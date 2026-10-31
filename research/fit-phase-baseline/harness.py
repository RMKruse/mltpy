"""Phase-level fit baseline for mltpy (wayfinder ticket #139).

Evidence-gathering harness only: wraps library internals at runtime, never
edits library code.  One case per process:

    python harness.py --case NAME --out results/NAME.json

Sequence inside one process (so "cold" means the first fit in a fresh
interpreter for that case):

1. build data, record RSS
2. cold fit (fresh model, untraced) + ru_maxrss after it
3. warm fresh-model reps, then same-instance refit reps (untraced)
4. one instrumented fit (phase attribution; wrapper overhead noted)
5. independent validation of the final theta + SLSQP cross-check
6. tracemalloc peak / retained memory on one more fit
7. cProfile of one more fit
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import os
import platform
import pstats
import resource
import sys
import time
import tracemalloc
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import nnls

REPO = Path(os.environ.get("MLTPY_REPO", "/Users/entropy/Git/mltpy"))
sys.path.insert(0, str(REPO / "benchmarks"))
REF = REPO / "reference"

import mltpy  # noqa: E402
import mltpy.basis as mb  # noqa: E402
import mltpy.likelihood as ml  # noqa: E402
import mltpy.model as mm  # noqa: E402
import mltpy.optimizer as mo  # noqa: E402
from bench_python import generate_dataset  # noqa: E402
from mltpy import (  # noqa: E402
    BernsteinBasis,
    BoxCox,
    CensoredData,
    CensoringType,
    ConditionalTransformationModel,
    ConvergenceWarning,
    Coxph,
    InteractionBasis,
    OptimizerConfig,
)

SEED = 0  # restart random_state for every fit


def cfg(**kw: Any) -> OptimizerConfig:
    return OptimizerConfig(random_state=SEED, **kw)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    family: str
    make: Callable[[], Any]  # fresh unfitted model
    fit_args: tuple[Any, ...]
    fit_kwargs: dict[str, Any] = field(default_factory=dict)
    reps_warm: int = 9
    reps_refit: int = 5
    profile_ci: bool = False  # timed op is confint(type="profile") after fit
    r_check: Callable[[Any], dict[str, Any]] | None = None
    n: int = 0


def _mlt(order: int, cens: CensoringType, **kw: Any) -> Callable[[], Any]:
    return lambda: mltpy.MLT(
        order=order, support=(0.0, 10.0), censoring=cens, optimizer_config=cfg(), **kw
    )


def _cov(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, (n, k))


def _ia_data(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261031)
    X = rng.uniform(-1, 1, (n, 1))
    y = np.clip(rng.normal(0.4 * X[:, 0], 1), -3.9, 3.9)
    return y, X


def _ia_model(oy: int, ox: int, scaling: np.ndarray | None = None) -> Callable[[], Any]:
    def make() -> Any:
        ib = InteractionBasis(
            y_basis=BernsteinBasis(order=oy, support=(-4.0, 4.0)),
            x_basis=BernsteinBasis(order=ox, support=(-1.0, 1.0)),
        )
        return ConditionalTransformationModel(
            basis=ib, base_distribution="normal", scaling=scaling, optimizer_config=cfg()
        )

    return make


def _r_interaction(label: str) -> Case:
    b = REF
    y = np.loadtxt(b / "interaction_bs_bs_y_train.txt")
    x = np.loadtxt(b / "interaction_bs_bs_x_train.txt")
    sup = tuple(np.loadtxt(b / "interaction_bs_bs_y_support.txt"))
    theta_R = np.loadtxt(b / f"interaction_bs_bs_{label}_theta.txt")
    ll_R = float(np.loadtxt(b / f"interaction_bs_bs_{label}_loglik.txt"))
    dist = "normal" if label == "normal" else "logistic"

    def make() -> Any:
        ib = InteractionBasis(
            y_basis=BernsteinBasis(order=2, support=sup),
            x_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
        )
        return ConditionalTransformationModel(
            basis=ib, base_distribution=dist, optimizer_config=cfg()
        )

    def check(m: Any) -> dict[str, Any]:
        Theta_R = theta_R.reshape(3, 3).T
        return {
            "max_abs_dTheta_vs_R": float(np.max(np.abs(m.Theta_ - Theta_R))),
            "ll_mltpy": float(m.result_.log_likelihood),
            "ll_R": ll_R,
            "abs_dll_vs_R": float(abs(m.result_.log_likelihood - ll_R)),
            "test_tolerance": "Theta rtol=atol=1e-4; ll rtol=1e-6",
        }

    return Case(
        f"ia_R_{label}_3x3", "interaction", make, (y, x.reshape(-1, 1)),
        r_check=check, n=len(y),
    )


def _r_stram() -> Case:
    p = "stram_bs_bs_normal_"
    ld = lambda s: np.loadtxt(REF / f"{p}{s}.txt")  # noqa: E731
    y, x, xs, sup = ld("y_train"), ld("x_train"), ld("x_s_train"), ld("y_support")

    def make() -> Any:
        ib = InteractionBasis(
            y_basis=BernsteinBasis(order=2, support=tuple(sup)),
            x_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
        )
        return ConditionalTransformationModel(
            basis=ib, scaling=xs.reshape(-1, 1), base_distribution="normal",
            optimizer_config=cfg(),
        )

    extra = {}
    for nm in ("loglik", "gamma", "theta"):
        f = REF / f"{p}{nm}.txt"
        if f.exists():
            extra[nm] = np.loadtxt(f)

    def check(m: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"ll_mltpy": float(m.result_.log_likelihood)}
        if "loglik" in extra:
            out["ll_R"] = float(extra["loglik"])
            out["abs_dll_vs_R"] = abs(out["ll_mltpy"] - out["ll_R"])
        if "gamma" in extra:
            out["abs_dgamma_vs_R"] = float(
                np.max(np.abs(np.atleast_1d(m.gamma_) - np.atleast_1d(extra["gamma"])))
            )
        return out

    return Case("ia_scaled_R_stram_3x3+1", "interaction+scaling", make,
                (y, x.reshape(-1, 1)), r_check=check, n=len(y))


def _profile_case(model_name: str) -> Case:
    ld = lambda s: np.loadtxt(REF / f"vcov_{model_name}_{s}.txt")  # noqa: E731
    y, x, sup = ld("y"), ld("x").reshape(-1, 1), tuple(ld("support"))
    ci_R = np.loadtxt(REF / f"profile_ci_{model_name}.txt").reshape(-1, 2)
    if model_name == "coxph":
        ev = ld("event").astype(int)
        yy: Any = CensoredData.right_censored(y, censored=ev == 0)
        make = lambda: Coxph(support=sup, order=4, optimizer_config=cfg())  # noqa: E731
    else:
        yy = y
        make = lambda: BoxCox(support=sup, order=4, optimizer_config=cfg())  # noqa: E731

    def check(m: Any) -> dict[str, Any]:
        ci = m._last_ci
        exp = ci_R.copy()
        if model_name == "boxcox":
            exp[5] = -exp[5][::-1]
        return {
            "max_abs_dCI_vs_R": float(np.nanmax(np.abs(ci - exp))),
            "test_tolerance": "rtol=1e-3, atol=1e-6",
            "nonfinite_ci_entries": int(np.sum(~np.isfinite(ci))),
        }

    return Case(f"profile_ci_{model_name}", "profile-ci", make, (yy,), {"X": x},
                reps_warm=2, reps_refit=0, profile_ci=True, r_check=check, n=len(y))


def build_cases() -> dict[str, Case]:
    C: dict[str, Case] = {}
    for n in (100, 1000, 5000):
        y, _ = generate_dataset(n, "NONE")
        C[f"shift_none_n{n}"] = Case(
            f"shift_none_n{n}", "shift", _mlt(6, CensoringType.NONE), (y,), n=n)
        yr, st = generate_dataset(n, "RIGHT")
        cd = CensoredData.right_censored(yr, st == 0)
        C[f"shift_right_n{n}"] = Case(
            f"shift_right_n{n}", "shift", _mlt(6, CensoringType.RIGHT), (cd,), n=n)
    # interval censoring
    y, _ = generate_dataset(1000, "NONE")
    ic = CensoredData.interval_censored(y * 0.9, np.minimum(y * 1.1, 9.99))
    C["shift_interval_n1000"] = Case(
        "shift_interval_n1000", "shift", _mlt(6, CensoringType.INTERVAL), (ic,), n=1000)
    # covariate shift (BoxCox / Coxph, 3 covariates)
    for n in (1000, 5000):
        X = _cov(n, 3, 7)
        rng = np.random.default_rng(11 + n)
        yb = np.clip(np.exp(0.5 * rng.normal(size=n) + 0.7 + X @ [0.3, -0.2, 0.1]),
                     0.05, 9.95)
        C[f"boxcox_x3_n{n}"] = Case(
            f"boxcox_x3_n{n}", "shift+X",
            lambda: BoxCox(support=(0.0, 10.0), order=6, optimizer_config=cfg()),
            (yb,), {"X": X}, n=n)
        cens = rng.uniform(size=n) < 0.3
        C[f"coxph_x3_right_n{n}"] = Case(
            f"coxph_x3_right_n{n}", "shift+X+right",
            lambda: Coxph(support=(0.0, 10.0), order=6, optimizer_config=cfg()),
            (CensoredData.right_censored(yb, cens),), {"X": X}, n=n)
    # scaling (shift + scale)
    for n in (1000, 5000):
        X = _cov(n, 1, 21)
        Xs = _cov(n, 1, 22)
        rng = np.random.default_rng(31 + n)
        ys = np.clip(np.exp(0.5 * rng.normal(size=n) * np.exp(0.4 * Xs[:, 0])
                            + 0.7 + 0.3 * X[:, 0]), 0.05, 9.95)
        C[f"boxcox_scaled_n{n}"] = Case(
            f"boxcox_scaled_n{n}", "shift+scaling",
            lambda Xs=Xs: BoxCox(support=(0.0, 10.0), order=6,
                                 optimizer_config=cfg(), scaling=Xs),
            (ys,), {"X": X}, n=n)
    # interaction: R fixtures + synthetic grid over n and m=p*q
    C["ia_R_normal_3x3"] = _r_interaction("normal")
    C["ia_R_logistic_3x3"] = _r_interaction("logistic")
    if (REF / "stram_bs_bs_normal_y_train.txt").exists():
        C["ia_scaled_R_stram_3x3+1"] = _r_stram()
    for (oy, ox), ns in {
        (3, 1): (250, 1000, 4000),   # m = 8
        (4, 2): (1000,),             # m = 15
        (6, 3): (250, 1000),         # m = 28 (recon case at n=1000)
    }.items():
        for n in ns:
            y, X = _ia_data(n)
            m = (oy + 1) * (ox + 1)
            big = m >= 28
            nm = f"ia_bs{oy}x{ox}_m{m}_n{n}"
            C[nm] = Case(nm, "interaction", _ia_model(oy, ox), (y, X),
                         reps_warm=2 if big else 5, reps_refit=1 if big else 3, n=n)
    # interaction + scaling (synthetic), m = 8 + 1
    y, X = _ia_data(1000)
    Xs = _cov(1000, 1, 41)
    C["ia_scaled_bs3x1_m9_n1000"] = Case(
        "ia_scaled_bs3x1_m9_n1000", "interaction+scaling", _ia_model(3, 1, Xs),
        (y, X), reps_warm=5, reps_refit=3, n=1000)
    # bounded profile-CI workloads (R fixtures)
    C["profile_ci_boxcox"] = _profile_case("boxcox")
    C["profile_ci_coxph"] = _profile_case("coxph")
    return C


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def rss_mib() -> float:
    # macOS ru_maxrss is bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def run_op(case: Case, model: Any) -> Any:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(*case.fit_args, **case.fit_kwargs)
        if case.profile_ci:
            model._last_ci = model.confint(level=0.95, type="profile")
    model._warn = [f"{x.category.__name__}: {str(x.message)[:160]}" for x in w]
    return model


def timed(case: Case, model: Any) -> tuple[float, Any]:
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        run_op(case, model)
        dt = time.perf_counter() - t0
    finally:
        gc.enable()
    return dt, model


def fit_status(m: Any) -> dict[str, Any]:
    r = m.result_
    return {
        "raw_converged": bool(r.converged),
        "kkt_residual_stored": None if r.kkt_residual is None else float(r.kkt_residual),
        "solver_message": str(r.solver_message)[:200],
        "n_restarts": int(r.n_restarts),
        "n_outer_iter": r.n_outer_iter,
        "n_inner_iter": int(r.n_iter),
        "log_likelihood": float(r.log_likelihood),
        "warnings": sorted(set(m._warn)),
    }


# ---------------------------------------------------------------------------
# Phase instrumentation
# ---------------------------------------------------------------------------


class Phases:
    def __init__(self) -> None:
        self.stack: list[str] = ["other"]
        self.t: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.obj_calls: dict[str, int] = {}
        self.ll_ia_calls: dict[str, int] = {}
        self.polish: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self._orig: list[tuple[Any, str, Any]] = []

    def _patch(self, mod: Any, name: str, new: Any) -> None:
        self._orig.append((mod, name, getattr(mod, name)))
        setattr(mod, name, new)

    def restore(self) -> None:
        for mod, name, f in reversed(self._orig):
            setattr(mod, name, f)

    def phase(self, label: str, f: Callable[..., Any], post: Callable[..., None] | None = None) -> Callable[..., Any]:
        def w(*a: Any, **k: Any) -> Any:
            self.stack.append(label)
            t0 = time.perf_counter()
            try:
                out = f(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                self.stack.pop()
                self.t[label] = self.t.get(label, 0.0) + dt
                self.calls[label] = self.calls.get(label, 0) + 1
            if post is not None:
                post(a, k, out, dt)
            return out
        return w

    def counter(self, store: dict[str, int], f: Callable[..., Any]) -> Callable[..., Any]:
        def w(*a: Any, **k: Any) -> Any:
            key = self.stack[-1]
            store[key] = store.get(key, 0) + 1
            return f(*a, **k)
        return w

    def install(self) -> None:
        def after_attempt(a: Any, k: Any, r: Any, dt: float) -> None:
            self.attempts.append({
                "seconds": dt, "converged": bool(r.converged),
                "kkt": float(r.kkt_residual), "fun": float(r.fun),
                "outer": int(r.n_outer_iter), "inner": int(r.n_inner_iter),
                "message": str(r.message)[:120],
            })

        def after_polish(a: Any, k: Any, out: Any, dt: float) -> None:
            theta0, obj = a[0], a[1]
            f0 = float(obj(theta0)[0])
            rec: dict[str, Any] = {"seconds": dt, "accepted": out is not None,
                                   "nll_before": f0, "theta_before": theta0.tolist()}
            if out is not None:
                rec["nll_after"] = float(obj(out)[0])
                rec["max_abs_dtheta"] = float(np.max(np.abs(out - theta0)))
            self.polish.append(rec)

        self._patch(mo, "auglag_minimize",
                    self.phase("solver_attempts", mo.auglag_minimize, after_attempt))
        self._patch(mo, "_ncg_polish", self.phase("polish", mo._ncg_polish, after_polish))
        self._patch(mo, "_hessian", self.phase("polish_hessian", mo._hessian))
        self._patch(mm, "_hessian", self.phase("final_hessian", mm._hessian))
        self._patch(mm, "_score_matrix", self.phase("score_matrix", mm._score_matrix))
        self._patch(mm, "optimize", self.phase("optimize_total", mm.optimize))
        self._patch(mo, "_negative_log_likelihood_from_dist",
                    self.counter(self.obj_calls, mo._negative_log_likelihood_from_dist))
        self._patch(ml, "_ll_interaction_none",
                    self.counter(self.ll_ia_calls, ml._ll_interaction_none))


# ---------------------------------------------------------------------------
# Independent validation
# ---------------------------------------------------------------------------


def _fit_inputs(case: Case, m: Any) -> dict[str, Any]:
    y = case.fit_args[0]
    X = case.fit_args[1] if len(case.fit_args) > 1 else case.fit_kwargs.get("X")
    cens = CensoringType.NONE if m.censoring is None else m.censoring
    return {"basis": m.basis, "y": y, "X": X, "censoring": cens,
            "base_distribution": m.base_distribution, "scaling": m.scaling}


def kkt_check(theta: np.ndarray, A: np.ndarray | None, inp: dict[str, Any], n: int,
              active_tol: float = 1e-6) -> dict[str, Any]:
    nll, g = ml.negative_log_likelihood(theta, gradient=True, **inp)
    g = np.asarray(g)
    out: dict[str, Any] = {"nll": float(nll), "grad_inf": float(np.max(np.abs(g)))}
    if A is None or A.size == 0:
        out.update(stationarity_inf=out["grad_inf"], min_slack=None, n_active=0)
    else:
        slack = A @ theta
        act = slack <= active_tol * (1.0 + np.abs(theta).max())
        if act.any():
            mu, _ = nnls(A[act].T, g)  # ∇f = Aᵀμ, μ ≥ 0
            r = g - A[act].T @ mu
        else:
            r = g
        out.update(stationarity_inf=float(np.max(np.abs(r))),
                   min_slack=float(slack.min()), n_active=int(act.sum()))
    out["stationarity_inf_per_obs"] = out["stationarity_inf"] / n
    return out


def validate(case: Case, m: Any, ph: Phases) -> dict[str, Any]:
    inp = _fit_inputs(case, m)
    A = m._A_ineq_
    v: dict[str, Any] = {"final": kkt_check(m.theta_, A, inp, case.n)}
    if ph.polish:
        tb = np.asarray(ph.polish[-1]["theta_before"])
        v["pre_polish"] = kkt_check(tb, A, inp, case.n)
    # SLSQP cross-check of the optimum (optimize() only: no eager Hessian)
    try:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mo.optimize(inp["basis"], inp["y"], X=inp["X"], censoring=inp["censoring"],
                            config=cfg(solver="slsqp", tol=1e-12, max_iter=5000),
                            base_distribution=inp["base_distribution"],
                            scaling=inp["scaling"])
        v["slsqp"] = {"seconds": time.perf_counter() - t0, "success": bool(r.converged),
                      "ll": float(r.log_likelihood),
                      "ll_auglag_minus_slsqp": float(m.result_.log_likelihood - r.log_likelihood),
                      "max_abs_dtheta": float(np.max(np.abs(r.theta - m.theta_)))}
    except Exception as e:  # recorded, not hidden
        v["slsqp"] = {"error": repr(e)[:200]}
    # Interaction: FD Hessian vs central difference of the analytic gradient
    if isinstance(m.basis, InteractionBasis):
        th = m.theta_
        k = th.size
        Hg = np.empty((k, k))
        t0 = time.perf_counter()
        for i in range(k):
            e = np.zeros(k)
            e[i] = 1e-5
            gp = ml.negative_log_likelihood(th + e, gradient=True, **inp)[1]
            gm = ml.negative_log_likelihood(th - e, gradient=True, **inp)[1]
            Hg[:, i] = (np.asarray(gp) - np.asarray(gm)) / 2e-5
        tg = time.perf_counter() - t0
        Hg = 0.5 * (Hg + Hg.T)
        H = m.hessian_
        ev = np.linalg.eigvalsh(H)
        v["hessian_fd_vs_gradfd"] = {
            "seconds_gradfd": tg,
            "max_abs_diff": float(np.max(np.abs(H - Hg))),
            "rel_diff": float(np.max(np.abs(H - Hg)) / np.max(np.abs(H))),
            "min_eig_fd": float(ev[0]), "cond_fd": float(ev[-1] / max(ev[0], 1e-300)),
        }
        try:
            se1 = np.sqrt(np.diag(np.linalg.inv(H)))
            se2 = np.sqrt(np.diag(np.linalg.inv(Hg)))
            v["hessian_fd_vs_gradfd"]["max_rel_se_diff"] = float(np.max(np.abs(se1 / se2 - 1)))
        except Exception as e:
            v["hessian_fd_vs_gradfd"]["se_error"] = repr(e)[:120]
    if case.r_check is not None:
        v["r_reference"] = case.r_check(m)
    return v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def env_info() -> dict[str, Any]:
    cfgs = io.StringIO()
    try:
        info = np.show_config(mode="dicts")
        blas = info.get("Build Dependencies", {}).get("blas", {})
    except TypeError:
        blas = {}
    return {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "mltpy": getattr(mltpy, "__version__", "?"),
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu": os.popen("sysctl -n machdep.cpu.brand_string").read().strip(),
        "ncpu": os.cpu_count(),
        "blas": {k: blas.get(k) for k in ("name", "version", "detection method")},
        "thread_env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "loadavg": os.getloadavg(),
        "git_head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "git_dirty": bool(os.popen(f"git -C {REPO} status --porcelain src").read().strip()),
        "restart_random_state": SEED,
        "_": cfgs.getvalue(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cases = build_cases()
    case = cases[a.case]
    res: dict[str, Any] = {"case": case.name, "family": case.family, "n": case.n,
                           "env": env_info()}
    res["rss_before_mib"] = rss_mib()

    # 1. cold + warm fresh-model fits
    times, statuses = [], []
    for i in range(1 + case.reps_warm):
        dt, m = timed(case, case.make())
        times.append(dt)
        statuses.append(fit_status(m))
        if i == 0:
            res["maxrss_after_cold_mib"] = rss_mib()
    res["fresh_seconds"] = times
    res["fresh_status"] = statuses
    res["n_params"] = int(m.theta_.size)
    # 2. same-instance refits
    rt = []
    for _ in range(case.reps_refit):
        dt, m = timed(case, m)
        rt.append(dt)
    res["refit_seconds"] = rt
    res["refit_status_last"] = fit_status(m) if rt else None

    # 3. instrumented fit
    ph = Phases()
    ph.install()
    try:
        t0 = time.perf_counter()
        mi = run_op(case, case.make())
        res["instrumented_seconds"] = time.perf_counter() - t0
    finally:
        ph.restore()
    res["phases"] = {
        "seconds": ph.t, "calls": ph.calls, "objective_calls_by_phase": ph.obj_calls,
        "ll_interaction_calls_by_phase": ph.ll_ia_calls, "solver_attempts": ph.attempts,
        "polish": [{k: v for k, v in p.items() if k != "theta_before"} for p in ph.polish],
    }
    res["instrumented_status"] = fit_status(mi)

    # 4. independent validation (on the instrumented fit: same seed -> same theta)
    res["validation"] = validate(case, mi, ph) if not case.profile_ci else (
        {"r_reference": case.r_check(mi)} if case.r_check else {})
    res["determinism_max_abs_dtheta_vs_fresh"] = float(np.max(np.abs(mi.theta_ - m.theta_)))

    # 5. memory: tracemalloc peak during one op + retained by the fitted model
    mb._bernstein_cache.clear()
    mb._bernstein_assembled_cache.clear()
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    mt = run_op(case, case.make())
    gc.collect()
    cur, peak = tracemalloc.get_traced_memory()
    mb._bernstein_cache.clear()
    mb._bernstein_assembled_cache.clear()
    gc.collect()
    cur2 = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    est = getattr(mt, "_estfun_cache_", None)
    res["memory"] = {
        "traced_peak_mib": (peak - base) / 2**20,
        "retained_with_caches_mib": (cur - base) / 2**20,
        "retained_model_only_mib": (cur2 - base) / 2**20,
        "score_matrix_mib": None if est is None else est.nbytes / 2**20,
        "maxrss_mib_end": rss_mib(),
    }
    del mt

    # 6. cProfile one op
    pr = cProfile.Profile()
    pr.enable()
    run_op(case, case.make())
    pr.disable()
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    st.sort_stats("cumulative").print_stats(18)
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(10)
    trim = lambda t: "\n".join(t.getvalue().splitlines()[4:])  # noqa: E731
    res["cprofile_cumulative"] = trim(s)
    res["cprofile_tottime"] = trim(s2)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\n".join(build_cases()))
    else:
        main()
