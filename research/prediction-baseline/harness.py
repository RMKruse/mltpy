"""Prediction runtime / memory baseline for mltpy (wayfinder ticket #140).

Evidence-gathering harness only.  It never edits library code; candidate
changes are *harness-local prototypes* validated bitwise against the current
library output.  One case (= one fitted model) per process:

    python harness.py --case NAME --out results/NAME.json

Per case:

1. fit the model (``random_state=0``) and validate it independently
   (auglag vs SLSQP log-likelihood / theta);
2. fit-entry eviction: score() on the training data before / after a burst of
   prediction calls with changing data;
3. per (operation, batch size m): first call on new data, repeated calls on
   the same data, repeated calls on changing data (untraced timings); one
   tracemalloc run (peak, retained, global-cache bytes, model-instance bytes,
   RSS); a 40-call changing-data burst (bounded cache retention, RSS);
4. instrumented run at m=1000: basis-cache hits/misses, evaluate/derivative
   calls, bytes hashed into cache keys;
5. candidate prototypes (path-specific).

All caches are cleared before each measurement so cells are independent.
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
import statistics
import sys
import time
import tracemalloc
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil
import scipy
from scipy.interpolate import CubicSpline
from scipy.special import comb

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

import mltpy  # noqa: E402
import mltpy.basis as mb  # noqa: E402
import mltpy.model as mm  # noqa: E402
from bench_python import generate_dataset  # noqa: E402
from mltpy import (  # noqa: E402
    BernsteinBasis,
    BoxCox,
    CensoredData,
    CensoringType,
    ConditionalTransformationModel,
    Coxph,
    InteractionBasis,
    OptimizerConfig,
    Survreg,
)
from mltpy.likelihood import _get_dist  # noqa: E402

SEED = 0
PROC = psutil.Process()
M_SIZES = (100, 1_000, 10_000, 100_000)
RSTYLE_MAX_M = 10_000  # per-row Python loop: linear, capped for wall time
BURST = 40


def cfg(**kw: Any) -> OptimizerConfig:
    return OptimizerConfig(random_state=SEED, **kw)


def rss_mib() -> float:
    return PROC.memory_info().rss / 2**20


def clear_caches() -> None:
    mb._bernstein_cache.clear()
    mb._bernstein_assembled_cache.clear()
    gc.collect()


def cache_stats() -> dict[str, Any]:
    """Bytes held by the two module-level Bernstein caches (dedup by id)."""
    seen: set[int] = set()
    arr = 0
    keys = 0

    def add(a: np.ndarray) -> None:
        nonlocal arr
        base = a if a.base is None else a.base
        if id(base) not in seen:
            seen.add(id(base))
            arr += base.nbytes

    for k, v in mb._bernstein_cache.items():
        keys += len(k[1])
        add(v)
    for k, (B, dB) in mb._bernstein_assembled_cache.items():
        keys += len(k[2])
        add(B)
        add(dB)
    return {
        "raw_entries": len(mb._bernstein_cache),
        "assembled_entries": len(mb._bernstein_assembled_cache),
        "array_mib": arr / 2**20,
        "key_mib": keys / 2**20,
        "total_mib": (arr + keys) / 2**20,
    }


def model_bytes(m: Any) -> float:
    """nbytes of ndarrays reachable from the model instance (1 level + result_)."""
    seen: set[int] = set()
    tot = 0

    def walk(obj: Any, depth: int) -> None:
        nonlocal tot
        if isinstance(obj, np.ndarray):
            if id(obj) not in seen:
                seen.add(id(obj))
                tot += obj.nbytes
            return
        if depth > 2:
            return
        if isinstance(obj, (list, tuple)):
            for o in obj:
                walk(o, depth + 1)
        elif isinstance(obj, dict):
            for o in obj.values():
                walk(o, depth + 1)
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            for o in vars(obj).values():
                walk(o, depth + 1)

    walk(m, 0)
    return tot / 2**20


def summarize(ts: list[float]) -> dict[str, float]:
    return {
        "median_ms": 1e3 * statistics.median(ts),
        "min_ms": 1e3 * min(ts),
        "max_ms": 1e3 * max(ts),
        "n": len(ts),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

Inputs = dict[str, Any]


@dataclass
class Case:
    name: str
    path: str  # shift-bisection | shift-scaling | rstyle-parent | rstyle-survreg | interaction
    make: Callable[[], Any]
    fit_args: tuple[Any, ...]
    fit_kwargs: dict[str, Any]
    # op -> (rng, m) -> predict kwargs
    ops: dict[str, Callable[[np.random.Generator, int], Inputs]] = field(
        default_factory=dict
    )
    y_train: Any = None


def _cov(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, (n, k))


def _probs(rng: np.random.Generator, m: int) -> np.ndarray:
    return rng.uniform(0.05, 0.95, m)


def _sim_u(rng: np.random.Generator, m: int) -> np.ndarray:
    # exactly what simulate() feeds to predict(what="quantile")
    return np.clip(rng.uniform(size=m), mm._SIMULATE_U_EPS, 1.0 - mm._SIMULATE_U_EPS)


def build_cases() -> dict[str, Case]:
    C: dict[str, Case] = {}
    n = 1000
    # --- covariate shift, exact (BoxCox, 3 covariates): #139 boxcox_x3_n1000
    X = _cov(n, 3, 7)
    rng = np.random.default_rng(11 + n)
    yb = np.clip(
        np.exp(0.5 * rng.normal(size=n) + 0.7 + X @ [0.3, -0.2, 0.1]), 0.05, 9.95
    )
    cens = rng.uniform(size=n) < 0.3

    def x3(r: np.random.Generator, m: int) -> np.ndarray:
        return r.uniform(-1, 1, (m, 3))

    C["boxcox_x3"] = Case(
        "boxcox_x3",
        "shift-bisection",
        lambda: BoxCox(support=(0.0, 10.0), order=6, optimizer_config=cfg()),
        (yb,),
        {"X": X},
        {
            "distribution": lambda r, m: {
                "y_new": r.uniform(0.05, 9.95, m), "X_new": x3(r, m),
                "what": "distribution"},
            "density": lambda r, m: {
                "y_new": r.uniform(0.05, 9.95, m), "X_new": x3(r, m),
                "what": "density"},
            "quantile": lambda r, m: {
                "y_new": _probs(r, m), "X_new": x3(r, m), "what": "quantile"},
            "simulate": lambda r, m: {
                "y_new": _sim_u(r, m), "X_new": x3(r, m), "what": "quantile"},
        },
        yb,
    )
    # --- shift + scaling: #139 boxcox_scaled_n1000
    Xd = _cov(n, 1, 21)
    Xs = _cov(n, 1, 22)
    rng = np.random.default_rng(31 + n)
    ys = np.clip(
        np.exp(0.5 * rng.normal(size=n) * np.exp(0.4 * Xs[:, 0]) + 0.7 + 0.3 * Xd[:, 0]),
        0.05,
        9.95,
    )
    C["boxcox_scaled"] = Case(
        "boxcox_scaled",
        "shift-scaling",
        lambda: BoxCox(support=(0.0, 10.0), order=6, optimizer_config=cfg(), scaling=Xs),
        (ys,),
        {"X": Xd},
        {
            "distribution": lambda r, m: {
                "y_new": r.uniform(0.05, 9.95, m), "X_new": r.uniform(-1, 1, (m, 1)),
                "X_scale_new": r.uniform(-1, 1, (m, 1)), "what": "distribution"},
            "quantile": lambda r, m: {
                "y_new": _probs(r, m), "X_new": r.uniform(-1, 1, (m, 1)),
                "X_scale_new": r.uniform(-1, 1, (m, 1)), "what": "quantile"},
        },
        ys,
    )
    # --- right-censored baseline (recon case family, order 6): R-style, all
    # shifts equal (0)
    yr, st = generate_dataset(n, "RIGHT")
    C["right_baseline"] = Case(
        "right_baseline",
        "rstyle-parent",
        lambda: mltpy.MLT(
            order=6, support=(0.0, 10.0), censoring=CensoringType.RIGHT,
            optimizer_config=cfg()),
        (CensoredData.right_censored(yr, st == 0),),
        {},
        {
            "distribution": lambda r, m: {
                "y_new": r.uniform(0.05, 9.95, m), "what": "distribution"},
            "quantile": lambda r, m: {"y_new": _probs(r, m), "what": "quantile"},
            "simulate": lambda r, m: {"y_new": _sim_u(r, m), "what": "quantile"},
        },
        CensoredData.right_censored(yr, st == 0),
    )

    def cox_ops(lo: float, hi: float) -> dict[str, Callable[..., Inputs]]:
        def few(r: np.random.Generator, m: int) -> np.ndarray:
            prof = np.random.default_rng(5).uniform(-1, 1, (10, 3))
            return prof[r.integers(0, 10, m)]

        return {
            "distribution": lambda r, m: {
                "y_new": r.uniform(lo, hi, m), "X_new": x3(r, m),
                "what": "distribution"},
            # every row its own covariate profile
            "quantile_varying": lambda r, m: {
                "y_new": _probs(r, m), "X_new": x3(r, m), "what": "quantile"},
            # one profile, many probabilities (quantile curve for a subject)
            "quantile_equal": lambda r, m: {
                "y_new": _probs(r, m),
                "X_new": np.tile(r.uniform(-1, 1, (1, 3)), (m, 1)),
                "what": "quantile"},
            # 10 profiles, interleaved (e.g. simulate over a design)
            "quantile_10prof": lambda r, m: {
                "y_new": _probs(r, m), "X_new": few(r, m), "what": "quantile"},
            # equal profile + varying offset (offsets enter the shift)
            "quantile_equal_offset": lambda r, m: {
                "y_new": _probs(r, m),
                "X_new": np.tile(r.uniform(-1, 1, (1, 3)), (m, 1)),
                "offset_new": r.normal(0, 0.3, m), "what": "quantile"},
            "simulate_equal": lambda r, m: {
                "y_new": _sim_u(r, m),
                "X_new": np.tile(r.uniform(-1, 1, (1, 3)), (m, 1)),
                "what": "quantile"},
        }

    C["coxph_x3"] = Case(
        "coxph_x3",
        "rstyle-parent",
        lambda: Coxph(support=(0.0, 10.0), order=6, optimizer_config=cfg()),
        (CensoredData.right_censored(yb, cens),),
        {"X": X},
        cox_ops(0.05, 9.95),
        CensoredData.right_censored(yb, cens),
    )
    C["survreg_x3"] = Case(
        "survreg_x3",
        "rstyle-survreg",
        lambda: Survreg(support=(0.04, 10.5), distribution="weibull", order=6,
                        optimizer_config=cfg()),
        (CensoredData.right_censored(yb, cens),),
        {"X": X},
        cox_ops(0.05, 9.95),
        CensoredData.right_censored(yb, cens),
    )

    # --- interaction (#139 synthetic generator; valid configurations only)
    def ia_data(nn: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(20261031)
        Xi = r.uniform(-1, 1, (nn, 1))
        yi = np.clip(r.normal(0.4 * Xi[:, 0], 1), -3.9, 3.9)
        return yi, Xi

    def ia_model(oy: int, ox: int, scaling: np.ndarray | None = None) -> Callable[[], Any]:
        def make() -> Any:
            ib = InteractionBasis(
                y_basis=BernsteinBasis(order=oy, support=(-4.0, 4.0)),
                x_basis=BernsteinBasis(order=ox, support=(-1.0, 1.0)),
            )
            return ConditionalTransformationModel(
                basis=ib, base_distribution="normal", scaling=scaling,
                optimizer_config=cfg())
        return make

    def ia_ops(scaled: bool) -> dict[str, Callable[..., Inputs]]:
        def xs(r: np.random.Generator, m: int) -> dict[str, Any]:
            return {"X_scale_new": r.uniform(-1, 1, (m, 1))} if scaled else {}

        return {
            "distribution": lambda r, m: {
                "y_new": r.uniform(-3.9, 3.9, m), "X_new": r.uniform(-1, 1, (m, 1)),
                "what": "distribution", **xs(r, m)},
            "density": lambda r, m: {
                "y_new": r.uniform(-3.9, 3.9, m), "X_new": r.uniform(-1, 1, (m, 1)),
                "what": "density", **xs(r, m)},
            "quantile": lambda r, m: {
                "y_new": _probs(r, m), "X_new": r.uniform(-1, 1, (m, 1)),
                "what": "quantile", **xs(r, m)},
            "simulate": lambda r, m: {
                "y_new": _sim_u(r, m), "X_new": r.uniform(-1, 1, (m, 1)),
                "what": "quantile", **xs(r, m)},
        }

    for (oy, ox, nn) in ((3, 1, 1000), (4, 2, 1000), (6, 3, 250)):
        yi, Xi = ia_data(nn)
        nm = f"ia_bs{oy}x{ox}_m{(oy + 1) * (ox + 1)}_n{nn}"
        C[nm] = Case(nm, "interaction", ia_model(oy, ox), (yi, Xi), {}, ia_ops(False),
                     yi)
        C[nm].fit_kwargs = {}
    yi, Xi = ia_data(1000)
    Xs_i = _cov(1000, 1, 41)
    C["ia_scaled_bs3x1_m9_n1000"] = Case(
        "ia_scaled_bs3x1_m9_n1000", "interaction", ia_model(3, 1, Xs_i), (yi, Xi), {},
        ia_ops(True), yi)
    return C


# ---------------------------------------------------------------------------
# Fit + validation
# ---------------------------------------------------------------------------


def fit_and_validate(case: Case) -> tuple[Any, dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        model = case.make().fit(*case.fit_args, **case.fit_kwargs)
        t_fit = time.perf_counter() - t0
        ref = case.make()
        ref.optimizer_config = cfg(solver="slsqp", tol=1e-12, max_iter=5000)
        ref.fit(*case.fit_args, **case.fit_kwargs)
    r = model.result_
    info = {
        "fit_seconds": t_fit,
        "n_params": int(model.theta_.size),
        "converged": bool(r.converged),
        "kkt_residual": None if r.kkt_residual is None else float(r.kkt_residual),
        "ll": float(r.log_likelihood),
        "ll_minus_slsqp": float(r.log_likelihood - ref.result_.log_likelihood),
        "max_abs_dtheta_vs_slsqp": float(np.max(np.abs(model.theta_ - ref.theta_))),
    }
    info["valid"] = bool(abs(info["ll_minus_slsqp"]) <= 1e-6 * abs(info["ll"]) + 1e-6)
    return model, info


def score_input(case: Case, model: Any) -> Callable[[], float]:
    kw = {}
    if "X" in case.fit_kwargs:
        kw["X"] = case.fit_kwargs["X"]
    if isinstance(model.basis, InteractionBasis):
        kw["X"] = case.fit_args[1]
    if model.scaling is not None:
        # score() drops scaling= (issue #144); evaluate the same likelihood
        # path directly with the fitted scaling design instead.
        from mltpy.likelihood import log_likelihood

        cens = CensoringType.NONE if model.censoring is None else model.censoring
        Xs = kw.get("X")
        return lambda: log_likelihood(
            model.theta_, model.basis, case.fit_args[0],
            None if Xs is None else np.asarray(Xs, dtype=float), cens,
            base_distribution=model.base_distribution, scaling=model.scaling)
    return lambda: model.score(case.fit_args[0], **kw)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def call(model: Any, inp: Inputs) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cast(np.ndarray, model.predict(**inp))


def measure_cell(model: Any, make: Callable[..., Inputs], m: int, reps: int) -> dict:
    out: dict[str, Any] = {"m": m}
    # timing: first call on new data (caches cleared), same data, changing data
    clear_caches()
    inp = make(np.random.default_rng(1000 + m), m)
    t0 = time.perf_counter()
    call(model, inp)
    out["first_call"] = 1e3 * (time.perf_counter() - t0)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        call(model, inp)
        ts.append(time.perf_counter() - t0)
    out["same_data"] = summarize(ts)
    clear_caches()
    inputs = [make(np.random.default_rng(2000 + m + i), m) for i in range(reps)]
    ts = []
    for x in inputs:
        t0 = time.perf_counter()
        call(model, x)
        ts.append(time.perf_counter() - t0)
    out["changing_data"] = summarize(ts)
    del inputs

    # memory: one op on new data, traced
    clear_caches()
    mb0 = model_bytes(model)
    rss0 = rss_mib()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    res = call(model, inp)
    res_mib = res.nbytes / 2**20
    del res
    gc.collect()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out["traced"] = {
        "peak_mib": (peak - base) / 2**20,
        "retained_mib": (cur - base) / 2**20,
        "output_mib": res_mib,
        "cache_after_one_call": cache_stats(),
        "model_mib_before": mb0,
        "model_mib_delta": model_bytes(model) - mb0,
        "rss_delta_mib": rss_mib() - rss0,
    }
    # burst of changing-data calls: bounded cache retention + RSS
    clear_caches()
    rss_b0 = rss_mib()
    for i in range(BURST if m <= 10_000 else 12):
        call(model, make(np.random.default_rng(5000 + i), m))
        gc.collect()
    cs = cache_stats()
    rss_b1 = rss_mib()
    clear_caches()
    rss_b2 = rss_mib()
    out["burst"] = {
        "calls": BURST if m <= 10_000 else 12,
        "cache": cs,
        "rss_before_mib": rss_b0,
        "rss_after_burst_mib": rss_b1,
        "rss_after_clear_mib": rss_b2,
    }
    return out


class Counters:
    """Runtime wrappers counting cache traffic (instrumented run only)."""

    def __init__(self) -> None:
        self.c = {k: 0 for k in (
            "raw_hit", "raw_miss", "asm_hit", "asm_miss", "derivative_calls",
            "key_bytes_hashed", "dB_built_on_value_only")}
        self._orig_bm = mb._bernstein_matrix
        self._orig_ewd = BernsteinBasis.evaluate_with_derivative
        self._orig_der = BernsteinBasis.derivative
        self._orig_eval = BernsteinBasis.evaluate
        self._in_eval = False

    def __enter__(self) -> Counters:
        c = self.c
        orig_bm, orig_ewd, orig_der, orig_eval = (
            self._orig_bm, self._orig_ewd, self._orig_der, self._orig_eval)
        outer = self

        def bm(t: np.ndarray, k: int) -> np.ndarray:
            key = (k, np.ascontiguousarray(t, dtype=float).tobytes())
            c["key_bytes_hashed"] += len(key[1])
            c["raw_hit" if key in mb._bernstein_cache else "raw_miss"] += 1
            return orig_bm(t, k)

        def ewd(self_: Any, y: np.ndarray) -> Any:
            key = (self_.order, self_.support,
                   np.ascontiguousarray(y, dtype=float).tobytes())
            c["key_bytes_hashed"] += len(key[2])
            hit = key in mb._bernstein_assembled_cache
            c["asm_hit" if hit else "asm_miss"] += 1
            if not hit and outer._in_eval:
                c["dB_built_on_value_only"] += 1
            return orig_ewd(self_, y)

        def ev(self_: Any, y: np.ndarray) -> Any:
            outer._in_eval = True
            try:
                return orig_eval(self_, y)
            finally:
                outer._in_eval = False

        def der(self_: Any, y: np.ndarray, order: int = 1) -> Any:
            c["derivative_calls"] += 1
            return orig_der(self_, y, order)

        mb._bernstein_matrix = bm
        BernsteinBasis.evaluate_with_derivative = ewd  # type: ignore[method-assign]
        BernsteinBasis.evaluate = ev  # type: ignore[method-assign]
        BernsteinBasis.derivative = der  # type: ignore[method-assign]
        return self

    def __exit__(self, *a: Any) -> None:
        mb._bernstein_matrix = self._orig_bm
        BernsteinBasis.evaluate_with_derivative = self._orig_ewd  # type: ignore[method-assign]
        BernsteinBasis.evaluate = self._orig_eval  # type: ignore[method-assign]
        BernsteinBasis.derivative = self._orig_der  # type: ignore[method-assign]


def instrumented(model: Any, make: Callable[..., Inputs], m: int) -> dict[str, Any]:
    clear_caches()
    inp = make(np.random.default_rng(7), m)
    with Counters() as k:
        call(model, inp)
        first = dict(k.c)
        for key in k.c:
            k.c[key] = 0
        call(model, inp)
        second = dict(k.c)
    return {"m": m, "first_call": first, "repeat_same_data": second}


def profile_top(model: Any, inp: Inputs, top: int = 12) -> list[dict[str, Any]]:
    clear_caches()
    pr = cProfile.Profile()
    pr.enable()
    call(model, inp)
    pr.disable()
    st = pstats.Stats(pr)
    total = st.total_tt  # type: ignore[attr-defined]
    rows = []
    for (fn, line, name), (cc, nc, tt, ct, _) in st.stats.items():  # type: ignore[attr-defined]
        rows.append({"func": f"{Path(fn).name}:{line}:{name}", "ncalls": nc,
                     "tottime_share": tt / total, "cumtime_share": ct / total})
    rows.sort(key=lambda r: -r["tottime_share"])
    return rows[:top]


def key_cost(sizes: tuple[int, ...] = M_SIZES) -> dict[str, Any]:
    """Cost of building and hashing a content key for a fresh array of size m."""
    out = {}
    for m in sizes:
        y = np.random.default_rng(m).uniform(0, 1, m)
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            kb = np.ascontiguousarray(y, dtype=float).tobytes()
            hash(kb)
            ts.append(time.perf_counter() - t0)
        out[str(m)] = {"tobytes_plus_hash_us": 1e6 * statistics.median(ts),
                       "key_bytes": m * 8}
    return out


# ---------------------------------------------------------------------------
# Candidate prototypes (harness-local; never patched into the library)
# ---------------------------------------------------------------------------


def rstyle(
    probs: np.ndarray,
    shift: np.ndarray,
    h_base_grid: np.ndarray,
    q_grid: np.ndarray,
    dist: Any,
    zero_first: bool,
    reuse: str,
) -> tuple[np.ndarray, bool, int]:
    """Copy of the R-style loop; ``reuse`` in {none, consecutive, unique}.

    The per-probability work up to (c_s, q_s, status) depends only on
    shift[i]; the final interp / saturation step depends on p.  ``reuse``
    keeps that curve call-locally: the previous one (consecutive) or one per
    distinct shift within this call (unique).
    """
    eps = float(np.sqrt(np.finfo(float).eps))
    out = np.empty(probs.size)
    saturated = False
    built = 0
    prev_key: float | None = None
    prev: Any = None
    memo: dict[float, Any] = {}

    def curve(s: float) -> Any:
        cdf_grid = dist.cdf(h_base_grid + s)
        if zero_first:
            cdf_grid[0] = 0.0
        finite = np.isfinite(cdf_grid)
        if not np.any(finite):
            return None
        cmin = float(np.min(cdf_grid[finite]))
        cmax = float(np.max(cdf_grid[finite]))
        remove = ~finite
        flat_low = np.where(cdf_grid < cmin + eps)[0]
        flat_high = np.where(cdf_grid > cmax - eps)[0]
        if flat_low.size > 1:
            remove[flat_low[:-1]] = True
        if flat_high.size > 1:
            remove[flat_high[1:]] = True
        keep = ~remove
        qk = q_grid[keep]
        ck = cdf_grid[keep]
        if qk.size < 2:
            return None
        n_spline = max(3 * qk.size, 3)
        q_s = np.linspace(float(qk[0]), float(qk[-1]), n_spline, dtype=np.float64)
        c_s = CubicSpline(qk, ck, bc_type="not-a-knot", extrapolate=False)(q_s)
        finite_s = np.isfinite(c_s)
        if not np.any(finite_s):
            return None
        q_s = q_s[finite_s]
        c_s = c_s[finite_s]
        order = np.argsort(c_s)
        c_s = c_s[order]
        q_s = q_s[order]
        uniq = np.concatenate(([True], np.diff(c_s) > 0.0))
        c_s = c_s[uniq]
        q_s = q_s[uniq]
        if c_s.size == 0:
            return None
        return (c_s, q_s)

    for i, p in enumerate(probs):
        s = float(shift[i])
        if reuse == "consecutive" and prev_key is not None and s == prev_key:
            cur = prev
        elif reuse == "unique" and s in memo:
            cur = memo[s]
        else:
            cur = curve(s)
            built += 1
            prev_key, prev = s, cur
            if reuse == "unique":
                memo[s] = cur
        if cur is None:
            out[i] = q_grid[0]
            saturated = True
            continue
        c_s, q_s = cur
        if p < c_s[0]:
            out[i] = q_grid[0]
            saturated = True
        elif p > c_s[-1]:
            out[i] = q_grid[-1]
            saturated = True
        else:
            out[i] = float(np.interp(p, c_s, q_s))
    return out, saturated, built


def rstyle_setup(model: Any, inp: Inputs) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    theta_b, beta, _g, _p, q_d, _qs = model._split_fitted_theta()
    probs = np.asarray(inp["y_new"], dtype=float)
    shift = np.zeros(probs.size)
    if q_d > 0:
        shift = np.asarray(inp["X_new"], dtype=float) @ beta
    if inp.get("offset_new") is not None:
        shift = shift + np.asarray(inp["offset_new"], dtype=float)
    a, b = model.basis.support
    if isinstance(model, Survreg):
        q_grid = np.linspace(a, b, mm._QMLT_GRID_POINTS)
        h = model.basis.evaluate(q_grid) @ theta_b
        return h, q_grid, shift, False
    k = model.basis.order
    q_grid = np.linspace(0.0, b, mm._QMLT_GRID_POINTS)
    i_arr = np.arange(k + 1, dtype=float)
    j_arr = k - i_arr
    binom_theta = comb(k, i_arr, exact=False) * theta_b
    t_grid = (q_grid - a) * (1.0 / (b - a))
    T_grid = (t_grid[:, None] ** i_arr) * ((1.0 - t_grid)[:, None] ** j_arr)
    return T_grid @ binom_theta, q_grid, shift, True


def proto_rstyle(model: Any, case: Case, ops: list[str]) -> dict[str, Any]:
    dist = _get_dist(model.base_distribution)
    res: dict[str, Any] = {}
    for op in ops:
        for m in (100, 1_000, 10_000):
            inp = case.ops[op](np.random.default_rng(3000 + m), m)
            h, qg, shift, zf = rstyle_setup(model, inp)
            probs = np.asarray(inp["y_new"], dtype=float)
            lib = call(model, inp)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model.predict(**inp)
                lib_warned = any("saturated" in str(x.message) for x in w)
            row: dict[str, Any] = {"m": m, "distinct_shifts": int(np.unique(shift).size),
                                   "library_warned": lib_warned}
            for reuse in ("none", "consecutive", "unique"):
                ts = []
                for _ in range(3 if m >= 10_000 else 5):
                    t0 = time.perf_counter()
                    o, sat, built = rstyle(probs, shift, h, qg, dist, zf, reuse)
                    ts.append(time.perf_counter() - t0)
                row[reuse] = {
                    **summarize(ts), "curves_built": built,
                    "bitwise_equal_library": bool(np.array_equal(o, lib)),
                    "saturated_flag": sat,
                }
            res[f"{op}_m{m}"] = row
    return res


def proto_interaction(model: Any, case: Case) -> dict[str, Any]:
    """Interaction bisection with value-only y-basis evaluation.

    variants: library (cached evaluate -> evaluate_with_derivative),
    value_only_uncached (B only, no cache), with_dB_uncached (B and dB built
    as the library does, but not cached) -> separates derivative work from
    cache insertion.
    """
    yb: BernsteinBasis = model.basis.y_basis
    k = yb.order
    a, b = yb.support
    i = np.arange(k + 1, dtype=float)
    binom = comb(k, i, exact=False)
    i1 = np.arange(k, dtype=float)
    binom1 = comb(k - 1, i1, exact=False)

    def val(y: np.ndarray) -> np.ndarray:
        t = mb._normalize_and_validate_support(y, yb.support)
        return cast(np.ndarray, binom * t[:, None] ** i * (1.0 - t[:, None]) ** (k - i))

    def val_db(y: np.ndarray) -> np.ndarray:
        t = mb._normalize_and_validate_support(y, yb.support)
        B = binom * t[:, None] ** i * (1.0 - t[:, None]) ** (k - i)
        Bl = binom1 * t[:, None] ** i1 * (1.0 - t[:, None]) ** (k - 1 - i1)
        Bp = np.pad(Bl, ((0, 0), (1, 1)))
        _dB = k * (Bp[:, :-1] - Bp[:, 1:]) / (b - a)
        return cast(np.ndarray, B)

    res: dict[str, Any] = {}
    for m in M_SIZES:
        inp = case.ops["quantile"](np.random.default_rng(4000 + m), m)
        clear_caches()
        lib = call(model, inp)
        row: dict[str, Any] = {"m": m}
        for name, fn in (("library", None), ("value_only_uncached", val),
                         ("with_dB_uncached", val_db)):
            ts = []
            for _ in range(5 if m < 100_000 else 3):
                clear_caches()
                if fn is not None:
                    yb.evaluate = fn  # type: ignore[method-assign]
                try:
                    t0 = time.perf_counter()
                    o = call(model, inp)
                    ts.append(time.perf_counter() - t0)
                finally:
                    if fn is not None:
                        del yb.evaluate
            clear_caches()
            if fn is not None:
                yb.evaluate = fn  # type: ignore[method-assign]
            tracemalloc.start()
            base = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            o = call(model, inp)
            del_o = o.copy()
            del o
            gc.collect()
            cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if fn is not None:
                del yb.evaluate
            row[name] = {**summarize(ts),
                         "bitwise_equal_library": bool(np.array_equal(del_o, lib)),
                         "peak_mib": (peak - base) / 2**20,
                         "retained_mib": (cur - base) / 2**20,
                         "cache_after": cache_stats()}
        res[str(m)] = row
    return res


def proto_shift_value_only(model: Any, case: Case) -> dict[str, Any]:
    """Shift CDF: share of predict() spent in the unused derivative()."""
    res = {}
    for m in M_SIZES:
        inp = case.ops["distribution"](np.random.default_rng(6000 + m), m)
        y = np.asarray(inp["y_new"], dtype=float)
        tot, der = [], []
        for _ in range(5):
            clear_caches()
            t0 = time.perf_counter()
            call(model, inp)
            tot.append(time.perf_counter() - t0)
            # derivative() after evaluate() — exactly what predict() does
            t0 = time.perf_counter()
            model.basis.derivative(y, order=1)
            der.append(time.perf_counter() - t0)
        res[str(m)] = {"predict_new_data_ms": 1e3 * statistics.median(tot),
                       "derivative_call_ms": 1e3 * statistics.median(der)}
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def env() -> dict[str, Any]:
    cfgs = io.StringIO()
    try:
        from contextlib import redirect_stdout

        with redirect_stdout(cfgs):
            np.show_config()
    except Exception:  # noqa: BLE001 - environment capture is best effort
        pass
    blas = "accelerate" if "accelerate" in cfgs.getvalue().lower() else "other"
    import subprocess

    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "machine": platform.machine(),
        "platform": platform.platform(), "blas": blas, "git_sha": sha,
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "veclib_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    }


def run(case: Case) -> dict[str, Any]:
    out: dict[str, Any] = {"case": case.name, "path": case.path, "env": env()}
    out["rss_start_mib"] = rss_mib()
    model, info = fit_and_validate(case)
    out["fit"] = info
    out["model_mib"] = model_bytes(model)
    out["cache_after_fit"] = cache_stats()

    # fit-entry eviction: score() before / after a prediction burst
    sc = score_input(case, model)
    model.fit(*case.fit_args, **case.fit_kwargs)  # repopulate fit entries
    fit_keys_raw = set(mb._bernstein_cache)
    fit_keys_asm = set(mb._bernstein_assembled_cache)
    t_hot = []
    for _ in range(5):
        t0 = time.perf_counter()
        sc()
        t_hot.append(time.perf_counter() - t0)
    q_op = next(o for o in case.ops if o.startswith("quantile"))
    for i in range(BURST):
        call(model, case.ops["distribution"](np.random.default_rng(9000 + i), 1000))
        call(model, case.ops[q_op](np.random.default_rng(9500 + i), 100))
    surv_raw = len(fit_keys_raw & set(mb._bernstein_cache))
    surv_asm = len(fit_keys_asm & set(mb._bernstein_assembled_cache))
    t0 = time.perf_counter()
    sc()
    t_cold = time.perf_counter() - t0
    out["fit_entry_eviction"] = {
        "fit_raw_entries": len(fit_keys_raw), "fit_asm_entries": len(fit_keys_asm),
        "surviving_raw": surv_raw, "surviving_asm": surv_asm,
        "score_hot_ms": 1e3 * statistics.median(t_hot),
        "score_after_burst_ms": 1e3 * t_cold,
    }

    cells: dict[str, list[dict[str, Any]]] = {}
    rstyle_path = case.path.startswith("rstyle")
    for op, make in case.ops.items():
        rows = []
        for m in M_SIZES:
            if rstyle_path and (op.startswith("quantile") or op.startswith("simulate")) \
                    and m > RSTYLE_MAX_M:
                continue
            reps = 5 if m < 100_000 else 3
            rows.append(measure_cell(model, make, m, reps))
        cells[op] = rows
        print(f"  {case.name}:{op} done", file=sys.stderr, flush=True)
    out["cells"] = cells
    out["instrumented"] = {op: instrumented(model, mk, 1000) for op, mk in case.ops.items()}
    q_ops = [o for o in case.ops if o.startswith("quantile") or o.startswith("simulate")]
    out["profile_quantile_m1000"] = profile_top(
        model, case.ops[q_ops[0]](np.random.default_rng(8), 1000))
    if rstyle_path:
        out["proto_rstyle"] = proto_rstyle(model, case, q_ops)
    if case.path == "interaction":
        out["proto_interaction"] = proto_interaction(model, case)
    if "distribution" in case.ops and case.path.startswith(("shift", "rstyle")):
        out["proto_shift_derivative_share"] = proto_shift_value_only(model, case)
    out["key_cost"] = key_cost()
    out["rss_end_mib"] = rss_mib()
    out["maxrss_mib"] = __import__("resource").getrusage(
        __import__("resource").RUSAGE_SELF).ru_maxrss / 2**20
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cases = build_cases()
    if a.case == "list":
        print("\n".join(cases))
        return
    res = run(cases[a.case])
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
