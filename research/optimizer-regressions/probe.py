"""Issue #171 evidence probe; run from repo root with the pinned src on PYTHONPATH."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import signal
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import numpy as np
import scipy
from scipy import stats

import mltpy._auglag as al
import mltpy.optimizer as opt
from mltpy.basis import BernsteinBasis
from mltpy.likelihood import negative_log_likelihood
from mltpy.variables import CensoredData, CensoringType

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "reference"
BASELINE = "2659daca9057ca7aa4f77589565dcbe8bada1297"
# Any is needed for heterogeneous recorded solver fields and optimize kwargs.
type Record = dict[str, Any]


def load(stem: str) -> np.ndarray:
    return np.loadtxt(REF / f"{stem}.txt")


def cases() -> list[Record]:
    result = []
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(150, 1))
    y = rng.exponential(size=150) / (2 * np.exp(0.3 * xs[:, 0]))
    xd = np.random.default_rng(50).uniform(size=(150, 1))
    c = float(np.quantile(y, 0.8))
    observed, event = np.minimum(y, c), y <= c
    for dist in ("exponential", "normal", "logistic"):
        for flipped in (True, False):
            flags = event if flipped else ~event
            data = CensoredData.right_censored(observed, flags)
            assert np.array_equal(np.isnan(data.exact), flags)
            result.append(
                dict(
                    name=f"{'161' if dist == 'exponential' else '162'}_{dist}_"
                    f"{'literal' if flipped else 'corrected'}",
                    basis=BernsteinBasis(4, (0.0, c * 1.01)),
                    y=data,
                    X=None if dist == "exponential" else xd,
                    scaling=xs,
                    censoring=CensoringType.RIGHT,
                    base_distribution=dist,
                )
            )
    grid = np.quantile(y, np.linspace(0.0, 0.8, 5)[1:])
    k = np.searchsorted(grid, y)
    lo, hi = np.r_[0.0, grid][k], np.r_[grid, np.inf][k]
    assert np.all((lo <= y) & (y <= hi))
    for mode, data, support in [
        (CensoringType.NONE, y, (0.0, float(y.max()) * 1.01)),
        (
            CensoringType.LEFT,
            CensoredData.left_censored(
                np.maximum(y, np.quantile(y, 0.2)), y <= np.quantile(y, 0.2)
            ),
            (0.0, float(y.max()) * 1.01),
        ),
        (
            CensoringType.INTERVAL,
            CensoredData.interval_censored(lo, hi),
            (0.0, float(grid[-1]) * 1.01),
        ),
    ]:
        result.append(
            dict(
                name=f"163_{mode.name}",
                basis=BernsteinBasis(4, support),
                y=data,
                X=None,
                scaling=xs,
                censoring=mode,
                base_distribution="exponential",
            )
        )
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, (200, 1))
    uniform_y = rng.uniform(0.05, 0.95, 200)
    for off in (-0.5, 0.5):
        result.append(
            dict(
                name=f"160_offset_{off:+}",
                basis=BernsteinBasis(3, (0.0, 1.0)),
                y=uniform_y,
                X=x,
                offset=np.full(200, off),
                base_distribution="exponential",
            )
        )
    stem = "mlt_exponential_scaling_shift"
    for scaled in (False, True):
        result.append(
            dict(
                name=f"165_{'scaled' if scaled else 'unscaled'}",
                basis=BernsteinBasis(4, tuple(load(stem + "_support"))),
                y=load(stem + "_y"),
                X=load(stem + "_xd")[:, None],
                scaling=load(stem + "_xs")[:, None] if scaled else None,
                offset=load(stem + "_offset"),
                base_distribution="exponential",
            )
        )
    # Wider offset/censoring coverage on the same inspection design, no new draw.
    for shift in (False, True):
        for off in (-0.2, 0.2):
            result.append(
                dict(
                    name=f"coverage_interval_shift_{shift}_offset_{off:+}",
                    basis=BernsteinBasis(4, (0.0, grid[-1] * 1.01)),
                    y=CensoredData.interval_censored(lo, hi),
                    X=xd if shift else None,
                    scaling=xs,
                    offset=np.full(150, off),
                    censoring=CensoringType.INTERVAL,
                    base_distribution="exponential",
                )
            )
    return result


def independent(case: Record, theta: np.ndarray) -> Record:
    """Unclipped likelihood using scipy distributions and the actual row encoding."""
    basis, data = case["basis"], case["y"]
    p = basis.order + 1
    n = data.n if isinstance(data, CensoredData) else len(data)
    x, xs = case.get("X"), case.get("scaling")
    q = 0 if x is None else x.shape[1]
    factor = np.ones(n) if xs is None else np.exp(0.5 * (xs @ theta[p + q :]))
    shift = np.zeros(n) if x is None else x @ theta[p : p + q]
    shift = shift + case.get("offset", 0.0)
    dist = {
        "exponential": stats.expon,
        "normal": stats.norm,
        "logistic": stats.logistic,
    }[case["base_distribution"]]

    def h(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.asarray((basis.evaluate(y) @ theta[:p]) * factor[mask] + shift[mask])

    cd = data if isinstance(data, CensoredData) else CensoredData.from_exact(data)
    exact = np.isfinite(cd.exact)
    ll = np.zeros(n)
    raw_h = h(cd.exact[exact], exact)
    hp = (basis.derivative(cd.exact[exact]) @ theta[:p]) * factor[exact]
    with np.errstate(divide="ignore", invalid="ignore"):
        ll[exact] = dist.logpdf(raw_h) + np.log(hp)
        right = ~exact & np.isposinf(cd.upper)
        left = ~exact & np.isneginf(cd.lower)
        interval = ~exact & ~right & ~left
        ll[right] = dist.logsf(h(cd.lower[right], right))
        ll[left] = dist.logcdf(h(cd.upper[left], left))
        lower, upper = h(cd.lower[interval], interval), h(cd.upper[interval], interval)
        # Survival differences avoid cancellation in the exponential upper tail.
        a, b = dist.logsf(lower), dist.logsf(upper)
        ll[interval] = a + np.log(-np.expm1(b - a))
    return dict(
        true_nll=float(-ll.sum()),
        min_increment=float(np.diff(theta[:p]).min()),
        min_support=float((theta[0] * factor + shift).min()),
        exact_rows=int(exact.sum()),
        censored_rows=int((~exact).sum()),
        exact_abs_h_max=float(np.max(np.abs(raw_h), initial=0)),
        exact_rows_beyond_30=int(np.sum(np.abs(raw_h) > 30)),
    )


def warning_counts(caught: list[Any]) -> Record:
    counts = Counter(
        (w.category.__name__, Path(w.filename).name, w.lineno, str(w.message))
        for w in caught
    )
    return {" | ".join(map(str, key)): n for key, n in counts.items()}


def run(
    case: Record, solver: Literal["auglag", "slsqp", "trust-constr"], restarts: int
) -> Record:
    kwargs = {k: v for k, v in case.items() if k not in ("name", "basis", "y")}
    row: Record = dict(
        case=case["name"],
        solver=solver,
        max_restarts=restarts,
        attempts=[],
        inner_results=[],
    )
    original_loop = opt._restart_loop

    def loop(
        theta: Any,
        config: Any,
        perturb: Any,
        run_fn: Any,
        success: Any,
        nll: Any,
        **kw: Any,
    ) -> Any:
        def recorded(start: np.ndarray) -> Any:
            attempt: Record = dict(start=start.copy())
            row["attempts"].append(attempt)
            try:
                res = run_fn(start)
                attempt.update(
                    raw={
                        k: getattr(res, k)
                        for k in (
                            "x",
                            "theta",
                            "fun",
                            "success",
                            "converged",
                            "status",
                            "message",
                            "nit",
                            "n_outer_iter",
                            "kkt_residual",
                            "rho_final",
                        )
                        if hasattr(res, k)
                    }
                )
                return res
            except Exception as exc:
                attempt["error"] = repr(exc)
                raise

        return original_loop(theta, config, perturb, recorded, success, nll, **kw)

    def spy_minimize(original: Any) -> Any:
        def recorded(*args: Any, **kw: Any) -> Any:
            res = original(*args, **kw)
            row["inner_results"].append(
                {
                    k: getattr(res, k)
                    for k in (
                        "nfev",
                        "njev",
                        "nhev",
                        "nit",
                        "status",
                        "success",
                        "message",
                    )
                    if hasattr(res, k)
                }
            )
            return res

        return recorded

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with (
                patch.object(opt, "_restart_loop", loop),
                patch.object(opt, "minimize", spy_minimize(getattr(opt, "minimize"))),
                patch.object(al, "minimize", spy_minimize(getattr(al, "minimize"))),
            ):
                res = opt.optimize(
                    case["basis"],
                    case["y"],
                    **kwargs,
                    config=opt.OptimizerConfig(
                        solver=solver, random_state=0, max_restarts=restarts
                    ),
                )
            row.update(
                theta=res.theta,
                reported_nll=-res.log_likelihood,
                converged=res.converged,
                iterations=res.n_iter,
                restarts=res.n_restarts,
                message=res.solver_message,
                kkt=res.kkt_residual,
                outer_iterations=res.n_outer_iter,
            )
        except Exception as exc:
            row["error"] = repr(exc)
    row.update(seconds=time.perf_counter() - start, fit_warnings=warning_counts(caught))
    if "theta" in row:
        with warnings.catch_warnings(record=True) as final_warnings:
            warnings.simplefilter("always")
            try:
                evaluation = negative_log_likelihood(
                    row["theta"], case["basis"], case["y"], gradient=True, **kwargs
                )
                assert isinstance(evaluation, tuple)
                value, grad = evaluation
                row.update(
                    final_nll=value,
                    finite_gradient=bool(np.isfinite(grad).all()),
                    **independent(case, row["theta"]),
                )
            except Exception as exc:
                row["final_error"] = repr(exc)
        row["final_warnings"] = warning_counts(final_warnings)
    return row


def clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, CensoredData):
        return clean(vars(value))
    if isinstance(value, CensoringType):
        return value.name
    if isinstance(value, BernsteinBasis):
        return dict(order=value.order, support=value.support)
    return value


def self_check() -> Record:
    """Check worked likelihoods and the separate exact-row clipping concern."""
    basis = BernsteinBasis(1, (0.0, 1.0))
    case = dict(basis=basis, y=np.array([0.25, 0.75]), base_distribution="exponential")
    theta = np.array([0.0, 2.0])
    np.testing.assert_allclose(
        independent(case, theta)["true_nll"], 2.0 - 2.0 * np.log(2.0)
    )
    case["y"] = CensoredData.right_censored(np.array([0.5]), np.array([True]))
    np.testing.assert_allclose(independent(case, theta)["true_nll"], 1.0)
    case["y"] = CensoredData.interval_censored(np.array([0.0]), np.array([0.5]))
    np.testing.assert_allclose(
        independent(case, theta)["true_nll"], -np.log1p(-np.exp(-1.0))
    )
    # Exact h beyond the clip is a distinct question from the reversed flags.
    clip_case: Record = dict(basis=basis, y=np.array([0.5]), base_distribution="normal")
    clip_theta = np.array([40.0, 41.0])
    true = independent(clip_case, clip_theta)["true_nll"]
    np.testing.assert_allclose(true, 0.5 * 40.5**2 + 0.5 * np.log(2 * np.pi))
    evaluated = negative_log_likelihood(clip_theta, basis, clip_case["y"])
    assert isinstance(evaluated, float)
    np.testing.assert_allclose(evaluated, 0.5 * 30**2 + 0.5 * np.log(2 * np.pi))
    historical = []
    for dist in ("normal", "logistic"):
        stem = f"scaling_fixed_time_flipped_{dist}"
        historical_theta = load(stem + "_mltpy_theta")
        historical_case: Record = dict(
            basis=BernsteinBasis(4, tuple(load("scaling_fixed_time_support"))),
            X=load("scaling_fixed_time_x_d")[:, None],
            scaling=load("scaling_fixed_time_x_s")[:, None],
            base_distribution=dist,
        )
        event = load("scaling_fixed_time_event").astype(bool)
        for literal in (True, False):
            historical_case["y"] = CensoredData.right_censored(
                load("scaling_fixed_time_y"), event if literal else ~event
            )
            check = independent(historical_case, historical_theta)
            if literal:
                np.testing.assert_allclose(
                    check["true_nll"],
                    -float(load(stem + "_loglik_at_mltpy")),
                    rtol=1e-6,
                )
            historical.append(dict(dist=dist, literal=literal, **check))
    return dict(
        description="separate exact normal boundary probe, not a fitted optimum",
        historical_162=historical,
        theta=clip_theta,
        true_nll=true,
        evaluated_nll=evaluated,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--case", help="Optional case-name substring")
    args = parser.parse_args()
    clip_probe = self_check()
    if args.self_check:
        print(json.dumps(clean(clip_probe), indent=2))
        return
    if args.out is None:
        parser.error("--out is required unless --self-check is used")
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    library_source = Path(opt.__file__).resolve().parent
    library_checkout = subprocess.check_output(
        ["git", "-C", str(library_source), "rev-parse", "HEAD"], text=True
    ).strip()
    source_hash = hashlib.sha256()
    for path in sorted(library_source.glob("*.py")):
        source_hash.update(path.read_bytes())
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    output: Record = dict(
        clipping_probe=clip_probe,
        baseline=BASELINE,
        checkout=commit,
        library_checkout=library_checkout,
        library_source=str(library_source),
        source_sha256=source_hash.hexdigest(),
        python=sys.version,
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        blas=config.getvalue(),
        threads={
            k: os.environ.get(k)
            for k in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        data_seed=0,
        shift_seed=50,
        restart_seed=0,
        timeout_seconds=90,
        config=clean(vars(opt.OptimizerConfig(random_state=0))),
        cases=[],
        results=[],
    )

    def timeout(*_: Any) -> None:
        raise TimeoutError("90-second evidence budget exceeded")

    signal.signal(signal.SIGALRM, timeout)
    for case in cases():
        if args.case and args.case not in case["name"]:
            continue
        output["cases"].append(clean(case))
        solvers: tuple[Literal["auglag", "slsqp", "trust-constr"], ...] = (
            "auglag",
            "slsqp",
            "trust-constr",
        )
        for solver in solvers:
            for restarts in (0, 3):
                signal.alarm(90)
                try:
                    row = run(case, solver, restarts)
                finally:
                    signal.alarm(0)
                output["results"].append(row)
                args.out.write_text(
                    json.dumps(clean(output), indent=2, allow_nan=False) + "\n"
                )
                print(
                    case["name"],
                    solver,
                    restarts,
                    row.get("converged"),
                    row.get("reported_nll", row.get("error")),
                    flush=True,
                )


if __name__ == "__main__":
    main()
