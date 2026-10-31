"""Compare research Hessian routes without changing the installed library.

Run from the repository root with its .venv; see README.md for limitations.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Any  # Existing baseline models and heterogeneous JSON are dynamic.

import numpy as np
import scipy
from numpy.typing import NDArray

import mltpy.likelihood as likelihood
import mltpy.optimizer as optimizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fit_baseline", ROOT / "research/fit-phase-baseline/harness.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load the existing fit baseline harness")
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)

Array = NDArray[np.float64]


def analytic(
    kron: likelihood._InteractionDesign,
    theta: Array,
    dist: likelihood.DistOps,
    weights: Array | None = None,
    offset: Array | None = None,
    scaling: Array | None = None,
) -> Array:
    """Smooth-region research formula; deliberately refuses clipping/kinks."""
    A, D = kron.design, kron.d_design
    k = A.shape[1]
    g, gp = A @ theta[:k], D @ theta[:k]
    f = np.ones_like(g) if scaling is None else np.exp(0.5 * (scaling @ theta[k:]))
    h = g * f
    if offset is not None:
        h = h + offset
    if np.any(np.abs(h) >= likelihood._H_CLIP):
        raise ValueError("Research analytic candidate excludes clipped transformations")
    if dist.kind == "laplace" and np.any(h == 0):
        raise ValueError("Laplace has no classical Hessian at h=0")
    if np.any(gp <= 0):
        raise ValueError("Research candidate requires a positive response derivative")
    w = np.ones_like(g) if weights is None else weights
    c = -likelihood._d2_logpdf(h, dist)
    H = np.empty((len(theta), len(theta)))
    H[:k, :k] = A.T @ ((w * c * f**2)[:, None] * A)
    H[:k, :k] += D.T @ ((w / gp**2)[:, None] * D)
    if scaling is not None:
        t = c * g * f + likelihood._neg_score(h, dist)
        H[:k, k:] = A.T @ ((0.5 * w * f * t)[:, None] * scaling)
        H[k:, :k] = H[:k, k:].T
        H[k:, k:] = scaling.T @ ((0.25 * w * g * f * t)[:, None] * scaling)
    return H


def gradient_difference(
    kron: likelihood._InteractionDesign,
    theta: Array,
    dist: likelihood.DistOps,
    weights: Array | None = None,
    offset: Array | None = None,
    scaling: Array | None = None,
) -> Array:
    """Excluded production route, measured here against the analytic reference."""
    H = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = 1e-5
        args = dict(dist=dist, weights=weights, offset=offset, scaling=scaling)
        _, plus = likelihood._ll_and_grad_interaction_none(kron, theta + e, **args)
        _, minus = likelihood._ll_and_grad_interaction_none(kron, theta - e, **args)
        H[:, j] = (plus - minus) / 2e-5
    return 0.5 * (H + H.T)


def errors(H: Array, reference: Array) -> dict[str, Any]:
    """Covariance/SE comparisons are meaningful only for positive-definite H."""
    result: dict[str, Any] = {
        "relative_max_H_error": float(
            np.max(np.abs(H - reference)) / np.max(np.abs(reference))
        ),
        "min_eigenvalue": float(np.linalg.eigvalsh(H)[0]),
    }
    if result["min_eigenvalue"] > 0 and np.linalg.eigvalsh(reference)[0] > 0:
        V, Vref = np.linalg.inv(H), np.linalg.inv(reference)
        result["relative_max_vcov_error"] = float(
            np.max(np.abs(V - Vref)) / np.max(np.abs(Vref))
        )
        result["max_relative_SE_error"] = float(
            np.max(np.abs(np.sqrt(np.diag(V) / np.diag(Vref)) - 1))
        )
    else:
        result["SE_status"] = "undefined: information is not positive definite"
    return result


def main() -> None:
    cases = {n: c for n, c in baseline.build_cases().items() if n.startswith("ia_")}
    for n in (2000, 10000):
        y, X = baseline._ia_data(n)
        name = f"ia_bs6x3_m28_n{n}"
        cases[name] = baseline.Case(
            name, "interaction", baseline._ia_model(6, 3), (y, X), n=n
        )
    original = likelihood._hessian_interaction_fd
    routes = {
        "scalar_fd": original,
        "gradient_fd": gradient_difference,
        "analytic": analytic,
    }
    output: dict[str, Any] = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {
            n: os.environ.get(n)
            for n in (
                "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "reps": 5,
        "cases": {},
    }
    try:
        for name, case in cases.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = case.make().fit(*case.fit_args, **case.fit_kwargs)
            inp = baseline._fit_inputs(case, model)
            kron = likelihood._InteractionDesign.build(model.basis, inp["y"], inp["X"])
            args = dict(
                dist=likelihood._get_dist(model.base_distribution),
                scaling=model.scaling,
            )
            Href = analytic(kron, model.theta_, **args)
            g = kron.design @ model.theta_[: kron.design.shape[1]]
            if model.scaling is not None:
                g = g * np.exp(0.5 * (model.scaling @ model.gamma_))
            result: dict[str, Any] = {
                "n": case.n,
                "m": len(model.theta_),
                "condition_analytic": float(np.linalg.cond(Href)),
                "min_eigenvalue_analytic": float(np.linalg.eigvalsh(Href)[0]),
                "max_abs_h": float(np.max(np.abs(g))),
                "baseline_validation": baseline.validate(
                    case, model, baseline.Phases()
                ),
                "warnings": [str(w.message) for w in caught],
                "routes": {},
            }
            # Same-theta R checks consume these rather than refitting a different MLE.
            np.savetxt(HERE / "results" / f"{name}_theta.txt", model.theta_)
            np.savetxt(HERE / "results" / f"{name}_H.txt", Href)
            # Interleave routes per repetition to reduce timing drift.
            timings = {route: [] for route in routes}
            for _ in range(5):
                for route, fn in routes.items():
                    start = time.perf_counter()
                    H = fn(kron, model.theta_, **args)
                    timings[route].append(1000 * (time.perf_counter() - start))
                    result["routes"][route] = errors(H, Href)
            for route, fn in routes.items():
                record = result["routes"][route]
                record["kernel_ms"] = timings[route]
                likelihood._hessian_interaction_fd = fn
                optimizer._hessian_interaction_fd = fn
                fits = []
                for _ in range(6):  # first route fit excluded from warm median
                    with warnings.catch_warnings(record=True) as fit_warnings:
                        warnings.simplefilter("always")
                        start = time.perf_counter()
                        fitted = case.make().fit(*case.fit_args, **case.fit_kwargs)
                        seconds = time.perf_counter() - start
                    fits.append(
                        {
                            "ms": seconds * 1000,
                            "converged": bool(fitted.result_.converged),
                            "max_abs_theta_change": float(
                                np.max(np.abs(fitted.theta_ - model.theta_))
                            ),
                            "ll_change": float(
                                fitted.result_.log_likelihood
                                - model.result_.log_likelihood
                            ),
                            "validation": baseline.kkt_check(
                                fitted.theta_, fitted._A_ineq_, inp, case.n
                            ),
                            "warnings": [str(w.message) for w in fit_warnings],
                        }
                    )
                record["fits"] = fits
                record["fit_warm_median_ms"] = float(
                    np.median([f["ms"] for f in fits[1:]])
                )
                tracemalloc.start()
                case.make().fit(*case.fit_args, **case.fit_kwargs)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                record["peak_MiB"] = peak / 2**20
            likelihood._hessian_interaction_fd = original
            optimizer._hessian_interaction_fd = original
            output["cases"][name] = result
            print(
                name,
                {
                    r: round(v["fit_warm_median_ms"], 3)
                    for r, v in result["routes"].items()
                },
                flush=True,
            )
            (HERE / "results" / "measurements.json").write_text(
                json.dumps(output, indent=2) + "\n"
            )
    finally:
        likelihood._hessian_interaction_fd = original
        optimizer._hessian_interaction_fd = original


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    main()
