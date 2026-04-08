#!/usr/bin/env python3
"""Validate pymlt against R mlt/tram reference values.

Iterates over all cases in validation/references/, fits the corresponding
pymlt model, and compares theta, log-likelihood, and CDF predictions
against the R reference.

Usage:
    python validation/run_validation.py
    python validation/run_validation.py --case case_01 --verbose

Exit codes:
    0 — all cases pass tolerance checks
    1 — one or more cases exceed tolerances
    2 — no reference data found
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

TOL_THETA = 0.05  # max absolute component-wise difference
TOL_LOGLIK = 0.1  # absolute difference in log-likelihood
TOL_CDF = 0.02  # max absolute CDF difference at grid points

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReferenceCase:
    """Reference data loaded from a single case directory."""

    case_id: str
    model: str
    censoring: str
    n: int
    order: int
    support: tuple[float, float]
    y: NDArray[np.float64]
    theta_r: NDArray[np.float64]
    loglik_r: float
    cdf_grid: NDArray[np.float64]
    cdf_values_r: NDArray[np.float64]
    status: NDArray[np.bool_] | None = None
    y_lower: NDArray[np.float64] | None = None
    y_upper: NDArray[np.float64] | None = None
    X: NDArray[np.float64] | None = None
    regression: bool = False
    n_covariates: int = 0


@dataclass
class FittedResult:
    """Result of fitting a pymlt model on reference data."""

    theta_py: NDArray[np.float64]
    loglik_py: float
    cdf_values_py: NDArray[np.float64]
    converged: bool
    runtime_s: float


@dataclass
class ValidationResult:
    """Comparison verdict for one case."""

    case_id: str
    model: str
    n: int
    order: int
    passed: bool
    max_delta_theta: float
    delta_loglik: float
    max_delta_cdf: float
    converged: bool
    runtime_s: float
    failure_reason: str | None = None


# ---------------------------------------------------------------------------
# load_reference
# ---------------------------------------------------------------------------


def load_reference(case_dir: Path) -> ReferenceCase:
    """Load all .npy files and metadata.json from a case directory.

    Parameters
    ----------
    case_dir:
        Path to a ``case_*`` directory containing .npy files and metadata.json.

    Returns
    -------
    ReferenceCase
        Populated dataclass with all reference data.

    Raises
    ------
    FileNotFoundError
        If required files are missing.
    """
    with open(case_dir / "metadata.json") as f:
        meta = json.load(f)

    sup = meta["support"]

    kwargs: dict[str, object] = {
        "case_id": case_dir.name,
        "model": meta["model"],
        "censoring": meta["censoring"],
        "n": meta["n"],
        "order": meta["order"],
        "support": (float(sup[0]), float(sup[1])),
        "theta_r": np.load(case_dir / "theta.npy"),
        "loglik_r": float(np.load(case_dir / "loglik.npy")),
        "cdf_grid": np.load(case_dir / "cdf_grid.npy"),
        "cdf_values_r": np.load(case_dir / "cdf_values.npy"),
    }

    # y — present for all cases except pure interval censoring
    y_path = case_dir / "y.npy"
    if y_path.exists():
        kwargs["y"] = np.load(y_path)
    else:
        # interval-censored: no y.npy, use midpoints for placeholder
        kwargs["y"] = np.array([], dtype=np.float64)

    # Optional arrays
    status_path = case_dir / "status.npy"
    if status_path.exists():
        kwargs["status"] = np.load(status_path)

    for name in ("y_lower", "y_upper", "X"):
        p = case_dir / f"{name}.npy"
        if p.exists():
            kwargs[name] = np.load(p)

    if meta.get("regression", False):
        kwargs["regression"] = True
        kwargs["n_covariates"] = meta.get("n_covariates", 0)

    return ReferenceCase(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# fit_python_model
# ---------------------------------------------------------------------------

_FAILED_FIT = FittedResult(
    theta_py=np.array([]),
    loglik_py=float("nan"),
    cdf_values_py=np.array([]),
    converged=False,
    runtime_s=0.0,
)


def fit_python_model(case: ReferenceCase) -> FittedResult:
    """Instantiate and fit the correct pymlt model for *case*.

    All exceptions are caught — a failed fit returns a ``FittedResult``
    with ``converged=False`` so the script never aborts on a single case.

    Parameters
    ----------
    case:
        Reference data specifying model type, censoring, order, support, etc.

    Returns
    -------
    FittedResult
    """
    # Lazy import so the script can be syntax-checked without pymlt installed
    try:
        from pymlt.model import MLT
        from pymlt.tram import BoxCox, Colr, Coxph
        from pymlt.variables import CensoredData, CensoringType
    except ImportError as e:
        print(f"  ERROR: pymlt not importable: {e}", file=sys.stderr)
        return _FAILED_FIT

    try:
        t0 = time.perf_counter()

        model_name = case.model
        cens = case.censoring
        support = case.support
        order = case.order

        # --- Instantiate model ---
        if model_name == "boxcox":
            model = BoxCox(support=support, order=order)
        elif model_name == "coxph":
            model = Coxph(support=support, order=order)
        elif model_name == "colr":
            model = Colr(support=support, order=order)
        elif model_name == "mlt":
            censoring_map = {
                "none": CensoringType.NONE,
                "right": CensoringType.RIGHT,
                "left": CensoringType.LEFT,
                "interval": CensoringType.INTERVAL,
            }
            model = MLT(
                order=order,
                support=support,
                censoring=censoring_map[cens],
            )
        else:
            print(f"  WARNING: unknown model '{model_name}'", file=sys.stderr)
            return _FAILED_FIT

        # --- Prepare data ---
        y_or_cd: object
        X_fit: NDArray[np.float64] | None = None

        if cens == "right":
            assert case.status is not None
            censored = ~case.status.astype(bool)
            y_or_cd = CensoredData.right_censored(case.y, censored=censored)
        elif cens == "left":
            assert case.status is not None
            censored = ~case.status.astype(bool)
            y_or_cd = CensoredData.left_censored(case.y, censored=censored)
        elif cens == "interval":
            assert case.y_lower is not None and case.y_upper is not None
            y_or_cd = CensoredData.interval_censored(case.y_lower, case.y_upper)
        else:
            y_or_cd = case.y

        if case.regression and case.X is not None:
            X_fit = case.X

        # --- Fit ---
        model.fit(y_or_cd, X=X_fit)  # type: ignore[arg-type]

        # --- Extract results ---
        assert model.theta_ is not None
        assert model.result_ is not None
        theta_py = model.theta_
        loglik_py = model.result_.log_likelihood
        converged = model.result_.converged

        # --- CDF at grid points ---
        X_pred: NDArray[np.float64] | None = None
        if case.regression:
            n_grid = len(case.cdf_grid)
            X_pred = np.zeros((n_grid, case.n_covariates))

        cdf_py = model.predict(case.cdf_grid, X_new=X_pred, what="distribution")

        elapsed = time.perf_counter() - t0
        return FittedResult(
            theta_py=theta_py,
            loglik_py=loglik_py,
            cdf_values_py=cdf_py,
            converged=converged,
            runtime_s=elapsed,
        )

    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ERROR fitting {case.case_id}: {e}", file=sys.stderr)
        return FittedResult(
            theta_py=np.array([]),
            loglik_py=float("nan"),
            cdf_values_py=np.array([]),
            converged=False,
            runtime_s=elapsed,
        )


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------


def compare_results(
    ref: ReferenceCase,
    fit: FittedResult,
    verbose: bool = False,
) -> ValidationResult:
    """Compare pymlt fit against R reference values.

    Parameters
    ----------
    ref:
        Reference data from R.
    fit:
        pymlt fit result.
    verbose:
        If True, print per-component details for failing cases.

    Returns
    -------
    ValidationResult
    """
    if not fit.converged or len(fit.theta_py) == 0:
        return ValidationResult(
            case_id=ref.case_id,
            model=ref.model,
            n=ref.n,
            order=ref.order,
            passed=False,
            max_delta_theta=float("nan"),
            delta_loglik=float("nan"),
            max_delta_cdf=float("nan"),
            converged=fit.converged,
            runtime_s=fit.runtime_s,
            failure_reason="fit failed or did not converge",
        )

    delta_theta = np.abs(fit.theta_py - ref.theta_r)
    max_delta_theta = float(np.max(delta_theta))
    delta_loglik = abs(fit.loglik_py - ref.loglik_r)

    if len(fit.cdf_values_py) == len(ref.cdf_values_r):
        delta_cdf = np.abs(fit.cdf_values_py - ref.cdf_values_r)
        max_delta_cdf = float(np.max(delta_cdf))
    else:
        max_delta_cdf = float("nan")

    failures: list[str] = []
    if max_delta_theta > TOL_THETA:
        failures.append(f"theta ({max_delta_theta:.4f} > {TOL_THETA})")
    if delta_loglik > TOL_LOGLIK:
        failures.append(f"loglik ({delta_loglik:.4f} > {TOL_LOGLIK})")
    if max_delta_cdf > TOL_CDF:
        failures.append(f"cdf ({max_delta_cdf:.4f} > {TOL_CDF})")

    passed = len(failures) == 0
    failure_reason = "; ".join(failures) if failures else None

    if verbose and not passed:
        print(f"  {ref.case_id}:")
        for i, (r, p) in enumerate(zip(ref.theta_r, fit.theta_py)):
            d = abs(r - p)
            marker = " <<<" if d > TOL_THETA else ""
            print(f"    theta[{i}]: R={r:.6f}  py={p:.6f}  Δ={d:.4f}{marker}")
        print(
            f"    loglik: R={ref.loglik_r:.4f}  py={fit.loglik_py:.4f}"
            f"  Δ={delta_loglik:.4f}"
        )
        if len(fit.cdf_values_py) == len(ref.cdf_values_r):
            for i, (r, p) in enumerate(zip(ref.cdf_values_r, fit.cdf_values_py)):
                d = abs(r - p)
                marker = " <<<" if d > TOL_CDF else ""
                print(f"    cdf[{i}]:  R={r:.4f}  py={p:.4f}  Δ={d:.4f}{marker}")

    return ValidationResult(
        case_id=ref.case_id,
        model=ref.model,
        n=ref.n,
        order=ref.order,
        passed=passed,
        max_delta_theta=max_delta_theta,
        delta_loglik=delta_loglik,
        max_delta_cdf=max_delta_cdf,
        converged=fit.converged,
        runtime_s=fit.runtime_s,
        failure_reason=failure_reason,
    )


# ---------------------------------------------------------------------------
# run_all_validations
# ---------------------------------------------------------------------------


def run_all_validations(
    ref_dir: Path,
    case_filter: str | None = None,
    verbose: bool = False,
) -> list[ValidationResult]:
    """Run validation for all (or filtered) cases.

    Parameters
    ----------
    ref_dir:
        Path to ``validation/references/``.
    case_filter:
        If set, only run cases whose directory name starts with this prefix.
    verbose:
        If True, print per-component details for failing cases.

    Returns
    -------
    list[ValidationResult]
    """
    case_dirs = sorted(d for d in ref_dir.glob("case_*") if d.is_dir())

    if case_filter:
        case_dirs = [d for d in case_dirs if d.name.startswith(case_filter)]

    results: list[ValidationResult] = []

    for case_dir in case_dirs:
        ref = load_reference(case_dir)
        fit = fit_python_model(ref)
        result = compare_results(ref, fit, verbose=verbose)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------

_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def print_report(results: list[ValidationResult]) -> None:
    """Print a formatted table to stdout.

    Uses ANSI colors for PASS/FAIL when stdout is a terminal.
    """
    use_color = sys.stdout.isatty()

    def _color(text: str, color: str) -> str:
        if use_color:
            return f"{color}{text}{_RESET}"
        return text

    print()
    print("pymlt validation — R reference comparison")
    print("=" * 78)
    header = (
        f"{'Case':<28}│ {'Model':<7}│ {'n':>5} │ {'Ord':>3} │ {'Status':<6}"
        f"│ {'Δθ_max':>7} │ {'Δll':>7} │ {'Δcdf':>6}"
    )
    print(header)
    print("─" * 78)

    for r in results:
        if r.passed:
            status = _color("PASS", _GREEN)
        else:
            status = _color("FAIL", _RED)

        def _fmt(val: float, width: int = 7) -> str:
            if np.isnan(val):
                return "   N/A".ljust(width)
            return f"{val:{width}.4f}"

        dt = _fmt(r.max_delta_theta)
        dl = _fmt(r.delta_loglik)
        dc = _fmt(r.max_delta_cdf, 6)
        print(
            f"{r.case_id:<28}│ {r.model:<7}│ {r.n:>5} "
            f"│ {r.order:>3} │ {status:<6}│ {dt} │ {dl} │ {dc}"
        )

    print("─" * 78)
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    pct = 100.0 * n_pass / n_total if n_total > 0 else 0.0
    print(f"Total: {n_pass}/{n_total} passed ({pct:.1f}%)")
    print()


def save_report(
    results: list[ValidationResult],
    out_dir: Path,
) -> None:
    """Save report as Markdown table and JSON.

    Parameters
    ----------
    results:
        List of validation results.
    out_dir:
        Directory for output files (created if absent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Markdown ---
    lines = [
        "# pymlt Validation Report",
        "",
        "| Case | Model | n | Order | Status | Δθ_max | Δll | Δcdf |",
        "|------|-------|---|-------|--------|--------|-----|------|",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"

        def _md(val: float) -> str:
            return "N/A" if np.isnan(val) else f"{val:.4f}"

        lines.append(
            f"| {r.case_id} | {r.model} | {r.n} | {r.order} "
            f"| {status} | {_md(r.max_delta_theta)} "
            f"| {_md(r.delta_loglik)} | {_md(r.max_delta_cdf)} |"
        )

    n_pass = sum(1 for r in results if r.passed)
    lines.append("")
    lines.append(f"**{n_pass}/{len(results)} passed**")
    lines.append("")

    (out_dir / "validation_report.md").write_text("\n".join(lines))

    # --- JSON ---
    records = []
    for r in results:
        records.append(
            {
                "case_id": r.case_id,
                "model": r.model,
                "n": r.n,
                "order": r.order,
                "passed": r.passed,
                "max_delta_theta": (
                    None if np.isnan(r.max_delta_theta) else r.max_delta_theta
                ),
                "delta_loglik": (None if np.isnan(r.delta_loglik) else r.delta_loglik),
                "max_delta_cdf": (
                    None if np.isnan(r.max_delta_cdf) else r.max_delta_cdf
                ),
                "converged": r.converged,
                "runtime_s": round(r.runtime_s, 4),
                "failure_reason": r.failure_reason,
            }
        )

    (out_dir / "validation_report.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the validation runner."""
    parser = argparse.ArgumentParser(
        description="Validate pymlt against R mlt/tram reference values."
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run only cases matching this prefix (e.g. 'case_01', 'case_05_boxcox')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-component delta details for failing cases",
    )
    args = parser.parse_args()

    ref_dir = Path(__file__).parent / "references"

    if not ref_dir.is_dir():
        print(
            f"ERROR: {ref_dir} does not exist.\n"
            "Run the R script first: Rscript validation/generate_all_references.R\n"
            "Then convert:           python validation/convert_references.py",
            file=sys.stderr,
        )
        sys.exit(2)

    case_dirs = sorted(d for d in ref_dir.glob("case_*") if d.is_dir())
    if not case_dirs:
        print(f"ERROR: No case_* directories in {ref_dir}.", file=sys.stderr)
        sys.exit(2)

    results = run_all_validations(ref_dir, case_filter=args.case, verbose=args.verbose)

    if not results:
        print(f"ERROR: No cases matched filter '{args.case}'.", file=sys.stderr)
        sys.exit(2)

    print_report(results)
    save_report(results, Path(__file__).parent / "results")

    n_failed = sum(1 for r in results if not r.passed)
    if n_failed > 0:
        print(f"{n_failed} case(s) exceeded tolerance thresholds.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
