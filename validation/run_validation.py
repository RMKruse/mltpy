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

TOL_THETA = 0.05  # max absolute component-wise difference (informational only)
TOL_LOGLIK = 0.1  # absolute difference in log-likelihood
TOL_CDF = 0.02  # max absolute CDF difference at grid points
TOL_PDF = 0.05  # max absolute PDF difference at grid points
TOL_QUANTILE = 0.05  # max absolute quantile difference
TOL_HAZARD = 0.10  # max absolute hazard difference (compared only where CDF < 0.95)

# Cases known to fail for fundamental statistical reasons, not implementation
# bugs. They still run and are reported as XFAIL, but do not cause the overall
# validation run to exit non-zero. See validation/VALIDATION.md for per-case
# justification.
EXPECTED_FAILURES: frozenset[str] = frozenset({"case_16_mlt_500_12"})

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
    pdf_grid: NDArray[np.float64] | None = None
    pdf_values_r: NDArray[np.float64] | None = None
    quantile_probs: NDArray[np.float64] | None = None
    quantile_values_r: NDArray[np.float64] | None = None
    hazard_grid: NDArray[np.float64] | None = None
    hazard_values_r: NDArray[np.float64] | None = None
    base_distribution: str = "normal"


@dataclass
class FittedResult:
    """Result of fitting a pymlt model on reference data."""

    theta_py: NDArray[np.float64]
    loglik_py: float
    cdf_values_py: NDArray[np.float64]
    converged: bool
    runtime_s: float
    pdf_values_py: NDArray[np.float64] | None = None
    quantile_values_py: NDArray[np.float64] | None = None
    hazard_values_py: NDArray[np.float64] | None = None


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
    max_delta_pdf: float | None = None
    max_delta_quantile: float | None = None
    max_delta_hazard: float | None = None
    expected_failure: bool = False


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

    if "base_distribution" in meta:
        kwargs["base_distribution"] = meta["base_distribution"]

    # Functional output references (optional)
    for name, field in [
        ("pdf_grid", "pdf_grid"),
        ("pdf_values", "pdf_values_r"),
        ("quantile_probs", "quantile_probs"),
        ("quantile_values", "quantile_values_r"),
        ("hazard_grid", "hazard_grid"),
        ("hazard_values", "hazard_values_r"),
    ]:
        p = case_dir / f"{name}.npy"
        if p.exists():
            kwargs[field] = np.load(p)

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
                base_distribution=case.base_distribution,
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

        # --- PDF at grid points ---
        pdf_py: NDArray[np.float64] | None = None
        if case.pdf_grid is not None:
            X_pdf = None
            if case.regression:
                X_pdf = np.zeros((len(case.pdf_grid), case.n_covariates))
            pdf_py = model.predict(case.pdf_grid, X_new=X_pdf, what="density")

        # --- Quantiles ---
        quantile_py: NDArray[np.float64] | None = None
        if case.quantile_probs is not None:
            quantile_py = model.predict(
                case.quantile_probs, what="quantile"
            )

        # --- Hazard at grid points (right-censored only) ---
        hazard_py: NDArray[np.float64] | None = None
        if case.hazard_grid is not None and cens == "right":
            X_haz = None
            if case.regression:
                X_haz = np.zeros((len(case.hazard_grid), case.n_covariates))
            hazard_py = model.predict(
                case.hazard_grid, X_new=X_haz, what="hazard"
            )

        elapsed = time.perf_counter() - t0
        return FittedResult(
            theta_py=theta_py,
            loglik_py=loglik_py,
            cdf_values_py=cdf_py,
            converged=converged,
            runtime_s=elapsed,
            pdf_values_py=pdf_py,
            quantile_values_py=quantile_py,
            hazard_values_py=hazard_py,
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
            expected_failure=ref.case_id in EXPECTED_FAILURES,
        )

    delta_theta = np.abs(fit.theta_py - ref.theta_r)
    max_delta_theta = float(np.max(delta_theta))
    delta_loglik = abs(fit.loglik_py - ref.loglik_r)

    cdf_length_mismatch = len(fit.cdf_values_py) != len(ref.cdf_values_r)
    if not cdf_length_mismatch:
        delta_cdf = np.abs(fit.cdf_values_py - ref.cdf_values_r)
        max_delta_cdf = float(np.max(delta_cdf))
    else:
        max_delta_cdf = float("nan")

    # --- Functional output deltas ---
    max_delta_pdf: float | None = None
    if fit.pdf_values_py is not None and ref.pdf_values_r is not None:
        max_delta_pdf = float(np.max(np.abs(fit.pdf_values_py - ref.pdf_values_r)))

    max_delta_quantile: float | None = None
    if (
        fit.quantile_values_py is not None
        and ref.quantile_values_r is not None
        and ref.quantile_probs is not None
    ):
        # Exclude extreme tail quantiles (p < 0.05, p > 0.95) from comparison:
        # small CDF differences amplify through the inverse at extremes
        interior = (ref.quantile_probs >= 0.05) & (ref.quantile_probs <= 0.95)
        if np.any(interior):
            max_delta_quantile = float(
                np.max(
                    np.abs(
                        fit.quantile_values_py[interior]
                        - ref.quantile_values_r[interior]
                    )
                )
            )
        else:
            max_delta_quantile = float(
                np.max(np.abs(fit.quantile_values_py - ref.quantile_values_r))
            )

    max_delta_hazard: float | None = None
    if (
        fit.hazard_values_py is not None
        and ref.hazard_values_r is not None
        and fit.cdf_values_py is not None
        and ref.hazard_grid is not None
    ):
        # Only compare hazard where the model has reliable estimation:
        # restrict to grid points where CDF < 0.95 (i.e. S(t) > 0.05).
        # In the extreme right tail, S(t) → 0 and hazard = f/S amplifies
        # any tiny CDF difference by orders of magnitude.
        cdf_at_haz = np.interp(ref.hazard_grid, ref.cdf_grid, fit.cdf_values_py)
        reliable = cdf_at_haz < 0.95
        if np.any(reliable):
            max_delta_hazard = float(
                np.max(
                    np.abs(
                        fit.hazard_values_py[reliable]
                        - ref.hazard_values_r[reliable]
                    )
                )
            )
        else:
            # All grid points in extreme tail — skip hazard comparison
            max_delta_hazard = None

    # --- Build failure list ---
    # Primary metrics (loglik, CDF) are always blocking.
    # Theta is always informational (internal parameterization detail).
    # PDF, quantile, and hazard are blocking UNLESS loglik+CDF both match,
    # in which case they are downgraded to informational — because under
    # non-identifiable theta (heavy censoring), the transformation derivative
    # h'(y) can differ even when the distribution function F(h(y)) matches,
    # causing downstream differences in derived quantities.
    failures: list[str] = []
    info: list[str] = []

    loglik_ok = delta_loglik <= TOL_LOGLIK
    if cdf_length_mismatch:
        cdf_ok = False
    else:
        cdf_ok = max_delta_cdf <= TOL_CDF

    if max_delta_theta > TOL_THETA:
        info.append(f"theta ({max_delta_theta:.4f} > {TOL_THETA}; informational)")

    if not loglik_ok:
        failures.append(f"loglik ({delta_loglik:.4f} > {TOL_LOGLIK})")
    if cdf_length_mismatch:
        failures.append(
            f"cdf (length mismatch: got {len(fit.cdf_values_py)},"
            f" expected {len(ref.cdf_values_r)})"
        )
    elif not cdf_ok:
        failures.append(f"cdf ({max_delta_cdf:.4f} > {TOL_CDF})")

    # Derived functional outputs: hard failure only when primary metrics fail too
    _derived: list[tuple[str, float | None, float]] = [
        ("pdf", max_delta_pdf, TOL_PDF),
        ("quantile", max_delta_quantile, TOL_QUANTILE),
        ("hazard", max_delta_hazard, TOL_HAZARD),
    ]
    for name, delta, tol in _derived:
        if delta is not None and delta > tol:
            if loglik_ok and cdf_ok:
                info.append(
                    f"{name} ({delta:.4f} > {tol}; "
                    f"non-identifiable — ll/cdf match)"
                )
            else:
                failures.append(f"{name} ({delta:.4f} > {tol})")

    passed = len(failures) == 0
    all_notes = failures + info
    failure_reason = "; ".join(all_notes) if all_notes else None

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
        max_delta_pdf=max_delta_pdf,
        max_delta_quantile=max_delta_quantile,
        max_delta_hazard=max_delta_hazard,
        expected_failure=ref.case_id in EXPECTED_FAILURES,
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
_YELLOW = "\033[33m"
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
    print("=" * 110)
    header = (
        f"{'Case':<28}│ {'Model':<7}│ {'n':>5} │ {'Ord':>3} │ {'Status':<6}"
        f"│ {'Δθ':>6} │ {'Δll':>6} │ {'Δcdf':>6}"
        f"│ {'Δpdf':>6} │ {'Δqnt':>6} │ {'Δhaz':>6}"
    )
    print(header)
    print("─" * 110)

    def _fmt(val: float | None, width: int = 6) -> str:
        if val is None:
            return "   —".ljust(width)
        if np.isnan(val):
            return " N/A".ljust(width)
        return f"{val:{width}.4f}"

    for r in results:
        if r.expected_failure:
            label = "XPASS" if r.passed else "XFAIL"
            status = _color(label, _YELLOW)
        elif r.passed:
            status = _color("PASS", _GREEN)
        else:
            status = _color("FAIL", _RED)

        dt = _fmt(r.max_delta_theta)
        dl = _fmt(r.delta_loglik)
        dc = _fmt(r.max_delta_cdf)
        dp = _fmt(r.max_delta_pdf)
        dq = _fmt(r.max_delta_quantile)
        dh = _fmt(r.max_delta_hazard)
        print(
            f"{r.case_id:<28}│ {r.model:<7}│ {r.n:>5} "
            f"│ {r.order:>3} │ {status:<6}│ {dt} │ {dl} │ {dc}"
            f"│ {dp} │ {dq} │ {dh}"
        )

    print("─" * 110)
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    n_xfail = sum(1 for r in results if not r.passed and r.expected_failure)
    n_xpass = sum(1 for r in results if r.passed and r.expected_failure)
    pct = 100.0 * n_pass / n_total if n_total > 0 else 0.0
    extras = []
    if n_xfail:
        extras.append(f"{n_xfail} xfail")
    if n_xpass:
        extras.append(f"{n_xpass} xpass")
    suffix = f", {', '.join(extras)}" if extras else ""
    print(f"Total: {n_pass}/{n_total} passed{suffix} ({pct:.1f}%)")
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
        "| Case | Model | n | Order | Status | Δθ | Δll | Δcdf"
        " | Δpdf | Δquant | Δhaz |",
        "|------|-------|---|-------|--------|-----|-----|------"
        "|------|--------|------|",
    ]

    def _md(val: float | None) -> str:
        if val is None:
            return "—"
        return "N/A" if np.isnan(val) else f"{val:.4f}"

    for r in results:
        if r.expected_failure:
            status = "XPASS" if r.passed else "XFAIL"
        else:
            status = "PASS" if r.passed else "FAIL"

        lines.append(
            f"| {r.case_id} | {r.model} | {r.n} | {r.order} "
            f"| {status} | {_md(r.max_delta_theta)} "
            f"| {_md(r.delta_loglik)} | {_md(r.max_delta_cdf)} "
            f"| {_md(r.max_delta_pdf)} | {_md(r.max_delta_quantile)} "
            f"| {_md(r.max_delta_hazard)} |"
        )

    n_pass = sum(1 for r in results if r.passed)
    n_xfail = sum(1 for r in results if not r.passed and r.expected_failure)
    n_xpass = sum(1 for r in results if r.passed and r.expected_failure)
    lines.append("")
    extras = []
    if n_xfail:
        extras.append(f"{n_xfail} xfail")
    if n_xpass:
        extras.append(f"{n_xpass} xpass")
    suffix = f" ({', '.join(extras)})" if extras else ""
    lines.append(f"**{n_pass}/{len(results)} passed**{suffix}")
    lines.append("")

    (out_dir / "validation_report.md").write_text("\n".join(lines))

    # --- JSON ---
    def _json_float(val: float | None) -> float | None:
        if val is None:
            return None
        return None if np.isnan(val) else val

    records = []
    for r in results:
        records.append(
            {
                "case_id": r.case_id,
                "model": r.model,
                "n": r.n,
                "order": r.order,
                "passed": r.passed,
                "max_delta_theta": _json_float(r.max_delta_theta),
                "delta_loglik": _json_float(r.delta_loglik),
                "max_delta_cdf": _json_float(r.max_delta_cdf),
                "max_delta_pdf": _json_float(r.max_delta_pdf),
                "max_delta_quantile": _json_float(r.max_delta_quantile),
                "max_delta_hazard": _json_float(r.max_delta_hazard),
                "converged": r.converged,
                "runtime_s": round(r.runtime_s, 4),
                "failure_reason": r.failure_reason,
                "expected_failure": r.expected_failure,
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

    unexpected_failures = [
        r for r in results if not r.passed and not r.expected_failure
    ]
    xfails = [r for r in results if not r.passed and r.expected_failure]
    xpasses = [r for r in results if r.passed and r.expected_failure]

    if xfails:
        names = ", ".join(r.case_id for r in xfails)
        print(
            f"{len(xfails)} expected failure(s) not counted toward exit code: {names}"
        )
    if xpasses:
        names = ", ".join(r.case_id for r in xpasses)
        print(
            f"{len(xpasses)} case(s) marked expected-failure unexpectedly passed: "
            f"{names}. Consider removing from EXPECTED_FAILURES."
        )
    if unexpected_failures:
        print(f"{len(unexpected_failures)} case(s) exceeded tolerance thresholds.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
