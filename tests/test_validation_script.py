"""Tests for validation/run_validation.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Adjust path so we can import from validation/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

from run_validation import (  # noqa: E402
    _NEW_METRIC_SPEC,
    _NEW_WHATS,
    FittedResult,
    ReferenceCase,
    ValidationResult,
    compare_results,
    load_reference,
    print_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REF_DIR = Path(__file__).resolve().parent.parent / "validation" / "references"


def _make_ref(
    case_id: str = "test_case",
    theta: np.ndarray | None = None,
    loglik: float = -100.0,
    cdf: np.ndarray | None = None,
    pdf: np.ndarray | None = None,
    quantile_probs: np.ndarray | None = None,
    quantile_values: np.ndarray | None = None,
    hazard: np.ndarray | None = None,
    **new_metrics: np.ndarray | None,
) -> ReferenceCase:
    """Build a minimal ReferenceCase for comparison tests.

    Additional keyword arguments matching the 10 new metric names (e.g.
    ``trafo``, ``logodds``) are accepted and routed to the corresponding
    ``<what>_values_r`` field on the dataclass.
    """
    if theta is None:
        theta = np.linspace(0, 1, 5)
    if cdf is None:
        cdf = np.linspace(0.1, 0.9, 10)
    cdf_grid = np.linspace(0.1, 0.9, len(cdf))
    new_kwargs: dict[str, np.ndarray | None] = {
        f"{w}_values_r": new_metrics.get(w) for w in _NEW_WHATS
    }
    return ReferenceCase(
        case_id=case_id,
        model="mlt",
        censoring="none",
        n=100,
        order=4,
        support=(0.0, 1.0),
        y=np.linspace(0.05, 0.95, 100),
        theta_r=theta,
        loglik_r=loglik,
        cdf_grid=cdf_grid,
        cdf_values_r=cdf,
        pdf_grid=cdf_grid if pdf is not None else None,
        pdf_values_r=pdf,
        quantile_probs=quantile_probs,
        quantile_values_r=quantile_values,
        hazard_grid=cdf_grid if hazard is not None else None,
        hazard_values_r=hazard,
        **new_kwargs,
    )


def _make_fit(
    ref: ReferenceCase,
    theta_offset: float = 0.0,
    loglik_offset: float = 0.0,
    cdf_offset: float = 0.0,
    pdf_offset: float = 0.0,
    quantile_offset: float = 0.0,
    hazard_offset: float = 0.0,
    converged: bool = True,
    **new_metric_offsets: float,
) -> FittedResult:
    """Build a FittedResult near the reference with controlled deltas.

    For each of the 10 new metric names, a ``<what>_offset`` keyword adds a
    uniform shift to the corresponding mltpy prediction (only when the
    reference provides a value; otherwise the mltpy field stays ``None``).
    """
    pdf_py = None
    if ref.pdf_values_r is not None:
        pdf_py = ref.pdf_values_r + pdf_offset
    quantile_py = None
    if ref.quantile_values_r is not None:
        quantile_py = ref.quantile_values_r + quantile_offset
    hazard_py = None
    if ref.hazard_values_r is not None:
        hazard_py = ref.hazard_values_r + hazard_offset

    new_py: dict[str, np.ndarray | None] = {}
    for what in _NEW_WHATS:
        ref_vals = getattr(ref, f"{what}_values_r", None)
        offset = float(new_metric_offsets.get(f"{what}_offset", 0.0))
        new_py[f"{what}_values_py"] = None if ref_vals is None else ref_vals + offset

    return FittedResult(
        theta_py=ref.theta_r + theta_offset,
        loglik_py=ref.loglik_r + loglik_offset,
        cdf_values_py=ref.cdf_values_r + cdf_offset,
        converged=converged,
        runtime_s=0.1,
        pdf_values_py=pdf_py,
        quantile_values_py=quantile_py,
        hazard_values_py=hazard_py,
        **new_py,
    )


# ---------------------------------------------------------------------------
# test_load_reference — synthetic data
# ---------------------------------------------------------------------------


def test_load_reference_synthetic(tmp_path: Path) -> None:
    """Write synthetic .npy + metadata.json, verify load_reference reads them."""
    case_dir = tmp_path / "case_99_test_100_4"
    case_dir.mkdir()

    y = np.linspace(0.05, 0.95, 100)
    theta = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    loglik = np.float64(-123.456)
    cdf_grid = np.linspace(0.1, 0.9, 10)
    cdf_values = np.linspace(0.05, 0.95, 10)

    np.save(case_dir / "y.npy", y)
    np.save(case_dir / "theta.npy", theta)
    np.save(case_dir / "loglik.npy", loglik)
    np.save(case_dir / "cdf_grid.npy", cdf_grid)
    np.save(case_dir / "cdf_values.npy", cdf_values)

    meta = {
        "model": "mlt",
        "censoring": "none",
        "n": 100,
        "order": 4,
        "support": [0.0, 1.0],
        "seed": 42,
    }
    (case_dir / "metadata.json").write_text(json.dumps(meta))

    ref = load_reference(case_dir)
    assert ref.case_id == "case_99_test_100_4"
    assert ref.model == "mlt"
    assert ref.censoring == "none"
    assert ref.n == 100
    assert ref.order == 4
    assert ref.support == (0.0, 1.0)
    np.testing.assert_array_equal(ref.y, y)
    np.testing.assert_array_equal(ref.theta_r, theta)
    assert ref.loglik_r == pytest.approx(-123.456)
    assert ref.status is None
    assert ref.X is None


def test_load_reference_with_status(tmp_path: Path) -> None:
    """Verify status.npy is loaded when present."""
    case_dir = tmp_path / "case_99_right_100_4"
    case_dir.mkdir()

    y = np.random.default_rng(0).uniform(0.1, 0.9, 50)
    status = np.array([True] * 35 + [False] * 15)
    np.save(case_dir / "y.npy", y)
    np.save(case_dir / "status.npy", status)
    np.save(case_dir / "theta.npy", np.linspace(0, 1, 5))
    np.save(case_dir / "loglik.npy", np.float64(-50.0))
    np.save(case_dir / "cdf_grid.npy", np.linspace(0.1, 0.9, 10))
    np.save(case_dir / "cdf_values.npy", np.linspace(0.1, 0.9, 10))

    meta = {
        "model": "mlt",
        "censoring": "right",
        "n": 50,
        "order": 4,
        "support": [0.0, 1.0],
        "seed": 42,
    }
    (case_dir / "metadata.json").write_text(json.dumps(meta))

    ref = load_reference(case_dir)
    assert ref.status is not None
    assert ref.status.shape == (50,)


# ---------------------------------------------------------------------------
# test_compare_results — boundary tests
# ---------------------------------------------------------------------------


def test_compare_results_pass() -> None:
    """Δθ=0.01, Δll=0.01, Δcdf=0.001 — all within tolerance."""
    ref = _make_ref()
    fit = _make_fit(ref, theta_offset=0.01, loglik_offset=0.01, cdf_offset=0.001)
    result = compare_results(ref, fit)
    assert result.passed is True
    assert result.failure_reason is None


def test_compare_results_theta_informational_only() -> None:
    """Δθ=0.06 > TOL_THETA — passes because theta is informational only."""
    ref = _make_ref()
    fit = _make_fit(ref, theta_offset=0.06)
    result = compare_results(ref, fit)
    assert result.passed is True
    assert result.failure_reason is not None
    assert "informational" in result.failure_reason


def test_compare_results_theta_with_cdf_fail() -> None:
    """Δθ=0.06 AND Δcdf=0.03 — fails on cdf, theta is informational."""
    ref = _make_ref()
    fit = _make_fit(ref, theta_offset=0.06, cdf_offset=0.03)
    result = compare_results(ref, fit)
    assert result.passed is False
    assert result.failure_reason is not None
    assert "cdf" in result.failure_reason


def test_compare_results_fail_loglik() -> None:
    """Δll=0.2 > TOL_LOGLIK=0.1 — must fail with loglik in reason."""
    ref = _make_ref()
    fit = _make_fit(ref, loglik_offset=0.2)
    result = compare_results(ref, fit)
    assert result.passed is False
    assert result.failure_reason is not None
    assert "loglik" in result.failure_reason


def test_compare_results_fail_cdf() -> None:
    """Δcdf=0.03 > TOL_CDF=0.02 — must fail with cdf in reason."""
    ref = _make_ref()
    fit = _make_fit(ref, cdf_offset=0.03)
    result = compare_results(ref, fit)
    assert result.passed is False
    assert result.failure_reason is not None
    assert "cdf" in result.failure_reason


def test_compare_results_not_converged_but_correct_passes() -> None:
    """Non-converged fit with correct theta must still pass on deltas.

    Auglag (the new default solver) can report ``converged=False`` on
    degenerate active sets — its strict KKT residual asymptotes just above
    ``outer_tol=1e-7`` even when θ matches R to many decimals.  The script
    now treats the convergence flag as informational and lets the Δθ / Δll /
    Δcdf tolerances decide pass/fail; only a true crash (empty θ) short-
    circuits.  ``test_compare_results_crash_short_circuits`` covers that path.
    """
    ref = _make_ref()
    fit = _make_fit(ref, converged=False)
    result = compare_results(ref, fit)
    assert result.passed is True


def test_compare_results_crash_short_circuits() -> None:
    """Empty θ (the ``_FAILED_FIT`` sentinel) must short-circuit to fail."""
    ref = _make_ref()
    fit = FittedResult(
        theta_py=np.array([]),
        loglik_py=float("nan"),
        cdf_values_py=np.array([]),
        converged=False,
        runtime_s=0.0,
    )
    result = compare_results(ref, fit)
    assert result.passed is False
    assert "crash" in (result.failure_reason or "").lower()


def test_compare_results_multiple_failures() -> None:
    """Multiple tolerances exceeded — all appear in failure_reason."""
    ref = _make_ref()
    fit = _make_fit(ref, loglik_offset=0.2, cdf_offset=0.03)
    result = compare_results(ref, fit)
    assert result.passed is False
    assert result.failure_reason is not None
    assert "loglik" in result.failure_reason
    assert "cdf" in result.failure_reason


# ---------------------------------------------------------------------------
# test_compare_results — functional output tests
# ---------------------------------------------------------------------------


def _make_ref_with_functional() -> ReferenceCase:
    """Build a ReferenceCase with all functional outputs populated."""
    return _make_ref(
        pdf=np.linspace(0.5, 1.5, 10),
        quantile_probs=np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
        quantile_values=np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
        hazard=np.linspace(0.1, 2.0, 10),
    )


def test_compare_results_pass_with_functional() -> None:
    """All functional metrics within tolerance — PASS."""
    ref = _make_ref_with_functional()
    fit = _make_fit(ref, pdf_offset=0.01, quantile_offset=0.01, hazard_offset=0.01)
    result = compare_results(ref, fit)
    assert result.passed is True
    assert result.max_delta_pdf is not None
    assert result.max_delta_quantile is not None
    assert result.max_delta_hazard is not None


def test_compare_results_fail_pdf_with_cdf() -> None:
    """Δpdf=0.06 AND Δcdf=0.03 — fails on both (derived failures are hard when CDF fails)."""
    ref = _make_ref_with_functional()
    fit = _make_fit(ref, pdf_offset=0.06, cdf_offset=0.03)
    result = compare_results(ref, fit)
    assert result.passed is False
    assert result.failure_reason is not None
    assert "pdf" in result.failure_reason
    assert "cdf" in result.failure_reason


def test_compare_results_pdf_informational_when_cdf_ok() -> None:
    """Δpdf=0.06 but ll/cdf match — passes (non-identifiable)."""
    ref = _make_ref_with_functional()
    fit = _make_fit(ref, pdf_offset=0.06)
    result = compare_results(ref, fit)
    assert result.passed is True
    assert result.failure_reason is not None
    assert "non-identifiable" in result.failure_reason
    assert "pdf" in result.failure_reason


def test_compare_results_fail_quantile_with_loglik() -> None:
    """Δquantile=0.06 AND Δll=0.2 — fails (derived failures are hard when loglik fails)."""
    ref = _make_ref_with_functional()
    fit = _make_fit(ref, quantile_offset=0.06, loglik_offset=0.2)
    result = compare_results(ref, fit)
    assert result.passed is False
    assert result.failure_reason is not None
    assert "quantile" in result.failure_reason


def test_compare_results_fail_hazard_with_cdf() -> None:
    """Δhazard=0.11 AND Δcdf=0.03 — fails on both."""
    ref = _make_ref_with_functional()
    fit = _make_fit(ref, hazard_offset=0.11, cdf_offset=0.03)
    result = compare_results(ref, fit)
    assert result.passed is False
    assert result.failure_reason is not None
    assert "hazard" in result.failure_reason


def test_compare_results_missing_optional_metrics() -> None:
    """No PDF/quantile/hazard reference — metrics are None, not failure."""
    ref = _make_ref()  # no functional outputs
    fit = _make_fit(ref)
    result = compare_results(ref, fit)
    assert result.passed is True
    assert result.max_delta_pdf is None
    assert result.max_delta_quantile is None
    assert result.max_delta_hazard is None


# ---------------------------------------------------------------------------
# test_print_report
# ---------------------------------------------------------------------------


def test_print_report_contains_pass_fail(capsys: pytest.CaptureFixture[str]) -> None:
    """print_report must include PASS and FAIL strings."""
    results = [
        ValidationResult(
            case_id="case_pass",
            model="mlt",
            n=200,
            order=4,
            passed=True,
            max_delta_theta=0.01,
            delta_loglik=0.001,
            max_delta_cdf=0.001,
            converged=True,
            runtime_s=0.1,
            max_delta_pdf=0.005,
            max_delta_quantile=0.002,
        ),
        ValidationResult(
            case_id="case_fail",
            model="mlt",
            n=200,
            order=6,
            passed=False,
            max_delta_theta=0.06,
            delta_loglik=0.2,
            max_delta_cdf=0.03,
            converged=True,
            runtime_s=0.2,
            failure_reason="cdf (0.03 > 0.02)",
            max_delta_pdf=0.06,
            max_delta_hazard=0.05,
        ),
    ]
    print_report(results)
    captured = capsys.readouterr().out
    assert "PASS" in captured
    assert "FAIL" in captured
    assert "1/2 passed" in captured
    # No new-metric deltas were set, so the extended block must not render.
    assert "Extended predict-type" not in captured


def test_print_report_renders_extended_table_when_new_metrics_present(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Extended predict-type table must render once any new-metric delta is set."""
    new_kwargs = {f"max_delta_{w}": 0.001 for w in _NEW_WHATS}
    results = [
        ValidationResult(
            case_id="case_ext",
            model="mlt",
            n=200,
            order=4,
            passed=True,
            max_delta_theta=0.01,
            delta_loglik=0.001,
            max_delta_cdf=0.001,
            converged=True,
            runtime_s=0.1,
            **new_kwargs,
        ),
    ]
    print_report(results)
    out = capsys.readouterr().out
    assert "Extended predict-type" in out
    assert "legend:" in out
    # Every short label from _NEW_TERMINAL_COLS must appear in the extended header.
    from run_validation import _NEW_TERMINAL_COLS

    for label, _, _ in _NEW_TERMINAL_COLS:
        assert label in out


# ---------------------------------------------------------------------------
# Integration test — runs the actual script on a known case
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# New-metric coverage — parametrized over all 10 new prediction types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("what,_field,tol,_mask", _NEW_METRIC_SPEC)
def test_new_metric_populated_when_reference_given(
    what: str, _field: str, tol: float, _mask: object
) -> None:
    """When a new-metric reference is provided, the delta appears on ValidationResult."""
    ref_vals = np.linspace(0.1, 0.9, 10)
    ref = _make_ref(**{what: ref_vals})
    fit = _make_fit(ref)
    result = compare_results(ref, fit)
    delta = getattr(result, f"max_delta_{what}")
    assert delta is not None
    assert delta == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("what,_field,_tol,_mask", _NEW_METRIC_SPEC)
def test_new_metric_none_when_reference_absent(
    what: str, _field: str, _tol: float, _mask: object
) -> None:
    """No reference for this metric → max_delta_* is None, case still PASS."""
    ref = _make_ref()  # no new metrics populated
    fit = _make_fit(ref)
    result = compare_results(ref, fit)
    assert getattr(result, f"max_delta_{what}") is None
    assert result.passed is True


@pytest.mark.parametrize("what,_field,tol,_mask", _NEW_METRIC_SPEC)
def test_new_metric_exceedance_is_informational_when_cdf_loglik_ok(
    what: str, _field: str, tol: float, _mask: object
) -> None:
    """Δ<metric> > tol but Δcdf/Δll tiny → non-identifiability demotion."""
    ref_vals = np.linspace(0.1, 0.9, 10)
    ref = _make_ref(**{what: ref_vals})
    offset = tol * 5.0  # Well above tolerance, outside any tail mask
    fit = _make_fit(ref, **{f"{what}_offset": offset})
    result = compare_results(ref, fit)
    assert result.passed is True, (
        f"{what}: exceedance should be demoted when cdf+loglik agree; "
        f"got failure_reason={result.failure_reason!r}"
    )
    assert result.failure_reason is not None
    assert "non-identifiable" in result.failure_reason


@pytest.fixture
def reference_npy_cache() -> None:
    """Ensure the .npy reference cache for the integration case exists.

    The .npy files are a gitignored local cache regenerated from the committed
    .csv ground truth (see validation/convert_references.py). Build them on
    demand so the integration test runs on a fresh checkout / CI instead of
    skipping. Only skip if the CSV ground truth itself is missing — that means
    a broken checkout, which cannot be reconstructed here.
    """
    case_dir = REF_DIR / "case_01_mlt_200_4"
    if not case_dir.is_dir() or not list(case_dir.glob("*.csv")):
        pytest.skip(f"Reference CSV ground truth missing under {case_dir}")
    if not (case_dir / "theta.npy").is_file():
        from convert_references import convert_case  # noqa: E402

        convert_case(case_dir)


def test_integration_run_validation_single_case(
    reference_npy_cache: None,
) -> None:
    """Run the validation script on case_01_mlt_200_4, expect exit code 0."""
    result = subprocess.run(
        [
            sys.executable,
            "validation/run_validation.py",
            "--case",
            "case_01_mlt_200_4",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Validation failed for case_01_mlt_200_4:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
