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
) -> ReferenceCase:
    """Build a minimal ReferenceCase for comparison tests."""
    if theta is None:
        theta = np.linspace(0, 1, 5)
    if cdf is None:
        cdf = np.linspace(0.1, 0.9, 10)
    cdf_grid = np.linspace(0.1, 0.9, len(cdf))
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
) -> FittedResult:
    """Build a FittedResult near the reference with controlled deltas."""
    pdf_py = None
    if ref.pdf_values_r is not None:
        pdf_py = ref.pdf_values_r + pdf_offset
    quantile_py = None
    if ref.quantile_values_r is not None:
        quantile_py = ref.quantile_values_r + quantile_offset
    hazard_py = None
    if ref.hazard_values_r is not None:
        hazard_py = ref.hazard_values_r + hazard_offset
    return FittedResult(
        theta_py=ref.theta_r + theta_offset,
        loglik_py=ref.loglik_r + loglik_offset,
        cdf_values_py=ref.cdf_values_r + cdf_offset,
        converged=converged,
        runtime_s=0.1,
        pdf_values_py=pdf_py,
        quantile_values_py=quantile_py,
        hazard_values_py=hazard_py,
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


def test_compare_results_not_converged() -> None:
    """Non-converged fit must produce passed=False."""
    ref = _make_ref()
    fit = _make_fit(ref, converged=False)
    result = compare_results(ref, fit)
    assert result.passed is False
    assert "converge" in (result.failure_reason or "").lower()


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


# ---------------------------------------------------------------------------
# Integration test — runs the actual script on a known case
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (REF_DIR / "case_01_mlt_200_4" / "theta.npy").is_file(),
    reason="Reference .npy data not available (run convert_references.py first)",
)
def test_integration_run_validation_single_case() -> None:
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
