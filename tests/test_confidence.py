"""Tests for ``confint`` and ``confband`` on :class:`mltpy.ConditionalTransformationModel`.

Covers:

* R parity — ``confband`` on a baseline (no-covariate) MLT normal fit against
  a hand-computed delta-method band in R (``confband_baseline_*.txt``);
  ``confint`` against R's ``±z·sqrt(diag(vcov))`` for the three tram fits
  already used by :mod:`tests.test_vcov`.
* Algebraic self-consistency — ``confint`` equals the same formula applied to
  ``standard_errors``; ``confband(..., what="survivor")`` equals the
  distribution band with endpoints mirrored.
* Edge / error cases — bad ``level``/``what``/``X``, pre-fit guard, narrower
  band at a smaller confidence level.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
from scipy.stats import norm

from mltpy import MLT, CensoredData, ConvergenceWarning, NotFittedError
from mltpy.tram import BoxCox, Colr, Coxph

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


# ---------------------------------------------------------------------------
# R parity: confband on the baseline (no-covariate) MLT normal fit
# ---------------------------------------------------------------------------


def _load_confband_baseline():
    required = [
        REF_DIR / "confband_baseline_theta.txt",
        REF_DIR / "confband_baseline_vcov.txt",
        REF_DIR / "confband_baseline_y_grid.txt",
        REF_DIR / "mlt_normal_y.txt",
    ]
    if not all(p.exists() for p in required):
        pytest.skip(
            "confband_baseline_* or mlt_normal_y.txt not yet generated — "
            "run Rscript reference/generate_reference.R"
        )
    theta = np.loadtxt(required[0])
    p = len(theta)
    vcov = np.loadtxt(required[1]).reshape(p, p)
    y_grid = np.loadtxt(required[2])
    y_fit = np.loadtxt(required[3])
    return {"theta": theta, "vcov": vcov, "y_grid": y_grid, "y": y_fit}


@pytest.mark.parametrize(
    "what",
    ["trafo", "distribution", "survivor", "density", "hazard"],
)
def test_confband_baseline_matches_R(what):
    """Mltpy's confband on the order-4 MLT normal baseline matches R."""
    ref = _load_confband_baseline()
    ref_band_path = REF_DIR / f"confband_baseline_{what}.txt"
    if not ref_band_path.exists():
        pytest.skip(f"{ref_band_path.name} not yet generated")

    m = len(ref["y_grid"])
    ref_band = np.loadtxt(ref_band_path).reshape(m, 3)

    # Fit mltpy on the same y, order=4, support=(0,1).  The MLE converges to
    # the same theta as R (verified by tests/test_mlt.py), so vcov and the
    # resulting band match by construction.
    model = MLT(order=4, support=(0.0, 1.0)).fit(ref["y"])
    np.testing.assert_allclose(model.theta_, ref["theta"], rtol=1e-4, atol=1e-6)

    band = model.confband(ref["y_grid"], level=0.95, what=what)
    assert band.shape == (m, 3)
    # Tolerance is loose enough to absorb the small residual between mltpy's
    # MLE and R's (≈1e-6 per-coefficient difference from the two optimisers
    # converging to the same point from different starting simplices).
    np.testing.assert_allclose(band, ref_band, rtol=5e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# R parity: confint for tram fits (BoxCox, Colr, Coxph)
# ---------------------------------------------------------------------------


def _load_confint_reference(model_name: str):
    """Return dict with y, x, event (Coxph only), support, theta_R, CI_R."""
    required = [
        REF_DIR / f"vcov_{model_name}_y.txt",
        REF_DIR / f"vcov_{model_name}_x.txt",
        REF_DIR / f"vcov_{model_name}_support.txt",
        REF_DIR / f"vcov_{model_name}_theta.txt",
        REF_DIR / f"confint_{model_name}.txt",
    ]
    if not all(p.exists() for p in required):
        pytest.skip(
            f"confint_{model_name}.txt / vcov_{model_name}_* not yet "
            "generated — run Rscript reference/generate_reference.R"
        )
    theta = np.loadtxt(required[3])
    k = len(theta)
    data = {
        "y": np.loadtxt(required[0]),
        "x": np.loadtxt(required[1]).reshape(-1, 1),
        "support": tuple(np.loadtxt(required[2])),
        "theta_R": theta,
        "ci_R": np.loadtxt(required[4]).reshape(k, 2),
    }
    event_path = REF_DIR / f"vcov_{model_name}_event.txt"
    if event_path.exists():
        data["event"] = np.loadtxt(event_path).astype(int)
    return data


def _apply_mltpy_sign(ci_R: np.ndarray, p: int, beta_sign: float) -> np.ndarray:
    """Convert an R-convention CI table to mltpy's sign convention.

    mltpy always uses ``h = h_b + x'β``.  tram's ``BoxCox`` uses
    ``negative=TRUE`` (``h = h_b - x'β``), so its β is the negative of
    mltpy's.  For those rows, flip sign and swap (lower, upper).
    """
    if beta_sign == 1.0:
        return ci_R.copy()
    out = ci_R.copy()
    # For β rows: mltpy_lower = -R_upper, mltpy_upper = -R_lower
    out[p:, :] = -ci_R[p:, ::-1]
    return out


def test_confint_boxcox_matches_R():
    ref = _load_confint_reference("boxcox")
    a, b = ref["support"]
    m = BoxCox(support=(float(a), float(b)), order=4).fit(ref["y"], X=ref["x"])
    ci_mltpy = m.confint(level=0.95)
    ci_expected = _apply_mltpy_sign(ref["ci_R"], p=5, beta_sign=-1.0)
    # Require fitted theta to match R (verified elsewhere) so CI matches.
    # Tolerance accounts for the small difference between R's and mltpy's
    # optimiser-returned MLEs (the vcov formula itself is validated tighter
    # in tests/test_vcov.py).
    np.testing.assert_allclose(ci_mltpy, ci_expected, rtol=1e-3, atol=1e-3)


def test_confint_colr_matches_R():
    ref = _load_confint_reference("colr")
    a, b = ref["support"]
    m = Colr(support=(float(a), float(b)), order=4).fit(ref["y"], X=ref["x"])
    ci_mltpy = m.confint(level=0.95)
    np.testing.assert_allclose(ci_mltpy, ref["ci_R"], rtol=1e-3, atol=1e-3)


def test_confint_coxph_matches_R():
    ref = _load_confint_reference("coxph")
    a, b = ref["support"]
    cd = CensoredData.right_censored(ref["y"], censored=ref["event"] == 0)
    m = Coxph(support=(float(a), float(b)), order=4).fit(cd, X=ref["x"])
    ci_mltpy = m.confint(level=0.95)
    np.testing.assert_allclose(ci_mltpy, ref["ci_R"], rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# R parity: profile-likelihood confint for tram fits (BoxCox, Colr, Coxph)
#
# Issue #88 — extends the Wald-CI parity block above by inverting the χ²_1
# LR test in both R and mltpy and comparing the resulting (k, 2) tables on
# the same three tram fixtures.  R-side reference is emitted by
# ``.write_profile_ci_ref`` in ``reference/generate_reference.R``; the
# mltpy side calls ``confint(level=0.95, type="profile")`` from #87.
#
# Tolerance is looser than the Wald block (rtol=1e-3, atol=1e-6 vs 1e-3)
# because each side runs its own root finder over an iteratively-refit
# constrained likelihood — bracket choice and the inner auglag's tolerance
# combine into ~1e-5 noise on log-likelihood-flat coordinates.  Coxph's
# Bernstein MLE sits on the monotonicity boundary (Bs2=Bs3=Bs4); both
# implementations land on the same constrained-refit fallback there, so
# parity still holds.
# ---------------------------------------------------------------------------


def _load_profile_ci_reference(model_name: str):
    """Return y, x, support, theta_R, profile_CI_R for one tram model.

    Parallels ``_load_confint_reference`` but loads
    ``profile_ci_<model>.txt`` instead of ``confint_<model>.txt``.
    """
    required = [
        REF_DIR / f"vcov_{model_name}_y.txt",
        REF_DIR / f"vcov_{model_name}_x.txt",
        REF_DIR / f"vcov_{model_name}_support.txt",
        REF_DIR / f"vcov_{model_name}_theta.txt",
        REF_DIR / f"profile_ci_{model_name}.txt",
    ]
    if not all(p.exists() for p in required):
        pytest.skip(
            f"profile_ci_{model_name}.txt / vcov_{model_name}_* not yet "
            "generated — run Rscript reference/generate_reference.R"
        )
    theta = np.loadtxt(required[3])
    k = len(theta)
    data = {
        "y": np.loadtxt(required[0]),
        "x": np.loadtxt(required[1]).reshape(-1, 1),
        "support": tuple(np.loadtxt(required[2])),
        "theta_R": theta,
        "ci_R": np.loadtxt(required[4]).reshape(k, 2),
    }
    event_path = REF_DIR / f"vcov_{model_name}_event.txt"
    if event_path.exists():
        data["event"] = np.loadtxt(event_path).astype(int)
    return data


def test_confint_profile_boxcox_matches_R():
    """Profile CI on BoxCox(y ~ x) matches R after the β sign flip.

    mltpy parameterises ``h + x'β``; tram's ``BoxCox(negative=TRUE)`` uses
    ``h - x'β``, so its β is the negative of mltpy's.  ``_apply_mltpy_sign``
    flips the β row (here row p=5, the single covariate) before comparing.
    """
    ref = _load_profile_ci_reference("boxcox")
    a, b = ref["support"]
    m = BoxCox(support=(float(a), float(b)), order=4).fit(ref["y"], X=ref["x"])
    ci_mltpy = m.confint(level=0.95, type="profile")
    ci_expected = _apply_mltpy_sign(ref["ci_R"], p=5, beta_sign=-1.0)
    np.testing.assert_allclose(ci_mltpy, ci_expected, rtol=1e-3, atol=1e-6)


def test_confint_profile_colr_matches_R():
    """Profile CI on Colr(y ~ x) matches R (no sign flip — β is aligned)."""
    ref = _load_profile_ci_reference("colr")
    a, b = ref["support"]
    m = Colr(support=(float(a), float(b)), order=4).fit(ref["y"], X=ref["x"])
    ci_mltpy = m.confint(level=0.95, type="profile")
    np.testing.assert_allclose(ci_mltpy, ref["ci_R"], rtol=1e-3, atol=1e-6)


def test_confint_profile_coxph_matches_R():
    """Profile CI on Coxph(Surv(y, event) ~ x) matches R.

    Right-censored fit, no sign flip.  The Bernstein MLE has Bs2=Bs3=Bs4
    stacked on the monotonicity boundary — both R and mltpy land on the
    same constrained-refit fallback there, so parity still holds at the
    standard tolerance.
    """
    ref = _load_profile_ci_reference("coxph")
    a, b = ref["support"]
    cd = CensoredData.right_censored(ref["y"], censored=ref["event"] == 0)
    m = Coxph(support=(float(a), float(b)), order=4).fit(cd, X=ref["x"])
    ci_mltpy = m.confint(level=0.95, type="profile")
    np.testing.assert_allclose(ci_mltpy, ref["ci_R"], rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Algebraic self-consistency
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_mlt():
    rng = np.random.default_rng(17)
    n = 150
    X = rng.normal(0, 1, (n, 2))
    y = 0.4 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(0, 1, n)
    return (
        MLT(order=5, support=(float(y.min() - 0.1), float(y.max() + 0.1))).fit(y, X=X),
        X,
        y,
    )


@pytest.fixture
def fitted_boxcox():
    rng = np.random.default_rng(23)
    y = rng.lognormal(size=200)
    return BoxCox(support=(float(y.min() - 0.05), float(y.max() + 0.05))).fit(y), y


def test_confint_equals_wald_formula(fitted_mlt):
    model, _, _ = fitted_mlt
    se = model.standard_errors()
    z = norm.ppf(0.975)
    expected = np.column_stack((model.theta_ - z * se, model.theta_ + z * se))
    np.testing.assert_allclose(model.confint(level=0.95), expected, atol=1e-12)


def test_confint_parm_subset(fitted_mlt):
    model, _, _ = fitted_mlt
    full = model.confint(level=0.9)
    idx = [0, 2, model.theta_.size - 1]
    subset = model.confint(level=0.9, parm=idx)
    np.testing.assert_allclose(subset, full[idx, :], atol=1e-12)


def test_confint_narrower_at_lower_level(fitted_mlt):
    model, _, _ = fitted_mlt
    ci_50 = model.confint(level=0.5)
    ci_95 = model.confint(level=0.95)
    widths_50 = ci_50[:, 1] - ci_50[:, 0]
    widths_95 = ci_95[:, 1] - ci_95[:, 0]
    assert np.all(widths_50 < widths_95)


def test_confband_distribution_vs_survivor(fitted_boxcox):
    model, y = fitted_boxcox
    a, b = model.basis.support
    grid = np.linspace(a + 0.05 * (b - a), b - 0.05 * (b - a), 40)
    band_F = model.confband(grid, what="distribution", level=0.95)
    band_S = model.confband(grid, what="survivor", level=0.95)
    # S estimate = 1 - F estimate, endpoints mirror (1-F monotone decreasing).
    np.testing.assert_allclose(band_S[:, 0], 1.0 - band_F[:, 0], atol=1e-12)
    np.testing.assert_allclose(band_S[:, 1], 1.0 - band_F[:, 2], atol=1e-12)
    np.testing.assert_allclose(band_S[:, 2], 1.0 - band_F[:, 1], atol=1e-12)


def test_confband_distribution_in_unit_interval(fitted_boxcox):
    model, y = fitted_boxcox
    a, b = model.basis.support
    grid = np.linspace(a + 1e-3, b - 1e-3, 50)
    band = model.confband(grid, what="distribution", level=0.99)
    assert np.all(band >= 0.0)
    assert np.all(band <= 1.0)
    # Lower ≤ estimate ≤ upper everywhere.
    assert np.all(band[:, 1] <= band[:, 0] + 1e-12)
    assert np.all(band[:, 0] <= band[:, 2] + 1e-12)


def test_confband_narrower_at_lower_level(fitted_boxcox):
    model, y = fitted_boxcox
    a, b = model.basis.support
    grid = np.linspace(a + 0.1, b - 0.1, 15)
    b50 = model.confband(grid, what="distribution", level=0.5)
    b95 = model.confband(grid, what="distribution", level=0.95)
    w50 = b50[:, 2] - b50[:, 1]
    w95 = b95[:, 2] - b95[:, 1]
    assert np.all(w50 < w95)


def test_confband_density_positive(fitted_boxcox):
    model, y = fitted_boxcox
    a, b = model.basis.support
    grid = np.linspace(a + 0.05 * (b - a), b - 0.05 * (b - a), 30)
    band = model.confband(grid, what="density", level=0.95)
    assert np.all(band > 0.0)
    assert np.all(band[:, 1] <= band[:, 0] + 1e-12)
    assert np.all(band[:, 0] <= band[:, 2] + 1e-12)


def test_confband_estimate_matches_predict(fitted_mlt):
    """Estimate column equals model.predict at the same grid / covariate."""
    model, X, _ = fitted_mlt
    a, b = model.basis.support
    grid = np.linspace(a + 0.1, b - 0.1, 20)
    x_row = X[0:1, :]
    for what in ("trafo", "distribution", "survivor", "density", "hazard"):
        band = model.confband(grid, X=x_row, what=what, level=0.95)
        # predict broadcasts a single covariate row by tiling.
        expected = model.predict(
            grid, X_new=np.repeat(x_row, grid.size, axis=0), what=what
        )
        np.testing.assert_allclose(band[:, 0], expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


class TestUnfittedGuards:
    def test_unfitted_confint_raises(self):
        m = MLT(order=3, support=(0.0, 1.0))
        with pytest.raises(NotFittedError):
            m.confint()

    def test_unfitted_confband_raises(self):
        m = MLT(order=3, support=(0.0, 1.0))
        with pytest.raises(NotFittedError):
            m.confband(np.array([0.5]))


class TestInvalidInputs:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.y = rng.uniform(0.1, 0.9, 80)
        self.model = MLT(order=4, support=(0.0, 1.0)).fit(self.y)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_confint_bad_level(self, level):
        with pytest.raises(ValueError, match="level"):
            self.model.confint(level=level)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_confband_bad_level(self, level):
        with pytest.raises(ValueError, match="level"):
            self.model.confband(np.array([0.5]), level=level)

    def test_confband_bad_what(self):
        with pytest.raises(ValueError, match="what"):
            self.model.confband(np.array([0.5]), what="logdistribution")

    def test_confint_bad_parm(self):
        with pytest.raises(ValueError, match="parm"):
            self.model.confint(parm=[0, 999])

    def test_confband_unexpected_X(self):
        with pytest.raises(ValueError, match="without covariates"):
            self.model.confband(np.array([0.5]), X=np.array([1.0]))


class TestXShapeValidation:
    def setup_method(self):
        rng = np.random.default_rng(3)
        n = 100
        self.X = rng.normal(0, 1, (n, 2))
        self.y = 0.3 * self.X[:, 0] + rng.normal(0, 1, n)
        self.model = MLT(
            order=4,
            support=(float(self.y.min() - 0.1), float(self.y.max() + 0.1)),
        ).fit(self.y, X=self.X)

    def test_missing_X_raises(self):
        with pytest.raises(ValueError, match="covariates"):
            self.model.confband(np.array([0.5]), X=None)

    def test_wrong_X_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            self.model.confband(np.array([0.5]), X=np.zeros((1, 3)))

    def test_multi_row_X_raises(self):
        with pytest.raises(ValueError, match="shape"):
            self.model.confband(np.array([0.5, 0.6]), X=np.zeros((2, 2)))

    def test_1d_X_accepted(self):
        grid = np.linspace(self.y.min() + 0.1, self.y.max() - 0.1, 5)
        band = self.model.confband(grid, X=np.array([0.0, 0.0]))
        assert band.shape == (5, 3)


def test_confband_intercept_only_no_X():
    rng = np.random.default_rng(5)
    y = rng.uniform(0.1, 0.9, 120)
    model = MLT(order=3, support=(0.0, 1.0)).fit(y)
    grid = np.linspace(0.1, 0.9, 10)
    # X=None works when the model was fit without covariates.
    band = model.confband(grid, X=None, what="distribution")
    assert band.shape == (10, 3)


# ---------------------------------------------------------------------------
# Profile-likelihood confidence intervals (issue #87)
#
# Inverts the χ²_1 likelihood-ratio test for each requested parameter to get
# a CI that respects the curvature of the log-likelihood (unlike the Wald
# interval, which assumes quadratic behaviour at the MLE).
# ---------------------------------------------------------------------------


def _load_profile_ci_baseline():
    required = [
        REF_DIR / "profile_ci_baseline.txt",
        REF_DIR / "mlt_normal_y.txt",
    ]
    if not all(p.exists() for p in required):
        pytest.skip(
            "profile_ci_baseline.txt or mlt_normal_y.txt not yet generated — "
            "run Rscript reference/generate_reference.R"
        )
    y_fit = np.loadtxt(required[1])
    ci_ref = np.loadtxt(required[0]).reshape(-1, 2)
    return {"y": y_fit, "ci": ci_ref}


def test_confint_type_wald_matches_default():
    """type='wald' must reproduce today's default Wald CI bit-for-bit."""
    ref = _load_profile_ci_baseline()  # reuses mlt_normal_y for the baseline fit
    model = MLT(order=4, support=(0.0, 1.0)).fit(ref["y"])
    ci_default = model.confint(level=0.95)
    ci_wald = model.confint(level=0.95, type="wald")
    np.testing.assert_array_equal(ci_wald, ci_default)


def test_confint_invalid_type_raises():
    """Unknown type values raise ValueError listing the accepted set."""
    ref = _load_profile_ci_baseline()
    model = MLT(order=4, support=(0.0, 1.0)).fit(ref["y"])
    with pytest.raises(ValueError, match=r"\{'wald', 'profile'\}"):
        model.confint(type="garbage")  # type: ignore[arg-type]


def test_confint_profile_baseline_matches_R():
    """Profile-CI on the order-4 no-covariate baseline matches R mlt.

    Reference is produced by inverting the χ²_1 LR test in R via
    ``mlt(..., fixed = c(name = value))`` re-fits — see the
    ``profile_ci_baseline`` block in ``reference/generate_reference.R``.
    """
    ref = _load_profile_ci_baseline()
    model = MLT(order=4, support=(0.0, 1.0)).fit(ref["y"])
    ci = model.confint(level=0.95, type="profile")
    assert ci.shape == ref["ci"].shape
    np.testing.assert_allclose(ci, ref["ci"], rtol=1e-3, atol=1e-6)


def test_confint_profile_bracket_failure_raises():
    """Bracket search that cannot widen far enough must raise actionably.

    Shrinks the initial bracket multiplier to a tiny value and caps the
    doubling budget at 1 so the search has no realistic chance of seeing
    a sign change.  The error message must name the parameter index and
    report the largest ``|f|`` value seen — both are needed for users to
    diagnose whether the issue is a vanishingly small SE or a parameter
    pinned against the monotonicity boundary.
    """
    ref = _load_profile_ci_baseline()
    model = MLT(order=4, support=(0.0, 1.0)).fit(ref["y"])
    # Squeeze the bracket so f never crosses zero within the budget.
    model._PROFILE_BRACKET_INIT = 1e-12
    model._PROFILE_BRACKET_MAX_DOUBLINGS = 1
    with pytest.raises(RuntimeError, match=r"parameter 0.*\|f\|"):
        model.confint(level=0.95, parm=[0], type="profile")


def test_confint_profile_parm_subset():
    """parm= restricts the (k, 2) output to the requested indices only.

    Cuts the 5-parameter baseline down to indices [0, 2] — the resulting
    rows must match the full profile-CI at those same indices, both in
    shape and value.
    """
    ref = _load_profile_ci_baseline()
    model = MLT(order=4, support=(0.0, 1.0)).fit(ref["y"])
    subset = model.confint(level=0.95, parm=[0, 2], type="profile")
    assert subset.shape == (2, 2)
    # The R fixture covers all p=5 parameters; rows 0 and 2 are what we
    # asked for.  Tolerance matches the full-parity test.
    np.testing.assert_allclose(subset, ref["ci"][[0, 2], :], rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("what", ["density", "hazard"])
def test_confband_density_hazard_clip_extreme_h_warns(what):
    """Saturated |h| > _H_CLIP on density/hazard must warn and stay finite.

    Forces the condition by hand-setting theta_ to values that drive
    h = B @ theta_b well past ±30 across the basis support, then checks
    that the band is fully finite and a UserWarning mentioning the clip
    fires from the model module (not leaking inf from dist.logsf(h)).
    """
    rng = np.random.default_rng(11)
    y = rng.uniform(0.1, 0.9, 100)
    model = MLT(order=4, support=(0.0, 1.0)).fit(y)
    # Override theta with values spanning ±50 so h crosses ±_H_CLIP on the grid.
    model.theta_ = np.linspace(-50.0, 50.0, model.theta_.size)
    # Invert the hessian at the original theta is still valid; keep hessian_
    # from the fit so vcov() is non-singular.
    grid = np.linspace(0.01, 0.99, 25)
    with pytest.warns(UserWarning, match="exceeds"):
        band = model.confband(grid, what=what, level=0.95)
    assert np.all(np.isfinite(band))


# ---------------------------------------------------------------------------
# Profile-CI robustness (issue #89): boundary semantics, bracket diagnostics,
# inner-fit failure handling.
#
# The MVP from #87 raises RuntimeError on any of three failure modes.  Issue
# #89 hardens this so that ``parm=None`` calls survive the failure of any
# single parameter (warn + NaN/±inf endpoint), while explicit ``parm=[j]``
# calls still raise so the caller can debug the one parameter they asked
# for.  Tests below force each failure mode through the public interface.
# ---------------------------------------------------------------------------


@pytest.fixture
def small_mlt_for_profile():
    """A small BoxCox fit used as the host for monkey-patched profile-loglik
    experiments.  BoxCox on lognormal data with N=120 converges interior
    (Wald + profile both succeed without any boundary or KKT issues), so
    the un-patched inner refits stay finite — letting the tests below
    isolate the patched-index failure path from background noise.
    Order-4 yields a 5-row theta_ vector before any covariate β.
    """
    rng = np.random.default_rng(8917)
    y = rng.lognormal(size=120)
    return BoxCox(
        order=4,
        support=(float(y.min() - 0.05), float(y.max() + 0.05)),
    ).fit(y)


def test_profile_ci_parm_none_warns_and_returns_inf_when_one_param_unbracketable(
    small_mlt_for_profile, monkeypatch
):
    """parm=None with one un-bracketable parameter → warn + ±inf row;
    finite endpoints in the other rows; call does not raise.

    Manufactures the failure by patching ``_profile_loglik_at`` to return
    the unrestricted MLE log-likelihood whenever ``j == 0`` (perfectly flat
    profile → f(v) = -χ²_crit < 0 for every v → bracket-search exhausts).
    The other indices use the genuine pinned refit and bracket normally.
    """
    model = small_mlt_for_profile
    ll_hat = float(model.result_.log_likelihood)
    real_loglik_at = model._profile_loglik_at

    def patched(j, v):
        if j == 0:
            return ll_hat  # perfectly flat → never crosses the LR threshold
        return real_loglik_at(j, v)

    monkeypatch.setattr(model, "_profile_loglik_at", patched)

    with pytest.warns(ConvergenceWarning, match=r"parameter 0\b"):
        ci = model.confint(level=0.95, type="profile")

    assert ci.shape == (model.theta_.size, 2)
    # The flat-profile row should be unbounded on both sides; explicit ±inf
    # (not NaN) per #89 because the failure mode is "no bracket", not
    # "inner fit non-convergent".
    assert ci[0, 0] == -np.inf
    assert ci[0, 1] == np.inf
    # Remaining rows finite — proves the call did not abort on the failure.
    assert np.all(np.isfinite(ci[1:, :]))


def test_profile_ci_parm_explicit_unbracketable_raises_with_diagnostic(
    small_mlt_for_profile, monkeypatch
):
    """parm=[j] singleton + un-bracketable parameter → RuntimeError whose
    message names j, the largest |f| observed, and the widest bracket
    multiplier tried.  Matches #89 acceptance criterion 3: explicit
    parm requests get the hard failure so the caller can debug.
    """
    model = small_mlt_for_profile
    ll_hat = float(model.result_.log_likelihood)
    real_loglik_at = model._profile_loglik_at

    def patched(j, v):
        if j == 0:
            return ll_hat
        return real_loglik_at(j, v)

    monkeypatch.setattr(model, "_profile_loglik_at", patched)

    with pytest.raises(RuntimeError) as excinfo:
        model.confint(level=0.95, type="profile", parm=[0])
    msg = str(excinfo.value)
    assert "parameter 0" in msg
    # Largest |f| reported (already in the MVP message).
    assert "largest |f|" in msg
    # New per-#89: widest bracket multiplier surfaced for caller to widen.
    assert "widest bracket multiplier" in msg


@pytest.mark.parametrize("failing_index", [0, 2])
def test_profile_ci_convergence_warning_names_parameter_index(
    small_mlt_for_profile, monkeypatch, failing_index
):
    """Every ConvergenceWarning emitted on the parm=None path must include
    the literal parameter index in its message — #89 acceptance criterion 5.

    Captures all warnings and checks that for the chosen failing index
    both the lower- and upper-side warnings name it.  Parametrized so the
    assertion does not pass vacuously when ``failing_index == 0`` happens
    to match an unrelated zero literal elsewhere in the text.
    """
    model = small_mlt_for_profile
    ll_hat = float(model.result_.log_likelihood)
    real_loglik_at = model._profile_loglik_at

    def patched(j, v):
        if j == failing_index:
            return ll_hat
        return real_loglik_at(j, v)

    monkeypatch.setattr(model, "_profile_loglik_at", patched)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.confint(level=0.95, type="profile")

    relevant = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    # Two warnings (lower + upper) for the failing parameter; none for
    # the others (otherwise the test would not isolate the assertion).
    assert len(relevant) == 2, (
        f"expected 2 warnings for parameter {failing_index}, got "
        f"{[str(w.message) for w in relevant]}"
    )
    needle = f"parameter {failing_index}"
    for w in relevant:
        assert needle in str(w.message), (
            f"warning did not name parameter {failing_index}: {w.message}"
        )


def test_profile_ci_boundary_failure_returns_signed_inf_under_parm_none(
    small_mlt_for_profile, monkeypatch
):
    """A boundary failure inside the pinned refit (the equality theta[j]=v
    can't co-exist with active monotonicity rows) maps to ±inf with a
    side-correct sign under parm=None — #89 acceptance criterion 1.

    Manufactures the failure by patching ``_profile_loglik_at`` to raise
    the private ``_ProfileInnerFailure`` for ``j == 0`` regardless of v
    (so both lower and upper sides hit the boundary path).  Asserts both
    endpoints saturate to ±inf with a warning naming the parameter and
    the failure kind, while other rows are finite.
    """
    from mltpy.model import _ProfileInnerFailure

    model = small_mlt_for_profile
    real_loglik_at = model._profile_loglik_at

    def patched(j, v):
        if j == 0:
            raise _ProfileInnerFailure(
                j=j,
                kind="boundary",
                diagnostic=(
                    f"inner fit could not honour theta_[{j}]={v}: post-fit "
                    "monotonicity active set is degenerate"
                ),
            )
        return real_loglik_at(j, v)

    monkeypatch.setattr(model, "_profile_loglik_at", patched)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        ci = model.confint(level=0.95, type="profile")

    # Side-correct ±inf saturation.
    assert ci[0, 0] == -np.inf
    assert ci[0, 1] == np.inf
    # Other rows intact.
    assert np.all(np.isfinite(ci[1:, :]))
    # Warnings name the parameter and the kind.
    relevant = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert len(relevant) >= 2  # one per side (more is OK if the search probed twice)
    for w in relevant:
        text = str(w.message)
        assert "parameter 0" in text
        assert "boundary" in text


def test_profile_ci_boundary_failure_under_parm_explicit_raises(
    small_mlt_for_profile, monkeypatch
):
    """The same boundary failure under parm=[j] re-raises RuntimeError
    naming the parameter and the kind — #89 acceptance criterion 1, strict
    side: explicit caller gets the hard failure to inspect.
    """
    from mltpy.model import _ProfileInnerFailure

    model = small_mlt_for_profile
    real_loglik_at = model._profile_loglik_at

    def patched(j, v):
        if j == 0:
            raise _ProfileInnerFailure(
                j=j,
                kind="boundary",
                diagnostic="inner fit hit monotonicity active set",
            )
        return real_loglik_at(j, v)

    monkeypatch.setattr(model, "_profile_loglik_at", patched)

    with pytest.raises(RuntimeError) as excinfo:
        model.confint(level=0.95, type="profile", parm=[0])
    msg = str(excinfo.value)
    assert "parameter 0" in msg
    assert "boundary" in msg


def _make_inner_optimize_patcher(real_optimize, target_j, *, mode, real_theta):
    """Wrap ``optimize`` so the inner refit for ``fixed_params={target_j: v}``
    returns a fabricated :class:`OptimizationResult` simulating either a
    pure convergence failure (``mode="convergence"`` — kkt_residual above
    threshold, theta honours the pin) or a boundary failure (``mode="boundary"``
    — theta drifts from the pin).  All other ``optimize`` calls pass through
    to the real implementation.
    """
    from mltpy.optimizer import OptimizationResult

    def patched(*args, **kwargs):
        cfg = kwargs.get("config")
        fp = getattr(cfg, "fixed_params", None)
        if fp is not None and target_j in fp:
            v = float(fp[target_j])
            theta = real_theta.copy()
            if mode == "boundary":
                # Equality theta[j] = v was not honoured — drift past tol.
                theta[target_j] = v + 1.0
                kkt = 1e-2  # below the convergence threshold (the drift is
                # what triggers boundary classification, not the KKT)
            else:  # convergence
                theta[target_j] = v  # equality honoured
                kkt = 1.0  # >> _PROFILE_INNER_KKT_THRESHOLD (0.1)
            return OptimizationResult(
                theta=theta,
                log_likelihood=-1234.5,  # arbitrary; never compared
                converged=False,
                n_iter=10,
                n_restarts=0,
                solver_message="synthetic-failure",
                n_outer_iter=5,
                kkt_residual=kkt,
            )
        return real_optimize(*args, **kwargs)

    return patched


def test_profile_ci_convergence_failure_returns_nan_under_parm_none(
    small_mlt_for_profile, monkeypatch
):
    """parm=None + inner-fit non-convergence (KKT residual >> 1e-3 with no
    theta-pin drift) → warn + NaN endpoint — #89 acceptance criterion 4.

    Patches ``mltpy.model.optimize`` to fabricate a non-convergent inner
    result whenever ``fixed_params={0: v}`` is requested.  Asserts the
    public surface translates this into NaN row-0 endpoints with the
    other rows still finite and a ConvergenceWarning naming the index
    and the kind.
    """
    import mltpy.model as pm

    model = small_mlt_for_profile
    patcher = _make_inner_optimize_patcher(
        pm.optimize, target_j=0, mode="convergence", real_theta=model.theta_
    )
    monkeypatch.setattr(pm, "optimize", patcher)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        ci = model.confint(level=0.95, type="profile")

    assert np.isnan(ci[0, 0])
    assert np.isnan(ci[0, 1])
    assert np.all(np.isfinite(ci[1:, :]))

    relevant = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert len(relevant) >= 2
    for w in relevant:
        text = str(w.message)
        assert "parameter 0" in text
        assert "convergence" in text


def test_profile_ci_inner_loglik_detects_boundary_vs_convergence(
    small_mlt_for_profile, monkeypatch
):
    """Detection-logic unit test: ``_profile_loglik_at`` raises
    ``_ProfileInnerFailure`` with the right ``kind`` based on whether the
    pinned refit returned ``theta[j]`` drifted from ``v`` (boundary) or
    just non-convergent KKT (convergence).
    """
    import mltpy.model as pm
    from mltpy.model import _ProfileInnerFailure

    model = small_mlt_for_profile

    # Boundary signal: theta drift.
    patcher_b = _make_inner_optimize_patcher(
        pm.optimize, target_j=0, mode="boundary", real_theta=model.theta_
    )
    monkeypatch.setattr(pm, "optimize", patcher_b)
    with pytest.raises(_ProfileInnerFailure) as exc_b:
        model._profile_loglik_at(0, 0.0)
    assert exc_b.value.kind == "boundary"
    assert exc_b.value.j == 0

    # Convergence signal: no drift, just high KKT.
    patcher_c = _make_inner_optimize_patcher(
        pm.optimize, target_j=0, mode="convergence", real_theta=model.theta_
    )
    monkeypatch.setattr(pm, "optimize", patcher_c)
    with pytest.raises(_ProfileInnerFailure) as exc_c:
        model._profile_loglik_at(0, 0.0)
    assert exc_c.value.kind == "convergence"
    assert exc_c.value.j == 0


def test_profile_ci_convergence_failure_under_parm_explicit_raises(
    small_mlt_for_profile, monkeypatch
):
    """parm=[j] singleton + inner-fit non-convergence → RuntimeError naming
    the parameter and the kind — #89 acceptance criterion 4, strict side.
    """
    import mltpy.model as pm

    model = small_mlt_for_profile
    patcher = _make_inner_optimize_patcher(
        pm.optimize, target_j=0, mode="convergence", real_theta=model.theta_
    )
    monkeypatch.setattr(pm, "optimize", patcher)

    with pytest.raises(RuntimeError) as excinfo:
        model.confint(level=0.95, type="profile", parm=[0])
    msg = str(excinfo.value)
    assert "parameter 0" in msg
    assert "convergence" in msg
