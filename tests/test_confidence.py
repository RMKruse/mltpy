"""Tests for ``confint`` and ``confband`` on :class:`pymlt.ConditionalTransformationModel`.

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

import numpy as np
import pytest
from scipy.stats import norm

from pymlt import MLT, CensoredData, NotFittedError
from pymlt.tram import BoxCox, Colr, Coxph

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
    """Pymlt's confband on the order-4 MLT normal baseline matches R."""
    ref = _load_confband_baseline()
    ref_band_path = REF_DIR / f"confband_baseline_{what}.txt"
    if not ref_band_path.exists():
        pytest.skip(f"{ref_band_path.name} not yet generated")

    m = len(ref["y_grid"])
    ref_band = np.loadtxt(ref_band_path).reshape(m, 3)

    # Fit pymlt on the same y, order=4, support=(0,1).  The MLE converges to
    # the same theta as R (verified by tests/test_mlt.py), so vcov and the
    # resulting band match by construction.
    model = MLT(order=4, support=(0.0, 1.0)).fit(ref["y"])
    np.testing.assert_allclose(model.theta_, ref["theta"], rtol=1e-4, atol=1e-6)

    band = model.confband(ref["y_grid"], level=0.95, what=what)
    assert band.shape == (m, 3)
    # Tolerance is loose enough to absorb the small residual between pymlt's
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


def _apply_pymlt_sign(ci_R: np.ndarray, p: int, beta_sign: float) -> np.ndarray:
    """Convert an R-convention CI table to pymlt's sign convention.

    pymlt always uses ``h = h_b + x'β``.  tram's ``BoxCox`` uses
    ``negative=TRUE`` (``h = h_b - x'β``), so its β is the negative of
    pymlt's.  For those rows, flip sign and swap (lower, upper).
    """
    if beta_sign == 1.0:
        return ci_R.copy()
    out = ci_R.copy()
    # For β rows: pymlt_lower = -R_upper, pymlt_upper = -R_lower
    out[p:, :] = -ci_R[p:, ::-1]
    return out


def test_confint_boxcox_matches_R():
    ref = _load_confint_reference("boxcox")
    a, b = ref["support"]
    m = BoxCox(support=(float(a), float(b)), order=4).fit(ref["y"], X=ref["x"])
    ci_pymlt = m.confint(level=0.95)
    ci_expected = _apply_pymlt_sign(ref["ci_R"], p=5, beta_sign=-1.0)
    # Require fitted theta to match R (verified elsewhere) so CI matches.
    # Tolerance accounts for the small difference between R's and pymlt's
    # optimiser-returned MLEs (the vcov formula itself is validated tighter
    # in tests/test_vcov.py).
    np.testing.assert_allclose(ci_pymlt, ci_expected, rtol=1e-3, atol=1e-3)


def test_confint_colr_matches_R():
    ref = _load_confint_reference("colr")
    a, b = ref["support"]
    m = Colr(support=(float(a), float(b)), order=4).fit(ref["y"], X=ref["x"])
    ci_pymlt = m.confint(level=0.95)
    np.testing.assert_allclose(ci_pymlt, ref["ci_R"], rtol=1e-3, atol=1e-3)


def test_confint_coxph_matches_R():
    ref = _load_confint_reference("coxph")
    a, b = ref["support"]
    cd = CensoredData.right_censored(ref["y"], censored=ref["event"] == 0)
    m = Coxph(support=(float(a), float(b)), order=4).fit(cd, X=ref["x"])
    ci_pymlt = m.confint(level=0.95)
    np.testing.assert_allclose(ci_pymlt, ref["ci_R"], rtol=1e-3, atol=1e-3)


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
