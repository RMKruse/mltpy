"""Tests for pymlt.tram — BoxCox, Coxph, Colr."""
from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import logistic as _logistic, norm

import pymlt
from pymlt.model import MLT
from pymlt.tram import BoxCox, Coxph, Colr, _TramModel
from pymlt.variables import CensoredData, CensoringType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_y(n: int = 80, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.05, 0.95, n)


def simple_survival(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    times = rng.exponential(scale=0.4, size=n).clip(0.01, 0.99)
    censored = rng.random(n) < 0.3
    return times, censored


# ---------------------------------------------------------------------------
# Smoke tests — fit + predict, no errors, correct shapes
# ---------------------------------------------------------------------------

class TestBoxCoxSmoke:
    def test_instantiate(self):
        model = BoxCox(support=(0.0, 1.0))
        assert isinstance(model, BoxCox)
        assert isinstance(model, _TramModel)

    def test_fit_predict(self):
        y = simple_y()
        model = BoxCox(support=(0.0, 1.0))
        model.fit(y)
        cdf = model.predict(y, what="distribution")
        assert cdf.shape == (len(y),)
        assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)

    def test_censoring_is_none(self):
        model = BoxCox(support=(0.0, 1.0))
        assert model.censoring is CensoringType.NONE

    def test_base_distribution_normal(self):
        model = BoxCox(support=(0.0, 1.0))
        assert model.base_distribution == "normal"

    def test_fitted_transformation_shape(self):
        y = simple_y()
        model = BoxCox(support=(0.0, 1.0)).fit(y)
        h = model.fitted_transformation(y)
        assert h.shape == (len(y),)

    def test_fitted_transformation_before_fit_raises(self):
        from pymlt.model import NotFittedError
        model = BoxCox(support=(0.0, 1.0))
        with pytest.raises(NotFittedError):
            model.fitted_transformation(simple_y())


class TestCoxphSmoke:
    def test_instantiate(self):
        model = Coxph(support=(0.0, 1.0))
        assert isinstance(model, Coxph)

    def test_censoring_is_right(self):
        model = Coxph(support=(0.0, 1.0))
        assert model.censoring is CensoringType.RIGHT

    def test_fit_with_censored_data(self):
        times, censored = simple_survival()
        cd = CensoredData.right_censored(times, censored)
        model = Coxph(support=(0.01, 1.0))
        model.fit(cd)
        assert model.is_fitted_

    def test_survival_shape_and_range(self):
        times, censored = simple_survival()
        cd = CensoredData.right_censored(times, censored)
        model = Coxph(support=(0.01, 1.0)).fit(cd)
        grid = np.linspace(0.05, 0.95, 30)
        s = model.survival(grid)
        assert s.shape == (30,)
        assert np.all(s >= 0.0) and np.all(s <= 1.0)

    def test_hazard_shape(self):
        times, censored = simple_survival()
        cd = CensoredData.right_censored(times, censored)
        model = Coxph(support=(0.01, 1.0)).fit(cd)
        grid = np.linspace(0.05, 0.95, 20)
        h = model.hazard(grid)
        assert h.shape == (20,)
        assert np.all(h >= 0.0)


class TestColrSmoke:
    def test_instantiate(self):
        model = Colr(support=(0.0, 1.0))
        assert isinstance(model, Colr)

    def test_base_distribution_logistic(self):
        model = Colr(support=(0.0, 1.0))
        assert model.base_distribution == "logistic"

    def test_censoring_is_none(self):
        model = Colr(support=(0.0, 1.0))
        assert model.censoring is CensoringType.NONE

    def test_fit_predict_distribution(self):
        y = simple_y()
        model = Colr(support=(0.0, 1.0)).fit(y)
        cdf = model.predict(y, what="distribution")
        assert cdf.shape == (len(y),)
        assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)

    def test_fit_predict_density_non_negative(self):
        y = simple_y()
        model = Colr(support=(0.0, 1.0)).fit(y)
        pdf = model.predict(y, what="density")
        assert np.all(pdf >= 0.0)


# ---------------------------------------------------------------------------
# BoxCox: fitted_transformation is monotone (Hypothesis)
# ---------------------------------------------------------------------------

@settings(max_examples=15, deadline=8000)
@given(seed=st.integers(min_value=0, max_value=99))
def test_boxcox_fitted_transformation_monotone(seed: int):
    """h(y) must be non-decreasing — the core monotonicity guarantee."""
    y = np.random.default_rng(seed).uniform(0.05, 0.95, 60)
    model = BoxCox(support=(0.0, 1.0), order=4).fit(y)
    grid = np.linspace(0.05, 0.95, 50)
    h = model.fitted_transformation(grid)
    assert np.all(np.diff(h) >= -1e-6), f"seed={seed}: min diff={np.diff(h).min():.2e}"


# ---------------------------------------------------------------------------
# Coxph: survival properties
# ---------------------------------------------------------------------------

class TestCoxphSurvival:
    def setup_method(self):
        times, censored = simple_survival(seed=7)
        cd = CensoredData.right_censored(times, censored)
        self.model = Coxph(support=(0.01, 1.0), order=3).fit(cd)
        self.grid = np.linspace(0.05, 0.95, 40)

    def test_survival_is_complement_of_cdf(self):
        s = self.model.survival(self.grid)
        cdf = self.model.predict(self.grid, what="distribution")
        np.testing.assert_allclose(s, 1.0 - cdf, atol=1e-10)

    def test_survival_monotone_decreasing(self):
        s = self.model.survival(self.grid)
        assert np.all(np.diff(s) <= 1e-6), f"survival not monotone: {np.diff(s).max():.2e}"

    def test_hazard_matches_predict(self):
        h1 = self.model.hazard(self.grid)
        h2 = self.model.predict(self.grid, what="hazard")
        np.testing.assert_array_equal(h1, h2)


# ---------------------------------------------------------------------------
# Colr uses logistic distribution (different theta from BoxCox)
# ---------------------------------------------------------------------------

def test_colr_uses_logistic_distribution():
    """Colr and BoxCox on the same data must produce different theta_."""
    y = simple_y(seed=42)
    boxcox = BoxCox(support=(0.0, 1.0), order=4).fit(y)
    colr   = Colr(support=(0.0, 1.0), order=4).fit(y)
    # Theta vectors differ because base distributions differ
    assert not np.allclose(boxcox.theta_, colr.theta_, atol=1e-3), (
        "BoxCox and Colr produced identical theta — logistic distribution not applied"
    )


# ---------------------------------------------------------------------------
# Colr: prediction uses logistic distribution
# ---------------------------------------------------------------------------

class TestColrPredictLogistic:
    """Colr.predict() must use the logistic distribution, not normal."""

    def setup_method(self):
        rng = np.random.default_rng(17)
        self.y = np.sort(rng.uniform(0.05, 0.95, 120))
        self.model = Colr(support=(0.0, 1.0), order=5).fit(self.y)
        self.grid = np.linspace(0.1, 0.9, 30)

    def _h(self, y_vals: np.ndarray) -> np.ndarray:
        p = self.model.basis.order + 1
        return self.model.basis.evaluate(y_vals) @ self.model.theta_[:p]

    def test_cdf_matches_logistic(self):
        h = self._h(self.grid)
        np.testing.assert_allclose(
            self.model.predict(self.grid, what="distribution"),
            _logistic.cdf(h),
        )

    def test_cdf_not_norm(self):
        h = self._h(self.grid)
        actual = self.model.predict(self.grid, what="distribution")
        assert not np.allclose(actual, norm.cdf(h), atol=1e-6), (
            "Colr.predict(distribution) returned norm.cdf values"
        )

    def test_density_matches_logistic(self):
        p = self.model.basis.order + 1
        D = self.model.basis.derivative(self.grid, order=1)
        hp = D @ self.model.theta_[:p]
        h = self._h(self.grid)
        expected = _logistic.pdf(h) * np.maximum(hp, 0.0)
        np.testing.assert_allclose(
            self.model.predict(self.grid, what="density"),
            expected,
        )

    def test_quantile_cdf_inverse(self):
        probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q = self.model.predict(probs, what="quantile")
        cdf_back = self.model.predict(q, what="distribution")
        np.testing.assert_allclose(cdf_back, probs, atol=1e-4)

    def test_quantile_h_equals_logistic_ppf(self):
        """h(quantile(p)) must equal logistic.ppf(p), not norm.ppf(p)."""
        probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q = self.model.predict(probs, what="quantile")
        h_at_q = self._h(q)
        np.testing.assert_allclose(h_at_q, _logistic.ppf(probs), atol=1e-4)
        assert not np.allclose(h_at_q, norm.ppf(probs), atol=1e-4), (
            "h(quantile(p)) matches norm.ppf — logistic.ppf not used as inversion target"
        )

    def test_median_near_data_median(self):
        """Fitted median quantile should be close to the data median."""
        q50 = self.model.predict(np.array([0.5]), what="quantile")
        np.testing.assert_allclose(q50[0], np.median(self.y), atol=0.05)

    def test_colr_cdf_differs_from_boxcox(self):
        """Same data, different base distribution → different CDF values."""
        boxcox = BoxCox(support=(0.0, 1.0), order=5).fit(self.y)
        cdf_colr = self.model.predict(self.grid, what="distribution")
        cdf_boxcox = boxcox.predict(self.grid, what="distribution")
        assert not np.allclose(cdf_colr, cdf_boxcox, atol=1e-3), (
            "Colr and BoxCox CDF predictions are identical — logistic distribution not applied"
        )


# ---------------------------------------------------------------------------
# Logistic hazard — base_distribution="logistic" with RIGHT censoring
# ---------------------------------------------------------------------------

class TestLogisticHazard:
    """predict(what='hazard') must use logistic pdf/sf for logistic-family models.

    Colr hard-codes censoring=NONE so hazard is not available on it directly.
    This class uses an MLT with base_distribution="logistic" and censoring=RIGHT,
    which exercises the same code path.
    """

    def setup_method(self):
        rng = np.random.default_rng(55)
        y = rng.uniform(0.05, 0.95, 100)
        censored = rng.random(100) < 0.3
        cd = CensoredData.right_censored(y, censored)
        self.model = MLT(
            order=5, support=(0.0, 1.0),
            base_distribution="logistic",
            censoring=CensoringType.RIGHT,
        ).fit(cd)
        self.grid = np.linspace(0.1, 0.9, 25)

    def _h(self, y_vals: np.ndarray) -> np.ndarray:
        p = self.model.basis.order + 1
        return self.model.basis.evaluate(y_vals) @ self.model.theta_[:p]

    def test_hazard_matches_logistic_ratio(self):
        """predict(hazard) == logistic.pdf(h) / logistic.sf(h)."""
        h = self._h(self.grid)
        expected = _logistic.pdf(h) / np.maximum(_logistic.sf(h), 1e-300)
        np.testing.assert_allclose(
            self.model.predict(self.grid, what="hazard"), expected
        )

    def test_hazard_not_norm_ratio(self):
        """Regression guard: hazard must not equal the normal-based ratio."""
        h = self._h(self.grid)
        wrong = norm.pdf(h) / np.maximum(norm.sf(h), 1e-300)
        actual = self.model.predict(self.grid, what="hazard")
        assert not np.allclose(actual, wrong, atol=1e-6), (
            "logistic hazard returned norm-based values"
        )


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_before_fit(self):
        model = BoxCox(support=(0.0, 1.0))
        s = model.summary()
        assert isinstance(s, str)
        assert "Fitted:       No" in s

    def test_summary_after_fit(self):
        model = BoxCox(support=(0.0, 1.0)).fit(simple_y())
        s = model.summary()
        assert "Log-lik" in s
        assert "Fitted:       Yes" in s
        assert "Converged" in s

    def test_summary_coxph(self):
        times, censored = simple_survival()
        cd = CensoredData.right_censored(times, censored)
        model = Coxph(support=(0.01, 1.0)).fit(cd)
        s = model.summary()
        assert "Coxph" in s
        assert "Log-lik" in s

    def test_summary_colr(self):
        model = Colr(support=(0.0, 1.0)).fit(simple_y())
        s = model.summary()
        assert "Colr" in s


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_shows_boxcox(self):
        assert "BoxCox" in repr(BoxCox(support=(0.0, 1.0)))

    def test_repr_shows_coxph(self):
        assert "Coxph" in repr(Coxph(support=(0.0, 1.0)))

    def test_repr_shows_colr(self):
        assert "Colr" in repr(Colr(support=(0.0, 1.0)))

    def test_repr_not_mlt(self):
        r = repr(BoxCox(support=(0.0, 1.0)))
        assert "MLT" not in r

    def test_repr_after_fit_has_ll(self):
        model = BoxCox(support=(0.0, 1.0)).fit(simple_y())
        assert "ll=" in repr(model)


# ---------------------------------------------------------------------------
# plot()
# ---------------------------------------------------------------------------

class TestPlot:
    def setup_method(self):
        pytest.importorskip("matplotlib")
        self.model = BoxCox(support=(0.0, 1.0)).fit(simple_y())
        self.y = simple_y()

    def test_plot_no_ax_returns_two_axes(self):
        result = self.model.plot(self.y)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_plot_no_ax_both_panels_have_data(self):
        ax_cdf, ax_pdf = self.model.plot(self.y)
        assert len(ax_cdf.lines) > 0, "CDF panel has no lines"
        assert len(ax_pdf.lines) > 0, "density panel has no lines"

    def test_plot_with_axes_tuple_returns_list(self):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)
        result = self.model.plot(self.y, ax=(ax1, ax2))
        assert result == [ax1, ax2]
        plt.close(fig)

    def test_plot_with_axes_tuple_plots_both(self):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.model.plot(self.y, ax=(ax1, ax2))
        assert len(ax1.lines) > 0, "CDF axis has no lines"
        assert len(ax2.lines) > 0, "density axis has no lines"
        plt.close(fig)

    def test_plot_single_axes_raises_type_error(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        with pytest.raises(TypeError):
            self.model.plot(self.y, ax=ax)
        plt.close(fig)

    def test_plot_raises_import_error_when_no_matplotlib(self):
        model = BoxCox(support=(0.0, 1.0)).fit(simple_y())
        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            with pytest.raises(ImportError, match="matplotlib"):
                model.plot(simple_y())


# ---------------------------------------------------------------------------
# Top-level pymlt import
# ---------------------------------------------------------------------------

def test_pymlt_top_level_import():
    assert hasattr(pymlt, "BoxCox")
    assert hasattr(pymlt, "Coxph")
    assert hasattr(pymlt, "Colr")
    model = pymlt.BoxCox(support=(0.0, 1.0))
    assert isinstance(model, BoxCox)
