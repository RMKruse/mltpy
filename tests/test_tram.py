"""Tests for mltpy.tram — BoxCox, Coxph, Colr, Lm."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import logistic as _logistic
from scipy.stats import norm

import mltpy
from mltpy.model import MLT
from mltpy.tram import BoxCox, Colr, Coxph, Lehmann, Lm, _format_wald_table, _TramModel
from mltpy.variables import CensoredData, CensoringType

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
        from mltpy.model import NotFittedError

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
        assert np.all(np.diff(s) <= 1e-6), (
            f"survival not monotone: {np.diff(s).max():.2e}"
        )

    def test_hazard_matches_predict(self):
        h1 = self.model.hazard(self.grid)
        h2 = self.model.predict(self.grid, what="hazard")
        np.testing.assert_array_equal(h1, h2)


# ---------------------------------------------------------------------------
# Lehmann — proportional reverse-time hazards
# ---------------------------------------------------------------------------


class TestLehmannSmoke:
    def test_instantiate(self):
        model = Lehmann(support=(0.0, 1.0))
        assert isinstance(model, Lehmann)

    def test_base_distribution_max_extreme_value(self):
        model = Lehmann(support=(0.0, 1.0))
        assert model.base_distribution == "max_extreme_value"

    def test_censoring_is_right(self):
        model = Lehmann(support=(0.0, 1.0))
        assert model.censoring is CensoringType.RIGHT

    def test_fit_with_censored_data(self):
        times, censored = simple_survival()
        cd = CensoredData.right_censored(times, censored)
        model = Lehmann(support=(0.01, 1.0))
        model.fit(cd)
        assert model.is_fitted_

    def test_survival_shape_and_range(self):
        times, censored = simple_survival()
        cd = CensoredData.right_censored(times, censored)
        model = Lehmann(support=(0.01, 1.0)).fit(cd)
        grid = np.linspace(0.05, 0.95, 30)
        s = model.survival(grid)
        assert s.shape == (30,)
        assert np.all(s >= 0.0) and np.all(s <= 1.0)

    def test_hazard_shape(self):
        times, censored = simple_survival()
        cd = CensoredData.right_censored(times, censored)
        model = Lehmann(support=(0.01, 1.0)).fit(cd)
        grid = np.linspace(0.05, 0.95, 20)
        h = model.hazard(grid)
        assert h.shape == (20,)
        assert np.all(h >= 0.0)


class TestLehmannSurvival:
    def setup_method(self):
        times, censored = simple_survival(seed=7)
        cd = CensoredData.right_censored(times, censored)
        self.model = Lehmann(support=(0.01, 1.0), order=3).fit(cd)
        self.grid = np.linspace(0.05, 0.95, 40)

    def test_survival_is_complement_of_cdf(self):
        s = self.model.survival(self.grid)
        cdf = self.model.predict(self.grid, what="distribution")
        np.testing.assert_allclose(s, 1.0 - cdf, atol=1e-10)

    def test_survival_monotone_decreasing(self):
        s = self.model.survival(self.grid)
        assert np.all(np.diff(s) <= 1e-6), (
            f"survival not monotone: {np.diff(s).max():.2e}"
        )

    def test_hazard_matches_predict(self):
        h1 = self.model.hazard(self.grid)
        h2 = self.model.predict(self.grid, what="hazard")
        np.testing.assert_array_equal(h1, h2)


def test_lehmann_differs_from_coxph():
    """Lehmann and Coxph on the same data must produce different theta_."""
    times, censored = simple_survival(seed=42)
    cd = CensoredData.right_censored(times, censored)
    lehmann = Lehmann(support=(0.01, 1.0), order=4).fit(cd)
    coxph = Coxph(support=(0.01, 1.0), order=4).fit(cd)
    assert not np.allclose(lehmann.theta_, coxph.theta_, atol=1e-3), (
        "Lehmann and Coxph produced identical theta — max_extreme_value not applied"
    )


# ---------------------------------------------------------------------------
# Colr uses logistic distribution (different theta from BoxCox)
# ---------------------------------------------------------------------------


def test_colr_uses_logistic_distribution():
    """Colr and BoxCox on the same data must produce different theta_."""
    y = simple_y(seed=42)
    boxcox = BoxCox(support=(0.0, 1.0), order=4).fit(y)
    colr = Colr(support=(0.0, 1.0), order=4).fit(y)
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
            order=5,
            support=(0.0, 1.0),
            base_distribution="logistic",
            censoring=CensoringType.RIGHT,
        ).fit(cd)
        self.grid = np.linspace(0.1, 0.9, 25)

    def _h(self, y_vals: np.ndarray) -> np.ndarray:
        p = self.model.basis.order + 1
        return self.model.basis.evaluate(y_vals) @ self.model.theta_[:p]

    def _hp(self, y_vals: np.ndarray) -> np.ndarray:
        p = self.model.basis.order + 1
        return self.model.basis.derivative(y_vals, order=1) @ self.model.theta_[:p]

    def test_hazard_matches_logistic_ratio(self):
        """predict(hazard) == logistic.pdf(h) * h' / logistic.sf(h)."""
        h = self._h(self.grid)
        hp = self._hp(self.grid)
        expected = (
            _logistic.pdf(h) * np.maximum(hp, 0.0) / np.maximum(_logistic.sf(h), 1e-300)
        )
        np.testing.assert_allclose(
            self.model.predict(self.grid, what="hazard"), expected
        )

    def test_hazard_not_norm_ratio(self):
        """Regression guard: hazard must not equal the normal-based ratio."""
        h = self._h(self.grid)
        hp = self._hp(self.grid)
        wrong = norm.pdf(h) * np.maximum(hp, 0.0) / np.maximum(norm.sf(h), 1e-300)
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
        assert "AIC:" in s
        assert "BIC:" in s
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
# _format_wald_table — zero / non-finite SE handling
# ---------------------------------------------------------------------------


class TestFormatWaldTableDegenerateSE:
    """Zero or non-finite SEs render as ``NA`` without RuntimeWarnings."""

    def test_zero_se_renders_na_quietly(self):
        names = ["X1", "X2"]
        estimates = np.array([0.5, 1.2])
        ses = np.array([0.0, 0.3])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            table = _format_wald_table(names, estimates, ses)

        rows = table.splitlines()
        assert "NA" in rows[1]
        assert "inf" not in rows[1]
        # The valid row still has numeric z and p columns.
        assert "NA" not in rows[2]

    def test_nonfinite_se_renders_na(self):
        names = ["X1"]
        estimates = np.array([0.5])
        ses = np.array([np.nan])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            table = _format_wald_table(names, estimates, ses)
        assert "NA" in table.splitlines()[1]

    def test_all_positive_se_unchanged(self):
        names = ["X1", "X2"]
        estimates = np.array([0.5, -1.0])
        ses = np.array([0.25, 0.5])
        table = _format_wald_table(names, estimates, ses)
        assert "NA" not in table


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

    def teardown_method(self):
        import matplotlib.pyplot as plt

        plt.close("all")

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

    def test_plot_single_axes_plots_cdf_only(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ret = self.model.plot(self.y, ax=ax)
        assert ret is ax
        assert len(ax.lines) == 1
        assert "CDF" in ax.get_title()
        plt.close(fig)

    def test_plot_one_tuple_raises_type_error(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        with pytest.raises(TypeError, match="2-tuple"):
            self.model.plot(self.y, ax=(ax,))
        plt.close(fig)

    def test_plot_three_tuple_raises_type_error(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3)
        with pytest.raises(TypeError, match="2-tuple"):
            self.model.plot(self.y, ax=tuple(axes))
        plt.close(fig)

    def test_plot_raises_import_error_when_no_matplotlib(self):
        model = BoxCox(support=(0.0, 1.0)).fit(simple_y())
        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            with pytest.raises(ImportError, match="matplotlib"):
                model.plot(simple_y())


# ---------------------------------------------------------------------------
# Top-level mltpy import
# ---------------------------------------------------------------------------


def test_mltpy_top_level_import():
    assert hasattr(mltpy, "BoxCox")
    assert hasattr(mltpy, "Coxph")
    assert hasattr(mltpy, "Colr")
    assert hasattr(mltpy, "Lm")
    model = mltpy.BoxCox(support=(0.0, 1.0))
    assert isinstance(model, BoxCox)


# ---------------------------------------------------------------------------
# Lm — linear model as a CTM
# ---------------------------------------------------------------------------

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"


def _support_from_y(y: np.ndarray, pad: float = 0.1) -> tuple[float, float]:
    return (float(y.min()) - pad, float(y.max()) + pad)


class TestLmSmoke:
    def test_instantiate(self):
        model = Lm(support=(0.0, 1.0))
        assert isinstance(model, Lm)
        assert isinstance(model, _TramModel)

    def test_censoring_is_none(self):
        assert Lm(support=(0.0, 1.0)).censoring is CensoringType.NONE

    def test_base_distribution_normal(self):
        assert Lm(support=(0.0, 1.0)).base_distribution == "normal"

    def test_order_is_one(self):
        model = Lm(support=(0.0, 1.0))
        assert model.basis.order == 1

    def test_order_kwarg_rejected(self):
        with pytest.raises(TypeError):
            Lm(support=(0.0, 1.0), order=2)  # type: ignore[call-arg]

    def test_fit_predict_shape_and_range(self):
        rng = np.random.default_rng(0)
        y = rng.normal(loc=1.0, scale=0.3, size=120)
        model = Lm(support=_support_from_y(y)).fit(y)
        cdf = model.predict(y, what="distribution")
        assert cdf.shape == (len(y),)
        assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)

    def test_theta_is_monotone(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=80)
        model = Lm(support=_support_from_y(y)).fit(y)
        assert model.theta_ is not None
        assert model.theta_[1] > model.theta_[0]

    def test_fitted_transformation_shape(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=60)
        model = Lm(support=_support_from_y(y)).fit(y)
        h = model.fitted_transformation(y)
        assert h.shape == (len(y),)


class TestLmAccessors:
    def setup_method(self):
        rng = np.random.default_rng(3)
        self.x = rng.normal(size=200)
        self.y = 2.0 + 3.0 * self.x + rng.normal(scale=0.5, size=200)
        self.support = _support_from_y(self.y)

    def test_sigma_positive_and_finite(self):
        model = Lm(support=self.support).fit(self.y)
        assert np.isfinite(model.sigma_)
        assert model.sigma_ > 0.0

    def test_intercept_finite(self):
        model = Lm(support=self.support).fit(self.y)
        assert np.isfinite(model.intercept_)

    def test_coef_empty_without_covariates(self):
        model = Lm(support=self.support).fit(self.y)
        assert model.coef_.shape == (0,)

    def test_coef_shape_with_covariate(self):
        model = Lm(support=self.support).fit(self.y, X=self.x.reshape(-1, 1))
        assert model.coef_.shape == (1,)

    def test_accessors_before_fit_raise(self):
        model = Lm(support=(0.0, 1.0))
        with pytest.raises(mltpy.NotFittedError):
            _ = model.sigma_
        with pytest.raises(mltpy.NotFittedError):
            _ = model.intercept_
        with pytest.raises(mltpy.NotFittedError):
            _ = model.coef_
        with pytest.raises(mltpy.NotFittedError):
            model.fitted_transformation(self.y)

    def test_degenerate_theta_raises(self):
        model = Lm(support=self.support).fit(self.y, X=self.x.reshape(-1, 1))
        model.theta_[1] = model.theta_[0]
        with pytest.raises(RuntimeError, match="Degenerate"):
            _ = model.sigma_
        with pytest.raises(RuntimeError, match="Degenerate"):
            _ = model.intercept_
        with pytest.raises(RuntimeError, match="Degenerate"):
            _ = model.coef_

    def test_accessors_support_invariant(self):
        a, b = self.support
        s1 = (a, b)
        s2 = (a - 1.0, b + 1.0)
        m1 = Lm(support=s1).fit(self.y, X=self.x.reshape(-1, 1))
        m2 = Lm(support=s2).fit(self.y, X=self.x.reshape(-1, 1))
        # theta_ is support-dependent
        assert not np.allclose(m1.theta_[:2], m2.theta_[:2], atol=1e-3)
        # Derived quantities are support-invariant
        np.testing.assert_allclose(m1.sigma_, m2.sigma_, atol=1e-4)
        np.testing.assert_allclose(m1.intercept_, m2.intercept_, atol=1e-4)
        np.testing.assert_allclose(m1.coef_, m2.coef_, atol=1e-4)


@pytest.mark.skipif(
    not (REFERENCE_DIR / "lm_uni_theta.txt").exists(),
    reason="R reference data not generated; run Rscript reference/generate_reference.R",
)
class TestLmReference:
    """Validate Lm against R tram::Lm() and base R lm() output."""

    def test_univariate_matches_r(self):
        y = np.loadtxt(REFERENCE_DIR / "lm_uni_y.txt")
        a, b = np.loadtxt(REFERENCE_DIR / "lm_uni_support.txt")
        intercept_r, sigma_r_ols = np.loadtxt(REFERENCE_DIR / "lm_uni_lm_coef.txt")

        model = Lm(support=(float(a), float(b))).fit(y)

        n, p = len(y), 1
        sigma_r_mle = float(sigma_r_ols) * np.sqrt((n - p) / n)

        np.testing.assert_allclose(model.intercept_, intercept_r, atol=1e-4)
        np.testing.assert_allclose(model.sigma_, sigma_r_mle, atol=1e-4)

    def test_covariate_matches_r(self):
        y = np.loadtxt(REFERENCE_DIR / "lm_cov_y.txt")
        x = np.loadtxt(REFERENCE_DIR / "lm_cov_x.txt")
        a, b = np.loadtxt(REFERENCE_DIR / "lm_cov_support.txt")
        intercept_r, slope_r, sigma_r_ols = np.loadtxt(
            REFERENCE_DIR / "lm_cov_lm_coef.txt"
        )

        model = Lm(support=(float(a), float(b))).fit(y, X=x.reshape(-1, 1))

        n, p = len(y), 2
        sigma_r_mle = float(sigma_r_ols) * np.sqrt((n - p) / n)

        np.testing.assert_allclose(model.intercept_, intercept_r, atol=1e-4)
        np.testing.assert_allclose(model.sigma_, sigma_r_mle, atol=1e-4)
        np.testing.assert_allclose(model.coef_[0], slope_r, atol=1e-4)


class TestLmEquivalence:
    """Cross-check Lm accessors against numpy.linalg.lstsq on synthetic data."""

    def test_matches_ols_lstsq(self):
        rng = np.random.default_rng(2026)
        n = 400
        x = rng.normal(size=n)
        y = 2.0 + 3.0 * x + rng.normal(scale=0.5, size=n)
        support = _support_from_y(y)

        model = Lm(support=support).fit(y, X=x.reshape(-1, 1))

        A = np.c_[np.ones(n), x]
        beta_ols, *_ = np.linalg.lstsq(A, y, rcond=None)
        residuals = y - A @ beta_ols
        # MLE sigma (divide by n), not OLS (divide by n-p) — CTM is MLE-based
        sigma_mle = float(np.sqrt(residuals @ residuals / n))

        np.testing.assert_allclose(model.intercept_, beta_ols[0], atol=0.02)
        np.testing.assert_allclose(model.coef_[0], beta_ols[1], atol=0.02)
        np.testing.assert_allclose(model.sigma_, sigma_mle, atol=0.02)


@pytest.mark.skipif(
    not (REFERENCE_DIR / "predict_quantile_coxph_expected.txt").exists(),
    reason="R reference data not generated; run Rscript reference/generate_reference.R",
)
class TestCoxphPredictQuantileReference:
    """Validate conditional quantile prediction against R tram::Coxph."""

    def test_conditional_quantile_matches_r(self):
        y = np.loadtxt(REFERENCE_DIR / "vcov_coxph_y.txt")
        event = np.loadtxt(REFERENCE_DIR / "vcov_coxph_event.txt").astype(bool)
        x = np.loadtxt(REFERENCE_DIR / "vcov_coxph_x.txt")
        a, b = np.loadtxt(REFERENCE_DIR / "vcov_coxph_support.txt")

        X_grid = np.loadtxt(REFERENCE_DIR / "predict_quantile_coxph_X.txt")
        probs = np.loadtxt(REFERENCE_DIR / "predict_quantile_coxph_probs.txt")
        expected = np.loadtxt(
            REFERENCE_DIR / "predict_quantile_coxph_expected.txt"
        ).reshape(len(X_grid), len(probs))

        # CensoredData uses censored-mask (True = censored), complement of event.
        cd = CensoredData.right_censored(y, ~event)
        model = Coxph(support=(float(a), float(b)), order=4).fit(cd, X=x.reshape(-1, 1))

        # For each X value, compute quantile vector at all probs (rows of expected).
        got = np.empty_like(expected)
        for i, xv in enumerate(X_grid):
            X_new = np.full((len(probs), 1), float(xv))
            got[i] = model.predict(probs, X_new=X_new, what="quantile")

        # R qmlt() inverts via a grid+spline approximation (not exact root-finding).
        # We mirror that workflow; small residuals remain due spline backend
        # differences between R's hyman spline and SciPy's cubic implementation.
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=7e-4)


# ---------------------------------------------------------------------------
# Coxph(interacting=...) — non-proportional Cox via tensor-product basis (#66)
# ---------------------------------------------------------------------------


class TestCoxphInteracting:
    """``Coxph(interacting=BernsteinBasis(...))`` wires the tensor-product path."""

    @staticmethod
    def _exact_data(n: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, n)
        y = rng.uniform(0.05, 0.95, n)
        return y, x

    def test_matches_mlt_with_interaction_basis(self):
        """Coxph(interacting=...) reproduces MLT(InteractionBasis(...))."""
        from mltpy import (
            ConditionalTransformationModel,
            InteractionBasis,
            OptimizerConfig,
        )
        from mltpy.basis import BernsteinBasis

        y, x = self._exact_data()
        y_basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)

        ref = ConditionalTransformationModel(
            basis=ib,
            censoring=CensoringType.RIGHT,
            optimizer_config=OptimizerConfig(random_state=0),
            base_distribution="min_extreme_value",
        )
        ref.fit(y, X=x)

        model = Coxph(
            support=(0.0, 1.0),
            order=3,
            optimizer_config=OptimizerConfig(random_state=0),
            interacting=x_basis,
        ).fit(y, X=x)

        assert model.theta_ is not None
        np.testing.assert_allclose(model.theta_, ref.theta_, rtol=1e-6, atol=1e-8)

    def test_survival_and_hazard_monotone_on_grid(self):
        """survival(y) is monotone non-increasing and hazard(y) is non-negative."""
        from mltpy import OptimizerConfig
        from mltpy.basis import BernsteinBasis

        y, x = self._exact_data(seed=1)
        x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        model = Coxph(
            support=(0.0, 1.0),
            order=3,
            optimizer_config=OptimizerConfig(random_state=0),
            interacting=x_basis,
        ).fit(y, X=x)

        y_grid = np.linspace(0.05, 0.95, 30)
        x_grid = np.full_like(y_grid, 0.5)
        s = model.survival(y_grid, X=x_grid)
        h = model.hazard(y_grid, X=x_grid)
        assert s.shape == (30,)
        assert np.all(s >= 0.0) and np.all(s <= 1.0)
        assert np.all(np.diff(s) <= 1e-6), (
            f"survival not monotone: max diff={np.diff(s).max():.2e}"
        )
        assert np.all(h >= 0.0)

    def test_stores_interaction_basis_on_self(self):
        """The model's basis attribute is the constructed InteractionBasis."""
        from mltpy import InteractionBasis
        from mltpy.basis import BernsteinBasis

        x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        model = Coxph(support=(0.0, 1.0), order=3, interacting=x_basis)
        assert isinstance(model.basis, InteractionBasis)
        assert model.basis.x_basis is x_basis
