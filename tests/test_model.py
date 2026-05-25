"""Tests for mltpy.model — ConditionalTransformationModel and MLT."""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import logistic as _logistic
from scipy.stats import norm

from mltpy.basis import BernsteinBasis
from mltpy.model import (
    MLT,
    AnovaResult,
    ConditionalTransformationModel,
    ConvergenceWarning,
    NotFittedError,
    anova,
)
from mltpy.optimizer import OptimizerConfig
from mltpy.variables import CensoredData, CensoringType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def simple_y(n: int = 80, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.05, 0.95, n)


def make_ctm(order: int = 3) -> ConditionalTransformationModel:
    basis = BernsteinBasis(order=order, support=(0.0, 1.0))
    return ConditionalTransformationModel(basis)


# ---------------------------------------------------------------------------
# NotFittedError
# ---------------------------------------------------------------------------


class TestNotFittedError:
    def test_predict_before_fit(self):
        model = make_ctm()
        with pytest.raises(NotFittedError):
            model.predict(np.array([0.5]))

    def test_score_before_fit(self):
        model = make_ctm()
        with pytest.raises(NotFittedError):
            model.score(simple_y())

    def test_simulate_before_fit(self):
        model = make_ctm()
        with pytest.raises(NotFittedError):
            model.simulate(10)

    def test_aic_before_fit(self):
        model = make_ctm()
        with pytest.raises(NotFittedError):
            model.aic()

    def test_bic_before_fit(self):
        model = make_ctm()
        with pytest.raises(NotFittedError):
            model.bic()


# ---------------------------------------------------------------------------
# fit() / method chaining
# ---------------------------------------------------------------------------


class TestFit:
    def test_method_chaining(self):
        model = make_ctm()
        result = model.fit(simple_y())
        assert result is model

    def test_is_fitted_after_fit(self):
        model = make_ctm()
        model.fit(simple_y())
        assert model.is_fitted_

    def test_theta_shape(self):
        order = 4
        basis = BernsteinBasis(order=order, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis)
        model.fit(simple_y())
        assert model.theta_.shape == (order + 1,)

    def test_theta_shape_with_x(self):
        rng = np.random.default_rng(7)
        n, q = 80, 2
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis)
        y = rng.uniform(0.05, 0.95, n)
        X = rng.standard_normal((n, q))
        model.fit(y, X=X)
        assert model.theta_.shape == (4 + q,)

    def test_duck_typed_columns_length_mismatch_raises(self):
        class _BadColumnsArray:
            def __init__(self, data: np.ndarray):
                self._data = data
                self.shape = data.shape
                self.columns = ("x1", "x2")

            def __array__(
                self, dtype: np.dtype[np.float64] | None = None
            ) -> np.ndarray:
                if dtype is None:
                    return self._data
                return self._data.astype(dtype, copy=False)

        model = make_ctm()
        y = simple_y(n=20)
        X = _BadColumnsArray(np.linspace(-1.0, 1.0, 20).reshape(-1, 1))
        with pytest.raises(ValueError, match="columns metadata"):
            model.fit(y, X=X)

    def test_result_has_log_likelihood(self):
        model = make_ctm()
        model.fit(simple_y())
        assert np.isfinite(model.result_.log_likelihood)

    def test_convergence_warning(self):
        """Very tight iteration limit → ConvergenceWarning.

        Pinned to ``solver="slsqp"`` so ``max_iter=1`` actually starves the
        scipy inner solve.  Auglag's outer budget is controlled separately by
        :attr:`~mltpy._auglag.AugLagOptions.max_outer_iter` and ignores
        ``max_iter``; the equivalent starvation path for auglag is covered in
        :mod:`tests.test_auglag`.
        """
        cfg = OptimizerConfig(solver="slsqp", max_iter=1, max_restarts=0)
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis, optimizer_config=cfg)
        with pytest.warns(ConvergenceWarning):
            model.fit(simple_y())


# ---------------------------------------------------------------------------
# _validate_input
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_out_of_support_raises(self):
        model = make_ctm()
        with pytest.raises(ValueError, match="support"):
            model.fit(np.array([0.5, 1.5]))

    def test_below_support_raises(self):
        model = make_ctm()
        with pytest.raises(ValueError, match="support"):
            model.fit(np.array([-0.1, 0.5]))

    def test_censored_out_of_support_raises(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis)
        cd = CensoredData.right_censored(np.array([0.5, 1.5]), np.array([False, False]))
        with pytest.raises(ValueError, match="support"):
            model.fit(cd)

    def test_x_shape_mismatch_raises(self):
        model = make_ctm()
        y = simple_y(n=10)
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="rows"):
            model.fit(y, X=X)

    def test_pandas_series_accepted(self):
        """Duck-typing: pd.Series is coerced via np.asarray."""
        pd = pytest.importorskip("pandas")
        model = make_ctm()
        s = pd.Series(simple_y())
        model.fit(s)  # must not raise
        assert model.is_fitted_

    def test_censored_upper_out_of_support_raises(self):
        """Interval-censored upper bound above support triggers fin_hi branch."""
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis)
        cd = CensoredData.interval_censored(
            np.array([0.2, 0.4]),
            np.array([0.6, 1.5]),
        )
        with pytest.raises(ValueError, match="upper"):
            model.fit(cd)

    def test_empty_y_raises(self):
        model = make_ctm()
        with pytest.raises(ValueError, match="at least one observation"):
            model.fit(np.array([], dtype=float))

    def test_x_1d_reshaped(self):
        """A 1-D X array is promoted to a column vector; fit must succeed."""
        model = make_ctm()
        y = simple_y(n=20)
        X = np.linspace(-1.0, 1.0, 20)  # shape (20,) — 1-D
        model.fit(y, X=X)
        assert model.is_fitted_
        p = model.basis.order + 1
        assert model.theta_.shape == (p + 1,)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def setup_method(self):
        self.model = make_ctm(order=4)
        self.model.fit(simple_y())
        self.y_grid = np.linspace(0.05, 0.95, 30)

    def test_distribution_shape(self):
        cdf = self.model.predict(self.y_grid, what="distribution")
        assert cdf.shape == (30,)

    def test_distribution_range(self):
        cdf = self.model.predict(self.y_grid, what="distribution")
        assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)

    def test_distribution_monotone(self):
        cdf = self.model.predict(self.y_grid, what="distribution")
        assert np.all(np.diff(cdf) >= -1e-6), (
            f"CDF not monotone: {np.diff(cdf).min():.2e}"
        )

    def test_density_non_negative(self):
        pdf = self.model.predict(self.y_grid, what="density")
        assert np.all(pdf >= 0.0)

    def test_quantile_shape(self):
        probs = np.array([0.1, 0.5, 0.9])
        q = self.model.predict(probs, what="quantile")
        assert q.shape == (3,)

    def test_quantile_in_support(self):
        probs = np.linspace(0.05, 0.95, 20)
        q = self.model.predict(probs, what="quantile")
        assert np.all(q >= 0.0) and np.all(q <= 1.0)

    def test_quantile_cdf_inverse(self):
        """CDF(quantile(p)) ≈ p."""
        probs = np.array([0.2, 0.5, 0.8])
        q = self.model.predict(probs, what="quantile")
        cdf_back = self.model.predict(q, what="distribution")
        np.testing.assert_allclose(cdf_back, probs, atol=1e-4)

    def test_quantile_empty_probs_returns_empty(self):
        out = self.model.predict(np.array([]), what="quantile")
        assert out.shape == (0,)

    def test_quantile_warns_when_bracket_clip_bites(self):
        probs = np.array([1e-15, 1.0 - 1e-15])
        with pytest.warns(UserWarning, match="saturated"):
            self.model.predict(probs, what="quantile")

    def test_quantile_no_warning_for_well_behaved_probs(self):
        probs = np.linspace(0.1, 0.9, 9)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.model.predict(probs, what="quantile")

    def test_invalid_what_raises(self):
        with pytest.raises(ValueError, match="is invalid"):
            self.model.predict(self.y_grid, what="banana")

    @pytest.mark.parametrize("what", ["density", "logdensity", "hazard", "loghazard"])
    def test_non_monotone_theta_raises_on_hp_dependent_what(self, what):
        """Manually corrupting theta_ to violate monotonicity must raise on
        any predict path that uses h'(y), not silently floor to log(tiny)."""
        from mltpy.likelihood import InfeasibleParameterError

        bad = self.model.theta_.copy()
        bad[: self.model.basis.order + 1] = bad[0]
        self.model.theta_ = bad
        with pytest.raises(InfeasibleParameterError, match="h'.y."):
            self.model.predict(self.y_grid, what=what)

    def test_non_monotone_theta_does_not_raise_on_h_only_what(self):
        """Distribution-only paths do not depend on h'(y); they should remain
        callable even when monotonicity is marginal."""
        bad = self.model.theta_.copy()
        bad[: self.model.basis.order + 1] = bad[0]
        self.model.theta_ = bad
        # Should not raise — hp is unused by the distribution path.
        self.model.predict(self.y_grid, what="distribution")

    def test_hazard_any_censoring(self):
        """Hazard is a pure functional of h; no censoring restriction."""
        h = self.model.predict(self.y_grid, what="hazard")
        assert h.shape == (30,)
        assert np.all(h >= 0.0)

    def test_hazard_right_censored(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis, censoring=CensoringType.RIGHT)
        rng = np.random.default_rng(5)
        y = rng.uniform(0.05, 0.95, 60)
        censored = rng.random(60) < 0.3
        cd = CensoredData.right_censored(y, censored)
        model.fit(cd)
        h = model.predict(np.linspace(0.1, 0.9, 10), what="hazard")
        assert np.all(h >= 0.0)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


class TestScore:
    def test_score_is_finite(self):
        model = make_ctm()
        y = simple_y()
        model.fit(y)
        assert np.isfinite(model.score(y))

    def test_score_geq_init(self):
        """Score after fit >= score at initial theta (linspace)."""
        from mltpy.likelihood import log_likelihood
        from mltpy.optimizer import _initial_theta

        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis)
        y = simple_y()
        model.fit(y)
        theta_init = _initial_theta(basis.order + 1, None)
        ll_init = log_likelihood(theta_init, basis, y)
        assert model.score(y) >= ll_init - 1e-6


# ---------------------------------------------------------------------------
# simulate()
# ---------------------------------------------------------------------------


class TestSimulate:
    def test_shape(self):
        model = make_ctm()
        model.fit(simple_y())
        samples = model.simulate(50, random_state=42)
        assert samples.shape == (50,)

    # Fixture's fitted θ_b spans ≈ [-2.15, 2.12], narrower than ppf(1-1e-10) ≈
    # 6.36, so simulate's u-clip cannot avoid bracket saturation in
    # _predict_quantile.  The warning is a real production signal but
    # irrelevant to what this test asserts (samples within basis support).
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_values_in_support(self):
        model = make_ctm()
        model.fit(simple_y())
        samples = model.simulate(200, random_state=0)
        assert np.all(samples >= 0.0) and np.all(samples <= 1.0)

    def test_reproducible(self):
        model = make_ctm()
        model.fit(simple_y())
        s1 = model.simulate(20, random_state=99)
        s2 = model.simulate(20, random_state=99)
        np.testing.assert_array_equal(s1, s2)

    def test_generator_accepted(self):
        model = make_ctm()
        model.fit(simple_y())
        rng = np.random.default_rng(1)
        samples = model.simulate(10, random_state=rng)
        assert samples.shape == (10,)


# ---------------------------------------------------------------------------
# Conditional quantile prediction with covariates
# ---------------------------------------------------------------------------


def _fit_mlt_with_covariate(
    seed: int = 0, n: int = 200, beta_true: float = 1.2
) -> tuple[MLT, np.ndarray, np.ndarray]:
    """Fit an MLT with a binary covariate and a strong shift effect.

    Data-generating process: baseline normal errors, shifted by `beta_true * x`.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=n).astype(float)
    z = rng.normal(size=n)
    y = (z - beta_true * x - 3.0) / 2.0
    support = (float(y.min()) - 0.2, float(y.max()) + 0.2)
    model = MLT(order=5, support=support).fit(y, X=x.reshape(-1, 1))
    return model, x, y


class TestPredictQuantileConditional:
    def test_requires_X_when_fit_with_covariates(self):
        model, _, _ = _fit_mlt_with_covariate()
        probs = np.array([0.25, 0.5, 0.75])
        with pytest.raises(ValueError, match="X_new must be provided"):
            model.predict(probs, what="quantile")

    def test_row_count_mismatch_raises(self):
        model, _, _ = _fit_mlt_with_covariate()
        probs = np.array([0.25, 0.5, 0.75])
        X_bad = np.zeros((2, 1))  # 2 rows vs 3 probs
        with pytest.raises(ValueError, match="rows"):
            model.predict(probs, X_new=X_bad, what="quantile")

    def test_column_count_mismatch_raises(self):
        model, _, _ = _fit_mlt_with_covariate()
        probs = np.array([0.25, 0.5, 0.75])
        X_bad = np.zeros((3, 2))  # 2 cols vs 1 beta
        with pytest.raises(ValueError, match="columns"):
            model.predict(probs, X_new=X_bad, what="quantile")

    def test_X_actually_shifts_quantiles(self):
        """Different X values must produce different quantiles.

        Under the pre-fix bug the returned quantiles were identical across X.
        """
        model, _, _ = _fit_mlt_with_covariate(beta_true=1.5)
        probs = np.full(2, 0.5)
        X0 = np.array([[0.0], [0.0]])
        X1 = np.array([[1.0], [1.0]])
        q0 = model.predict(probs, X_new=X0, what="quantile")
        q1 = model.predict(probs, X_new=X1, what="quantile")
        # With a positive beta shift, the covariate=1 group has a *smaller* y
        # at the median (baseline h(q) = F⁻¹(p) − β).
        assert np.all(q1 != q0)
        assert np.all(q1 < q0)

    def test_conditional_cdf_round_trip(self):
        """predict(quantile) then predict(distribution) at same X returns p."""
        model, _, _ = _fit_mlt_with_covariate()
        probs = np.array([0.2, 0.5, 0.8])
        X = np.array([[0.0], [1.0], [0.0]])
        q = model.predict(probs, X_new=X, what="quantile")
        p_back = model.predict(q, X_new=X, what="distribution")
        np.testing.assert_allclose(p_back, probs, atol=1e-4)

    def test_no_covariates_unchanged(self):
        """The baseline (no-X, no-beta) path is unaffected by the fix."""
        model = make_ctm()
        model.fit(simple_y())
        probs = np.array([0.1, 0.5, 0.9])
        q = model.predict(probs, what="quantile")
        cdf_back = model.predict(q, what="distribution")
        np.testing.assert_allclose(cdf_back, probs, atol=1e-4)


class TestSimulateConditional:
    def test_requires_X_row_count_matching_n(self):
        model, _, _ = _fit_mlt_with_covariate()
        X = np.zeros((5, 1))
        with pytest.raises(ValueError, match="rows"):
            model.simulate(10, X=X, random_state=0)

    def test_requires_X_when_fit_with_covariates(self):
        model, _, _ = _fit_mlt_with_covariate()
        with pytest.raises(ValueError, match="X_new must be provided"):
            model.simulate(10, random_state=0)

    # Fixture's fitted θ_b spans ≈ [-4.93, 3.18], narrower than ppf(1-1e-10) ≈
    # 6.36, so simulate's u-clip cannot avoid bracket saturation in
    # _predict_quantile.  The warning is a real production signal but
    # irrelevant to what this test asserts (covariate-group mean separation).
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_simulate_recovers_shift(self):
        """Simulated samples should separate by covariate group.

        With a positive β the X=1 group has smaller y values than X=0 (since
        h_baseline(q) = F⁻¹(p) − β, a larger shift pulls q downward).
        """
        model, _, _ = _fit_mlt_with_covariate(beta_true=1.5, n=300, seed=11)
        n = 2000
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=n).astype(float).reshape(-1, 1)
        samples = model.simulate(n, X=x, random_state=rng)
        mean0 = samples[x[:, 0] == 0].mean()
        mean1 = samples[x[:, 0] == 1].mean()
        assert mean1 < mean0 - 0.1


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_before_fit(self):
        model = make_ctm()
        r = repr(model)
        assert "fitted=False" in r

    def test_after_fit(self):
        model = make_ctm()
        model.fit(simple_y())
        r = repr(model)
        assert "fitted=True" in r
        assert "ll=" in r

    def test_mlt_repr_before(self):
        model = MLT(order=4, support=(0.0, 1.0))
        r = repr(model)
        assert "fitted=False" in r
        assert "MLT" in r

    def test_mlt_repr_after(self):
        model = MLT(order=4, support=(0.0, 1.0))
        model.fit(simple_y())
        r = repr(model)
        assert "fitted=True" in r
        assert "ll=" in r


# ---------------------------------------------------------------------------
# MLT convenience class
# ---------------------------------------------------------------------------


class TestMLT:
    def test_defaults(self):
        model = MLT()
        assert model.basis.order == 6
        assert model.basis.support == (0.0, 1.0)

    def test_custom(self):
        model = MLT(order=4, support=(0.0, 100.0))
        assert model.basis.order == 4
        assert model.basis.support == (0.0, 100.0)

    def test_fit_predict(self):
        rng = np.random.default_rng(3)
        y = rng.uniform(1.0, 99.0, 80)
        model = MLT(order=4, support=(0.0, 100.0))
        model.fit(y)
        cdf = model.predict(np.linspace(10.0, 90.0, 20), what="distribution")
        assert cdf.shape == (20,)
        assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)


# ---------------------------------------------------------------------------
# End-to-end with covariates
# ---------------------------------------------------------------------------


class TestWithCovariates:
    def test_fit_predict_with_x(self):
        rng = np.random.default_rng(11)
        n, q = 80, 2
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis)
        y = rng.uniform(0.05, 0.95, n)
        X = rng.standard_normal((n, q))
        model.fit(y, X=X)
        assert model.theta_.shape == (4 + q,)
        X_new = rng.standard_normal((20, q))
        cdf = model.predict(rng.uniform(0.1, 0.9, 20), X_new=X_new, what="distribution")
        assert cdf.shape == (20,)
        assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)


# ---------------------------------------------------------------------------
# End-to-end with censored data
# ---------------------------------------------------------------------------


class TestWithCensoredData:
    def test_right_censored(self):
        rng = np.random.default_rng(22)
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis, censoring=CensoringType.RIGHT)
        y = rng.uniform(0.05, 0.95, 60)
        censored = rng.random(60) < 0.3
        cd = CensoredData.right_censored(y, censored)
        model.fit(cd)
        assert model.is_fitted_
        assert np.isfinite(model.result_.log_likelihood)

    def test_interval_censored(self):
        rng = np.random.default_rng(33)
        centers = rng.uniform(0.1, 0.9, 40)
        cd = CensoredData.interval_censored(centers - 0.05, centers + 0.05)
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis, censoring=CensoringType.INTERVAL)
        model.fit(cd)
        assert np.isfinite(model.result_.log_likelihood)


# ---------------------------------------------------------------------------
# Property-based: CDF monotonicity
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=5000)
@given(
    order=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=999),
)
def test_cdf_is_monotone_hypothesis(order: int, seed: int):
    """Fitted CDF must be non-decreasing on a dense grid."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.05, 0.95, 60)
    model = MLT(order=order, support=(0.0, 1.0))
    model.fit(y)
    grid = np.linspace(0.05, 0.95, 50)
    cdf = model.predict(grid, what="distribution")
    diffs = np.diff(cdf)
    assert np.all(diffs >= -1e-6), (
        f"order={order}, seed={seed}: min diff={diffs.min():.2e}"
    )


def test_fit_regression_no_runtime_warning_on_flat_hp_boundary():
    """Regression: this seed/order previously emitted divide-by-zero warnings."""
    rng = np.random.default_rng(24)
    y = rng.uniform(0.05, 0.95, 60)
    model = MLT(order=2, support=(0.0, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(y)
    assert model.is_fitted_


# ---------------------------------------------------------------------------
# R reference integration test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# base_distribution validation — invalid values raise at construction
# ---------------------------------------------------------------------------


class TestBaseDistributionValidation:
    def test_ctm_invalid_raises(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        with pytest.raises(ValueError, match="base_distribution"):
            ConditionalTransformationModel(basis, base_distribution="student-t")

    def test_mlt_invalid_raises(self):
        with pytest.raises(ValueError, match="base_distribution"):
            MLT(order=4, support=(0.0, 1.0), base_distribution="student-t")

    @pytest.mark.parametrize("bad", ["Normal", "LOGISTIC", "gauss", "", "t"])
    def test_case_sensitive_and_aliases_rejected(self, bad):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        with pytest.raises(ValueError, match="base_distribution"):
            ConditionalTransformationModel(basis, base_distribution=bad)

    def test_normal_accepted(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        m = ConditionalTransformationModel(basis, base_distribution="normal")
        assert m.base_distribution == "normal"

    def test_logistic_accepted(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        m = ConditionalTransformationModel(basis, base_distribution="logistic")
        assert m.base_distribution == "logistic"

    def test_min_extreme_value_accepted(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        m = ConditionalTransformationModel(basis, base_distribution="min_extreme_value")
        assert m.base_distribution == "min_extreme_value"

    def test_max_extreme_value_accepted(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        m = ConditionalTransformationModel(basis, base_distribution="max_extreme_value")
        assert m.base_distribution == "max_extreme_value"

    def test_exponential_accepted(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        m = ConditionalTransformationModel(basis, base_distribution="exponential")
        assert m.base_distribution == "exponential"

    def test_error_raised_before_fit(self):
        """ValueError must be raised at __init__, not lazily at fit()."""
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        with pytest.raises(ValueError, match="base_distribution"):
            # The exception must come from __init__, not from fit()
            ConditionalTransformationModel(basis, base_distribution="bad")


# ---------------------------------------------------------------------------
# base_distribution="logistic" — predictions must use logistic, not normal
# ---------------------------------------------------------------------------


class TestPredictLogistic:
    """Prediction output must reflect base_distribution="logistic"."""

    def setup_method(self):
        rng = np.random.default_rng(7)
        self.y = np.sort(rng.uniform(0.05, 0.95, 120))
        self.model = MLT(order=5, support=(0.0, 1.0), base_distribution="logistic").fit(
            self.y
        )
        self.grid = np.linspace(0.1, 0.9, 30)

    def _h(self, y_vals: np.ndarray) -> np.ndarray:
        """Compute transformation h = B @ theta_b for given y values."""
        p = self.model.basis.order + 1
        B = self.model.basis.evaluate(y_vals)
        return B @ self.model.theta_[:p]

    def test_cdf_matches_logistic_cdf(self):
        """predict(distribution) must equal logistic.cdf(h), not norm.cdf(h)."""
        h = self._h(self.grid)
        expected = _logistic.cdf(h)
        actual = self.model.predict(self.grid, what="distribution")
        np.testing.assert_allclose(actual, expected)

    def test_cdf_differs_from_normal_cdf(self):
        """Logistic CDF must not be equal to normal CDF for the same h."""
        h = self._h(self.grid)
        wrong = norm.cdf(h)
        actual = self.model.predict(self.grid, what="distribution")
        assert not np.allclose(actual, wrong, atol=1e-6), (
            "logistic model predict(distribution) returned norm.cdf values"
        )

    def test_density_matches_logistic_pdf(self):
        """predict(density) must use logistic.pdf(h), not norm.pdf(h)."""
        p = self.model.basis.order + 1
        D = self.model.basis.derivative(self.grid, order=1)
        hp = D @ self.model.theta_[:p]
        h = self._h(self.grid)
        expected = _logistic.pdf(h) * np.maximum(hp, 0.0)
        actual = self.model.predict(self.grid, what="density")
        np.testing.assert_allclose(actual, expected)

    def test_density_differs_from_normal_density(self):
        p = self.model.basis.order + 1
        D = self.model.basis.derivative(self.grid, order=1)
        hp = D @ self.model.theta_[:p]
        h = self._h(self.grid)
        wrong = norm.pdf(h) * np.maximum(hp, 0.0)
        actual = self.model.predict(self.grid, what="density")
        assert not np.allclose(actual, wrong, atol=1e-6), (
            "logistic model predict(density) returned norm.pdf values"
        )

    def test_quantile_cdf_inverse(self):
        """CDF(quantile(p)) ≈ p for logistic model."""
        probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q = self.model.predict(probs, what="quantile")
        cdf_back = self.model.predict(q, what="distribution")
        np.testing.assert_allclose(cdf_back, probs, atol=1e-4)

    def test_quantile_in_support(self):
        probs = np.linspace(0.05, 0.95, 20)
        q = self.model.predict(probs, what="quantile")
        assert np.all(q >= 0.0) and np.all(q <= 1.0)

    def test_hazard_matches_logistic_ratio(self):
        """predict(hazard) must use logistic pdf/sf (with h' Jacobian), not normal."""
        basis = BernsteinBasis(order=4, support=(0.0, 1.0))
        model = ConditionalTransformationModel(
            basis, censoring=CensoringType.RIGHT, base_distribution="logistic"
        )
        rng = np.random.default_rng(3)
        y = rng.uniform(0.05, 0.95, 80)
        cd = CensoredData.right_censored(y, rng.random(80) < 0.3)
        model.fit(cd)

        grid = np.linspace(0.1, 0.9, 20)
        p = model.basis.order + 1
        B = model.basis.evaluate(grid)
        D = model.basis.derivative(grid, order=1)
        h = B @ model.theta_[:p]
        hp = D @ model.theta_[:p]

        expected = (
            _logistic.pdf(h) * np.maximum(hp, 0.0) / np.maximum(_logistic.sf(h), 1e-300)
        )
        actual = model.predict(grid, what="hazard")
        np.testing.assert_allclose(actual, expected)

    def test_hazard_differs_from_normal_hazard(self):
        basis = BernsteinBasis(order=4, support=(0.0, 1.0))
        model = ConditionalTransformationModel(
            basis, censoring=CensoringType.RIGHT, base_distribution="logistic"
        )
        rng = np.random.default_rng(3)
        y = rng.uniform(0.05, 0.95, 80)
        cd = CensoredData.right_censored(y, rng.random(80) < 0.3)
        model.fit(cd)

        grid = np.linspace(0.1, 0.9, 20)
        p = model.basis.order + 1
        B = model.basis.evaluate(grid)
        D = model.basis.derivative(grid, order=1)
        h = B @ model.theta_[:p]
        hp = D @ model.theta_[:p]

        wrong = norm.pdf(h) * np.maximum(hp, 0.0) / np.maximum(norm.sf(h), 1e-300)
        actual = model.predict(grid, what="hazard")
        assert not np.allclose(actual, wrong, atol=1e-6), (
            "logistic model predict(hazard) returned norm-based values"
        )

    def test_normal_model_unchanged(self):
        """Sanity: normal model still uses norm.cdf (default behaviour)."""
        model_n = MLT(order=5, support=(0.0, 1.0), base_distribution="normal").fit(
            self.y
        )
        p = model_n.basis.order + 1
        B = model_n.basis.evaluate(self.grid)
        h = B @ model_n.theta_[:p]
        expected = norm.cdf(h)
        actual = model_n.predict(self.grid, what="distribution")
        np.testing.assert_allclose(actual, expected)


# ---------------------------------------------------------------------------
# All 14 predict(what=...) types — closed-form consistency against scipy
# ---------------------------------------------------------------------------

_NEW_WHATS = (
    "trafo",
    "logdistribution",
    "survivor",
    "logsurvivor",
    "logdensity",
    "loghazard",
    "cumhazard",
    "logcumhazard",
    "odds",
    "logodds",
)


class TestPredictAllWhats:
    """Consistency tests for the 10 new predict() targets + regression guards."""

    @pytest.fixture(params=["normal", "logistic"])
    def fitted(self, request):
        rng = np.random.default_rng(11)
        y = np.sort(rng.uniform(0.05, 0.95, 150))
        model = MLT(order=5, support=(0.0, 1.0), base_distribution=request.param).fit(y)
        grid = np.linspace(0.1, 0.9, 40)
        p = model.basis.order + 1
        B = model.basis.evaluate(grid)
        D = model.basis.derivative(grid, order=1)
        h = B @ model.theta_[:p]
        hp = D @ model.theta_[:p]
        from mltpy.likelihood import _get_dist

        dist = _get_dist(model.base_distribution)
        return model, grid, h, hp, dist

    def test_trafo_equals_h(self, fitted):
        model, grid, h, _hp, _dist = fitted
        np.testing.assert_allclose(model.predict(grid, what="trafo"), h)

    def test_logdistribution_matches_dist_logcdf(self, fitted):
        model, grid, h, _hp, dist = fitted
        np.testing.assert_allclose(
            model.predict(grid, what="logdistribution"), dist.logcdf(h)
        )

    def test_survivor_matches_dist_sf(self, fitted):
        model, grid, h, _hp, dist = fitted
        np.testing.assert_allclose(model.predict(grid, what="survivor"), dist.sf(h))

    def test_logsurvivor_matches_dist_logsf(self, fitted):
        model, grid, h, _hp, dist = fitted
        np.testing.assert_allclose(
            model.predict(grid, what="logsurvivor"), dist.logsf(h)
        )

    def test_logdensity_matches_logpdf_plus_log_hp(self, fitted):
        model, grid, h, hp, dist = fitted
        expected = dist.logpdf(h) + np.log(np.maximum(hp, np.finfo(np.float64).tiny))
        np.testing.assert_allclose(model.predict(grid, what="logdensity"), expected)

    def test_loghazard_equals_logdensity_minus_logsurvivor(self, fitted):
        """Identity: loghazard ≡ logdensity − logsurvivor."""
        model, grid, _h, _hp, _dist = fitted
        lhzd = model.predict(grid, what="loghazard")
        lden = model.predict(grid, what="logdensity")
        lsurv = model.predict(grid, what="logsurvivor")
        np.testing.assert_allclose(lhzd, lden - lsurv)

    def test_cumhazard_equals_neg_log_survivor(self, fitted):
        model, grid, _h, _hp, _dist = fitted
        ch = model.predict(grid, what="cumhazard")
        lsurv = model.predict(grid, what="logsurvivor")
        np.testing.assert_allclose(ch, -lsurv)

    def test_logcumhazard_matches_log_of_cumhazard(self, fitted):
        model, grid, _h, _hp, _dist = fitted
        lch = model.predict(grid, what="logcumhazard")
        ch = model.predict(grid, what="cumhazard")
        np.testing.assert_allclose(lch, np.log(ch))

    def test_odds_matches_cdf_over_sf(self, fitted):
        model, grid, h, _hp, dist = fitted
        expected = dist.cdf(h) / dist.sf(h)
        np.testing.assert_allclose(
            model.predict(grid, what="odds"), expected, rtol=1e-10
        )

    def test_logodds_matches_logcdf_minus_logsf(self, fitted):
        model, grid, h, _hp, dist = fitted
        expected = dist.logcdf(h) - dist.logsf(h)
        np.testing.assert_allclose(model.predict(grid, what="logodds"), expected)

    def test_logodds_equals_log_of_odds(self, fitted):
        """Identity: logodds ≡ log(odds) where odds > 0."""
        model, grid, _h, _hp, _dist = fitted
        lo = model.predict(grid, what="logodds")
        o = model.predict(grid, what="odds")
        np.testing.assert_allclose(lo, np.log(o), rtol=1e-10)

    def test_hazard_equals_density_over_survivor(self, fitted):
        """Identity (fixed formula): hazard ≡ density / survivor."""
        model, grid, _h, _hp, _dist = fitted
        hzd = model.predict(grid, what="hazard")
        dens = model.predict(grid, what="density")
        surv = model.predict(grid, what="survivor")
        np.testing.assert_allclose(hzd, dens / np.maximum(surv, 1e-300))

    def test_exp_loghazard_equals_hazard(self, fitted):
        """Identity: exp(loghazard) ≡ hazard (lock hazard-family consistency)."""
        model, grid, _h, _hp, _dist = fitted
        hzd = model.predict(grid, what="hazard")
        lhzd = model.predict(grid, what="loghazard")
        np.testing.assert_allclose(np.exp(lhzd), hzd, rtol=1e-10)


class TestPredictLogScaleTailStability:
    """The numerical-stability justification for log-scale variants:
    log(primal) can underflow to -inf, but predict(what='logX') stays finite."""

    def setup_method(self):
        rng = np.random.default_rng(23)
        y = np.sort(rng.uniform(0.05, 0.95, 200))
        # Large support so the tails of h reach extreme z-values
        self.model = MLT(order=6, support=(-5.0, 5.0), base_distribution="normal").fit(
            rng.normal(0.0, 1.0, 200).clip(-4.9, 4.9)
        )
        del y

    def test_logsurvivor_finite_where_survivor_underflows(self):
        # Far-right point: S(h) underflows to 0, logS stays finite
        y_tail = np.array([4.95])
        surv = self.model.predict(y_tail, what="survivor")
        lsurv = self.model.predict(y_tail, what="logsurvivor")
        assert np.isfinite(lsurv).all()
        # Either survivor underflowed to zero (→ log(0) = -inf, bad) OR
        # it is nonzero and logsurvivor is more precise. Either way logsurvivor
        # must be finite and equal to log(surv) when surv > 0.
        if np.all(surv > 0):
            np.testing.assert_allclose(lsurv, np.log(surv), rtol=1e-6)

    def test_logdistribution_finite_where_cdf_underflows(self):
        # Far-left point: F(h) underflows to 0, logF stays finite
        y_tail = np.array([-4.95])
        cdf = self.model.predict(y_tail, what="distribution")
        lcdf = self.model.predict(y_tail, what="logdistribution")
        assert np.isfinite(lcdf).all()
        if np.all(cdf > 0):
            np.testing.assert_allclose(lcdf, np.log(cdf), rtol=1e-6)

    def test_log_variants_match_log_of_primal_in_bulk(self):
        grid = np.linspace(-2.0, 2.0, 30)
        for primal, log_variant in [
            ("distribution", "logdistribution"),
            ("survivor", "logsurvivor"),
            ("density", "logdensity"),
            ("cumhazard", "logcumhazard"),
            ("odds", "logodds"),
            ("hazard", "loghazard"),
        ]:
            p = self.model.predict(grid, what=primal)
            lp = self.model.predict(grid, what=log_variant)
            mask = p > 0
            np.testing.assert_allclose(
                lp[mask],
                np.log(p[mask]),
                rtol=1e-8,
                err_msg=f"{log_variant} != log({primal}) in bulk",
            )


class TestPredictCovariateAware:
    """Every new what must honour the covariate shift X @ beta."""

    def setup_method(self):
        rng = np.random.default_rng(31)
        n = 120
        self.X = rng.normal(0.0, 1.0, (n, 2))
        y = np.clip(0.5 + 0.1 * self.X[:, 0] + rng.normal(0.0, 0.1, n), 0.05, 0.95)
        self.model = MLT(order=4, support=(0.0, 1.0), base_distribution="normal").fit(
            y, X=self.X
        )
        self.grid = np.linspace(0.1, 0.9, 15)

    @pytest.mark.parametrize("what", _NEW_WHATS + ("distribution", "density", "hazard"))
    def test_prediction_shifts_with_X(self, what):
        m = len(self.grid)
        X0 = np.zeros((m, 2))
        X1 = np.ones((m, 2))
        v0 = self.model.predict(self.grid, X_new=X0, what=what)
        v1 = self.model.predict(self.grid, X_new=X1, what=what)
        # Shapes match the grid
        assert v0.shape == (m,)
        assert v1.shape == (m,)
        # Non-trivial covariate impact (beta is fitted non-zero)
        assert not np.allclose(v0, v1, atol=1e-8), (
            f"what={what!r} appears not to use X (v0 == v1)"
        )


class TestInvalidWhatListsAll:
    def test_error_mentions_new_types(self):
        model = MLT(order=3, support=(0.0, 1.0)).fit(simple_y())
        with pytest.raises(ValueError) as excinfo:
            model.predict(np.array([0.5]), what="banana")
        msg = str(excinfo.value)
        for w in _NEW_WHATS + ("distribution", "density", "hazard", "quantile"):
            assert w in msg, f"{w!r} missing from error message"


# ---------------------------------------------------------------------------
# AIC / BIC / anova
# ---------------------------------------------------------------------------


class TestModelSelectionAttributes:
    def test_n_obs_and_n_free_params_after_fit(self):
        y = simple_y(n=80)
        model = MLT(order=4, support=(0.0, 1.0)).fit(y)
        assert model.n_obs_ == 80
        assert model.n_free_params_ == 5  # order + 1

    def test_n_obs_with_covariates(self):
        rng = np.random.default_rng(7)
        n, q = 60, 2
        y = rng.uniform(0.05, 0.95, n)
        X = rng.standard_normal((n, q))
        model = MLT(order=3, support=(0.0, 1.0)).fit(y, X=X)
        assert model.n_obs_ == n
        assert model.n_free_params_ == 4 + q

    def test_n_obs_uses_censored_data_n(self):
        rng = np.random.default_rng(5)
        n = 50
        y = rng.uniform(0.05, 0.95, n)
        cd = CensoredData.right_censored(y, rng.random(n) < 0.3)
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(
            basis, censoring=CensoringType.RIGHT
        ).fit(cd)
        assert model.n_obs_ == n
        assert model.n_obs_ == cd.n

    def test_attrs_none_before_fit(self):
        model = make_ctm()
        assert model.n_obs_ is None
        assert model.n_free_params_ is None


class TestAIC:
    def test_formula(self):
        y = simple_y(n=80)
        model = MLT(order=4, support=(0.0, 1.0)).fit(y)
        expected = -2.0 * model.result_.log_likelihood + 2.0 * model.n_free_params_
        assert model.aic() == pytest.approx(expected)

    def test_returns_float(self):
        model = MLT(order=3, support=(0.0, 1.0)).fit(simple_y())
        assert isinstance(model.aic(), float)

    def test_aic_finite(self):
        model = MLT(order=4, support=(0.0, 1.0)).fit(simple_y())
        assert np.isfinite(model.aic())


class TestBIC:
    def test_formula(self):
        y = simple_y(n=80)
        model = MLT(order=4, support=(0.0, 1.0)).fit(y)
        expected = (
            -2.0 * model.result_.log_likelihood
            + np.log(model.n_obs_) * model.n_free_params_
        )
        assert model.bic() == pytest.approx(expected)

    def test_returns_float(self):
        model = MLT(order=3, support=(0.0, 1.0)).fit(simple_y())
        assert isinstance(model.bic(), float)

    def test_bic_penalises_more_than_aic_for_n_gt_7(self):
        y = simple_y(n=80)
        model = MLT(order=4, support=(0.0, 1.0)).fit(y)
        assert model.bic() > model.aic()


class TestAnova:
    def _two_nested(self, n: int = 100, seed: int = 0):
        y = np.random.default_rng(seed).uniform(0.05, 0.95, n)
        small = MLT(order=3, support=(0.0, 1.0)).fit(y)
        large = MLT(order=6, support=(0.0, 1.0)).fit(y)
        return small, large

    def test_returns_anova_result(self):
        small, large = self._two_nested()
        result = anova(small, large)
        assert isinstance(result, AnovaResult)

    def test_columns_have_correct_length(self):
        small, large = self._two_nested()
        result = anova(small, large)
        for col in (
            result.model_names,
            result.n_params,
            result.log_lik,
            result.df,
            result.deviance,
            result.p_value,
        ):
            assert len(col) == 2

    def test_first_row_test_columns_are_none(self):
        small, large = self._two_nested()
        result = anova(small, large)
        assert result.df[0] is None
        assert result.deviance[0] is None
        assert result.p_value[0] is None

    def test_models_sorted_ascending_by_n_params(self):
        small, large = self._two_nested()
        # Pass in reverse to confirm internal sort
        result = anova(large, small)
        assert result.n_params == (4, 7)  # order+1 for 3 and 6

    def test_model_names_preserve_caller_input_order(self):
        """Labels must reflect the caller's argument position, not the
        post-sort row index — otherwise multi-model tables are easy to misread."""
        small, large = self._two_nested()
        # Caller passes large first (#0), small second (#1)
        result = anova(large, small)
        # After sort: row 0 is small (caller arg #1), row 1 is large (caller arg #0)
        assert result.model_names[0].endswith("#1")
        assert result.model_names[1].endswith("#0")

    def test_df_equals_param_diff(self):
        small, large = self._two_nested()
        result = anova(small, large)
        assert result.df[1] == 3  # 7 − 4

    def test_deviance_non_negative_for_nested_fit(self):
        small, large = self._two_nested()
        result = anova(small, large)
        # Larger model should fit at least as well; deviance ≥ 0 modulo solver noise
        assert result.deviance[1] >= -1e-6

    def test_p_value_in_unit_interval(self):
        small, large = self._two_nested()
        result = anova(small, large)
        assert 0.0 <= result.p_value[1] <= 1.0

    def test_p_value_matches_chi2_sf(self):
        from scipy.stats import chi2

        small, large = self._two_nested()
        result = anova(small, large)
        d = max(result.deviance[1], 0.0)
        expected = float(chi2.sf(d, result.df[1]))
        assert result.p_value[1] == pytest.approx(expected)

    def test_three_models_chain(self):
        y = simple_y(n=120)
        m3 = MLT(order=3, support=(0.0, 1.0)).fit(y)
        m5 = MLT(order=5, support=(0.0, 1.0)).fit(y)
        m7 = MLT(order=7, support=(0.0, 1.0)).fit(y)
        result = anova(m3, m5, m7)
        assert result.n_params == (4, 6, 8)
        assert result.df == (None, 2, 2)
        assert result.df[1] == 2 and result.df[2] == 2

    def test_repr_is_string(self):
        small, large = self._two_nested()
        s = repr(anova(small, large))
        assert isinstance(s, str)
        assert "Pr(>Chisq)" in s

    def test_too_few_models_raises(self):
        model = MLT(order=3, support=(0.0, 1.0)).fit(simple_y())
        with pytest.raises(ValueError, match="at least 2"):
            anova(model)

    def test_unfitted_model_raises(self):
        small = MLT(order=3, support=(0.0, 1.0)).fit(simple_y())
        unfit = MLT(order=5, support=(0.0, 1.0))
        with pytest.raises(ValueError, match="is not fitted"):
            anova(small, unfit)

    def test_different_n_obs_raises(self):
        y1 = simple_y(n=60, seed=1)
        y2 = simple_y(n=80, seed=2)
        m1 = MLT(order=3, support=(0.0, 1.0)).fit(y1)
        m2 = MLT(order=5, support=(0.0, 1.0)).fit(y2)
        with pytest.raises(ValueError, match="sample size"):
            anova(m1, m2)

    def test_equal_n_params_raises(self):
        y = simple_y(n=80)
        m1 = MLT(order=4, support=(0.0, 1.0)).fit(y)
        m2 = MLT(order=4, support=(0.0, 1.0)).fit(y)
        with pytest.raises(ValueError, match="parameter counts"):
            anova(m1, m2)

    def test_negative_deviance_warns(self):
        """Non-nested models can produce D < 0; anova should warn rather than
        silently return an impossible LR statistic."""
        import dataclasses

        y = simple_y(n=80)
        small = MLT(order=3, support=(0.0, 1.0)).fit(y)
        large = MLT(order=5, support=(0.0, 1.0)).fit(y)
        # Force the larger model's ll below the smaller's to guarantee D < 0.
        large.result_ = dataclasses.replace(
            large.result_,
            log_likelihood=small.result_.log_likelihood - 1.0,
        )
        with pytest.warns(UserWarning, match="negative"):
            result = anova(small, large)
        assert result.deviance[1] is not None and result.deviance[1] < 0


# ---------------------------------------------------------------------------
# R reference integration test
# ---------------------------------------------------------------------------

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


@pytest.mark.skipif(
    not (REF_DIR / "mlt_aic_bic.txt").exists(),
    reason="R AIC/BIC reference file not generated yet",
)
def test_integration_r_reference_aic_bic():
    """AIC and BIC for both reference models match R's mlt::AIC / BIC.

    The reference models (order 3 and order 6) are fit on the same data
    written to reference/mlt_normal_y.txt so that AIC/BIC are directly
    comparable. Tolerance accounts for cross-solver differences in the
    optimum (the LL differs by < 0.5 nats per the existing reference test).
    """
    aic_s_r, bic_s_r, aic_l_r, bic_l_r = np.loadtxt(REF_DIR / "mlt_aic_bic.txt")
    y_ref = np.loadtxt(REF_DIR / "mlt_normal_y.txt")

    fit_small = MLT(order=3, support=(0.0, 1.0)).fit(y_ref)
    fit_large = MLT(order=6, support=(0.0, 1.0)).fit(y_ref)

    # AIC = -2*ll + 2*k → LL agreement within 0.5 nats translates to AIC
    # agreement within 1.0 (and identical k means the penalty terms match
    # exactly).
    assert abs(fit_small.aic() - aic_s_r) < 1.0
    assert abs(fit_small.bic() - bic_s_r) < 1.0
    assert abs(fit_large.aic() - aic_l_r) < 1.0
    assert abs(fit_large.bic() - bic_l_r) < 1.0


@pytest.mark.skipif(
    not (REF_DIR / "mlt_anova.txt").exists(),
    reason="R anova reference file not generated yet",
)
def test_integration_r_reference_anova():
    """anova(small, large) Chisq, df, and p-value match R's anova.mlt."""
    chisq_r, df_r, p_r = np.loadtxt(REF_DIR / "mlt_anova.txt")
    y_ref = np.loadtxt(REF_DIR / "mlt_normal_y.txt")

    fit_small = MLT(order=3, support=(0.0, 1.0)).fit(y_ref)
    fit_large = MLT(order=6, support=(0.0, 1.0)).fit(y_ref)
    result = anova(fit_small, fit_large)

    assert result.df[1] == int(df_r)
    # Deviance is 2 * (ll_large - ll_small); each LL agrees with R within
    # 0.5 nats, so deviance agrees within 2 nats.
    assert abs(result.deviance[1] - chisq_r) < 2.0
    # p-values can shift visibly when chisq shifts; bound loosely
    assert abs(result.p_value[1] - p_r) < 0.05


def test_integration_r_reference():
    """Fitted theta is close to R's mlt() output."""
    theta_r = np.loadtxt(REF_DIR / "mlt_normal_theta.txt")
    y_ref = np.loadtxt(REF_DIR / "mlt_normal_y.txt")

    order = len(theta_r) - 1
    model = MLT(order=order, support=(0.0, 1.0))
    model.fit(y_ref)

    # Log-likelihoods must agree (within tolerance from different optimisers)
    from mltpy.basis import BernsteinBasis
    from mltpy.likelihood import log_likelihood

    basis = BernsteinBasis(order=order, support=(0.0, 1.0))
    ll_r = log_likelihood(theta_r, basis, y_ref)
    ll_py = model.score(y_ref)
    assert ll_py >= ll_r - 0.5, (
        f"Python LL={ll_py:.4f} worse than R LL={ll_r:.4f} by more than 0.5 nats"
    )


# ---------------------------------------------------------------------------
# R reference: max_extreme_value and exponential base distributions
#
# Validates that mltpy.log_likelihood agrees with R's mlt::logLik at R's
# fitted theta for the new base distributions, and that mltpy's own fit
# reaches at least that log-likelihood.
# ---------------------------------------------------------------------------


class TestExponentialWithCovariates:
    """Exponential support ([0, ∞)) must hold per observation when covariates
    are present: ``h(y_i|x_i) >= 0`` for every training row ``i``.

    This reduces to ``theta_b[0] + X_i @ beta >= 0`` since h is monotone in y
    and ``min_y B_k(y) · theta_b = theta_b[0]``.
    """

    def _fit(self, seed: int = 31, n: int = 100, q: int = 2):
        from mltpy.basis import BernsteinBasis

        rng = np.random.default_rng(seed)
        y = rng.uniform(0.05, 0.95, n)
        X = rng.normal(0.0, 0.5, (n, q))
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        model = ConditionalTransformationModel(basis, base_distribution="exponential")
        model.fit(y, X=X)
        return model, basis, y, X

    def test_fit_converges(self):
        """Auglag and SLSQP must land at the same optimum on this fixture.

        ``y`` and ``X`` are independent random draws, so the MLE is degenerate:
        ``beta = 0`` and ``theta_b[0] = 0`` make every per-row support
        constraint ``theta_b[0] + X_i @ beta >= 0`` active simultaneously.
        100+ stacked active inequalities inflate the auglag KKT-stationarity
        residual to ~1e-2 even at the true optimum — a numerical artefact of
        PHR on this kind of degenerate active set, not a real divergence.
        Stronger than checking ``.converged``: assert both solvers reach the
        same log-likelihood, which is the property a user actually cares about.
        """
        from mltpy.optimizer import OptimizerConfig

        model, basis, y, X = self._fit()
        slsqp_model = ConditionalTransformationModel(
            basis,
            base_distribution="exponential",
            optimizer_config=OptimizerConfig(solver="slsqp"),
        )
        slsqp_model.fit(y, X=X)
        assert slsqp_model.result_.converged
        np.testing.assert_allclose(
            model.result_.log_likelihood,
            slsqp_model.result_.log_likelihood,
            rtol=1e-6,
            atol=1e-8,
        )

    def test_fitted_h_is_nonnegative_at_training_rows(self):
        """At the fitted parameters, h(y_i|x_i) >= 0 for every training row."""
        model, basis, y, X = self._fit()
        p = basis.order + 1
        theta_b = model.theta_[:p]
        beta = model.theta_[p:]
        h = basis.evaluate(y) @ theta_b + X @ beta
        # Feasibility within SLSQP tolerance
        assert h.min() >= -1e-6, f"min h(y|x) = {h.min():.3e}"

    def test_fitted_h_nonnegative_at_y_min_per_row(self):
        """Tightest feasibility point: min_y h(y|x_i) = theta_b[0] + X_i @ beta.

        This value must also be >= 0 — the actual constraint the optimiser
        imposes.
        """
        model, basis, _, X = self._fit()
        p = basis.order + 1
        theta_b0 = model.theta_[0]
        beta = model.theta_[p:]
        min_h_per_row = theta_b0 + X @ beta
        assert min_h_per_row.min() >= -1e-6

    def test_ll_finite_with_covariates(self):
        """Exponential + covariates produces a finite LL at the fitted theta."""
        model, _, y, X = self._fit()
        assert np.isfinite(model.result_.log_likelihood)
        assert np.isfinite(model.score(y, X=X))


@pytest.mark.parametrize(
    ("name", "theta_file", "y_file", "ll_file"),
    [
        (
            "max_extreme_value",
            "mlt_maxextrval_theta.txt",
            "mlt_maxextrval_y.txt",
            "mlt_maxextrval_ll.txt",
        ),
        (
            "exponential",
            "mlt_exponential_theta.txt",
            "mlt_exponential_y.txt",
            "mlt_exponential_ll.txt",
        ),
    ],
)
def test_integration_r_reference_new_distributions(name, theta_file, y_file, ll_file):
    """LL at R's theta matches R; mltpy fit reaches ≥ R's LL minus 0.5 nats."""
    required = [REF_DIR / f for f in (theta_file, y_file, ll_file)]
    if not all(p.exists() for p in required):
        pytest.skip(
            f"{name} reference files not yet generated — "
            "run Rscript reference/generate_reference.R"
        )

    theta_r = np.loadtxt(required[0])
    y_ref = np.loadtxt(required[1])
    ll_r = float(np.loadtxt(required[2]))

    order = len(theta_r) - 1
    from mltpy.basis import BernsteinBasis
    from mltpy.likelihood import log_likelihood

    basis = BernsteinBasis(order=order, support=(0.0, 1.0))
    ll_py_at_theta_r = log_likelihood(theta_r, basis, y_ref, base_distribution=name)
    # Same formula, same data, same theta → exact agreement with R mlt
    np.testing.assert_allclose(ll_py_at_theta_r, ll_r, rtol=1e-6, atol=1e-8)

    # mltpy's own fit should reach at least R's LL (minus optimiser slack)
    model = MLT(order=order, support=(0.0, 1.0), base_distribution=name).fit(y_ref)
    ll_py = model.score(y_ref)
    assert ll_py >= ll_r - 0.5, (
        f"{name}: Python LL={ll_py:.4f} worse than R LL={ll_r:.4f}"
    )
    if name == "exponential":
        # Feasibility: theta_b[0] >= 0 ensures h(y) >= 0 across support
        assert model.theta_[0] >= -1e-6


# ---------------------------------------------------------------------------
# residuals()
# ---------------------------------------------------------------------------


def _refit_at_r_theta(model, theta_R, beta_sign):
    """Inject R's theta (with sign-flipped beta block if needed) into a fitted
    mltpy model so residuals() evaluates at R's MLE rather than mltpy's.

    Mirrors the convention used in tests/test_vcov.py — tram::BoxCox uses
    ``negative = TRUE`` (β sign flipped), Colr / Coxph match mltpy directly.
    """
    p = model.basis.order + 1
    theta_mltpy = theta_R.copy()
    if beta_sign != 1.0 and len(theta_mltpy) > p:
        theta_mltpy[p:] *= beta_sign
    model.theta_ = theta_mltpy
    return model


class TestResidualsRReference:
    """Element-wise parity with ``mlt::residuals`` for the canonical fits."""

    def _load(self, model: str):
        required = [
            REF_DIR / f"vcov_{model}_y.txt",
            REF_DIR / f"vcov_{model}_x.txt",
            REF_DIR / f"vcov_{model}_support.txt",
            REF_DIR / f"vcov_{model}_theta.txt",
            REF_DIR / f"residuals_{model}_score.txt",
            REF_DIR / f"residuals_{model}_coxsnell.txt",
            REF_DIR / f"residuals_{model}_deviance.txt",
        ]
        if not all(p.exists() for p in required):
            pytest.skip(
                f"residuals_{model}_* reference files not yet generated — "
                "run Rscript reference/generate_reference.R"
            )
        data = {
            "y": np.loadtxt(required[0]),
            "x": np.loadtxt(required[1]).reshape(-1, 1),
            "support": tuple(np.loadtxt(required[2])),
            "theta": np.loadtxt(required[3]),
            "score": np.loadtxt(required[4]),
            "coxsnell": np.loadtxt(required[5]),
            "deviance": np.loadtxt(required[6]),
        }
        event_path = REF_DIR / f"vcov_{model}_event.txt"
        if event_path.exists():
            data["event"] = np.loadtxt(event_path).astype(int)
        return data

    def _fit_and_inject(self, model_cls, ref, beta_sign, censoring=None):
        if "event" in ref:
            cd = CensoredData.right_censored(ref["y"], censored=ref["event"] == 0)
            y_obj: np.ndarray | CensoredData = cd
        else:
            y_obj = ref["y"]
        order = len(ref["theta"]) - ref["x"].shape[1] - 1
        m = model_cls(order=order, support=ref["support"]).fit(y_obj, X=ref["x"])
        return _refit_at_r_theta(m, ref["theta"], beta_sign)

    def test_boxcox(self):
        from mltpy.tram import BoxCox

        ref = self._load("boxcox")
        m = self._fit_and_inject(BoxCox, ref, beta_sign=-1.0)
        np.testing.assert_allclose(
            m.residuals("score"), ref["score"], rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            m.residuals("cox-snell"), ref["coxsnell"], rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            m.residuals("deviance"), ref["deviance"], rtol=1e-6, atol=1e-10
        )

    def test_colr(self):
        from mltpy.tram import Colr

        ref = self._load("colr")
        m = self._fit_and_inject(Colr, ref, beta_sign=+1.0)
        np.testing.assert_allclose(
            m.residuals("score"), ref["score"], rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            m.residuals("cox-snell"), ref["coxsnell"], rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            m.residuals("deviance"), ref["deviance"], rtol=1e-6, atol=1e-10
        )

    def test_coxph(self):
        from mltpy.tram import Coxph

        ref = self._load("coxph")
        m = self._fit_and_inject(Coxph, ref, beta_sign=+1.0)
        np.testing.assert_allclose(
            m.residuals("score"), ref["score"], rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            m.residuals("cox-snell"), ref["coxsnell"], rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            m.residuals("deviance"), ref["deviance"], rtol=1e-6, atol=1e-10
        )


class TestResidualsProperties:
    def test_score_residuals_sum_to_zero_at_mle_uncensored(self):
        """Score equation: sum of intercept-score residuals at the MLE is 0."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 200)
        from mltpy.tram import BoxCox

        m = BoxCox(support=(float(y.min() - 0.1), float(y.max() + 0.1)), order=4).fit(y)
        # Sum of intercept-score residuals == ∂(-ℓ)/∂α at the MLE; bounded by
        # the optimiser's gradient tolerance (~1e-4 by default).
        assert abs(m.residuals("score").sum()) < 1e-3

    def test_cox_snell_mean_near_one_under_correct_model(self):
        """Cox-Snell residuals ~ Exp(1) under a correctly specified model."""
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 1000)
        from mltpy.tram import BoxCox

        m = BoxCox(support=(float(y.min() - 0.1), float(y.max() + 0.1)), order=6).fit(y)
        r = m.residuals("cox-snell")
        # Exp(1) has mean 1, var 1.  Sample SE of mean ≈ 1/sqrt(n) ≈ 0.032.
        assert abs(r.mean() - 1.0) < 0.1
        assert (r > 0).all()

    def test_residuals_shape_matches_n_obs(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(0.1, 0.9, 50)
        m = MLT(order=3, support=(0.0, 1.0)).fit(y)
        for kind in ("score", "cox-snell", "deviance"):
            assert m.residuals(kind).shape == (m.n_obs_,)

    def test_residuals_with_right_censoring_runs(self):
        """Right-censored fits produce length-n vectors with sane signs."""
        from mltpy.tram import Coxph

        rng = np.random.default_rng(3)
        n = 80
        y = np.abs(rng.normal(1.5, 0.5, n)) + 0.1
        event = rng.random(n) > 0.3
        cd = CensoredData.right_censored(y, censored=~event)
        m = Coxph(order=4, support=(0.0, float(y.max() + 0.5))).fit(cd)
        r_score = m.residuals("score")
        r_cs = m.residuals("cox-snell")
        assert r_score.shape == (n,)
        assert r_cs.shape == (n,)
        # Cox-Snell residuals are -log S, always > 0 for finite S < 1.
        assert (r_cs > 0).all()


class TestResidualsErrors:
    def test_unfitted_raises(self):
        m = make_ctm()
        with pytest.raises(NotFittedError):
            m.residuals()

    def test_invalid_type_raises(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0.1, 0.9, 30)
        m = MLT(order=3, support=(0.0, 1.0)).fit(y)
        with pytest.raises(ValueError, match="invalid"):
            m.residuals(type="bogus")  # type: ignore[arg-type]

    def test_default_type_is_score(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0.1, 0.9, 30)
        m = MLT(order=3, support=(0.0, 1.0)).fit(y)
        np.testing.assert_array_equal(m.residuals(), m.residuals("score"))

    def test_training_data_isolated_from_caller_mutation(self):
        """Mutating y after fit() must not change residuals()."""
        rng = np.random.default_rng(7)
        y = rng.uniform(0.1, 0.9, 40)
        m = MLT(order=3, support=(0.0, 1.0)).fit(y)
        r_before = m.residuals("cox-snell").copy()
        y[:] = 0.0
        np.testing.assert_array_equal(m.residuals("cox-snell"), r_before)


class TestResidualsIntervalCensoring:
    """Interval-censored model exercises the _log_diff_ndtr Taylor branch."""

    def test_narrow_interval_runs(self):
        rng = np.random.default_rng(11)
        n = 60
        y = rng.uniform(0.1, 0.9, n)
        # Very narrow intervals trigger the Taylor fallback in _log_diff_ndtr
        cd = CensoredData(
            lower=y - 1e-7,
            upper=y + 1e-7,
            exact=np.full(n, np.nan),
        )
        m = ConditionalTransformationModel(
            BernsteinBasis(order=4, support=(0.0, 1.0)),
            censoring=CensoringType.INTERVAL,
        ).fit(cd)
        r_score = m.residuals("score")
        r_cs = m.residuals("cox-snell")
        assert r_score.shape == (n,)
        assert np.all(np.isfinite(r_score))
        assert np.all(np.isfinite(r_cs))


class TestScalingStub:
    """ADR 0002 surface: scaling= kwarg behaviour on supported / unsupported
    paths.  After #71 closes the censoring + base-distribution coverage,
    every censoring type and every link except ``"exponential"`` accepts
    ``scaling=`` (InteractionBasis combinations remain out of scope and
    raise at construction).
    """

    def test_scaling_none_is_default_and_byte_identical(self):
        m1 = MLT(order=3, support=(0.0, 1.0))
        m2 = MLT(order=3, support=(0.0, 1.0), scaling=None)
        assert m1.scaling is None
        assert m2.scaling is None

    def test_scaling_ndarray_accepted_on_supported_path(self):
        X_s = np.ones((20, 1), dtype=float)
        model = MLT(order=3, support=(0.0, 1.0), scaling=X_s)
        assert model.scaling is not None
        assert model.scaling.shape == (20, 1)

    def test_scaling_rejected_for_exponential_base(self):
        X_s = np.ones((20, 1), dtype=float)
        with pytest.raises(ValueError, match="0002-scaling-terms"):
            MLT(
                order=3,
                support=(0.0, 1.0),
                scaling=X_s,
                base_distribution="exponential",
            )

    def test_scaling_accepted_for_each_censoring_type(self):
        """Issue #71: every censoring type now accepts scaling= at __init__.

        Construction succeeds for RIGHT / LEFT / INTERVAL just as it does
        for NONE; fitting parity is exercised in tests/test_scaling_censoring.py.
        """
        X_s = np.ones((20, 1), dtype=float)
        for cens in (
            CensoringType.NONE,
            CensoringType.RIGHT,
            CensoringType.LEFT,
            CensoringType.INTERVAL,
        ):
            model = MLT(order=3, support=(0.0, 1.0), scaling=X_s, censoring=cens)
            assert model.scaling is not None
            assert model.censoring is cens
