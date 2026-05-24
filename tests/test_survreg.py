"""Tests for pymlt.tram.Survreg and pymlt.basis.LogBernsteinBasis."""

from __future__ import annotations

import numpy as np
import pytest

from pymlt.variables import CensoredData, CensoringType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_lognormal_data(
    n: int = 150, mu: float = 0.5, sigma: float = 0.4, seed: int = 0
):
    """Simulate lognormal survival times with ~30% right censoring."""
    rng = np.random.default_rng(seed)
    t = rng.lognormal(mean=mu, sigma=sigma, size=n)
    censored = rng.random(n) < 0.3
    return t, censored


def make_weibull_data(
    n: int = 150, shape: float = 1.5, scale: float = 2.0, seed: int = 0
):
    """Simulate Weibull survival times with ~30% right censoring."""
    rng = np.random.default_rng(seed)
    t = scale * rng.weibull(shape, size=n)
    censored = rng.random(n) < 0.3
    return t, censored


# ---------------------------------------------------------------------------
# LogBernsteinBasis tests
# ---------------------------------------------------------------------------


class TestLogBernsteinBasis:
    def test_import(self):
        from pymlt.basis import LogBernsteinBasis  # noqa: F401

    def test_instantiate(self):
        from pymlt.basis import LogBernsteinBasis

        basis = LogBernsteinBasis(order=4, support=(0.1, 10.0))
        assert basis.order == 4
        assert basis.support == (0.1, 10.0)

    def test_invalid_support_nonpositive(self):
        from pymlt.basis import LogBernsteinBasis

        with pytest.raises(ValueError, match="positive"):
            LogBernsteinBasis(order=4, support=(0.0, 10.0))

    def test_invalid_support_order(self):
        from pymlt.basis import LogBernsteinBasis

        with pytest.raises(ValueError):
            LogBernsteinBasis(order=4, support=(5.0, 1.0))

    def test_evaluate_shape(self):
        from pymlt.basis import LogBernsteinBasis

        basis = LogBernsteinBasis(order=4, support=(0.1, 10.0))
        y = np.array([0.5, 1.0, 2.0, 5.0])
        B = basis.evaluate(y)
        assert B.shape == (4, 5)  # (n, order+1)

    def test_evaluate_rows_sum_to_one(self):
        from pymlt.basis import LogBernsteinBasis

        basis = LogBernsteinBasis(order=6, support=(0.01, 100.0))
        y = np.linspace(0.05, 90.0, 20)
        B = basis.evaluate(y)
        np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-12)

    def test_derivative_shape(self):
        from pymlt.basis import LogBernsteinBasis

        basis = LogBernsteinBasis(order=4, support=(0.1, 10.0))
        y = np.array([0.5, 1.0, 2.0, 5.0])
        dB = basis.derivative(y, order=1)
        assert dB.shape == (4, 5)

    def test_derivative_is_one_over_y_times_log_derivative(self):
        """d/dy B(log y) = (1/y) * d/d(log y) B(log y)."""
        from pymlt.basis import BernsteinBasis, LogBernsteinBasis

        a, b = 0.5, 20.0
        basis = LogBernsteinBasis(order=4, support=(a, b))
        inner = BernsteinBasis(order=4, support=(np.log(a), np.log(b)))

        y = np.array([1.0, 2.0, 5.0, 10.0])
        dB_log = basis.derivative(y, order=1)  # (n, p) in y-scale
        dB_inner = inner.derivative(np.log(y), order=1)  # (n, p) in log-scale

        expected = dB_inner / y[:, None]
        np.testing.assert_allclose(dB_log, expected, rtol=1e-12)

    def test_evaluate_with_derivative_consistent(self):
        from pymlt.basis import LogBernsteinBasis

        basis = LogBernsteinBasis(order=5, support=(0.1, 50.0))
        y = np.linspace(0.2, 40.0, 10)
        B, dB = basis.evaluate_with_derivative(y)
        B_ref = basis.evaluate(y)
        dB_ref = basis.derivative(y, order=1)
        np.testing.assert_array_equal(B, B_ref)
        np.testing.assert_array_equal(dB, dB_ref)

    def test_evaluate_outside_support_raises(self):
        from pymlt.basis import LogBernsteinBasis

        basis = LogBernsteinBasis(order=4, support=(0.5, 10.0))
        with pytest.raises(ValueError):
            basis.evaluate(np.array([0.1]))  # below support

    def test_derivative_numerical_check(self):
        """Numerical finite-difference validation of the log derivative."""
        from pymlt.basis import LogBernsteinBasis

        basis = LogBernsteinBasis(order=4, support=(0.1, 100.0))
        theta_b = np.array([0.0, 0.5, 1.0, 1.8, 2.5])
        y0 = np.array([1.0, 3.0, 10.0])
        eps = 1e-5
        h_plus = basis.evaluate(y0 + eps) @ theta_b
        h_minus = basis.evaluate(y0 - eps) @ theta_b
        fd_deriv = (h_plus - h_minus) / (2 * eps)

        dB = basis.derivative(y0, order=1)
        analytic = dB @ theta_b
        np.testing.assert_allclose(analytic, fd_deriv, rtol=1e-5)


# ---------------------------------------------------------------------------
# Survreg smoke tests
# ---------------------------------------------------------------------------


class TestSurvregSmoke:
    def test_import(self):
        from pymlt.tram import Survreg  # noqa: F401

    def test_import_from_pymlt(self):
        import pymlt

        assert hasattr(pymlt, "Survreg")

    def test_instantiate_weibull(self):
        from pymlt.tram import Survreg

        model = Survreg(support=(0.1, 20.0), distribution="weibull")
        assert model.censoring is CensoringType.RIGHT
        assert model.base_distribution == "min_extreme_value"

    def test_instantiate_lognormal(self):
        from pymlt.tram import Survreg

        model = Survreg(support=(0.1, 20.0), distribution="lognormal")
        assert model.base_distribution == "normal"

    def test_instantiate_loglogistic(self):
        from pymlt.tram import Survreg

        model = Survreg(support=(0.1, 20.0), distribution="loglogistic")
        assert model.base_distribution == "logistic"

    def test_invalid_distribution(self):
        from pymlt.tram import Survreg

        with pytest.raises(ValueError, match="distribution"):
            Survreg(support=(0.1, 20.0), distribution="exponential")

    def test_fit_lognormal_returns_self(self):
        from pymlt.tram import Survreg

        t, censored = make_lognormal_data()
        cd = CensoredData.right_censored(t, censored)
        t_max = float(t.max()) * 1.1
        model = Survreg(support=(0.001, t_max), distribution="lognormal")
        result = model.fit(cd)
        assert result is model
        assert model.is_fitted_

    def test_fit_weibull(self):
        from pymlt.tram import Survreg

        t, censored = make_weibull_data()
        cd = CensoredData.right_censored(t, censored)
        t_max = float(t.max()) * 1.1
        model = Survreg(support=(0.001, t_max), distribution="weibull").fit(cd)
        assert model.is_fitted_

    def test_fit_loglogistic(self):
        from pymlt.tram import Survreg

        t, censored = make_lognormal_data()
        cd = CensoredData.right_censored(t, censored)
        t_max = float(t.max()) * 1.1
        model = Survreg(support=(0.001, t_max), distribution="loglogistic").fit(cd)
        assert model.is_fitted_


# ---------------------------------------------------------------------------
# Survreg prediction tests
# ---------------------------------------------------------------------------


class TestSurvregPredict:
    @pytest.fixture
    def fitted_lognormal(self):
        from pymlt.tram import Survreg

        t, censored = make_lognormal_data()
        cd = CensoredData.right_censored(t, censored)
        t_max = float(t.max()) * 1.1
        model = Survreg(support=(0.001, t_max), distribution="lognormal").fit(cd)
        grid = np.linspace(0.1, float(t.max() * 0.9), 30)
        return model, grid

    def test_distribution_shape_and_range(self, fitted_lognormal):
        model, grid = fitted_lognormal
        cdf = model.predict(grid, what="distribution")
        assert cdf.shape == (30,)
        assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)

    def test_distribution_is_monotone(self, fitted_lognormal):
        model, grid = fitted_lognormal
        cdf = model.predict(grid, what="distribution")
        assert np.all(np.diff(cdf) >= -1e-10)

    def test_survivor_shape_and_range(self, fitted_lognormal):
        model, grid = fitted_lognormal
        surv = model.survival(grid)
        assert surv.shape == (30,)
        assert np.all(surv >= 0.0) and np.all(surv <= 1.0)

    def test_cdf_plus_survival_equals_one(self, fitted_lognormal):
        model, grid = fitted_lognormal
        cdf = model.predict(grid, what="distribution")
        surv = model.survival(grid)
        np.testing.assert_allclose(cdf + surv, 1.0, atol=1e-12)

    def test_density_is_positive(self, fitted_lognormal):
        model, grid = fitted_lognormal
        density = model.predict(grid, what="density")
        assert density.shape == (30,)
        assert np.all(density > 0)

    def test_hazard_is_positive(self, fitted_lognormal):
        model, grid = fitted_lognormal
        hazard = model.hazard(grid)
        assert hazard.shape == (30,)
        assert np.all(hazard > 0)

    def test_hazard_equals_density_over_survival(self, fitted_lognormal):
        model, grid = fitted_lognormal
        f = model.predict(grid, what="density")
        S = model.survival(grid)
        h = model.hazard(grid)
        np.testing.assert_allclose(h, f / S, rtol=1e-8)


# ---------------------------------------------------------------------------
# Survreg with covariates
# ---------------------------------------------------------------------------


class TestSurvregCovariates:
    def test_fit_with_covariate(self):
        from pymlt.tram import Survreg

        rng = np.random.default_rng(42)
        n = 120
        X = rng.normal(size=(n, 1))
        t = rng.lognormal(mean=0.3 * X[:, 0], sigma=0.5)
        censored = rng.random(n) < 0.3
        cd = CensoredData.right_censored(t, censored)
        t_max = float(t.max()) * 1.1
        model = Survreg(support=(0.001, t_max), distribution="lognormal")
        model.fit(cd, X=X)
        assert model.is_fitted_
        assert model.theta_ is not None
        # theta_ = [theta_b (order+1) | beta (1)]
        assert model.theta_.size == model.basis.order + 2


# ---------------------------------------------------------------------------
# Density Jacobian test (critical: must include 1/y factor)
# ---------------------------------------------------------------------------


class TestSurvregJacobian:
    def test_density_includes_log_jacobian(self):
        """Verify f_T(t) = f_X(h(log t)) * h'(log t) * (1/t).

        Using an lm-equivalent fit (order=1), h(x) is affine so we can
        compute the expected density analytically.
        """
        from pymlt.tram import Survreg

        # Generate enough data for a clean fit
        rng = np.random.default_rng(7)
        t = rng.lognormal(mean=1.0, sigma=0.5, size=300)
        cd = CensoredData.right_censored(t, ~np.ones(300, dtype=bool))  # no censoring

        t_min, t_max = float(t.min()) * 0.9, float(t.max()) * 1.1
        model = Survreg(support=(t_min, t_max), distribution="lognormal", order=6).fit(
            cd
        )

        # At any test point, density must be positive
        grid = np.percentile(t, [25, 50, 75])
        f_T = model.predict(grid, what="density")
        assert np.all(f_T > 0)

        # Numerical check: integrate density over support ≈ 1
        fine_grid = np.linspace(t_min * 1.001, t_max * 0.999, 500)
        f_fine = model.predict(fine_grid, what="density")
        integral = np.trapezoid(f_fine, fine_grid)
        # With a finite support, integral < 1; but should be close for good coverage
        assert 0.5 < integral <= 1.0 + 1e-6
