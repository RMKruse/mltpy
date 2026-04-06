"""Tests for pymlt.model — ConditionalTransformationModel and MLT."""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import pymlt
from pymlt.basis import BernsteinBasis
from pymlt.model import (
    ConditionalTransformationModel,
    ConvergenceWarning,
    MLT,
    NotFittedError,
)
from pymlt.optimizer import OptimizerConfig
from pymlt.variables import CensoredData, CensoringType


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

    def test_result_has_log_likelihood(self):
        model = make_ctm()
        model.fit(simple_y())
        assert np.isfinite(model.result_.log_likelihood)

    def test_convergence_warning(self):
        """Very tight iteration limit → ConvergenceWarning."""
        cfg = OptimizerConfig(max_iter=1, max_restarts=0)
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
        cd = CensoredData.right_censored(
            np.array([0.5, 1.5]), np.array([False, False])
        )
        with pytest.raises(ValueError, match="support"):
            model.fit(cd)

    def test_x_shape_mismatch_raises(self):
        model = make_ctm()
        y = simple_y(n=10)
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="Zeilen"):
            model.fit(y, X=X)

    def test_pandas_series_accepted(self):
        """Duck-typing: pd.Series is coerced via np.asarray."""
        pd = pytest.importorskip("pandas")
        model = make_ctm()
        s = pd.Series(simple_y())
        model.fit(s)  # must not raise
        assert model.is_fitted_


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
        assert np.all(np.diff(cdf) >= -1e-6), f"CDF not monotone: {np.diff(cdf).min():.2e}"

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

    def test_invalid_what_raises(self):
        with pytest.raises(ValueError, match="ungültig"):
            self.model.predict(self.y_grid, what="banana")

    def test_hazard_requires_right_censoring(self):
        with pytest.raises(NotImplementedError):
            self.model.predict(self.y_grid, what="hazard")

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
        from pymlt.likelihood import log_likelihood
        from pymlt.optimizer import _initial_theta

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
    assert np.all(diffs >= -1e-6), f"order={order}, seed={seed}: min diff={diffs.min():.2e}"


# ---------------------------------------------------------------------------
# R reference integration test
# ---------------------------------------------------------------------------

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


@pytest.mark.skipif(
    not (REF_DIR / "mlt_normal_theta.txt").exists(),
    reason="R reference files not generated yet",
)
def test_integration_r_reference():
    """Fitted theta is close to R's mlt() output."""
    theta_r = np.loadtxt(REF_DIR / "mlt_normal_theta.txt")
    y_ref = np.loadtxt(REF_DIR / "mlt_normal_y.txt")

    order = len(theta_r) - 1
    model = MLT(order=order, support=(0.0, 1.0))
    model.fit(y_ref)

    # Log-likelihoods must agree (within tolerance from different optimisers)
    from pymlt.likelihood import log_likelihood
    from pymlt.basis import BernsteinBasis
    basis = BernsteinBasis(order=order, support=(0.0, 1.0))
    ll_r = log_likelihood(theta_r, basis, y_ref)
    ll_py = model.score(y_ref)
    assert ll_py >= ll_r - 0.5, (
        f"Python LL={ll_py:.4f} worse than R LL={ll_r:.4f} by more than 0.5 nats"
    )
