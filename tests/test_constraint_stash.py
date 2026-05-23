"""Tests for constraint-matrix stashing on fitted models — issue #80.

After fit() the model should expose:
  _A_ineq_  — inequality constraint matrix (auglag only; None for SLSQP/trust-constr)
  _C_eq_    — equality constraint matrix when lower/upper are pinned; None otherwise
"""

from __future__ import annotations

import numpy as np

from pymlt.basis import BernsteinBasis, InteractionBasis, OneHotBasis
from pymlt.model import MLT, ConditionalTransformationModel
from pymlt.optimizer import OptimizerConfig
from pymlt.tram import Coxph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _exact_y(n: int = 80) -> np.ndarray:
    return RNG.uniform(0.05, 0.95, n)


def _survival_data(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """(y, X) for a Coxph fit — survival times + one covariate."""
    y = RNG.exponential(scale=3.0, size=n) + 0.1
    X = RNG.standard_normal((n, 1))
    return y, X


# ---------------------------------------------------------------------------
# Tracer bullet: _A_ineq_ is stashed after an auglag fit
# ---------------------------------------------------------------------------


class TestAineqStashedAuglag:
    def test_a_ineq_not_none_after_auglag_fit(self):
        model = MLT(order=3, support=(0.0, 1.0)).fit(_exact_y())
        assert model._A_ineq_ is not None

    def test_a_ineq_is_ndarray(self):
        model = MLT(order=3, support=(0.0, 1.0)).fit(_exact_y())
        assert isinstance(model._A_ineq_, np.ndarray)

    def test_a_ineq_shape_no_covariates(self):
        # order=3 → p=4; monotonicity rows = p-1 = 3; total_params=4
        model = MLT(order=3, support=(0.0, 1.0)).fit(_exact_y())
        assert model._A_ineq_.shape == (3, 4)

    def test_a_ineq_shape_with_covariates(self):
        # order=3, p=4; q_d=1 covariate; total_params=5; m_ineq=3
        y, X = _survival_data()
        model = Coxph(order=3, support=(0, y.max() + 0.5)).fit(y, X)
        p = 4
        q_d = 1
        assert model._A_ineq_.shape == (3, p + q_d)

    def test_a_ineq_shape_matches_issue_spec(self):
        """Shape is (m_ineq, p + q_d + q_s) as specified in #80."""
        y, X = _survival_data()
        model = Coxph(order=3, support=(0, y.max() + 0.5)).fit(y, X)
        p = model.basis.order + 1
        q_d = X.shape[1]
        q_s = 0
        m_ineq, total = model._A_ineq_.shape
        assert total == p + q_d + q_s
        assert m_ineq == model.basis.order  # p-1 monotonicity rows


# ---------------------------------------------------------------------------
# Constraint slacks: A_ineq @ theta_ >= 0
# ---------------------------------------------------------------------------


class TestConstraintSlacks:
    def test_slacks_nonneg_no_covariates(self):
        model = MLT(order=4, support=(0.0, 1.0)).fit(_exact_y())
        slacks = model._A_ineq_ @ model.theta_
        # Auglag KKT tolerance is ~1e-5; allow small numerical slack violations.
        assert np.all(slacks >= -1e-7)

    def test_slacks_nonneg_with_covariates(self):
        y, X = _survival_data()
        model = Coxph(order=3, support=(0, y.max() + 0.5)).fit(y, X)
        slacks = model._A_ineq_ @ model.theta_
        assert np.all(slacks >= -1e-7)

    def test_slacks_dtype_float64(self):
        model = MLT(order=3, support=(0.0, 1.0)).fit(_exact_y())
        slacks = model._A_ineq_ @ model.theta_
        assert slacks.dtype == np.float64


# ---------------------------------------------------------------------------
# _C_eq_: None for unpinned fits, non-None when lower/upper are pinned
# ---------------------------------------------------------------------------


class TestCeqStash:
    def test_c_eq_none_no_pinning(self):
        model = MLT(order=3, support=(0.0, 1.0)).fit(_exact_y())
        assert model._C_eq_ is None

    def test_c_eq_not_none_lower_pinned(self):
        config = OptimizerConfig(lower=0.0)
        model = MLT(order=3, support=(0.0, 1.0), optimizer_config=config).fit(
            _exact_y()
        )
        assert model._C_eq_ is not None

    def test_c_eq_shape_lower_pinned(self):
        # One equality row pinning theta[0] = 0.0; total_params = 4
        config = OptimizerConfig(lower=0.0)
        model = MLT(order=3, support=(0.0, 1.0), optimizer_config=config).fit(
            _exact_y()
        )
        assert model._C_eq_.shape == (1, 4)

    def test_c_eq_not_none_upper_pinned(self):
        config = OptimizerConfig(upper=5.0)
        model = MLT(order=3, support=(0.0, 1.0), optimizer_config=config).fit(
            _exact_y()
        )
        assert model._C_eq_ is not None

    def test_c_eq_shape_both_pinned(self):
        config = OptimizerConfig(lower=0.0, upper=1.0)
        model = MLT(order=3, support=(0.0, 1.0), optimizer_config=config).fit(
            _exact_y()
        )
        assert model._C_eq_.shape == (2, 4)


# ---------------------------------------------------------------------------
# Non-auglag solvers: both attributes must be None
# ---------------------------------------------------------------------------


class TestNonAuglagSolversNone:
    def test_slsqp_a_ineq_none(self):
        config = OptimizerConfig(solver="slsqp")
        model = MLT(order=3, support=(0.0, 1.0), optimizer_config=config).fit(
            _exact_y()
        )
        assert model._A_ineq_ is None

    def test_slsqp_c_eq_none(self):
        config = OptimizerConfig(solver="slsqp")
        model = MLT(order=3, support=(0.0, 1.0), optimizer_config=config).fit(
            _exact_y()
        )
        assert model._C_eq_ is None

    def test_trust_constr_a_ineq_none(self):
        config = OptimizerConfig(solver="trust-constr")
        model = MLT(order=3, support=(0.0, 1.0), optimizer_config=config).fit(
            _exact_y()
        )
        assert model._A_ineq_ is None

    def test_trust_constr_c_eq_none(self):
        config = OptimizerConfig(solver="trust-constr")
        model = MLT(order=3, support=(0.0, 1.0), optimizer_config=config).fit(
            _exact_y()
        )
        assert model._C_eq_ is None


# ---------------------------------------------------------------------------
# Before fit: both attributes are None
# ---------------------------------------------------------------------------


class TestBeforeFitNone:
    def test_a_ineq_none_before_fit(self):
        model = MLT(order=3, support=(0.0, 1.0))
        assert model._A_ineq_ is None

    def test_c_eq_none_before_fit(self):
        model = MLT(order=3, support=(0.0, 1.0))
        assert model._C_eq_ is None


# ---------------------------------------------------------------------------
# Interaction basis: _A_ineq_ has the Kronecker shape (p-1)*q × p*q
# ---------------------------------------------------------------------------


class TestInteractionAuglagStash:
    def test_interaction_a_ineq_not_none(self):
        rng = np.random.default_rng(7)
        n, p, q = 60, 4, 3
        y = rng.uniform(0.05, 0.95, n)
        X = rng.integers(0, q, size=n)
        y_basis = BernsteinBasis(order=p - 1, support=(0.0, 1.0))
        x_basis = OneHotBasis(K=q)
        basis = InteractionBasis(y_basis, x_basis)
        model = ConditionalTransformationModel(basis).fit(y, X)
        assert model._A_ineq_ is not None

    def test_interaction_a_ineq_shape(self):
        rng = np.random.default_rng(7)
        n, p, q = 60, 4, 3
        y = rng.uniform(0.05, 0.95, n)
        X = rng.integers(0, q, size=n)
        y_basis = BernsteinBasis(order=p - 1, support=(0.0, 1.0))
        x_basis = OneHotBasis(K=q)
        basis = InteractionBasis(y_basis, x_basis)
        model = ConditionalTransformationModel(basis).fit(y, X)
        # Kronecker shape: (p-1)*q × p*q
        assert model._A_ineq_.shape == ((p - 1) * q, p * q)

    def test_interaction_c_eq_none(self):
        rng = np.random.default_rng(7)
        n, p, q = 60, 4, 3
        y = rng.uniform(0.05, 0.95, n)
        X = rng.integers(0, q, size=n)
        y_basis = BernsteinBasis(order=p - 1, support=(0.0, 1.0))
        x_basis = OneHotBasis(K=q)
        basis = InteractionBasis(y_basis, x_basis)
        model = ConditionalTransformationModel(basis).fit(y, X)
        assert model._C_eq_ is None
