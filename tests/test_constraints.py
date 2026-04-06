"""Tests for pymlt.constraints — monotonicity and boundary constraints."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.optimize import LinearConstraint, minimize

from pymlt.constraints import (
    BoundaryConstraint,
    MonotonicityConstraint,
    build_constraints,
)


# ---------------------------------------------------------------------------
# MonotonicityConstraint — matrix
# ---------------------------------------------------------------------------

class TestMonotonicityMatrix:
    def test_shape(self):
        D = MonotonicityConstraint(4).as_matrix()
        assert D.shape == (3, 4)

    def test_exact_values_n4(self):
        D = MonotonicityConstraint(4).as_matrix()
        expected = np.array([
            [-1,  1,  0,  0],
            [ 0, -1,  1,  0],
            [ 0,  0, -1,  1],
        ], dtype=float)
        np.testing.assert_array_equal(D, expected)

    def test_shape_general(self):
        for n in range(2, 8):
            D = MonotonicityConstraint(n).as_matrix()
            assert D.shape == (n - 1, n)

    def test_ascending_theta_satisfies(self):
        D = MonotonicityConstraint(5).as_matrix()
        theta = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert np.all(D @ theta >= 0)

    def test_equal_theta_satisfies(self):
        D = MonotonicityConstraint(4).as_matrix()
        theta = np.array([2.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_equal(D @ theta, 0.0)

    def test_descending_theta_violates(self):
        D = MonotonicityConstraint(4).as_matrix()
        theta = np.array([4.0, 3.0, 2.0, 1.0])
        assert np.all(D @ theta < 0)

    def test_n_params_1_raises(self):
        with pytest.raises(ValueError, match="n_params"):
            MonotonicityConstraint(1)

    def test_returns_copy(self):
        mc = MonotonicityConstraint(3)
        D1 = mc.as_matrix()
        D1[0, 0] = 999.0
        D2 = mc.as_matrix()
        assert D2[0, 0] != 999.0


# ---------------------------------------------------------------------------
# MonotonicityConstraint — scipy_constraint dict
# ---------------------------------------------------------------------------

class TestMonotonicityScipyConstraint:
    def test_type_is_ineq(self):
        c = MonotonicityConstraint(4).as_scipy_constraint()
        assert c["type"] == "ineq"

    def test_fun_ascending(self):
        c = MonotonicityConstraint(4).as_scipy_constraint()
        theta = np.array([1.0, 2.0, 3.0, 4.0])
        result = c["fun"](theta)
        assert np.all(result >= 0)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_fun_descending_negative(self):
        c = MonotonicityConstraint(4).as_scipy_constraint()
        theta = np.array([4.0, 3.0, 2.0, 1.0])
        assert np.all(c["fun"](theta) < 0)

    def test_jac_equals_matrix(self):
        mc = MonotonicityConstraint(5)
        c = mc.as_scipy_constraint()
        theta = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(c["jac"](theta), mc.as_matrix())

    def test_jac_constant(self):
        """Jacobian must be independent of theta (linear constraint)."""
        mc = MonotonicityConstraint(4)
        c = mc.as_scipy_constraint()
        J1 = c["jac"](np.array([0.0, 1.0, 2.0, 3.0]))
        J2 = c["jac"](np.array([10.0, 20.0, 30.0, 40.0]))
        np.testing.assert_array_equal(J1, J2)


# ---------------------------------------------------------------------------
# MonotonicityConstraint — LinearConstraint
# ---------------------------------------------------------------------------

class TestMonotonicityLinearConstraint:
    def test_returns_linear_constraint(self):
        lc = MonotonicityConstraint(4).as_LinearConstraint()
        assert isinstance(lc, LinearConstraint)

    def test_lb_is_zero(self):
        lc = MonotonicityConstraint(4).as_LinearConstraint()
        np.testing.assert_array_equal(lc.lb, 0.0)

    def test_ub_is_inf(self):
        lc = MonotonicityConstraint(4).as_LinearConstraint()
        assert np.all(np.isinf(lc.ub))

    def test_A_equals_matrix(self):
        mc = MonotonicityConstraint(5)
        lc = mc.as_LinearConstraint()
        np.testing.assert_array_equal(lc.A.toarray() if hasattr(lc.A, "toarray") else lc.A,
                                      mc.as_matrix())


# ---------------------------------------------------------------------------
# BoundaryConstraint
# ---------------------------------------------------------------------------

class TestBoundaryConstraint:
    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            BoundaryConstraint(4, lower=None, upper=None)

    def test_lower_only(self):
        bc = BoundaryConstraint(4, lower=0.0, upper=None)
        cs = bc.as_scipy_constraint()
        assert len(cs) == 1
        assert cs[0]["type"] == "eq"
        theta = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(cs[0]["fun"](theta), 0.0)

    def test_upper_only(self):
        bc = BoundaryConstraint(4, lower=None, upper=5.0)
        cs = bc.as_scipy_constraint()
        assert len(cs) == 1
        assert cs[0]["type"] == "eq"
        theta = np.array([1.0, 2.0, 3.0, 5.0])
        np.testing.assert_allclose(cs[0]["fun"](theta), 0.0)

    def test_both_bounds(self):
        bc = BoundaryConstraint(4, lower=1.0, upper=4.0)
        cs = bc.as_scipy_constraint()
        assert len(cs) == 2
        theta = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(cs[0]["fun"](theta), 0.0)
        np.testing.assert_allclose(cs[1]["fun"](theta), 0.0)

    def test_lower_violated(self):
        bc = BoundaryConstraint(4, lower=1.0, upper=None)
        cs = bc.as_scipy_constraint()
        theta = np.array([2.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(cs[0]["fun"](theta), 1.0)  # 2 - 1 = 1

    def test_upper_violated(self):
        bc = BoundaryConstraint(4, lower=None, upper=3.0)
        cs = bc.as_scipy_constraint()
        theta = np.array([1.0, 2.0, 3.0, 5.0])
        np.testing.assert_allclose(cs[0]["fun"](theta), 2.0)  # 5 - 3 = 2

    def test_jac_lower(self):
        bc = BoundaryConstraint(4, lower=0.0, upper=None)
        cs = bc.as_scipy_constraint()
        jac = cs[0]["jac"](np.zeros(4))
        expected = np.eye(4)[0]
        np.testing.assert_array_equal(jac, expected)

    def test_jac_upper(self):
        bc = BoundaryConstraint(4, lower=None, upper=1.0)
        cs = bc.as_scipy_constraint()
        jac = cs[0]["jac"](np.zeros(4))
        expected = np.eye(4)[-1]
        np.testing.assert_array_equal(jac, expected)

    def test_linear_constraint_equality(self):
        bc = BoundaryConstraint(4, lower=1.0, upper=4.0)
        lc = bc.as_LinearConstraint()
        assert isinstance(lc, LinearConstraint)
        np.testing.assert_array_equal(lc.lb, lc.ub)
        np.testing.assert_array_equal(lc.lb, [1.0, 4.0])


# ---------------------------------------------------------------------------
# build_constraints
# ---------------------------------------------------------------------------

class TestBuildConstraints:
    def test_slsqp_returns_list_of_dicts(self):
        result = build_constraints(4, solver="slsqp")
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_trust_constr_returns_list_of_lc(self):
        result = build_constraints(4, solver="trust-constr")
        assert isinstance(result, list)
        assert all(isinstance(c, LinearConstraint) for c in result)

    def test_no_boundary_slsqp_length(self):
        result = build_constraints(5, solver="slsqp")
        assert len(result) == 1  # only monotonicity

    def test_lower_only_slsqp_length(self):
        result = build_constraints(5, lower=0.0, solver="slsqp")
        assert len(result) == 2  # mono + lower boundary

    def test_both_bounds_slsqp_length(self):
        result = build_constraints(5, lower=0.0, upper=1.0, solver="slsqp")
        assert len(result) == 3  # mono + 2 boundary

    def test_trust_constr_no_boundary_length(self):
        result = build_constraints(5, solver="trust-constr")
        assert len(result) == 1

    def test_trust_constr_both_bounds_length(self):
        result = build_constraints(5, lower=0.0, upper=1.0, solver="trust-constr")
        assert len(result) == 2

    def test_minimize_slsqp_enforces_monotonicity(self):
        """Constrained QP: nearest monotone vector to a descending target."""
        # target = [4, 3, 2, 1] (strictly descending)
        # Nearest monotone (non-decreasing) solution under L2: [2.5, 2.5, 2.5, 2.5]
        target = np.array([4.0, 3.0, 2.0, 1.0])
        constraints = build_constraints(4, solver="slsqp")

        def obj(t):
            return np.sum((t - target) ** 2)

        def grad(t):
            return 2 * (t - target)

        result = minimize(
            obj, x0=target, jac=grad, method="SLSQP", constraints=constraints
        )
        assert result.success, result.message
        # Solution must be non-decreasing
        assert np.all(np.diff(result.x) >= -1e-6)

    def test_minimize_trust_constr_enforces_monotonicity(self):
        """Same QP with trust-constr."""
        target = np.array([5.0, 3.0, 4.0, 1.0])
        constraints = build_constraints(4, solver="trust-constr")

        def obj(t):
            return np.sum((t - target) ** 2)

        def grad(t):
            return 2 * (t - target)

        result = minimize(
            obj, x0=np.zeros(4), jac=grad,
            method="trust-constr", constraints=constraints
        )
        assert result.success or result.status in (1, 2)
        assert np.all(np.diff(result.x) >= -1e-5)

    def test_minimize_with_boundary_constraints(self):
        """Constrained QP with fixed endpoints."""
        target = np.array([4.0, 3.0, 2.0, 1.0])
        constraints = build_constraints(4, lower=0.0, upper=1.0, solver="slsqp")

        def obj(t):
            return np.sum((t - target) ** 2)

        def grad(t):
            return 2 * (t - target)

        result = minimize(
            obj, x0=np.array([0.0, 0.3, 0.6, 1.0]),
            jac=grad, method="SLSQP", constraints=constraints
        )
        assert result.success, result.message
        assert np.all(np.diff(result.x) >= -1e-6)
        np.testing.assert_allclose(result.x[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.x[-1], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(
    n_params=st.integers(2, 10),
    seed=st.integers(0, 2**31 - 1),
)
@settings(max_examples=300)
def test_monotone_property_ascending(n_params, seed):
    """D @ theta >= 0 for any non-decreasing theta."""
    rng = np.random.default_rng(seed)
    theta = np.sort(rng.uniform(-100, 100, size=n_params))
    D = MonotonicityConstraint(n_params).as_matrix()
    assert np.all(D @ theta >= -1e-12)


@given(
    n_params=st.integers(2, 10),
    seed=st.integers(0, 2**31 - 1),
)
@settings(max_examples=300)
def test_monotone_property_strictly_descending(n_params, seed):
    """D @ theta < 0 for any strictly decreasing theta."""
    rng = np.random.default_rng(seed)
    increments = rng.uniform(0.01, 1.0, size=n_params)
    theta = np.cumsum(increments)[::-1]  # strictly decreasing
    D = MonotonicityConstraint(n_params).as_matrix()
    assert np.all(D @ theta < 0)
