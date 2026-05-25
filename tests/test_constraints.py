"""Tests for mltpy.constraints — monotonicity and boundary constraints."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import LinearConstraint, minimize

from mltpy.constraints import (
    BoundaryConstraint,
    MonotonicityConstraint,
    build_constraint_matrices,
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
        expected = np.array(
            [
                [-1, 1, 0, 0],
                [0, -1, 1, 0],
                [0, 0, -1, 1],
            ],
            dtype=float,
        )
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

    def test_n_params_0_raises(self):
        with pytest.raises(ValueError, match="n_params"):
            MonotonicityConstraint(0)

    def test_n_params_1_empty_matrix(self):
        # K=2 ordinal models need n_params == 1: one cutpoint, no
        # monotonicity rows.  Matrix shape is (0, 1); D @ theta is empty.
        D = MonotonicityConstraint(1).as_matrix()
        assert D.shape == (0, 1)
        result = MonotonicityConstraint(1).as_scipy_constraint()["fun"](np.array([1.0]))
        assert result.shape == (0,)

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
        np.testing.assert_array_equal(
            lc.A.toarray() if hasattr(lc.A, "toarray") else lc.A, mc.as_matrix()
        )


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

    def test_total_params_pads_A_shape(self):
        bc = BoundaryConstraint(4, lower=0.0, upper=1.0, total_params=6)
        assert bc._A.shape == (2, 6)

    def test_total_params_upper_selects_correct_index(self):
        # upper must pin theta[n_params-1]=theta[3], not theta[-1]=theta[5]
        bc = BoundaryConstraint(4, lower=None, upper=1.0, total_params=6)
        cs = bc.as_scipy_constraint()
        theta = np.array([0.0, 0.5, 0.8, 1.0, 99.0, 99.0])
        np.testing.assert_allclose(cs[0]["fun"](theta), 0.0)

    def test_linear_constraint_padded_shape(self):
        bc = BoundaryConstraint(4, lower=0.0, upper=1.0, total_params=6)
        lc = bc.as_LinearConstraint()
        A = lc.A.toarray() if hasattr(lc.A, "toarray") else lc.A
        assert A.shape == (2, 6)
        np.testing.assert_array_equal(A[:, 4:], 0.0)


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
            obj,
            x0=np.zeros(4),
            jac=grad,
            method="trust-constr",
            constraints=constraints,
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
            obj,
            x0=np.array([0.0, 0.3, 0.6, 1.0]),
            jac=grad,
            method="SLSQP",
            constraints=constraints,
        )
        assert result.success, result.message
        assert np.all(np.diff(result.x) >= -1e-6)
        np.testing.assert_allclose(result.x[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.x[-1], 1.0, atol=1e-6)

    @pytest.mark.parametrize("solver", ["slsqp", "trust-constr"])
    def test_boundary_with_total_params_correct_shape(self, solver):
        n_params, total = 4, 6
        result = build_constraints(
            n_params, lower=0.0, upper=1.0, solver=solver, total_params=total
        )
        if solver == "slsqp":
            for c in result[1:]:  # skip monotonicity
                j = c["jac"](np.zeros(total))
                assert j.shape == (total,)
        else:
            bc_lc = result[1]
            A = bc_lc.A.toarray() if hasattr(bc_lc.A, "toarray") else bc_lc.A
            assert A.shape == (2, total)

    def test_minimize_slsqp_boundary_with_betas(self):
        """Boundary constraints work end-to-end when beta parameters are present."""
        n_params, n_beta = 4, 2
        total = n_params + n_beta
        constraints = build_constraints(
            n_params, lower=0.0, upper=1.0, solver="slsqp", total_params=total
        )
        target = np.concatenate([np.array([4.0, 3.0, 2.0, 1.0]), np.zeros(n_beta)])

        def obj(t):
            return np.sum((t - target) ** 2)

        def grad(t):
            return 2 * (t - target)

        result = minimize(
            obj,
            x0=np.array([0.0, 0.3, 0.6, 1.0, 0.0, 0.0]),
            jac=grad,
            method="SLSQP",
            constraints=constraints,
        )
        assert result.success, result.message
        np.testing.assert_allclose(result.x[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.x[n_params - 1], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# nonneg_lower support for exponential base distribution
# ---------------------------------------------------------------------------


class TestNonnegLower:
    def test_no_covariates_single_row(self):
        """Without X, nonneg_lower adds one inequality theta[0] >= 0."""
        result = build_constraints(4, solver="slsqp", nonneg_lower=True)
        # monotonicity + single nonneg row
        assert len(result) == 2
        theta_feasible = np.array([0.0, 1.0, 2.0, 3.0])
        theta_infeasible = np.array([-0.1, 1.0, 2.0, 3.0])
        # Second constraint row is the support row
        support = result[1]
        assert np.all(support["fun"](theta_feasible) >= 0)
        assert np.any(support["fun"](theta_infeasible) < 0)

    def test_covariates_one_row_per_observation(self):
        """With X, nonneg_lower adds one row per observation.

        Row i encodes theta_b[0] + X_i @ beta >= 0.
        """
        n_params = 4
        X = np.array(
            [
                [1.0, -0.5],
                [-2.0, 0.3],
                [0.5, 1.0],
            ]
        )
        total = n_params + X.shape[1]
        result = build_constraints(
            n_params,
            solver="slsqp",
            total_params=total,
            nonneg_lower=True,
            X=X,
        )
        assert len(result) == 2  # monotonicity + support
        support = result[1]

        # theta = [theta_b | beta]
        theta_b = np.array([0.5, 1.0, 2.0, 3.0])  # theta_b[0] = 0.5
        beta = np.array([0.0, 0.0])
        theta = np.concatenate([theta_b, beta])
        # With beta = 0, each row evaluates to theta_b[0] = 0.5 >= 0
        np.testing.assert_allclose(support["fun"](theta), [0.5, 0.5, 0.5])

        # Now beta that makes row 1 infeasible:
        # theta_b[0] + X_1 @ beta = 0.5 + (-2)*1 + 0.3*0 = -1.5 < 0
        beta = np.array([1.0, 0.0])
        theta = np.concatenate([theta_b, beta])
        vals = support["fun"](theta)
        assert vals[0] == pytest.approx(0.5 + 1.0 * 1.0 + (-0.5) * 0.0)
        assert vals[1] == pytest.approx(0.5 + (-2.0) * 1.0 + 0.3 * 0.0)
        assert vals[1] < 0
        assert vals[2] == pytest.approx(0.5 + 0.5 * 1.0 + 1.0 * 0.0)

    def test_covariates_jacobian_matches_support_matrix(self):
        n_params = 3
        X = np.array([[0.7, -1.2], [0.1, 2.0]])
        total = n_params + X.shape[1]
        result = build_constraints(
            n_params,
            solver="slsqp",
            total_params=total,
            nonneg_lower=True,
            X=X,
        )
        support = result[1]
        theta = np.zeros(total)
        jac = support["jac"](theta)
        # Expected: [1, 0, 0 | X_i] per row
        expected = np.zeros((X.shape[0], total))
        expected[:, 0] = 1.0
        expected[:, n_params:] = X
        np.testing.assert_array_equal(jac, expected)

    def test_trust_constr_covariates(self):
        """trust-constr path emits a LinearConstraint with n_obs rows."""
        n_params = 3
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
        total = n_params + X.shape[1]
        result = build_constraints(
            n_params,
            solver="trust-constr",
            total_params=total,
            nonneg_lower=True,
            X=X,
        )
        # monotonicity + support
        assert len(result) == 2
        support = result[1]
        assert isinstance(support, LinearConstraint)
        assert support.A.shape == (X.shape[0], total)
        np.testing.assert_array_equal(support.A[:, 0], 1.0)
        np.testing.assert_array_equal(support.A[:, n_params:], X)

    def test_wrong_X_shape_raises(self):
        with pytest.raises(ValueError, match="X must be 2-D"):
            build_constraints(
                3,
                solver="slsqp",
                total_params=5,
                nonneg_lower=True,
                X=np.array([1.0, 2.0, 3.0]),
            )

    @pytest.mark.parametrize("solver", ["slsqp", "trust-constr"])
    def test_missing_total_params_with_covariates_raises(self, solver):
        with pytest.raises(ValueError, match="total_params must be provided"):
            build_constraints(
                3,
                solver=solver,
                nonneg_lower=True,
                X=np.zeros((4, 2)),
            )

    def test_wrong_X_columns_raises(self):
        with pytest.raises(ValueError, match="columns"):
            build_constraints(
                3,
                solver="slsqp",
                total_params=5,
                nonneg_lower=True,
                X=np.zeros((4, 3)),  # 3 cols vs expected 5-3=2
            )


# ---------------------------------------------------------------------------
# build_constraint_matrices — boundary equality (auglag path)
# ---------------------------------------------------------------------------


class TestBuildConstraintMatricesBoundary:
    """Parallel to :class:`TestBoundaryConstraint` for the auglag entry point.

    Each case checks that the equality rows in ``C_eq``/``d_eq`` pin the
    correct index of ``theta`` to the requested value — mirroring the
    behaviour of :class:`BoundaryConstraint` but expressed as the dense
    matrices the augmented Lagrangian solver consumes directly.
    """

    def test_no_boundary_returns_zero_row_c_eq(self):
        cm = build_constraint_matrices(4)
        assert cm.C_eq.shape == (0, 4)
        assert cm.d_eq.shape == (0,)

    def test_lower_only_one_row(self):
        cm = build_constraint_matrices(4, lower=0.0)
        assert cm.C_eq.shape == (1, 4)
        # Row picks out theta[0]; equality residual C @ theta - d at the
        # boundary value must be zero.
        theta = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(cm.C_eq @ theta - cm.d_eq, 0.0)
        np.testing.assert_array_equal(cm.C_eq[0], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(cm.d_eq, [0.0])

    def test_upper_only_one_row(self):
        cm = build_constraint_matrices(4, upper=5.0)
        assert cm.C_eq.shape == (1, 4)
        # Row picks out theta[n_params-1].
        theta = np.array([1.0, 2.0, 3.0, 5.0])
        np.testing.assert_allclose(cm.C_eq @ theta - cm.d_eq, 0.0)
        np.testing.assert_array_equal(cm.C_eq[0], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(cm.d_eq, [5.0])

    def test_both_bounds_two_rows(self):
        cm = build_constraint_matrices(4, lower=1.0, upper=4.0)
        assert cm.C_eq.shape == (2, 4)
        # First row pins theta[0]=1.0; second pins theta[3]=4.0.
        theta = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(cm.C_eq @ theta - cm.d_eq, 0.0)
        np.testing.assert_array_equal(cm.d_eq, [1.0, 4.0])

    def test_lower_residual_non_zero(self):
        cm = build_constraint_matrices(4, lower=1.0)
        theta = np.array([2.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(cm.C_eq @ theta - cm.d_eq, [1.0])  # 2 - 1

    def test_upper_residual_non_zero(self):
        cm = build_constraint_matrices(4, upper=3.0)
        theta = np.array([1.0, 2.0, 3.0, 5.0])
        np.testing.assert_allclose(cm.C_eq @ theta - cm.d_eq, [2.0])  # 5 - 3

    def test_total_params_pads_c_eq(self):
        # With beta entries, equality rows must be padded with zero columns
        # so they apply to the Bernstein block only.  upper still points at
        # theta[n_params-1], not the last column of the padded vector.
        cm = build_constraint_matrices(4, lower=0.0, upper=1.0, total_params=6)
        assert cm.C_eq.shape == (2, 6)
        np.testing.assert_array_equal(cm.C_eq[:, 4:], 0.0)
        theta = np.array([0.0, 0.5, 0.8, 1.0, 99.0, 99.0])
        np.testing.assert_allclose(cm.C_eq @ theta - cm.d_eq, 0.0)

    def test_total_params_preserves_monotonicity_padding(self):
        cm = build_constraint_matrices(4, lower=0.0, total_params=6)
        assert cm.A_ineq.shape == (3, 6)
        np.testing.assert_array_equal(cm.A_ineq[:, 4:], 0.0)

    @pytest.mark.parametrize(
        "lower,upper,expected_rows",
        [(None, None, 0), (0.0, None, 1), (None, 1.0, 1), (0.0, 1.0, 2)],
    )
    def test_row_count(self, lower, upper, expected_rows):
        cm = build_constraint_matrices(5, lower=lower, upper=upper)
        assert cm.C_eq.shape[0] == expected_rows
        assert cm.d_eq.shape == (expected_rows,)


# ---------------------------------------------------------------------------
# build_constraint_matrices — exponential support inequality (auglag path)
# ---------------------------------------------------------------------------


class TestBuildConstraintMatricesNonnegLower:
    """Parallel to :class:`TestNonnegLower` for the auglag entry point.

    Verifies that the per-observation support-feasibility rows appended to
    ``A_ineq`` match the ones :func:`build_constraints` emits as SLSQP
    dictionaries — same row layout, ``b_ineq = 0``.
    """

    def test_no_covariates_single_support_row(self):
        """Without X, nonneg_lower adds one row theta[0] >= 0."""
        cm = build_constraint_matrices(4, nonneg_lower=True)
        # Monotonicity has n_params - 1 = 3 rows; plus one support row.
        assert cm.A_ineq.shape == (3 + 1, 4)
        np.testing.assert_array_equal(cm.A_ineq[-1], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(cm.b_ineq, np.zeros(4))

    def test_no_covariates_feasibility_evaluation(self):
        cm = build_constraint_matrices(4, nonneg_lower=True)
        theta_feasible = np.array([0.0, 1.0, 2.0, 3.0])
        theta_infeasible = np.array([-0.1, 1.0, 2.0, 3.0])
        # Last row of A_ineq enforces the support condition.
        assert (cm.A_ineq @ theta_feasible - cm.b_ineq)[-1] >= 0
        assert (cm.A_ineq @ theta_infeasible - cm.b_ineq)[-1] < 0

    def test_covariates_one_row_per_observation(self):
        """With X, one row per observation: theta_b[0] + X_i @ beta >= 0."""
        n_params = 4
        X = np.array([[1.0, -0.5], [-2.0, 0.3], [0.5, 1.0]])
        total = n_params + X.shape[1]
        cm = build_constraint_matrices(
            n_params, total_params=total, nonneg_lower=True, X=X
        )
        n_mono = n_params - 1  # monotonicity rows
        # 3 monotonicity rows + 3 support rows
        assert cm.A_ineq.shape == (n_mono + X.shape[0], total)
        # Support rows: first column is 1, last q cols equal X
        support = cm.A_ineq[n_mono:]
        np.testing.assert_array_equal(support[:, 0], 1.0)
        np.testing.assert_array_equal(support[:, 1:n_params], 0.0)
        np.testing.assert_array_equal(support[:, n_params:], X)
        np.testing.assert_array_equal(cm.b_ineq, np.zeros(cm.A_ineq.shape[0]))

    def test_covariates_feasibility_evaluation(self):
        """Support row i is feasible iff theta_b[0] + X_i @ beta >= 0."""
        n_params = 4
        X = np.array([[1.0, -0.5], [-2.0, 0.3], [0.5, 1.0]])
        total = n_params + X.shape[1]
        cm = build_constraint_matrices(
            n_params, total_params=total, nonneg_lower=True, X=X
        )
        n_mono = n_params - 1
        # theta_b[0] = 0.5, beta = 0  →  every row evaluates to 0.5.
        theta = np.concatenate([np.array([0.5, 1.0, 2.0, 3.0]), np.zeros(2)])
        support_vals = (cm.A_ineq @ theta - cm.b_ineq)[n_mono:]
        np.testing.assert_allclose(support_vals, [0.5, 0.5, 0.5])

        # beta = (1.0, 0.0): row 1 = 0.5 + (-2)*1 = -1.5 < 0 (infeasible).
        theta = np.concatenate([np.array([0.5, 1.0, 2.0, 3.0]), [1.0, 0.0]])
        support_vals = (cm.A_ineq @ theta - cm.b_ineq)[n_mono:]
        assert support_vals[1] < 0

    def test_covariates_monotonicity_preserved(self):
        """nonneg_lower must not disturb the leading monotonicity block."""
        n_params = 3
        X = np.array([[0.7, -1.2], [0.1, 2.0], [-0.4, 0.5], [1.5, 0.3]])
        total = n_params + X.shape[1]
        cm = build_constraint_matrices(
            n_params, total_params=total, nonneg_lower=True, X=X
        )
        # First (n_params - 1) rows = forward-difference monotonicity (padded).
        D = np.diff(np.eye(n_params), axis=0)
        np.testing.assert_array_equal(cm.A_ineq[: n_params - 1, :n_params], D)
        # Beta columns of the monotonicity block must be zero.
        np.testing.assert_array_equal(cm.A_ineq[: n_params - 1, n_params:], 0.0)

    def test_b_ineq_always_zero(self):
        n_params = 3
        X = np.array([[0.7, -1.2], [0.1, 2.0]])
        total = n_params + X.shape[1]
        cm = build_constraint_matrices(
            n_params, total_params=total, nonneg_lower=True, X=X
        )
        np.testing.assert_array_equal(cm.b_ineq, 0.0)

    def test_wrong_X_shape_raises(self):
        with pytest.raises(ValueError, match="X must be 2-D"):
            build_constraint_matrices(
                3, total_params=5, nonneg_lower=True, X=np.array([1.0, 2.0, 3.0])
            )

    def test_missing_total_params_with_covariates_raises(self):
        with pytest.raises(ValueError, match="total_params must be provided"):
            build_constraint_matrices(3, nonneg_lower=True, X=np.zeros((4, 2)))

    def test_wrong_X_columns_raises(self):
        with pytest.raises(ValueError, match="columns"):
            build_constraint_matrices(
                3, total_params=5, nonneg_lower=True, X=np.zeros((4, 3))
            )

    def test_no_X_with_extra_total_params_raises(self):
        """nonneg_lower=True, X=None, but total_params has room for beta — ambiguous."""
        with pytest.raises(ValueError, match="X must be provided"):
            build_constraint_matrices(3, total_params=5, nonneg_lower=True)

    def test_matches_slsqp_layout_no_covariates(self):
        """Same row layout as build_constraints(..., solver='slsqp')."""
        cm = build_constraint_matrices(4, nonneg_lower=True)
        slsqp = build_constraints(4, solver="slsqp", nonneg_lower=True)
        # build_constraints emits monotonicity dict (n-1 rows) + support dict (1 row).
        np.testing.assert_array_equal(cm.A_ineq[:3], slsqp[0]["jac"](np.zeros(4)))
        np.testing.assert_array_equal(cm.A_ineq[3:], slsqp[1]["jac"](np.zeros(4)))

    def test_matches_slsqp_layout_with_covariates(self):
        """Same per-observation rows as build_constraints with covariates."""
        n_params = 3
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        total = n_params + X.shape[1]
        cm = build_constraint_matrices(
            n_params, total_params=total, nonneg_lower=True, X=X
        )
        slsqp = build_constraints(
            n_params, solver="slsqp", total_params=total, nonneg_lower=True, X=X
        )
        # Support block (rows from monotonicity onwards) must match.
        np.testing.assert_array_equal(
            cm.A_ineq[n_params - 1 :], slsqp[1]["jac"](np.zeros(total))
        )

    def test_combines_with_boundary_equality(self):
        """nonneg_lower (A_ineq rows) and lower= (C_eq row) can coexist."""
        n_params = 4
        X = np.array([[0.5], [-0.3]])
        total = n_params + X.shape[1]
        cm = build_constraint_matrices(
            n_params, lower=0.0, total_params=total, nonneg_lower=True, X=X
        )
        # Inequalities: 3 monotonicity + 2 support.
        assert cm.A_ineq.shape == (3 + 2, total)
        # Equality: one row pinning theta[0]=0.
        assert cm.C_eq.shape == (1, total)
        np.testing.assert_array_equal(cm.C_eq[0], [1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(cm.d_eq, [0.0])


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
