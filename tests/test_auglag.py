"""Tests for pymlt._auglag: AugLagOptions, AugLagResult, auglag_minimize."""

from __future__ import annotations

import numpy as np
import pytest

from pymlt._auglag import AugLagOptions, auglag_minimize
from pymlt.constraints import ConstraintMatrices, build_constraint_matrices

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def quadratic_obj(
    center: np.ndarray,
) -> callable:
    """Return (f, grad) for f(x) = 0.5 * ||x - center||^2."""

    def obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
        diff = theta - center
        return 0.5 * float(diff @ diff), diff.copy()

    return obj


# ---------------------------------------------------------------------------
# AugLagOptions defaults
# ---------------------------------------------------------------------------


class TestAugLagOptionsDefaults:
    def test_defaults(self) -> None:
        opts = AugLagOptions()
        assert opts.rho_init == 10.0
        assert opts.rho_growth == 10.0
        assert opts.violation_shrink_factor == 0.25
        assert opts.outer_tol == 1e-7
        assert opts.max_outer_iter == 50
        assert opts.inner_method == "L-BFGS-B"
        assert opts.inner_options["gtol"] == 1e-8
        assert opts.inner_options["ftol"] == 1e-15
        assert opts.inner_options["maxiter"] == 500
        assert opts.rho_max == 1e8

    def test_custom(self) -> None:
        opts = AugLagOptions(rho_init=1.0, outer_tol=1e-5, max_outer_iter=20)
        assert opts.rho_init == 1.0
        assert opts.outer_tol == 1e-5
        assert opts.max_outer_iter == 20


# ---------------------------------------------------------------------------
# AugLagResult fields
# ---------------------------------------------------------------------------


class TestAugLagResultFields:
    def test_all_fields_present(self) -> None:
        obj = quadratic_obj(np.array([1.0, 2.0]))
        result = auglag_minimize(
            obj, np.zeros(2), A_ineq=None, b_ineq=None, C_eq=None, d_eq=None
        )
        assert hasattr(result, "theta")
        assert hasattr(result, "fun")
        assert hasattr(result, "n_outer_iter")
        assert hasattr(result, "n_inner_iter")
        assert hasattr(result, "converged")
        assert hasattr(result, "kkt_residual")
        assert hasattr(result, "message")
        assert hasattr(result, "lambda_eq")
        assert hasattr(result, "mu_ineq")
        assert hasattr(result, "rho_final")

    def test_types(self) -> None:
        obj = quadratic_obj(np.array([0.5]))
        result = auglag_minimize(
            obj, np.zeros(1), A_ineq=None, b_ineq=None, C_eq=None, d_eq=None
        )
        assert isinstance(result.theta, np.ndarray)
        assert isinstance(result.fun, float)
        assert isinstance(result.n_outer_iter, int)
        assert isinstance(result.n_inner_iter, int)
        assert isinstance(result.converged, bool)
        assert isinstance(result.kkt_residual, float)
        assert isinstance(result.message, str)
        assert isinstance(result.lambda_eq, np.ndarray)
        assert isinstance(result.mu_ineq, np.ndarray)
        assert isinstance(result.rho_final, float)


# ---------------------------------------------------------------------------
# Unconstrained quadratic: auglag must reduce to inner solver result
# ---------------------------------------------------------------------------


class TestUnconstrained:
    @pytest.mark.parametrize(
        "center",
        [
            np.array([1.0, 2.0]),
            np.array([-3.0, 0.5, 4.0]),
            np.array([0.0]),
        ],
    )
    def test_unconstrained_reaches_minimum(self, center: np.ndarray) -> None:
        """No constraints → auglag delegates purely to inner solver."""
        obj = quadratic_obj(center)
        x0 = np.zeros_like(center)
        result = auglag_minimize(
            obj, x0, A_ineq=None, b_ineq=None, C_eq=None, d_eq=None
        )
        np.testing.assert_allclose(result.theta, center, atol=1e-6)
        assert result.converged

    def test_unconstrained_zero_multipliers(self) -> None:
        """No constraints → multipliers remain zero."""
        obj = quadratic_obj(np.array([3.0, -1.0]))
        result = auglag_minimize(
            obj, np.zeros(2), A_ineq=None, b_ineq=None, C_eq=None, d_eq=None
        )
        assert result.lambda_eq.shape == (0,)
        assert result.mu_ineq.shape == (0,)

    def test_fun_equals_objective_at_optimum(self) -> None:
        """result.fun must equal f(theta), not the augmented Lagrangian."""
        center = np.array([2.0, -1.0])
        obj = quadratic_obj(center)
        result = auglag_minimize(
            obj, np.zeros(2), A_ineq=None, b_ineq=None, C_eq=None, d_eq=None
        )
        f_direct, _ = obj(result.theta)
        assert abs(result.fun - f_direct) < 1e-10


# ---------------------------------------------------------------------------
# Inequality-constrained QP: known active-set
# ---------------------------------------------------------------------------


class TestInequalityQP:
    """Minimise 0.5*||x - [3, 1]||^2  s.t.  x[1] >= x[0].

    Unconstrained opt at (3, 1) violates x[1] >= x[0].
    Constrained opt is x* = (2, 2) with constraint active.
    KKT multiplier μ = 2 (positive).
    Complementary slackness: μ * (x[1]-x[0]) = 2 * 0 = 0.
    """

    @pytest.fixture
    def setup(self) -> tuple:
        center = np.array([3.0, 1.0])
        obj = quadratic_obj(center)
        # Constraint: x[1] - x[0] >= 0  →  A = [-1, 1], b = 0
        A = np.array([[-1.0, 1.0]])
        b = np.array([0.0])
        x0 = np.array([0.0, 0.0])
        return obj, A, b, x0

    def test_solution(self, setup: tuple) -> None:
        obj, A, b, x0 = setup
        result = auglag_minimize(obj, x0, A_ineq=A, b_ineq=b, C_eq=None, d_eq=None)
        np.testing.assert_allclose(result.theta, [2.0, 2.0], atol=1e-5)

    def test_converged(self, setup: tuple) -> None:
        obj, A, b, x0 = setup
        result = auglag_minimize(obj, x0, A_ineq=A, b_ineq=b, C_eq=None, d_eq=None)
        assert result.converged

    def test_kkt_mu_nonneg(self, setup: tuple) -> None:
        """Dual variables must be non-negative at the optimum."""
        obj, A, b, x0 = setup
        result = auglag_minimize(obj, x0, A_ineq=A, b_ineq=b, C_eq=None, d_eq=None)
        assert np.all(result.mu_ineq >= -1e-8)

    def test_complementary_slackness(self, setup: tuple) -> None:
        """μ_i * g_i(x*) ≈ 0 at the optimum."""
        obj, A, b, x0 = setup
        result = auglag_minimize(obj, x0, A_ineq=A, b_ineq=b, C_eq=None, d_eq=None)
        g = A @ result.theta - b
        cs = result.mu_ineq * g
        np.testing.assert_allclose(cs, 0.0, atol=1e-5)

    def test_constraint_satisfied(self, setup: tuple) -> None:
        """g(x*) >= 0 at the optimum."""
        obj, A, b, x0 = setup
        result = auglag_minimize(obj, x0, A_ineq=A, b_ineq=b, C_eq=None, d_eq=None)
        g = A @ result.theta - b
        assert np.all(g >= -1e-6)

    def test_inactive_constraint_no_multiplier(self) -> None:
        """When unconstrained opt is feasible, μ ≈ 0 (constraint inactive)."""
        center = np.array([1.0, 3.0])  # 3 > 1, so x[1]-x[0]>0 at unconstrained opt
        obj = quadratic_obj(center)
        A = np.array([[-1.0, 1.0]])
        b = np.array([0.0])
        result = auglag_minimize(
            obj, np.zeros(2), A_ineq=A, b_ineq=b, C_eq=None, d_eq=None
        )
        np.testing.assert_allclose(result.theta, center, atol=1e-5)
        assert result.mu_ineq[0] < 1e-5


# ---------------------------------------------------------------------------
# Equality-constrained QP: closed-form optimum
# ---------------------------------------------------------------------------


class TestEqualityQP:
    """Minimise 0.5 * ||x||^2  s.t.  sum(x) = 1.

    By symmetry: x* = [1/3, 1/3, 1/3],  λ* = 1/3.
    """

    @pytest.fixture
    def setup(self) -> tuple:
        obj = quadratic_obj(np.zeros(3))
        C = np.ones((1, 3))
        d = np.array([1.0])
        x0 = np.zeros(3)
        return obj, C, d, x0

    def test_solution(self, setup: tuple) -> None:
        obj, C, d, x0 = setup
        result = auglag_minimize(obj, x0, A_ineq=None, b_ineq=None, C_eq=C, d_eq=d)
        np.testing.assert_allclose(result.theta, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_converged(self, setup: tuple) -> None:
        obj, C, d, x0 = setup
        result = auglag_minimize(obj, x0, A_ineq=None, b_ineq=None, C_eq=C, d_eq=d)
        assert result.converged

    def test_equality_satisfied(self, setup: tuple) -> None:
        """C @ x* = d at the optimum."""
        obj, C, d, x0 = setup
        result = auglag_minimize(obj, x0, A_ineq=None, b_ineq=None, C_eq=C, d_eq=d)
        residual = C @ result.theta - d
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)

    def test_lambda_eq_correct_sign(self, setup: tuple) -> None:
        """Equality multiplier λ* = 1/3 (positive, pushing toward constraint)."""
        obj, C, d, x0 = setup
        result = auglag_minimize(obj, x0, A_ineq=None, b_ineq=None, C_eq=C, d_eq=d)
        # lambda_eq sign: our PHR sign convention is L_A = f - λᵀh + ...
        # KKT: ∇f = Cᵀλ  →  θ = λ·ones  →  λ = 1/3
        np.testing.assert_allclose(result.lambda_eq[0], 1 / 3, atol=1e-5)

    def test_zero_row_c_eq_treated_as_unconstrained(self) -> None:
        """Zero-row C_eq is treated as no equality constraints."""
        obj = quadratic_obj(np.array([2.0, -1.0]))
        C = np.zeros((0, 2))
        d = np.zeros(0)
        result = auglag_minimize(
            obj, np.zeros(2), A_ineq=None, b_ineq=None, C_eq=C, d_eq=d
        )
        np.testing.assert_allclose(result.theta, [2.0, -1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# None vs zero-row matrix: both accepted
# ---------------------------------------------------------------------------


class TestNoneVsZeroRowMatrix:
    @pytest.mark.parametrize(
        "a_ineq,b_ineq",
        [
            (None, None),
            (np.zeros((0, 2)), np.zeros(0)),
        ],
    )
    def test_none_and_zero_row_equivalent_ineq(
        self, a_ineq: object, b_ineq: object
    ) -> None:
        obj = quadratic_obj(np.array([1.0, 1.0]))
        r = auglag_minimize(
            obj, np.zeros(2), A_ineq=a_ineq, b_ineq=b_ineq, C_eq=None, d_eq=None
        )
        np.testing.assert_allclose(r.theta, [1.0, 1.0], atol=1e-5)

    @pytest.mark.parametrize(
        "c_eq,d_eq",
        [
            (None, None),
            (np.zeros((0, 2)), np.zeros(0)),
        ],
    )
    def test_none_and_zero_row_equivalent_eq(self, c_eq: object, d_eq: object) -> None:
        obj = quadratic_obj(np.array([2.0, 3.0]))
        r = auglag_minimize(
            obj, np.zeros(2), A_ineq=None, b_ineq=None, C_eq=c_eq, d_eq=d_eq
        )
        np.testing.assert_allclose(r.theta, [2.0, 3.0], atol=1e-5)


# ---------------------------------------------------------------------------
# build_constraint_matrices: monotonicity only (slice 1)
# ---------------------------------------------------------------------------


class TestBuildConstraintMatrices:
    def test_returns_constraint_matrices(self) -> None:
        cm = build_constraint_matrices(4)
        assert isinstance(cm, ConstraintMatrices)

    def test_a_ineq_shape(self) -> None:
        """A_ineq should be the forward-difference matrix (n_params-1, n_params)."""
        cm = build_constraint_matrices(4)
        assert cm.A_ineq.shape == (3, 4)

    def test_b_ineq_zeros(self) -> None:
        cm = build_constraint_matrices(4)
        np.testing.assert_array_equal(cm.b_ineq, np.zeros(3))

    def test_c_eq_zero_row(self) -> None:
        """No equality constraints this slice — C_eq has 0 rows."""
        cm = build_constraint_matrices(4)
        assert cm.C_eq.shape[0] == 0

    def test_d_eq_empty(self) -> None:
        cm = build_constraint_matrices(4)
        assert cm.d_eq.shape == (0,)

    def test_monotonicity_constraint_values(self) -> None:
        """A_ineq @ theta >= 0 for a non-decreasing theta."""
        cm = build_constraint_matrices(3)
        theta_mono = np.array([0.0, 0.5, 1.0])
        assert np.all(cm.A_ineq @ theta_mono >= 0)

    def test_monotonicity_constraint_violation(self) -> None:
        """A_ineq @ theta < 0 for a strictly decreasing theta."""
        cm = build_constraint_matrices(3)
        theta_bad = np.array([1.0, 0.5, 0.0])
        assert np.all(cm.A_ineq @ theta_bad <= 0)

    def test_beta_padding(self) -> None:
        """total_params > n_params: A_ineq padded with zero columns for beta."""
        n_params, n_beta = 3, 2
        total = n_params + n_beta
        cm = build_constraint_matrices(n_params, total_params=total)
        assert cm.A_ineq.shape == (n_params - 1, total)
        # Beta columns must be zero
        np.testing.assert_array_equal(cm.A_ineq[:, n_params:], 0.0)

    @pytest.mark.parametrize("n_params", [1, 2, 5, 10])
    def test_shape_parametrized(self, n_params: int) -> None:
        cm = build_constraint_matrices(n_params)
        assert cm.A_ineq.shape == (max(n_params - 1, 0), n_params)


# ---------------------------------------------------------------------------
# auglag_minimize via optimizer.py: auglag solver integrates correctly
# ---------------------------------------------------------------------------


class TestOptimizerAuglag:
    def test_solver_auglag_converges(self) -> None:
        """optimizer.optimize() with solver='auglag' converges on simple data."""
        from pymlt.basis import BernsteinBasis
        from pymlt.optimizer import OptimizationResult, OptimizerConfig, optimize

        rng = np.random.default_rng(0)
        y = rng.uniform(0.05, 0.95, 60)
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        config = OptimizerConfig(solver="auglag")
        result = optimize(basis, y, config=config)

        assert isinstance(result, OptimizationResult)
        assert result.converged
        assert np.isfinite(result.log_likelihood)
        assert result.n_outer_iter is not None
        assert result.kkt_residual is not None

    def test_auglag_theta_non_decreasing(self) -> None:
        """Monotonicity constraint must be satisfied at the auglag optimum."""
        from pymlt.basis import BernsteinBasis
        from pymlt.optimizer import OptimizerConfig, optimize

        rng = np.random.default_rng(1)
        y = rng.uniform(0.05, 0.95, 80)
        basis = BernsteinBasis(order=4, support=(0.0, 1.0))
        result = optimize(basis, y, config=OptimizerConfig(solver="auglag"))
        n_params = basis.order + 1
        theta_b = result.theta[:n_params]
        assert np.all(np.diff(theta_b) >= -1e-6)

    def test_auglag_slsqp_agree(self) -> None:
        """Auglag and SLSQP find the same log-likelihood on the same data."""
        from pymlt.basis import BernsteinBasis
        from pymlt.optimizer import OptimizerConfig, optimize

        rng = np.random.default_rng(7)
        y = rng.uniform(0.05, 0.95, 60)
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        r_slsqp = optimize(basis, y, config=OptimizerConfig(solver="slsqp"))
        r_auglag = optimize(basis, y, config=OptimizerConfig(solver="auglag"))
        assert abs(r_auglag.log_likelihood - r_slsqp.log_likelihood) < 1e-4

    def test_auglag_result_auglag_only_fields(self) -> None:
        """n_outer_iter and kkt_residual are None for SLSQP, set for auglag."""
        from pymlt.basis import BernsteinBasis
        from pymlt.optimizer import OptimizerConfig, optimize

        rng = np.random.default_rng(2)
        y = rng.uniform(0.05, 0.95, 40)
        basis = BernsteinBasis(order=2, support=(0.0, 1.0))

        r_slsqp = optimize(basis, y, config=OptimizerConfig(solver="slsqp"))
        assert r_slsqp.n_outer_iter is None
        assert r_slsqp.kkt_residual is None

        r_auglag = optimize(basis, y, config=OptimizerConfig(solver="auglag"))
        assert r_auglag.n_outer_iter is not None
        assert r_auglag.kkt_residual is not None
        assert r_auglag.kkt_residual >= 0.0

    def test_auglag_default_options(self) -> None:
        """None auglag_options falls back to AugLagOptions defaults."""
        from pymlt.basis import BernsteinBasis
        from pymlt.optimizer import OptimizerConfig, optimize

        rng = np.random.default_rng(3)
        y = rng.uniform(0.05, 0.95, 40)
        basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        config = OptimizerConfig(solver="auglag", auglag_options=None)
        result = optimize(basis, y, config=config)
        assert result.converged
