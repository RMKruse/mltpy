"""Tests for mltpy._auglag: AugLagOptions, AugLagResult, auglag_minimize."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from mltpy._auglag import AugLagOptions, _projected_stationarity, auglag_minimize
from mltpy.constraints import ConstraintMatrices, build_constraint_matrices

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
# Stall-based convergence: projected stationarity (issue #134)
# ---------------------------------------------------------------------------


def _pinned_obj(
    x0: NDArray[np.float64],
    f0: float = 0.0,
    grad: NDArray[np.float64] | float = 1.0,
) -> Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]:
    """Objective the inner solver cannot leave: f = ``f0`` with ∇f = ``grad``
    (a vector, or a scalar broadcast to every component) at ``x0`` and
    ``(inf, 0)`` everywhere else.

    Mimics the issue-#134 failure mode — every L-BFGS-B trial step lands on a
    non-finite value, so θ never moves.  Whether the resulting stall is a
    constrained optimum or a failed solve is then decided purely by ∇f and the
    constraint geometry at ``x0``.
    """
    g0 = np.broadcast_to(np.asarray(grad, dtype=np.float64), x0.shape).copy()

    def obj(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        if np.array_equal(theta, x0):
            return f0, g0.copy()
        return float("inf"), np.zeros_like(theta)

    return obj


_X0 = np.array([0.5, 0.5])
_E1 = np.array([[1.0, 0.0]])
_E2 = np.array([[0.0, 1.0]])
_ONES = np.array([[1.0, 1.0]])


class TestStallConvergenceProjectedStationarity:
    """A feasible iterate whose θ stopped moving is accepted as converged only
    when ∇f is explained, direction by direction, by the constraints that are
    active there: equality normals with any multiplier sign, active inequality
    normals with non-negative multipliers.  Whatever is left over is a
    feasible descent direction, so a stall that leaves a residual above the
    stationarity floor is a failed inner solve, never an active-set floor."""

    @pytest.mark.parametrize(
        "grad,A,b,C,d",
        [
            # λ = 1 on the equality 1ᵀθ = 1 explains ∇f = (1, 1).
            ((1.0, 1.0), None, None, _ONES, np.array([1.0])),
            # μ = 1 on θ₁ ≥ 0.5 (at its boundary) explains ∇f = (1, 0).
            ((1.0, 0.0), _E1, np.array([0.5]), None, None),
            # Degenerate stack: three rows active in two dimensions; ∇f lies
            # in the cone of their normals.
            (
                (1.0, 1.0),
                np.vstack([_E1, _E2, _ONES]),
                np.array([0.5, 0.5, 1.0]),
                None,
                None,
            ),
            # Equality multiplier of either sign (λ = −3) plus μ = 2 on θ₂.
            ((-3.0, 2.0), _E2, np.array([0.5]), _E1, np.array([0.5])),
            # Inequality violated by 5e-8 < feas_tol, so μ > 0 after the first
            # multiplier update; μ = 1 explains ∇f = (1, 0).
            ((1.0, 0.0), _E1, np.array([0.5 + 5e-8]), None, None),
        ],
        ids=[
            "equality-normal",
            "inequality-at-boundary",
            "stacked-inequalities",
            "equality-and-inequality",
            "inequality-positive-multiplier",
        ],
    )
    def test_stall_at_constrained_optimum_converges(
        self,
        grad: tuple[float, float],
        A: NDArray[np.float64] | None,
        b: NDArray[np.float64] | None,
        C: NDArray[np.float64] | None,
        d: NDArray[np.float64] | None,
    ) -> None:
        """The alabama-parity stall exit is kept at genuine KKT points, even
        though the solver's running multipliers (still zero or tiny) leave the
        augmented-Lagrangian stationarity far above ``outer_tol``."""
        result = auglag_minimize(
            _pinned_obj(_X0, grad=np.array(grad)),
            _X0,
            A_ineq=A,
            b_ineq=b,
            C_eq=C,
            d_eq=d,
        )
        assert result.converged
        assert "projected stationarity" in result.message
        np.testing.assert_array_equal(result.theta, _X0)

    @pytest.mark.parametrize(
        "grad,A,b,C,d",
        [
            # No constraints at all.
            ((1.0, 1.0), None, None, None, None),
            # Slack inequality: g = 0.5 > 0 at x0, μ stays 0.
            ((1.0, 1.0), _E1, np.array([0.0]), None, None),
            # An equality on θ₁ cannot explain ∇f in θ₂ (review of #134).
            ((1.0, 1.0), None, None, _E1, np.array([0.5])),
            # Positive multiplier on θ₁ ≥ 0.5, but ∇f also has a θ₂ component.
            ((1.0, 1.0), _E1, np.array([0.5 + 5e-8]), None, None),
            # Inequality at its boundary with ∇f pointing out of the feasible
            # side: +e₁ is a feasible descent direction (would need μ = −1).
            ((-1.0, 0.0), _E1, np.array([0.5]), None, None),
            # ∇f orthogonal to the equality normal: descent along 1ᵀθ = 1.
            ((1.0, -1.0), None, None, _ONES, np.array([1.0])),
            # ∇f outside the cone spanned by two active inequality normals.
            (
                (1.0, -1.0),
                np.vstack([_E1, _ONES]),
                np.array([0.5, 1.0]),
                None,
                None,
            ),
        ],
        ids=[
            "unconstrained",
            "slack-inequality",
            "equality-other-coordinate",
            "positive-multiplier-other-coordinate",
            "inequality-wrong-sign",
            "descent-along-equality",
            "outside-inequality-cone",
        ],
    )
    def test_stall_with_feasible_descent_not_converged(
        self,
        grad: tuple[float, float],
        A: NDArray[np.float64] | None,
        b: NDArray[np.float64] | None,
        C: NDArray[np.float64] | None,
        d: NDArray[np.float64] | None,
    ) -> None:
        options = AugLagOptions()
        result = auglag_minimize(
            _pinned_obj(_X0, grad=np.array(grad)),
            _X0,
            A_ineq=A,
            b_ineq=b,
            C_eq=C,
            d_eq=d,
            options=options,
        )
        assert not result.converged
        np.testing.assert_array_equal(result.theta, _X0)
        assert "stall" in result.message.lower()
        # Gives up as soon as the stall is detected rather than burning the
        # remaining outer budget on an immobile θ.
        assert result.n_outer_iter < options.max_outer_iter

    @pytest.mark.parametrize(
        "f0,slope,expect_converged",
        [
            # Bound θ_tol^(2/3)·(1+|f|) ≈ 4.6e-6 · 1001 ≈ 4.6e-3.
            (1000.0, 1e-6, True),  # numerical floor of a large-|f| fit
            (1000.0, 1e-1, False),  # far above the floor
            # |f| ≈ 0: bound ≈ 4.6e-6.
            (0.0, 1e-6, True),
            (0.0, 1e-4, False),
        ],
    )
    def test_stall_stationarity_bound(
        self, f0: float, slope: float, expect_converged: bool
    ) -> None:
        """An unexplained gradient is tolerated only up to the
        Gill–Murray–Wright U3 floor ‖r‖∞ ≤ θ_tol^(2/3)·(1 + |f|) — the
        gradient floor attainable when f itself is only resolved to relative
        precision θ_tol²."""
        result = auglag_minimize(
            _pinned_obj(_X0, f0=f0, grad=slope),
            _X0,
            A_ineq=_E1,
            b_ineq=np.array([0.0]),
            C_eq=None,
            d_eq=None,
        )
        assert result.converged is expect_converged


class TestStallOnSmoothQuadratic:
    """End-to-end form of the #134 review counter-example: a smooth quadratic
    f = ½‖θ − (0, 1)‖² with θ₁ fixed at 1 by an equality, started at (1, 2).
    The constrained optimum is (1, 1).  Starving L-BFGS-B's line search
    (``maxls=1``) makes the inner solve stall at the start point; that stall
    previously came back ``converged=True`` with KKT residual 1.0 because an
    equality row existed at all."""

    _C = np.array([[1.0, 0.0]])
    _D = np.array([1.0])
    _X_START = np.array([1.0, 2.0])
    _X_OPT = np.array([1.0, 1.0])

    @pytest.mark.parametrize(
        "inner_options",
        [
            {"maxls": 1},
            AugLagOptions().inner_options,
        ],
        ids=["starved-line-search", "default"],
    )
    def test_converged_implies_constrained_optimum(
        self, inner_options: dict[str, object]
    ) -> None:
        result = auglag_minimize(
            quadratic_obj(np.array([0.0, 1.0])),
            self._X_START,
            A_ineq=None,
            b_ineq=None,
            C_eq=self._C,
            d_eq=self._D,
            options=AugLagOptions(inner_options=inner_options),
        )
        if result.converged:
            np.testing.assert_allclose(result.theta, self._X_OPT, atol=1e-6)
        else:
            assert "stall" in result.message.lower()

    def test_starved_line_search_reports_failure(self) -> None:
        """Pin the failure path: with ``maxls=1`` θ₂ never leaves 2, and the
        solver must say so instead of declaring the stall a KKT floor."""
        result = auglag_minimize(
            quadratic_obj(np.array([0.0, 1.0])),
            self._X_START,
            A_ineq=None,
            b_ineq=None,
            C_eq=self._C,
            d_eq=self._D,
            options=AugLagOptions(inner_options={"maxls": 1}),
        )
        if np.allclose(result.theta, self._X_OPT, atol=1e-6):
            pytest.skip("L-BFGS-B reached the optimum despite maxls=1")
        assert not result.converged
        assert result.kkt_residual > 0.5

    def test_default_options_reach_optimum(self) -> None:
        result = auglag_minimize(
            quadratic_obj(np.array([0.0, 1.0])),
            self._X_START,
            A_ineq=None,
            b_ineq=None,
            C_eq=self._C,
            d_eq=self._D,
        )
        assert result.converged
        np.testing.assert_allclose(result.theta, self._X_OPT, atol=1e-6)


class TestProjectedStationarity:
    """Unit tests for ``_projected_stationarity``: the residual of the best
    representation ∇f = Cᵀλ + A_𝒜ᵀμ (λ free, μ ≥ 0) and the duality gap
    μ*ᵀmax(g, 0) of the working-set rows it uses."""

    @pytest.mark.parametrize(
        "grad,C,A,g,expected_resid,expected_gap",
        [
            # Nothing to explain the gradient with.
            ((3.0, -4.0), None, None, None, 4.0, 0.0),
            # Equality normal absorbs ∇f entirely.
            ((2.0, 2.0), _ONES, None, None, 0.0, 0.0),
            # Equality on θ₁ leaves the θ₂ component.
            ((1.0, 1.0), _E1, None, None, 1.0, 0.0),
            # Rank-deficient C (duplicated direction) behaves like one row.
            ((1.0, 1.0), np.array([[1.0, 0.0], [2.0, 0.0]]), None, None, 1.0, 0.0),
            # Inequality, correct sign: μ = 2.
            ((2.0, 0.0), None, _E1, (0.0,), 0.0, 0.0),
            # Inequality, wrong sign: μ clipped to 0, nothing absorbed.
            ((-2.0, 0.0), None, _E1, (0.0,), 2.0, 0.0),
            # Working-set row with slack 0.1 used at μ = 2: gap 0.2.
            ((2.0, 0.0), None, _E1, (0.1,), 0.0, 0.2),
            # Slightly violated row (g < 0) contributes no gap.
            ((2.0, 0.0), None, _E1, (-1e-8,), 0.0, 0.0),
            # Equality (free sign, λ = −3) and inequality (μ = 2) together.
            ((-3.0, 2.0), _E1, _E2, (0.0,), 0.0, 0.0),
            # Inequality normal projected onto null(C): the θ₁ part of
            # (1, 1) is irrelevant once θ₁ is fixed, μ = 1 absorbs θ₂.
            ((5.0, 1.0), _E1, _ONES, (0.0,), 0.0, 0.0),
        ],
        ids=[
            "unconstrained",
            "equality-absorbs",
            "equality-other-coordinate",
            "equality-rank-deficient",
            "inequality-right-sign",
            "inequality-wrong-sign",
            "slack-row-gap",
            "violated-row-no-gap",
            "equality-and-inequality",
            "inequality-projected-on-equality",
        ],
    )
    def test_known_values(
        self,
        grad: tuple[float, float],
        C: NDArray[np.float64] | None,
        A: NDArray[np.float64] | None,
        g: tuple[float, ...] | None,
        expected_resid: float,
        expected_gap: float,
    ) -> None:
        C_ = np.zeros((0, 2)) if C is None else C
        A_ = np.zeros((0, 2)) if A is None else A
        g_ = np.zeros(0) if g is None else np.array(g)
        resid, gap = _projected_stationarity(np.array(grad), C_, A_, g_)
        np.testing.assert_allclose(resid, expected_resid, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(gap, expected_gap, rtol=1e-10, atol=1e-12)

    def test_empty_parameter_vector(self) -> None:
        resid, gap = _projected_stationarity(
            np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
        )
        assert resid == 0.0
        assert gap == 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_residual_is_feasible_descent_direction(self, seed: int) -> None:
        """Any unexplained residual r yields d = −r with C d = 0, A_𝒜 d ≥ 0
        and ∇fᵀd = −‖r‖₂² < 0 — the descent direction a stall left behind."""
        from scipy.optimize import nnls

        rng = np.random.default_rng(seed)
        n, m_eq, m_act = 6, 2, 4
        grad = rng.normal(size=n)
        C = rng.normal(size=(m_eq, n))
        A = rng.normal(size=(m_act, n))
        resid_inf, _ = _projected_stationarity(grad, C, A, np.zeros(m_act))
        # Rebuild r independently: project onto null(C), then NNLS.
        P = np.eye(n) - np.linalg.pinv(C) @ C
        mu, _ = nnls(P @ A.T, P @ grad)
        r = P @ grad - P @ A.T @ mu
        np.testing.assert_allclose(np.max(np.abs(r)), resid_inf, rtol=1e-8)
        d = -r
        np.testing.assert_allclose(C @ d, 0.0, atol=1e-10)
        assert np.all(A @ d >= -1e-10)
        np.testing.assert_allclose(grad @ d, -(r @ r), rtol=1e-8, atol=1e-12)


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
        from mltpy.basis import BernsteinBasis
        from mltpy.optimizer import OptimizationResult, OptimizerConfig, optimize

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
        from mltpy.basis import BernsteinBasis
        from mltpy.optimizer import OptimizerConfig, optimize

        rng = np.random.default_rng(1)
        y = rng.uniform(0.05, 0.95, 80)
        basis = BernsteinBasis(order=4, support=(0.0, 1.0))
        result = optimize(basis, y, config=OptimizerConfig(solver="auglag"))
        n_params = basis.order + 1
        theta_b = result.theta[:n_params]
        assert np.all(np.diff(theta_b) >= -1e-6)

    def test_auglag_slsqp_agree(self) -> None:
        """Auglag and SLSQP find the same log-likelihood on the same data."""
        from mltpy.basis import BernsteinBasis
        from mltpy.optimizer import OptimizerConfig, optimize

        rng = np.random.default_rng(7)
        y = rng.uniform(0.05, 0.95, 60)
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        r_slsqp = optimize(basis, y, config=OptimizerConfig(solver="slsqp"))
        r_auglag = optimize(basis, y, config=OptimizerConfig(solver="auglag"))
        assert abs(r_auglag.log_likelihood - r_slsqp.log_likelihood) < 1e-4

    def test_auglag_result_auglag_only_fields(self) -> None:
        """n_outer_iter and kkt_residual are None for SLSQP, set for auglag."""
        from mltpy.basis import BernsteinBasis
        from mltpy.optimizer import OptimizerConfig, optimize

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
        from mltpy.basis import BernsteinBasis
        from mltpy.optimizer import OptimizerConfig, optimize

        rng = np.random.default_rng(3)
        y = rng.uniform(0.05, 0.95, 40)
        basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        config = OptimizerConfig(solver="auglag", auglag_options=None)
        result = optimize(basis, y, config=config)
        assert result.converged


# ---------------------------------------------------------------------------
# Box bounds on the inner solve (issue #114)
# ---------------------------------------------------------------------------


def _runaway_obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
    """``f(x) = x₀·exp(x₁) + x₁²/2``, unbounded below as ``x₀ → −∞, x₁ → ∞``.

    Mirrors the scaled exponential likelihood's smooth extension: below the
    support row ``x₀ ≥ 0`` the objective falls exponentially in ``x₁`` while a
    PHR penalty on the row grows only quadratically.  Over ``x₀ ≥ 0`` the
    minimum is ``(0, 0)`` with multiplier ``∂f/∂x₀ = exp(0) = 1``.
    """
    x0, x1 = float(theta[0]), float(theta[1])
    e = np.exp(x1)
    return x0 * e + 0.5 * x1 * x1, np.array([e, x0 * e + x1])


_NO_BOUNDS = np.array([-np.inf, -np.inf])


class TestBoxBounds:
    """``bounds=(lb, ub)`` is passed to the inner solve (box-constrained AL).

    Birgin & Martínez (2014) minimise each PHR subproblem over a box Ω; the
    bound multipliers are read off the projected augmented-Lagrangian gradient.
    """

    @pytest.mark.parametrize(
        ("center", "lb", "ub", "expected", "nu_lo", "nu_up"),
        [
            # Lower bound active: x₀ pushed to 0, ν_lo = ∂f/∂x₀ = 0 − (−1) = 1.
            (
                [-1.0, 2.0],
                [0.0, -np.inf],
                [np.inf, np.inf],
                [0.0, 2.0],
                [1.0, 0],
                [0, 0],
            ),
            # Upper bound active on x₁: ν_up = −∂f/∂x₁ = 3 − 2 = 1.
            (
                [0.5, 3.0],
                [-np.inf, -np.inf],
                [np.inf, 2.0],
                [0.5, 2.0],
                [0, 0],
                [0, 1.0],
            ),
            # Interior optimum: bounds inactive, multipliers zero.
            ([0.5, 0.5], [0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0, 0], [0, 0]),
        ],
    )
    def test_quadratic(
        self,
        center: list[float],
        lb: list[float],
        ub: list[float],
        expected: list[float],
        nu_lo: list[float],
        nu_up: list[float],
    ) -> None:
        result = auglag_minimize(
            quadratic_obj(np.array(center)),
            np.array([0.5, 0.5]),
            A_ineq=None,
            b_ineq=None,
            C_eq=None,
            d_eq=None,
            bounds=(np.array(lb), np.array(ub)),
        )
        assert result.converged
        np.testing.assert_allclose(result.theta, expected, atol=1e-8)
        np.testing.assert_allclose(result.nu_lower, nu_lo, atol=1e-6)
        np.testing.assert_allclose(result.nu_upper, nu_up, atol=1e-6)
        assert result.kkt_residual < 1e-6

    def test_box_keeps_every_evaluation_inside(self) -> None:
        """The PHR subproblem is unbounded below for x₀ < 0; the box keeps every
        objective evaluation at x₀ ≥ 0, so no inner solve can enter that region.
        """
        A = np.array([[1.0, 0.0]])
        seen_x0: list[float] = []

        def recording(theta: np.ndarray) -> tuple[float, np.ndarray]:
            seen_x0.append(float(theta[0]))
            return _runaway_obj(theta)

        # L_A at μ = 0, ρ = 10 along x₀ = −1: f + 5 → −∞ as x₁ grows.
        assert _runaway_obj(np.array([-1.0, 20.0]))[0] + 5.0 < -1e8
        boxed = auglag_minimize(
            recording,
            np.array([0.5, 0.0]),
            A_ineq=A,
            b_ineq=None,
            C_eq=None,
            d_eq=None,
            bounds=(np.array([0.0, -np.inf]), np.array([np.inf, np.inf])),
        )
        assert boxed.converged
        assert min(seen_x0) >= 0.0
        np.testing.assert_allclose(boxed.theta, [0.0, 0.0], atol=1e-7)
        # The duplicated PHR row never goes infeasible, so its own multiplier
        # stays 0 and the force is carried by the bound multiplier.
        np.testing.assert_allclose(boxed.mu_ineq, [0.0], atol=1e-12)
        np.testing.assert_allclose(boxed.nu_lower, [1.0, 0.0], atol=1e-6)

    def test_with_linear_rows(self) -> None:
        """Box + inequality: min ½‖x − (−1, −1)‖² s.t. x₀ ≥ 0, x₁ − x₀ ≥ 0."""
        result = auglag_minimize(
            quadratic_obj(np.array([-1.0, -1.0])),
            np.array([0.5, 1.0]),
            A_ineq=np.array([[-1.0, 1.0]]),
            b_ineq=None,
            C_eq=None,
            d_eq=None,
            bounds=(np.array([0.0, -np.inf]), np.array([np.inf, np.inf])),
        )
        assert result.converged
        np.testing.assert_allclose(result.theta, [0.0, 0.0], atol=1e-7)
        # ∇f = (1, 1) = ν_lo e₀ + μ (−1, 1)  ⇒  μ = 1, ν_lo = 2.
        np.testing.assert_allclose(result.mu_ineq, [1.0], atol=1e-5)
        np.testing.assert_allclose(result.nu_lower, [2.0, 0.0], atol=1e-5)

    def test_x0_outside_box_is_projected(self) -> None:
        result = auglag_minimize(
            quadratic_obj(np.array([1.0, 1.0])),
            np.array([-5.0, 9.0]),
            A_ineq=None,
            b_ineq=None,
            C_eq=None,
            d_eq=None,
            bounds=(np.array([0.0, 0.0]), np.array([2.0, 2.0])),
        )
        assert result.converged
        np.testing.assert_allclose(result.theta, [1.0, 1.0], atol=1e-8)

    def test_no_bounds_matches_default(self) -> None:
        """``bounds=None`` leaves the solve unchanged; ν are zero vectors."""
        obj = quadratic_obj(np.array([-1.0, 2.0]))
        A = np.array([[1.0, 0.0]])
        a = auglag_minimize(
            obj, np.ones(2), A_ineq=A, b_ineq=None, C_eq=None, d_eq=None
        )
        b = auglag_minimize(
            obj,
            np.ones(2),
            A_ineq=A,
            b_ineq=None,
            C_eq=None,
            d_eq=None,
            bounds=None,
        )
        np.testing.assert_array_equal(a.theta, b.theta)
        np.testing.assert_array_equal(a.mu_ineq, b.mu_ineq)
        np.testing.assert_array_equal(a.nu_lower, np.zeros(2))
        np.testing.assert_array_equal(a.nu_upper, np.zeros(2))

    @pytest.mark.parametrize(
        ("bounds", "match"),
        [
            ((np.zeros(3), np.ones(3)), "shape"),
            ((np.zeros(2), np.ones(3)), "shape"),
            ((np.ones(2), np.zeros(2)), "lb <= ub"),
            ((np.array([np.nan, 0.0]), np.ones(2)), "NaN"),
            ((np.array([np.inf, 0.0]), np.full(2, np.inf)), "lb"),
        ],
    )
    def test_invalid_bounds(
        self, bounds: tuple[np.ndarray, np.ndarray], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            auglag_minimize(
                quadratic_obj(np.zeros(2)),
                np.zeros(2),
                A_ineq=None,
                b_ineq=None,
                C_eq=None,
                d_eq=None,
                bounds=bounds,
            )

    def test_bounds_require_bounded_inner_method(self) -> None:
        with pytest.raises(ValueError, match="inner_method"):
            auglag_minimize(
                quadratic_obj(np.zeros(2)),
                np.zeros(2),
                A_ineq=None,
                b_ineq=None,
                C_eq=None,
                d_eq=None,
                bounds=(np.zeros(2), np.ones(2)),
                options=AugLagOptions(inner_method="BFGS"),
            )
