"""Tests for pymlt.optimizer — convergence, feasibility, result structure."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pymlt._auglag import AugLagOptions
from pymlt.basis import BernsteinBasis
from pymlt.constraints import MonotonicityConstraint
from pymlt.likelihood import negative_log_likelihood
from pymlt.optimizer import (
    OptimizationResult,
    OptimizerConfig,
    _initial_theta,
    _make_objective,
    _perturb_and_project,
    _project_to_feasible,
    optimize,
)
from pymlt.variables import CensoredData, CensoringType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_basis(order: int = 3, support: tuple = (0.0, 1.0)) -> BernsteinBasis:
    return BernsteinBasis(order=order, support=support)


def simple_data(n: int = 60, seed: int = 0) -> np.ndarray:
    """Uniform observations on (0.05, 0.95) — easy test case."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.95, n)


# ---------------------------------------------------------------------------
# OptimizerConfig defaults
# ---------------------------------------------------------------------------


class TestOptimizerConfig:
    def test_defaults(self):
        cfg = OptimizerConfig()
        # auglag is the default (PHR augmented Lagrangian) — matches R `mlt`
        # which uses `alabama::auglag`.  SLSQP / trust-constr remain opt-in.
        assert cfg.solver == "auglag"
        assert cfg.max_iter == 1000
        assert cfg.tol == 1e-8
        assert cfg.max_restarts == 3
        assert cfg.use_gradient is True
        assert cfg.verbose is False

    def test_polish_defaults_true(self):
        assert OptimizerConfig().polish is True

    def test_polish_can_be_disabled(self):
        assert OptimizerConfig(polish=False).polish is False

    def test_custom(self):
        cfg = OptimizerConfig(solver="trust-constr", max_iter=500, verbose=True)
        assert cfg.solver == "trust-constr"
        assert cfg.max_iter == 500
        assert cfg.verbose is True


# ---------------------------------------------------------------------------
# OptimizationResult structure
# ---------------------------------------------------------------------------


class TestOptimizationResultFields:
    def test_all_fields_present(self):
        basis = make_basis()
        y = simple_data()
        result = optimize(basis, y)
        assert hasattr(result, "theta")
        assert hasattr(result, "log_likelihood")
        assert hasattr(result, "converged")
        assert hasattr(result, "n_iter")
        assert hasattr(result, "n_restarts")
        assert hasattr(result, "solver_message")

    def test_theta_shape(self):
        order = 4
        basis = make_basis(order=order)
        y = simple_data()
        result = optimize(basis, y)
        assert result.theta.shape == (order + 1,)

    def test_log_likelihood_is_float(self):
        basis = make_basis()
        result = optimize(basis, simple_data())
        assert isinstance(result.log_likelihood, float)
        assert np.isfinite(result.log_likelihood)

    def test_converged_is_bool(self):
        result = optimize(make_basis(), simple_data())
        assert isinstance(result.converged, bool)

    def test_solver_message_is_str(self):
        result = optimize(make_basis(), simple_data())
        assert isinstance(result.solver_message, str)

    def test_auglag_multiplier_fields_present(self):
        result = optimize(make_basis(), simple_data())
        assert hasattr(result, "rho_final")
        assert hasattr(result, "mu_ineq")
        assert hasattr(result, "lambda_eq")

    def test_auglag_multiplier_types(self):
        result = optimize(make_basis(), simple_data())
        assert isinstance(result.rho_final, float)
        assert isinstance(result.mu_ineq, np.ndarray)
        assert isinstance(result.lambda_eq, np.ndarray)

    def test_auglag_mu_ineq_shape(self):
        order = 4
        result = optimize(make_basis(order=order), simple_data())
        # one inequality per adjacent-pair difference → order constraints
        assert result.mu_ineq.shape == (order,)

    def test_auglag_lambda_eq_shape(self):
        result = optimize(make_basis(order=3), simple_data())
        # shift model has no equality constraints
        assert result.lambda_eq.shape == (0,)

    def test_auglag_mu_ineq_nonneg(self):
        result = optimize(make_basis(), simple_data())
        assert np.all(result.mu_ineq >= -1e-8)

    def test_slsqp_multiplier_fields_are_none(self):
        cfg = OptimizerConfig(solver="slsqp")
        result = optimize(make_basis(), simple_data(), config=cfg)
        assert result.rho_final is None
        assert result.mu_ineq is None
        assert result.lambda_eq is None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestProjectToFeasible:
    def test_already_sorted(self):
        theta = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(_project_to_feasible(theta), theta)

    def test_reversed(self):
        theta = np.array([4.0, 3.0, 2.0, 1.0])
        result = _project_to_feasible(theta)
        assert np.all(np.diff(result) >= 0)

    def test_random_is_monotone(self):
        rng = np.random.default_rng(5)
        theta = rng.standard_normal(8)
        result = _project_to_feasible(theta)
        assert np.all(np.diff(result) >= 0)


class TestInitialTheta:
    def test_no_x(self):
        theta = _initial_theta(5, None)
        assert theta.shape == (5,)
        assert np.all(np.diff(theta) >= 0)

    def test_with_x(self):
        X = np.ones((10, 3))
        theta = _initial_theta(4, X)
        assert theta.shape == (7,)
        np.testing.assert_array_equal(theta[4:], 0.0)

    def test_is_linspace(self):
        theta = _initial_theta(5, None)
        np.testing.assert_allclose(theta, np.linspace(0, 1, 5))


class TestPerturbAndProject:
    def test_output_is_feasible(self):
        rng = np.random.default_rng(42)
        theta = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
        for _ in range(20):
            result = _perturb_and_project(theta, 5, rng)
            assert result.shape == (5,)
            assert np.all(np.diff(result) >= 0)

    def test_beta_unchanged(self):
        rng = np.random.default_rng(0)
        theta = np.array([0.0, 0.5, 1.0, 2.5, 3.0])  # 3 basis + 2 beta
        for _ in range(10):
            result = _perturb_and_project(theta, 3, rng)
            np.testing.assert_array_equal(result[3:], theta[3:])


# ---------------------------------------------------------------------------
# optimize() — convergence and feasibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["auglag", "slsqp", "trust-constr"])
class TestOptimizeConvergence:
    """Convergence and feasibility under every supported solver.

    Parametrising over all three solvers turns the full assertion set into a
    regression net for the new auglag default while keeping SLSQP / trust-constr
    exercised for as long as they remain opt-in alternatives.
    """

    def test_converges_on_simple_data(self, solver):
        cfg = OptimizerConfig(solver=solver)
        basis = make_basis(order=3)
        y = simple_data(n=80)
        result = optimize(basis, y, config=cfg)
        assert result.converged, f"Did not converge: {result.solver_message}"

    def test_theta_is_non_decreasing(self, solver):
        cfg = OptimizerConfig(solver=solver)
        basis = make_basis(order=4)
        y = simple_data(n=80)
        result = optimize(basis, y, config=cfg)
        n_params = basis.order + 1
        theta_b = result.theta[:n_params]
        assert np.all(np.diff(theta_b) >= -1e-6), f"theta not non-decreasing: {theta_b}"

    def test_nll_decreased_from_init(self, solver):
        """Optimised NLL ≤ initial NLL."""
        from pymlt.likelihood import negative_log_likelihood

        cfg = OptimizerConfig(solver=solver)
        basis = make_basis(order=3)
        y = simple_data(n=60)
        theta_init = _initial_theta(basis.order + 1, None)
        nll_init = negative_log_likelihood(theta_init, basis, y)
        result = optimize(basis, y, config=cfg)
        assert -result.log_likelihood <= nll_init + 1e-6

    def test_monotonicity_constraint_satisfied(self, solver):
        """D @ theta_b >= 0 for the optimised parameters."""
        cfg = OptimizerConfig(solver=solver)
        basis = make_basis(order=5)
        y = simple_data(n=100)
        result = optimize(basis, y, config=cfg)
        D = MonotonicityConstraint(basis.order + 1).as_matrix()
        violations = D @ result.theta[: basis.order + 1]
        assert np.all(violations >= -1e-6), (
            f"Constraint violated: {violations.min():.2e}"
        )

    def test_log_likelihood_is_finite(self, solver):
        cfg = OptimizerConfig(solver=solver)
        basis = make_basis(order=3)
        y = simple_data(n=50)
        result = optimize(basis, y, config=cfg)
        assert np.isfinite(result.log_likelihood)


class TestOptimizeWithCensoredData:
    def test_right_censored(self):
        rng = np.random.default_rng(1)
        basis = make_basis(order=3)
        y = rng.uniform(0.05, 0.95, 60)
        censored = rng.random(60) < 0.3
        cd = CensoredData.right_censored(y, censored)
        result = optimize(basis, cd, censoring=CensoringType.RIGHT)
        assert np.isfinite(result.log_likelihood)
        assert np.all(np.diff(result.theta) >= -1e-6)

    def test_interval_censored(self):
        rng = np.random.default_rng(2)
        basis = make_basis(order=3)
        centers = rng.uniform(0.1, 0.9, 40)
        cd = CensoredData.interval_censored(centers - 0.05, centers + 0.05)
        result = optimize(basis, cd, censoring=CensoringType.INTERVAL)
        assert np.isfinite(result.log_likelihood)
        assert np.all(np.diff(result.theta) >= -1e-6)


class TestOptimizeWithCovariates:
    def test_with_x(self):
        rng = np.random.default_rng(3)
        n, q = 60, 2
        basis = make_basis(order=3)
        y = rng.uniform(0.05, 0.95, n)
        X = rng.standard_normal((n, q))
        result = optimize(basis, y, X=X)
        assert result.theta.shape == (basis.order + 1 + q,)
        assert np.isfinite(result.log_likelihood)
        # Only the basis part must be monotone
        assert np.all(np.diff(result.theta[: basis.order + 1]) >= -1e-6)


class TestOptimizeSolvers:
    def test_slsqp_solver(self):
        cfg = OptimizerConfig(solver="slsqp")
        result = optimize(make_basis(), simple_data(), config=cfg)
        assert result.converged

    def test_trust_constr_solver(self):
        cfg = OptimizerConfig(solver="trust-constr", max_iter=500)
        result = optimize(make_basis(order=3), simple_data(n=40), config=cfg)
        assert np.isfinite(result.log_likelihood)
        assert np.all(np.diff(result.theta) >= -1e-6)

    def test_no_gradient(self):
        """use_gradient=False still converges (slower, no gradient)."""
        cfg = OptimizerConfig(use_gradient=False, max_iter=2000)
        basis = make_basis(order=2)
        y = simple_data(n=40)
        result = optimize(basis, y, config=cfg)
        assert np.isfinite(result.log_likelihood)


class TestOptimizeVerbose:
    def test_verbose_no_crash(self):
        cfg = OptimizerConfig(verbose=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = optimize(make_basis(), simple_data(), config=cfg)
        assert isinstance(result, OptimizationResult)


class TestOptimizeRestarts:
    def test_n_restarts_field(self):
        cfg = OptimizerConfig(max_restarts=2)
        result = optimize(make_basis(), simple_data(), config=cfg)
        assert result.n_restarts <= cfg.max_restarts

    def test_first_attempt_succeeds_no_restart(self):
        """Easy data → converges first try → n_restarts == 0."""
        cfg = OptimizerConfig(max_restarts=5)
        result = optimize(make_basis(order=2), simple_data(n=50), config=cfg)
        if result.converged:
            assert result.n_restarts == 0


class TestOptimizeReproducibility:
    """``OptimizerConfig.random_state`` controls the restart-perturbation RNG.

    Pinned to ``solver="slsqp"``: the restart-RNG plumbing is shared with the
    auglag path, but ``max_iter=1`` is the SLSQP-specific knob that forces
    scipy to bail early and trigger the restart loop.  Auglag has its own outer
    budget via :class:`~pymlt._auglag.AugLagOptions` and does not honour
    ``max_iter``; on this easy fixture it converges on the first attempt so the
    RNG is never consulted.  Testing the same code path through SLSQP is
    sufficient — both solvers wrap the same ``_perturb_and_project`` /
    ``np.random.default_rng`` machinery in ``optimize()``.
    """

    def _run(self, random_state):
        cfg = OptimizerConfig(
            solver="slsqp",
            max_iter=1,
            max_restarts=3,
            random_state=random_state,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return optimize(make_basis(), simple_data(), config=cfg)

    def test_same_seed_same_theta(self):
        r1 = self._run(42)
        r2 = self._run(42)
        np.testing.assert_allclose(r1.theta, r2.theta)

    def test_different_seed_different_theta(self):
        r1 = self._run(1)
        r2 = self._run(2)
        assert not np.allclose(r1.theta, r2.theta)

    def test_generator_instance_accepted(self):
        r_int = self._run(7)
        r_gen = self._run(np.random.default_rng(7))
        np.testing.assert_allclose(r_int.theta, r_gen.theta)


# ---------------------------------------------------------------------------
# base_distribution validation in optimize()
# ---------------------------------------------------------------------------


class TestBaseDistributionValidation:
    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="base_distribution"):
            optimize(make_basis(), simple_data(), base_distribution="student-t")

    @pytest.mark.parametrize("bad", ["Normal", "LOGISTIC", "gauss", "", "t"])
    def test_case_sensitive_aliases_rejected(self, bad):
        with pytest.raises(ValueError, match="base_distribution"):
            optimize(make_basis(), simple_data(), base_distribution=bad)

    def test_normal_accepted(self):
        result = optimize(make_basis(), simple_data(), base_distribution="normal")
        assert isinstance(result, OptimizationResult)

    def test_logistic_accepted(self):
        result = optimize(make_basis(), simple_data(), base_distribution="logistic")
        assert isinstance(result, OptimizationResult)

    def test_min_extreme_value_accepted(self):
        result = optimize(
            make_basis(), simple_data(), base_distribution="min_extreme_value"
        )
        assert isinstance(result, OptimizationResult)

    def test_max_extreme_value_accepted(self):
        result = optimize(
            make_basis(), simple_data(), base_distribution="max_extreme_value"
        )
        assert isinstance(result, OptimizationResult)

    def test_exponential_accepted_and_nonnegative_lower(self):
        """Exponential fit must respect h(y_min) = theta_b[0] >= 0."""
        result = optimize(make_basis(), simple_data(), base_distribution="exponential")
        assert isinstance(result, OptimizationResult)
        # Feasibility (within SLSQP tolerance): theta_b[0] >= 0
        assert result.theta[0] >= -1e-6

    def test_optimize_resolves_distribution_once(self, monkeypatch):
        calls = 0

        def count_optimizer_get_dist(base_distribution):
            nonlocal calls
            calls += 1
            from pymlt.likelihood import _NORM_OPS

            return _NORM_OPS

        def fail_likelihood_get_dist(base_distribution):
            raise AssertionError(
                "_get_dist should not run inside optimize() evaluations"
            )

        monkeypatch.setattr("pymlt.optimizer._get_dist", count_optimizer_get_dist)
        monkeypatch.setattr("pymlt.likelihood._get_dist", fail_likelihood_get_dist)

        result = optimize(make_basis(order=2), simple_data(n=20))

        assert isinstance(result, OptimizationResult)
        assert calls == 1


# ---------------------------------------------------------------------------
# _make_objective — ValueError fallback penalty
# ---------------------------------------------------------------------------


class TestMakeObjectivePenalty:
    """Infeasible theta (h' ≤ 0) must trigger the ValueError catch that
    returns a large penalty instead of crashing the optimiser."""

    def test_infeasible_theta_grad_returns_penalty(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        y = np.array([0.3, 0.5, 0.7])
        obj = _make_objective(basis, y, None, CensoringType.NONE, use_gradient=True)
        theta_bad = np.array([10.0, 5.0, 0.0, -5.0])  # strictly decreasing
        val, grad = obj(theta_bad)
        assert val == 1e10
        # Gradient must point away from infeasibility: a small step along the
        # descent direction -grad should reduce the monotonicity violation.
        assert np.any(grad != 0.0)
        D = MonotonicityConstraint(n_params=theta_bad.size).as_matrix()
        violation_before = np.sum(np.minimum(D @ theta_bad, 0.0) ** 2)
        theta_step = theta_bad - 1e-3 * grad
        violation_after = np.sum(np.minimum(D @ theta_step, 0.0) ** 2)
        assert violation_after < violation_before

    def test_infeasible_theta_with_beta_grad_zero_on_beta_block(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        y = np.array([0.3, 0.5, 0.7])
        X = np.array([[1.0], [2.0], [3.0]])
        obj = _make_objective(basis, y, X, CensoringType.NONE, use_gradient=True)
        theta_bad = np.array([10.0, 5.0, 0.0, -5.0, 0.25])  # last entry is beta
        val, grad = obj(theta_bad)
        assert val == 1e10
        assert grad.shape == theta_bad.shape
        # Beta slice must be zero — beta does not enter monotonicity.
        assert grad[-1] == 0.0
        # Theta_b slice must be non-trivial.
        assert np.any(grad[:-1] != 0.0)

    def test_infeasible_theta_nograd_returns_penalty(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        y = np.array([0.3, 0.5, 0.7])
        obj = _make_objective(basis, y, None, CensoringType.NONE, use_gradient=False)
        theta_bad = np.array([10.0, 5.0, 0.0, -5.0])
        val = obj(theta_bad)
        assert val == 1e10

    def test_infeasible_subgradient_does_not_stall(self):
        """Gradient descent using only the closure's infeasibility-path
        output must drive a deeply non-monotone theta back into the feasible
        cone.  Regression guard: the previous zero-gradient fallback left
        gradient descent stationary forever."""
        basis = BernsteinBasis(order=4, support=(0.0, 1.0))
        y = np.array([0.2, 0.4, 0.6, 0.8])
        obj = _make_objective(basis, y, None, CensoringType.NONE, use_gradient=True)
        D = MonotonicityConstraint(n_params=basis.order + 1).as_matrix()

        theta = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # strictly decreasing
        step = 0.5
        for _ in range(200):
            violation = np.minimum(D @ theta, 0.0)
            if np.all(violation == 0.0):
                break
            _, grad = obj(theta)
            theta = theta - step * grad
        assert np.all(D @ theta >= -1e-9)


# ---------------------------------------------------------------------------
# optimize() — scipy-exception / all-fail fallback
# ---------------------------------------------------------------------------


class TestOptimizeExceptionFallback:
    """SLSQP-specific failure handling: the SLSQP / trust-constr path calls
    ``scipy.optimize.minimize`` directly, so monkeypatching that import
    is the lever for forcing a ``LinAlgError``.  The auglag path goes through
    :func:`~pymlt._auglag.auglag_minimize`; equivalent failure-mode coverage
    for that path lives in ``tests/test_auglag.py``."""

    def test_linalg_error_retries_and_falls_back(self, monkeypatch):
        """Every scipy call raises LinAlgError → fallback returns initial theta."""
        from numpy.linalg import LinAlgError

        def boom(*args, **kwargs):
            raise LinAlgError("singular matrix")

        monkeypatch.setattr("pymlt.optimizer.minimize", boom)

        cfg = OptimizerConfig(solver="slsqp", max_restarts=2)
        basis = make_basis(order=3)
        result = optimize(basis, simple_data(), config=cfg)

        assert not result.converged
        assert result.n_iter == 0
        assert "linalgerror" in result.solver_message.lower()
        np.testing.assert_allclose(result.theta, np.linspace(0.0, 1.0, basis.order + 1))
        assert np.isfinite(result.log_likelihood)

    def test_unrelated_exception_propagates(self, monkeypatch):
        """A non-LinAlgError from minimize must bubble out — not be silenced."""

        def boom(*args, **kwargs):
            raise RuntimeError("scipy blew up")

        monkeypatch.setattr("pymlt.optimizer.minimize", boom)

        cfg = OptimizerConfig(solver="slsqp", max_restarts=2)
        with pytest.raises(RuntimeError, match="scipy blew up"):
            optimize(make_basis(order=3), simple_data(), config=cfg)

    def test_linalg_error_with_verbose_warns(self, monkeypatch):
        """verbose=True → RuntimeWarning emitted on each LinAlgError retry."""
        from numpy.linalg import LinAlgError

        def boom(*args, **kwargs):
            raise LinAlgError("singular matrix")

        monkeypatch.setattr("pymlt.optimizer.minimize", boom)

        cfg = OptimizerConfig(solver="slsqp", max_restarts=0, verbose=True)
        with pytest.warns(RuntimeWarning, match="hit"):
            optimize(make_basis(), simple_data(), config=cfg)


# ---------------------------------------------------------------------------
# _make_objective — narrow catch (infeasibility only)
# ---------------------------------------------------------------------------


class TestMakeObjectiveNarrowCatch:
    """Only InfeasibleParameterError is absorbed into the penalty; other
    ValueErrors — e.g. a shape bug or an unsupported distribution — must
    propagate out so that genuine bugs are surfaced."""

    def test_unrelated_value_error_propagates_grad(self, monkeypatch):
        basis = make_basis(order=3)
        y = simple_data()
        obj = _make_objective(basis, y, None, CensoringType.NONE, use_gradient=True)

        def bad_nll(*args, **kwargs):
            raise ValueError("bad shape")

        monkeypatch.setattr(
            "pymlt.optimizer._negative_log_likelihood_from_dist", bad_nll
        )
        with pytest.raises(ValueError, match="bad shape"):
            obj(np.linspace(0.0, 1.0, basis.order + 1))

    def test_unrelated_value_error_propagates_nograd(self, monkeypatch):
        basis = make_basis(order=3)
        y = simple_data()
        obj = _make_objective(basis, y, None, CensoringType.NONE, use_gradient=False)

        def bad_nll(*args, **kwargs):
            raise ValueError("bad shape")

        monkeypatch.setattr(
            "pymlt.optimizer._negative_log_likelihood_from_dist", bad_nll
        )
        with pytest.raises(ValueError, match="bad shape"):
            obj(np.linspace(0.0, 1.0, basis.order + 1))


# ---------------------------------------------------------------------------
# optimize() — verbose warning on genuine non-convergence
# ---------------------------------------------------------------------------


class TestOptimizeVerboseNonConvergence:
    def test_verbose_warns_on_non_convergence(self):
        """Tight iter budget → scipy returns success=False → verbose warning.

        Pinned to SLSQP: ``max_iter`` is the SLSQP/trust-constr iteration cap.
        Auglag uses :attr:`~pymlt._auglag.AugLagOptions.max_outer_iter` (its
        own verbose-warning path is unit-tested in ``test_auglag.py``).
        """
        cfg = OptimizerConfig(solver="slsqp", max_iter=1, max_restarts=3, verbose=True)
        with pytest.warns(RuntimeWarning, match="did not converge"):
            optimize(make_basis(order=3), simple_data(n=80), config=cfg)


# ---------------------------------------------------------------------------
# Newton-CG polish step
# ---------------------------------------------------------------------------


class TestPolishStep:
    """Newton-CG polish step applied after auglag for interior-MLE fits."""

    @pytest.fixture
    def interior_setup(self):
        """Logistic data clipped to (-5, 5) — all monotonicity constraints inactive."""
        rng = np.random.default_rng(42)
        y_raw = rng.logistic(0.0, 1.0, 300)
        y = np.clip(y_raw, -5.0, 5.0)
        basis = BernsteinBasis(order=6, support=(-5.0, 5.0))
        return basis, y

    # Deliberately very loose: 1 outer iter, coarse inner solver — leaves
    # gradient ~1.3, well above the trust-ncg convergence threshold.
    _LOOSE = AugLagOptions(
        outer_tol=1e-1,
        max_outer_iter=1,
        inner_options={"maxiter": 20, "gtol": 1e-2, "ftol": 1e-3},
    )

    def test_polish_does_not_regress_nll(self, interior_setup):
        """polish=True returns NLL at least as good as polish=False."""
        basis, y = interior_setup
        r_no = optimize(basis, y, config=OptimizerConfig(solver="auglag", polish=False))
        r_yes = optimize(basis, y, config=OptimizerConfig(solver="auglag", polish=True))
        assert r_yes.log_likelihood >= r_no.log_likelihood - 1e-10

    def test_polish_tightens_gradient_on_loose_auglag(self, interior_setup):
        """Polish tightens the gradient by orders of magnitude on a coarse auglag."""
        basis, y = interior_setup
        r_no = optimize(
            basis,
            y,
            config=OptimizerConfig(
                solver="auglag", auglag_options=self._LOOSE, polish=False
            ),
        )
        r_yes = optimize(
            basis,
            y,
            config=OptimizerConfig(
                solver="auglag", auglag_options=self._LOOSE, polish=True
            ),
        )
        _, g_no = negative_log_likelihood(r_no.theta, basis, y, gradient=True)
        _, g_yes = negative_log_likelihood(r_yes.theta, basis, y, gradient=True)
        assert np.linalg.norm(g_yes) < np.linalg.norm(g_no) * 1e-3

    def test_polish_skipped_when_disabled(self, interior_setup):
        """With the same seed, polish=False and polish=True start at the same auglag
        theta; polish=True then reduces the gradient strictly."""
        basis, y = interior_setup
        r_no = optimize(
            basis,
            y,
            config=OptimizerConfig(
                solver="auglag",
                auglag_options=self._LOOSE,
                polish=False,
                random_state=0,
            ),
        )
        r_yes = optimize(
            basis,
            y,
            config=OptimizerConfig(
                solver="auglag", auglag_options=self._LOOSE, polish=True, random_state=0
            ),
        )
        _, g_no = negative_log_likelihood(r_no.theta, basis, y, gradient=True)
        _, g_yes = negative_log_likelihood(r_yes.theta, basis, y, gradient=True)
        assert np.linalg.norm(g_yes) < np.linalg.norm(g_no)

    def test_polished_theta_feasible(self, interior_setup):
        """Polish must not push theta_b outside the monotone cone."""
        basis, y = interior_setup
        result = optimize(
            basis,
            y,
            config=OptimizerConfig(
                solver="auglag", auglag_options=self._LOOSE, polish=True
            ),
        )
        n_params = basis.order + 1
        theta_b = result.theta[:n_params]
        assert np.all(np.diff(theta_b) >= -1e-8)

    def test_polish_skipped_for_slsqp(self, interior_setup):
        """polish flag has no effect on non-auglag solvers."""
        basis, y = interior_setup
        r_no = optimize(basis, y, config=OptimizerConfig(solver="slsqp", polish=False))
        r_yes = optimize(basis, y, config=OptimizerConfig(solver="slsqp", polish=True))
        np.testing.assert_array_equal(r_no.theta, r_yes.theta)

    def test_polish_skipped_when_constraints_active(self):
        """Coxph has active monotonicity constraints — polish must be skipped."""
        import warnings

        from pymlt.tram import Coxph

        rng = np.random.default_rng(0)
        t = rng.exponential(1.0, 100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_no = Coxph(
                order=5,
                support=(0.0, 10.0),
                optimizer_config=OptimizerConfig(
                    solver="auglag", polish=False, random_state=7
                ),
            )
            m_no.fit(t)
            m_yes = Coxph(
                order=5,
                support=(0.0, 10.0),
                optimizer_config=OptimizerConfig(
                    solver="auglag", polish=True, random_state=7
                ),
            )
            m_yes.fit(t)

        # When constraints are active the polish condition fails → same theta
        np.testing.assert_array_equal(m_no.theta_, m_yes.theta_)


# ---------------------------------------------------------------------------
# fixed_params support (auglag path) — issue #85
# ---------------------------------------------------------------------------


class TestFixedParamsAuglag:
    """``OptimizerConfig.fixed_params`` pins arbitrary indices of the full
    parameter vector ``[theta_b | beta | gamma]`` at user-supplied values
    during optimisation.  Implemented as equality rows on ``C_eq``/``d_eq``,
    so the constraint reuses the same scaffolding as ``lower``/``upper``
    and is honoured by the auglag solver natively.

    Scope (this iteration): ``solver="auglag"`` + shift basis only.
    SLSQP / trust-constr and ``InteractionBasis`` are rejected with
    ``NotImplementedError``.
    """

    def test_fixed_params_pins_theta_index_to_value(self):
        """fixed_params={i: v} + auglag → result.theta[i] == v at feasibility tol."""
        cfg = OptimizerConfig(solver="auglag", fixed_params={1: 0.3})
        basis = make_basis(order=3)
        y = simple_data(n=60)
        result = optimize(basis, y, config=cfg)
        np.testing.assert_allclose(result.theta[1], 0.3, atol=1e-8)

    def test_fixed_params_off_mle_reduces_log_likelihood(self):
        """Pinning a parameter away from the unconstrained MLE must strictly
        reduce the log-likelihood — proves the equality is a real constraint
        seen by the inner solver, not a stub that the optimiser ignores."""
        basis = make_basis(order=3)
        y = simple_data(n=60)
        free = optimize(basis, y, config=OptimizerConfig(solver="auglag"))
        # Pin theta[1] far from its free value
        off = float(free.theta[1]) + 2.0
        pinned = optimize(
            basis,
            y,
            config=OptimizerConfig(solver="auglag", fixed_params={1: off}),
        )
        np.testing.assert_allclose(pinned.theta[1], off, atol=1e-8)
        # A genuine constraint can only lower the log-likelihood.  Allow a
        # tiny float-round slack so an exactly-at-the-MLE pin would still pass.
        assert pinned.log_likelihood < free.log_likelihood - 1e-6

    def test_fixed_params_adds_one_c_eq_row_per_pinned_index(self):
        """constraint_C_eq grows by exactly len(fixed_params) rows when no
        lower/upper are set.  Each row is a unit vector at the pinned index."""
        basis = make_basis(order=3)
        y = simple_data(n=40)
        cfg = OptimizerConfig(solver="auglag", fixed_params={0: 0.0, 2: 0.5})
        result = optimize(basis, y, config=cfg)

        assert result.constraint_C_eq is not None
        # Two pinned indices → two equality rows over the full theta_ (length 4)
        assert result.constraint_C_eq.shape == (2, basis.order + 1)
        # Each row is a unit vector at the pinned index
        expected = np.zeros((2, basis.order + 1))
        expected[0, 0] = 1.0
        expected[1, 2] = 1.0
        np.testing.assert_array_equal(result.constraint_C_eq, expected)

    def test_fixed_params_stacks_with_lower_upper(self):
        """fixed_params rows are appended after the lower/upper rows; none
        of the existing pins are clobbered."""
        basis = make_basis(order=3)
        y = simple_data(n=40)
        cfg = OptimizerConfig(
            solver="auglag", lower=0.0, upper=1.0, fixed_params={2: 0.4}
        )
        result = optimize(basis, y, config=cfg)

        assert result.constraint_C_eq is not None
        # 2 boundary rows (lower=theta[0], upper=theta[-1]) + 1 fixed_params row
        assert result.constraint_C_eq.shape == (3, basis.order + 1)
        # All three pins are honoured at the fitted theta.  Tolerance is the
        # auglag KKT residual budget — three stacked equalities tighten the
        # joint feasibility a touch beyond the single-side R-parity case.
        np.testing.assert_allclose(result.theta[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.theta[-1], 1.0, atol=1e-6)
        np.testing.assert_allclose(result.theta[2], 0.4, atol=1e-6)

    def test_fixed_params_out_of_range_index_raises(self):
        """Indices outside [0, total_params) must be caught before auglag."""
        basis = make_basis(order=3)  # total_params = 4 (no covariates)
        y = simple_data(n=40)
        cfg = OptimizerConfig(solver="auglag", fixed_params={9: 0.0})
        with pytest.raises(ValueError, match="out-of-range"):
            optimize(basis, y, config=cfg)

    @pytest.mark.parametrize("solver", ["slsqp", "trust-constr"])
    def test_fixed_params_rejected_on_non_auglag_solvers(self, solver):
        """SLSQP / trust-constr would need a parallel equality-constraint
        extension — deferred per issue #85 scope."""
        basis = make_basis(order=3)
        y = simple_data(n=40)
        cfg = OptimizerConfig(solver=solver, fixed_params={1: 0.3})
        with pytest.raises(NotImplementedError, match="fixed_params"):
            optimize(basis, y, config=cfg)

    def test_fixed_params_empty_dict_is_treated_as_no_pin(self):
        """``{}`` is falsy in Python — the NotImplementedError gate and
        the C_eq-row build should both treat it as 'no pins'."""
        basis = make_basis(order=3)
        y = simple_data(n=40)
        # Empty dict on a non-auglag solver must NOT raise (no pins ⇒ no
        # scope-guard trigger)
        cfg = OptimizerConfig(solver="slsqp", fixed_params={})
        result = optimize(basis, y, config=cfg)
        assert result.converged

    def test_fixed_params_negative_index_raises(self):
        """Negative indices are rejected — keeps the API non-ambiguous (we
        do not silently wrap ``-1`` to ``total_params - 1``)."""
        basis = make_basis(order=3)
        y = simple_data(n=40)
        cfg = OptimizerConfig(solver="auglag", fixed_params={-1: 0.0})
        with pytest.raises(ValueError, match="out-of-range"):
            optimize(basis, y, config=cfg)

    def test_fixed_params_rejected_with_interaction_basis(self):
        """InteractionBasis uses a vec_C(Θ) layout — pinning by index needs
        its own design decision (column/row indexing, Kronecker padding).
        Deferred per issue #85 scope."""
        from pymlt.basis import BernsteinBasis, InteractionBasis, OneHotBasis

        rng = np.random.default_rng(0)
        n = 40
        y = rng.uniform(0.05, 0.95, n)
        X = rng.integers(0, 2, size=(n, 1)).astype(float)
        basis = InteractionBasis(
            BernsteinBasis(order=2, support=(0.0, 1.0)),
            OneHotBasis(K=2),
        )
        cfg = OptimizerConfig(solver="auglag", fixed_params={0: 0.0})
        with pytest.raises(NotImplementedError, match="fixed_params"):
            optimize(basis, y, X=X, config=cfg)
