"""Tests for pymlt.optimizer — convergence, feasibility, result structure."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pymlt.basis import BernsteinBasis
from pymlt.constraints import MonotonicityConstraint
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
        assert cfg.solver == "slsqp"
        assert cfg.max_iter == 1000
        assert cfg.tol == 1e-8
        assert cfg.max_restarts == 3
        assert cfg.use_gradient is True
        assert cfg.verbose is False

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


class TestOptimizeConvergence:
    def test_converges_on_simple_data(self):
        basis = make_basis(order=3)
        y = simple_data(n=80)
        result = optimize(basis, y)
        assert result.converged, f"Did not converge: {result.solver_message}"

    def test_theta_is_non_decreasing(self):
        basis = make_basis(order=4)
        y = simple_data(n=80)
        result = optimize(basis, y)
        n_params = basis.order + 1
        theta_b = result.theta[:n_params]
        assert np.all(np.diff(theta_b) >= -1e-6), f"theta not non-decreasing: {theta_b}"

    def test_nll_decreased_from_init(self):
        """Optimised NLL ≤ initial NLL."""
        from pymlt.likelihood import negative_log_likelihood

        basis = make_basis(order=3)
        y = simple_data(n=60)
        theta_init = _initial_theta(basis.order + 1, None)
        nll_init = negative_log_likelihood(theta_init, basis, y)
        result = optimize(basis, y)
        assert -result.log_likelihood <= nll_init + 1e-6

    def test_monotonicity_constraint_satisfied(self):
        """D @ theta_b >= 0 for the optimised parameters."""
        basis = make_basis(order=5)
        y = simple_data(n=100)
        result = optimize(basis, y)
        D = MonotonicityConstraint(basis.order + 1).as_matrix()
        violations = D @ result.theta[: basis.order + 1]
        assert np.all(violations >= -1e-6), (
            f"Constraint violated: {violations.min():.2e}"
        )

    def test_log_likelihood_is_finite(self):
        basis = make_basis(order=3)
        y = simple_data(n=50)
        result = optimize(basis, y)
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
    """`OptimizerConfig.random_state` controls the restart-perturbation RNG."""

    def _run(self, random_state):
        cfg = OptimizerConfig(
            max_iter=1, max_restarts=3, random_state=random_state, verbose=False
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
            optimize(make_basis(), simple_data(), base_distribution="cauchy")

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
        np.testing.assert_array_equal(grad, np.zeros_like(theta_bad))

    def test_infeasible_theta_nograd_returns_penalty(self):
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        y = np.array([0.3, 0.5, 0.7])
        obj = _make_objective(basis, y, None, CensoringType.NONE, use_gradient=False)
        theta_bad = np.array([10.0, 5.0, 0.0, -5.0])
        val = obj(theta_bad)
        assert val == 1e10


# ---------------------------------------------------------------------------
# optimize() — scipy-exception / all-fail fallback
# ---------------------------------------------------------------------------


class TestOptimizeExceptionFallback:
    def test_linalg_error_retries_and_falls_back(self, monkeypatch):
        """Every scipy call raises LinAlgError → fallback returns initial theta."""
        from numpy.linalg import LinAlgError

        def boom(*args, **kwargs):
            raise LinAlgError("singular matrix")

        monkeypatch.setattr("pymlt.optimizer.minimize", boom)

        cfg = OptimizerConfig(max_restarts=2)
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

        cfg = OptimizerConfig(max_restarts=2)
        with pytest.raises(RuntimeError, match="scipy blew up"):
            optimize(make_basis(order=3), simple_data(), config=cfg)

    def test_linalg_error_with_verbose_warns(self, monkeypatch):
        """verbose=True → RuntimeWarning emitted on each LinAlgError retry."""
        from numpy.linalg import LinAlgError

        def boom(*args, **kwargs):
            raise LinAlgError("singular matrix")

        monkeypatch.setattr("pymlt.optimizer.minimize", boom)

        cfg = OptimizerConfig(max_restarts=0, verbose=True)
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
        """Tight iter budget → scipy returns success=False → verbose warning."""
        cfg = OptimizerConfig(max_iter=1, max_restarts=3, verbose=True)
        with pytest.warns(RuntimeWarning, match="did not converge"):
            optimize(make_basis(order=3), simple_data(n=80), config=cfg)
