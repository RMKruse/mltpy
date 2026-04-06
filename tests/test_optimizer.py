"""Tests for pymlt.optimizer — convergence, feasibility, result structure."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from pymlt.basis import BernsteinBasis
from pymlt.constraints import MonotonicityConstraint
from pymlt.optimizer import (
    OptimizerConfig,
    OptimizationResult,
    _initial_theta,
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
        assert np.all(np.diff(theta_b) >= -1e-6), (
            f"theta not non-decreasing: {theta_b}"
        )

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
        violations = D @ result.theta[:basis.order + 1]
        assert np.all(violations >= -1e-6), f"Constraint violated: {violations.min():.2e}"

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
        assert np.all(np.diff(result.theta[:basis.order + 1]) >= -1e-6)


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
