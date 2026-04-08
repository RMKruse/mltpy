"""Optimisation wrapper for conditional transformation models.

This module contains no mathematical logic — it only orchestrates calls to
:func:`~pymlt.likelihood.negative_log_likelihood` and
:func:`~pymlt.constraints.build_constraints`.

Analogue to R's ``mltoptim.R`` (sequential solver attempts) and the ``maxtry``
restart mechanism in ``mlt()``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pymlt.basis import BernsteinBasis
from pymlt.constraints import build_constraints
from pymlt.likelihood import negative_log_likelihood, _get_dist
from pymlt.variables import CensoredData, CensoringType


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    """Settings for the optimisation run.

    Parameters
    ----------
    solver:
        ``"slsqp"`` (default) or ``"trust-constr"``.
        SLSQP is faster; trust-constr handles ill-conditioned problems better.
    max_iter:
        Maximum number of iterations passed to scipy.
    tol:
        Convergence tolerance.  Mapped to ``ftol`` (SLSQP) or ``gtol``
        (trust-constr).
    max_restarts:
        Number of *additional* attempts after the first one.  Analogous to
        ``maxtry`` in R's ``mlt()``.  On each restart the starting point is
        perturbed and projected back to the feasible region.
    use_gradient:
        If ``True`` (default), the analytical gradient from
        :func:`~pymlt.likelihood.negative_log_likelihood` is passed to scipy.
        Set to ``False`` only for debugging.
    verbose:
        If ``True``, print a warning on each failed attempt.
    """

    solver: Literal["slsqp", "trust-constr"] = "slsqp"
    max_iter: int = 1000
    tol: float = 1e-8
    max_restarts: int = 3
    use_gradient: bool = True
    verbose: bool = False


@dataclass
class OptimizationResult:
    """Result of a :func:`optimize` call.

    Parameters
    ----------
    theta:
        Optimised parameter vector ``[theta_basis | beta]``.
    log_likelihood:
        Log-likelihood value at the optimum (i.e. ``−nll``).
    converged:
        Whether scipy reported successful convergence on at least one attempt.
    n_iter:
        Number of iterations used by the final (best) scipy run.
    n_restarts:
        Number of restarts that were needed (0 if first attempt succeeded).
    solver_message:
        scipy's ``result.message`` from the best attempt, unchanged.
    """

    theta: NDArray[np.float64]
    log_likelihood: float
    converged: bool
    n_iter: int
    n_restarts: int
    solver_message: str


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _project_to_feasible(theta_b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project an arbitrary vector to the monotone-non-decreasing cone.

    Uses the simplest valid projection: ``np.sort``.  The result satisfies
    ``D @ theta_b >= 0`` (the :class:`~pymlt.constraints.MonotonicityConstraint`).

    Parameters
    ----------
    theta_b:
        Bernstein coefficient vector of length p (may be non-monotone).

    Returns
    -------
    NDArray of length p, sorted ascending.
    """
    return cast(NDArray[np.float64], np.sort(theta_b))


def _make_objective(
    basis: BernsteinBasis,
    y: NDArray[np.float64] | CensoredData,
    X: Optional[NDArray[np.float64]],
    censoring: CensoringType,
    use_gradient: bool,
    base_distribution: Literal["normal", "logistic"] = "normal",
) -> Callable[[NDArray[np.float64]], Any]:
    """Return a closure suitable for ``scipy.optimize.minimize``.

    When ``use_gradient=True`` the closure returns ``(nll, grad)`` and
    ``jac=True`` should be passed to scipy.  When False it returns a scalar
    and ``jac=None`` should be used.

    ValueError from infeasible theta (h' ≤ 0) is caught and replaced by a
    large penalty so that the optimiser can back off rather than crash.
    """
    _BIG = 1e10

    if use_gradient:
        def obj(theta: NDArray[np.float64]) -> Any:
            try:
                return negative_log_likelihood(
                    theta, basis, y, X, censoring, gradient=True,
                    base_distribution=base_distribution,
                )
            except ValueError:
                return _BIG, np.zeros_like(theta)
    else:
        def obj(theta: NDArray[np.float64]) -> Any:
            try:
                return negative_log_likelihood(
                    theta, basis, y, X, censoring, gradient=False,
                    base_distribution=base_distribution,
                )
            except ValueError:
                return _BIG

    return obj


def _scipy_options(config: OptimizerConfig) -> dict[str, Any]:
    """Build the ``options`` dict for the chosen solver."""
    if config.solver == "slsqp":
        return {"maxiter": config.max_iter, "ftol": config.tol}
    return {"maxiter": config.max_iter, "gtol": config.tol}


def _initial_theta(n_params: int, X: Optional[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Default starting point: linearly spaced basis coefficients + zero beta.

    ``np.linspace(0, 1, n_params)`` is non-decreasing by construction, so it
    satisfies the monotonicity constraint at the first call.
    """
    theta_b = np.linspace(0.0, 1.0, n_params)
    if X is not None:
        beta = np.zeros(X.shape[1])
        return cast(NDArray[np.float64], np.concatenate([theta_b, beta]))
    return theta_b


def _perturb_and_project(
    theta: NDArray[np.float64],
    n_params: int,
    rng: np.random.Generator,
    scale: float = 0.1,
) -> NDArray[np.float64]:
    """Perturb theta_basis, project back to feasible, keep beta unchanged."""
    theta_b = theta[:n_params] + rng.normal(0.0, scale, size=n_params)
    theta_b = _project_to_feasible(theta_b)
    if len(theta) > n_params:
        return cast(NDArray[np.float64], np.concatenate([theta_b, theta[n_params:]]))
    return theta_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize(
    basis: BernsteinBasis,
    y: NDArray[np.float64] | CensoredData,
    X: Optional[NDArray[np.float64]] = None,
    censoring: CensoringType = CensoringType.NONE,
    config: Optional[OptimizerConfig] = None,
    base_distribution: Literal["normal", "logistic"] = "normal",
) -> OptimizationResult:
    """Fit Bernstein transformation model parameters by maximising log-likelihood.

    Parameters
    ----------
    basis:
        :class:`~pymlt.basis.BernsteinBasis` instance defining the response
        transformation.
    y:
        Observations — plain ``NDArray`` for exact data, or
        :class:`~pymlt.variables.CensoredData` for censored data.
    X:
        Optional covariate matrix, shape (n, q).  If given, the last
        ``q`` entries of the returned ``theta`` are regression coefficients.
    censoring:
        Censoring type; passed through to the likelihood.
    config:
        Optimisation settings.  Defaults to :class:`OptimizerConfig` with
        all defaults.

    Returns
    -------
    :class:`OptimizationResult`
        Contains the optimised parameters, convergence status, and diagnostics.
        If all restarts fail, the best result found so far is returned with
        ``converged=False``.  The caller (``model.py``) decides whether to
        raise or warn.
    """
    _get_dist(base_distribution)  # fail fast; raises ValueError for unsupported values
    if config is None:
        config = OptimizerConfig()

    n_params = basis.order + 1
    total_params = n_params + (X.shape[1] if X is not None else 0)
    constraints = build_constraints(n_params, solver=config.solver, total_params=total_params)
    obj = _make_objective(basis, y, X, censoring, config.use_gradient,
                          base_distribution=base_distribution)
    jac = True if config.use_gradient else None
    options = _scipy_options(config)
    rng = np.random.default_rng()

    theta_init = _initial_theta(n_params, X)

    best_scipy_result = None
    best_nll = float("inf")
    n_restarts_used = 0

    for attempt in range(config.max_restarts + 1):
        if attempt == 0:
            theta_try = theta_init.copy()
        else:
            n_restarts_used = attempt
            theta_try = _perturb_and_project(theta_init, n_params, rng)

        try:
            scipy_result = minimize(
                obj,
                theta_try,
                method=config.solver,
                jac=jac,
                constraints=constraints,
                options=options,
            )
        except Exception as exc:
            if config.verbose:
                warnings.warn(
                    f"optimizer.py: attempt {attempt + 1} raised {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

        if scipy_result.fun < best_nll:
            best_nll = float(scipy_result.fun)
            best_scipy_result = scipy_result

        if scipy_result.success:
            break

        if config.verbose:
            warnings.warn(
                f"optimizer.py: attempt {attempt + 1}/{config.max_restarts + 1} "
                f"did not converge — {scipy_result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

    if best_scipy_result is None:
        # All attempts raised exceptions — return the initial point as fallback
        _nll = cast(float, negative_log_likelihood(
            theta_init, basis, y, X, censoring,
            base_distribution=base_distribution,
        ))
        return OptimizationResult(
            theta=theta_init,
            log_likelihood=float(-_nll),
            converged=False,
            n_iter=0,
            n_restarts=n_restarts_used,
            solver_message="All optimisation attempts raised an exception.",
        )

    return OptimizationResult(
        theta=best_scipy_result.x,
        log_likelihood=float(-best_scipy_result.fun),
        converged=bool(best_scipy_result.success),
        n_iter=int(getattr(best_scipy_result, "nit", 0)),
        n_restarts=n_restarts_used,
        solver_message=str(best_scipy_result.message),
    )
