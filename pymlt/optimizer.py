"""Optimisation wrapper for conditional transformation models.

This module contains no mathematical logic — it only orchestrates calls to
:func:`~pymlt.likelihood.negative_log_likelihood` and
:func:`~pymlt.constraints.build_constraints`.

Analogue to R's ``mltoptim.R`` (sequential solver attempts) and the ``maxtry``
restart mechanism in ``mlt()``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pymlt.basis import BernsteinBasis
from pymlt.constraints import build_constraints
from pymlt.likelihood import BaseDistribution, _get_dist, negative_log_likelihood
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
    theta_b : NDArray[np.float64]
        Bernstein coefficient vector of length p (may be non-monotone).

    Returns
    -------
    NDArray[np.float64]
        NDArray of length p, sorted ascending.
    """
    return cast(NDArray[np.float64], np.sort(theta_b))


def _make_objective(
    basis: BernsteinBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None,
    censoring: CensoringType,
    use_gradient: bool,
    base_distribution: BaseDistribution = "normal",
) -> Callable[[NDArray[np.float64]], Any]:
    """Return a closure suitable for ``scipy.optimize.minimize``.

    When ``use_gradient=True`` the closure returns ``(nll, grad)`` and
    ``jac=True`` should be passed to scipy.  When False it returns a scalar
    and ``jac=None`` should be used.

    ValueError from infeasible theta (h' ≤ 0) is caught and replaced by a
    large penalty so that the optimiser can back off rather than crash.

    Parameters
    ----------
    basis : BernsteinBasis
        Polynomial basis defining the response transformation.
    y : NDArray[np.float64] | CensoredData
        Response observations.
    X : NDArray[np.float64] | None
        Covariate matrix, if any.
    censoring : CensoringType
        Type of censoring for the response variables.
    use_gradient : bool
        If True, return analytical gradients along with the negative log-likelihood.
    base_distribution : BaseDistribution, default="normal"
        The base distribution to estimate transformations against.

    Returns
    -------
    Callable[[NDArray[np.float64]], Any]
        Objective function mapping a parameter vector ``theta`` to a scalar
        negative log-likelihood (and optionally its gradient vector).
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
    """Build the dictionary of configuration options passed directly to
    `scipy.optimize.minimize`.

    Parameters
    ----------
    config : OptimizerConfig
        Object containing settings for the optimization run.

    Returns
    -------
    dict[str, Any]
        A dictionary with optimization solver-specific kwargs.
    """
    if config.solver == "slsqp":
        return {"maxiter": config.max_iter, "ftol": config.tol}
    return {"maxiter": config.max_iter, "gtol": config.tol}


def _initial_theta(n_params: int, X: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """Default starting point: linearly spaced basis coefficients and zero
    beta components.

    Parameters
    ----------
    n_params : int
        Number of Bernstein basis coefficients (p).
    X : NDArray[np.float64] | None
        Covariate matrix, shape `(n, q)`. If covariates exist, their initial weights
        will be zero-initialized and concatenated with the starting theta vector.

    Returns
    -------
    NDArray[np.float64]
        Initial concatenated parameter vector of shape `(p + q,)`.
        ``np.linspace(0, 1, n_params)`` is non-decreasing by construction, therefore
        guaranteeing the constraint is met for the first trial.
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
    """Perturb the current parameter vector randomly and re-project to a feasible state.

    This function adds random Gaussian noise to the Bernstein coefficients (theta_b),
    in order to break out of poor local minima, while leaving the beta
    components (if any) unchanged. The perturbed vector is then projected to
    preserve the monotonic non-decreasing constraints.

    Parameters
    ----------
    theta : NDArray[np.float64]
        The current complete parameter vector `[theta_b | beta]`.
    n_params : int
        The length of the `theta_b` sub-vector.
    rng : np.random.Generator
        Initialized NumPy random number generator for normal sampling.
    scale : float, default=0.1
        Scale (standard deviation) of the normal noise added.

    Returns
    -------
    NDArray[np.float64]
        A valid, non-decreasing parameter vector of identical shape.
    """
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
    base_distribution: BaseDistribution = "normal",
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
    constraints = build_constraints(
        n_params, solver=config.solver, total_params=total_params
    )
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
