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
from typing import Any, Callable, Literal, cast

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize

from pymlt._auglag import AugLagOptions, AugLagResult, auglag_minimize
from pymlt.basis import BernsteinBasis, InteractionBasis
from pymlt.constraints import (
    build_constraint_matrices,
    build_constraint_matrices_interaction,
    build_constraints,
)
from pymlt.likelihood import (
    BaseDistribution,
    DistOps,
    InfeasibleParameterError,
    _get_dist,
    _negative_log_likelihood_from_dist,
)
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
        ``"auglag"`` (default), ``"slsqp"``, or ``"trust-constr"``.  Auglag is
        the PHR augmented Lagrangian (matches R ``mlt`` / ``alabama::auglag``
        and gives the best parity with the reference implementation).  SLSQP
        and trust-constr remain opt-in alternatives — SLSQP is faster on
        easy problems, trust-constr handles ill-conditioned ones better.
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
    random_state:
        If an ``int``, seeds the RNG used to perturb restart starting points
        so repeated fits with the same config and data are bit-identical.
        If a :class:`numpy.random.Generator`, it is used directly.
        If ``None`` (default), draws are non-reproducible across runs.
    auglag_options:
        :class:`~pymlt._auglag.AugLagOptions` controlling the PHR outer loop.
        Only consulted when ``solver="auglag"``; ignored otherwise.  ``None``
        (default) uses :class:`AugLagOptions` defaults (alabama parity).
    lower:
        If not ``None``, fixes ``θ[0] = lower`` as an equality constraint
        (pins the lower-boundary Bernstein coefficient).  Honoured by every
        solver: passes through to :func:`~pymlt.constraints.build_constraints`
        for SLSQP/trust-constr and
        :func:`~pymlt.constraints.build_constraint_matrices` for auglag.
    upper:
        If not ``None``, fixes ``θ[n_params−1] = upper`` analogously.
    """

    solver: Literal["auglag", "slsqp", "trust-constr"] = "auglag"
    max_iter: int = 1000
    tol: float = 1e-8
    max_restarts: int = 3
    use_gradient: bool = True
    verbose: bool = False
    random_state: int | np.random.Generator | None = None
    auglag_options: AugLagOptions | None = None
    lower: float | None = None
    upper: float | None = None


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
    n_outer_iter:
        Number of PHR outer iterations the auglag solver used.  ``None``
        for SLSQP / trust-constr fits (those solvers have no outer loop).
        Appears in ``repr()`` so users can see at a glance how many
        Lagrange-multiplier updates the fit required.
    kkt_residual:
        Final KKT residual reported by the auglag solver — the
        ``max(‖h(θ)‖∞, ‖min(g(θ), μ/ρ)‖∞, ‖∇L_A(θ)‖∞)`` value at the
        returned ``theta``.  ``None`` for SLSQP / trust-constr fits.
        Useful when ``converged=False`` to judge how close the run got
        before exhausting its outer-iteration budget.
    """

    theta: NDArray[np.float64]
    log_likelihood: float
    converged: bool
    n_iter: int
    n_restarts: int
    solver_message: str
    n_outer_iter: int | None = None
    kkt_residual: float | None = None


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
    *,
    dist: DistOps | None = None,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> Callable[[NDArray[np.float64]], Any]:
    """Return a closure suitable for ``scipy.optimize.minimize``.

    When ``use_gradient=True`` the closure returns ``(nll, grad)`` and
    ``jac=True`` should be passed to scipy.  When False it returns a scalar
    and ``jac=None`` should be used.

    :class:`~pymlt.likelihood.InfeasibleParameterError` from an infeasible
    ``theta`` (h' ≤ 0) is caught and replaced by a large penalty so that the
    optimiser can back off rather than crash.  Any other exception — including
    plain ``ValueError`` for unsupported ``base_distribution`` or shape
    mismatches — propagates out so that genuine bugs are not silenced.

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
    dist : DistOps | None, default=None
        Optional pre-resolved distribution wrapper. When provided,
        objective evaluations reuse it instead of re-dispatching from
        ``base_distribution`` on every call.
    weights : NDArray[np.float64] | None, default=None
        Per-observation weights (already validated by the caller).
    offset : NDArray[np.float64] | None, default=None
        Per-observation offsets added to ``h`` before distribution calls
        (already validated by the caller).

    Returns
    -------
    Callable[[NDArray[np.float64]], Any]
        Objective function mapping a parameter vector ``theta`` to a scalar
        negative log-likelihood (and optionally its gradient vector).
    """
    _BIG = 1e10
    resolved_dist = _get_dist(base_distribution) if dist is None else dist
    n_params = basis.order + 1
    # Forward-difference matrix D: monotonicity is D @ theta_b >= 0.  Identical
    # to MonotonicityConstraint(n_params).as_matrix(); built inline to avoid a
    # cross-module coupling in the hot path.  Shape (n_params-1, n_params);
    # empty when n_params < 2 (order=0 — monotonicity is vacuous).
    _D = np.diff(np.eye(n_params), axis=0) if n_params >= 2 else None

    if use_gradient:

        def obj(theta: NDArray[np.float64]) -> Any:
            try:
                return _negative_log_likelihood_from_dist(
                    theta,
                    basis,
                    y,
                    X,
                    censoring,
                    gradient=True,
                    dist=resolved_dist,
                    weights=weights,
                    offset=offset,
                    scaling=scaling,
                )
            except InfeasibleParameterError:
                # Subgradient of the quadratic monotonicity-violation penalty
                # P(theta_b) = 0.5 · ||max(0, -(D @ theta_b))||².
                # Gives SLSQP a descent direction toward the monotone cone
                # instead of the stationary-point signal a zero gradient
                # conveys.  Magnitude is left unscaled (natural units: adjacent
                # theta_b differences) — pairing _BIG with a huge gradient
                # would misrepresent the local slope.  Beta block stays zero
                # because beta does not enter the monotonicity constraint.
                grad = np.zeros_like(theta)
                if _D is not None:
                    g = _D @ theta[:n_params]
                    grad[:n_params] = _D.T @ np.minimum(g, 0.0)
                return _BIG, grad
    else:

        def obj(theta: NDArray[np.float64]) -> Any:
            try:
                return _negative_log_likelihood_from_dist(
                    theta,
                    basis,
                    y,
                    X,
                    censoring,
                    gradient=False,
                    dist=resolved_dist,
                    weights=weights,
                    offset=offset,
                    scaling=scaling,
                )
            except InfeasibleParameterError:
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


def _initial_theta(
    n_params: int,
    X: NDArray[np.float64] | None,
    lower: float | None = None,
    upper: float | None = None,
    q_s: int = 0,
) -> NDArray[np.float64]:
    """Default starting point: linearly spaced basis coefficients and zero
    beta components.

    Parameters
    ----------
    n_params : int
        Number of Bernstein basis coefficients (p).
    X : NDArray[np.float64] | None
        Covariate matrix, shape `(n, q)`. If covariates exist, their initial weights
        will be zero-initialized and concatenated with the starting theta vector.
    lower : float | None, default=None
        When provided alongside ``upper``, the basis coefficients are seeded
        from ``np.linspace(lower, upper, n_params)`` so the start already
        satisfies both boundary-equality constraints (``θ[0] = lower``,
        ``θ[n_params-1] = upper``).  When only one side is pinned, the start
        is shifted so that side matches but the spacing is unchanged.
    upper : float | None, default=None
        See ``lower``.

    Returns
    -------
    NDArray[np.float64]
        Initial concatenated parameter vector of shape `(p + q,)`.
        Always non-decreasing by construction, so the monotonicity constraint
        is satisfied at the first trial.  When ``lower`` / ``upper`` are
        provided the starting point also satisfies the boundary equality
        constraints, sparing the augmented-Lagrangian penalty an unnecessary
        initial gradient swing.
    """
    if lower is not None and upper is not None:
        theta_b = np.linspace(float(lower), float(upper), n_params)
    elif lower is not None:
        theta_b = np.linspace(0.0, 1.0, n_params) + (float(lower) - 0.0)
    elif upper is not None:
        theta_b = np.linspace(0.0, 1.0, n_params) + (float(upper) - 1.0)
    else:
        theta_b = np.linspace(0.0, 1.0, n_params)
    parts: list[NDArray[np.float64]] = [theta_b]
    if X is not None:
        parts.append(np.zeros(X.shape[1]))
    if q_s > 0:
        # γ = 0 ⇒ exp(X_s · γ) = 1, so the scaled likelihood collapses to the
        # shift likelihood at the start — the optimiser walks γ away from zero
        # only if the data carry heteroskedastic signal (ADR 0002, Decision 3).
        parts.append(np.zeros(q_s))
    if len(parts) == 1:
        return theta_b
    return cast(NDArray[np.float64], np.concatenate(parts))


def _perturb_and_project(
    theta: NDArray[np.float64],
    n_params: int,
    rng: np.random.Generator,
    scale: float = 0.1,
    nonneg_lower: bool = False,
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
    nonneg_lower : bool, default=False
        If True, shift the projected coefficients so ``theta_b[0] >= 0``
        (needed when ``base_distribution="exponential"``).

    Returns
    -------
    NDArray[np.float64]
        A valid, non-decreasing parameter vector of identical shape.
    """
    theta_b = theta[:n_params] + rng.normal(0.0, scale, size=n_params)
    theta_b = _project_to_feasible(theta_b)
    if nonneg_lower and theta_b[0] < 0.0:
        theta_b = theta_b - theta_b[0]
    if len(theta) > n_params:
        return cast(NDArray[np.float64], np.concatenate([theta_b, theta[n_params:]]))
    return theta_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize(
    basis: BernsteinBasis | InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    config: OptimizerConfig | None = None,
    base_distribution: BaseDistribution = "normal",
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
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
    weights:
        Optional per-observation weights, shape ``(n,)``.  Passed unchanged
        to the likelihood; no normalisation is applied.
    offset:
        Optional per-observation offset, shape ``(n,)``.  Added to ``h``
        before distribution calls on every likelihood evaluation.

    Returns
    -------
    :class:`OptimizationResult`
        Contains the optimised parameters, convergence status, and diagnostics.
        If all restarts fail, the best result found so far is returned with
        ``converged=False``.  The caller (``model.py``) decides whether to
        raise or warn.
    """
    # Fail fast for unsupported base distributions before entering scipy.
    dist = _get_dist(base_distribution)
    if config is None:
        config = OptimizerConfig()

    if isinstance(basis, InteractionBasis):
        if scaling is not None:
            raise NotImplementedError(
                "scaling= is not supported with InteractionBasis in v0.4 "
                "(see docs/adr/0002-scaling-terms.md, Decision 2)."
            )
        return _optimize_interaction(
            basis=basis,
            y=y,
            X=X,
            config=config,
            dist=dist,
            base_distribution=base_distribution,
            weights=weights,
            offset=offset,
        )

    q_s = scaling.shape[1] if scaling is not None else 0
    if scaling is not None and base_distribution == "exponential":
        # ADR 0002, Decision 3 — combining scaling with the exponential link
        # would turn the support-feasibility row into a non-linear constraint
        # that the current constraint scaffolding does not support.
        raise NotImplementedError(
            "scaling= is not supported with base_distribution='exponential' "
            "(see docs/adr/0002-scaling-terms.md, Decision 3)."
        )
    n_params = basis.order + 1
    total_params = n_params + (X.shape[1] if X is not None else 0) + q_s
    nonneg_lower = base_distribution == "exponential"

    if isinstance(config.random_state, np.random.Generator):
        rng = config.random_state
    else:
        rng = np.random.default_rng(config.random_state)

    theta_init = _initial_theta(
        n_params, X, lower=config.lower, upper=config.upper, q_s=q_s
    )

    if config.solver == "auglag":
        return _optimize_auglag(
            basis=basis,
            y=y,
            X=X,
            censoring=censoring,
            config=config,
            dist=dist,
            base_distribution=base_distribution,
            weights=weights,
            offset=offset,
            n_params=n_params,
            total_params=total_params,
            nonneg_lower=nonneg_lower,
            rng=rng,
            theta_init=theta_init,
            scaling=scaling,
        )

    # ------------------------------------------------------------------
    # SLSQP / trust-constr path
    # ------------------------------------------------------------------
    # Exponential has support [0, ∞); enforce h(y|x) >= 0.  Without covariates
    # this reduces to theta_b[0] >= 0; with covariates we add one linear
    # inequality per training row: theta_b[0] + X_i @ beta >= 0.
    constraints = build_constraints(
        n_params,
        lower=config.lower,
        upper=config.upper,
        solver=config.solver,
        total_params=total_params,
        nonneg_lower=nonneg_lower,
        X=X if nonneg_lower else None,
    )
    obj = _make_objective(
        basis,
        y,
        X,
        censoring,
        config.use_gradient,
        base_distribution=base_distribution,
        dist=dist,
        weights=weights,
        offset=offset,
        scaling=scaling,
    )
    jac = True if config.use_gradient else None
    options = _scipy_options(config)

    best_scipy_result = None
    best_nll = float("inf")
    n_restarts_used = 0

    for attempt in range(config.max_restarts + 1):
        if attempt == 0:
            theta_try = theta_init.copy()
        else:
            n_restarts_used = attempt
            theta_try = _perturb_and_project(
                theta_init,
                n_params,
                rng,
                nonneg_lower=nonneg_lower,
            )

        try:
            scipy_result = minimize(
                obj,
                theta_try,
                method=config.solver,
                jac=jac,
                constraints=constraints,
                options=options,
            )
        except LinAlgError as exc:
            if config.verbose:
                warnings.warn(
                    f"optimizer.py: attempt {attempt + 1} hit {exc!r}; retrying",
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
        # Every attempt hit a numerical linear-algebra failure — return the
        # initial point as fallback so the caller can surface a
        # ConvergenceWarning rather than re-raising.
        _nll = cast(
            float,
            _negative_log_likelihood_from_dist(
                theta_init,
                basis,
                y,
                X,
                censoring,
                gradient=False,
                dist=dist,
                weights=weights,
                offset=offset,
                scaling=scaling,
            ),
        )
        return OptimizationResult(
            theta=theta_init,
            log_likelihood=float(-_nll),
            converged=False,
            n_iter=0,
            n_restarts=n_restarts_used,
            solver_message="All optimisation attempts raised LinAlgError.",
        )

    return OptimizationResult(
        theta=best_scipy_result.x,
        log_likelihood=float(-best_scipy_result.fun),
        converged=bool(best_scipy_result.success),
        n_iter=int(getattr(best_scipy_result, "nit", 0)),
        n_restarts=n_restarts_used,
        solver_message=str(best_scipy_result.message),
    )


def _optimize_interaction(
    *,
    basis: InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None,
    config: OptimizerConfig,
    dist: DistOps,
    base_distribution: BaseDistribution,
    weights: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
) -> OptimizationResult:
    """Optimisation path for InteractionBasis (stratified / fully-interacting CTM).

    Dispatches on ``config.solver`` and uses the Kronecker monotonicity
    constraint ``(D ⊗ I_q) @ vec(Θ) ≥ 0`` (built once via
    :func:`~pymlt.constraints.build_constraint_matrices_interaction`) in the
    form each solver expects.  X is required — it cannot be ``None`` for an
    interaction model.
    """
    if X is None:
        raise ValueError(
            "InteractionBasis.fit() requires X (covariate matrix). "
            "Pass the stratum labels as X."
        )
    x_arr = basis._coerce_x(X)

    cm = build_constraint_matrices_interaction(basis)

    p = basis.n_y_params
    q = basis.n_x_params
    total_params = p * q

    Theta_init = np.outer(np.linspace(0.0, 1.0, p), np.ones(q))
    theta_init = Theta_init.ravel().copy()

    if isinstance(config.random_state, np.random.Generator):
        rng = config.random_state
    else:
        rng = np.random.default_rng(config.random_state)

    def obj(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        try:
            result = _negative_log_likelihood_from_dist(
                theta,
                basis,
                y,
                x_arr,
                CensoringType.NONE,
                gradient=True,
                dist=dist,
                weights=weights,
                offset=offset,
            )
            return cast(tuple[float, NDArray[np.float64]], result)
        except InfeasibleParameterError:
            return float("inf"), np.zeros(total_params)

    def perturb(theta_seed: NDArray[np.float64]) -> NDArray[np.float64]:
        Theta_try = theta_seed.reshape(p, q) + rng.normal(0.0, 0.1, size=(p, q))
        for j in range(q):
            Theta_try[:, j] = np.maximum.accumulate(Theta_try[:, j])
        return cast(NDArray[np.float64], Theta_try.ravel())

    if config.solver == "auglag":
        return _interaction_auglag(
            obj=obj,
            theta_init=theta_init,
            cm=cm,
            config=config,
            basis=basis,
            y=y,
            x_arr=x_arr,
            dist=dist,
            weights=weights,
            offset=offset,
            perturb=perturb,
        )

    return _interaction_scipy(
        obj=obj,
        theta_init=theta_init,
        cm=cm,
        config=config,
        basis=basis,
        y=y,
        x_arr=x_arr,
        dist=dist,
        weights=weights,
        offset=offset,
        perturb=perturb,
    )


def _interaction_auglag(
    *,
    obj: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    theta_init: NDArray[np.float64],
    cm: Any,
    config: OptimizerConfig,
    basis: InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    x_arr: NDArray[np.float64],
    dist: DistOps,
    weights: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
    perturb: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> OptimizationResult:
    """PHR augmented-Lagrangian path for the interaction basis."""
    auglag_opts = (
        config.auglag_options if config.auglag_options is not None else AugLagOptions()
    )

    best_result: AugLagResult | None = None
    best_nll = float("inf")
    n_restarts_used = 0

    for attempt in range(config.max_restarts + 1):
        theta_try = theta_init.copy() if attempt == 0 else perturb(theta_init)
        if attempt > 0:
            n_restarts_used = attempt

        try:
            result = auglag_minimize(
                obj,
                theta_try,
                A_ineq=cm.A_ineq,
                b_ineq=cm.b_ineq,
                C_eq=cm.C_eq,
                d_eq=cm.d_eq,
                options=auglag_opts,
            )
        except LinAlgError as exc:
            if config.verbose:
                warnings.warn(
                    f"optimizer.py (interaction): attempt {attempt + 1} "
                    f"hit {exc!r}; retrying",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result

        if result.converged:
            break

    if best_result is None:
        _nll = cast(
            float,
            _negative_log_likelihood_from_dist(
                theta_init,
                basis,
                y,
                x_arr,
                CensoringType.NONE,
                gradient=False,
                dist=dist,
                weights=weights,
                offset=offset,
            ),
        )
        return OptimizationResult(
            theta=theta_init,
            log_likelihood=float(-_nll),
            converged=False,
            n_iter=0,
            n_restarts=n_restarts_used,
            solver_message="All interaction-model auglag attempts raised LinAlgError.",
            n_outer_iter=None,
            kkt_residual=None,
        )

    return OptimizationResult(
        theta=best_result.theta,
        log_likelihood=float(-best_nll),
        converged=best_result.converged,
        n_iter=best_result.n_inner_iter,
        n_restarts=n_restarts_used,
        solver_message=best_result.message,
        n_outer_iter=best_result.n_outer_iter,
        kkt_residual=best_result.kkt_residual,
    )


def _interaction_scipy(
    *,
    obj: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    theta_init: NDArray[np.float64],
    cm: Any,
    config: OptimizerConfig,
    basis: InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    x_arr: NDArray[np.float64],
    dist: DistOps,
    weights: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
    perturb: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> OptimizationResult:
    """SLSQP / trust-constr path for the interaction basis.

    Both methods consume the same ``A_ineq`` block-diagonal matrix that
    :func:`~pymlt.constraints.build_constraint_matrices_interaction` builds —
    SLSQP via a ``type='ineq'`` dict and trust-constr via a
    :class:`scipy.optimize.LinearConstraint` — without code changes anywhere
    else.
    """
    A_ineq = cm.A_ineq

    if config.solver == "slsqp":
        constraints: Any = [
            {
                "type": "ineq",
                "fun": lambda theta, _A=A_ineq: _A @ theta,
                "jac": lambda theta, _A=A_ineq: _A,
            }
        ]
    else:  # trust-constr
        constraints = [LinearConstraint(A_ineq, lb=0.0, ub=np.inf)]

    options = _scipy_options(config)
    best_scipy_result = None
    best_nll = float("inf")
    n_restarts_used = 0

    for attempt in range(config.max_restarts + 1):
        theta_try = theta_init.copy() if attempt == 0 else perturb(theta_init)
        if attempt > 0:
            n_restarts_used = attempt
        try:
            scipy_result = minimize(
                obj,
                theta_try,
                method=config.solver,
                jac=True,
                constraints=constraints,
                options=options,
            )
        except LinAlgError as exc:
            if config.verbose:
                warnings.warn(
                    f"optimizer.py (interaction): attempt {attempt + 1} "
                    f"hit {exc!r}; retrying",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

        if scipy_result.fun < best_nll:
            best_nll = float(scipy_result.fun)
            best_scipy_result = scipy_result

        if scipy_result.success:
            break

    if best_scipy_result is None:
        _nll = cast(
            float,
            _negative_log_likelihood_from_dist(
                theta_init,
                basis,
                y,
                x_arr,
                CensoringType.NONE,
                gradient=False,
                dist=dist,
                weights=weights,
                offset=offset,
            ),
        )
        return OptimizationResult(
            theta=theta_init,
            log_likelihood=float(-_nll),
            converged=False,
            n_iter=0,
            n_restarts=n_restarts_used,
            solver_message=("All interaction-model scipy attempts raised LinAlgError."),
        )

    return OptimizationResult(
        theta=best_scipy_result.x,
        log_likelihood=float(-best_scipy_result.fun),
        converged=bool(best_scipy_result.success),
        n_iter=int(getattr(best_scipy_result, "nit", 0)),
        n_restarts=n_restarts_used,
        solver_message=str(best_scipy_result.message),
    )


def _optimize_auglag(
    *,
    basis: BernsteinBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None,
    censoring: CensoringType,
    config: OptimizerConfig,
    dist: DistOps,
    base_distribution: BaseDistribution,
    weights: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
    n_params: int,
    total_params: int,
    nonneg_lower: bool,
    rng: np.random.Generator,
    theta_init: NDArray[np.float64],
    scaling: NDArray[np.float64] | None = None,
) -> OptimizationResult:
    """Augmented Lagrangian optimisation path (PHR).

    Mirrors the restart logic of the SLSQP path and reuses ``_make_objective``
    so ``InfeasibleParameterError`` handling is inherited.  The inner solver
    always uses gradients regardless of ``config.use_gradient`` because
    L-BFGS-B requires them.
    """
    cm = build_constraint_matrices(
        n_params,
        lower=config.lower,
        upper=config.upper,
        total_params=total_params,
        nonneg_lower=nonneg_lower,
        X=X if nonneg_lower else None,
    )
    # auglag always needs gradients for the L-BFGS-B inner solver
    obj = _make_objective(
        basis,
        y,
        X,
        censoring,
        use_gradient=True,
        base_distribution=base_distribution,
        dist=dist,
        weights=weights,
        offset=offset,
        scaling=scaling,
    )
    auglag_opts = (
        config.auglag_options if config.auglag_options is not None else AugLagOptions()
    )

    best_auglag_result: AugLagResult | None = None
    best_nll = float("inf")
    n_restarts_used = 0

    for attempt in range(config.max_restarts + 1):
        if attempt == 0:
            theta_try = theta_init.copy()
        else:
            n_restarts_used = attempt
            theta_try = _perturb_and_project(
                theta_init,
                n_params,
                rng,
                nonneg_lower=nonneg_lower,
            )

        try:
            result = auglag_minimize(
                obj,
                theta_try,
                A_ineq=cm.A_ineq,
                b_ineq=cm.b_ineq,
                C_eq=cm.C_eq,
                d_eq=cm.d_eq,
                options=auglag_opts,
            )
        except LinAlgError as exc:
            if config.verbose:
                warnings.warn(
                    f"optimizer.py: attempt {attempt + 1} hit {exc!r}; retrying",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

        if result.fun < best_nll:
            best_nll = result.fun
            best_auglag_result = result

        if result.converged:
            break

        if config.verbose:
            warnings.warn(
                f"optimizer.py: attempt {attempt + 1}/{config.max_restarts + 1} "
                f"did not converge — {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

    if best_auglag_result is None:
        _nll = cast(
            float,
            _negative_log_likelihood_from_dist(
                theta_init,
                basis,
                y,
                X,
                censoring,
                gradient=False,
                dist=dist,
                weights=weights,
                offset=offset,
                scaling=scaling,
            ),
        )
        return OptimizationResult(
            theta=theta_init,
            log_likelihood=float(-_nll),
            converged=False,
            n_iter=0,
            n_restarts=n_restarts_used,
            solver_message="All auglag attempts raised LinAlgError.",
            n_outer_iter=None,
            kkt_residual=None,
        )

    return OptimizationResult(
        theta=best_auglag_result.theta,
        log_likelihood=float(-best_nll),
        converged=best_auglag_result.converged,
        n_iter=best_auglag_result.n_inner_iter,
        n_restarts=n_restarts_used,
        solver_message=best_auglag_result.message,
        n_outer_iter=best_auglag_result.n_outer_iter,
        kkt_residual=best_auglag_result.kkt_residual,
    )
