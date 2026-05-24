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
from pymlt.likelihood import (
    hessian as _hessian,
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
    polish:
        If ``True`` (default), run a Newton-CG polish step after auglag
        converges when no monotonicity constraints are active (interior-MLE
        fits).  Uses ``trust-ncg`` seeded at auglag's θ-hat with the
        analytical Hessian from :func:`~pymlt.likelihood.hessian`.  The
        polished θ is accepted only when NLL does not increase by more than
        ``1e-12`` and the monotonicity cone is preserved.  Has no effect on
        ``slsqp`` / ``trust-constr`` solvers.
    fixed_params:
        Optional ``{index: value}`` mapping that pins arbitrary entries of
        the full parameter vector ``[theta_b | beta | gamma]`` at the given
        values during optimisation.  Useful for profile likelihood, score
        tests, and nested-model fits.

        * ``solver="auglag"`` (issue #85) — each entry is appended as an
          equality row ``e_i · θ = value`` on the ``C_eq``/``d_eq`` block,
          stacked under any ``lower``/``upper`` rows.  The pin holds to the
          auglag KKT tolerance (~1e-8); the equality row remains visible on
          ``OptimizationResult.constraint_C_eq`` so downstream consumers
          (``vcov(regularize='active')``) see it.
        * ``solver="slsqp"`` / ``"trust-constr"`` (issue #86) — the pinned
          indices are *eliminated* from the optimisation problem entirely:
          scipy sees the smaller free-subvector objective and constraint
          matrix sliced to the free columns.  The pin therefore holds to
          machine precision regardless of solver tolerance.  ``constraint_C_eq``
          is ``None`` on this path (no equality row exists).
        * :class:`~pymlt.basis.InteractionBasis` is not yet supported —
          generalising to ``vec_C(Θ)`` indices needs an explicit ADR
          decision and raises :class:`NotImplementedError`.

        Indices outside ``[0, total_params)`` raise :class:`ValueError`.
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
    polish: bool = True
    fixed_params: dict[int, float] | None = None


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
    rho_final:
        Final penalty parameter ρ from the PHR auglag solver.  ``None``
        for SLSQP / trust-constr fits.
    mu_ineq:
        Final inequality multipliers (≥ 0), shape ``(m_ineq,)``, from the
        auglag solver.  For shift models ``m_ineq = order``; for interaction
        models ``m_ineq = order * q``.  ``None`` for SLSQP / trust-constr.
    lambda_eq:
        Final equality multipliers, shape ``(m_eq,)``, from the auglag
        solver.  Currently always shape ``(0,)`` for all built-in models
        (no equality constraints are imposed).  ``None`` for SLSQP /
        trust-constr fits.
    constraint_A_ineq:
        Inequality constraint matrix used during optimisation, shape
        ``(m_ineq, total_params)``.  Set for auglag fits; ``None`` for
        SLSQP / trust-constr.  Consumed by ``model.fit()`` to stash
        ``_A_ineq_`` for downstream inference (e.g. penalty-augmented
        Hessian in ``vcov(regularize='active')``).
    constraint_C_eq:
        Equality constraint matrix used during optimisation, shape
        ``(m_eq, total_params)``.  Non-``None`` for auglag fits whenever any
        equality row is imposed — by ``lower`` / ``upper`` (boundary pins) or
        by ``fixed_params`` (arbitrary-index pins, issue #85), stacked in
        that order.  ``None`` when no equality constraints were imposed or
        when the solver is not auglag.
    """

    theta: NDArray[np.float64]
    log_likelihood: float
    converged: bool
    n_iter: int
    n_restarts: int
    solver_message: str
    n_outer_iter: int | None = None
    kkt_residual: float | None = None
    rho_final: float | None = None
    mu_ineq: NDArray[np.float64] | None = None
    lambda_eq: NDArray[np.float64] | None = None
    constraint_A_ineq: NDArray[np.float64] | None = None
    constraint_C_eq: NDArray[np.float64] | None = None


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


def _fixed_params_split(
    fixed_params: dict[int, float],
    total_params: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Validate a ``fixed_params`` mapping and split it into index/value arrays.

    Used by the SLSQP / trust-constr branch of :func:`optimize` to reduce the
    problem to its free subvector (issue #86).  The auglag path keeps the
    equality-row treatment introduced in #85 because its ``C_eq`` block is
    consumed by ``vcov(regularize='active')`` downstream — see ADR-less note
    in the SLSQP/trust-constr branch comment.

    Parameters
    ----------
    fixed_params:
        ``{index: value}`` pin mapping.  Caller is expected to gate on falsy
        values (``if config.fixed_params:``) so an empty dict never reaches
        this function.
    total_params:
        Length of the full parameter vector ``[theta_b | beta | gamma]``.

    Returns
    -------
    tuple
        ``(free_idx, fixed_idx, fixed_vals)``:

        * ``free_idx`` — sorted ``np.intp`` indices NOT pinned.
        * ``fixed_idx`` — sorted ``np.intp`` indices that ARE pinned.
        * ``fixed_vals`` — pin values in the same order as ``fixed_idx``.

    Raises
    ------
    ValueError
        If any index lies outside ``[0, total_params)``.  Message contains
        ``"out-of-range"`` to match the auglag-path validation text so a
        single test pattern covers both branches.
    """
    bad = sorted(i for i in fixed_params if not (0 <= i < total_params))
    if bad:
        raise ValueError(
            f"fixed_params indices must lie in [0, {total_params}); "
            f"got out-of-range indices {bad}."
        )
    fixed_idx = np.array(sorted(fixed_params.keys()), dtype=np.intp)
    fixed_vals = np.array([float(fixed_params[i]) for i in fixed_idx], dtype=np.float64)
    free_mask = np.ones(total_params, dtype=bool)
    free_mask[fixed_idx] = False
    free_idx = np.flatnonzero(free_mask).astype(np.intp)
    return free_idx, fixed_idx, fixed_vals


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
        if config.fixed_params:
            # Issue #85 — fixed_params currently lives only on the shift /
            # auglag path.  Generalising to vec_C(Θ) indices needs an
            # explicit decision (column-vs-row layout, Kronecker padding),
            # so we defer until that ADR lands.
            raise NotImplementedError(
                "fixed_params= is not supported with InteractionBasis "
                "(auglag shift-basis path only in this release)."
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

    # fixed_params (issue #86) — reduction approach for the scipy paths.
    # Rather than appending equality constraints (the auglag treatment, #85),
    # the pinned indices are eliminated from the optimisation problem
    # altogether: scipy sees a smaller objective on theta_free and a
    # constraint matrix sliced to the free columns, with the constant
    # contribution of the pinned values absorbed into the right-hand side.
    # The pin then holds to machine precision regardless of solver tolerance,
    # and SLSQP avoids its known mixed-equality+inequality weakness.
    free_idx: NDArray[np.intp] | None = None
    fixed_idx: NDArray[np.intp] | None = None
    fixed_vals: NDArray[np.float64] | None = None
    obj_used: Callable[[NDArray[np.float64]], Any] = obj
    constraints_used: Any = constraints
    if config.fixed_params:
        free_idx, fixed_idx, fixed_vals = _fixed_params_split(
            config.fixed_params, total_params
        )
        theta_init[fixed_idx] = fixed_vals
        _free_idx = free_idx
        _fixed_idx = fixed_idx
        _fixed_vals = fixed_vals

        def _reconstruct(theta_free: NDArray[np.float64]) -> NDArray[np.float64]:
            full = np.empty(total_params, dtype=np.float64)
            full[_free_idx] = theta_free
            full[_fixed_idx] = _fixed_vals
            return full

        if config.use_gradient:

            def obj_used(
                tf: NDArray[np.float64], _o: Any = obj, _fi: Any = _free_idx
            ) -> Any:
                res = _o(_reconstruct(tf))
                if isinstance(res, tuple):
                    nll, grad = res
                    return nll, np.asarray(grad)[_fi]
                return res
        else:

            def obj_used(tf: NDArray[np.float64], _o: Any = obj) -> Any:
                return _o(_reconstruct(tf))

        if config.solver == "slsqp":

            def _wrap_dict(d: dict[str, Any]) -> dict[str, Any]:
                f, j, t = d["fun"], d["jac"], d["type"]
                return {
                    "type": t,
                    "fun": lambda tf, _f=f: _f(_reconstruct(tf)),
                    "jac": lambda tf, _j=j: np.asarray(_j(_reconstruct(tf)))[
                        ..., _free_idx
                    ],
                }

            constraints_used = [_wrap_dict(d) for d in constraints]
        else:  # trust-constr

            def _slice_lc(lc: LinearConstraint) -> LinearConstraint:
                A = np.atleast_2d(np.asarray(lc.A, dtype=np.float64))
                shift = A[:, _fixed_idx] @ _fixed_vals  # shape (m,)
                lb_arr = (
                    np.broadcast_to(
                        np.asarray(lc.lb, dtype=np.float64), shift.shape
                    ).astype(np.float64)
                    - shift
                )
                ub_arr = (
                    np.broadcast_to(
                        np.asarray(lc.ub, dtype=np.float64), shift.shape
                    ).astype(np.float64)
                    - shift
                )
                return LinearConstraint(A[:, _free_idx], lb=lb_arr, ub=ub_arr)

            constraints_used = [_slice_lc(lc) for lc in constraints]

    best_scipy_result = None
    best_nll = float("inf")
    n_restarts_used = 0

    for attempt in range(config.max_restarts + 1):
        if attempt == 0:
            theta_try_full = theta_init.copy()
        else:
            n_restarts_used = attempt
            theta_try_full = _perturb_and_project(
                theta_init,
                n_params,
                rng,
                nonneg_lower=nonneg_lower,
            )
            if fixed_idx is not None:
                # Perturbation projects theta_b onto the monotone cone, which
                # may shift pinned entries.  Re-applying the pins here keeps
                # the reduced theta_try aligned with the constraint slicing
                # (lb/ub were shifted by ``A[:, fixed_idx] @ fixed_vals``).
                theta_try_full[fixed_idx] = fixed_vals

        theta_try = theta_try_full[free_idx] if free_idx is not None else theta_try_full

        try:
            scipy_result = minimize(
                obj_used,
                theta_try,
                method=config.solver,
                jac=jac,
                constraints=constraints_used,
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

    # Reduction path returns the optimum on the free subvector; lift it back
    # to the full layout (fixed entries restored to their pinned values) so
    # downstream consumers see the unchanged ``[theta_b | beta | gamma]``
    # representation regardless of whether fixed_params was used.
    if free_idx is not None and fixed_idx is not None and fixed_vals is not None:
        final_theta = np.empty(total_params, dtype=np.float64)
        final_theta[free_idx] = best_scipy_result.x
        final_theta[fixed_idx] = fixed_vals
    else:
        final_theta = best_scipy_result.x

    return OptimizationResult(
        theta=final_theta,
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
            base_distribution=base_distribution,
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
    base_distribution: BaseDistribution,
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

    # Newton-CG polish: only when all Kronecker-constraint multipliers are inactive
    ia_final_theta = best_result.theta
    ia_final_nll = best_nll
    if config.polish and np.all(best_result.mu_ineq < 1e-6):
        theta_polished = _ncg_polish(
            best_result.theta,
            obj,
            0,
            basis,
            y,
            x_arr,
            CensoringType.NONE,
            base_distribution,
            weights,
            offset,
            None,
            A_ineq=cm.A_ineq,
        )
        if theta_polished is not None:
            ia_final_theta = theta_polished
            ia_final_nll = float(obj(theta_polished)[0])

    ia_c_eq: NDArray[np.float64] | None = cm.C_eq if cm.C_eq.shape[0] > 0 else None
    return OptimizationResult(
        theta=ia_final_theta,
        log_likelihood=float(-ia_final_nll),
        converged=best_result.converged,
        n_iter=best_result.n_inner_iter,
        n_restarts=n_restarts_used,
        solver_message=best_result.message,
        n_outer_iter=best_result.n_outer_iter,
        kkt_residual=best_result.kkt_residual,
        rho_final=best_result.rho_final,
        mu_ineq=best_result.mu_ineq,
        lambda_eq=best_result.lambda_eq,
        constraint_A_ineq=cm.A_ineq,
        constraint_C_eq=ia_c_eq,
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


def _ncg_polish(
    theta: NDArray[np.float64],
    obj: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    n_params: int,
    basis: BernsteinBasis | InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None,
    censoring: CensoringType,
    base_distribution: BaseDistribution,
    weights: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
    scaling: NDArray[np.float64] | None,
    *,
    A_ineq: NDArray[np.float64] | None = None,
) -> NDArray[np.float64] | None:
    """Newton-CG polish: trust-ncg from auglag's θ-hat using the analytical Hessian.

    Parameters
    ----------
    theta:
        Starting point (auglag's best θ).
    obj:
        NLL objective returning ``(nll, grad)``.
    n_params:
        Number of Bernstein basis coefficients (shift path) or ``None``-equivalent
        when ``A_ineq`` is provided (interaction path uses A_ineq for feasibility).
    A_ineq:
        Optional inequality constraint matrix.  When provided, feasibility is
        checked via ``A_ineq @ theta_polished >= 0`` instead of the shift-model
        ``D @ theta_b >= 0`` check.

    Returns
    -------
    NDArray[np.float64] or None
        Polished parameter vector, or ``None`` if polish failed or was rejected.
    """
    nll_before, _ = obj(theta)

    def hess_fn(t: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            return _hessian(
                t,
                basis,
                y,
                X,
                censoring,
                base_distribution,
                weights=weights,
                offset=offset,
                scaling=scaling,
            )
        except (InfeasibleParameterError, Exception):
            return np.eye(len(t), dtype=np.float64) * 1e6

    try:
        result = minimize(obj, theta, method="trust-ncg", jac=True, hess=hess_fn)
    except Exception:
        return None

    theta_new = np.asarray(result.x, dtype=np.float64)
    nll_new, _ = obj(theta_new)

    if nll_new > nll_before + 1e-12:
        return None

    if A_ineq is not None:
        if np.any(A_ineq @ theta_new < -1e-8):
            return None
    elif n_params >= 2 and np.any(np.diff(theta_new[:n_params]) < -1e-8):
        return None

    return theta_new


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
    # fixed_params (issue #85): append one equality row per pinned index.
    # Each row is e_i with rhs v_i, stacked under any existing rows from
    # lower / upper.  Indices must reference the full theta_ vector
    # [theta_b | beta | gamma] of length total_params.  The starting point
    # is overwritten on those indices so the very first inner iteration
    # already satisfies the new equalities (cuts down auglag's outer-loop
    # multiplier-update count).
    if config.fixed_params:
        bad = [i for i in config.fixed_params if not (0 <= i < total_params)]
        if bad:
            raise ValueError(
                f"fixed_params indices must lie in [0, {total_params}); "
                f"got out-of-range indices {bad}."
            )
        extra_rows = np.zeros((len(config.fixed_params), total_params))
        extra_rhs = np.empty(len(config.fixed_params), dtype=np.float64)
        for row, (idx, val) in enumerate(config.fixed_params.items()):
            extra_rows[row, idx] = 1.0
            extra_rhs[row] = float(val)
            theta_init[idx] = float(val)
        cm = type(cm)(
            A_ineq=cm.A_ineq,
            b_ineq=cm.b_ineq,
            C_eq=np.vstack([cm.C_eq, extra_rows]),
            d_eq=np.concatenate([cm.d_eq, extra_rhs]),
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

    # Newton-CG polish: only for interior-MLE fits (no active inequalities, no
    # equality constraints — the unconstrained trust-ncg would violate lower/upper
    # boundary pins if cm.C_eq has rows).
    final_theta = best_auglag_result.theta
    final_nll = best_nll
    if (
        config.polish
        and cm.C_eq.shape[0] == 0
        and np.all(best_auglag_result.mu_ineq < 1e-6)
    ):
        theta_polished = _ncg_polish(
            best_auglag_result.theta,
            obj,
            n_params,
            basis,
            y,
            X,
            censoring,
            base_distribution,
            weights,
            offset,
            scaling,
        )
        if theta_polished is not None:
            final_theta = theta_polished
            final_nll = float(obj(theta_polished)[0])

    c_eq: NDArray[np.float64] | None = cm.C_eq if cm.C_eq.shape[0] > 0 else None
    return OptimizationResult(
        theta=final_theta,
        log_likelihood=float(-final_nll),
        converged=best_auglag_result.converged,
        n_iter=best_auglag_result.n_inner_iter,
        n_restarts=n_restarts_used,
        solver_message=best_auglag_result.message,
        n_outer_iter=best_auglag_result.n_outer_iter,
        kkt_residual=best_auglag_result.kkt_residual,
        rho_final=best_auglag_result.rho_final,
        mu_ineq=best_auglag_result.mu_ineq,
        lambda_eq=best_auglag_result.lambda_eq,
        constraint_A_ineq=cm.A_ineq,
        constraint_C_eq=c_eq,
    )
