"""Optimisation wrapper for conditional transformation models.

This module contains no mathematical logic — it only orchestrates calls to
:func:`~mltpy.likelihood.negative_log_likelihood` and
:func:`~mltpy.constraints.build_constraints`.

Analogue to R's ``mltoptim.R`` (sequential solver attempts) and the ``maxtry``
restart mechanism in ``mlt()``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, cast

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from scipy.optimize import (
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
    minimize,
)

from mltpy._auglag import (
    AugLagOptions,
    AugLagResult,
    NonlinearInequality,
    auglag_minimize,
)
from mltpy.basis import BernsteinBasis, InteractionBasis
from mltpy.constraints import (
    ConstraintMatrices,
    MonotonicityConstraint,
    ScaledSupportConstraint,
    build_constraint_matrices,
    build_constraint_matrices_interaction,
    build_constraints,
)
from mltpy.likelihood import (
    BaseDistribution,
    DistOps,
    InfeasibleParameterError,
    _get_dist,
    _hessian_interaction_fd,
    _interaction_nll,
    _interaction_nll_and_grad,
    _InteractionDesign,
    _negative_log_likelihood_from_dist,
)
from mltpy.likelihood import (
    hessian as _hessian,
)
from mltpy.variables import CensoredData, CensoringType

# Objective value returned for an infeasible θ (non-finite log-likelihood,
# typically h' ≤ 0 after a trial step leaves the monotone cone).  Finite so
# that line searches (SLSQP, L-BFGS-B inside auglag) can backtrack from it; an
# ``inf`` value stalls L-BFGS-B's line search (issue #134).  Paired with the
# subgradient of the monotonicity-violation penalty — see ``_make_objective``
# and ``_optimize_interaction``.
#
# ``_INFEASIBLE_NLL`` is only the *floor*.  A weighted likelihood can exceed
# any fixed constant (n = 1000 with every weight 1e8 sits near 9e10), and a
# penalty below the feasible objective turns the infeasible region into the
# minimum the solver descends to.  Each fit therefore scales the penalty to
# its own objective via ``_infeasible_penalty``, and ``_restart_loop`` checks
# the true likelihood before retaining a candidate (ADR 0004, Decision 1).
_INFEASIBLE_NLL = 1e10
# Penalty headroom over ``1 + |NLL(θ_init)|``.  The unweighted R-parity fits
# see a penalty-to-objective ratio of ~1e6–1e7 (1e10 against |f| ~ 1e3), so
# the floor keeps those fits unchanged and a rescaled objective gets the same
# ratio.
_INFEASIBLE_NLL_HEADROOM = 1e6


def _infeasible_penalty(
    theta_init: NDArray[np.float64],
    true_nll: Callable[[NDArray[np.float64]], float],
) -> float:
    """Penalty value for an infeasible θ, scaled to this fit's objective.

    Parameters
    ----------
    theta_init : NDArray[np.float64]
        The feasible starting point every restart is derived from.
    true_nll : Callable[[NDArray[np.float64]], float]
        Unpenalised negative log-likelihood; raises
        :class:`~mltpy.likelihood.InfeasibleParameterError` when θ is
        infeasible.

    Returns
    -------
    float
        ``max(_INFEASIBLE_NLL, _INFEASIBLE_NLL_HEADROOM · (1 + |NLL(θ_init)|))``.
        Falls back to the floor when the start itself is infeasible or its
        likelihood is not finite; ``_restart_loop`` still rejects any penalty
        value that a solver returns as its result.
    """
    try:
        f_init = float(true_nll(theta_init))
    except InfeasibleParameterError:
        return _INFEASIBLE_NLL
    if not np.isfinite(f_init):
        return _INFEASIBLE_NLL
    return max(_INFEASIBLE_NLL, _INFEASIBLE_NLL_HEADROOM * (1.0 + abs(f_init)))


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
        (trust-constr).  SLSQP minimises the likelihood under weights
        normalised to mean 1, so ``ftol`` means the same whatever the scale of
        ``weights`` (issue #137).
    max_restarts:
        Number of *additional* attempts after the first one.  Analogous to
        ``maxtry`` in R's ``mlt()``.  On each restart the starting point is
        perturbed and projected back to the feasible region.
    use_gradient:
        If ``True`` (default), the analytical gradient from
        :func:`~mltpy.likelihood.negative_log_likelihood` is passed to scipy.
        Set to ``False`` only for debugging.
    verbose:
        If ``True``, print a warning on each failed attempt.
    random_state:
        If an ``int``, seeds the RNG used to perturb restart starting points
        so repeated fits with the same config and data are bit-identical.
        If a :class:`numpy.random.Generator`, it is used directly.
        If ``None`` (default), draws are non-reproducible across runs.
    auglag_options:
        :class:`~mltpy._auglag.AugLagOptions` controlling the PHR outer loop.
        Only consulted when ``solver="auglag"``; ignored otherwise.  ``None``
        (default) uses :class:`AugLagOptions` defaults (alabama parity).
    lower:
        If not ``None``, fixes ``θ[0] = lower`` as an equality constraint
        (pins the lower-boundary Bernstein coefficient).  Honoured by every
        solver: passes through to :func:`~mltpy.constraints.build_constraints`
        for SLSQP/trust-constr and
        :func:`~mltpy.constraints.build_constraint_matrices` for auglag.
    upper:
        If not ``None``, fixes ``θ[n_params−1] = upper`` analogously.
    polish:
        If ``True`` (default), run a Newton-CG polish step after auglag
        converges when no monotonicity constraints are active (interior-MLE
        fits).  Uses ``trust-ncg`` seeded at auglag's θ-hat with the
        analytical Hessian from :func:`~mltpy.likelihood.hessian`.  The
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
        * :class:`~mltpy.basis.InteractionBasis` is not yet supported —
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
    ``D @ theta_b >= 0`` (the :class:`~mltpy.constraints.MonotonicityConstraint`).

    Parameters
    ----------
    theta_b : NDArray[np.float64]
        Bernstein coefficient vector of length p (may be non-monotone).

    Returns
    -------
    NDArray[np.float64]
        NDArray of length p, sorted ascending.
    """
    return np.sort(theta_b)


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
    penalty: float = _INFEASIBLE_NLL,
) -> Callable[[NDArray[np.float64]], Any]:
    """Return a closure suitable for ``scipy.optimize.minimize``.

    When ``use_gradient=True`` the closure returns ``(nll, grad)`` and
    ``jac=True`` should be passed to scipy.  When False it returns a scalar
    and ``jac=None`` should be used.

    :class:`~mltpy.likelihood.InfeasibleParameterError` from an infeasible
    ``theta`` (h' ≤ 0) is caught and replaced by ``penalty`` so that the
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
    penalty : float, default=_INFEASIBLE_NLL
        Value returned for an infeasible ``theta``.  Solver paths pass
        :func:`_infeasible_penalty` so that it dominates this fit's feasible
        objective values whatever their scale (e.g. large weights).

    Returns
    -------
    Callable[[NDArray[np.float64]], Any]
        Objective function mapping a parameter vector ``theta`` to a scalar
        negative log-likelihood (and optionally its gradient vector).
    """
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
                # theta_b differences) — pairing the penalty value with a huge
                # gradient would misrepresent the local slope.  Beta block stays zero
                # because beta does not enter the monotonicity constraint.
                grad = np.zeros_like(theta)
                if _D is not None:
                    g = _D @ theta[:n_params]
                    grad[:n_params] = _D.T @ np.minimum(g, 0.0)
                return penalty, grad
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
                return penalty

    return obj


def _make_true_nll(
    basis: BernsteinBasis | InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None,
    censoring: CensoringType,
    dist: DistOps,
    weights: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
    scaling: NDArray[np.float64] | None,
) -> Callable[[NDArray[np.float64]], float]:
    """Unpenalised negative log-likelihood ``θ ↦ NLL(θ)`` for one fit.

    Unlike the solver objective, it does not absorb
    :class:`~mltpy.likelihood.InfeasibleParameterError` into a penalty: an
    infeasible θ raises.  Used to anchor :func:`_infeasible_penalty`, to
    verify candidates in :func:`_restart_loop`, and to score the
    fallback / polished θ, so that a reported log-likelihood is always a real
    one.
    """

    def true_nll(theta: NDArray[np.float64]) -> float:
        return float(
            cast(
                float,
                _negative_log_likelihood_from_dist(
                    theta,
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
        )

    return true_nll


def _solver_weights(
    weights: NDArray[np.float64] | None,
    solver: Literal["auglag", "slsqp", "trust-constr"],
) -> tuple[NDArray[np.float64] | None, float]:
    """Weights a scipy solver's objective is built with, and the factor removed.

    SLSQP's stopping test (``ftol``) is absolute in the objective, and its QP
    subproblem sets the objective gradient against unit-scale monotonicity
    rows.  A uniform weight ``w`` multiplies the NLL and its gradient by ``w``
    without moving the MLE, yet from ``w ≈ 1e4`` the stopping test cannot be
    met in floating point and from ``w ≈ 1e6`` the QP is reported
    incompatible after one step (issue #137).  Normalising the weights to
    mean 1 gives SLSQP an objective on the scale of an unweighted fit with
    the same argmin; uniform weights normalise to exactly 1, so such a fit
    takes the unweighted fit's path.

    Only the objective the solver sees changes.  Candidates are still scored,
    and the log-likelihood reported, under the caller's weights.

    The normalisation covers an objective made large by its weights.  An
    unweighted objective made large by sheer sample size (``|NLL| ≳ 1e6``,
    millions of rows) still meets an absolute ``ftol``.

    Parameters
    ----------
    weights : NDArray[np.float64] | None
        Validated per-observation weights (finite, non-negative, at least one
        positive entry), or ``None``.
    solver : Literal["auglag", "slsqp", "trust-constr"]
        ``config.solver``.  Only ``"slsqp"`` is normalised: trust-constr and
        auglag reach the MLE at every weight scale and keep their objective.

    Returns
    -------
    tuple[NDArray[np.float64] | None, float]
        ``(weights / s, s)`` with ``s = mean(weights)`` for a weighted SLSQP
        fit, otherwise ``(weights, 1.0)``.  The solver's objective times ``s``
        is the NLL under the caller's weights.
    """
    if weights is None or solver != "slsqp":
        return weights, 1.0
    scale = float(np.mean(weights))
    return weights / scale, scale


def _unscale_scipy_result(result: OptimizeResult, scale: float) -> OptimizeResult:
    """Restore ``.fun`` / ``.jac`` of a scipy result to the caller's weights.

    Multiplies both by ``scale`` (from :func:`_solver_weights`) in place and
    returns the same ``result``; a no-op when ``scale == 1``.
    """
    if scale != 1.0:
        result.fun = result.fun * scale
        if getattr(result, "jac", None) is not None:
            result.jac = result.jac * scale
    return result


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
    support_floor: float = 0.0,
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
    q_s : int, default=0
        Number of scaling columns; their γ entries start at zero.
    support_floor : float, default=0.0
        Exponential support floor ``c = max(0, max_i(−offset_i))`` from
        :func:`_support_floor`.  When ``lower`` is not pinned and
        ``theta_b[0] < c``, the basis coefficients are lifted so
        ``theta_b[0] = c`` while staying strictly increasing (see Notes).

    Returns
    -------
    NDArray[np.float64]
        Initial concatenated parameter vector of shape `(p + q,)`.
        Always non-decreasing by construction, so the monotonicity constraint
        is satisfied at the first trial.  When ``lower`` / ``upper`` are
        provided the starting point also satisfies the boundary equality
        constraints, sparing the augmented-Lagrangian penalty an unnecessary
        initial gradient swing.

    Notes
    -----
    The support lift is a parallel shift, ``theta_b + (c − theta_b[0])``,
    which keeps the spacing of the default start (the same lift as
    :func:`_perturb_and_project` and the interaction path's column lift).
    With ``upper`` pinned it is ``linspace(c, max(c, upper), p)`` instead, so
    the pin holds whenever ``c ≤ upper``; for ``c > upper`` no feasible
    increasing start exists and the start is flat at ``c``.

    A lift by ``np.maximum(theta_b, c)`` would set the leading coefficients
    equal.  With order ``M = p − 1`` and ``ỹ = (y − a)/(b − a)``,
    ``h'(y) = M/(b − a) · Σ_{k<M} Δθ_k B_{k,M−1}(ỹ)``; zero leading
    ``Δθ_k`` make it vanish to high order at the lower support end, so
    ``∂ log h'(y_i)/∂θ_b`` is huge for small ``y_i`` (‖∇ℓ‖ ~8e7 on the #115
    fixture against ~5e2 here); SLSQP's first quasi-Newton step, taken with
    ``B = I``, then lands far outside the useful region (issue #165).
    """
    if lower is not None and upper is not None:
        theta_b = np.linspace(float(lower), float(upper), n_params)
    elif lower is not None:
        theta_b = np.linspace(0.0, 1.0, n_params) + (float(lower) - 0.0)
    elif upper is not None:
        theta_b = np.linspace(0.0, 1.0, n_params) + (float(upper) - 1.0)
    else:
        theta_b = np.linspace(0.0, 1.0, n_params)
    if lower is None and theta_b[0] < support_floor:
        if upper is None:
            theta_b = theta_b + (support_floor - theta_b[0])
        else:
            theta_b = np.linspace(
                support_floor, max(support_floor, float(upper)), n_params
            )
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
    support_floor: float = 0.0,
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
        If True, shift the projected coefficients in parallel so
        ``theta_b[0] >= support_floor`` (needed when
        ``base_distribution="exponential"``).
    support_floor : float, default=0.0
        Lower bound for ``theta_b[0]`` under ``nonneg_lower``:
        ``max(0, max_i(−offset_i))`` from :func:`_support_floor`.  With
        ``β = 0`` (the default start) it makes every exponential support row
        ``theta_b[0] + x_i·β ≥ −offset_i`` feasible (issue #160).

    Returns
    -------
    NDArray[np.float64]
        A valid, non-decreasing parameter vector of identical shape.
    """
    theta_b = theta[:n_params] + rng.normal(0.0, scale, size=n_params)
    theta_b = _project_to_feasible(theta_b)
    if nonneg_lower and theta_b[0] < support_floor:
        theta_b = theta_b - theta_b[0] + support_floor
    if len(theta) > n_params:
        return cast(NDArray[np.float64], np.concatenate([theta_b, theta[n_params:]]))
    return theta_b


def _support_floor(nonneg_lower: bool, offset: NDArray[np.float64] | None) -> float:
    """Lower bound ``c = max(0, max_i(−offset_i))`` for a feasible ``theta_b[0]``.

    Parameters
    ----------
    nonneg_lower : bool
        Whether the exponential support rows are active.
    offset : NDArray[np.float64] | None
        Per-observation offset, or ``None``.

    Returns
    -------
    float
        ``0.0`` unless ``nonneg_lower`` and a non-empty ``offset`` with some
        ``offset_i < 0`` are given.

    Notes
    -----
    The shift-path support rows read ``theta_b[0] + x_i·β ≥ −offset_i``
    (ADR 0005 Decision 1, issue #160).  At ``β = 0`` — the default start —
    every row holds once ``theta_b[0] ≥ c``; the ``max(0, ·)`` keeps the
    pre-#160 lift to ``theta_b[0] ≥ 0``, so ``offset=None`` is unchanged.
    The interaction path computes the same ``c`` for its column lift.
    """
    if not nonneg_lower or offset is None:
        return 0.0
    off = np.asarray(offset, dtype=np.float64)
    if off.size == 0:
        return 0.0
    return max(0.0, float(np.max(-off)))


def _restart_loop(
    theta_init: NDArray[np.float64],
    config: OptimizerConfig,
    perturb_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    run_fn: Callable[[NDArray[np.float64]], Any],
    success_fn: Callable[[Any], bool],
    nll_fn: Callable[[Any], float],
    *,
    label: str = "optimizer.py",
    message_fn: Callable[[Any], str] | None = None,
) -> tuple[Any, float, int]:
    """Run the shared perturb-and-project restart loop and return the best result.

    Every solver path (auglag / SLSQP / trust-constr, shift and interaction
    bases) wraps the same skeleton: attempt 0 starts from ``theta_init``;
    each subsequent attempt perturbs and re-projects the start, retrying up to
    ``config.max_restarts`` extra times.  A :class:`numpy.linalg.LinAlgError`
    from any attempt is caught and (optionally) warned, then skipped.  The best
    result by *verified* negative log-likelihood is retained even when
    convergence is never reported, so the caller can return
    ``converged=False`` with the closest fit rather than nothing.

    The solver's own ``.fun`` is not trusted for this: on an infeasible θ the
    objective returns a finite penalty, which is an ordinary number to the
    solver and to a comparison.  Each result is re-scored with ``nll_fn``
    (the unpenalised likelihood), and a result whose θ is infeasible or whose
    likelihood is not finite is discarded like a ``LinAlgError`` attempt — it
    is neither retained nor allowed to end the loop as converged.

    Parameters
    ----------
    theta_init : NDArray[np.float64]
        Feasible starting point.  Attempt 0 uses ``theta_init.copy()``; later
        attempts call ``perturb_fn(theta_init)``.
    config : OptimizerConfig
        Supplies ``max_restarts`` and ``verbose``.
    perturb_fn : Callable[[NDArray[np.float64]], NDArray[np.float64]]
        Maps the (unperturbed) ``theta_init`` to a fresh feasible start for a
        restart attempt.
    run_fn : Callable[[NDArray[np.float64]], Any]
        Runs the underlying solver on a trial start and returns a result object
        exposing a ``.fun`` attribute.  May raise ``LinAlgError`` to signal a
        recoverable numerical failure.
    success_fn : Callable[[Any], bool]
        Predicate marking a result as converged (``r.success`` for scipy,
        ``r.converged`` for auglag).  A truthy value breaks the loop.
    nll_fn : Callable[[Any], float]
        Unpenalised negative log-likelihood at a result's θ.  Raises
        :class:`~mltpy.likelihood.InfeasibleParameterError` (or returns a
        non-finite value) when that θ is infeasible.
    label : str, default="optimizer.py"
        Prefix for the ``RuntimeWarning`` messages (e.g.
        ``"optimizer.py (interaction)"``).
    message_fn : Callable[[Any], str] | None, default=None
        When provided *and* ``config.verbose``, emits a "did not converge"
        warning carrying ``message_fn(result)`` after each unsuccessful
        attempt.  ``None`` suppresses that warning (the interaction paths).

    Returns
    -------
    tuple[Any, float, int]
        ``(best_result, best_nll, n_restarts_used)``.  ``best_nll`` is the
        verified likelihood of ``best_result``.  ``best_result`` is ``None``
        (and ``best_nll`` is ``inf``) only when every attempt raised
        ``LinAlgError`` or ended at an infeasible θ.
    """
    best_result: Any = None
    best_nll = float("inf")
    n_restarts_used = 0

    for attempt in range(config.max_restarts + 1):
        if attempt == 0:
            theta_try = theta_init.copy()
        else:
            n_restarts_used = attempt
            theta_try = perturb_fn(theta_init)

        try:
            result = run_fn(theta_try)
        except LinAlgError as exc:
            if config.verbose:
                warnings.warn(
                    f"{label}: attempt {attempt + 1} hit {exc!r}; retrying",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

        try:
            nll = float(nll_fn(result))
        except InfeasibleParameterError:
            nll = float("nan")
        if not np.isfinite(nll):
            if config.verbose:
                warnings.warn(
                    f"{label}: attempt {attempt + 1} ended at an infeasible θ "
                    f"(objective {float(result.fun):.6g} is the infeasibility "
                    "penalty, not a likelihood); discarding it",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

        if nll < best_nll:
            best_nll = nll
            best_result = result

        if success_fn(result):
            break

        if config.verbose and message_fn is not None:
            warnings.warn(
                f"{label}: attempt {attempt + 1}/{config.max_restarts + 1} "
                f"did not converge — {message_fn(result)}",
                RuntimeWarning,
                stacklevel=2,
            )

    return best_result, best_nll, n_restarts_used


def _fallback_result(
    theta_init: NDArray[np.float64],
    n_restarts_used: int,
    message: str,
    compute_nll: Callable[[], float],
) -> OptimizationResult:
    """Build the ``converged=False`` result when no restart left a usable θ.

    That is, every attempt raised ``LinAlgError`` or ended at an infeasible θ.

    Returns ``theta_init`` (the only feasible point we are certain of) with its
    log-likelihood, leaving the auglag-only diagnostic fields at their ``None``
    defaults.  The caller (``model.py``) decides whether to warn or raise.

    Parameters
    ----------
    theta_init : NDArray[np.float64]
        Feasible starting point returned as the fallback ``theta``.
    n_restarts_used : int
        Number of restarts attempted before giving up.
    message : str
        ``solver_message`` describing why the fit fell back.
    compute_nll : Callable[[], float]
        Thunk returning the negative log-likelihood at ``theta_init``.  Deferred
        so the (path-specific) likelihood call is only made on the failure path.

    Returns
    -------
    OptimizationResult
        Fallback result with ``converged=False`` and ``n_iter=0``.
    """
    return OptimizationResult(
        theta=theta_init,
        log_likelihood=float(-compute_nll()),
        converged=False,
        n_iter=0,
        n_restarts=n_restarts_used,
        solver_message=message,
    )


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
        :class:`~mltpy.basis.BernsteinBasis` instance defining the response
        transformation.
    y:
        Observations — plain ``NDArray`` for exact data, or
        :class:`~mltpy.variables.CensoredData` for censored data.
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

    Raises
    ------
    NotImplementedError
        For an :class:`~mltpy.basis.InteractionBasis` when ``y`` is a
        :class:`~mltpy.variables.CensoredData` with censored rows (the
        interaction likelihood is exact-data only), when ``scaling`` is
        combined with ``base_distribution="exponential"``, or when
        ``config.fixed_params`` is set.

    Notes
    -----
    On the shift basis ``scaling`` combined with
    ``base_distribution="exponential"`` fits on every solver.  The support
    rule ``θ_b[0]·exp(0.5·x_s,i·γ) + x_i·β ≥ −offset_i`` collapses to the
    linear row ``θ_b[0] ≥ 0`` without a shift block and with a zero offset
    (issue #114); otherwise it is non-linear in ``γ`` (ADR 0005 Decision 2):
    auglag carries it as its nonlinear inequality block (issue #115), SLSQP
    as an ``ineq`` dict and trust-constr as a ``NonlinearConstraint`` with
    the analytic Jacobian and Hessian (issue #116).
    """
    # Fail fast for unsupported base distributions before entering scipy.
    dist = _get_dist(base_distribution)
    if config is None:
        config = OptimizerConfig()

    if isinstance(basis, InteractionBasis):
        if isinstance(y, CensoredData) and y.n_censored > 0:
            # The interaction likelihood evaluates y.exact only; censored rows
            # there are NaN and would surface as an opaque NaN log-likelihood.
            raise NotImplementedError(
                "InteractionBasis supports exact (uncensored) data only; got "
                f"CensoredData with {y.n_censored} censored row(s) out of {y.n}. "
                "Censored likelihoods are not implemented on the interaction "
                "path."
            )
        if scaling is not None and base_distribution == "exponential":
            # ADR 0003, Decision 5 — the exponential link's support row
            # becomes non-linear in γ per interacting column; carried over
            # from ADR 0002 Decision 3.
            raise NotImplementedError(
                "scaling= is not supported with base_distribution='exponential' "
                "(see docs/adr/0003-scaling-with-interaction.md, Decision 5)."
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
            weights=weights,
            offset=offset,
            scaling=scaling,
            nonneg_lower=base_distribution == "exponential",
        )

    # ADR 0005 Decision 2 — under scaling the exponential support rule
    # h(a|x_i) ≥ 0 reads θ_b[0]·f_i + x_i·β ≥ −offset_i with
    # f_i = exp(0.5·x_s,i·γ).  Without a shift block and with a zero offset
    # it collapses to the linear row θ_b[0] ≥ 0 (issue #114); every other
    # case is the nonlinear block, which every solver carries: auglag as
    # ``nonlinear_ineq`` (issue #115), SLSQP / trust-constr natively
    # (issue #116).
    q_s = scaling.shape[1] if scaling is not None else 0
    n_params = basis.order + 1
    total_params = n_params + (X.shape[1] if X is not None else 0) + q_s
    nonneg_lower = base_distribution == "exponential"

    if isinstance(config.random_state, np.random.Generator):
        rng = config.random_state
    else:
        rng = np.random.default_rng(config.random_state)

    # Exponential support rows read θ_b[0] + x_i·β ≥ −offset_i (issue #160).
    # At the default β = 0 they all hold once θ_b[0] ≥ c = max(0,
    # max_i(−offset_i)); the start is lifted there in parallel, keeping θ_b
    # strictly increasing (issue #165), unless θ_b[0] is pinned by ``lower``.
    support_floor = _support_floor(nonneg_lower, offset)
    theta_init = _initial_theta(
        n_params,
        X,
        lower=config.lower,
        upper=config.upper,
        q_s=q_s,
        support_floor=support_floor,
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
            support_floor=support_floor,
            rng=rng,
            theta_init=theta_init,
            scaling=scaling,
        )

    return _optimize_scipy(
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
        support_floor=support_floor,
        rng=rng,
        theta_init=theta_init,
        scaling=scaling,
    )


def _optimize_scipy(
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
    support_floor: float,
    rng: np.random.Generator,
    theta_init: NDArray[np.float64],
    scaling: NDArray[np.float64] | None = None,
) -> OptimizationResult:
    """SLSQP / trust-constr optimisation path for the shift basis.

    Sibling of :func:`_optimize_auglag` and :func:`_optimize_interaction`:
    builds the linear constraints, applies the issue-#86 ``fixed_params``
    reduction (pinned indices eliminated from the free subvector), and drives
    the shared :func:`_restart_loop`.  The inner solver is scipy's ``minimize``
    with ``method=config.solver``.
    """
    # This path only ever runs for the two scipy solvers; narrow the type so
    # the overloaded ``build_constraints`` matches (auglag is dispatched away
    # before we get here).
    solver = cast("Literal['slsqp', 'trust-constr']", config.solver)
    # Exponential has support [0, ∞); enforce h(a|x_i) >= 0 with the offset
    # included (ADR 0005 Decision 1, issue #160).  Without covariates this
    # reduces to theta_b[0] >= max_i(-offset_i); with covariates we add one
    # linear inequality per training row: theta_b[0] + X_i @ beta >= -offset_i.
    # Under scaling with no shift block and a zero offset it is
    # theta_b[0] >= 0 (issue #114); otherwise the rows
    # theta_b[0]·exp(0.5·x_s,i·γ) + X_i @ beta >= -offset_i are non-linear in
    # γ and enter natively — an SLSQP ``ineq`` dict or a trust-constr
    # ``NonlinearConstraint`` with the analytic Jacobian (issue #116).
    constraints = build_constraints(
        n_params,
        lower=config.lower,
        upper=config.upper,
        solver=solver,
        total_params=total_params,
        nonneg_lower=nonneg_lower,
        X=X if nonneg_lower else None,
        offset=offset if nonneg_lower else None,
        q_s=scaling.shape[1] if scaling is not None else 0,
        X_scale=scaling if nonneg_lower else None,
    )

    true_nll = _make_true_nll(basis, y, X, censoring, dist, weights, offset, scaling)

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
    if config.fixed_params:
        free_idx, fixed_idx, fixed_vals = _fixed_params_split(
            config.fixed_params, total_params
        )
        theta_init[fixed_idx] = fixed_vals

    # SLSQP sees the likelihood under mean-1 weights (issue #137); scoring
    # below keeps the caller's weights via ``true_nll``.
    obj_weights, obj_scale = _solver_weights(weights, config.solver)
    obj_true_nll = _make_true_nll(
        basis, y, X, censoring, dist, obj_weights, offset, scaling
    )

    obj = _make_objective(
        basis,
        y,
        X,
        censoring,
        config.use_gradient,
        base_distribution=base_distribution,
        dist=dist,
        weights=obj_weights,
        offset=offset,
        scaling=scaling,
        penalty=_infeasible_penalty(theta_init, obj_true_nll),
    )
    jac = True if config.use_gradient else None
    options = _scipy_options(config)

    obj_used: Callable[[NDArray[np.float64]], Any] = obj
    constraints_used: Any = constraints
    if free_idx is not None and fixed_idx is not None and fixed_vals is not None:
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

            def _slice_lc(
                lc: LinearConstraint | NonlinearConstraint,
            ) -> LinearConstraint | NonlinearConstraint:
                if isinstance(lc, NonlinearConstraint):
                    # Scaled exponential support rows (issue #116): evaluate
                    # on the full θ, keep the free columns of the Jacobian
                    # and the free block of the Hessian term.
                    f, j, hs = lc.fun, lc.jac, lc.hess
                    free_block = np.ix_(_free_idx, _free_idx)
                    return NonlinearConstraint(
                        lambda tf, _f=f: _f(_reconstruct(tf)),
                        lc.lb,
                        lc.ub,
                        jac=lambda tf, _j=j: np.asarray(_j(_reconstruct(tf)))[
                            :, _free_idx
                        ],
                        hess=lambda tf, v, _h=hs: np.asarray(_h(_reconstruct(tf), v))[
                            free_block
                        ],
                    )
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

    def _perturb(t: NDArray[np.float64]) -> NDArray[np.float64]:
        theta_try_full = _perturb_and_project(
            t,
            n_params,
            rng,
            nonneg_lower=nonneg_lower,
            support_floor=support_floor,
        )
        if fixed_idx is not None:
            # Perturbation projects theta_b onto the monotone cone, which may
            # shift pinned entries.  Re-applying the pins here keeps the reduced
            # theta_try aligned with the constraint slicing (lb/ub were shifted
            # by ``A[:, fixed_idx] @ fixed_vals``).
            theta_try_full[fixed_idx] = fixed_vals
        return theta_try_full

    def _run(theta_try_full: NDArray[np.float64]) -> Any:
        theta_try = theta_try_full[free_idx] if free_idx is not None else theta_try_full
        return _unscale_scipy_result(
            minimize(
                obj_used,
                theta_try,
                method=config.solver,
                jac=jac,
                constraints=constraints_used,
                options=options,
            ),
            obj_scale,
        )

    def _full_theta(x: NDArray[np.float64]) -> NDArray[np.float64]:
        # Reduction path returns the optimum on the free subvector; lift it
        # back to the full layout (fixed entries restored to their pinned
        # values) so downstream consumers see the unchanged
        # ``[theta_b | beta | gamma]`` representation regardless of whether
        # fixed_params was used.
        if free_idx is None or fixed_idx is None or fixed_vals is None:
            return np.asarray(x, dtype=np.float64)
        full = np.empty(total_params, dtype=np.float64)
        full[free_idx] = x
        full[fixed_idx] = fixed_vals
        return full

    best_scipy_result, best_nll, n_restarts_used = _restart_loop(
        theta_init,
        config,
        _perturb,
        _run,
        lambda r: bool(r.success),
        lambda r: true_nll(_full_theta(r.x)),
        message_fn=lambda r: str(r.message),
    )

    if best_scipy_result is None:
        # No attempt left a usable θ (numerical linear-algebra failure or an
        # infeasible end point) — return the initial point as fallback so the
        # caller can surface a ConvergenceWarning rather than re-raising.
        return _fallback_result(
            theta_init,
            n_restarts_used,
            "All optimisation attempts raised LinAlgError or ended at an infeasible θ.",
            lambda: true_nll(theta_init),
        )

    return OptimizationResult(
        theta=_full_theta(best_scipy_result.x),
        log_likelihood=float(-best_nll),
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
    weights: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
    scaling: NDArray[np.float64] | None = None,
    nonneg_lower: bool = False,
) -> OptimizationResult:
    """Optimisation path for InteractionBasis (stratified / fully-interacting CTM).

    Dispatches on ``config.solver`` and uses the Kronecker monotonicity
    constraint ``(D ⊗ I_q) @ vec(Θ) ≥ 0`` (built once via
    :func:`~mltpy.constraints.build_constraint_matrices_interaction`) in the
    form each solver expects.  X is required — it cannot be ``None`` for an
    interaction model.

    When ``scaling is not None`` the parameter vector is
    ``[vec_C(Θ) | γ]`` (ADR 0003): the γ block is initialised to zeros, the
    Kronecker constraint is padded with ``q_s`` zero columns, and perturb-
    and-project restarts touch only the ``vec(Θ)`` block.

    When ``nonneg_lower`` is ``True`` (``base_distribution="exponential"``)
    the constraint set also carries one support row per training observation,
    ``h(a|x_i) = (a(a) ⊗ b(x_i))ᵀ vec(Θ) + offset_i ≥ 0`` (ADR 0005,
    Decision 1).  The start point and every restart are lifted so that each
    column's ``h(a)`` value is at least ``c = max(0, max_i(−offset_i))``; with
    a partition-of-unity x-basis that makes every support row feasible.
    """
    if X is None:
        raise ValueError(
            "InteractionBasis.fit() requires X (covariate matrix). "
            "Pass the stratum labels as X."
        )
    x_arr = basis._coerce_x(X)

    q_s = scaling.shape[1] if scaling is not None else 0
    cm = build_constraint_matrices_interaction(
        basis,
        q_s=q_s,
        nonneg_lower=nonneg_lower,
        X=x_arr if nonneg_lower else None,
        offset=offset if nonneg_lower else None,
    )

    p = basis.n_y_params
    q = basis.n_x_params
    pq = p * q

    # Exponential support (ADR 0005): every column's lower-end value
    # a(a)ᵀ Θ[:, j] must reach c = max(0, max_i(−offset_i)).  The x-basis is
    # a non-negative partition of unity, so h(a|x_i) + offset_i ≥ c +
    # offset_i ≥ 0 for every row.  ``basis_at_lower`` = a(a), which is e_0
    # for a Bernstein y-basis.
    basis_at_lower = basis.y_basis.evaluate(
        np.full(1, basis.support[0], dtype=np.float64)
    )[0]
    support_floor = _support_floor(nonneg_lower, offset)

    def lift_to_support(Theta: NDArray[np.float64]) -> NDArray[np.float64]:
        if not nonneg_lower:
            return Theta
        # Adding δ to a whole column raises h(a) for that column by
        # δ·Σ_k a_k(a) = δ for a partition-of-unity y-basis, and keeps the
        # column non-decreasing.
        deficit = np.maximum(support_floor - basis_at_lower @ Theta, 0.0)  # (q,)
        return cast(NDArray[np.float64], Theta + deficit[None, :])

    Theta_init = lift_to_support(np.outer(np.linspace(0.0, 1.0, p), np.ones(q)))
    theta_init = np.concatenate([Theta_init.ravel(), np.zeros(q_s, dtype=np.float64)])

    if isinstance(config.random_state, np.random.Generator):
        rng = config.random_state
    else:
        rng = np.random.default_rng(config.random_state)

    # The dense Kronecker designs a(y)⊗b(x) and a'(y)⊗b(x) do not depend on θ:
    # build them once and share them across every objective, gradient,
    # penalty-anchor and polish-Hessian evaluation of this fit (issue #147).
    # Only the closures below hold ``kron``, so it is released when this
    # function returns; it is never stored on the result or the model.
    kron = _InteractionDesign.build(basis, y, x_arr)

    def true_nll(theta: NDArray[np.float64]) -> float:
        # Unpenalised, like ``_make_true_nll``: an infeasible θ raises.
        return _interaction_nll(
            kron, theta, dist, weights=weights, offset=offset, scaling=scaling
        )

    # SLSQP sees the likelihood under mean-1 weights (issue #137); scoring
    # keeps the caller's weights via ``true_nll``.
    obj_weights, obj_scale = _solver_weights(weights, config.solver)
    penalty = _infeasible_penalty(
        theta_init,
        lambda t: _interaction_nll(
            kron, t, dist, weights=obj_weights, offset=offset, scaling=scaling
        ),
    )

    def obj(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        try:
            return _interaction_nll_and_grad(
                kron,
                theta,
                dist,
                weights=obj_weights,
                offset=offset,
                scaling=scaling,
            )
        except InfeasibleParameterError:
            # Same contract as the shift path's ``_make_objective``: a finite
            # penalty plus the subgradient of the column-wise monotonicity
            # violation P(θ) = 0.5·‖min(0, A_ineq @ θ − b_ineq)‖², with
            # A_ineq = (D ⊗ I_q) padded by q_s zero γ columns.  Returning
            # ``(inf, 0)`` here stalled L-BFGS-B's line search on the first
            # trial step that left the cone, and auglag then reported the
            # unmoved start point as converged (issue #134).  When θ is inside
            # the cone (h' = 0 at a data point without a negative difference)
            # the subgradient is zero and the finite value alone makes the
            # line search backtrack.  ``penalty`` is scaled to this fit's
            # objective so that it stays above every feasible value even
            # under large weights (see ``_infeasible_penalty``).
            viol = np.minimum(cm.A_ineq @ theta - cm.b_ineq, 0.0)
            return penalty, cm.A_ineq.T @ viol

    def perturb(theta_seed: NDArray[np.float64]) -> NDArray[np.float64]:
        # Perturb-and-project only the vec(Θ) block; leave γ untouched.
        Theta_try = theta_seed[:pq].reshape(p, q) + rng.normal(0.0, 0.1, size=(p, q))
        for j in range(q):
            Theta_try[:, j] = np.maximum.accumulate(Theta_try[:, j])
        Theta_try = lift_to_support(Theta_try)
        return np.concatenate([Theta_try.ravel(), theta_seed[pq:]])

    def hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return _hessian_interaction_fd(
            kron, theta, dist, weights=weights, offset=offset, scaling=scaling
        )

    if config.solver == "auglag":
        return _interaction_auglag(
            obj=obj,
            true_nll=true_nll,
            hess=hess,
            theta_init=theta_init,
            cm=cm,
            config=config,
            perturb=perturb,
        )

    return _interaction_scipy(
        obj=obj,
        true_nll=true_nll,
        theta_init=theta_init,
        cm=cm,
        config=config,
        perturb=perturb,
        obj_scale=obj_scale,
    )


def _interaction_auglag(
    *,
    obj: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    true_nll: Callable[[NDArray[np.float64]], float],
    hess: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    theta_init: NDArray[np.float64],
    cm: Any,
    config: OptimizerConfig,
    perturb: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> OptimizationResult:
    """PHR augmented-Lagrangian path for the interaction basis.

    ``hess`` is the observed-information Hessian used by the Newton-CG polish;
    ``obj``, ``true_nll`` and ``hess`` all evaluate on the same fit-scoped
    designs built in :func:`_optimize_interaction`.
    """
    auglag_opts = (
        config.auglag_options if config.auglag_options is not None else AugLagOptions()
    )

    best_result, best_nll, n_restarts_used = _restart_loop(
        theta_init,
        config,
        perturb,
        lambda t: auglag_minimize(
            obj,
            t,
            A_ineq=cm.A_ineq,
            b_ineq=cm.b_ineq,
            C_eq=cm.C_eq,
            d_eq=cm.d_eq,
            options=auglag_opts,
        ),
        lambda r: bool(r.converged),
        lambda r: true_nll(r.theta),
        label="optimizer.py (interaction)",
    )

    if best_result is None:
        return _fallback_result(
            theta_init,
            n_restarts_used,
            "All interaction-model auglag attempts raised LinAlgError or "
            "ended at an infeasible θ.",
            lambda: true_nll(theta_init),
        )

    # Newton-CG polish: only when all Kronecker-constraint multipliers are inactive
    ia_final_theta = best_result.theta
    ia_final_nll = best_nll
    if config.polish and np.all(best_result.mu_ineq < 1e-6):
        theta_polished = _ncg_polish(
            best_result.theta, obj, hess, 0, A_ineq=cm.A_ineq, b_ineq=cm.b_ineq
        )
        if theta_polished is not None:
            ia_final_theta = theta_polished
            ia_final_nll = true_nll(theta_polished)

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
    true_nll: Callable[[NDArray[np.float64]], float],
    theta_init: NDArray[np.float64],
    cm: Any,
    config: OptimizerConfig,
    perturb: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    obj_scale: float,
) -> OptimizationResult:
    """SLSQP / trust-constr path for the interaction basis.

    Both methods consume the same ``A_ineq θ ≥ b_ineq`` rows that
    :func:`~mltpy.constraints.build_constraint_matrices_interaction` builds
    (Kronecker monotonicity, plus the exponential support rows whose
    right-hand side is ``−offset``) — SLSQP via a ``type='ineq'`` dict and
    trust-constr via a :class:`scipy.optimize.LinearConstraint`.
    ``obj_scale`` is the factor ``obj`` was divided by
    (:func:`_solver_weights`); it restores the solver's ``.fun`` / ``.jac``.
    """
    A_ineq = cm.A_ineq
    b_ineq = cm.b_ineq

    if config.solver == "slsqp":
        constraints: Any = [
            {
                "type": "ineq",
                "fun": lambda theta, _A=A_ineq, _b=b_ineq: _A @ theta - _b,
                "jac": lambda theta, _A=A_ineq: _A,
            }
        ]
    else:  # trust-constr
        constraints = [LinearConstraint(A_ineq, lb=b_ineq, ub=np.inf)]

    options = _scipy_options(config)

    best_scipy_result, best_nll, n_restarts_used = _restart_loop(
        theta_init,
        config,
        perturb,
        lambda t: _unscale_scipy_result(
            minimize(
                obj,
                t,
                method=config.solver,
                jac=True,
                constraints=constraints,
                options=options,
            ),
            obj_scale,
        ),
        lambda r: bool(r.success),
        lambda r: true_nll(r.x),
        label="optimizer.py (interaction)",
    )

    if best_scipy_result is None:
        return _fallback_result(
            theta_init,
            n_restarts_used,
            "All interaction-model scipy attempts raised LinAlgError or "
            "ended at an infeasible θ.",
            lambda: true_nll(theta_init),
        )

    return OptimizationResult(
        theta=best_scipy_result.x,
        log_likelihood=float(-best_nll),
        converged=bool(best_scipy_result.success),
        n_iter=int(getattr(best_scipy_result, "nit", 0)),
        n_restarts=n_restarts_used,
        solver_message=str(best_scipy_result.message),
    )


def _ncg_polish(
    theta: NDArray[np.float64],
    obj: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    hess: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_params: int,
    *,
    A_ineq: NDArray[np.float64] | None = None,
    b_ineq: NDArray[np.float64] | None = None,
    nonlinear_ineq: NonlinearInequality | None = None,
) -> NDArray[np.float64] | None:
    """Newton-CG polish: trust-ncg from auglag's θ-hat using the analytical Hessian.

    Parameters
    ----------
    theta:
        Starting point (auglag's best θ).
    obj:
        NLL objective returning ``(nll, grad)``.
    hess:
        Hessian of the NLL, ``θ ↦ ∂²(-ℓ)/∂θ∂θ'``.  Any exception it raises
        (e.g. :class:`~mltpy.likelihood.InfeasibleParameterError` at a trial
        point outside the monotone cone) is absorbed into a stiff ``1e6·I``
        so that trust-ncg shrinks its step.
    n_params:
        Number of Bernstein basis coefficients ``θ_b`` (shift path).  Ignored
        when ``A_ineq`` is provided; the interaction path passes ``0``.
    A_ineq:
        Optional inequality constraint matrix.  When provided, feasibility is
        checked via ``A_ineq @ theta_polished >= b_ineq`` instead of the
        shift-model ``D @ theta_b >= 0`` check.
    b_ineq:
        Right-hand side paired with ``A_ineq``; zeros when ``None``.
    nonlinear_ineq:
        Optional nonlinear inequality block ``θ ↦ (g(θ), J(θ))``; the
        polished θ is rejected when some ``g_i(θ) < −1e-8`` (the scaled
        exponential support rows, issue #115).

    Returns
    -------
    NDArray[np.float64] or None
        Polished parameter vector, or ``None`` if polish failed or was rejected.
    """
    nll_before, _ = obj(theta)

    def hess_fn(t: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            return hess(t)
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
        rhs = b_ineq if b_ineq is not None else 0.0
        if np.any(A_ineq @ theta_new - rhs < -1e-8):
            return None
    elif n_params >= 2 and np.any(np.diff(theta_new[:n_params]) < -1e-8):
        return None

    if nonlinear_ineq is not None and np.any(nonlinear_ineq(theta_new)[0] < -1e-8):
        return None

    return theta_new


def _increment_auglag(
    obj: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    cm: ConstraintMatrices,
    support_nl: ScaledSupportConstraint | None,
    n_params: int,
    total_params: int,
    options: AugLagOptions,
    *,
    intercept_lower: float = -np.inf,
) -> Callable[[NDArray[np.float64]], AugLagResult]:
    """auglag runner for the scaled exponential paths (issues #115, #161).

    Solves in the increment coordinates

        δ = T θ,   δ_0 = θ_b[0],   δ_k = θ_b[k] − θ_b[k−1]  (k = 1, …, p−1),

    identity on ``[β | γ]``, where the monotonicity rows ``D θ_b ≥ 0`` are the
    bounds ``δ_k ≥ 0``, enforced inside every inner L-BFGS-B solve
    (box-constrained AL, :func:`~mltpy._auglag.auglag_minimize` Notes).

    Parameters
    ----------
    obj:
        NLL objective in θ, returning ``(nll, ∇nll)``.
    cm:
        Constraint matrices in θ (monotonicity rows, equality rows, and on
        the scale-only path the collapsed support row) with
        ``nonlinear_ineq = support_nl``.
    support_nl:
        The scaled support block ``θ ↦ (g(θ), J(θ))`` (shift block or
        non-zero offset, #115), or ``None`` on the scale-only path (#114),
        where the support rule is the single linear row ``θ_b[0] ≥ b``.
    n_params:
        Number of Bernstein coefficients ``p``.
    total_params:
        Length of ``θ = [θ_b | β | γ]``.
    options:
        auglag options.
    intercept_lower:
        Lower bound on ``δ_0 = θ_b[0]``: the right-hand side ``b`` of the
        collapsed support row on the scale-only path (#114), ``−inf`` (no
        bound) when the support rows are the nonlinear block.  Mutually
        exclusive with ``support_nl``: each path has exactly one support form.

    Returns
    -------
    Callable
        ``θ₀ ↦ AugLagResult`` with ``theta`` mapped back to θ.  Multipliers
        refer to the same constraint functions (``A_ineq`` rows, then the
        support rows; equality rows; ``nu_lower[k]`` is the bound
        ``δ_k ≥ 0``, i.e. monotonicity row ``k − 1``, and ``nu_lower[0]`` the
        bound ``θ_b[0] ≥ intercept_lower``), so they need no transformation.

    Raises
    ------
    ValueError
        If both ``support_nl`` and a finite ``intercept_lower`` are given.

    Notes
    -----
    Under the exponential link's smooth extension ``log f(h) = −h`` (and
    ``log S(h) = −h``) the NLL is unbounded below where ``h → −∞``.  The PHR
    penalty on the support rows ``g_i(θ) = h(a|x_i)`` controls ``h(·|x_i)``
    only while ``h_0`` is non-decreasing (then ``h(y|x_i) ≥ g_i`` for every
    ``y``).  A violated monotonicity row lets ``h_0(y) < θ_b[0]`` and, through
    ``f_i = exp(0.5·x_s,i·γ)``, the NLL fall exponentially in ``γ``, which
    a penalty quadratic in ``Dθ_b`` cannot bound — the #114 mechanism, here
    entering through the monotonicity rows on censored data.  A censored
    row has no ``−log h'`` term to push back, so the effect shows wherever
    censored rows sit in a region of ``y`` that no exact row covers (#161:
    rows below a common time ``c`` censored at their own times, exact rows
    tied at ``c``).  Keeping ``D θ_b ≥ 0`` (and, on the scale-only path,
    ``θ_b[0] ≥ b``) as hard bounds leaves every remaining violation linear
    in ``g_i`` against a quadratic penalty.  ``T`` is unit lower-triangular
    on the ``θ_b`` block, so the reparametrisation is exact and the fitted
    model is unchanged.
    """
    if support_nl is not None and np.isfinite(intercept_lower):
        raise ValueError(
            "Expected either the nonlinear support block (support_nl) or a "
            "finite intercept_lower bound, not both; got support_nl and "
            f"intercept_lower={intercept_lower}."
        )
    # T: θ ↦ δ (first differences on θ_b); T_inv: cumulative sums on δ_b.
    T = np.eye(total_params, dtype=np.float64)
    T[1:n_params, :n_params] = MonotonicityConstraint(n_params).as_matrix()
    T_inv = np.eye(total_params, dtype=np.float64)
    T_inv[:n_params, :n_params] = np.tril(np.ones((n_params, n_params)))
    lb = np.full(total_params, -np.inf)
    lb[0] = intercept_lower
    lb[1:n_params] = 0.0
    box = (lb, np.full(total_params, np.inf))
    A_d = cm.A_ineq @ T_inv
    C_d = cm.C_eq @ T_inv

    def obj_d(d: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        # PHR inner trial points may leave the support (h(c|x_i) < 0 on a
        # left-censored or (−∞, c] row makes log F NaN).  The likelihood turns
        # that NaN into InfeasibleParameterError and ``obj`` into the
        # infeasibility penalty, so numpy's "invalid value" notice is noise.
        # A both-finite interval row scores R's pexp value instead, log F(h_u)
        # or −inf, which is at most 0 and so keeps the subproblem bounded
        # (ADR 0005, #164 amendment).
        with np.errstate(invalid="ignore"):
            f, grad = obj(T_inv @ d)
        return float(f), np.asarray(T_inv.T @ np.asarray(grad), dtype=np.float64)

    # Scale-only path (#114): no nonlinear block; θ_b[0] ≥ b is the δ_0 bound.
    support_d: (
        Callable[
            [NDArray[np.float64]],
            tuple[NDArray[np.float64], NDArray[np.float64]],
        ]
        | None
    ) = None
    if support_nl is not None:
        support_block = support_nl

        def _support_d(
            d: NDArray[np.float64],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            values, jac = support_block(T_inv @ d)
            return values, jac @ T_inv

        support_d = _support_d

    def run(t: NDArray[np.float64]) -> AugLagResult:
        res = auglag_minimize(
            obj_d,
            T @ t,
            A_ineq=A_d,
            b_ineq=cm.b_ineq,
            C_eq=C_d,
            d_eq=cm.d_eq,
            options=options,
            bounds=box,
            nonlinear_ineq=support_d,
        )
        return replace(res, theta=np.asarray(T_inv @ res.theta, dtype=np.float64))

    return run


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
    support_floor: float,
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
        offset=offset if nonneg_lower else None,
        q_s=scaling.shape[1] if scaling is not None else 0,
        X_scale=scaling if nonneg_lower else None,
    )
    # Issue #115 — scaled exponential path with a shift block or a non-zero
    # offset: the support rows θ_b[0]·f_i + x_i·β ≥ −offset_i are the
    # nonlinear block cm.nonlinear_ineq (A_ineq holds monotonicity only).
    # Unlike the collapsed row of #114 below, the PHR penalty here is on
    # g_i = h(a|x_i) itself, which bounds every h(y_i|x_i) ≥ g_i from below
    # (h_0 non-decreasing): the smooth extension log f(h) = −h makes the NLL
    # fall at most linearly in a violated g_i, against a quadratic penalty, so
    # the subproblem stays bounded without a box.
    support_nl = cm.nonlinear_ineq
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
        # ``replace`` keeps the nonlinear support block (issue #115): the pins
        # are equality rows and compose with it unchanged.
        cm = replace(
            cm,
            C_eq=np.vstack([cm.C_eq, extra_rows]),
            d_eq=np.concatenate([cm.d_eq, extra_rhs]),
        )
    # Issue #114 — scaled exponential path without a shift block and with a
    # zero offset (the other cases are the nonlinear block above).  The
    # support rule has collapsed to the single row θ_b[0] ≥ 0, the last row
    # of A_ineq, but the smooth extension log f(h) = −h makes the objective
    # unbounded below once it is violated: with θ_b[0] = −t, the NLL falls
    # like −t·Σ_i exp(0.5·x_s,i·γ), exponentially in γ, against a PHR penalty
    # quadratic in t, so the PHR subproblem over ℝⁿ has no minimiser.  The
    # row is a pure bound, so it becomes the bound on δ_0 = θ_b[0] of the
    # increment-coordinate solve below; the row stays in A_ineq for reporting
    # and its multiplier is the bound multiplier ν_lo[0] (folded in below).
    support_row: int | None = None
    intercept_lower = -np.inf
    if nonneg_lower and scaling is not None and support_nl is None:
        support_row = cm.A_ineq.shape[0] - 1
        e0 = np.zeros(total_params, dtype=np.float64)
        e0[0] = 1.0
        if not np.array_equal(cm.A_ineq[support_row], e0):
            raise RuntimeError(
                "Expected the collapsed exponential support row [1, 0, …, 0] "
                "as the last inequality row on the scaled path; got "
                f"{cm.A_ineq[support_row].tolist()}."
            )
        intercept_lower = float(cm.b_ineq[support_row])
    true_nll = _make_true_nll(basis, y, X, censoring, dist, weights, offset, scaling)

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
        penalty=_infeasible_penalty(theta_init, true_nll),
    )
    auglag_opts = (
        config.auglag_options if config.auglag_options is not None else AugLagOptions()
    )

    def run_linear(t: NDArray[np.float64]) -> AugLagResult:
        return auglag_minimize(
            obj,
            t,
            A_ineq=cm.A_ineq,
            b_ineq=cm.b_ineq,
            C_eq=cm.C_eq,
            d_eq=cm.d_eq,
            options=auglag_opts,
        )

    # Every scaled exponential fit (#114 scale-only, #115 shift / offset)
    # solves in increment coordinates, where D θ_b ≥ 0 is a hard box in each
    # inner solve.  As a PHR row only, a violated monotonicity row lets
    # censored rows (no −log h' term) drive the NLL down exponentially in γ
    # (issue #161, _increment_auglag Notes).  Every other fit keeps the
    # θ-coordinate solve unchanged.
    use_increment_coords = nonneg_lower and scaling is not None
    run_auglag: Callable[[NDArray[np.float64]], AugLagResult] = run_linear
    if use_increment_coords:
        run_auglag = _increment_auglag(
            obj,
            cm,
            support_nl,
            n_params,
            total_params,
            auglag_opts,
            intercept_lower=intercept_lower,
        )

    best_auglag_result, best_nll, n_restarts_used = _restart_loop(
        theta_init,
        config,
        lambda t: _perturb_and_project(
            t,
            n_params,
            rng,
            nonneg_lower=nonneg_lower,
            support_floor=support_floor,
        ),
        run_auglag,
        lambda r: bool(r.converged),
        lambda r: true_nll(r.theta),
        message_fn=lambda r: str(r.message),
    )

    if best_auglag_result is None:
        return _fallback_result(
            theta_init,
            n_restarts_used,
            "All auglag attempts raised LinAlgError or ended at an infeasible θ.",
            lambda: true_nll(theta_init),
        )

    # Newton-CG polish: only for interior-MLE fits (no active inequalities, no
    # equality constraints — the unconstrained trust-ncg would violate lower/upper
    # boundary pins if cm.C_eq has rows).
    mu_ineq = best_auglag_result.mu_ineq
    if use_increment_coords:
        # Increment coordinates (_increment_auglag): monotonicity row k is the
        # bound δ_{k+1} ≥ 0, whose multiplier is ν_lo[k+1]; the PHR copy of
        # the row (the first n_params − 1 rows of A_ineq) never goes
        # infeasible and keeps a zero multiplier.
        mu_ineq = mu_ineq.copy()
        mu_ineq[: n_params - 1] += best_auglag_result.nu_lower[1:n_params]
        if support_row is not None:
            # Likewise the boxed support row θ_b[0] ≥ b (scale-only path):
            # its force is the bound multiplier on δ_0 = θ_b[0].
            mu_ineq[support_row] += best_auglag_result.nu_lower[0]
    final_theta = best_auglag_result.theta
    final_nll = best_nll
    if config.polish and cm.C_eq.shape[0] == 0 and np.all(mu_ineq < 1e-6):

        def shift_hess(t: NDArray[np.float64]) -> NDArray[np.float64]:
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

        # Feasibility of the polished θ is checked against every inequality
        # row with its right-hand side: monotonicity (b = 0) and, under the
        # exponential link, the support rows (b = −offset, issue #160).
        theta_polished = _ncg_polish(
            best_auglag_result.theta,
            obj,
            shift_hess,
            n_params,
            A_ineq=cm.A_ineq,
            b_ineq=cm.b_ineq,
            nonlinear_ineq=support_nl,
        )
        if theta_polished is not None:
            final_theta = theta_polished
            final_nll = true_nll(theta_polished)

    # Reported inequality rows line up with ``mu_ineq`` (linear rows, then the
    # nonlinear block).  The nonlinear rows enter as their linearisation
    # J(θ̂) at the returned θ — the constraint normals that the KKT
    # conditions and the active-set covariance (model._constrained_vcov_active)
    # use.
    a_ineq_report = cm.A_ineq
    if support_nl is not None:
        a_ineq_report = np.vstack([cm.A_ineq, support_nl(final_theta)[1]])
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
        mu_ineq=mu_ineq,
        lambda_eq=best_auglag_result.lambda_eq,
        constraint_A_ineq=a_ineq_report,
        constraint_C_eq=c_eq,
    )
