"""Augmented Lagrangian (PHR) optimiser for linear equality/inequality constraints.

Implements the Penalty-Homotopy-Relaxation (PHR) augmented Lagrangian
algorithm of Birgin & Martínez (2014).  All constraints must be linear:

    A_ineq @ theta >= b_ineq   (inequality)
    C_eq   @ theta  = d_eq     (equality)

The augmented Lagrangian is:

    L_A(θ; λ, μ, ρ) = f(θ)
                     − λᵀh(θ) + (ρ/2)‖h(θ)‖²
                     + (1/(2ρ)) Σᵢ [max(0, μᵢ − ρ gᵢ(θ))² − μᵢ²]

where h(θ) = C θ − d (equality residuals) and g(θ) = A θ − b (should be ≥ 0).

Gradient:
    ∇L_A = ∇f − Cᵀλ + ρ Cᵀh(θ) − Aᵀ max(0, μ − ρ g(θ))

Outer (multiplier) updates after each inner solve at θ⁺:
    λ ← λ − ρ h(θ⁺)
    μ ← max(0, μ − ρ g(θ⁺))
    ρ ← γρ   only when max-violation failed to shrink by factor τ

Convergence: max(‖h‖_∞, ‖min(g, μ/ρ)‖_∞, ‖∇L_A‖_∞) < outer_tol.

Default parameters are chosen to match R's alabama::auglag behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass
class AugLagOptions:
    """Configuration for the augmented Lagrangian (PHR) solver.

    Parameters
    ----------
    rho_init:
        Initial penalty parameter ρ₀.  alabama default: 10.
    rho_growth:
        Penalty growth factor γ: ρ ← γ·ρ when violation fails to shrink.
        alabama default: 10.
    violation_shrink_factor:
        Shrink threshold τ: penalty update fires when
        ``max_violation > τ · prev_max_violation``.  alabama default: 0.25.
    outer_tol:
        Convergence tolerance on the KKT residual.  alabama default: 1e-7.
    max_outer_iter:
        Maximum number of outer (multiplier-update) iterations.
        alabama default: 50.
    rho_max:
        Upper bound on the penalty parameter ρ.  PHR theory only requires ρ
        to be "large enough"; once it exceeds this cap further growth is
        suppressed.  Without a cap, near-machine-precision residuals can
        repeatedly fail the shrink test and inflate ρ until ``ρ·g`` swamps
        ``μ`` in the multiplier update, zeroing the dual estimate.
        Default ``1e8``.
    feas_tol:
        Feasibility tolerance used by the stall-based convergence test.  An
        iterate counts as *feasible* when ``max(‖h‖∞, ‖max(0, −g)‖∞) <
        feas_tol``.  Default ``1e-7`` (matches ``outer_tol``).
    theta_tol:
        Relative parameter-change tolerance for the stall-based convergence
        test.  The outer loop also declares convergence when the iterate is
        feasible *and* the inter-outer-iteration step has stalled,
        ``‖θ⁺ − θ‖∞ ≤ theta_tol · (1 + ‖θ⁺‖∞)``.  This mirrors
        ``alabama::auglag`` (which R ``mlt`` uses), whose outer loop stops on a
        small parameter change rather than on the raw KKT residual.  It matters
        on degenerate active sets (stacked monotonicity boundaries) where the
        augmented-Lagrangian stationarity floors at ~1e-5 even though θ has
        stopped moving and matches the reference fit to many decimals.  Default
        ``1e-8``.
    inner_method:
        ``scipy.optimize.minimize`` method used for each unconstrained inner
        solve.  Default ``"L-BFGS-B"`` matches alabama's inner solver.
    inner_options:
        Options dict passed to ``scipy.optimize.minimize`` for the inner
        solve.  Default ``{"gtol": 1e-8, "ftol": 1e-15, "maxiter": 500}`` —
        a near-zero ``ftol`` prevents L-BFGS-B from terminating on the
        function-value criterion (which would leave the gradient looser
        than the requested ``gtol``) and would otherwise stall the outer
        KKT check at ``stationarity ≈ 1e-5``.
    """

    rho_init: float = 10.0
    rho_growth: float = 10.0
    violation_shrink_factor: float = 0.25
    outer_tol: float = 1e-7
    max_outer_iter: int = 50
    rho_max: float = 1e8
    feas_tol: float = 1e-7
    theta_tol: float = 1e-8
    inner_method: str = "L-BFGS-B"
    inner_options: dict[str, object] = field(
        default_factory=lambda: {"gtol": 1e-8, "ftol": 1e-15, "maxiter": 500}
    )


@dataclass
class AugLagResult:
    """Result of an :func:`auglag_minimize` call.

    Parameters
    ----------
    theta:
        Optimised parameter vector, shape (n,).
    fun:
        Objective value f(θ) at the final θ (not the augmented Lagrangian).
    n_outer_iter:
        Number of outer (multiplier-update) iterations completed.
    n_inner_iter:
        Total inner (L-BFGS-B) iterations summed across all outer steps.
    converged:
        Whether the KKT residual fell below ``outer_tol``.
    kkt_residual:
        KKT residual at θ:
        ``max(‖h‖_∞, ‖min(g, μ/ρ)‖_∞, ‖∇L_A‖_∞)``.
    message:
        Human-readable convergence status.
    lambda_eq:
        Final equality multipliers, shape (m_eq,).
    mu_ineq:
        Final inequality multipliers (≥ 0), shape (m_ineq,).
    rho_final:
        Final penalty parameter ρ.
    """

    theta: NDArray[np.float64]
    fun: float
    n_outer_iter: int
    n_inner_iter: int
    converged: bool
    kkt_residual: float
    message: str
    lambda_eq: NDArray[np.float64]
    mu_ineq: NDArray[np.float64]
    rho_final: float


def _make_auglag_obj(
    fun: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    lambda_eq: NDArray[np.float64],
    mu_ineq: NDArray[np.float64],
    rho: float,
    A_ineq: NDArray[np.float64],
    b_ineq: NDArray[np.float64],
    C_eq: NDArray[np.float64],
    d_eq: NDArray[np.float64],
    m_ineq: int,
    m_eq: int,
) -> Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]:
    """Factory: return the augmented Lagrangian closure for one outer iteration.

    Copies multipliers by value so updates in the outer loop do not bleed
    into an already-dispatched inner solve.
    """
    lam = lambda_eq.copy()
    mu = mu_ineq.copy()
    rho_ = float(rho)

    def obj(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        f, grad_f = fun(theta)
        L_A = float(f)
        grad = np.array(grad_f, dtype=np.float64)

        if m_eq > 0:
            h = C_eq @ theta - d_eq
            L_A += -float(lam @ h) + 0.5 * rho_ * float(h @ h)
            grad = grad - C_eq.T @ lam + rho_ * (C_eq.T @ h)

        if m_ineq > 0:
            g = A_ineq @ theta - b_ineq
            phi = np.maximum(0.0, mu - rho_ * g)
            L_A += (0.5 / rho_) * (float(phi @ phi) - float(mu @ mu))
            grad = grad - A_ineq.T @ phi

        return L_A, grad

    return obj


def auglag_minimize(
    fun: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    x0: NDArray[np.float64],
    *,
    A_ineq: NDArray[np.float64] | None,
    b_ineq: NDArray[np.float64] | None,
    C_eq: NDArray[np.float64] | None,
    d_eq: NDArray[np.float64] | None,
    options: AugLagOptions | None = None,
) -> AugLagResult:
    """Minimise a smooth objective subject to linear equality/inequality constraints.

    Parameters
    ----------
    fun:
        Objective returning ``(f, ∇f)``.  Must be smooth; ``∇f`` is the
        gradient at ``theta``.
    x0:
        Starting point, shape (n,).
    A_ineq:
        Inequality constraint matrix, shape (m_ineq, n): ``A_ineq @ θ ≥ b_ineq``.
        Pass ``None`` or a zero-row matrix to omit.
    b_ineq:
        Right-hand side for inequality constraints, shape (m_ineq,).
        Defaults to zeros when ``A_ineq`` is provided without ``b_ineq``.
    C_eq:
        Equality constraint matrix, shape (m_eq, n): ``C_eq @ θ = d_eq``.
        Pass ``None`` or a zero-row matrix to omit.
    d_eq:
        Right-hand side for equality constraints, shape (m_eq,).
        Defaults to zeros when ``C_eq`` is provided without ``d_eq``.
    options:
        :class:`AugLagOptions` controlling the outer loop.  Defaults to
        :class:`AugLagOptions` with all default values (alabama-parity).

    Returns
    -------
    AugLagResult
        Contains the optimised θ, multipliers, convergence status, and KKT
        residual.

    Notes
    -----
    The PHR augmented Lagrangian is minimised by an outer loop that updates
    multipliers and (occasionally) the penalty ρ.  Each outer step delegates
    an unconstrained inner solve to L-BFGS-B (configurable via
    ``AugLagOptions.inner_method``).

    References
    ----------
    Birgin, E. G. & Martínez, J. M. (2014). *Practical Augmented Lagrangian
    Methods for Constrained Optimization*.  SIAM.
    """
    if options is None:
        options = AugLagOptions()

    x0_arr = np.asarray(x0, dtype=np.float64)
    n = x0_arr.size

    # Normalise both sides to concrete arrays; zero-row when absent.
    if A_ineq is None or np.asarray(A_ineq).shape[0] == 0:
        m_ineq = 0
        A_ = np.zeros((0, n), dtype=np.float64)
        b_ = np.zeros(0, dtype=np.float64)
    else:
        A_ = np.asarray(A_ineq, dtype=np.float64)
        m_ineq = A_.shape[0]
        b_ = (
            np.zeros(m_ineq, dtype=np.float64)
            if b_ineq is None
            else np.asarray(b_ineq, dtype=np.float64)
        )

    if C_eq is None or np.asarray(C_eq).shape[0] == 0:
        m_eq = 0
        C_ = np.zeros((0, n), dtype=np.float64)
        d_ = np.zeros(0, dtype=np.float64)
    else:
        C_ = np.asarray(C_eq, dtype=np.float64)
        m_eq = C_.shape[0]
        d_ = (
            np.zeros(m_eq, dtype=np.float64)
            if d_eq is None
            else np.asarray(d_eq, dtype=np.float64)
        )

    lambda_eq = np.zeros(m_eq, dtype=np.float64)
    mu_ineq = np.zeros(m_ineq, dtype=np.float64)
    rho = float(options.rho_init)

    theta = x0_arr.copy()
    prev_max_violation = np.inf
    n_outer_iter = 0
    n_inner_iter = 0
    converged = False
    kkt_residual = np.inf
    message = "Maximum outer iterations reached without convergence."

    # Track the best (lowest-KKT) feasible iterate seen.  On degenerate active
    # sets the per-outer-iteration KKT residual can bounce around its floor; we
    # return the best iterate rather than whichever one the loop happened to
    # exit on.  Seeded with the starting point so this is never ``None``.
    best_theta = theta.copy()
    best_kkt = np.inf
    best_lambda_eq = lambda_eq.copy()
    best_mu_ineq = mu_ineq.copy()

    for outer_iter in range(options.max_outer_iter):
        n_outer_iter = outer_iter + 1

        inner_obj = _make_auglag_obj(
            fun, lambda_eq, mu_ineq, rho, A_, b_, C_, d_, m_ineq, m_eq
        )
        try:
            result = minimize(
                inner_obj,
                theta,
                method=options.inner_method,
                jac=True,
                options=options.inner_options,
            )
        except LinAlgError:
            break
        theta_new = np.asarray(result.x, dtype=np.float64)
        n_inner_iter += int(getattr(result, "nit", 0))
        # Step taken by this outer iteration, used by the stall test below.
        theta_step = float(np.max(np.abs(theta_new - theta))) if theta.size else 0.0
        theta = theta_new

        # Constraint residuals at new θ
        h = C_ @ theta - d_ if m_eq > 0 else np.zeros(0, dtype=np.float64)
        g = A_ @ theta - b_ if m_ineq > 0 else np.zeros(0, dtype=np.float64)

        # Gradient of L_A at new θ with current multipliers
        _f, grad_f = fun(theta)
        grad_L_A = np.array(grad_f, dtype=np.float64)
        if m_eq > 0:
            grad_L_A = grad_L_A - C_.T @ lambda_eq + rho * (C_.T @ h)
        if m_ineq > 0:
            phi = np.maximum(0.0, mu_ineq - rho * g)
            grad_L_A = grad_L_A - A_.T @ phi

        # KKT residual: equality violation, inequality complementarity, and
        # stationarity.  ``feasibility`` is the raw primal constraint violation
        # (no complementarity), used by the penalty update and the stall test.
        eq_viol = float(np.max(np.abs(h))) if m_eq > 0 else 0.0
        ineq_kkt = (
            float(np.max(np.abs(np.minimum(g, mu_ineq / rho)))) if m_ineq > 0 else 0.0
        )
        ineq_viol = float(np.max(np.maximum(0.0, -g))) if m_ineq > 0 else 0.0
        stationarity = float(np.max(np.abs(grad_L_A)))
        kkt_residual = max(eq_viol, ineq_kkt, stationarity)
        feasibility = max(eq_viol, ineq_viol)

        if kkt_residual < best_kkt:
            best_kkt = kkt_residual
            best_theta = theta.copy()
            best_lambda_eq = lambda_eq.copy()
            best_mu_ineq = mu_ineq.copy()

        if kkt_residual < options.outer_tol:
            converged = True
            message = "KKT conditions satisfied."
            break

        # Stall-based convergence (alabama parity): a feasible iterate whose
        # parameters have stopped moving between outer iterations is accepted
        # even when the augmented-Lagrangian stationarity has floored above
        # ``outer_tol``.  This is the documented degenerate-active-set regime
        # (stacked monotonicity boundaries) — further outer iterations only
        # perturb θ within noise while inflating wasted work.  Guard with
        # ``outer_iter >= 1`` so the very first (often large) inner step never
        # short-circuits the loop.
        if (
            outer_iter >= 1
            and feasibility < options.feas_tol
            and theta_step <= options.theta_tol * (1.0 + float(np.max(np.abs(theta))))
        ):
            converged = True
            message = (
                "Converged: feasible and parameter change below tolerance "
                f"(KKT residual {kkt_residual:.2e} floored on an active set)."
            )
            # The current θ is the converged point; record it as best.
            best_kkt = kkt_residual
            best_theta = theta.copy()
            best_lambda_eq = lambda_eq.copy()
            best_mu_ineq = mu_ineq.copy()
            break

        # Multiplier updates
        if m_eq > 0:
            lambda_eq = lambda_eq - rho * h
        if m_ineq > 0:
            mu_ineq = np.maximum(0.0, mu_ineq - rho * g)

        # Penalty update: grow ρ only when the iterate is still infeasible AND
        # the violation failed to shrink by the required factor AND ρ is below
        # its cap.  Gating on ``feasibility > feas_tol`` is essential: once the
        # constraints are satisfied, continuing to grow ρ (as the bare
        # shrink-test would, since a tiny residual still "fails" relative to an
        # even tinier previous one) drives ρ to ``rho_max``, wrecks the inner
        # problem's conditioning, and *degrades* an already-good iterate
        # (stationarity climbing back from ~1e-5 to ~1e-2).
        current_max_violation = feasibility
        insufficient_decrease = (
            current_max_violation > options.violation_shrink_factor * prev_max_violation
        )
        if (
            current_max_violation > options.feas_tol
            and insufficient_decrease
            and rho < options.rho_max
        ):
            rho = min(options.rho_growth * rho, options.rho_max)
        prev_max_violation = current_max_violation

    theta = best_theta
    lambda_eq = best_lambda_eq
    mu_ineq = best_mu_ineq
    kkt_residual = best_kkt
    f_final = float(fun(theta)[0])

    return AugLagResult(
        theta=theta,
        fun=f_final,
        n_outer_iter=n_outer_iter,
        n_inner_iter=n_inner_iter,
        converged=converged,
        kkt_residual=float(kkt_residual),
        message=message,
        lambda_eq=lambda_eq,
        mu_ineq=mu_ineq,
        rho_final=rho,
    )
