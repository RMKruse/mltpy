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
        theta = np.asarray(result.x, dtype=np.float64)
        n_inner_iter += int(getattr(result, "nit", 0))

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

        # KKT residual
        eq_viol = float(np.max(np.abs(h))) if m_eq > 0 else 0.0
        ineq_kkt = (
            float(np.max(np.abs(np.minimum(g, mu_ineq / rho)))) if m_ineq > 0 else 0.0
        )
        stationarity = float(np.max(np.abs(grad_L_A)))
        kkt_residual = max(eq_viol, ineq_kkt, stationarity)

        if kkt_residual < options.outer_tol:
            converged = True
            message = "KKT conditions satisfied."
            break

        # Multiplier updates
        if m_eq > 0:
            lambda_eq = lambda_eq - rho * h
        if m_ineq > 0:
            mu_ineq = np.maximum(0.0, mu_ineq - rho * g)

        # Penalty update: grow ρ only when violation failed to shrink enough
        # AND we have not yet hit ``rho_max``.  The cap matters because once ρ
        # is so large that ``ρ·g`` exceeds ``μ`` for residuals near machine
        # precision, the multiplier update collapses to zero — see the
        # ``rho_max`` docstring on :class:`AugLagOptions`.
        current_max_violation = max(eq_viol, ineq_kkt)
        if (
            current_max_violation > options.violation_shrink_factor * prev_max_violation
            and rho < options.rho_max
        ):
            rho = min(options.rho_growth * rho, options.rho_max)
        prev_max_violation = current_max_violation

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
