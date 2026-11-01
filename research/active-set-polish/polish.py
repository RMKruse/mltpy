"""Prototype: active-set null-space Newton polish for auglag fits (#177).

Research code, not library code. It implements the #177 decisions (as
revised after #178) so their gain and cost can be measured before any
production change.

Problem
-------
Minimise the negative log-likelihood ``f(θ)`` subject to::

    A θ ≥ b          linear inequalities (monotonicity, support rows)
    g(θ) ≥ 0         smooth nonlinear inequalities (scaled support rows)
    C θ = d          equalities (lower / upper pins, fixed_params)

starting from auglag's returned point ``θ₀``.

Method (Nocedal & Wright, *Numerical Optimization*, 2nd ed., §16.5, §18.4)
--------------------------------------------------------------------------
1. **Working set** ``W``: every equality row and every inequality row whose
   normalised slack ``s_i/‖∇s_i‖`` is at most ``act_tol`` (Decision 2).
2. **Restoration**: move θ onto the working rows with minimum-norm
   Gauss-Newton corrections ``θ ← θ − M_W⁺ r_W(θ)``.  For linear rows one
   step is exact; nonlinear rows converge quadratically (Decision 3).
3. **Reduced Newton step**: ``Z`` spans ``null(M_W)`` from an SVD with a rank
   tolerance, so linearly dependent working rows are harmless (Decision 1).
   The step is ``p = −Z (Zᵀ ∇²L Z)⁻¹ Zᵀ ∇f``, with the Lagrangian Hessian
   ``∇²L = ∇²f − Σ_{i∈W} λ_i ∇²g_i`` (nonlinear rows only).
4. **Ratio test**: if the step would cross an inactive row (to first order),
   it is shortened to that row, and the row joins ``W`` (Decision 2
   revised).
5. **Multipliers**: ``λ`` from least squares on ``∇f = M_Wᵀ λ``.  When the
   reduced gradient is small but an inequality multiplier is clearly
   negative, the most negative row leaves ``W``.
6. **Stop** when the reduced gradient is below ``gtol`` and no multiplier is
   clearly negative, or after ``max_iter`` iterations.

Acceptance (Decision 4) is applied by :func:`polish` to the final point:

- feasibility: every inequality row ≥ ``−feas_tol`` (normalised) and every
  equality row within ``feas_tol``;
- the stationarity residual is not higher than at ``θ₀``, up to round-off
  ``kkt_floor · max(1, ‖∇f(θ)‖∞)``.  At a constrained optimum ``∇f`` has the
  size of the multipliers, so residuals of 1e-11 to 1e-10 on a gradient of
  that size are round-off; the old "strictly lower" rule rejected exact
  boundary points on the 50-of-51-active fixture over such differences;
- the likelihood is not worse than ``f(θ₀) + Σ_{i∈W} |λ_i| |c_i(θ₀)|``.
  auglag leaves its active rows ~1e-10 to 1e-7 on either side of the
  boundary; moving them exactly onto it changes ``f`` by ``λ_i |c_i|`` to
  first order, which is the price of exact feasibility, not a worse fit
  (measured on the #160 fixture: 5.6e-9 ≈ 70 × 8e-11).

Otherwise ``θ₀`` is returned unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

Vec = NDArray[np.float64]
Mat = NDArray[np.float64]


@dataclass
class Problem:
    """The constrained problem around one auglag fit."""

    f: Callable[[Vec], tuple[float, Vec]]
    hess: Callable[[Vec], Mat]
    A: Mat
    b: Vec
    C: Mat
    d: Vec
    g: Callable[[Vec], tuple[Vec, Mat]] | None = None
    g_hess: Callable[[Vec, Vec], Mat] | None = None


@dataclass
class PolishResult:
    theta: Vec
    candidate: Vec
    accepted: bool
    reason: str
    n_iter: int
    n_active: int
    n_added: int
    n_dropped: int
    kkt_before: float
    kkt_after: float
    nll_before: float
    nll_after: float
    viol_before: float
    viol_after: float
    history: list[str] = field(default_factory=list)


class _Rows:
    """Inequality rows (linear, then nonlinear) and equality rows at θ."""

    def __init__(self, prob: Problem) -> None:
        self.p = prob
        self.m_lin = prob.A.shape[0]

    def ineq(self, theta: Vec) -> tuple[Vec, Mat]:
        vals = [self.p.A @ theta - self.p.b]
        jacs = [self.p.A]
        if self.p.g is not None:
            gv, gj = self.p.g(theta)
            vals.append(np.asarray(gv, dtype=np.float64))
            jacs.append(np.atleast_2d(np.asarray(gj, dtype=np.float64)))
        n = theta.size
        v = np.concatenate(vals) if vals else np.zeros(0)
        J = np.vstack(jacs) if jacs else np.zeros((0, n))
        return v, J

    def eq(self, theta: Vec) -> tuple[Vec, Mat]:
        return self.p.C @ theta - self.p.d, self.p.C

    @staticmethod
    def normalise(v: Vec, J: Mat) -> Vec:
        norms = np.linalg.norm(J, axis=1)
        return v / np.where(norms > 0.0, norms, 1.0)


def _null_space(M: Mat, n: int) -> Mat:
    if M.shape[0] == 0:
        return np.eye(n)
    _, s, vt = np.linalg.svd(M, full_matrices=True)
    tol = max(M.shape) * np.finfo(np.float64).eps * (s[0] if s.size else 0.0)
    rank = int(np.sum(s > max(tol, 1e-12 * (s[0] if s.size else 0.0))))
    return vt[rank:].T


def _working(rows: _Rows, theta: Vec, W: list[int]) -> tuple[Vec, Mat, int]:
    """Residuals and Jacobian of the working rows: [ineq W | eq]."""
    vi, Ji = rows.ineq(theta)
    ve, Je = rows.eq(theta)
    r = np.concatenate([vi[W], ve])
    M = np.vstack([Ji[W], Je]) if (W or Je.shape[0]) else np.zeros((0, theta.size))
    return r, M, len(W)


def _restore(rows: _Rows, theta: Vec, W: list[int], tol: float = 1e-13) -> Vec:
    for _ in range(8):
        r, M, _ = _working(rows, theta, W)
        if r.size == 0 or np.max(np.abs(r)) <= tol * (1.0 + np.max(np.abs(theta))):
            break
        delta, *_ = np.linalg.lstsq(M, r, rcond=None)
        theta = theta - delta
    return theta


def _multipliers(M: Mat, n_ineq: int, grad: Vec) -> tuple[Vec, Vec]:
    """(sign-constrained λ, min-norm unconstrained λ) for ∇f = Mᵀλ."""
    if M.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    lam_free, *_ = np.linalg.lstsq(M.T, grad, rcond=None)
    lb = np.concatenate([np.zeros(n_ineq), np.full(M.shape[0] - n_ineq, -np.inf)])
    ub = np.full(M.shape[0], np.inf)
    lam_pos = lsq_linear(M.T, grad, bounds=(lb, ub), method="bvls").x
    return lam_pos, lam_free


def stationarity(prob: Problem, theta: Vec, act_tol: float) -> tuple[float, float]:
    """(KKT stationarity residual, max violation) with sign-constrained λ.

    The active rows are those within ``act_tol`` (normalised); the residual is
    ``‖∇f − M_Aᵀλ‖∞`` with ``λ ≥ 0`` on inequality rows.  The same measure is
    applied before and after the polish.
    """
    rows = _Rows(prob)
    vi, Ji = rows.ineq(theta)
    ve, _ = rows.eq(theta)
    ns = rows.normalise(vi, Ji)
    A_idx = [int(i) for i in np.flatnonzero(ns <= act_tol)]
    _, grad = prob.f(theta)
    _, M, k = _working(rows, theta, A_idx)
    lam, _ = _multipliers(M, k, grad)
    res = grad - (M.T @ lam if M.shape[0] else 0.0)
    viol = max(
        float(-np.min(ns)) if ns.size else 0.0,
        float(np.max(np.abs(ve))) if ve.size else 0.0,
        0.0,
    )
    return float(np.max(np.abs(res))), viol


def _feasibility_price(prob: Problem, theta0: Vec, theta1: Vec, W: list[int]) -> float:
    """First-order NLL cost of moving θ₀'s working rows onto the boundary.

    ``Σ_i |λ_i| |c_i(θ₀)|`` over the final working set and the equality rows,
    with the sign-constrained multipliers at the polished point.
    """
    rows = _Rows(prob)
    _, grad = prob.f(theta1)
    r0, _, _ = _working(rows, theta0, W)
    _, M1, k = _working(rows, theta1, W)
    lam, _ = _multipliers(M1, k, grad)
    return float(np.sum(np.abs(lam) * np.abs(r0))) if lam.size else 0.0


def polish(
    prob: Problem,
    theta0: Vec,
    *,
    act_tol: float = 1e-6,
    gtol: float = 1e-10,
    mult_tol: float = 1e-8,
    feas_tol: float = 1e-9,
    kkt_floor: float = 1e-10,
    max_iter: int = 30,
) -> PolishResult:
    """Polish ``theta0`` on its active set; return θ₀ unless it improves."""
    rows = _Rows(prob)
    theta0 = np.asarray(theta0, dtype=np.float64)
    n = theta0.size
    nll0, _ = prob.f(theta0)
    kkt0, viol0 = stationarity(prob, theta0, act_tol)
    vi, Ji = rows.ineq(theta0)
    W = [int(i) for i in np.flatnonzero(rows.normalise(vi, Ji) <= act_tol)]
    n_added = n_dropped = 0
    history: list[str] = []
    theta = _restore(rows, theta0.copy(), W)
    it = 0
    gscale = max(1.0, abs(nll0))
    for it in range(1, max_iter + 1):
        fval, grad = prob.f(theta)
        _, M, k = _working(rows, theta, W)
        Z = _null_space(M, n)
        lam_pos, lam_free = _multipliers(M, k, grad)
        rg = Z.T @ grad if Z.size else np.zeros(0)
        rg_norm = float(np.max(np.abs(rg))) if rg.size else 0.0
        if rg_norm <= gtol * gscale:
            neg = lam_free[:k]
            if k and np.min(neg) < -mult_tol * gscale:
                j = int(np.argmin(neg))
                history.append(f"it{it}: drop row {W[j]} (λ={neg[j]:.1e})")
                W.pop(j)
                n_dropped += 1
                continue
            history.append(f"it{it}: converged |Zᵀ∇f|={rg_norm:.1e}")
            break
        H = prob.hess(theta)
        if prob.g_hess is not None and prob.g is not None and k:
            v = np.zeros(rows.ineq(theta)[0].size - rows.m_lin)
            for pos, row in enumerate(W):
                if row >= rows.m_lin:
                    v[row - rows.m_lin] = lam_pos[pos]
            if np.any(v):
                H = H - prob.g_hess(theta, v)
        Hr = Z.T @ H @ Z
        Hr = 0.5 * (Hr + Hr.T)
        w, V = np.linalg.eigh(Hr)
        floor = 1e-10 * max(1.0, float(np.max(np.abs(w))))
        w = np.maximum(w, floor)
        dstep = -(V @ ((V.T @ rg) / w))
        step = Z @ dstep
        # Ratio test against inactive rows (first order).
        vi, Ji = rows.ineq(theta)
        inactive = [i for i in range(vi.size) if i not in W]
        alpha_max, blocking = 1.0, None
        for i in inactive:
            rate = float(Ji[i] @ step)
            if rate < 0.0:
                a = max(float(vi[i]), 0.0) / -rate
                if a < alpha_max:
                    alpha_max, blocking = a, i
        alpha = alpha_max
        slope = float(grad @ step)
        # Near the optimum the predicted decrease −slope (~|Zᵀ∇f|²/H) falls to
        # the round-off of f, and Armijo then accepts or rejects at random.
        # Below that level, judge the step by the reduced gradient instead.
        f_resolution = 64 * np.finfo(np.float64).eps * gscale
        accepted_step = False
        for _ in range(40):
            trial = _restore(rows, theta + alpha * step, W)
            try:
                ftrial, gtrial = prob.f(trial)
            # Research prototype: any error counts as an infeasible trial
            # point. Production must catch only InfeasibleParameterError /
            # LinAlgError (#177 Decision 4; README finding 8).
            except Exception:
                ftrial, gtrial = np.inf, grad
            if not np.isfinite(ftrial):
                alpha *= 0.5
                continue
            if -alpha * slope <= f_resolution:
                if float(np.max(np.abs(Z.T @ gtrial))) < rg_norm:
                    accepted_step = True
                    break
            elif ftrial <= fval + 1e-4 * alpha * slope:
                accepted_step = True
                break
            alpha *= 0.5
        if not accepted_step:
            history.append(f"it{it}: line search failed |Zᵀ∇f|={rg_norm:.1e}")
            break
        theta = trial
        if blocking is not None and alpha == alpha_max:
            W.append(blocking)
            n_added += 1
            history.append(f"it{it}: add row {blocking} (α={alpha:.2e})")

    nll1, grad1 = prob.f(theta)
    kkt1, viol1 = stationarity(prob, theta, act_tol)
    allowance = _feasibility_price(prob, theta0, theta, W)
    kkt_limit = max(kkt0, kkt_floor * max(1.0, float(np.max(np.abs(grad1)))))
    reason = "accepted"
    ok = True
    if viol1 > feas_tol:
        ok, reason = False, f"infeasible ({viol1:.1e})"
    elif kkt1 > kkt_limit:
        ok, reason = False, f"KKT not lower ({kkt1:.1e} > {kkt_limit:.1e})"
    elif nll1 > nll0 + allowance + 1e-12 * gscale:
        ok, reason = (
            False,
            f"likelihood worse by {nll1 - nll0:.1e} (allowance {allowance:.1e})",
        )
    return PolishResult(
        theta=theta if ok else theta0,
        candidate=theta,
        accepted=ok,
        reason=reason,
        n_iter=it,
        n_active=len(W),
        n_added=n_added,
        n_dropped=n_dropped,
        kkt_before=kkt0,
        kkt_after=kkt1,
        nll_before=float(nll0),
        nll_after=float(nll1),
        viol_before=viol0,
        viol_after=viol1,
        history=history,
    )
