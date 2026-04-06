"""Log-likelihood functions for conditional transformation models.

All censoring types (exact, left, right, interval) are supported.
Optional linear regression shift via covariate matrix X.

Mathematical convention
-----------------------
Given a Bernstein basis B_k and coefficient vector theta_b (length p = order+1):

    h(y)  = B_k(y) @ theta_b  [+ X @ beta  if covariates are present]
    h'(y) = B_k'(y) @ theta_b  (first derivative; beta does not appear)

The target distribution is Standard Normal:  Z ~ N(0, 1).

Log-likelihood formulae
-----------------------
NONE (exact):
    ℓ = Σ_i [log φ(h_i) + log h'_i]

RIGHT (exact + right-censored):
    ℓ = Σ_{exact}    [log φ(h_i) + log h'_i]
      + Σ_{censored}  log Φ̄(h_i)         [Φ̄ = 1 - Φ, via norm.logsf]

LEFT (exact + left-censored):
    ℓ = Σ_{exact}    [log φ(h_i) + log h'_i]
      + Σ_{censored}  log Φ(h_i)          [via scipy.special.log_ndtr]

INTERVAL (interval-censored [l_i, u_i]):
    ℓ = Σ_i log(Φ(h(u_i)) − Φ(h(l_i)))  [via _log_diff_ndtr]

Numerical stability
-------------------
- All h values are clipped to [-30, 30] before norm calls.
- log CDF is computed via scipy.special.log_ndtr, not log(cdf(...)).
- log(1 − F) is computed via norm.logsf, not log(1 − cdf(...)).
- log(Φ(b) − Φ(a)) uses a logsumexp trick with a Taylor fallback for
  very narrow intervals (see _log_diff_ndtr).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr
from scipy.stats import norm

from pymlt.basis import BernsteinBasis
from pymlt.variables import CensoredData, CensoringType

# Clipping range for h before normal-distribution calls.
_H_CLIP = 30.0


# ---------------------------------------------------------------------------
# Numerically stable log(Φ(b) − Φ(a))
# ---------------------------------------------------------------------------

def _log_diff_ndtr(a: NDArray, b: NDArray) -> NDArray:
    """Compute log(Φ(b) − Φ(a)) in a numerically stable way.

    Requires b >= a element-wise.

    For *wide* intervals the logsumexp identity is used::

        log(Φ(b) − Φ(a)) = log Φ(b) + log(1 − exp(log Φ(a) − log Φ(b)))

    For *narrow* intervals (Φ(a) ≈ Φ(b)) the first-order Taylor approximation
    is used::

        Φ(b) − Φ(a) ≈ φ(mid) · (b − a),   mid = (a + b) / 2

    Parameters
    ----------
    a, b:
        Lower and upper argument.  Must satisfy a <= b.

    Returns
    -------
    NDArray, same shape as a/b.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    log_Fa = log_ndtr(a)          # log Φ(a)
    log_Fb = log_ndtr(b)          # log Φ(b)
    ratio  = log_Fa - log_Fb      # <= 0  (since a <= b)

    # Wide-interval path: ratio well below 0 → log1p stable
    # Clip ratio away from 0 to prevent log1p(-exp(0)) = -inf
    ratio_safe = np.minimum(ratio, -1e-15)
    wide = log_Fb + np.log1p(-np.exp(ratio_safe))

    # Narrow-interval fallback: Φ(b)-Φ(a) ≈ φ(mid)·(b−a)
    mid  = 0.5 * (a + b)
    width = np.maximum(b - a, np.finfo(float).tiny)
    narrow = norm.logpdf(mid) + np.log(width)

    # Use fallback when ratio > -1e-6  (i.e. Φ(a)/Φ(b) > 1-1e-6)
    return np.where(ratio < -1e-6, wide, narrow)


# ---------------------------------------------------------------------------
# Helpers: split theta into basis coefficients and (optional) regression beta
# ---------------------------------------------------------------------------

def _split_theta(
    theta: NDArray, p: int, X: Optional[NDArray]
) -> tuple[NDArray, Optional[NDArray]]:
    """Return (theta_b, beta) where theta_b has length p."""
    theta_b = theta[:p]
    beta    = theta[p:] if X is not None else None
    return theta_b, beta


def _shift(h: NDArray, X: Optional[NDArray], beta: Optional[NDArray]) -> NDArray:
    """Add regression shift X @ beta to h if X is provided."""
    if X is not None and beta is not None:
        return h + X @ beta
    return h


# ---------------------------------------------------------------------------
# Private log-likelihood functions — one per censoring type
# ---------------------------------------------------------------------------

def _ll_none(
    y: NDArray,
    theta: NDArray,
    basis: BernsteinBasis,
    X: Optional[NDArray],
) -> float:
    """ℓ = Σ [log φ(h_i) + log h'_i]  (exact observations)."""
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)

    B  = basis.evaluate(y)              # (n, p)
    D  = basis.derivative(y, order=1)   # (n, p)
    h  = np.clip(_shift(B @ theta_b, X, beta), -_H_CLIP, _H_CLIP)
    hp = D @ theta_b                    # h-prime; must be > 0

    with np.errstate(invalid="ignore", divide="ignore"):
        # np.log(hp) produces -inf/nan when hp <= 0 (monotonicity violated).
        # log_likelihood() detects and raises ValueError after this call.
        return float(np.sum(norm.logpdf(h)) + np.sum(np.log(hp)))


def _ll_right(
    cd: CensoredData,
    theta: NDArray,
    basis: BernsteinBasis,
    X: Optional[NDArray],
) -> float:
    """ℓ = Σ_exact [log φ(h) + log h'] + Σ_censored log Φ̄(h)."""
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)
    ll = 0.0

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        B_e = basis.evaluate(y_e)
        D_e = basis.derivative(y_e, order=1)
        h_e  = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        ll  += float(np.sum(norm.logpdf(h_e)) + np.sum(np.log(hp_e)))

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]          # last known lower bound
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        ll += float(np.sum(norm.logsf(h_c)))

    return ll


def _ll_left(
    cd: CensoredData,
    theta: NDArray,
    basis: BernsteinBasis,
    X: Optional[NDArray],
) -> float:
    """ℓ = Σ_exact [log φ(h) + log h'] + Σ_censored log Φ(h)."""
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)
    ll = 0.0

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        B_e = basis.evaluate(y_e)
        D_e = basis.derivative(y_e, order=1)
        h_e  = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        ll  += float(np.sum(norm.logpdf(h_e)) + np.sum(np.log(hp_e)))

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]          # last known upper bound
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        ll += float(np.sum(log_ndtr(h_c)))

    return ll


def _ll_interval(
    cd: CensoredData,
    theta: NDArray,
    basis: BernsteinBasis,
    X: Optional[NDArray],
) -> float:
    """ℓ = Σ log(Φ(h(upper_i)) − Φ(h(lower_i)))  [+ exact terms if present]."""
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)
    ll = 0.0

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        B_e = basis.evaluate(y_e)
        D_e = basis.derivative(y_e, order=1)
        h_e  = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        ll  += float(np.sum(norm.logpdf(h_e)) + np.sum(np.log(hp_e)))

    mask_i = cd.is_interval_censored_mask
    if mask_i.any():
        lo = cd.lower[mask_i]
        hi = cd.upper[mask_i]
        X_i = X[mask_i] if X is not None else None
        B_lo = basis.evaluate(lo)
        B_hi = basis.evaluate(hi)
        shift = (X_i @ beta) if (X_i is not None and beta is not None) else 0.0
        h_lo = np.clip(B_lo @ theta_b + shift, -_H_CLIP, _H_CLIP)
        h_hi = np.clip(B_hi @ theta_b + shift, -_H_CLIP, _H_CLIP)
        ll += float(np.sum(_log_diff_ndtr(h_lo, h_hi)))

    return ll


# ---------------------------------------------------------------------------
# Private gradient functions — one per censoring type
# (all return gradient of the NEGATIVE log-likelihood)
# ---------------------------------------------------------------------------

def _grad_none(
    y: NDArray,
    theta: NDArray,
    basis: BernsteinBasis,
    X: Optional[NDArray],
) -> NDArray:
    """∂(-ℓ)/∂θ = B.T @ h − D.T @ (1/h')  [+ X.T @ h for beta part]."""
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)

    B  = basis.evaluate(y)              # (n, p)
    D  = basis.derivative(y, order=1)   # (n, p)
    h  = np.clip(_shift(B @ theta_b, X, beta), -_H_CLIP, _H_CLIP)
    hp = D @ theta_b

    grad_b = B.T @ h - D.T @ (1.0 / hp)          # (p,)

    if X is not None and beta is not None:
        grad_beta = X.T @ h                        # (q,)
        return np.concatenate([grad_b, grad_beta])
    return grad_b


def _grad_right(
    cd: CensoredData,
    theta: NDArray,
    basis: BernsteinBasis,
    X: Optional[NDArray],
) -> NDArray:
    """Gradient of -ℓ for right-censored data."""
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    theta_b, beta = _split_theta(theta, p, X)
    grad = np.zeros(p + q)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        B_e = basis.evaluate(y_e)
        D_e = basis.derivative(y_e, order=1)
        h_e  = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        grad[:p] += B_e.T @ h_e - D_e.T @ (1.0 / hp_e)
        if X_e is not None:
            grad[p:] += X_e.T @ h_e

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        # ∂(-ℓ)/∂θ_b from censored = +B_c.T @ [φ(h)/Φ̄(h)]
        hazard = np.exp(norm.logpdf(h_c) - norm.logsf(h_c))  # φ/Φ̄
        grad[:p] += B_c.T @ hazard
        if X_c is not None:
            grad[p:] += X_c.T @ hazard

    return grad


def _grad_left(
    cd: CensoredData,
    theta: NDArray,
    basis: BernsteinBasis,
    X: Optional[NDArray],
) -> NDArray:
    """Gradient of -ℓ for left-censored data."""
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    theta_b, beta = _split_theta(theta, p, X)
    grad = np.zeros(p + q)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        B_e = basis.evaluate(y_e)
        D_e = basis.derivative(y_e, order=1)
        h_e  = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        grad[:p] += B_e.T @ h_e - D_e.T @ (1.0 / hp_e)
        if X_e is not None:
            grad[p:] += X_e.T @ h_e

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        # ∂ log Φ(h)/∂θ = +[φ(h)/Φ(h)] · B
        # ∂(-ℓ)/∂θ_b from censored = -B_c.T @ [φ(h)/Φ(h)]
        inv_mills = np.exp(norm.logpdf(h_c) - log_ndtr(h_c))   # φ/Φ
        grad[:p] -= B_c.T @ inv_mills
        if X_c is not None:
            grad[p:] -= X_c.T @ inv_mills

    return grad


def _grad_interval(
    cd: CensoredData,
    theta: NDArray,
    basis: BernsteinBasis,
    X: Optional[NDArray],
) -> NDArray:
    """Gradient of -ℓ for interval-censored data."""
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    theta_b, beta = _split_theta(theta, p, X)
    grad = np.zeros(p + q)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        B_e = basis.evaluate(y_e)
        D_e = basis.derivative(y_e, order=1)
        h_e  = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        grad[:p] += B_e.T @ h_e - D_e.T @ (1.0 / hp_e)
        if X_e is not None:
            grad[p:] += X_e.T @ h_e

    mask_i = cd.is_interval_censored_mask
    if mask_i.any():
        lo = cd.lower[mask_i]
        hi = cd.upper[mask_i]
        X_i = X[mask_i] if X is not None else None
        B_lo = basis.evaluate(lo)
        B_hi = basis.evaluate(hi)
        shift = (X_i @ beta) if (X_i is not None and beta is not None) else 0.0
        h_lo = np.clip(B_lo @ theta_b + shift, -_H_CLIP, _H_CLIP)
        h_hi = np.clip(B_hi @ theta_b + shift, -_H_CLIP, _H_CLIP)

        log_p  = _log_diff_ndtr(h_lo, h_hi)                    # log(Φ(h_hi) − Φ(h_lo))
        w_hi   = np.exp(norm.logpdf(h_hi) - log_p)             # φ(h_hi) / p
        w_lo   = np.exp(norm.logpdf(h_lo) - log_p)             # φ(h_lo) / p

        # ∂ℓ/∂θ_b = B_hi.T@w_hi − B_lo.T@w_lo
        # ∂(-ℓ)/∂θ_b = -(B_hi.T@w_hi − B_lo.T@w_lo)
        grad[:p] -= B_hi.T @ w_hi - B_lo.T @ w_lo
        if X_i is not None:
            grad[p:] -= X_i.T @ (w_hi - w_lo)

    return grad


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_likelihood(
    theta: NDArray,
    basis: BernsteinBasis,
    y: NDArray | CensoredData,
    X: Optional[NDArray] = None,
    censoring: CensoringType = CensoringType.NONE,
) -> float:
    """Log-likelihood of a conditional transformation model.

    Parameters
    ----------
    theta:
        Parameter vector.  First ``basis.order + 1`` entries are Bernstein
        coefficients (must be non-decreasing for a valid model).  Remaining
        entries are regression coefficients beta (only if X is given).
    basis:
        ``BernsteinBasis`` instance defining the transformation.
    y:
        Observations.  If ``NDArray``, treated as exact (ignores ``censoring``).
        If ``CensoredData``, the censoring type must be specified via
        ``censoring``.
    X:
        Covariate matrix of shape (n, q), or None for no regression.
    censoring:
        Censoring regime.  Only used when ``y`` is a ``CensoredData`` object.

    Returns
    -------
    float  (log-likelihood value)

    Raises
    ------
    ValueError
        If the result is ``-inf`` or ``NaN``.  Most likely causes: theta
        violates monotonicity (h' ≤ 0), observations outside support, or
        numerical overflow in the basis evaluation.
    """
    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        result = _ll_none(y_arr, theta, basis, X)
    else:
        if censoring is CensoringType.NONE:
            result = _ll_none(y.exact, theta, basis, X)
        elif censoring is CensoringType.RIGHT:
            result = _ll_right(y, theta, basis, X)
        elif censoring is CensoringType.LEFT:
            result = _ll_left(y, theta, basis, X)
        else:  # INTERVAL
            result = _ll_interval(y, theta, basis, X)

    if not np.isfinite(result):
        raise ValueError(
            f"log_likelihood returned {result}.  Possible causes: theta "
            "violates monotonicity (h'(y) ≤ 0), observations outside basis "
            "support, or extreme h values despite clipping."
        )
    return result


def negative_log_likelihood(
    theta: NDArray,
    basis: BernsteinBasis,
    y: NDArray | CensoredData,
    X: Optional[NDArray] = None,
    censoring: CensoringType = CensoringType.NONE,
    gradient: bool = False,
) -> float | tuple[float, NDArray]:
    """Negative log-likelihood (objective for minimisation) with optional gradient.

    Parameters
    ----------
    theta, basis, y, X, censoring:
        Same as :func:`log_likelihood`.
    gradient:
        If ``True``, return a ``(nll, grad)`` tuple where ``grad`` is the
        analytical gradient of the *negative* log-likelihood w.r.t. ``theta``.
        Computed analytically — no finite-difference approximation.

    Returns
    -------
    float  when ``gradient=False``
    (float, NDArray)  when ``gradient=True``
    """
    nll = -log_likelihood(theta, basis, y, X, censoring)

    if not gradient:
        return nll

    # Analytical gradient of the negative log-likelihood
    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        grad = _grad_none(y_arr, theta, basis, X)
    else:
        if censoring is CensoringType.NONE:
            grad = _grad_none(y.exact, theta, basis, X)
        elif censoring is CensoringType.RIGHT:
            grad = _grad_right(y, theta, basis, X)
        elif censoring is CensoringType.LEFT:
            grad = _grad_left(y, theta, basis, X)
        else:
            grad = _grad_interval(y, theta, basis, X)

    return nll, grad
