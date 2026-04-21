"""Log-likelihood functions for conditional transformation models.

All censoring types (exact, left, right, interval) are supported.
Optional linear regression shift via covariate matrix X.

Mathematical convention
-----------------------
Given a Bernstein basis B_k and coefficient vector theta_b (length p = order+1):

    h(y)  = B_k(y) @ theta_b  [+ X @ beta  if covariates are present]
    h'(y) = B_k'(y) @ theta_b  (first derivative; beta does not appear)

The target distribution Z follows one of:

* ``"normal"``             — N(0, 1)
* ``"logistic"``           — Logistic(0, 1)
* ``"min_extreme_value"``  — standard minimum extreme value (reversed Gumbel),
                             the link that gives the Cox proportional hazards
                             model, since ``log[-log S(t)] = h(t)``.
* ``"max_extreme_value"``  — standard (right) Gumbel, the link that realises
                             the Lehmann / reverse-time proportional-hazards
                             model: ``-log F(t) = h(t) + x'β``.
* ``"exponential"``        — standard exponential with rate 1.  Support is
                             ``[0, ∞)``, enforced during optimisation by
                             requiring ``h(y|x) >= 0``.  Without covariates
                             this reduces to ``theta_b[0] >= 0``; with
                             covariates a per-row inequality
                             ``theta_b[0] + X_i · β >= 0`` is added for each
                             training observation (see
                             :func:`pymlt.constraints.build_constraints`).

Log-likelihood formulae
-----------------------
NONE (exact):
    ℓ = Σ_i [log f(h_i) + log h'_i]

RIGHT (exact + right-censored):
    ℓ = Σ_{exact}    [log f(h_i) + log h'_i]
      + Σ_{censored}  log F̄(h_i)

LEFT (exact + left-censored):
    ℓ = Σ_{exact}    [log f(h_i) + log h'_i]
      + Σ_{censored}  log F(h_i)

INTERVAL (interval-censored [l_i, u_i]):
    ℓ = Σ_i log(F(h(u_i)) − F(h(l_i)))  [+ exact terms if present]

Numerical stability
-------------------
- All h values are clipped to [-30, 30] before distribution calls.
- log CDF for normal is computed via scipy.special.log_ndtr, not log(cdf(...)).
- log(1 − F) is computed via dist.logsf, not log(1 − cdf(...)).
- log(F(b) − F(a)) uses a logsumexp trick with a Taylor fallback for
  very narrow intervals (see _log_diff_ndtr).
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr
from scipy.stats import expon as _expon
from scipy.stats import gumbel_l as _mev
from scipy.stats import gumbel_r as _maxev
from scipy.stats import logistic as _logistic
from scipy.stats import norm

from pymlt.basis import BernsteinBasis
from pymlt.variables import CensoredData, CensoringType

BaseDistribution = Literal[
    "normal",
    "logistic",
    "min_extreme_value",
    "max_extreme_value",
    "exponential",
]
_VALID_BASE_DISTRIBUTIONS = (
    "normal",
    "logistic",
    "min_extreme_value",
    "max_extreme_value",
    "exponential",
)


def _get_dist(base_distribution: str) -> Any:
    """Return the scipy.stats distribution for *base_distribution*.

    The supported choices are:

    * ``"normal"``             — :data:`scipy.stats.norm`
    * ``"logistic"``           — :data:`scipy.stats.logistic`
    * ``"min_extreme_value"``  — :data:`scipy.stats.gumbel_l`, the standard
      minimum extreme value (reversed Gumbel) distribution.  This is the
      inverse link that realises the Cox proportional hazards model:
      if ``h(T) ~ MinExtrVal`` then ``log[-log S(t)] = h(t) + x'beta``.
    * ``"max_extreme_value"``  — :data:`scipy.stats.gumbel_r`, the standard
      (right) Gumbel distribution.  The link that realises the Lehmann /
      reverse-time proportional-hazards model:
      if ``h(T) ~ MaxExtrVal`` then ``-log F(t) = h(t) + x'beta``.
    * ``"exponential"``        — :data:`scipy.stats.expon`, the standard
      exponential (rate 1).  Support is ``[0, ∞)``; the optimiser enforces
      ``h(y|x) >= 0`` via :func:`pymlt.constraints.build_constraints`.  With
      no covariates this collapses to ``theta_b[0] >= 0``; with covariates,
      one inequality ``theta_b[0] + X_i · β >= 0`` is added per training row.

    Raises
    ------
    ValueError
        For any value not in ``_VALID_BASE_DISTRIBUTIONS``, so misconfiguration
        is never silently swallowed.
    """
    if base_distribution == "normal":
        return norm
    if base_distribution == "logistic":
        return _logistic
    if base_distribution == "min_extreme_value":
        return _mev
    if base_distribution == "max_extreme_value":
        return _maxev
    if base_distribution == "exponential":
        return _expon
    raise ValueError(
        f"base_distribution={base_distribution!r} is not supported. "
        f"Choose one of {_VALID_BASE_DISTRIBUTIONS}."
    )


class InfeasibleParameterError(ValueError):
    """Raised when the log-likelihood is non-finite at the given ``theta``.

    Subclass of :class:`ValueError` so callers that pre-date this class keep
    working.  The optimiser catches this specific subclass to distinguish
    "this parameter point is infeasible, retreat" from "something is
    genuinely wrong" (e.g. a shape bug or an unsupported ``base_distribution``,
    which should propagate).
    """


# Clipping range for h before distribution calls.
_H_CLIP = 30.0

# Maximum exponent that np.exp can handle without overflow.
_LOG_FLOAT_MAX: float = float(np.log(np.finfo(np.float64).max))


# ---------------------------------------------------------------------------
# Numerically stable log(F(b) − F(a))
# ---------------------------------------------------------------------------


def _log_diff_ndtr(
    a: NDArray[np.float64], b: NDArray[np.float64], dist: Any = norm
) -> NDArray[np.float64]:
    """Compute log(F(b) − F(a)) in a numerically stable way.

    Requires b >= a element-wise.

    For *wide* intervals the logsumexp identity is used::

        log(F(b) − F(a)) = log F(b) + log(1 − exp(log F(a) − log F(b)))

    For *narrow* intervals (F(a) ≈ F(b)) the first-order Taylor approximation
    is used::

        F(b) − F(a) ≈ f(mid) · (b − a),   mid = (a + b) / 2

    Parameters
    ----------
    a: NDArray[np.float64]
        Lower arguments. Must satisfy a <= b.
    b: NDArray[np.float64]
        Upper arguments. Must satisfy a <= b.
    dist: Any, default=scipy.stats.norm
        Base distribution object (e.g., norm or logistic) providing
        ``logcdf`` and ``logpdf``.

    Returns
    -------
    NDArray[np.float64]
        Array of log(F(b) - F(a)) values, same shape as inputs.

    Notes
    -----
    Computing `log(F(b) - F(a))` directly can lead to catastrophic cancellation
    when `F(b) ≈ F(a)`, resulting in `log(0) = -inf`. This function uses a
    piecewise approach:

    1. For **wide intervals** (`log(F(a)) - log(F(b)) < -1e-6`), it uses the
       logsumexp trick:
       `log(F(b)) + log1p(-exp(log(F(a)) - log(F(b))))`

    2. For **narrow intervals** (`log(F(a)) - log(F(b)) >= -1e-6`), it uses a
       first-order Taylor approximation around the midpoint `mid = (a + b) / 2`
       to avoid evaluating identical CDFs:
       `F(b) - F(a) ≈ f(mid) * (b - a)`
       which in log-space becomes:
       `log(f(mid)) + log(b - a)`
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Use log_ndtr for normal (more stable at extreme values); dist.logcdf otherwise
    _logcdf = log_ndtr if dist is norm else dist.logcdf

    log_Fa = _logcdf(a)  # log F(a)
    log_Fb = _logcdf(b)  # log F(b)
    ratio = log_Fa - log_Fb  # <= 0  (since a <= b)

    # Wide-interval path: ratio well below 0 → log1p stable
    ratio_safe = np.minimum(ratio, -1e-15)
    wide = log_Fb + np.log1p(-np.exp(ratio_safe))

    # Narrow-interval fallback: F(b)-F(a) ≈ f(mid)·(b−a)
    mid = 0.5 * (a + b)
    width = np.maximum(b - a, np.finfo(float).tiny)
    narrow = dist.logpdf(mid) + np.log(width)

    # Use fallback when ratio > -1e-6  (i.e. F(a)/F(b) > 1-1e-6)
    return cast(NDArray[np.float64], np.where(ratio < -1e-6, wide, narrow))


# ---------------------------------------------------------------------------
# Helpers: split theta, apply shift, compute score
# ---------------------------------------------------------------------------


def _split_theta(
    theta: NDArray[np.float64], p: int, X: NDArray[np.float64] | None
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Split the full parameter vector `theta` into basis coefficients and
    regression shifts.

    Parameters
    ----------
    theta : NDArray[np.float64]
        Fitted parameter vector of shape `(p + q,)` where `p` is the order of the
        Bernstein basis + 1, and `q` is the number of covariates (if any).
    p : int
        Number of Bernstein basis coefficients (`order + 1`).
    X : NDArray[np.float64] | None
        Covariate matrix of shape `(n, q)`, or None if no covariates are used.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64] | None]
        A tuple `(theta_b, beta)` where `theta_b` is the vector of basis coefficients
        and `beta` is the vector of regression coefficients (if `X` is provided).
    """
    theta_b = theta[:p]
    beta = theta[p:] if X is not None else None
    return theta_b, beta


def _shift(
    h: NDArray[np.float64],
    X: NDArray[np.float64] | None,
    beta: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Add regression shift X @ beta to the baseline transformation h.

    Parameters
    ----------
    h : NDArray[np.float64]
        Transformation function evaluated at observations.
    X : NDArray[np.float64] | None
        Covariates of shape `(n, q)`. If None, `h` is returned unchanged.
    beta : NDArray[np.float64] | None
        Regression coefficients for the linear shift `X @ beta`.

    Returns
    -------
    NDArray[np.float64]
        Shifted transformation vector: `h + X @ beta`.
    """
    if X is not None and beta is not None:
        return h + X @ beta
    return h


def _neg_score(h: NDArray[np.float64], dist: Any) -> NDArray[np.float64]:
    """Compute -(∂ log f(h) / ∂h) for the base distribution.

    Parameters
    ----------
    h : NDArray[np.float64]
        Values of the transformation function.
    dist : Any
        scipy.stats distribution object (``norm``, ``logistic``, ``gumbel_l``,
        ``gumbel_r``, or ``expon``).

    Returns
    -------
    NDArray[np.float64]
        The negative derivative of the log-density.

        * normal (N(0,1)):            ``h``
        * logistic:                   ``2 F(h) - 1``
        * min_extreme_value (gumbel_l): ``exp(h) - 1``
        * max_extreme_value (gumbel_r): ``1 - exp(-h)``
        * exponential:                ``1`` (constant)
    """
    if dist is norm:
        return h
    if dist is _mev:
        return np.exp(h) - 1.0
    if dist is _maxev:
        return 1.0 - np.exp(-h)
    if dist is _expon:
        return np.ones_like(h)
    # logistic (remaining case)
    return cast(NDArray[np.float64], 2.0 * dist.cdf(h) - 1.0)


def _d2_logpdf(h: NDArray[np.float64], dist: Any) -> NDArray[np.float64]:
    """Compute ``ψ'(h) = d² log f(h) / dh²`` for the base distribution.

    Parameters
    ----------
    h : NDArray[np.float64]
        Values of the transformation function.
    dist : Any
        scipy.stats distribution object (``norm``, ``logistic``, ``gumbel_l``,
        ``gumbel_r``, or ``expon``).

    Returns
    -------
    NDArray[np.float64]
        The second derivative of ``log f`` w.r.t. ``h``.

        * normal (N(0,1)):              ``-1``
        * logistic:                     ``-2 · f(h)`` where ``f`` is logistic pdf
        * min_extreme_value (gumbel_l): ``-exp(h)``
        * max_extreme_value (gumbel_r): ``-exp(-h)``
        * exponential:                  ``0``  (log f is linear in h)

    Notes
    -----
    Used to assemble the analytical Hessian of the log-likelihood.  For
    log-concave base distributions (all five supported choices),
    ``ψ'(h) ≤ 0`` for every h, which makes the exact-observation Hessian
    of the *negative* log-likelihood positive on the ``β`` block.
    """
    if dist is norm:
        return np.full_like(h, -1.0)
    if dist is _mev:
        return cast(NDArray[np.float64], -np.exp(h))
    if dist is _maxev:
        return cast(NDArray[np.float64], -np.exp(-h))
    if dist is _expon:
        return np.zeros_like(h)
    # logistic (remaining case)
    return cast(NDArray[np.float64], -2.0 * dist.pdf(h))


# ---------------------------------------------------------------------------
# Private log-likelihood functions — one per censoring type
# ---------------------------------------------------------------------------


def _ll_none(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> float:
    """Computes the log-likelihood for exactly observed (uncensored) data.

    Formula
    -------
    ℓ = Σ [log f(h_i) + log h'_i]

    Parameters
    ----------
    y : NDArray[np.float64]
        Exact observations.
    theta : NDArray[np.float64]
        Concatenated parameter vector `[theta_b | beta]`.
    basis : BernsteinBasis
        Polynomial basis object.
    X : NDArray[np.float64] | None
        Covariates.
    dist : Any, default=scipy.stats.norm
        Base distribution.

    Returns
    -------
    float
        Computed log-likelihood.
    """
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)

    B = basis.evaluate(y)  # (n, p)
    D = basis.derivative(y, order=1)  # (n, p)
    h = np.clip(_shift(B @ theta_b, X, beta), -_H_CLIP, _H_CLIP)
    hp = D @ theta_b  # h-prime; must be > 0

    with np.errstate(invalid="ignore", divide="ignore"):
        # np.log(hp) produces -inf/nan when hp <= 0 (monotonicity violated).
        # log_likelihood() detects and raises ValueError after this call.
        return float(np.sum(dist.logpdf(h)) + np.sum(np.log(hp)))


def _ll_right(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> float:
    """Computes the log-likelihood for right-censored data.

    Formula
    -------
    ℓ = Σ_exact [log f(h) + log h'] + Σ_censored log S(h)
    where S(h) = 1 - F(h) is the survival function.

    Parameters
    ----------
    cd : CensoredData
        Object containing exact and censored bounds.
    theta : NDArray[np.float64]
        Concatenated parameter vector `[theta_b | beta]`.
    basis : BernsteinBasis
        Polynomial basis object.
    X : NDArray[np.float64] | None
        Covariates.
    dist : Any, default=scipy.stats.norm
        Base distribution.

    Returns
    -------
    float
        Computed log-likelihood.
    """
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)
    ll = 0.0

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        B_e = basis.evaluate(y_e)
        D_e = basis.derivative(y_e, order=1)
        h_e = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        ll += float(np.sum(dist.logpdf(h_e)) + np.sum(np.log(hp_e)))

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]  # last known lower bound
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        ll += float(np.sum(dist.logsf(h_c)))

    return ll


def _ll_left(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> float:
    """Computes the log-likelihood for left-censored data.

    Formula
    -------
    ℓ = Σ_exact [log f(h) + log h'] + Σ_censored log F(h)

    Parameters
    ----------
    cd : CensoredData
        Object containing exact and censored bounds.
    theta : NDArray[np.float64]
        Concatenated parameter vector `[theta_b | beta]`.
    basis : BernsteinBasis
        Polynomial basis object.
    X : NDArray[np.float64] | None
        Covariates.
    dist : Any, default=scipy.stats.norm
        Base distribution.

    Returns
    -------
    float
        Computed log-likelihood.
    """
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)
    ll = 0.0

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        B_e = basis.evaluate(y_e)
        D_e = basis.derivative(y_e, order=1)
        h_e = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        ll += float(np.sum(dist.logpdf(h_e)) + np.sum(np.log(hp_e)))

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]  # last known upper bound
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        _logcdf = log_ndtr if dist is norm else dist.logcdf
        ll += float(np.sum(_logcdf(h_c)))

    return ll


def _ll_interval(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> float:
    """ℓ = Σ log(F(h(upper_i)) − F(h(lower_i)))  [+ exact terms if present]."""
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)
    ll = 0.0

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        B_e = basis.evaluate(y_e)
        D_e = basis.derivative(y_e, order=1)
        h_e = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        ll += float(np.sum(dist.logpdf(h_e)) + np.sum(np.log(hp_e)))

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
        ll += float(np.sum(_log_diff_ndtr(h_lo, h_hi, dist=dist)))

    return ll


# ---------------------------------------------------------------------------
# Private gradient functions — one per censoring type
# (all return gradient of the NEGATIVE log-likelihood)
# ---------------------------------------------------------------------------


def _grad_none(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
    """∂(-ℓ)/∂θ for exact observations."""
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)

    B = basis.evaluate(y)  # (n, p)
    D = basis.derivative(y, order=1)  # (n, p)
    h = np.clip(_shift(B @ theta_b, X, beta), -_H_CLIP, _H_CLIP)
    hp = D @ theta_b

    ns = _neg_score(h, dist)  # -(∂ log f(h)/∂h)
    grad_b = B.T @ ns - D.T @ (1.0 / hp)

    if X is not None and beta is not None:
        grad_beta = X.T @ ns
        return cast(NDArray[np.float64], np.concatenate([grad_b, grad_beta]))
    return grad_b


def _grad_right(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
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
        h_e = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        ns = _neg_score(h_e, dist)
        grad[:p] += B_e.T @ ns - D_e.T @ (1.0 / hp_e)
        if X_e is not None:
            grad[p:] += X_e.T @ ns

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        # ∂(-ℓ)/∂θ_b from censored = +B_c.T @ [f(h)/F̄(h)]
        # Cap the exponent to avoid overflow; consistent with logsf in the LL.
        log_hazard = dist.logpdf(h_c) - dist.logsf(h_c)
        hazard = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
        grad[:p] += B_c.T @ hazard
        if X_c is not None:
            grad[p:] += X_c.T @ hazard

    return cast(NDArray[np.float64], grad)


def _grad_left(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
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
        h_e = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        ns = _neg_score(h_e, dist)
        grad[:p] += B_e.T @ ns - D_e.T @ (1.0 / hp_e)
        if X_e is not None:
            grad[p:] += X_e.T @ ns

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        # ∂(-ℓ)/∂θ_b from censored = -B_c.T @ [f(h)/F(h)]
        _logcdf = log_ndtr if dist is norm else dist.logcdf
        inv_mills = np.exp(dist.logpdf(h_c) - _logcdf(h_c))
        grad[:p] -= B_c.T @ inv_mills
        if X_c is not None:
            grad[p:] -= X_c.T @ inv_mills

    return cast(NDArray[np.float64], grad)


def _grad_interval(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
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
        h_e = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        ns = _neg_score(h_e, dist)
        grad[:p] += B_e.T @ ns - D_e.T @ (1.0 / hp_e)
        if X_e is not None:
            grad[p:] += X_e.T @ ns

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

        log_p = _log_diff_ndtr(h_lo, h_hi, dist=dist)
        # When interval probability ≈ 0, log_p = -inf and the subtraction
        # produces +inf before exp, raising overflow/invalid warnings.  Suppress
        # them here; nan_to_num zeros the resulting inf/NaN so they never affect
        # the gradient.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            w_hi = np.exp(dist.logpdf(h_hi) - log_p)
            w_lo = np.exp(dist.logpdf(h_lo) - log_p)
        w_hi = np.nan_to_num(w_hi, nan=0.0, posinf=0.0, neginf=0.0)
        w_lo = np.nan_to_num(w_lo, nan=0.0, posinf=0.0, neginf=0.0)

        with np.errstate(invalid="ignore"):
            grad[:p] -= B_hi.T @ w_hi - B_lo.T @ w_lo
            if X_i is not None:
                grad[p:] -= X_i.T @ (w_hi - w_lo)

    return cast(NDArray[np.float64], grad)


# ---------------------------------------------------------------------------
# Private per-observation score functions — one per censoring type.
# Each returns an (n, p+q) matrix of gradients of the POSITIVE log-likelihood
# ℓ_i w.r.t. theta, with rows aligned to the input ordering.
# ---------------------------------------------------------------------------


def _scores_none(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
    """Per-observation ∂ℓ/∂θ for exact observations, shape ``(n, p+q)``."""
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    theta_b, beta = _split_theta(theta, p, X)

    B = basis.evaluate(y)  # (n, p)
    D = basis.derivative(y, order=1)  # (n, p)
    h = np.clip(_shift(B @ theta_b, X, beta), -_H_CLIP, _H_CLIP)
    hp = D @ theta_b  # (n,)

    psi = -_neg_score(h, dist)  # ψ(h) = d log f / dh, shape (n,)
    # ∂ℓ_i/∂θ_b = B_i · ψ(h_i) + D_i / h'_i
    scores_b = B * psi[:, None] + D / hp[:, None]

    scores = np.empty((len(y), p + q), dtype=np.float64)
    scores[:, :p] = scores_b
    if X is not None:
        # ∂ℓ_i/∂β = x_i · ψ(h_i)
        scores[:, p:] = X * psi[:, None]
    return scores


def _scores_right(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
    """Per-observation ∂ℓ/∂θ for right-censored data, shape ``(n, p+q)``."""
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    n = cd.n
    theta_b, beta = _split_theta(theta, p, X)
    scores = np.zeros((n, p + q), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        s_e = _scores_none(y_e, theta, basis, X_e, dist=dist)
        scores[mask_e] = s_e

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        # ∂ℓ_i/∂h = -λ(h) = -f(h)/S(h)
        log_hazard = dist.logpdf(h_c) - dist.logsf(h_c)
        hazard = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
        scores[mask_c, :p] = -B_c * hazard[:, None]
        if X_c is not None:
            scores[mask_c, p:] = -X_c * hazard[:, None]

    return scores


def _scores_left(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
    """Per-observation ∂ℓ/∂θ for left-censored data, shape ``(n, p+q)``."""
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    n = cd.n
    theta_b, beta = _split_theta(theta, p, X)
    scores = np.zeros((n, p + q), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        scores[mask_e] = _scores_none(y_e, theta, basis, X_e, dist=dist)

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        # ∂ℓ_i/∂h = µ(h) = f(h)/F(h)
        _logcdf = log_ndtr if dist is norm else dist.logcdf
        inv_mills = np.exp(dist.logpdf(h_c) - _logcdf(h_c))
        scores[mask_c, :p] = B_c * inv_mills[:, None]
        if X_c is not None:
            scores[mask_c, p:] = X_c * inv_mills[:, None]

    return scores


def _scores_interval(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
    """Per-observation ∂ℓ/∂θ for interval-censored data, shape ``(n, p+q)``."""
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    n = cd.n
    theta_b, beta = _split_theta(theta, p, X)
    scores = np.zeros((n, p + q), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        scores[mask_e] = _scores_none(y_e, theta, basis, X_e, dist=dist)

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

        log_p = _log_diff_ndtr(h_lo, h_hi, dist=dist)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            w_hi = np.exp(dist.logpdf(h_hi) - log_p)
            w_lo = np.exp(dist.logpdf(h_lo) - log_p)
        w_hi = np.nan_to_num(w_hi, nan=0.0, posinf=0.0, neginf=0.0)
        w_lo = np.nan_to_num(w_lo, nan=0.0, posinf=0.0, neginf=0.0)

        # ∂ℓ_i/∂θ_b = B_hi_i · w_hi - B_lo_i · w_lo
        scores[mask_i, :p] = B_hi * w_hi[:, None] - B_lo * w_lo[:, None]
        if X_i is not None:
            scores[mask_i, p:] = X_i * (w_hi - w_lo)[:, None]

    return scores


# ---------------------------------------------------------------------------
# Private Hessian functions — one per censoring type.
# Each returns the (p+q, p+q) Hessian of the NEGATIVE log-likelihood.
# ---------------------------------------------------------------------------


def _assemble_hessian(
    B: NDArray[np.float64],
    w: NDArray[np.float64],
    X: NDArray[np.float64] | None,
    p: int,
    q: int,
) -> NDArray[np.float64]:
    """Assemble ``[B, X]^T · diag(w) · [B, X]`` block structure.

    Reused for every censoring type's shared-h chain-rule term: a diagonal
    weight ``w`` (shape ``(n_group,)``) times the outer product of the same
    design row on both sides.  Returns a ``(p+q, p+q)`` symmetric block.
    """
    Bw = B * w[:, None]
    H = np.zeros((p + q, p + q), dtype=np.float64)
    H[:p, :p] = Bw.T @ B
    if X is not None and q > 0:
        Xw = X * w[:, None]
        H_bx = Bw.T @ X
        H[:p, p:] = H_bx
        H[p:, :p] = H_bx.T
        H[p:, p:] = Xw.T @ X
    return H


def _hess_none(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
    """Hessian of -ℓ for exact observations, shape ``(p+q, p+q)``.

    Per-observation contribution to ``∂²(-ℓ)/∂θ∂θ'``::

        [θ_b θ_b]:  -ψ'(h) · B_i B_i' + (D_i D_i') / (h'_i)²
        [θ_b β  ]:  -ψ'(h) · B_i x_i'
        [β   β  ]:  -ψ'(h) · x_i x_i'

    where ``ψ'(h) = d² log f / dh²`` comes from :func:`_d2_logpdf`.  The
    ``(D_i D_i')/(h'_i)²`` term comes from ``-∂²/∂θ_b² log(h')``; it is
    absent for ``β`` because ``h'`` does not depend on ``β``.
    """
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    theta_b, beta = _split_theta(theta, p, X)

    B = basis.evaluate(y)  # (n, p)
    D = basis.derivative(y, order=1)  # (n, p)
    h = np.clip(_shift(B @ theta_b, X, beta), -_H_CLIP, _H_CLIP)
    hp = D @ theta_b

    w = -_d2_logpdf(h, dist)  # -ψ'(h), ≥ 0 for log-concave f
    H = _assemble_hessian(B, w, X, p, q)
    # Add D^T diag(1/h'²) D term on the θ_b block only
    Dw = D * (1.0 / (hp * hp))[:, None]
    H[:p, :p] += Dw.T @ D
    return H


def _hess_right(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
    """Hessian of -ℓ for right-censored data, shape ``(p+q, p+q)``.

    Exact rows contribute via :func:`_hess_none`.  Right-censored rows at
    lower bound ``h_l`` contribute ``λ(h)·(ψ(h) + λ(h))`` on the shared
    ``[B, x]`` design, where ``λ = f/S`` is the hazard, ``ψ = d log f / dh``.
    """
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    theta_b, beta = _split_theta(theta, p, X)
    H = np.zeros((p + q, p + q), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        H += _hess_none(y_e, theta, basis, X_e, dist=dist)

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        log_hazard = dist.logpdf(h_c) - dist.logsf(h_c)
        lam = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
        psi = -_neg_score(h_c, dist)
        w = lam * (psi + lam)  # = -d²logS/dh² → NLL contribution
        H += _assemble_hessian(B_c, w, X_c, p, q)

    return H


def _hess_left(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
    """Hessian of -ℓ for left-censored data, shape ``(p+q, p+q)``.

    Left-censored rows at upper bound ``h_u`` contribute
    ``µ(h)·(µ(h) - ψ(h))`` where ``µ = f/F`` is the inverse Mills ratio.
    """
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    theta_b, beta = _split_theta(theta, p, X)
    H = np.zeros((p + q, p + q), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        H += _hess_none(y_e, theta, basis, X_e, dist=dist)

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        _logcdf = log_ndtr if dist is norm else dist.logcdf
        mu = np.exp(dist.logpdf(h_c) - _logcdf(h_c))
        psi = -_neg_score(h_c, dist)
        w = mu * (mu - psi)  # = -d²logF/dh² → NLL contribution
        H += _assemble_hessian(B_c, w, X_c, p, q)

    return H


def _hess_interval(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
    """Hessian of -ℓ for interval-censored data, shape ``(p+q, p+q)``.

    For each interval ``[h_l, h_u]`` with ``p = F(h_u) - F(h_l)``,
    ``w_lo = f(h_l)/p``, ``w_hi = f(h_u)/p``, the 2x2 Hessian of
    ``log p`` w.r.t. ``(h_l, h_u)`` has entries::

        ∂²/∂h_l² = -ψ(h_l) w_lo - w_lo²
        ∂²/∂h_u² =  ψ(h_u) w_hi - w_hi²
        ∂²/∂h_l ∂h_u = w_hi · w_lo

    Chained through the Jacobian ``∂(h_l, h_u)/∂(θ_b, β) = [[B_lo, x],
    [B_hi, x]]`` and negated for NLL.
    """
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    theta_b, beta = _split_theta(theta, p, X)
    H = np.zeros((p + q, p + q), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        H += _hess_none(y_e, theta, basis, X_e, dist=dist)

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

        log_p = _log_diff_ndtr(h_lo, h_hi, dist=dist)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            w_hi = np.exp(dist.logpdf(h_hi) - log_p)
            w_lo = np.exp(dist.logpdf(h_lo) - log_p)
        w_hi = np.nan_to_num(w_hi, nan=0.0, posinf=0.0, neginf=0.0)
        w_lo = np.nan_to_num(w_lo, nan=0.0, posinf=0.0, neginf=0.0)

        psi_lo = -_neg_score(h_lo, dist)
        psi_hi = -_neg_score(h_hi, dist)

        # Entries of the 2x2 Hessian of log p (per obs, same for ℓ and -ℓ sign
        # flipped below).
        a = -psi_lo * w_lo - w_lo * w_lo  # ∂²log p/∂h_l²
        c = psi_hi * w_hi - w_hi * w_hi  # ∂²log p/∂h_u²
        b = w_hi * w_lo  # ∂²log p/∂h_l ∂h_u

        # NLL contribution = - (chain through J):
        #   -[ J_l^T diag(a) J_l + J_l^T diag(b) J_u + J_u^T diag(b) J_l
        #      + J_u^T diag(c) J_u ]
        # where J_l = [B_lo, X_i], J_u = [B_hi, X_i].
        def _outer(
            U: NDArray[np.float64],
            Xu: NDArray[np.float64] | None,
            V: NDArray[np.float64],
            Xv: NDArray[np.float64] | None,
            w_vec: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """Compute U^T diag(w_vec) V split into (p+q, p+q) block."""
            Uw = U * w_vec[:, None]
            out = np.zeros((p + q, p + q), dtype=np.float64)
            out[:p, :p] = Uw.T @ V
            if Xu is not None and Xv is not None and q > 0:
                out[:p, p:] = Uw.T @ Xv
                out[p:, :p] = (Xu * w_vec[:, None]).T @ V
                out[p:, p:] = (Xu * w_vec[:, None]).T @ Xv
            return out

        block = (
            _outer(B_lo, X_i, B_lo, X_i, a)
            + _outer(B_lo, X_i, B_hi, X_i, b)
            + _outer(B_hi, X_i, B_lo, X_i, b)
            + _outer(B_hi, X_i, B_hi, X_i, c)
        )
        H -= block  # NLL = -ℓ

    return H


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_likelihood(
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    base_distribution: BaseDistribution = "normal",
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
    base_distribution:
        One of ``"normal"`` (default), ``"logistic"``, ``"min_extreme_value"``,
        ``"max_extreme_value"``, or ``"exponential"``.  Selects the target
        distribution Z such that h(Y|X) ~ Z.

    Returns
    -------
    float  (log-likelihood value)

    Raises
    ------
    InfeasibleParameterError
        If the result is ``-inf`` or ``NaN``.  Most likely causes: theta
        violates monotonicity (h' ≤ 0), observations outside support, or
        numerical overflow in the basis evaluation.  Subclass of
        :class:`ValueError`.
    ValueError
        From :func:`_get_dist` if ``base_distribution`` is not supported.
    """
    dist = _get_dist(base_distribution)

    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        result = _ll_none(y_arr, theta, basis, X, dist=dist)
    else:
        if censoring is CensoringType.NONE:
            result = _ll_none(y.exact, theta, basis, X, dist=dist)
        elif censoring is CensoringType.RIGHT:
            result = _ll_right(y, theta, basis, X, dist=dist)
        elif censoring is CensoringType.LEFT:
            result = _ll_left(y, theta, basis, X, dist=dist)
        else:  # INTERVAL
            result = _ll_interval(y, theta, basis, X, dist=dist)

    if not np.isfinite(result):
        raise InfeasibleParameterError(
            f"log_likelihood returned {result}.  Possible causes: theta "
            "violates monotonicity (h'(y) ≤ 0), observations outside basis "
            "support, or extreme h values despite clipping."
        )
    return result


def negative_log_likelihood(
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    gradient: bool = False,
    base_distribution: BaseDistribution = "normal",
) -> float | tuple[float, NDArray[np.float64]]:
    """Negative log-likelihood (objective for minimisation) with optional gradient.

    Parameters
    ----------
    theta, basis, y, X, censoring:
        Same as :func:`log_likelihood`.
    gradient:
        If ``True``, return a ``(nll, grad)`` tuple where ``grad`` is the
        analytical gradient of the *negative* log-likelihood w.r.t. ``theta``.
        Computed analytically — no finite-difference approximation.
    base_distribution:
        One of ``"normal"`` (default), ``"logistic"``, ``"min_extreme_value"``,
        ``"max_extreme_value"``, or ``"exponential"``.

    Returns
    -------
    float  when ``gradient=False``
    (float, NDArray)  when ``gradient=True``
    """
    nll = -log_likelihood(
        theta, basis, y, X, censoring, base_distribution=base_distribution
    )

    if not gradient:
        return nll

    dist = _get_dist(base_distribution)

    # Analytical gradient of the negative log-likelihood
    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        grad = _grad_none(y_arr, theta, basis, X, dist=dist)
    else:
        if censoring is CensoringType.NONE:
            grad = _grad_none(y.exact, theta, basis, X, dist=dist)
        elif censoring is CensoringType.RIGHT:
            grad = _grad_right(y, theta, basis, X, dist=dist)
        elif censoring is CensoringType.LEFT:
            grad = _grad_left(y, theta, basis, X, dist=dist)
        else:
            grad = _grad_interval(y, theta, basis, X, dist=dist)

    return nll, grad


def hessian(
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    base_distribution: BaseDistribution = "normal",
) -> NDArray[np.float64]:
    """Analytical Hessian of the negative log-likelihood.

    The returned matrix is the observed information ``∂²(-ℓ)/∂θ∂θ'`` at
    ``theta``.  Inverting it yields the asymptotic covariance matrix of the
    maximum-likelihood estimator (see
    :meth:`~pymlt.model.ConditionalTransformationModel.vcov`).

    Parameters
    ----------
    theta, basis, y, X, censoring, base_distribution:
        Same as :func:`log_likelihood`.

    Returns
    -------
    NDArray[np.float64]
        Symmetric ``(p+q, p+q)`` Hessian of ``-ℓ`` where ``p = basis.order + 1``
        and ``q = X.shape[1]`` (``0`` if ``X is None``).

    Raises
    ------
    InfeasibleParameterError
        If any entry of the Hessian is non-finite — most commonly because
        ``theta`` violates monotonicity (``h'(y) ≤ 0``) or because observations
        fall outside the basis support.  Subclass of :class:`ValueError`.
    ValueError
        If ``base_distribution`` is not supported (propagated from
        :func:`_get_dist`).

    Notes
    -----
    All five base distributions are log-concave, so the ``β`` block of the
    Hessian of ``-ℓ`` is positive semidefinite at any ``h``.  The full
    Hessian is additionally positive definite at the unconstrained MLE
    (and invertible) for non-degenerate data.
    """
    dist = _get_dist(base_distribution)

    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        result = _hess_none(y_arr, theta, basis, X, dist=dist)
    elif censoring is CensoringType.NONE:
        result = _hess_none(y.exact, theta, basis, X, dist=dist)
    elif censoring is CensoringType.RIGHT:
        result = _hess_right(y, theta, basis, X, dist=dist)
    elif censoring is CensoringType.LEFT:
        result = _hess_left(y, theta, basis, X, dist=dist)
    else:
        result = _hess_interval(y, theta, basis, X, dist=dist)

    if not np.all(np.isfinite(result)):
        raise InfeasibleParameterError(
            "hessian() produced non-finite entries.  Possible causes: theta "
            "violates monotonicity (h'(y) ≤ 0), observations outside basis "
            "support, or extreme h values despite clipping."
        )
    return result


def score_matrix(
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    base_distribution: BaseDistribution = "normal",
) -> NDArray[np.float64]:
    """Per-observation score contributions ``∂ℓ_i/∂θ``.

    Returns the ``(n, p+q)`` matrix of per-observation gradients of the
    *positive* log-likelihood, often referred to as ``estfun`` in the R
    ``sandwich`` package.  ``score_matrix(...).sum(axis=0)`` equals the full
    log-likelihood gradient (the negative of
    :func:`negative_log_likelihood` gradient).

    Parameters
    ----------
    theta, basis, y, X, censoring, base_distribution:
        Same as :func:`log_likelihood`.

    Returns
    -------
    NDArray[np.float64]
        Per-observation score matrix of shape ``(n, p+q)``.  Row ``i`` gives
        ``∂ℓ_i/∂θ``.  Rows of observations that contribute nothing to the
        log-likelihood under the chosen censoring regime (should not occur
        for well-formed inputs) are zero.

    Raises
    ------
    InfeasibleParameterError
        If any entry of the score matrix is non-finite — most commonly
        because ``theta`` violates monotonicity (``h'(y) ≤ 0``) or because
        observations fall outside the basis support.  Subclass of
        :class:`ValueError`.
    ValueError
        If ``base_distribution`` is not supported.
    """
    dist = _get_dist(base_distribution)

    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        result = _scores_none(y_arr, theta, basis, X, dist=dist)
    elif censoring is CensoringType.NONE:
        # Exact observations stored in CensoredData — only .exact is used.
        result = _scores_none(y.exact, theta, basis, X, dist=dist)
    elif censoring is CensoringType.RIGHT:
        result = _scores_right(y, theta, basis, X, dist=dist)
    elif censoring is CensoringType.LEFT:
        result = _scores_left(y, theta, basis, X, dist=dist)
    else:
        result = _scores_interval(y, theta, basis, X, dist=dist)

    if not np.all(np.isfinite(result)):
        raise InfeasibleParameterError(
            "score_matrix() produced non-finite entries.  Possible causes: "
            "theta violates monotonicity (h'(y) ≤ 0), observations outside "
            "basis support, or extreme h values despite clipping."
        )
    return result
