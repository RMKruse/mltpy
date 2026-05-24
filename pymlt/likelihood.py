"""Log-likelihood functions for conditional transformation models.

All censoring types (exact, left, right, interval) are supported.
Optional linear regression shift via covariate matrix X.
Optional per-observation weights and fixed offset are supported.

Mathematical convention
-----------------------
Given a Bernstein basis B_k and coefficient vector theta_b (length p = order+1):

    h(y)  = B_k(y) @ theta_b  [+ X @ beta  if covariates are present]
              [+ offset         if offset is provided]
    h'(y) = B_k'(y) @ theta_b  (first derivative; beta and offset do not appear)

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

Weighted log-likelihood
-----------------------
With per-observation weights ``w_i ≥ 0`` and offset ``o_i ∈ ℝ``:

    h_i   = B_i · θ_b + x_i · β + o_i     (offset enters as a constant)
    ℓ(θ)  = Σ_i w_i ℓ_i(θ)
    ∇ℓ    = Σ_i w_i s_i
    ∇²ℓ   = Σ_i w_i H_i

The score matrix (estfun) convention follows R ``sandwich``: row i = ``w_i · s_i``,
so column sums equal the full gradient.  ``weights=None`` ≡ unit weights;
``offset=None`` ≡ zero offset.

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

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, expm1, log1p, log_expit, log_ndtr, ndtr
from scipy.stats import cauchy as _cauchy
from scipy.stats import expon as _expon
from scipy.stats import gumbel_l as _mev
from scipy.stats import gumbel_r as _maxev
from scipy.stats import laplace as _laplace
from scipy.stats import logistic as _logistic
from scipy.stats import norm

from pymlt.basis import BernsteinBasis, InteractionBasis
from pymlt.variables import CensoredData, CensoringType

BaseDistribution = Literal[
    "normal",
    "logistic",
    "min_extreme_value",
    "max_extreme_value",
    "exponential",
    "laplace",
    "cauchy",
]
_VALID_BASE_DISTRIBUTIONS = (
    "normal",
    "logistic",
    "min_extreme_value",
    "max_extreme_value",
    "exponential",
    "laplace",
    "cauchy",
)

# Logarithmic constants used by the closed-form DistOps fast paths.
_LOG2: float = float(np.log(2.0))
_LOG_PI: float = float(np.log(np.pi))
_HALF_LOG_2PI: float = float(0.5 * np.log(2.0 * np.pi))


@dataclass(frozen=True)
class DistOps:
    """Typed wrapper around a scipy.stats distribution used by the likelihood.

    Dispatch in the hot paths (analytical score, second derivative, log-CDF
    fast path) goes through :attr:`kind` — a plain string comparison against
    the :data:`BaseDistribution` Literal — rather than ``dist is norm`` style
    identity checks.  Identity-based dispatch is fragile: anything that
    reimports scipy, pickles the distribution, or wraps it for instrumentation
    can silently break the identity while leaving attribute access intact,
    which would previously fall through to the logistic branch in
    :func:`_neg_score` and :func:`_d2_logpdf` and produce wrong gradients.

    The five methods :meth:`logpdf`, :meth:`logcdf`, :meth:`logsf`,
    :meth:`cdf`, and :meth:`pdf` are defined as closed forms built on
    :mod:`scipy.special` ufuncs (``log_ndtr``, ``ndtr``, ``expit``,
    ``log_expit``, ``expm1``, ``log1p``), dispatching on :attr:`kind`.  This
    avoids the ~20× ``rv_continuous`` wrapper tax of the corresponding
    ``scipy.stats`` methods in the optimiser's hot loop (issue #94).  Inputs
    arrive clipped to ``±_H_CLIP``; results match the scipy objects to
    ``rtol≈1e-10`` (the one intentional divergence is exponential
    :meth:`logpdf`, which returns ``-h`` on the whole line rather than scipy's
    ``-inf`` for ``h < 0`` — see :func:`_ll_none`).

    Every *other* method (``ppf``, ``sf``, ``isf``, ...) is exposed unchanged
    through the ``__getattr__`` forwarder to the underlying scipy distribution.
    Because methods defined on the class take priority over ``__getattr__``,
    only those five names are intercepted; existing call sites that read e.g.
    ``dist.logpdf(h)`` get the fast path for free.
    """

    kind: BaseDistribution
    scipy: Any

    def __getattr__(self, name: str) -> Any:
        # Dataclass attributes (``kind``, ``scipy``) are resolved normally;
        # this method only runs for missing names, so recursion is impossible.
        return getattr(self.scipy, name)

    # -- closed-form fast paths (issue #94) --------------------------------
    # The naive ``log1p(-exp(-exp(±h)))`` form for gumbel_l logcdf / gumbel_r
    # logsf loses ~6 digits at h=-30 and returns -inf at h=-40; the
    # ``log(-expm1(...))`` form below is load-bearing — do not "simplify" it.

    def logpdf(self, h: NDArray[np.float64]) -> NDArray[np.float64]:
        """Closed-form log density, dispatched on :attr:`kind`."""
        kind = self.kind
        if kind == "normal":
            return -0.5 * h * h - _HALF_LOG_2PI
        if kind == "logistic":
            return cast(NDArray[np.float64], log_expit(h) + log_expit(-h))
        if kind == "min_extreme_value":
            return h - np.exp(h)
        if kind == "max_extreme_value":
            return -h - np.exp(-h)
        if kind == "exponential":
            # Full-domain extension log f = -h (scipy returns -inf for h<0).
            return -h
        if kind == "laplace":
            return -np.abs(h) - _LOG2
        if kind == "cauchy":
            return cast(NDArray[np.float64], -_LOG_PI - log1p(h * h))
        raise AssertionError(f"unhandled dist.kind={kind!r}")

    def logcdf(self, h: NDArray[np.float64]) -> NDArray[np.float64]:
        """Closed-form log CDF, dispatched on :attr:`kind`."""
        kind = self.kind
        if kind == "normal":
            return cast(NDArray[np.float64], log_ndtr(h))
        if kind == "logistic":
            return cast(NDArray[np.float64], log_expit(h))
        if kind == "min_extreme_value":
            return cast(NDArray[np.float64], np.log(-expm1(-np.exp(h))))
        if kind == "max_extreme_value":
            return cast(NDArray[np.float64], -np.exp(-h))
        if kind == "exponential":
            return cast(NDArray[np.float64], np.log(-expm1(-h)))
        if kind == "laplace":
            with np.errstate(divide="ignore", invalid="ignore"):
                return cast(
                    NDArray[np.float64],
                    np.where(h <= 0.0, h - _LOG2, log1p(-0.5 * np.exp(-h))),
                )
        if kind == "cauchy":
            return cast(NDArray[np.float64], np.log(0.5 + np.arctan(h) / np.pi))
        raise AssertionError(f"unhandled dist.kind={kind!r}")

    def logsf(self, h: NDArray[np.float64]) -> NDArray[np.float64]:
        """Closed-form log survival function, dispatched on :attr:`kind`."""
        kind = self.kind
        if kind == "normal":
            return cast(NDArray[np.float64], log_ndtr(-h))
        if kind == "logistic":
            return cast(NDArray[np.float64], log_expit(-h))
        if kind == "min_extreme_value":
            return cast(NDArray[np.float64], -np.exp(h))
        if kind == "max_extreme_value":
            return cast(NDArray[np.float64], np.log(-expm1(-np.exp(-h))))
        if kind == "exponential":
            return -h
        if kind == "laplace":
            with np.errstate(divide="ignore", invalid="ignore"):
                return cast(
                    NDArray[np.float64],
                    np.where(h >= 0.0, -h - _LOG2, log1p(-0.5 * np.exp(h))),
                )
        if kind == "cauchy":
            return cast(NDArray[np.float64], np.log(0.5 - np.arctan(h) / np.pi))
        raise AssertionError(f"unhandled dist.kind={kind!r}")

    def cdf(self, h: NDArray[np.float64]) -> NDArray[np.float64]:
        """Closed-form CDF, dispatched on :attr:`kind`."""
        kind = self.kind
        if kind == "normal":
            return cast(NDArray[np.float64], ndtr(h))
        if kind == "logistic":
            return cast(NDArray[np.float64], expit(h))
        if kind == "min_extreme_value":
            return cast(NDArray[np.float64], -expm1(-np.exp(h)))
        if kind == "max_extreme_value":
            return cast(NDArray[np.float64], np.exp(-np.exp(-h)))
        if kind == "exponential":
            return cast(NDArray[np.float64], -expm1(-h))
        if kind == "laplace":
            return cast(NDArray[np.float64], np.exp(self.logcdf(h)))
        if kind == "cauchy":
            return 0.5 + np.arctan(h) / np.pi
        raise AssertionError(f"unhandled dist.kind={kind!r}")

    def pdf(self, h: NDArray[np.float64]) -> NDArray[np.float64]:
        """Closed-form density, ``exp(logpdf(h))`` for every :attr:`kind`."""
        return cast(NDArray[np.float64], np.exp(self.logpdf(h)))


_NORM_OPS = DistOps("normal", norm)
_LOGIS_OPS = DistOps("logistic", _logistic)
_MEV_OPS = DistOps("min_extreme_value", _mev)
_MAXEV_OPS = DistOps("max_extreme_value", _maxev)
_EXPON_OPS = DistOps("exponential", _expon)
_LAPLACE_OPS = DistOps("laplace", _laplace)
_CAUCHY_OPS = DistOps("cauchy", _cauchy)


def _get_dist(base_distribution: str) -> DistOps:
    """Return the :class:`DistOps` wrapper for *base_distribution*.

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
    * ``"laplace"``            — :data:`scipy.stats.laplace`, the standard
      Laplace (double exponential) distribution.  Score function is
      ``sign(h)``; the link realises a median regression model.
    * ``"cauchy"``             — :data:`scipy.stats.cauchy`, the standard
      Cauchy distribution.  Score function is ``2h/(1+h²)``.  Note that
      Cauchy is **not** log-concave, so the Hessian of the negative
      log-likelihood may not be positive semi-definite.

    The returned :class:`DistOps` forwards attribute access to the underlying
    scipy distribution, so ``_get_dist("normal").logpdf(h)`` is equivalent to
    ``scipy.stats.norm.logpdf(h)``.

    Raises
    ------
    ValueError
        For any value not in ``_VALID_BASE_DISTRIBUTIONS``, so misconfiguration
        is never silently swallowed.
    """
    if base_distribution == "normal":
        return _NORM_OPS
    if base_distribution == "logistic":
        return _LOGIS_OPS
    if base_distribution == "min_extreme_value":
        return _MEV_OPS
    if base_distribution == "max_extreme_value":
        return _MAXEV_OPS
    if base_distribution == "exponential":
        return _EXPON_OPS
    if base_distribution == "laplace":
        return _LAPLACE_OPS
    if base_distribution == "cauchy":
        return _CAUCHY_OPS
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

# _log_diff_ndtr: floor on the log-CDF ratio  r = log F(a) − log F(b)  before
# evaluating  log1p(−exp(r))  on the wide-interval path.  Without the floor,
# exact zero-width intervals (a == b ⇒ r == 0) yield exp(r) == 1 and
# log1p(−1) = −∞.  −1e-15 is the smallest magnitude that survives a single
# float64 round-trip through exp/log1p (≈ machine epsilon for ones), so the
# wide branch returns a finite (very negative) number which the
# narrow-branch selector below then discards anyway.
_LOG_DIFF_NDTR_RATIO_CAP = -1e-15

# _log_diff_ndtr: split point between the wide-interval logsumexp identity
# and the narrow-interval Taylor fallback, in the same  r = log F(a) − log F(b)
# space.  Once F(a)/F(b) exceeds 1 − 1e-6, the cancellation in
# log1p(−exp(r)) loses more than ~6 significant digits, while the midpoint
# Taylor rule  F(b)−F(a) ≈ f(mid)·(b−a)  remains accurate.  −1e-6 is the
# crossover where both rules agree to ~6 digits in float64.
_LOG_DIFF_NDTR_NARROW_THRESHOLD = -1e-6


# ---------------------------------------------------------------------------
# Numerically stable log(F(b) − F(a))
# ---------------------------------------------------------------------------


def _log_diff_ndtr(
    a: NDArray[np.float64], b: NDArray[np.float64], dist: DistOps = _NORM_OPS
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
    dist: DistOps, default=:data:`_NORM_OPS`
        Distribution wrapper providing ``logcdf`` / ``logpdf`` via attribute
        forwarding to the underlying scipy distribution.

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

    # Closed-form log CDF (normal → log_ndtr) via the DistOps fast path.
    _logcdf = dist.logcdf

    log_Fa = _logcdf(a)  # log F(a)
    log_Fb = _logcdf(b)  # log F(b)
    ratio = log_Fa - log_Fb  # <= 0  (since a <= b)

    # Wide-interval path: ratio well below 0 → log1p stable
    ratio_safe = np.minimum(ratio, _LOG_DIFF_NDTR_RATIO_CAP)
    wide = log_Fb + np.log1p(-np.exp(ratio_safe))

    # Narrow-interval fallback: F(b)-F(a) ≈ f(mid)·(b−a).
    # Zero-width intervals (a == b) give true probability 0 → log = -inf.
    # Tiny negative widths can arise transiently when both bounds clip to
    # ±_H_CLIP and round in opposite directions — mask out the invalid log.
    mid = 0.5 * (a + b)
    width = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        log_width = np.log(width)
    narrow = dist.logpdf(mid) + log_width

    # Use fallback when ratio > _LOG_DIFF_NDTR_NARROW_THRESHOLD  (F(a)/F(b)
    # within 1e-6 of 1 — see constant definition for the rationale).
    return cast(
        NDArray[np.float64],
        np.where(ratio < _LOG_DIFF_NDTR_NARROW_THRESHOLD, wide, narrow),
    )


def _pair_density_weights(
    h_lo: NDArray[np.float64],
    h_hi: NDArray[np.float64],
    log_p: NDArray[np.float64],
    dist: DistOps,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mills-style density weights ``(f(h_hi)/P, f(h_lo)/P)`` shared by the
    both-finite branch of every interval-style block (interval-censored
    likelihood / gradient / Hessian / scores / intercept score, plus the
    truncation correction).

    In a very narrow window ``log_p`` is strongly negative, so
    ``logpdf - log_p`` can exceed ``log(float64.max) ≈ 709`` and a bare
    ``np.exp`` would overflow to ``+inf``.  Mapping ``+inf`` to ``0`` (as a
    plain ``nan_to_num(posinf=0.0)`` would) silently zeroes a genuinely
    large derivative weight, biasing gradient and Hessian *toward* the
    cases where the correction matters most.  We instead clip the log-arg
    at :data:`_LOG_FLOAT_MAX`, preserving the (large but finite) magnitude;
    the only remaining NaN source is ``-inf - -inf`` from a degenerate
    point-mass interval, where zero is the correct contribution.
    """
    with np.errstate(invalid="ignore"):
        log_w_hi = dist.logpdf(h_hi) - log_p
        log_w_lo = dist.logpdf(h_lo) - log_p
    w_hi = np.exp(np.minimum(log_w_hi, _LOG_FLOAT_MAX))
    w_lo = np.exp(np.minimum(log_w_lo, _LOG_FLOAT_MAX))
    return (
        np.nan_to_num(w_hi, nan=0.0),
        np.nan_to_num(w_lo, nan=0.0),
    )


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


def _split_theta_scaled(
    theta: NDArray[np.float64],
    p: int,
    q_d: int,
    q_s: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    """Split ``theta = [theta_b | beta | gamma]`` (ADR 0002, Decision 2).

    Generalised three-way split used by the scaled-likelihood path.  Reduces
    to the existing shift split when ``q_s = 0`` (``gamma is None``), so the
    new branch is dead code for every non-scaling call site.

    Parameters
    ----------
    theta:
        Parameter vector of length ``p + q_d + q_s``.
    p:
        Number of Bernstein basis coefficients (``basis.order + 1``).
    q_d:
        Number of shift-design columns (``X.shape[1]``; ``0`` if no ``X``).
    q_s:
        Number of scaling-design columns (``scaling.shape[1]``; ``0`` if
        ``scaling is None``).
    """
    theta_b = theta[:p]
    beta = theta[p : p + q_d] if q_d > 0 else None
    gamma = theta[p + q_d : p + q_d + q_s] if q_s > 0 else None
    return theta_b, beta, gamma


def _eval_h_censored(
    y_c: NDArray[np.float64],
    basis: BernsteinBasis,
    theta_b: NDArray[np.float64],
    X_c: NDArray[np.float64] | None,
    beta: NDArray[np.float64] | None,
    scaling_c: NDArray[np.float64] | None,
    gamma: NDArray[np.float64] | None,
    offset_c: NDArray[np.float64] | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
]:
    """Evaluate ``h`` at censored rows, supporting the scaled-baseline form.

    For each row ``i``::

        h_0(y_i)  := B_basis(y_i) · θ_b
        f_i       := exp(0.5 · x_s,i · γ)             if scaling_c is given
        h_i       := h_0(y_i) · f_i + X_d,i · β       (+ offset_i)

    Returns ``(h, B_c, h0, f)`` where ``h`` is clipped to ``[-_H_CLIP, _H_CLIP]``,
    ``B_c`` is the Bernstein evaluation matrix, ``h0`` is the unscaled / unshifted
    baseline transformation (``B_c · θ_b``), and ``f`` is the scaling factor
    vector (``None`` on the shift-only path).
    """
    B_c = basis.evaluate(y_c)
    h0 = B_c @ theta_b
    if scaling_c is not None and gamma is not None:
        f: NDArray[np.float64] | None = np.exp(0.5 * (scaling_c @ gamma))
        h_raw = h0 * cast(NDArray[np.float64], f)
    else:
        f = None
        h_raw = h0
    if X_c is not None and beta is not None:
        h_raw = h_raw + X_c @ beta
    if offset_c is not None:
        h_raw = h_raw + offset_c
    h = np.clip(h_raw, -_H_CLIP, _H_CLIP)
    return h, B_c, h0, f


def _add_scaled_gamma_blocks_h(
    H: NDArray[np.float64],
    B_c: NDArray[np.float64],
    X_c: NDArray[np.float64] | None,
    S_c: NDArray[np.float64],
    f_c: NDArray[np.float64],
    h0_c: NDArray[np.float64],
    w_chain: NDArray[np.float64],
    b_grad: NDArray[np.float64],
    p: int,
    q_d: int,
    q_s: int,
) -> None:
    """In-place add (θ_b, γ), (β, γ), (γ, γ) NLL Hessian blocks for one sub-group.

    For a single-endpoint group sharing
    ``h_i = h_0(y_i)·f_i + X_d,i·β`` (right/left censored or exact), the
    per-row Hessian decomposes as
    ``w_chain · (∂h/∂θ)(∂h/∂θ)' + b_grad · ∂²h/∂θ∂θ'`` where ``w_chain``
    is the diagonal chain kernel (``-ψ'`` exact, ``λ(ψ+λ)`` right,
    ``µ(µ-ψ)`` left), ``b_grad = ∂NLL/∂h`` is the per-row gradient
    coefficient (``-ψ`` exact, ``+λ`` right, ``-µ`` left), and both are
    already row-weighted.  Writing ``m := w_chain · h_0 · f + b_grad``::

        (θ_b, γ): 0.5 · f · m · B X_s'
        (β,   γ): 0.5 · w_chain · h_0 · f · X_d X_s'   (chain only; bias = 0)
        (γ,   γ): 0.25 · h_0 · f · m · X_s X_s'

    The (θ_b, θ_b), (θ_b, β), (β, β) sub-blocks are handled by the caller
    via :func:`_assemble_hessian` with ``B̃ = f · B`` and ``w_chain``.
    """
    m_i = w_chain * h0_c * f_c + b_grad
    c_b = 0.5 * f_c * m_i
    H_bg = (B_c * c_b[:, None]).T @ S_c
    H[:p, p + q_d : p + q_d + q_s] += H_bg
    H[p + q_d : p + q_d + q_s, :p] += H_bg.T
    if X_c is not None and q_d > 0:
        c_d = 0.5 * w_chain * h0_c * f_c
        H_dg = (X_c * c_d[:, None]).T @ S_c
        H[p : p + q_d, p + q_d : p + q_d + q_s] += H_dg
        H[p + q_d : p + q_d + q_s, p : p + q_d] += H_dg.T
    c_g = 0.25 * h0_c * f_c * m_i
    H[p + q_d : p + q_d + q_s, p + q_d : p + q_d + q_s] += (S_c * c_g[:, None]).T @ S_c


def _add_grad_censored(
    grad: NDArray[np.float64],
    B_c: NDArray[np.float64],
    X_c: NDArray[np.float64] | None,
    scaling_c: NDArray[np.float64] | None,
    f_c: NDArray[np.float64] | None,
    h0_c: NDArray[np.float64],
    weight: NDArray[np.float64],
    p: int,
    q_d: int,
) -> None:
    """Accumulate ``weight · ∂h/∂θ`` into ``grad`` for one censored sub-group.

    The signed ``weight`` (already multiplied by any row weights) is the
    per-row coefficient that multiplies ``∂h/∂θ`` in the gradient of the
    NLL contribution.  With the scaled-baseline form
    ``h_i = h_0(y_i)·f_i + X_d,i·β`` and ``f_i = exp(0.5 · x_s,i · γ)``::

        ∂h_i/∂θ_b = B_c,i · f_i           (or B_c,i when f_c is None)
        ∂h_i/∂β   = x_d,i
        ∂h_i/∂γ   = 0.5 · h_0(y_i) · f_i · x_s,i

    Modifies ``grad`` in place.
    """
    if f_c is not None:
        grad[:p] += B_c.T @ (weight * f_c)
    else:
        grad[:p] += B_c.T @ weight
    if X_c is not None and q_d > 0:
        grad[p : p + q_d] += X_c.T @ weight
    if scaling_c is not None and f_c is not None:
        grad[p + q_d :] += 0.5 * (scaling_c.T @ (weight * h0_c * f_c))


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


def _neg_score(h: NDArray[np.float64], dist: DistOps) -> NDArray[np.float64]:
    """Compute -(∂ log f(h) / ∂h) for the base distribution.

    Parameters
    ----------
    h : NDArray[np.float64]
        Values of the transformation function.
    dist : DistOps
        Distribution wrapper (dispatched on :attr:`DistOps.kind`).

    Returns
    -------
    NDArray[np.float64]
        The negative derivative of the log-density.

        * normal (N(0,1)):              ``h``
        * logistic:                     ``2 F(h) - 1``
        * min_extreme_value (gumbel_l): ``exp(h) - 1``
        * max_extreme_value (gumbel_r): ``1 - exp(-h)``
        * exponential:                  ``1`` (constant)
        * laplace:                      ``sign(h)``
        * cauchy:                       ``2h / (1 + h²)``
    """
    kind = dist.kind
    if kind == "normal":
        return h
    if kind == "logistic":
        return 2.0 * dist.cdf(h) - 1.0
    if kind == "min_extreme_value":
        return np.exp(h) - 1.0
    if kind == "max_extreme_value":
        return 1.0 - np.exp(-h)
    if kind == "exponential":
        return np.ones_like(h)
    if kind == "laplace":
        return cast(NDArray[np.float64], np.sign(h))
    if kind == "cauchy":
        return 2.0 * h / (1.0 + h**2)
    raise AssertionError(f"unhandled dist.kind={kind!r}")


def _d2_logpdf(h: NDArray[np.float64], dist: DistOps) -> NDArray[np.float64]:
    """Compute ``ψ'(h) = d² log f(h) / dh²`` for the base distribution.

    Parameters
    ----------
    h : NDArray[np.float64]
        Values of the transformation function.
    dist : DistOps
        Distribution wrapper (dispatched on :attr:`DistOps.kind`).

    Returns
    -------
    NDArray[np.float64]
        The second derivative of ``log f`` w.r.t. ``h``.

        * normal (N(0,1)):              ``-1``
        * logistic:                     ``-2 · f(h)`` where ``f`` is logistic pdf
        * min_extreme_value (gumbel_l): ``-exp(h)``
        * max_extreme_value (gumbel_r): ``-exp(-h)``
        * exponential:                  ``0``  (log f is linear in h)
        * laplace:                      ``0``  (log f is piecewise linear; a.e.)
        * cauchy:                       ``2(h² - 1) / (1 + h²)²``
                                        (not log-concave: positive for |h| > 1)

    Notes
    -----
    Used to assemble the analytical Hessian of the log-likelihood.  For
    log-concave base distributions, ``ψ'(h) ≤ 0`` for every h.  Note that
    Cauchy is *not* log-concave, so ``ψ'(h)`` can be positive for ``|h| > 1``,
    meaning the Hessian of the negative log-likelihood may not be PSD.
    """
    kind = dist.kind
    if kind == "normal":
        return np.full_like(h, -1.0)
    if kind == "logistic":
        return -2.0 * dist.pdf(h)
    if kind == "min_extreme_value":
        return cast(NDArray[np.float64], -np.exp(h))
    if kind == "max_extreme_value":
        return cast(NDArray[np.float64], -np.exp(-h))
    if kind == "exponential":
        return np.zeros_like(h)
    if kind == "laplace":
        return np.zeros_like(h)
    if kind == "cauchy":
        denom = 1.0 + h**2
        return 2.0 * (h**2 - 1.0) / denom**2
    raise AssertionError(f"unhandled dist.kind={kind!r}")


def _validate_weights_offset(
    weights: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
    n: int,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Validate and coerce per-observation weights and offset arrays.

    Parameters
    ----------
    weights:
        Per-observation weights. Must have shape ``(n,)``, be finite, and be
        non-negative. ``None`` is equivalent to unit weights (no array
        allocated).
    offset:
        Per-observation fixed linear predictor offset. Must have shape
        ``(n,)`` and be finite. ``None`` is equivalent to zero offset.
    n:
        Expected number of observations.

    Returns
    -------
    (weights, offset)
        Validated, float64-cast arrays or ``None``.

    Raises
    ------
    ValueError
        If ``weights`` or ``offset`` have the wrong shape, contain non-finite
        values, or ``weights`` contains negative values.
    """
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights.shape}.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite (no NaN or inf).")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative.")
        if not np.any(weights > 0.0):
            raise ValueError(
                "weights must have at least one positive entry; "
                "all-zero weights provide no information."
            )
    if offset is not None:
        offset = _validate_offset(offset, n)
    return weights, offset


def _validate_offset(
    offset: NDArray[np.float64],
    n: int,
) -> NDArray[np.float64]:
    """Validate and coerce a per-observation offset array.

    Parameters
    ----------
    offset:
        Array to validate.  Must have shape ``(n,)`` and be finite.
    n:
        Expected length.

    Returns
    -------
    NDArray[np.float64]
        Validated, float64-cast array of shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``offset`` has the wrong shape or contains non-finite values.
    """
    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (n,):
        raise ValueError(f"offset must have shape ({n},), got {offset.shape}.")
    if not np.all(np.isfinite(offset)):
        raise ValueError("offset must be finite (no NaN or inf).")
    return offset


def _inverse_hp(
    hp: NDArray[np.float64], weights: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    """Return ``1 / hp`` (or ``weights / hp``) while silencing divide warnings."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (1.0 / hp) if weights is None else (weights / hp)
    return result


def _outer(
    U: NDArray[np.float64],
    Xu: NDArray[np.float64] | None,
    V: NDArray[np.float64],
    Xv: NDArray[np.float64] | None,
    w_vec: NDArray[np.float64],
    p: int,
    q: int,
) -> NDArray[np.float64]:
    """Compute ``[U, Xu]^T diag(w_vec) [V, Xv]`` as a ``(p+q, p+q)`` block.

    Reused by the interval-censored Hessian and the truncation Hessian to chain
    a 2x2 second-derivative kernel through the design matrices ``[B, X]``.
    """
    Uw = U * w_vec[:, None]
    out = np.zeros((p + q, p + q), dtype=np.float64)
    out[:p, :p] = Uw.T @ V
    if Xu is not None and Xv is not None and q > 0:
        out[:p, p:] = Uw.T @ Xv
        out[p:, :p] = (Xu * w_vec[:, None]).T @ V
        out[p:, p:] = (Xu * w_vec[:, None]).T @ Xv
    return out


# ---------------------------------------------------------------------------
# Truncation correction
# ---------------------------------------------------------------------------
#
# Under truncation [l_i, u_i] each observation is conditioned on being inside
# the observable window:
#
#     ℓ_i(θ) = ℓ_uncond_i(θ) − log P_i(θ),
#     P_i(θ) = F(h(u_i|x_i)) − F(h(l_i|x_i))
#
# This factorisation requires the observation event A_i = [lower_i, upper_i]
# to be contained in B_i = [trunc_lower_i, trunc_upper_i] — otherwise the
# numerator should be P(A_i ∩ B_i), not P(A_i).  CensoredData enforces
# A_i ⊆ B_i at construction (see variables.py), so the helpers below assume
# the precondition holds.
#
# The censoring branch produces ℓ_uncond; the helpers below add the
# −log P_i correction (and its derivatives) on top.  The five public
# dispatchers (log_likelihood, negative_log_likelihood, score_matrix,
# hessian, intercept_score) call into these helpers when the CensoredData
# input carries trunc_lower / trunc_upper.
#
# Per-observation cases (mirrors the fin_lo / fin_hi split used by
# _ll_interval):
#
#   trunc_lower finite, trunc_upper finite  →  log P = log[F(h_u) − F(h_l)]
#   trunc_lower = −∞,    trunc_upper finite →  log P = log F(h_u)
#   trunc_lower finite,  trunc_upper = +∞   →  log P = log S(h_l)
#   both infinite                            →  log P = 0  (no contribution)


def _has_truncation(cd: CensoredData) -> bool:
    """Whether ``cd`` carries any truncation information."""
    return cd.trunc_lower is not None or cd.trunc_upper is not None


def _trunc_bounds(
    cd: CensoredData,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(lo, hi)`` per-observation truncation arrays.

    A missing side is expanded to ``±inf`` so the rest of the helpers can
    treat both bounds uniformly via ``np.isfinite`` masks.
    """
    n = cd.n
    lo = (
        cd.trunc_lower
        if cd.trunc_lower is not None
        else np.full(n, -np.inf, dtype=np.float64)
    )
    hi = (
        cd.trunc_upper
        if cd.trunc_upper is not None
        else np.full(n, np.inf, dtype=np.float64)
    )
    return lo, hi


@dataclass
class _TruncContext:
    """Pre-computed pieces shared across truncation-correction outputs."""

    n: int
    both: NDArray[np.bool_]
    only_hi: NDArray[np.bool_]
    only_lo: NDArray[np.bool_]
    # "both finite" subset
    B_lo_b: NDArray[np.float64]
    B_hi_b: NDArray[np.float64]
    h_lo_b: NDArray[np.float64]
    h_hi_b: NDArray[np.float64]
    X_b: NDArray[np.float64] | None
    # "only upper finite" subset
    B_hi_o: NDArray[np.float64]
    h_hi_o: NDArray[np.float64]
    X_only_hi: NDArray[np.float64] | None
    # "only lower finite" subset
    B_lo_o: NDArray[np.float64]
    h_lo_o: NDArray[np.float64]
    X_only_lo: NDArray[np.float64] | None


def _build_truncation_context(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
) -> _TruncContext:
    """Evaluate basis matrices and clipped ``h`` at the truncation bounds.

    Returns a :class:`_TruncContext` carrying the three sub-mask buffers.
    Empty subsets get zero-row arrays so downstream helpers can blindly
    iterate without ``mask.any()`` guards.
    """
    n = cd.n
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, X)
    lo, hi = _trunc_bounds(cd)
    fin_lo = np.isfinite(lo)
    fin_hi = np.isfinite(hi)
    both = fin_lo & fin_hi
    only_hi = ~fin_lo & fin_hi
    only_lo = fin_lo & ~fin_hi

    def _eval(
        sub_mask: NDArray[np.bool_], y_vals: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        if not sub_mask.any():
            empty_B = np.zeros((0, p), dtype=np.float64)
            empty_h = np.zeros(0, dtype=np.float64)
            empty_X = None if X is None else np.zeros((0, X.shape[1]), dtype=np.float64)
            return empty_B, empty_h, empty_X
        B = basis.evaluate(y_vals)
        X_sub = X[sub_mask] if X is not None else None
        shift = (X_sub @ beta) if (X_sub is not None and beta is not None) else 0.0
        if offset is not None:
            shift = shift + offset[sub_mask]
        h = np.clip(B @ theta_b + shift, -_H_CLIP, _H_CLIP)
        return B, h, X_sub

    B_lo_b, h_lo_b, X_b = _eval(both, lo[both])
    B_hi_b, h_hi_b, _ = _eval(both, hi[both])

    B_hi_o, h_hi_o, X_only_hi = _eval(only_hi, hi[only_hi])
    B_lo_o, h_lo_o, X_only_lo = _eval(only_lo, lo[only_lo])

    return _TruncContext(
        n=n,
        both=both,
        only_hi=only_hi,
        only_lo=only_lo,
        B_lo_b=B_lo_b,
        B_hi_b=B_hi_b,
        h_lo_b=h_lo_b,
        h_hi_b=h_hi_b,
        X_b=X_b,
        B_hi_o=B_hi_o,
        h_hi_o=h_hi_o,
        X_only_hi=X_only_hi,
        B_lo_o=B_lo_o,
        h_lo_o=h_lo_o,
        X_only_lo=X_only_lo,
    )


def _truncation_log_p(ctx: _TruncContext, dist: DistOps) -> NDArray[np.float64]:
    """Per-observation ``log P_i`` (zero where neither bound is finite)."""
    log_p = np.zeros(ctx.n, dtype=np.float64)
    if ctx.both.any():
        log_p[ctx.both] = _log_diff_ndtr(ctx.h_lo_b, ctx.h_hi_b, dist=dist)
    if ctx.only_hi.any():
        _logcdf = dist.logcdf
        log_p[ctx.only_hi] = _logcdf(ctx.h_hi_o)
    if ctx.only_lo.any():
        log_p[ctx.only_lo] = dist.logsf(ctx.h_lo_o)
    return log_p


def _truncation_ll(
    ctx: _TruncContext,
    dist: DistOps,
    weights: NDArray[np.float64] | None,
) -> float:
    """Scalar correction ``-Σ w_i · log P_i`` to add to the log-likelihood."""
    log_p = _truncation_log_p(ctx, dist)
    if weights is not None:
        return float(-np.dot(weights, log_p))
    return float(-np.sum(log_p))


def _truncation_weights(
    ctx: _TruncContext,
    dist: DistOps,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return the four (sub-mask aligned) ``f / P`` weights used by the
    derivative corrections.

    Outputs:

    * ``w_lo_b``, ``w_hi_b`` for the "both finite" subset
      (``= f(h_l) / P``, ``= f(h_u) / P`` respectively).
    * ``w_hi_only`` for the "only upper finite" subset (inverse Mills ratio
      ``f(h_u) / F(h_u)``).
    * ``w_lo_only`` for the "only lower finite" subset (hazard
      ``f(h_l) / S(h_l)``).
    """
    if ctx.both.any():
        log_p_b = _log_diff_ndtr(ctx.h_lo_b, ctx.h_hi_b, dist=dist)
        w_hi_b, w_lo_b = _pair_density_weights(ctx.h_lo_b, ctx.h_hi_b, log_p_b, dist)
    else:
        w_hi_b = np.zeros(0, dtype=np.float64)
        w_lo_b = np.zeros(0, dtype=np.float64)

    if ctx.only_hi.any():
        _logcdf = dist.logcdf
        w_hi_only = np.exp(
            np.minimum(dist.logpdf(ctx.h_hi_o) - _logcdf(ctx.h_hi_o), _LOG_FLOAT_MAX)
        )
    else:
        w_hi_only = np.zeros(0, dtype=np.float64)

    if ctx.only_lo.any():
        w_lo_only = np.exp(
            np.minimum(dist.logpdf(ctx.h_lo_o) - dist.logsf(ctx.h_lo_o), _LOG_FLOAT_MAX)
        )
    else:
        w_lo_only = np.zeros(0, dtype=np.float64)

    return w_lo_b, w_hi_b, w_hi_only, w_lo_only


def _truncation_grad_nll(
    ctx: _TruncContext,
    dist: DistOps,
    weights: NDArray[np.float64] | None,
    p: int,
    q: int,
) -> NDArray[np.float64]:
    """Correction added to ``∂(-ℓ)/∂θ``.

    For each observation ``ℓ_trunc = -log P``, so the contribution to the
    gradient of the *negative* log-likelihood is ``+∂log P/∂θ``:

    * **both finite**: ``+(B(u)·w_hi − B(l)·w_lo)`` for ``θ_b``
      and ``+(w_hi − w_lo)·x`` for ``β``.
    * **only upper finite**: ``+B(u)·µ`` for ``θ_b``, ``+µ·x`` for ``β``,
      where ``µ = f(h_u)/F(h_u)`` (inverse Mills ratio).
    * **only lower finite**: ``-B(l)·λ`` for ``θ_b``, ``-λ·x`` for ``β``,
      where ``λ = f(h_l)/S(h_l)`` (hazard).
    """
    grad = np.zeros(p + q, dtype=np.float64)
    w_lo_b, w_hi_b, w_hi_only, w_lo_only = _truncation_weights(ctx, dist)

    if ctx.both.any():
        if weights is not None:
            ww = weights[ctx.both]
            w_hi_b = ww * w_hi_b
            w_lo_b = ww * w_lo_b
        grad[:p] += ctx.B_hi_b.T @ w_hi_b - ctx.B_lo_b.T @ w_lo_b
        if ctx.X_b is not None:
            grad[p:] += ctx.X_b.T @ (w_hi_b - w_lo_b)

    if ctx.only_hi.any():
        wh = w_hi_only
        if weights is not None:
            wh = weights[ctx.only_hi] * wh
        grad[:p] += ctx.B_hi_o.T @ wh
        if ctx.X_only_hi is not None:
            grad[p:] += ctx.X_only_hi.T @ wh

    if ctx.only_lo.any():
        wl = w_lo_only
        if weights is not None:
            wl = weights[ctx.only_lo] * wl
        grad[:p] -= ctx.B_lo_o.T @ wl
        if ctx.X_only_lo is not None:
            grad[p:] -= ctx.X_only_lo.T @ wl

    return grad


def _truncation_scores(
    ctx: _TruncContext,
    dist: DistOps,
    weights: NDArray[np.float64] | None,
    n: int,
    p: int,
    q: int,
) -> NDArray[np.float64]:
    """Per-observation correction ``∂log P_i/∂θ`` (shape ``(n, p+q)``).

    Caller subtracts this from the unconditional ``∂ℓ_i/∂θ`` matrix.  Rows
    are weighted following the R ``sandwich`` ``estfun`` convention so that
    column sums equal the full weighted gradient of ``-Σ_i log P_i``.
    """
    out = np.zeros((n, p + q), dtype=np.float64)
    w_lo_b, w_hi_b, w_hi_only, w_lo_only = _truncation_weights(ctx, dist)

    if ctx.both.any():
        rows = np.flatnonzero(ctx.both)
        if weights is not None:
            ww = weights[ctx.both]
            w_hi_b = ww * w_hi_b
            w_lo_b = ww * w_lo_b
        out[rows, :p] = ctx.B_hi_b * w_hi_b[:, None] - ctx.B_lo_b * w_lo_b[:, None]
        if ctx.X_b is not None:
            out[rows, p:] = ctx.X_b * (w_hi_b - w_lo_b)[:, None]

    if ctx.only_hi.any():
        rows = np.flatnonzero(ctx.only_hi)
        wh = w_hi_only
        if weights is not None:
            wh = weights[ctx.only_hi] * wh
        out[rows, :p] = ctx.B_hi_o * wh[:, None]
        if ctx.X_only_hi is not None:
            out[rows, p:] = ctx.X_only_hi * wh[:, None]

    if ctx.only_lo.any():
        rows = np.flatnonzero(ctx.only_lo)
        wl = w_lo_only
        if weights is not None:
            wl = weights[ctx.only_lo] * wl
        out[rows, :p] = -ctx.B_lo_o * wl[:, None]
        if ctx.X_only_lo is not None:
            out[rows, p:] = -ctx.X_only_lo * wl[:, None]

    return out


def _truncation_hess_nll(
    ctx: _TruncContext,
    dist: DistOps,
    weights: NDArray[np.float64] | None,
    p: int,
    q: int,
) -> NDArray[np.float64]:
    """Correction added to ``∂²(-ℓ)/∂θ²`` (the negative log-likelihood Hessian).

    Since ``ℓ_trunc = -log P``, the contribution to ``∂²(-ℓ)/∂θ²`` equals
    ``+∂²log P/∂θ²`` — the *negation* of the corresponding interval /
    left / right censoring Hessian block evaluated at the truncation
    bounds.
    """
    H = np.zeros((p + q, p + q), dtype=np.float64)
    w_lo_b, w_hi_b, w_hi_only, w_lo_only = _truncation_weights(ctx, dist)

    if ctx.both.any():
        psi_lo = -_neg_score(ctx.h_lo_b, dist)
        psi_hi = -_neg_score(ctx.h_hi_b, dist)
        a = -psi_lo * w_lo_b - w_lo_b * w_lo_b
        c = psi_hi * w_hi_b - w_hi_b * w_hi_b
        b = w_hi_b * w_lo_b
        if weights is not None:
            ww = weights[ctx.both]
            a = ww * a
            b = ww * b
            c = ww * c
        # ∂²(log P)/∂θ² = chained 2x2 [a b; b c] outer product.  This is the
        # SAME block computed by _hess_interval for the "both finite" path,
        # added with the OPPOSITE sign (interval Hessian subtracts; truncation
        # Hessian — which targets -ℓ_trunc = +log P — adds).
        H += (
            _outer(ctx.B_lo_b, ctx.X_b, ctx.B_lo_b, ctx.X_b, a, p, q)
            + _outer(ctx.B_lo_b, ctx.X_b, ctx.B_hi_b, ctx.X_b, b, p, q)
            + _outer(ctx.B_hi_b, ctx.X_b, ctx.B_lo_b, ctx.X_b, b, p, q)
            + _outer(ctx.B_hi_b, ctx.X_b, ctx.B_hi_b, ctx.X_b, c, p, q)
        )

    if ctx.only_hi.any():
        # log P = log F(h_u);   ∂²log F/∂h² = -µ(µ - ψ).
        psi = -_neg_score(ctx.h_hi_o, dist)
        w_block = w_hi_only * (w_hi_only - psi)
        if weights is not None:
            w_block = weights[ctx.only_hi] * w_block
        H -= _assemble_hessian(ctx.B_hi_o, w_block, ctx.X_only_hi, p, q)

    if ctx.only_lo.any():
        # log P = log S(h_l);   ∂²log S/∂h² = -λ(ψ + λ).
        psi = -_neg_score(ctx.h_lo_o, dist)
        w_block = w_lo_only * (psi + w_lo_only)
        if weights is not None:
            w_block = weights[ctx.only_lo] * w_block
        H -= _assemble_hessian(ctx.B_lo_o, w_block, ctx.X_only_lo, p, q)

    return H


def _truncation_intercept_score(
    ctx: _TruncContext,
    dist: DistOps,
    weights: NDArray[np.float64] | None,
    n: int,
) -> NDArray[np.float64]:
    """Per-observation correction ``∂log P_i/∂α`` (length ``n``).

    Caller subtracts this from the unconditional intercept score.

    Closed forms:

    * **both finite**: ``w_hi − w_lo``
    * **only upper finite**: ``+µ`` (inverse Mills ratio at ``h_u``)
    * **only lower finite**: ``−λ`` (negative hazard at ``h_l``)
    """
    out = np.zeros(n, dtype=np.float64)
    w_lo_b, w_hi_b, w_hi_only, w_lo_only = _truncation_weights(ctx, dist)

    if ctx.both.any():
        vals = w_hi_b - w_lo_b
        if weights is not None:
            vals = weights[ctx.both] * vals
        out[ctx.both] = vals

    if ctx.only_hi.any():
        vals = w_hi_only
        if weights is not None:
            vals = weights[ctx.only_hi] * vals
        out[ctx.only_hi] = vals

    if ctx.only_lo.any():
        vals = -w_lo_only
        if weights is not None:
            vals = weights[ctx.only_lo] * vals
        out[ctx.only_lo] = vals

    return out


# ---------------------------------------------------------------------------
# Private log-likelihood functions — one per censoring type
# ---------------------------------------------------------------------------


def _ll_none(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> np.float64:
    """Computes the log-likelihood for exactly observed (uncensored) data.

    Formula
    -------
    Shift-only path (``scaling is None``)::

        h_i  = B_i · θ_b + X_i · β    (+ offset)
        h'_i = B'_i · θ_b
        ℓ    = Σ w_i [log f(h_i) + log h'_i]

    Scaled path (``scaling is not None``; ADR 0002)::

        f_i  = exp(0.5 · X_s_i · γ)        positive scaling factor
        h_i  = (B_i · θ_b) · f_i + X_i · β (+ offset)
        h'_i = (B'_i · θ_b) · f_i

    The factor of ``0.5`` in the exponent matches mlt's internal convention
    (``mlt:::tmlt`` evaluates ``sterm <- exp(0.5 * <scaling_predict>)``), so
    pymlt's γ is sign- *and* magnitude-aligned with R ``tram``'s scaling
    coefficient.  Without the 0.5, pymlt's γ would be half R's.

    The parameter vector is ``theta = [theta_b | beta | gamma]`` of length
    ``p + q_d + q_s``.

    Parameters
    ----------
    y : NDArray[np.float64]
        Exact observations.
    theta : NDArray[np.float64]
        Concatenated parameter vector ``[theta_b | beta | gamma]``.
    basis : BernsteinBasis
        Polynomial basis object.
    X : NDArray[np.float64] | None
        Covariates.
    dist : DistOps, default=:data:`_NORM_OPS`
        Base distribution.
    weights : NDArray[np.float64] | None
        Per-observation weights of shape ``(len(y),)``. ``None`` = unit weights.
    offset : NDArray[np.float64] | None
        Per-observation offset of shape ``(len(y),)``. ``None`` = zero offset.
    scaling : NDArray[np.float64] | None
        Scaling-design matrix of shape ``(len(y), q_s)``.  ``None`` selects
        the shift-only path.

    Returns
    -------
    float
        Computed log-likelihood.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)

    B, D = basis.evaluate_with_derivative(y)  # (n, p)
    h0 = B @ theta_b
    hp0 = D @ theta_b
    if scaling is not None and gamma is not None:
        # f_i = exp(0.5 · X_s_i · γ); positive, scales both h_0 and h_0'
        # uniformly.  The 0.5 matches mlt's internal convention
        # (mlt:::tmlt uses exp(0.5 * <scaling_predict>)), so γ is
        # sign- and magnitude-aligned with R `tram`'s scaling coefficient.
        f = np.exp(0.5 * (scaling @ gamma))
        h_raw = h0 * f
        hp = hp0 * f
    else:
        h_raw = h0
        hp = hp0
    h_raw = _shift(h_raw, X, beta)
    if offset is not None:
        h_raw = h_raw + offset
    h = np.clip(h_raw, -_H_CLIP, _H_CLIP)

    with np.errstate(invalid="ignore", divide="ignore"):
        # DistOps.logpdf gives exponential log f_exp(h) = -h for all h
        # (matches R mlt; scipy returns -inf for h<0, which would crash
        # penalty-based optimisers that legitimately evaluate slightly-
        # infeasible iterates).  Support feasibility h >= 0 is enforced
        # separately via build_constraints / build_constraint_matrices.
        log_pdf_h = dist.logpdf(h)
        per_obs = log_pdf_h + np.log(hp)
        if weights is not None:
            return cast(np.float64, np.dot(weights, per_obs))
        return np.sum(per_obs)


def _ll_right(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> np.float64:
    """Computes the log-likelihood for right-censored data.

    Formula
    -------
    ℓ = Σ_exact w_i [log f(h) + log h'] + Σ_censored w_i log S(h)
    where S(h) = 1 - F(h) is the survival function.

    When ``scaling`` is provided, ``h(y|x) = h_0(y)·exp(0.5·x_s·γ) + x_d·β``
    (ADR 0002).  Exact-row contributions delegate to :func:`_ll_none`, whose
    scaled-baseline branch is used unchanged.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    ll = np.float64(0.0)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        ll += _ll_none(
            y_e,
            theta,
            basis,
            X_e,
            dist=dist,
            weights=w_e,
            offset=o_e,
            scaling=S_e,
        )

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]  # last known lower bound
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        h_c, _, _, _ = _eval_h_censored(y_c, basis, theta_b, X_c, beta, S_c, gamma, o_c)
        logsf_c = dist.logsf(h_c)
        if w_c is not None:
            ll += np.dot(w_c, logsf_c)
        else:
            ll += np.sum(logsf_c)

    return ll


def _ll_left(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> np.float64:
    """Computes the log-likelihood for left-censored data.

    Formula
    -------
    ℓ = Σ_exact w_i [log f(h) + log h'] + Σ_censored w_i log F(h)

    When ``scaling`` is provided, the scaled-baseline form of ADR 0002
    applies to both blocks; exact rows delegate to :func:`_ll_none`.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    ll = np.float64(0.0)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        ll += _ll_none(
            y_e,
            theta,
            basis,
            X_e,
            dist=dist,
            weights=w_e,
            offset=o_e,
            scaling=S_e,
        )

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]  # last known upper bound
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        h_c, _, _, _ = _eval_h_censored(y_c, basis, theta_b, X_c, beta, S_c, gamma, o_c)
        _logcdf = dist.logcdf
        logcdf_c = _logcdf(h_c)
        if w_c is not None:
            ll += np.dot(w_c, logcdf_c)
        else:
            ll += np.sum(logcdf_c)

    return ll


def _ll_interval(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> np.float64:
    """ℓ = Σ w_i log(F(h(upper_i)) − F(h(lower_i)))  [+ exact terms if present].

    Scaled-baseline form (``scaling`` not ``None``): ``f_i = exp(0.5·x_s,i·γ)``
    is shared between the lower and upper endpoints of each interval (it does
    not depend on ``y``), so the same row-wise factor multiplies both
    ``h_0(lower)`` and ``h_0(upper)``.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    ll = np.float64(0.0)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        ll += _ll_none(
            y_e,
            theta,
            basis,
            X_e,
            dist=dist,
            weights=w_e,
            offset=o_e,
            scaling=S_e,
        )

    mask_c = ~cd.is_exact_mask
    if mask_c.any():
        lo = cd.lower[mask_c]
        hi = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        # Sub-masks within mask_c (relative to its compacted index space):
        # both finite → true interval; only-hi finite → left-open
        # (lower=-∞); only-lo finite → right-open (upper=+∞).
        fin_lo = np.isfinite(lo)
        fin_hi = np.isfinite(hi)
        both = fin_lo & fin_hi
        only_hi = ~fin_lo & fin_hi
        only_lo = fin_lo & ~fin_hi

        log_p = np.zeros(mask_c.sum(), dtype=np.float64)

        def _h_at(
            sub_mask: NDArray[np.bool_],
            y_vals: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            X_sub = X_c[sub_mask] if X_c is not None else None
            S_sub = S_c[sub_mask] if S_c is not None else None
            o_sub = o_c[sub_mask] if o_c is not None else None
            h_sub, _, _, _ = _eval_h_censored(
                y_vals, basis, theta_b, X_sub, beta, S_sub, gamma, o_sub
            )
            return h_sub

        if both.any():
            h_lo_b = _h_at(both, lo[both])
            h_hi_b = _h_at(both, hi[both])
            log_p[both] = _log_diff_ndtr(h_lo_b, h_hi_b, dist=dist)
        if only_hi.any():
            h_hi_o = _h_at(only_hi, hi[only_hi])
            _logcdf = dist.logcdf
            log_p[only_hi] = _logcdf(h_hi_o)
        if only_lo.any():
            h_lo_o = _h_at(only_lo, lo[only_lo])
            log_p[only_lo] = dist.logsf(h_lo_o)

        if w_c is not None:
            ll += np.dot(w_c, log_p)
        else:
            ll += np.sum(log_p)

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
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """∂(-ℓ)/∂θ for exact observations.

    For ``theta = [theta_b | beta | gamma]`` and ``f_i = exp(X_s_i · γ)``::

        ∂h_i/∂θ_b  = B_i · f_i       ∂h'_i/∂θ_b  = B'_i · f_i
        ∂h_i/∂β   = X_i             ∂h'_i/∂β   = 0
        ∂h_i/∂γ   = h_0(y_i)·f_i·X_s_i   ∂h'_i/∂γ = h_0'(y_i)·f_i·X_s_i

    Since ``ns_i = -∂log f(h_i)/∂h_i``, the gradient of ``-ℓ`` is

        ∂(-ℓ)/∂θ = Σ_i w_i · [ns_i · ∂h_i/∂θ − (1/h'_i) · ∂h'_i/∂θ].

    For γ, the ``(1/h'_i) · ∂h'_i/∂γ`` term simplifies to ``X_s_i`` because
    ``h'_i = h_0'(y_i)·f_i``, leaving

        ∂(-ℓ)/∂γ = X_s.T @ (w · ns · h_0 · f − w).
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)

    B, D = basis.evaluate_with_derivative(y)  # (n, p)
    h0 = B @ theta_b
    hp0 = D @ theta_b
    if scaling is not None and gamma is not None:
        # f = exp(0.5 · X_s · γ) — see _ll_none for the 0.5 rationale.
        f = np.exp(0.5 * (scaling @ gamma))
        h_raw = h0 * f
        hp = hp0 * f
    else:
        f = None
        h_raw = h0
        hp = hp0
    h_raw = _shift(h_raw, X, beta)
    if offset is not None:
        h_raw = h_raw + offset
    h = np.clip(h_raw, -_H_CLIP, _H_CLIP)

    ns = _neg_score(h, dist)  # -(∂ log f(h)/∂h)
    wns = ns if weights is None else weights * ns
    if f is not None:
        # ∂h_i/∂θ_b = B_i · f_i;   ∂h'_i/∂θ_b = B'_i · f_i, but
        # (1/h'_i) · ∂h'_i/∂θ_b = B'_i / h_0'(y_i) — the f cancels.
        ihp0 = _inverse_hp(hp0, weights)
        grad_b = (B * f[:, None]).T @ wns - D.T @ ihp0
    else:
        ihp = _inverse_hp(hp, weights)
        grad_b = B.T @ wns - D.T @ ihp

    parts: list[NDArray[np.float64]] = [grad_b]
    if X is not None and beta is not None:
        parts.append(X.T @ wns)
    if scaling is not None and gamma is not None and f is not None:
        # f = exp(0.5 · X_s · γ) ⇒ ∂f/∂γ = 0.5 · X_s · f.
        # ∂(-ℓ)/∂γ = 0.5 · X_s.T @ (w · (ns · h_0 · f − 1)).
        if weights is None:
            term = ns * h0 * f - 1.0
        else:
            term = weights * (ns * h0 * f - 1.0)
        parts.append(0.5 * (scaling.T @ term))
    if len(parts) == 1:
        return grad_b
    return cast(NDArray[np.float64], np.concatenate(parts))


def _grad_right(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Gradient of -ℓ for right-censored data.

    Censored-row contribution to ``∂(-ℓ)/∂θ`` is the hazard
    ``λ(h)=f(h)/S(h)`` chained through ``∂h/∂θ`` (the scaled-baseline
    Jacobian rebuilt per call; see :func:`_add_grad_censored`).
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    grad = np.zeros(p + q_d + q_s, dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        grad += _grad_none(
            y_e,
            theta,
            basis,
            X_e,
            dist=dist,
            weights=w_e,
            offset=o_e,
            scaling=S_e,
        )

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        h_c, B_c, h0_c, f_c = _eval_h_censored(
            y_c, basis, theta_b, X_c, beta, S_c, gamma, o_c
        )
        # ∂(-ℓ)/∂h_c from censored = +hazard
        log_hazard = dist.logpdf(h_c) - dist.logsf(h_c)
        hazard = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
        whazard = hazard if w_c is None else w_c * hazard
        _add_grad_censored(grad, B_c, X_c, S_c, f_c, h0_c, whazard, p, q_d)

    return grad


def _grad_left(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Gradient of -ℓ for left-censored data.

    Censored-row contribution to ``∂(-ℓ)/∂θ`` is ``-µ(h) = -f(h)/F(h)``
    chained through ``∂h/∂θ``.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    grad = np.zeros(p + q_d + q_s, dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        grad += _grad_none(
            y_e,
            theta,
            basis,
            X_e,
            dist=dist,
            weights=w_e,
            offset=o_e,
            scaling=S_e,
        )

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        h_c, B_c, h0_c, f_c = _eval_h_censored(
            y_c, basis, theta_b, X_c, beta, S_c, gamma, o_c
        )
        # ∂(-ℓ)/∂h_c from censored = -inv_mills
        _logcdf = dist.logcdf
        inv_mills = np.exp(np.minimum(dist.logpdf(h_c) - _logcdf(h_c), _LOG_FLOAT_MAX))
        winv = inv_mills if w_c is None else w_c * inv_mills
        _add_grad_censored(grad, B_c, X_c, S_c, f_c, h0_c, -winv, p, q_d)

    return grad


def _grad_interval(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Gradient of -ℓ for interval-censored data.

    Both endpoints of an interval share the same scaling factor
    ``f_i = exp(0.5·x_s,i·γ)`` (depends only on ``x_s``, not on ``y``), so
    ``f_i`` is computed once per row from the row's scaling design and reused
    for both ``h_lo`` and ``h_hi`` via :func:`_eval_h_censored`.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    grad = np.zeros(p + q_d + q_s, dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        grad += _grad_none(
            y_e,
            theta,
            basis,
            X_e,
            dist=dist,
            weights=w_e,
            offset=o_e,
            scaling=S_e,
        )

    mask_c = ~cd.is_exact_mask
    if mask_c.any():
        lo = cd.lower[mask_c]
        hi = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        fin_lo = np.isfinite(lo)
        fin_hi = np.isfinite(hi)
        both = fin_lo & fin_hi
        only_hi = ~fin_lo & fin_hi
        only_lo = fin_lo & ~fin_hi

        if both.any():
            X_b = X_c[both] if X_c is not None else None
            S_b = S_c[both] if S_c is not None else None
            o_b = o_c[both] if o_c is not None else None
            h_lo_b, B_lo_b, h0_lo_b, f_b = _eval_h_censored(
                lo[both], basis, theta_b, X_b, beta, S_b, gamma, o_b
            )
            h_hi_b, B_hi_b, h0_hi_b, _ = _eval_h_censored(
                hi[both], basis, theta_b, X_b, beta, S_b, gamma, o_b
            )
            log_p_b = _log_diff_ndtr(h_lo_b, h_hi_b, dist=dist)
            w_hi_b, w_lo_b = _pair_density_weights(h_lo_b, h_hi_b, log_p_b, dist)
            if w_c is not None:
                ww = w_c[both]
                w_hi_b = ww * w_hi_b
                w_lo_b = ww * w_lo_b
            with np.errstate(invalid="ignore"):
                # ∂(-ℓ)/∂h_hi = -w_hi; ∂(-ℓ)/∂h_lo = +w_lo
                _add_grad_censored(
                    grad, B_hi_b, X_b, S_b, f_b, h0_hi_b, -w_hi_b, p, q_d
                )
                _add_grad_censored(grad, B_lo_b, X_b, S_b, f_b, h0_lo_b, w_lo_b, p, q_d)

        if only_hi.any():
            # Left-open row: lower=-∞, upper=h_hi.  Same form as _grad_left.
            X_o = X_c[only_hi] if X_c is not None else None
            S_o = S_c[only_hi] if S_c is not None else None
            o_o = o_c[only_hi] if o_c is not None else None
            h_hi_o, B_hi_o, h0_hi_o, f_o = _eval_h_censored(
                hi[only_hi], basis, theta_b, X_o, beta, S_o, gamma, o_o
            )
            _logcdf = dist.logcdf
            inv_mills = np.exp(
                np.minimum(dist.logpdf(h_hi_o) - _logcdf(h_hi_o), _LOG_FLOAT_MAX)
            )
            if w_c is not None:
                inv_mills = w_c[only_hi] * inv_mills
            _add_grad_censored(grad, B_hi_o, X_o, S_o, f_o, h0_hi_o, -inv_mills, p, q_d)

        if only_lo.any():
            # Right-open row: lower=h_lo, upper=+∞.  Same form as _grad_right.
            X_o = X_c[only_lo] if X_c is not None else None
            S_o = S_c[only_lo] if S_c is not None else None
            o_o = o_c[only_lo] if o_c is not None else None
            h_lo_o, B_lo_o, h0_lo_o, f_o = _eval_h_censored(
                lo[only_lo], basis, theta_b, X_o, beta, S_o, gamma, o_o
            )
            log_hazard = dist.logpdf(h_lo_o) - dist.logsf(h_lo_o)
            hazard = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
            if w_c is not None:
                hazard = w_c[only_lo] * hazard
            _add_grad_censored(grad, B_lo_o, X_o, S_o, f_o, h0_lo_o, hazard, p, q_d)

    return grad


# ---------------------------------------------------------------------------
# Private combined LL + gradient helpers — one per censoring type.
# Used by negative_log_likelihood(gradient=True) so the optimiser pays for
# basis.evaluate / basis.derivative and the mask slicing once per iteration
# instead of twice.  Return (ll, grad_of_negative_ll).
# ---------------------------------------------------------------------------


def _ll_and_grad_none(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> tuple[np.float64, NDArray[np.float64]]:
    """Combined ℓ and ∂(-ℓ)/∂θ for exact observations.  See :func:`_grad_none`
    for the scaled-path formulae."""
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)

    B, D = basis.evaluate_with_derivative(y)
    h0 = B @ theta_b
    hp0 = D @ theta_b
    if scaling is not None and gamma is not None:
        # f = exp(0.5 · X_s · γ) — see _ll_none for the 0.5 rationale.
        f = np.exp(0.5 * (scaling @ gamma))
        h_raw = h0 * f
        hp = hp0 * f
    else:
        f = None
        h_raw = h0
        hp = hp0
    h_raw = _shift(h_raw, X, beta)
    if offset is not None:
        h_raw = h_raw + offset
    h = np.clip(h_raw, -_H_CLIP, _H_CLIP)

    ns = _neg_score(h, dist)
    wns = ns if weights is None else weights * ns
    with np.errstate(invalid="ignore", divide="ignore"):
        # Smooth analytical extension for exponential at h<0 — see _ll_none.
        log_pdf_h = dist.logpdf(h)
        if weights is not None:
            ll = np.dot(weights, log_pdf_h + np.log(hp))
        else:
            ll = np.sum(log_pdf_h) + np.sum(np.log(hp))
        if f is not None:
            ihp0 = _inverse_hp(hp0, weights)
            grad_b = (B * f[:, None]).T @ wns - D.T @ ihp0
        else:
            ihp = _inverse_hp(hp, weights)
            grad_b = B.T @ wns - D.T @ ihp
    parts: list[NDArray[np.float64]] = [grad_b]
    if X is not None and beta is not None:
        parts.append(X.T @ wns)
    if scaling is not None and gamma is not None and f is not None:
        # f = exp(0.5 · X_s · γ) — factor 0.5 (see _ll_none / _grad_none).
        if weights is None:
            term = ns * h0 * f - 1.0
        else:
            term = weights * (ns * h0 * f - 1.0)
        parts.append(0.5 * (scaling.T @ term))
    grad: NDArray[np.float64] = grad_b if len(parts) == 1 else np.concatenate(parts)
    return ll, grad


def _ll_and_grad_right(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> tuple[np.float64, NDArray[np.float64]]:
    """Combined ℓ and ∂(-ℓ)/∂θ for right-censored data."""
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    ll = np.float64(0.0)
    grad = np.zeros(p + q_d + q_s, dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        ll_e, grad_e = _ll_and_grad_none(
            y_e,
            theta,
            basis,
            X_e,
            dist=dist,
            weights=w_e,
            offset=o_e,
            scaling=S_e,
        )
        ll += ll_e
        grad += grad_e

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        h_c, B_c, h0_c, f_c = _eval_h_censored(
            y_c, basis, theta_b, X_c, beta, S_c, gamma, o_c
        )
        logsf_c = dist.logsf(h_c)
        if w_c is not None:
            ll += np.dot(w_c, logsf_c)
        else:
            ll += np.sum(logsf_c)
        log_hazard = dist.logpdf(h_c) - logsf_c
        hazard = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
        whazard = hazard if w_c is None else w_c * hazard
        _add_grad_censored(grad, B_c, X_c, S_c, f_c, h0_c, whazard, p, q_d)

    return ll, grad


def _ll_and_grad_left(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> tuple[np.float64, NDArray[np.float64]]:
    """Combined ℓ and ∂(-ℓ)/∂θ for left-censored data."""
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    ll = np.float64(0.0)
    grad = np.zeros(p + q_d + q_s, dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        ll_e, grad_e = _ll_and_grad_none(
            y_e,
            theta,
            basis,
            X_e,
            dist=dist,
            weights=w_e,
            offset=o_e,
            scaling=S_e,
        )
        ll += ll_e
        grad += grad_e

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        h_c, B_c, h0_c, f_c = _eval_h_censored(
            y_c, basis, theta_b, X_c, beta, S_c, gamma, o_c
        )
        _logcdf = dist.logcdf
        log_Fc = _logcdf(h_c)
        if w_c is not None:
            ll += np.dot(w_c, log_Fc)
        else:
            ll += np.sum(log_Fc)
        inv_mills = np.exp(np.minimum(dist.logpdf(h_c) - log_Fc, _LOG_FLOAT_MAX))
        winv = inv_mills if w_c is None else w_c * inv_mills
        _add_grad_censored(grad, B_c, X_c, S_c, f_c, h0_c, -winv, p, q_d)

    return ll, grad


def _ll_and_grad_interval(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> tuple[np.float64, NDArray[np.float64]]:
    """Combined ℓ and ∂(-ℓ)/∂θ for interval-censored data."""
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    ll = np.float64(0.0)
    grad = np.zeros(p + q_d + q_s, dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        ll_e, grad_e = _ll_and_grad_none(
            y_e,
            theta,
            basis,
            X_e,
            dist=dist,
            weights=w_e,
            offset=o_e,
            scaling=S_e,
        )
        ll += ll_e
        grad += grad_e

    mask_c = ~cd.is_exact_mask
    if mask_c.any():
        lo = cd.lower[mask_c]
        hi = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        fin_lo = np.isfinite(lo)
        fin_hi = np.isfinite(hi)
        both = fin_lo & fin_hi
        only_hi = ~fin_lo & fin_hi
        only_lo = fin_lo & ~fin_hi

        if both.any():
            X_b = X_c[both] if X_c is not None else None
            S_b = S_c[both] if S_c is not None else None
            o_b = o_c[both] if o_c is not None else None
            h_lo_b, B_lo_b, h0_lo_b, f_b = _eval_h_censored(
                lo[both], basis, theta_b, X_b, beta, S_b, gamma, o_b
            )
            h_hi_b, B_hi_b, h0_hi_b, _ = _eval_h_censored(
                hi[both], basis, theta_b, X_b, beta, S_b, gamma, o_b
            )
            log_p_b = _log_diff_ndtr(h_lo_b, h_hi_b, dist=dist)
            ww_b = w_c[both] if w_c is not None else None
            if ww_b is not None:
                ll += np.dot(ww_b, log_p_b)
            else:
                ll += np.sum(log_p_b)
            w_hi_b, w_lo_b = _pair_density_weights(h_lo_b, h_hi_b, log_p_b, dist)
            if ww_b is not None:
                w_hi_b = ww_b * w_hi_b
                w_lo_b = ww_b * w_lo_b
            with np.errstate(invalid="ignore"):
                _add_grad_censored(
                    grad, B_hi_b, X_b, S_b, f_b, h0_hi_b, -w_hi_b, p, q_d
                )
                _add_grad_censored(grad, B_lo_b, X_b, S_b, f_b, h0_lo_b, w_lo_b, p, q_d)

        if only_hi.any():
            X_o = X_c[only_hi] if X_c is not None else None
            S_o = S_c[only_hi] if S_c is not None else None
            o_o = o_c[only_hi] if o_c is not None else None
            h_hi_o, B_hi_o, h0_hi_o, f_o = _eval_h_censored(
                hi[only_hi], basis, theta_b, X_o, beta, S_o, gamma, o_o
            )
            _logcdf = dist.logcdf
            log_Fc = _logcdf(h_hi_o)
            ww_o = w_c[only_hi] if w_c is not None else None
            if ww_o is not None:
                ll += np.dot(ww_o, log_Fc)
            else:
                ll += np.sum(log_Fc)
            inv_mills = np.exp(np.minimum(dist.logpdf(h_hi_o) - log_Fc, _LOG_FLOAT_MAX))
            if ww_o is not None:
                inv_mills = ww_o * inv_mills
            _add_grad_censored(grad, B_hi_o, X_o, S_o, f_o, h0_hi_o, -inv_mills, p, q_d)

        if only_lo.any():
            X_o = X_c[only_lo] if X_c is not None else None
            S_o = S_c[only_lo] if S_c is not None else None
            o_o = o_c[only_lo] if o_c is not None else None
            h_lo_o, B_lo_o, h0_lo_o, f_o = _eval_h_censored(
                lo[only_lo], basis, theta_b, X_o, beta, S_o, gamma, o_o
            )
            logsf_o = dist.logsf(h_lo_o)
            ww_o = w_c[only_lo] if w_c is not None else None
            if ww_o is not None:
                ll += np.dot(ww_o, logsf_o)
            else:
                ll += np.sum(logsf_o)
            log_hazard = dist.logpdf(h_lo_o) - logsf_o
            hazard = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
            if ww_o is not None:
                hazard = ww_o * hazard
            _add_grad_censored(grad, B_lo_o, X_o, S_o, f_o, h0_lo_o, hazard, p, q_d)

    return ll, grad


# ---------------------------------------------------------------------------
# Private per-observation score functions — one per censoring type.
# Each returns an (n, p+q) matrix of gradients of the POSITIVE log-likelihood
# ℓ_i w.r.t. theta, with rows aligned to the input ordering.
# Row i is multiplied by weights[i] when weights is provided (estfun convention).
# ---------------------------------------------------------------------------


def _scores_none(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Per-observation ∂ℓ/∂θ for exact observations, shape ``(n, p+q_d+q_s)``.

    With ``theta = [theta_b | beta | gamma]`` and
    ``f_i = exp(0.5 · X_s,i · γ)`` (ADR 0002, Decision 4)::

        ∂ℓ_i/∂θ_b = ψ(h_i) · B_i · f_i + B'_i / h_0'(y_i)
        ∂ℓ_i/∂β   = ψ(h_i) · x_d,i
        ∂ℓ_i/∂γ   = 0.5 · X_s,i · (ψ(h_i) · h_0(y_i) · f_i + 1)
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)

    B, D = basis.evaluate_with_derivative(y)  # (n, p)
    h0 = B @ theta_b
    hp0 = D @ theta_b
    if scaling is not None and gamma is not None:
        f = np.exp(0.5 * (scaling @ gamma))
        h_raw = h0 * f
        hp = hp0 * f
    else:
        f = None
        h_raw = h0
        hp = hp0
    h_raw = _shift(h_raw, X, beta)
    if offset is not None:
        h_raw = h_raw + offset
    h = np.clip(h_raw, -_H_CLIP, _H_CLIP)

    psi = -_neg_score(h, dist)  # ψ(h) = d log f / dh, shape (n,)
    # ∂ℓ_i/∂θ_b = ψ(h_i) · B_i · f_i + D_i / h_0'(y_i)
    if f is not None:
        scores_b = (B * f[:, None]) * psi[:, None] + D / hp0[:, None]
    else:
        scores_b = B * psi[:, None] + D / hp[:, None]

    scores = np.empty((len(y), p + q_d + q_s), dtype=np.float64)
    scores[:, :p] = scores_b
    if X is not None:
        # ∂ℓ_i/∂β = x_d,i · ψ(h_i)
        scores[:, p : p + q_d] = X * psi[:, None]
    if scaling is not None and f is not None:
        # ∂ℓ_i/∂γ = 0.5 · X_s,i · (ψ(h_i) · h_0(y_i) · f_i + 1)
        gamma_factor = 0.5 * (psi * h0 * f + 1.0)
        scores[:, p + q_d :] = scaling * gamma_factor[:, None]
    if weights is not None:
        scores *= weights[:, None]
    return scores


def _scores_right(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Per-observation ∂ℓ/∂θ for right-censored data, shape ``(n, p+q_d+q_s)``.

    Scaled-baseline form (``scaling is not None``): for censored rows at
    ``h_c = h_0(y_i)·f_i + X_d,i·β`` with ``f_i = exp(0.5 X_s,i γ)``::

        ∂ℓ_i/∂θ_b = -λ · f · B_i
        ∂ℓ_i/∂β   = -λ · X_d,i
        ∂ℓ_i/∂γ   = -λ · 0.5 · h_0(y_i) · f · X_s,i
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    n = cd.n
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    scores = np.zeros((n, p + q_d + q_s), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        scores[mask_e] = _scores_none(
            y_e, theta, basis, X_e, dist=dist, weights=w_e, offset=o_e, scaling=S_e
        )

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        h_c, B_c, h0_c, f_c = _eval_h_censored(
            y_c, basis, theta_b, X_c, beta, S_c, gamma, o_c
        )
        # ∂ℓ_i/∂h = -λ(h) = -f(h)/S(h)
        log_hazard = dist.logpdf(h_c) - dist.logsf(h_c)
        hazard = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
        if w_c is not None:
            hazard = w_c * hazard
        B_eff = B_c if f_c is None else B_c * f_c[:, None]
        scores[mask_c, :p] = -B_eff * hazard[:, None]
        if X_c is not None:
            scores[mask_c, p : p + q_d] = -X_c * hazard[:, None]
        if S_c is not None and f_c is not None:
            scores[mask_c, p + q_d :] = -S_c * (0.5 * h0_c * f_c * hazard)[:, None]

    return scores


def _scores_left(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Per-observation ∂ℓ/∂θ for left-censored data, shape ``(n, p+q_d+q_s)``.

    Scaled-baseline form: censored rows use ``∂ℓ_i/∂h = +µ`` (inverse Mills)
    chained through ``∂h/∂θ`` with the γ Jacobian ``0.5·h_0·f·X_s``.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    n = cd.n
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    scores = np.zeros((n, p + q_d + q_s), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        scores[mask_e] = _scores_none(
            y_e, theta, basis, X_e, dist=dist, weights=w_e, offset=o_e, scaling=S_e
        )

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        h_c, B_c, h0_c, f_c = _eval_h_censored(
            y_c, basis, theta_b, X_c, beta, S_c, gamma, o_c
        )
        # ∂ℓ_i/∂h = µ(h) = f(h)/F(h)
        _logcdf = dist.logcdf
        inv_mills = np.exp(np.minimum(dist.logpdf(h_c) - _logcdf(h_c), _LOG_FLOAT_MAX))
        if w_c is not None:
            inv_mills = w_c * inv_mills
        B_eff = B_c if f_c is None else B_c * f_c[:, None]
        scores[mask_c, :p] = B_eff * inv_mills[:, None]
        if X_c is not None:
            scores[mask_c, p : p + q_d] = X_c * inv_mills[:, None]
        if S_c is not None and f_c is not None:
            scores[mask_c, p + q_d :] = S_c * (0.5 * h0_c * f_c * inv_mills)[:, None]

    return scores


def _scores_interval(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Per-observation ∂ℓ/∂θ for interval-censored data, shape ``(n, p+q_d+q_s)``.

    Scaled-baseline form: for two-sided rows at ``[lo, hi]``::

        ∂ℓ_i/∂θ_b = f · (w_hi · B_hi - w_lo · B_lo)
        ∂ℓ_i/∂β   = (w_hi - w_lo) · X_d,i
        ∂ℓ_i/∂γ   = 0.5 · f · X_s,i · (w_hi · h_0(hi_i) - w_lo · h_0(lo_i))

    Right-open (``only_lo``) rows reduce to right-censored at ``lo``;
    left-open (``only_hi``) rows reduce to left-censored at ``hi``.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    n = cd.n
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    scores = np.zeros((n, p + q_d + q_s), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        scores[mask_e] = _scores_none(
            y_e, theta, basis, X_e, dist=dist, weights=w_e, offset=o_e, scaling=S_e
        )

    mask_c = ~cd.is_exact_mask
    if mask_c.any():
        idx_c = np.flatnonzero(mask_c)
        lo = cd.lower[mask_c]
        hi = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        fin_lo = np.isfinite(lo)
        fin_hi = np.isfinite(hi)
        both = fin_lo & fin_hi
        only_hi = ~fin_lo & fin_hi
        only_lo = fin_lo & ~fin_hi

        if both.any():
            rows = idx_c[both]
            X_b = X_c[both] if X_c is not None else None
            w_b = w_c[both] if w_c is not None else None
            o_b = o_c[both] if o_c is not None else None
            S_b = S_c[both] if S_c is not None else None
            h_lo_b, B_lo_b, h0_lo_b, f_b = _eval_h_censored(
                lo[both], basis, theta_b, X_b, beta, S_b, gamma, o_b
            )
            h_hi_b, B_hi_b, h0_hi_b, _ = _eval_h_censored(
                hi[both], basis, theta_b, X_b, beta, S_b, gamma, o_b
            )
            log_p_b = _log_diff_ndtr(h_lo_b, h_hi_b, dist=dist)
            w_hi_b, w_lo_b = _pair_density_weights(h_lo_b, h_hi_b, log_p_b, dist)
            if w_b is not None:
                w_hi_b = w_b * w_hi_b
                w_lo_b = w_b * w_lo_b
            if f_b is None:
                B_eff_lo = B_lo_b
                B_eff_hi = B_hi_b
            else:
                B_eff_lo = B_lo_b * f_b[:, None]
                B_eff_hi = B_hi_b * f_b[:, None]
            scores[rows, :p] = B_eff_hi * w_hi_b[:, None] - B_eff_lo * w_lo_b[:, None]
            if X_b is not None:
                scores[rows, p : p + q_d] = X_b * (w_hi_b - w_lo_b)[:, None]
            if S_b is not None and f_b is not None:
                gamma_coef = 0.5 * f_b * (w_hi_b * h0_hi_b - w_lo_b * h0_lo_b)
                scores[rows, p + q_d :] = S_b * gamma_coef[:, None]

        if only_hi.any():
            rows = idx_c[only_hi]
            X_o = X_c[only_hi] if X_c is not None else None
            w_o = w_c[only_hi] if w_c is not None else None
            o_o = o_c[only_hi] if o_c is not None else None
            S_o = S_c[only_hi] if S_c is not None else None
            h_hi_o, B_hi_o, h0_hi_o, f_o = _eval_h_censored(
                hi[only_hi], basis, theta_b, X_o, beta, S_o, gamma, o_o
            )
            _logcdf = dist.logcdf
            inv_mills = np.exp(
                np.minimum(dist.logpdf(h_hi_o) - _logcdf(h_hi_o), _LOG_FLOAT_MAX)
            )
            if w_o is not None:
                inv_mills = w_o * inv_mills
            B_eff = B_hi_o if f_o is None else B_hi_o * f_o[:, None]
            scores[rows, :p] = B_eff * inv_mills[:, None]
            if X_o is not None:
                scores[rows, p : p + q_d] = X_o * inv_mills[:, None]
            if S_o is not None and f_o is not None:
                scores[rows, p + q_d :] = (
                    S_o * (0.5 * h0_hi_o * f_o * inv_mills)[:, None]
                )

        if only_lo.any():
            rows = idx_c[only_lo]
            X_o = X_c[only_lo] if X_c is not None else None
            w_o = w_c[only_lo] if w_c is not None else None
            o_o = o_c[only_lo] if o_c is not None else None
            S_o = S_c[only_lo] if S_c is not None else None
            h_lo_o, B_lo_o, h0_lo_o, f_o = _eval_h_censored(
                lo[only_lo], basis, theta_b, X_o, beta, S_o, gamma, o_o
            )
            log_hazard = dist.logpdf(h_lo_o) - dist.logsf(h_lo_o)
            hazard = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
            if w_o is not None:
                hazard = w_o * hazard
            B_eff = B_lo_o if f_o is None else B_lo_o * f_o[:, None]
            scores[rows, :p] = -B_eff * hazard[:, None]
            if X_o is not None:
                scores[rows, p : p + q_d] = -X_o * hazard[:, None]
            if S_o is not None and f_o is not None:
                scores[rows, p + q_d :] = -S_o * (0.5 * h0_lo_o * f_o * hazard)[:, None]

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
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Hessian of -ℓ for exact observations, shape ``(p+q_d+q_s, p+q_d+q_s)``.

    Shift-only per-observation contribution to ``∂²(-ℓ)/∂θ∂θ'``::

        [θ_b θ_b]:  -ψ'(h) · B_i B_i' + (D_i D_i') / (h'_i)²
        [θ_b β  ]:  -ψ'(h) · B_i x_d,i'
        [β   β  ]:  -ψ'(h) · x_d,i x_d,i'

    Scaled path (``scaling is not None``; ADR 0002, Decision 4).  With
    ``f_i = exp(0.5 · X_s,i · γ)``, ``B̃_i = f_i · B_i`` and
    ``m_i := w_chain_i · h_0(y_i) · f_i − ψ(h_i)``, the additional/modified
    blocks are::

        [θ_b θ_b]:  w_chain · B̃_i B̃_i' + (D_i D_i')/(h_0')²   (f cancels in h')
        [θ_b β  ]:  w_chain · B̃_i x_d,i'
        [θ_b γ  ]:  0.5 · f_i · m_i · B_i X_s,i'
        [β   γ  ]:  0.5 · w_chain · h_0 · f · X_d,i X_s,i'
        [γ   γ  ]:  0.25 · h_0 · f · m_i · X_s,i X_s,i'

    where ``w_chain := -ψ'(h)`` and ``ψ(h) = d log f_Z / dh``.  The two
    terms inside ``m_i`` come from chain-ruling ``-ψ'·(∂h/∂θ)(∂h/∂θ)'``
    and ``-ψ·∂²h/∂θ∂θ'`` (the latter is non-zero only for the γ blocks
    because ``h`` is non-linear in γ).
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)

    B, D = basis.evaluate_with_derivative(y)  # (n, p)
    h0 = B @ theta_b
    hp0 = D @ theta_b
    if scaling is not None and gamma is not None:
        f = np.exp(0.5 * (scaling @ gamma))
        h_raw = h0 * f
    else:
        f = None
        h_raw = h0
    h_raw = _shift(h_raw, X, beta)
    if offset is not None:
        h_raw = h_raw + offset
    h = np.clip(h_raw, -_H_CLIP, _H_CLIP)

    w_chain = -_d2_logpdf(h, dist)  # -ψ'(h), ≥ 0 for log-concave f
    if weights is not None:
        w_chain = weights * w_chain

    # θ_b / β block: use B̃ = f · B (or B if shift-only) for the chain-rule
    # outer product term.  The ``-log h'`` curvature term lives on (θ_b, θ_b)
    # and uses h_0' (the ``0.5·X_s·γ`` piece of log h' is linear in γ → 0).
    B_chain = B if f is None else B * f[:, None]
    H_shift = _assemble_hessian(B_chain, w_chain, X, p, q_d)
    inv_hp2 = 1.0 / (hp0 * hp0)
    if weights is not None:
        inv_hp2 = weights * inv_hp2
    Dw = D * inv_hp2[:, None]
    H_shift[:p, :p] += Dw.T @ D

    if scaling is None or f is None:
        return H_shift

    H = np.zeros((p + q_d + q_s, p + q_d + q_s), dtype=np.float64)
    H[: p + q_d, : p + q_d] = H_shift
    # γ blocks.  ∂NLL_exact/∂h = -ψ (already weighted via psi_w below).
    psi = -_neg_score(h, dist)
    psi_w = psi if weights is None else weights * psi
    _add_scaled_gamma_blocks_h(H, B, X, scaling, f, h0, w_chain, -psi_w, p, q_d, q_s)
    return H


def _hess_right(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Hessian of -ℓ for right-censored data, shape ``(p+q_d+q_s, p+q_d+q_s)``.

    Exact rows contribute via :func:`_hess_none`.  Right-censored rows at
    lower bound ``h_l`` contribute ``λ(h)·(ψ(h) + λ(h))`` on the shared
    ``[B, x]`` design (with ``B̃ = f·B`` under scaling), where ``λ = f/S``
    is the hazard.  Scaled γ blocks use ``∂NLL/∂h = +λ`` as the bias
    coefficient via :func:`_add_scaled_gamma_blocks_h`.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    H = np.zeros((p + q_d + q_s, p + q_d + q_s), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        H += _hess_none(
            y_e, theta, basis, X_e, dist=dist, weights=w_e, offset=o_e, scaling=S_e
        )

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        h_c, B_c, h0_c, f_c = _eval_h_censored(
            y_c, basis, theta_b, X_c, beta, S_c, gamma, o_c
        )
        log_hazard = dist.logpdf(h_c) - dist.logsf(h_c)
        lam = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
        psi = -_neg_score(h_c, dist)
        w_chain = lam * (psi + lam)  # = -d²logS/dh² → NLL contribution
        if w_c is not None:
            w_chain = w_c * w_chain
            lam_w = w_c * lam
        else:
            lam_w = lam
        B_eff = B_c if f_c is None else B_c * f_c[:, None]
        H[: p + q_d, : p + q_d] += _assemble_hessian(B_eff, w_chain, X_c, p, q_d)
        if S_c is not None and f_c is not None:
            _add_scaled_gamma_blocks_h(
                H, B_c, X_c, S_c, f_c, h0_c, w_chain, lam_w, p, q_d, q_s
            )

    return H


def _hess_left(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Hessian of -ℓ for left-censored data, shape ``(p+q_d+q_s, p+q_d+q_s)``.

    Left-censored rows at upper bound ``h_u`` contribute
    ``µ(h)·(µ(h) - ψ(h))`` (chain kernel) where ``µ = f/F`` is the inverse
    Mills ratio.  Under scaling, ``∂NLL/∂h = -µ`` feeds the γ-block bias
    term through :func:`_add_scaled_gamma_blocks_h`.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    H = np.zeros((p + q_d + q_s, p + q_d + q_s), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        H += _hess_none(
            y_e, theta, basis, X_e, dist=dist, weights=w_e, offset=o_e, scaling=S_e
        )

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        h_c, B_c, h0_c, f_c = _eval_h_censored(
            y_c, basis, theta_b, X_c, beta, S_c, gamma, o_c
        )
        _logcdf = dist.logcdf
        mu = np.exp(np.minimum(dist.logpdf(h_c) - _logcdf(h_c), _LOG_FLOAT_MAX))
        psi = -_neg_score(h_c, dist)
        w_chain = mu * (mu - psi)
        if w_c is not None:
            w_chain = w_c * w_chain
            mu_w = w_c * mu
        else:
            mu_w = mu
        B_eff = B_c if f_c is None else B_c * f_c[:, None]
        H[: p + q_d, : p + q_d] += _assemble_hessian(B_eff, w_chain, X_c, p, q_d)
        if S_c is not None and f_c is not None:
            # ∂NLL_left/∂h = -µ  (since NLL = -log F)
            _add_scaled_gamma_blocks_h(
                H, B_c, X_c, S_c, f_c, h0_c, w_chain, -mu_w, p, q_d, q_s
            )

    return H


def _hess_interval(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Hessian of -ℓ for interval-censored data, shape ``(p+q_d+q_s, p+q_d+q_s)``.

    For each interval ``[h_l, h_u]`` with ``p = F(h_u) - F(h_l)``,
    ``w_lo = f(h_l)/p``, ``w_hi = f(h_u)/p``, the 2x2 Hessian of
    ``log p`` w.r.t. ``(h_l, h_u)`` has entries::

        ∂²/∂h_l² = -ψ(h_l) w_lo - w_lo²
        ∂²/∂h_u² =  ψ(h_u) w_hi - w_hi²
        ∂²/∂h_l ∂h_u = w_hi · w_lo

    Chained through the Jacobian
    ``∂(h_l, h_u)/∂(θ_b, β, γ) = [[f·B_lo, X_d, 0.5·h_0(lo)·f·X_s],
    [f·B_hi, X_d, 0.5·h_0(hi)·f·X_s]]`` and negated for NLL.  The bias
    correction ``b_lo·∂²h_lo + b_hi·∂²h_hi`` with NLL gradients
    ``b_lo = +w_lo`` and ``b_hi = -w_hi`` contributes to the γ blocks
    (``∂²h/∂θ_b∂γ`` and ``∂²h/∂γ²`` are non-zero) and is added inline.
    Right-open / left-open sub-cases reduce to right / left censoring at
    the finite endpoint.
    """
    p = basis.order + 1
    q_d = X.shape[1] if X is not None else 0
    q_s = scaling.shape[1] if scaling is not None else 0
    theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
    H = np.zeros((p + q_d + q_s, p + q_d + q_s), dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        S_e = scaling[mask_e] if scaling is not None else None
        H += _hess_none(
            y_e, theta, basis, X_e, dist=dist, weights=w_e, offset=o_e, scaling=S_e
        )

    mask_c = ~cd.is_exact_mask
    if mask_c.any():
        lo = cd.lower[mask_c]
        hi = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        S_c = scaling[mask_c] if scaling is not None else None
        fin_lo = np.isfinite(lo)
        fin_hi = np.isfinite(hi)
        both = fin_lo & fin_hi
        only_hi = ~fin_lo & fin_hi
        only_lo = fin_lo & ~fin_hi

        if both.any():
            X_b = X_c[both] if X_c is not None else None
            w_b = w_c[both] if w_c is not None else None
            o_b = o_c[both] if o_c is not None else None
            S_b = S_c[both] if S_c is not None else None
            h_lo_b, B_lo_b, h0_lo_b, f_b = _eval_h_censored(
                lo[both], basis, theta_b, X_b, beta, S_b, gamma, o_b
            )
            h_hi_b, B_hi_b, h0_hi_b, _ = _eval_h_censored(
                hi[both], basis, theta_b, X_b, beta, S_b, gamma, o_b
            )
            log_p_b = _log_diff_ndtr(h_lo_b, h_hi_b, dist=dist)
            w_hi_b, w_lo_b = _pair_density_weights(h_lo_b, h_hi_b, log_p_b, dist)

            psi_lo = -_neg_score(h_lo_b, dist)
            psi_hi = -_neg_score(h_hi_b, dist)
            a = -psi_lo * w_lo_b - w_lo_b * w_lo_b
            c = psi_hi * w_hi_b - w_hi_b * w_hi_b
            b = w_hi_b * w_lo_b
            if w_b is not None:
                a = w_b * a
                b = w_b * b
                c = w_b * c
                w_lo_w = w_b * w_lo_b
                w_hi_w = w_b * w_hi_b
            else:
                w_lo_w = w_lo_b
                w_hi_w = w_hi_b

            # Shift block via _outer, with B̃ = f·B under scaling.
            if f_b is None:
                B_eff_lo = B_lo_b
                B_eff_hi = B_hi_b
            else:
                B_eff_lo = B_lo_b * f_b[:, None]
                B_eff_hi = B_hi_b * f_b[:, None]
            block = (
                _outer(B_eff_lo, X_b, B_eff_lo, X_b, a, p, q_d)
                + _outer(B_eff_lo, X_b, B_eff_hi, X_b, b, p, q_d)
                + _outer(B_eff_hi, X_b, B_eff_lo, X_b, b, p, q_d)
                + _outer(B_eff_hi, X_b, B_eff_hi, X_b, c, p, q_d)
            )
            H[: p + q_d, : p + q_d] -= block  # NLL = -log p

            if S_b is not None and f_b is not None:
                # 2D chain through γ Jacobian + bias term.
                # alpha_i := row of (a b; b c) · (h_0(lo), h_0(hi))
                alpha_lo = a * h0_lo_b + b * h0_hi_b
                alpha_hi = b * h0_lo_b + c * h0_hi_b
                # (θ_b, γ): NLL chain = -0.5·f²·X_s·(α_lo·B_lo + α_hi·B_hi)'
                #          NLL bias  = +0.5·f·X_s·(w_lo·B_lo - w_hi·B_hi)'
                coef_b_lo = 0.5 * f_b * (w_lo_w - f_b * alpha_lo)
                coef_b_hi = -0.5 * f_b * (w_hi_w + f_b * alpha_hi)
                H_bg = (B_lo_b * coef_b_lo[:, None]).T @ S_b + (
                    B_hi_b * coef_b_hi[:, None]
                ).T @ S_b
                H[:p, p + q_d :] += H_bg
                H[p + q_d :, :p] += H_bg.T
                # (β, γ): chain only = -0.5·f·X_d·X_s'·(α_lo + α_hi)
                if X_b is not None and q_d > 0:
                    coef_dg = -0.5 * f_b * (alpha_lo + alpha_hi)
                    H_dg = (X_b * coef_dg[:, None]).T @ S_b
                    H[p : p + q_d, p + q_d :] += H_dg
                    H[p + q_d :, p : p + q_d] += H_dg.T
                # (γ, γ): chain = -0.25·f²·(a·h0_lo² + 2b·h0_lo·h0_hi + c·h0_hi²)
                #         bias  = +0.25·f·(w_lo·h0_lo - w_hi·h0_hi)
                coef_gg = -0.25 * f_b * f_b * (
                    a * h0_lo_b * h0_lo_b
                    + 2.0 * b * h0_lo_b * h0_hi_b
                    + c * h0_hi_b * h0_hi_b
                ) + 0.25 * f_b * (w_lo_w * h0_lo_b - w_hi_w * h0_hi_b)
                H[p + q_d :, p + q_d :] += (S_b * coef_gg[:, None]).T @ S_b

        if only_hi.any():
            # Left-open: same Hessian form as _hess_left at h_hi.
            X_o = X_c[only_hi] if X_c is not None else None
            w_o = w_c[only_hi] if w_c is not None else None
            o_o = o_c[only_hi] if o_c is not None else None
            S_o = S_c[only_hi] if S_c is not None else None
            h_hi_o, B_hi_o, h0_hi_o, f_o = _eval_h_censored(
                hi[only_hi], basis, theta_b, X_o, beta, S_o, gamma, o_o
            )
            _logcdf = dist.logcdf
            log_mu = dist.logpdf(h_hi_o) - _logcdf(h_hi_o)
            mu = np.exp(np.minimum(log_mu, _LOG_FLOAT_MAX))
            psi = -_neg_score(h_hi_o, dist)
            w_chain = mu * (mu - psi)
            if w_o is not None:
                w_chain = w_o * w_chain
                mu_w = w_o * mu
            else:
                mu_w = mu
            B_eff = B_hi_o if f_o is None else B_hi_o * f_o[:, None]
            H[: p + q_d, : p + q_d] += _assemble_hessian(B_eff, w_chain, X_o, p, q_d)
            if S_o is not None and f_o is not None:
                _add_scaled_gamma_blocks_h(
                    H, B_hi_o, X_o, S_o, f_o, h0_hi_o, w_chain, -mu_w, p, q_d, q_s
                )

        if only_lo.any():
            # Right-open: same Hessian form as _hess_right at h_lo.
            X_o = X_c[only_lo] if X_c is not None else None
            w_o = w_c[only_lo] if w_c is not None else None
            o_o = o_c[only_lo] if o_c is not None else None
            S_o = S_c[only_lo] if S_c is not None else None
            h_lo_o, B_lo_o, h0_lo_o, f_o = _eval_h_censored(
                lo[only_lo], basis, theta_b, X_o, beta, S_o, gamma, o_o
            )
            log_hazard = dist.logpdf(h_lo_o) - dist.logsf(h_lo_o)
            lam = np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
            psi = -_neg_score(h_lo_o, dist)
            w_chain = lam * (psi + lam)
            if w_o is not None:
                w_chain = w_o * w_chain
                lam_w = w_o * lam
            else:
                lam_w = lam
            B_eff = B_lo_o if f_o is None else B_lo_o * f_o[:, None]
            H[: p + q_d, : p + q_d] += _assemble_hessian(B_eff, w_chain, X_o, p, q_d)
            if S_o is not None and f_o is not None:
                _add_scaled_gamma_blocks_h(
                    H, B_lo_o, X_o, S_o, f_o, h0_lo_o, w_chain, lam_w, p, q_d, q_s
                )

    return H


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _ll_interaction_none(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    basis: InteractionBasis,
    X: NDArray[np.float64],
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
) -> np.float64:
    """Log-likelihood for exact data with an InteractionBasis."""
    design, d_design = basis.evaluate_with_derivative(y, X)  # (n, p*q)
    h_raw = design @ theta
    if offset is not None:
        h_raw = h_raw + offset
    h = np.clip(h_raw, -_H_CLIP, _H_CLIP)
    hp = d_design @ theta

    with np.errstate(invalid="ignore", divide="ignore"):
        log_pdf_h = dist.logpdf(h)
        per_obs = log_pdf_h + np.log(hp)
        if weights is not None:
            return np.float64(np.dot(weights, per_obs))
        return np.float64(np.sum(per_obs))


def _ll_and_grad_interaction_none(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    basis: InteractionBasis,
    X: NDArray[np.float64],
    dist: DistOps = _NORM_OPS,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
) -> tuple[np.float64, NDArray[np.float64]]:
    """Combined ℓ and ∂(-ℓ)/∂θ for exact data with InteractionBasis."""
    design, d_design = basis.evaluate_with_derivative(y, X)  # (n, p*q)
    h_raw = design @ theta
    if offset is not None:
        h_raw = h_raw + offset
    h = np.clip(h_raw, -_H_CLIP, _H_CLIP)
    hp = d_design @ theta

    ns = _neg_score(h, dist)
    wns = ns if weights is None else weights * ns
    ihp = _inverse_hp(hp, weights)

    with np.errstate(invalid="ignore", divide="ignore"):
        log_pdf_h = dist.logpdf(h)
        if weights is not None:
            ll = np.float64(np.dot(weights, log_pdf_h + np.log(hp)))
        else:
            ll = np.float64(np.sum(log_pdf_h) + np.sum(np.log(hp)))
        grad = design.T @ wns - d_design.T @ ihp

    return ll, grad


def _log_likelihood_from_dist(
    theta: NDArray[np.float64],
    basis: BernsteinBasis | InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None,
    censoring: CensoringType,
    dist: DistOps,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> float:
    """Internal log-likelihood evaluator for a pre-resolved base distribution.

    ``scaling`` is only honoured on the exact / ``CensoringType.NONE`` branch
    in v0.4 (issue #70 tracer slice).  Other censoring types and the
    interaction-basis path raise :class:`NotImplementedError` when a
    non-``None`` scaling is supplied.
    """
    if scaling is not None and isinstance(basis, InteractionBasis):
        raise NotImplementedError(
            "scaling= is not supported with InteractionBasis in v0.4 "
            "(see docs/adr/0002-scaling-terms.md, Decision 2)."
        )
    if isinstance(basis, InteractionBasis):
        if X is None:
            raise ValueError(
                "InteractionBasis requires X to be provided for likelihood evaluation."
            )
        y_arr = (
            np.asarray(y, dtype=float).ravel() if isinstance(y, np.ndarray) else y.exact
        )
        result = _ll_interaction_none(
            y_arr, theta, basis, X, dist=dist, weights=weights, offset=offset
        )
        if not np.isfinite(result):
            raise InfeasibleParameterError(
                f"log_likelihood returned {result}.  Possible causes: theta "
                "violates monotonicity (h'(y) ≤ 0), observations outside basis "
                "support, or extreme h values despite clipping."
            )
        return float(result)

    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        result = _ll_none(
            y_arr,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    else:
        if censoring is CensoringType.NONE:
            result = _ll_none(
                y.exact,
                theta,
                basis,
                X,
                dist=dist,
                weights=weights,
                offset=offset,
                scaling=scaling,
            )
        elif censoring is CensoringType.RIGHT:
            result = _ll_right(
                y,
                theta,
                basis,
                X,
                dist=dist,
                weights=weights,
                offset=offset,
                scaling=scaling,
            )
        elif censoring is CensoringType.LEFT:
            result = _ll_left(
                y,
                theta,
                basis,
                X,
                dist=dist,
                weights=weights,
                offset=offset,
                scaling=scaling,
            )
        else:  # INTERVAL
            result = _ll_interval(
                y,
                theta,
                basis,
                X,
                dist=dist,
                weights=weights,
                offset=offset,
                scaling=scaling,
            )

        if _has_truncation(y):
            ctx = _build_truncation_context(y, theta, basis, X, offset)
            result = result + _truncation_ll(ctx, dist, weights)

    if not np.isfinite(result):
        raise InfeasibleParameterError(
            f"log_likelihood returned {result}.  Possible causes: theta "
            "violates monotonicity (h'(y) ≤ 0), observations outside basis "
            "support, or extreme h values despite clipping."
        )
    return float(result)


def _negative_log_likelihood_from_dist(
    theta: NDArray[np.float64],
    basis: BernsteinBasis | InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None,
    censoring: CensoringType,
    gradient: bool,
    dist: DistOps,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> float | tuple[float, NDArray[np.float64]]:
    """Internal NLL evaluator for a pre-resolved base distribution.

    ``scaling`` is only honoured on the exact / ``CensoringType.NONE`` branch
    in v0.4 (issue #70 tracer slice).
    """
    if scaling is not None and isinstance(basis, InteractionBasis):
        raise NotImplementedError(
            "scaling= is not supported with InteractionBasis in v0.4 "
            "(see docs/adr/0002-scaling-terms.md, Decision 2)."
        )
    if isinstance(basis, InteractionBasis):
        if X is None:
            raise ValueError(
                "InteractionBasis requires X to be provided for likelihood evaluation."
            )
        y_arr = (
            np.asarray(y, dtype=float).ravel() if isinstance(y, np.ndarray) else y.exact
        )
        if not gradient:
            result = _ll_interaction_none(
                y_arr, theta, basis, X, dist=dist, weights=weights, offset=offset
            )
            if not np.isfinite(result):
                raise InfeasibleParameterError(
                    f"log_likelihood returned {result}.  Possible causes: theta "
                    "violates monotonicity (h'(y) ≤ 0), observations outside basis "
                    "support, or extreme h values despite clipping."
                )
            return float(-result)
        ll, grad = _ll_and_grad_interaction_none(
            y_arr, theta, basis, X, dist=dist, weights=weights, offset=offset
        )
        if not np.isfinite(ll):
            raise InfeasibleParameterError(
                f"log_likelihood returned {ll}.  Possible causes: theta "
                "violates monotonicity (h'(y) ≤ 0), observations outside basis "
                "support, or extreme h values despite clipping."
            )
        return float(-ll), grad

    if not gradient:
        return -_log_likelihood_from_dist(
            theta,
            basis,
            y,
            X,
            censoring,
            dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )

    # Single pass: share basis.evaluate / basis.derivative and mask slicing
    # between the log-likelihood and its gradient.
    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        ll, grad = _ll_and_grad_none(
            y_arr,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    elif censoring is CensoringType.NONE:
        ll, grad = _ll_and_grad_none(
            y.exact,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    elif censoring is CensoringType.RIGHT:
        ll, grad = _ll_and_grad_right(
            y,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    elif censoring is CensoringType.LEFT:
        ll, grad = _ll_and_grad_left(
            y,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    else:
        ll, grad = _ll_and_grad_interval(
            y,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )

    if isinstance(y, CensoredData) and _has_truncation(y):
        p = basis.order + 1
        q = X.shape[1] if X is not None else 0
        ctx = _build_truncation_context(y, theta, basis, X, offset)
        ll = ll + _truncation_ll(ctx, dist, weights)
        grad = grad + _truncation_grad_nll(ctx, dist, weights, p, q)

    if not np.isfinite(ll):
        raise InfeasibleParameterError(
            f"log_likelihood returned {ll}.  Possible causes: theta "
            "violates monotonicity (h'(y) ≤ 0), observations outside basis "
            "support, or extreme h values despite clipping."
        )

    return float(-ll), grad


def log_likelihood(
    theta: NDArray[np.float64],
    basis: BernsteinBasis | InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    base_distribution: BaseDistribution = "normal",
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
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
    weights:
        Per-observation weights of shape ``(n,)``.  Non-negative, finite.
        ``None`` is equivalent to unit weights.
    offset:
        Per-observation fixed linear predictor added to ``h(y|x)`` before
        distribution calls.  Shape ``(n,)``, finite.  Not optimised.
        ``None`` is equivalent to zero offset.

    Returns
    -------
    float  (log-likelihood value)

    Raises
    ------
    InfeasibleParameterError
        If the result is ``-inf`` or ``NaN``.
    ValueError
        From :func:`_get_dist` if ``base_distribution`` is not supported, or
        from :func:`_validate_weights_offset` if ``weights``/``offset`` are
        invalid.
    """
    dist = _get_dist(base_distribution)
    n = y.n if isinstance(y, CensoredData) else len(np.asarray(y).ravel())
    weights, offset = _validate_weights_offset(weights, offset, n)
    return _log_likelihood_from_dist(
        theta,
        basis,
        y,
        X,
        censoring,
        dist,
        weights=weights,
        offset=offset,
        scaling=scaling,
    )


def negative_log_likelihood(
    theta: NDArray[np.float64],
    basis: BernsteinBasis | InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    gradient: bool = False,
    base_distribution: BaseDistribution = "normal",
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
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
    weights:
        Per-observation weights. See :func:`log_likelihood`.
    offset:
        Per-observation offset. See :func:`log_likelihood`.

    Returns
    -------
    float  when ``gradient=False``
    (float, NDArray)  when ``gradient=True``
    """
    dist = _get_dist(base_distribution)
    n = y.n if isinstance(y, CensoredData) else len(np.asarray(y).ravel())
    weights, offset = _validate_weights_offset(weights, offset, n)
    return _negative_log_likelihood_from_dist(
        theta,
        basis,
        y,
        X,
        censoring,
        gradient,
        dist,
        weights=weights,
        offset=offset,
        scaling=scaling,
    )


def _hessian_interaction_fd(
    theta: NDArray[np.float64],
    basis: InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64],
    dist: DistOps,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    h_fd: float = 1e-5,
) -> NDArray[np.float64]:
    """Finite-difference Hessian of NLL for InteractionBasis models."""
    m = theta.size
    y_arr = np.asarray(y, dtype=float).ravel() if isinstance(y, np.ndarray) else y.exact

    def nll(t: NDArray[np.float64]) -> float:
        return float(
            -_ll_interaction_none(
                y_arr, t, basis, X, dist=dist, weights=weights, offset=offset
            )
        )

    H = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        ei = np.zeros(m)
        ei[i] = h_fd
        for j in range(i, m):
            ej = np.zeros(m)
            ej[j] = h_fd
            val = (
                nll(theta + ei + ej)
                - nll(theta + ei - ej)
                - nll(theta - ei + ej)
                + nll(theta - ei - ej)
            ) / (4 * h_fd * h_fd)
            H[i, j] = val
            H[j, i] = val
    return H


def hessian(
    theta: NDArray[np.float64],
    basis: BernsteinBasis | InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    base_distribution: BaseDistribution = "normal",
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
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
    weights:
        Per-observation weights. See :func:`log_likelihood`.
    offset:
        Per-observation offset. See :func:`log_likelihood`.

    Returns
    -------
    NDArray[np.float64]
        Symmetric Hessian of ``-ℓ``.

    Raises
    ------
    InfeasibleParameterError
        If any entry of the Hessian is non-finite.
    ValueError
        If ``base_distribution`` is not supported.
    """
    dist = _get_dist(base_distribution)
    n = y.n if isinstance(y, CensoredData) else len(np.asarray(y).ravel())
    weights, offset = _validate_weights_offset(weights, offset, n)

    if scaling is not None and isinstance(basis, InteractionBasis):
        raise NotImplementedError(
            "scaling= is not supported with InteractionBasis "
            "(see docs/adr/0002-scaling-terms.md, Decision 2)."
        )

    if isinstance(basis, InteractionBasis):
        if X is None:
            raise ValueError("InteractionBasis requires X for hessian computation.")
        result = _hessian_interaction_fd(
            theta, basis, y, X, dist=dist, weights=weights, offset=offset
        )
        if not np.all(np.isfinite(result)):
            raise InfeasibleParameterError(
                "hessian() produced non-finite entries for InteractionBasis."
            )
        return result

    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        result = _hess_none(
            y_arr,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    elif censoring is CensoringType.NONE:
        result = _hess_none(
            y.exact,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    elif censoring is CensoringType.RIGHT:
        result = _hess_right(
            y,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    elif censoring is CensoringType.LEFT:
        result = _hess_left(
            y,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    else:
        result = _hess_interval(
            y,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )

    if isinstance(y, CensoredData) and _has_truncation(y):
        p = basis.order + 1
        q = X.shape[1] if X is not None else 0
        ctx = _build_truncation_context(y, theta, basis, X, offset)
        result = result + _truncation_hess_nll(ctx, dist, weights, p, q)

    if not np.all(np.isfinite(result)):
        raise InfeasibleParameterError(
            "hessian() produced non-finite entries.  Possible causes: theta "
            "violates monotonicity (h'(y) ≤ 0), observations outside basis "
            "support, or extreme h values despite clipping."
        )
    return result


def _score_matrix_interaction(
    theta: NDArray[np.float64],
    basis: InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64],
    dist: DistOps,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Per-observation score matrix for exact data with InteractionBasis."""
    y_arr = np.asarray(y, dtype=float).ravel() if isinstance(y, np.ndarray) else y.exact
    design, d_design = basis.evaluate_with_derivative(y_arr, X)  # (n, p*q)
    h_raw = design @ theta
    if offset is not None:
        h_raw = h_raw + offset
    h = np.clip(h_raw, -_H_CLIP, _H_CLIP)
    hp = d_design @ theta

    ns = _neg_score(h, dist)  # (n,) — negative score of log-density
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_hp = 1.0 / hp  # (n,)
    # ∂ℓ_i/∂θ = -ns_i * design_i + inv_hp_i * d_design_i
    score = -ns[:, None] * design + inv_hp[:, None] * d_design  # (n, p*q)
    if weights is not None:
        score = weights[:, None] * score
    return score


def score_matrix(
    theta: NDArray[np.float64],
    basis: BernsteinBasis | InteractionBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    base_distribution: BaseDistribution = "normal",
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Per-observation score contributions ``∂ℓ_i/∂θ``.

    Returns the ``(n, p+q)`` matrix of per-observation gradients of the
    *positive* log-likelihood, often referred to as ``estfun`` in the R
    ``sandwich`` package.  When ``weights`` are provided, row ``i`` is
    ``w_i · ∂ℓ_i/∂θ`` so that ``score_matrix(...).sum(axis=0)`` equals
    the full weighted log-likelihood gradient.

    Parameters
    ----------
    theta, basis, y, X, censoring, base_distribution:
        Same as :func:`log_likelihood`.
    weights:
        Per-observation weights. See :func:`log_likelihood`.
    offset:
        Per-observation offset. See :func:`log_likelihood`.

    Returns
    -------
    NDArray[np.float64]
        Per-observation score matrix of shape ``(n, p+q)``.

    Raises
    ------
    InfeasibleParameterError
        If any entry of the score matrix is non-finite.
    ValueError
        If ``base_distribution`` is not supported.
    """
    dist = _get_dist(base_distribution)
    n = y.n if isinstance(y, CensoredData) else len(np.asarray(y).ravel())
    weights, offset = _validate_weights_offset(weights, offset, n)

    if scaling is not None and isinstance(basis, InteractionBasis):
        raise NotImplementedError(
            "scaling= is not supported with InteractionBasis "
            "(see docs/adr/0002-scaling-terms.md, Decision 2)."
        )

    if isinstance(basis, InteractionBasis):
        if X is None:
            raise ValueError(
                "InteractionBasis requires X for score_matrix computation."
            )
        result = _score_matrix_interaction(
            theta, basis, y, X, dist=dist, weights=weights, offset=offset
        )
        if not np.all(np.isfinite(result)):
            raise InfeasibleParameterError(
                "score_matrix() produced non-finite entries for InteractionBasis."
            )
        return result

    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        result = _scores_none(
            y_arr,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    elif censoring is CensoringType.NONE:
        result = _scores_none(
            y.exact,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    elif censoring is CensoringType.RIGHT:
        result = _scores_right(
            y,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    elif censoring is CensoringType.LEFT:
        result = _scores_left(
            y,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
    else:
        result = _scores_interval(
            y,
            theta,
            basis,
            X,
            dist=dist,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )

    if isinstance(y, CensoredData) and _has_truncation(y):
        p = basis.order + 1
        q = X.shape[1] if X is not None else 0
        ctx = _build_truncation_context(y, theta, basis, X, offset)
        result = result - _truncation_scores(ctx, dist, weights, n, p, q)

    if not np.all(np.isfinite(result)):
        raise InfeasibleParameterError(
            "score_matrix() produced non-finite entries.  Possible causes: "
            "theta violates monotonicity (h'(y) ≤ 0), observations outside "
            "basis support, or extreme h values despite clipping."
        )
    return result


def intercept_score(
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    y: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    base_distribution: BaseDistribution = "normal",
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scaling: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Per-observation score w.r.t. an artificial intercept on ``h(y|x)``.

    For each observation ``i``, returns ``∂ℓ_i/∂α`` evaluated at ``α = 0``,
    where ``α`` is a hypothetical intercept added to the transformation:
    ``h̃(y|x) = h(y|x) + α``.  When ``weights`` are provided, row ``i`` is
    multiplied by ``w_i`` (same R ``sandwich`` convention as
    :func:`score_matrix`).

    The closed forms by censoring type are:

    * **exact** ``y_i``:           ``ψ(h_i) = d log f(h_i)/dh``
    * **right-censored**:           ``-f(h_i)/S(h_i)`` (negative hazard)
    * **left-censored**:            ``f(h_i)/F(h_i)`` (inverse Mills ratio)
    * **interval-censored** [a,b]:  ``(f(h_b) - f(h_a)) / (F(h_b) - F(h_a))``

    Under scaling (``scaling`` not ``None``) the closed forms are unchanged
    — the artificial intercept ``α`` is added to the *final* h (post-scaling
    and post-shift), so ``∂h̃/∂α = 1`` for every row regardless of γ.  γ
    enters only through the value at which the score is evaluated:
    ``h_i = h_0(y_i) · exp(0.5·x_s_i·γ) + x_d_i·β + offset_i``.

    Parameters
    ----------
    theta, basis, y, X, censoring, base_distribution:
        Same as :func:`score_matrix`.
    weights:
        Per-observation weights. See :func:`log_likelihood`.
    offset:
        Per-observation offset. See :func:`log_likelihood`.
    scaling : NDArray[np.float64] | None
        Scaling-design matrix ``(n, q_s)``.  When provided, ``theta`` is
        split as ``[θ_b | β | γ]`` and h is evaluated at the heteroskedastic
        value ``h_0(y) · exp(0.5·X_s·γ) + Xβ`` per ADR 0002.

    Returns
    -------
    NDArray[np.float64]
        Vector of length ``n``.

    Raises
    ------
    InfeasibleParameterError
        If any entry is non-finite (typically because ``theta`` violates
        monotonicity).
    ValueError
        If ``base_distribution`` is not supported.
    """
    dist = _get_dist(base_distribution)
    n = y.n if isinstance(y, CensoredData) else len(np.asarray(y).ravel())
    weights, offset = _validate_weights_offset(weights, offset, n)
    p = basis.order + 1
    if scaling is None:
        theta_b, beta = _split_theta(theta, p, X)
        scale_factor = None
    else:
        q_d = 0 if X is None else X.shape[1]
        q_s = scaling.shape[1]
        theta_b, beta, gamma = _split_theta_scaled(theta, p, q_d, q_s)
        # ``gamma`` is non-None whenever ``scaling is not None`` (q_s > 0).
        gamma_arr = cast(NDArray[np.float64], gamma)
        # Scaling factor f_i = exp(0.5 · x_s_i · γ).  ``intercept_score``
        # evaluates the per-row score at the scaled h, so the only effect
        # of γ on the score is via this multiplicative factor on ``h_0``.
        scale_factor = np.exp(0.5 * scaling @ gamma_arr)

    if isinstance(y, np.ndarray):
        y_arr = np.asarray(y, dtype=float).ravel()
        result = _intercept_score_exact(
            y_arr,
            theta_b,
            basis,
            X,
            beta,
            dist,
            weights=weights,
            offset=offset,
            scale_factor=scale_factor,
        )
    elif censoring is CensoringType.NONE:
        result = _intercept_score_exact(
            y.exact,
            theta_b,
            basis,
            X,
            beta,
            dist,
            weights=weights,
            offset=offset,
            scale_factor=scale_factor,
        )
    elif censoring is CensoringType.RIGHT:
        result = _intercept_score_right(
            y,
            theta_b,
            basis,
            X,
            beta,
            dist,
            weights=weights,
            offset=offset,
            scale_factor=scale_factor,
        )
    elif censoring is CensoringType.LEFT:
        result = _intercept_score_left(
            y,
            theta_b,
            basis,
            X,
            beta,
            dist,
            weights=weights,
            offset=offset,
            scale_factor=scale_factor,
        )
    else:
        result = _intercept_score_interval(
            y,
            theta_b,
            basis,
            X,
            beta,
            dist,
            weights=weights,
            offset=offset,
            scale_factor=scale_factor,
        )

    if isinstance(y, CensoredData) and _has_truncation(y):
        ctx = _build_truncation_context(y, theta, basis, X, offset)
        result = result - _truncation_intercept_score(ctx, dist, weights, n)

    if not np.all(np.isfinite(result)):
        raise InfeasibleParameterError(
            "intercept_score() produced non-finite entries.  Possible causes: "
            "theta violates monotonicity (h'(y) ≤ 0), observations outside "
            "basis support, or extreme h values despite clipping."
        )
    return result


def _intercept_score_exact(
    y: NDArray[np.float64],
    theta_b: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    beta: NDArray[np.float64] | None,
    dist: DistOps,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scale_factor: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    B = basis.evaluate(y)
    h0 = B @ theta_b
    if scale_factor is not None:
        h0 = h0 * scale_factor
    h_raw = _shift(h0, X, beta)
    if offset is not None:
        h_raw = h_raw + offset
    h = np.clip(h_raw, -_H_CLIP, _H_CLIP)
    result = -_neg_score(h, dist)
    if weights is not None:
        result = weights * result
    return result


def _intercept_score_right(
    cd: CensoredData,
    theta_b: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    beta: NDArray[np.float64] | None,
    dist: DistOps,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scale_factor: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    out = np.zeros(cd.n, dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        f_e = scale_factor[mask_e] if scale_factor is not None else None
        out[mask_e] = _intercept_score_exact(
            cd.exact[mask_e],
            theta_b,
            basis,
            X_e,
            beta,
            dist,
            weights=w_e,
            offset=o_e,
            scale_factor=f_e,
        )

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        f_c = scale_factor[mask_c] if scale_factor is not None else None
        B_c = basis.evaluate(y_c)
        h0_c = B_c @ theta_b
        if f_c is not None:
            h0_c = h0_c * f_c
        h_raw_c = _shift(h0_c, X_c, beta)
        if o_c is not None:
            h_raw_c = h_raw_c + o_c
        h_c = np.clip(h_raw_c, -_H_CLIP, _H_CLIP)
        log_hazard = dist.logpdf(h_c) - dist.logsf(h_c)
        vals = -np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
        if w_c is not None:
            vals = w_c * vals
        out[mask_c] = vals

    return out


def _intercept_score_left(
    cd: CensoredData,
    theta_b: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    beta: NDArray[np.float64] | None,
    dist: DistOps,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scale_factor: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    out = np.zeros(cd.n, dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        f_e = scale_factor[mask_e] if scale_factor is not None else None
        out[mask_e] = _intercept_score_exact(
            cd.exact[mask_e],
            theta_b,
            basis,
            X_e,
            beta,
            dist,
            weights=w_e,
            offset=o_e,
            scale_factor=f_e,
        )

    mask_c = cd.is_left_censored_mask
    if mask_c.any():
        y_c = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        f_c = scale_factor[mask_c] if scale_factor is not None else None
        B_c = basis.evaluate(y_c)
        h0_c = B_c @ theta_b
        if f_c is not None:
            h0_c = h0_c * f_c
        h_raw_c = _shift(h0_c, X_c, beta)
        if o_c is not None:
            h_raw_c = h_raw_c + o_c
        h_c = np.clip(h_raw_c, -_H_CLIP, _H_CLIP)
        _logcdf = dist.logcdf
        log_inv_mills = dist.logpdf(h_c) - _logcdf(h_c)
        vals = np.exp(np.minimum(log_inv_mills, _LOG_FLOAT_MAX))
        if w_c is not None:
            vals = w_c * vals
        out[mask_c] = vals

    return out


def _intercept_score_interval(
    cd: CensoredData,
    theta_b: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    beta: NDArray[np.float64] | None,
    dist: DistOps,
    weights: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
    scale_factor: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    out = np.zeros(cd.n, dtype=np.float64)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        X_e = X[mask_e] if X is not None else None
        w_e = weights[mask_e] if weights is not None else None
        o_e = offset[mask_e] if offset is not None else None
        f_e = scale_factor[mask_e] if scale_factor is not None else None
        out[mask_e] = _intercept_score_exact(
            cd.exact[mask_e],
            theta_b,
            basis,
            X_e,
            beta,
            dist,
            weights=w_e,
            offset=o_e,
            scale_factor=f_e,
        )

    mask_c = ~cd.is_exact_mask
    if mask_c.any():
        idx_c = np.flatnonzero(mask_c)
        lo = cd.lower[mask_c]
        hi = cd.upper[mask_c]
        X_c = X[mask_c] if X is not None else None
        w_c = weights[mask_c] if weights is not None else None
        o_c = offset[mask_c] if offset is not None else None
        f_c = scale_factor[mask_c] if scale_factor is not None else None
        fin_lo = np.isfinite(lo)
        fin_hi = np.isfinite(hi)
        both = fin_lo & fin_hi
        only_hi = ~fin_lo & fin_hi
        only_lo = fin_lo & ~fin_hi

        if both.any():
            rows = idx_c[both]
            B_lo_b = basis.evaluate(lo[both])
            B_hi_b = basis.evaluate(hi[both])
            X_b = X_c[both] if X_c is not None else None
            f_b = f_c[both] if f_c is not None else None
            shift_b = (X_b @ beta) if (X_b is not None and beta is not None) else 0.0
            if o_c is not None:
                shift_b = shift_b + o_c[both]
            h0_lo = B_lo_b @ theta_b
            h0_hi = B_hi_b @ theta_b
            if f_b is not None:
                h0_lo = h0_lo * f_b
                h0_hi = h0_hi * f_b
            h_lo_b = np.clip(h0_lo + shift_b, -_H_CLIP, _H_CLIP)
            h_hi_b = np.clip(h0_hi + shift_b, -_H_CLIP, _H_CLIP)
            log_p_b = _log_diff_ndtr(h_lo_b, h_hi_b, dist=dist)
            w_hi_b, w_lo_b = _pair_density_weights(h_lo_b, h_hi_b, log_p_b, dist)
            vals = w_hi_b - w_lo_b
            if w_c is not None:
                vals = w_c[both] * vals
            out[rows] = vals

        if only_hi.any():
            rows = idx_c[only_hi]
            B_hi_o = basis.evaluate(hi[only_hi])
            X_o = X_c[only_hi] if X_c is not None else None
            f_o = f_c[only_hi] if f_c is not None else None
            shift_o = (X_o @ beta) if (X_o is not None and beta is not None) else 0.0
            if o_c is not None:
                shift_o = shift_o + o_c[only_hi]
            h0_hi_o = B_hi_o @ theta_b
            if f_o is not None:
                h0_hi_o = h0_hi_o * f_o
            h_hi_o = np.clip(h0_hi_o + shift_o, -_H_CLIP, _H_CLIP)
            _logcdf = dist.logcdf
            log_inv_mills = dist.logpdf(h_hi_o) - _logcdf(h_hi_o)
            vals = np.exp(np.minimum(log_inv_mills, _LOG_FLOAT_MAX))
            if w_c is not None:
                vals = w_c[only_hi] * vals
            out[rows] = vals

        if only_lo.any():
            rows = idx_c[only_lo]
            B_lo_o = basis.evaluate(lo[only_lo])
            X_o = X_c[only_lo] if X_c is not None else None
            f_o = f_c[only_lo] if f_c is not None else None
            shift_o = (X_o @ beta) if (X_o is not None and beta is not None) else 0.0
            if o_c is not None:
                shift_o = shift_o + o_c[only_lo]
            h0_lo_o = B_lo_o @ theta_b
            if f_o is not None:
                h0_lo_o = h0_lo_o * f_o
            h_lo_o = np.clip(h0_lo_o + shift_o, -_H_CLIP, _H_CLIP)
            log_hazard = dist.logpdf(h_lo_o) - dist.logsf(h_lo_o)
            vals = -np.exp(np.minimum(log_hazard, _LOG_FLOAT_MAX))
            if w_c is not None:
                vals = w_c[only_lo] * vals
            out[rows] = vals

    return out
