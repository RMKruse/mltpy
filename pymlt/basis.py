"""Bernstein polynomial basis and analytical derivatives for transformation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import betainc, comb

# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _bernstein_matrix(t: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Evaluate the (n, k+1) Bernstein basis matrix at normalised t ∈ [0, 1].

    B[j, i] = C(k, i) · t[j]^i · (1 − t[j])^(k − i)

    Uses ``scipy.special.comb`` (exact=False) for vectorised float binomials.
    Fully vectorised — no Python loop over observations.

    Parameters
    ----------
    t:
        Normalised evaluation points, shape (n,).  The caller is responsible
        for validating that all entries lie in ``[0, 1]``.
    k:
        Polynomial degree.  Returns k+1 basis functions.

    Returns
    -------
    NDArray of shape (n, k+1).
    """
    t = np.asarray(t, dtype=float)
    i = np.arange(k + 1, dtype=float)  # shape (k+1,)
    binom = comb(k, i, exact=False)  # shape (k+1,)
    # Broadcasting: t[:, None] × i[None, :] → (n, k+1)
    return cast(
        NDArray[np.float64], binom * t[:, None] ** i * (1.0 - t[:, None]) ** (k - i)
    )


def _normalize_and_validate_support(
    y: NDArray[np.float64], support: tuple[float, float]
) -> NDArray[np.float64]:
    """Map ``y`` from ``support`` to ``[0, 1]`` after validating support."""
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
    if y_arr.size == 0:
        return y_arr

    a, b = support
    y_min = float(np.min(y_arr))
    y_max = float(np.max(y_arr))
    if y_min < a or y_max > b:
        raise ValueError(
            f"y contains values outside support [{a}, {b}]. "
            f"Adjust BernsteinBasis(support=...) accordingly. "
            f"(min={y_min:.4g}, max={y_max:.4g})"
        )
    return (y_arr - a) / (b - a)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


@dataclass
class BernsteinBasis:
    """Bernstein polynomial basis of degree `order` on a compact support.

    Coefficient ordering: ascending from degree 0 to `order` — identical to
    R's ``basefun::Bernstein_basis``.  This differs from ``numpy.poly1d``,
    which stores coefficients in *descending* degree order.

    Parameters
    ----------
    order:
        Polynomial degree k.  The basis has k+1 functions.
    support:
        Closed interval (a, b) with a < b.  Maps y → t = (y − a) / (b − a).
    """

    order: int
    support: tuple[float, float]

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        if not (np.isfinite(self.support[0]) and np.isfinite(self.support[1])):
            raise ValueError(f"support bounds must be finite, got {self.support}")
        if self.support[0] >= self.support[1]:
            raise ValueError(f"support must satisfy a < b, got {self.support}")

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def evaluate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Bernstein design matrix at observations y.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in the closed interval
            ``[support[0], support[1]]``.

        Returns
        -------
        NDArray of shape (n, order+1).  Row i is [B_{0,k}(y_i), …, B_{k,k}(y_i)].

        Raises
        ------
        ValueError
            If any observation lies outside ``support``.
        """
        t = _normalize_and_validate_support(y, self.support)
        return _bernstein_matrix(t, self.order)

    def derivative(self, y: NDArray[np.float64], order: int = 1) -> NDArray[np.float64]:
        """Analytical derivative of the Bernstein design matrix.

        Uses the recurrence relation — no finite differences.

        First derivative (order=1):
            dB_{i,k}/dy = k/(b−a) · [B_{i−1,k−1}(t) − B_{i,k−1}(t)]

        Second derivative (order=2):
            d²B_{i,k}/dy² = k(k−1)/(b−a)² ·
                            [B_{i−2,k−2}(t) − 2·B_{i−1,k−2}(t) + B_{i,k−2}(t)]

        Boundary terms (B_{j,·} with j < 0 or j > k) are treated as zero.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in the closed interval
            ``[support[0], support[1]]``.
        order:
            Derivative order: 1 (default) or 2.  Order 0 is intentionally not
            supported; use ``evaluate(y)`` instead.

        Returns
        -------
        NDArray of shape (n, self.order+1).

        Raises
        ------
        ValueError
            If ``order`` is not 1 or 2 (order 0 is not supported; call
            ``evaluate(y)`` directly), or if any observation lies outside
            ``support``.
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")

        k = self.order
        a, b = self.support
        t = _normalize_and_validate_support(y, self.support)
        n = len(t)

        if order == 1:
            if k == 0:
                return np.zeros((n, 1))
            # B_{k-1}(t): shape (n, k)
            B_low = _bernstein_matrix(t, k - 1)
            # Zero-pad to (n, k+2): B_{-1,k-1} = 0, B_{k,k-1} = 0
            B_pad = np.pad(B_low, ((0, 0), (1, 1)))
            # result[:, i] = k * (B_pad[:, i] - B_pad[:, i+1])
            result = k * (B_pad[:, :-1] - B_pad[:, 1:])
            return result / (b - a)

        else:  # order == 2
            if k <= 1:
                return np.zeros((n, k + 1))
            # B_{k-2}(t): shape (n, k-1)
            B_low = _bernstein_matrix(t, k - 2)
            # Zero-pad to (n, k+3): two zeros on each side
            B_pad = np.pad(B_low, ((0, 0), (2, 2)))
            # result[:, i] = k(k-1) * (B_pad[:,i] - 2*B_pad[:,i+1] + B_pad[:,i+2])
            result = k * (k - 1) * (B_pad[:, :-2] - 2 * B_pad[:, 1:-1] + B_pad[:, 2:])
            return result / (b - a) ** 2

    def evaluate_with_derivative(
        self, y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Bernstein design matrix and its first derivative in one pass.

        Normalises and validates ``y`` once, then returns both the evaluation
        matrix (degree k) and the first-derivative matrix (degree k−1
        recurrence).  Equivalent to calling ``evaluate(y)`` followed by
        ``derivative(y, order=1)`` but avoids the redundant support scan.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``[support[0], support[1]]``.

        Returns
        -------
        B : NDArray of shape (n, order+1)
            Same as ``evaluate(y)``.
        dB : NDArray of shape (n, order+1)
            Same as ``derivative(y, order=1)``.
        """
        k = self.order
        a, b = self.support
        t = _normalize_and_validate_support(y, self.support)
        n = len(t)
        B = _bernstein_matrix(t, k)
        if k == 0:
            return B, np.zeros((n, 1))
        B_low = _bernstein_matrix(t, k - 1)
        B_pad = np.pad(B_low, ((0, 0), (1, 1)))
        dB = k * (B_pad[:, :-1] - B_pad[:, 1:]) / (b - a)
        return B, dB

    def integrate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Running integral of each basis function from a to y.

        Uses the regularised incomplete beta function:

            ∫_a^y B_{i,k}(s) ds = (b−a)/(k+1) · I_t(i+1, k−i+1)

        where t = (y−a)/(b−a) and I is ``scipy.special.betainc``.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in the closed interval
            ``[support[0], support[1]]``.

        Returns
        -------
        NDArray of shape (n, order+1).

        Raises
        ------
        ValueError
            If any observation lies outside ``support``.
        """
        k = self.order
        a, b = self.support
        t = _normalize_and_validate_support(y, self.support)

        i = np.arange(k + 1)  # shape (k+1,)
        a_param = (i + 1).astype(float)  # shape (k+1,)
        b_param = (k - i + 1).astype(float)  # shape (k+1,)

        # betainc is vectorised over all arguments via broadcasting
        # t[:, None]: (n, 1),  a_param/b_param: (k+1,)  →  result: (n, k+1)
        result = betainc(a_param[None, :], b_param[None, :], t[:, None])
        return cast(NDArray[np.float64], result * (b - a) / (k + 1))


# ---------------------------------------------------------------------------
# Log-scale Bernstein basis (for parametric survival on positive outcomes)
# ---------------------------------------------------------------------------


@dataclass
class LogBernsteinBasis:
    """Bernstein polynomial basis evaluated at log(y) for log-scale survival models.

    Models the transformation h(y) = B_k(log(y)) · θ, where B_k is a standard
    Bernstein basis on (log a, log b).  This parameterises Survreg models:
    Weibull (min_extreme_value), log-normal (normal), and log-logistic (logistic).

    The derivative on the original scale follows the chain rule:

        dh/dy = (1/y) · dB_k(log y)/d(log y) · θ

    Parameters
    ----------
    order:
        Polynomial degree k.  The basis has k+1 functions.
    support:
        Closed interval (a, b) with 0 < a < b on the *original* positive scale.
        Internally maps y → t = (log y − log a) / (log b − log a).
    """

    order: int
    support: tuple[float, float]

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        a, b = self.support
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f"support bounds must be finite, got {self.support}")
        if a <= 0.0:
            raise ValueError(
                f"support lower bound must be strictly positive for "
                f"LogBernsteinBasis, got a={a}"
            )
        if a >= b:
            raise ValueError(f"support must satisfy a < b, got {self.support}")
        self._log_basis = BernsteinBasis(
            order=self.order, support=(float(np.log(a)), float(np.log(b)))
        )

    # ------------------------------------------------------------------
    # Core methods (duck-type BernsteinBasis)
    # ------------------------------------------------------------------

    def evaluate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate B_k(log y) at each observation.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in (support[0], support[1]).

        Returns
        -------
        NDArray of shape (n, order+1).

        Raises
        ------
        ValueError
            If any observation lies outside ``support``.
        """
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
        a, b = self.support
        if y_arr.size > 0 and (float(y_arr.min()) < a or float(y_arr.max()) > b):
            raise ValueError(
                f"y contains values outside support [{a}, {b}]. "
                f"(min={float(y_arr.min()):.4g}, max={float(y_arr.max()):.4g})"
            )
        return self._log_basis.evaluate(np.log(y_arr))

    def derivative(self, y: NDArray[np.float64], order: int = 1) -> NDArray[np.float64]:
        """Analytical derivative d/dy [B_k(log y)] = (1/y) · dB_k/d(log y).

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``support``.
        order:
            Derivative order: 1 (default).  Order 2 is not supported.

        Returns
        -------
        NDArray of shape (n, self.order+1).

        Raises
        ------
        ValueError
            If ``order`` is not 1, or any observation lies outside ``support``.
        """
        if order != 1:
            raise ValueError(
                f"LogBernsteinBasis.derivative only supports order=1, got {order}"
            )
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        a, b = self.support
        if y_arr.size > 0 and (float(y_arr.min()) < a or float(y_arr.max()) > b):
            raise ValueError(
                f"y contains values outside support [{a}, {b}]. "
                f"(min={float(y_arr.min()):.4g}, max={float(y_arr.max()):.4g})"
            )
        dB_log = self._log_basis.derivative(np.log(y_arr), order=1)
        return cast(NDArray[np.float64], dB_log / y_arr[:, None])

    def evaluate_with_derivative(
        self, y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return B_k(log y) and d/dy B_k(log y) in one pass.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``support``.

        Returns
        -------
        B : NDArray of shape (n, order+1)
        dB : NDArray of shape (n, order+1) — derivative w.r.t. y (includes 1/y)
        """
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        a, b = self.support
        if y_arr.size > 0 and (float(y_arr.min()) < a or float(y_arr.max()) > b):
            raise ValueError(
                f"y contains values outside support [{a}, {b}]. "
                f"(min={float(y_arr.min()):.4g}, max={float(y_arr.max()):.4g})"
            )
        B, dB_log = self._log_basis.evaluate_with_derivative(np.log(y_arr))
        return B, cast(NDArray[np.float64], dB_log / y_arr[:, None])

    def integrate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Not implemented for LogBernsteinBasis."""
        raise NotImplementedError(
            "LogBernsteinBasis.integrate() is not implemented. "
            "Numerical integration should be performed on the log scale."
        )


# ---------------------------------------------------------------------------
# Ordinal cutpoint basis
# ---------------------------------------------------------------------------


@dataclass
class OrdinalBasis:
    """Degenerate "one-hot cutpoint" basis used by ordinal regression (Polr).

    For ``K`` ordered levels the transformation has ``K-1`` cutpoints
    ``θ = (θ_1, ..., θ_{K-1})``.  Given an integer cut position
    ``k ∈ {1, ..., K-1}`` (representing the boundary between level ``k`` and
    level ``k+1``), the basis returns the one-hot row ``e_k`` of length
    ``K-1``, so ``B(k) @ θ = θ_k`` exactly — the basis *selects* the cutpoint.
    Combined with :class:`~pymlt.constraints.MonotonicityConstraint` of
    ``n_params = K-1`` this yields ``θ_1 ≤ ... ≤ θ_{K-1}``.

    The class duck-types :class:`BernsteinBasis` (``order``, ``support``,
    ``evaluate``, ``derivative``, ``integrate``) so that it drops into the
    existing likelihood / optimisation code paths unchanged.

    Parameters
    ----------
    K:
        Number of ordered levels.  Must satisfy ``K >= 2``.

    Notes
    -----
    The transformation ``h(y) = B(y) @ θ`` is a step function across cut
    positions, so its analytical derivative w.r.t. ``y`` is zero almost
    everywhere.  :meth:`derivative` returns zero accordingly; the exact-
    likelihood paths in :mod:`pymlt.likelihood` would log(0) but are never
    invoked for ordinal data — every observation is interval-censored
    (or one-sided open) and routes through the censored likelihoods.
    """

    K: int  # noqa: N815 — match standard ordinal-regression notation

    def __post_init__(self) -> None:
        if self.K < 2:
            raise ValueError(f"K must be >= 2, got {self.K}")

    @property
    def order(self) -> int:
        """Polynomial-degree analogue: ``K - 2`` so that ``order + 1 == K - 1``."""
        return self.K - 2

    @property
    def support(self) -> tuple[float, float]:
        """Wide enough to bracket integer cut positions ``1..K-1``.

        Rows that resolve to ``±∞`` bypass the support check in
        :meth:`pymlt.model.ConditionalTransformationModel._validate_input`.
        """
        return (0.0, float(self.K))

    # ------------------------------------------------------------------
    # Core methods (duck-type :class:`BernsteinBasis`)
    # ------------------------------------------------------------------

    def evaluate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return one-hot rows for integer cut positions in ``{1, ..., K-1}``.

        Parameters
        ----------
        y:
            Observations, shape ``(n,)``.  Each value must be an integer in
            ``{1, ..., K-1}`` (the synthetic cut positions emitted by
            :meth:`pymlt.variables.OrderedVariable.from_labels`).

        Returns
        -------
        NDArray of shape ``(n, K-1)`` with row ``i`` equal to ``e_{int(y[i])-1}``.

        Raises
        ------
        ValueError
            If any element of ``y`` is not an integer in ``{1, ..., K-1}``.
        """
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
        n = y_arr.size
        m = self.K - 1
        if n == 0:
            return np.zeros((0, m), dtype=np.float64)
        codes = y_arr.astype(np.intp)
        if not np.all(codes == y_arr):
            raise ValueError(
                "OrdinalBasis.evaluate expects integer cut positions; "
                "received non-integer values."
            )
        if codes.min() < 1 or codes.max() > m:
            raise ValueError(
                f"OrdinalBasis cut positions must be in [1, {m}], got "
                f"min={int(codes.min())}, max={int(codes.max())}."
            )
        out = np.zeros((n, m), dtype=np.float64)
        out[np.arange(n), codes - 1] = 1.0
        return out

    def derivative(self, y: NDArray[np.float64], order: int = 1) -> NDArray[np.float64]:
        """Derivative w.r.t. ``y`` — zero, because ``h`` is a step function."""
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        return np.zeros((y_arr.size, self.K - 1), dtype=np.float64)

    def integrate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Not defined for the ordinal basis — raises ``NotImplementedError``."""
        raise NotImplementedError(
            "OrdinalBasis has no continuous integral; use evaluate() instead."
        )


# ---------------------------------------------------------------------------
# Polynomial (monomial) basis
# ---------------------------------------------------------------------------


@dataclass
class PolynomialBasis:
    """Power/monomial basis [1, t, t², …, tᵏ] on a compact support.

    Normalises ``y`` to ``t = (y − a) / (b − a) ∈ [0, 1]`` and returns the
    Vandermonde matrix ``[1, t, t², …, tᵏ]``.  Unlike Bernstein polynomials
    these basis functions are not non-negative and the coefficient vector has
    no built-in monotonicity; constraints must be imposed externally.

    Parameters
    ----------
    order:
        Polynomial degree k.  The basis has k+1 functions.
    support:
        Closed interval (a, b) with a < b.
    """

    order: int
    support: tuple[float, float]

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        if not (np.isfinite(self.support[0]) and np.isfinite(self.support[1])):
            raise ValueError(f"support bounds must be finite, got {self.support}")
        if self.support[0] >= self.support[1]:
            raise ValueError(f"support must satisfy a < b, got {self.support}")

    def evaluate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vandermonde design matrix at observations y.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``[support[0], support[1]]``.

        Returns
        -------
        NDArray of shape (n, order+1) with columns [1, t, t², …, tᵏ].

        Raises
        ------
        ValueError
            If any observation lies outside ``support``.
        """
        t = _normalize_and_validate_support(y, self.support)
        if t.size == 0:
            return np.zeros((0, self.order + 1), dtype=np.float64)
        i = np.arange(self.order + 1, dtype=float)
        return cast(NDArray[np.float64], t[:, None] ** i)

    def derivative(self, y: NDArray[np.float64], order: int = 1) -> NDArray[np.float64]:
        """Analytical derivative of the monomial design matrix.

        Parameters
        ----------
        y:
            Observations, shape (n,).
        order:
            Derivative order: 1 or 2.

        Returns
        -------
        NDArray of shape (n, self.order+1).

        Raises
        ------
        ValueError
            If ``order`` is not 1 or 2.
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        a, b = self.support
        t = _normalize_and_validate_support(y, self.support)
        k = self.order
        n = t.size
        i = np.arange(k + 1, dtype=float)

        if order == 1:
            # d/dy [t^i] = i * t^(i-1) / (b-a),  with 0^(-1) ≡ 0
            if n == 0:
                return np.zeros((0, k + 1), dtype=np.float64)
            exponents = np.maximum(i - 1, 0.0)
            dB = i[None, :] * t[:, None] ** exponents[None, :]
            dB[:, 0] = 0.0
            return cast(NDArray[np.float64], dB / (b - a))
        else:  # order == 2
            # d²/dy² [t^i] = i*(i-1) * t^(i-2) / (b-a)²
            if n == 0:
                return np.zeros((0, k + 1), dtype=np.float64)
            exponents = np.maximum(i - 2, 0.0)
            dB2 = i[None, :] * (i - 1)[None, :] * t[:, None] ** exponents[None, :]
            dB2[:, 0] = 0.0
            if k >= 1:
                dB2[:, 1] = 0.0
            return cast(NDArray[np.float64], dB2 / (b - a) ** 2)

    def evaluate_with_derivative(
        self, y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return design matrix and first derivative in one pass.

        Parameters
        ----------
        y:
            Observations, shape (n,).

        Returns
        -------
        B  : NDArray of shape (n, order+1)
        dB : NDArray of shape (n, order+1)
        """
        return self.evaluate(y), self.derivative(y, order=1)

    def integrate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Running integral ∫_a^y tⁱ · (b−a) dt for each basis function.

        Uses the closed-form antiderivative: ∫_0^t sⁱ ds = tⁱ⁺¹/(i+1),
        scaled by (b−a) to convert dt → dy.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``[support[0], support[1]]``.

        Returns
        -------
        NDArray of shape (n, order+1).
        """
        a, b = self.support
        t = _normalize_and_validate_support(y, self.support)
        if t.size == 0:
            return np.zeros((0, self.order + 1), dtype=np.float64)
        i = np.arange(self.order + 1, dtype=float)
        result = t[:, None] ** (i + 1) / (i + 1) * (b - a)
        return cast(NDArray[np.float64], result)


# ---------------------------------------------------------------------------
# Legendre polynomial basis
# ---------------------------------------------------------------------------


def _legendre_matrix(t: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Evaluate Legendre polynomials P_0(t), …, P_k(t) via 3-term recurrence.

    (n+1) P_{n+1}(t) = (2n+1) t P_n(t) − n P_{n-1}(t)

    Parameters
    ----------
    t:
        Evaluation points in [−1, 1], shape (n_obs,).
    k:
        Maximum degree.

    Returns
    -------
    NDArray of shape (n_obs, k+1).
    """
    n_obs = t.shape[0]
    P = np.zeros((n_obs, k + 1), dtype=np.float64)
    if k >= 0:
        P[:, 0] = 1.0
    if k >= 1:
        P[:, 1] = t
    for m in range(1, k):
        P[:, m + 1] = ((2 * m + 1) * t * P[:, m] - m * P[:, m - 1]) / (m + 1)
    return P


def _legendre_derivative_matrix(
    t: NDArray[np.float64], k: int, P: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Derivatives P_0'(t), …, P_k'(t) using the recurrence.

    P_0'(t) = 0,  P_1'(t) = 1,
    P_n'(t) = P_{n-2}'(t) + (2n−1) P_{n-1}(t)  for n ≥ 2.

    Parameters
    ----------
    t:
        Evaluation points, shape (n_obs,).  Unused but accepted for symmetry.
    k:
        Maximum degree.
    P:
        Legendre matrix of shape (n_obs, k+1) from :func:`_legendre_matrix`.

    Returns
    -------
    NDArray of shape (n_obs, k+1).
    """
    n_obs = P.shape[0]
    dP = np.zeros((n_obs, k + 1), dtype=np.float64)
    if k >= 1:
        dP[:, 1] = 1.0
    for m in range(2, k + 1):
        dP[:, m] = dP[:, m - 2] + (2 * m - 1) * P[:, m - 1]
    return dP


@dataclass
class LegendreBasis:
    """Legendre polynomial basis P_0, P_1, …, P_k on a compact support.

    Maps ``y`` to ``t = 2 · (y − a) / (b − a) − 1 ∈ [−1, 1]`` and evaluates
    Legendre polynomials via the 3-term recurrence.  The basis is orthogonal
    with respect to the uniform measure on ``[a, b]``:

        ∫_a^b P_m(t(y)) P_n(t(y)) dy = (b−a) / (2n+1) · δ_{mn}

    Parameters
    ----------
    order:
        Maximum degree k.  The basis has k+1 functions.
    support:
        Closed interval (a, b) with a < b.
    """

    order: int
    support: tuple[float, float]

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        if not (np.isfinite(self.support[0]) and np.isfinite(self.support[1])):
            raise ValueError(f"support bounds must be finite, got {self.support}")
        if self.support[0] >= self.support[1]:
            raise ValueError(f"support must satisfy a < b, got {self.support}")

    def _normalize_to_legendre(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate support and map y → t ∈ [−1, 1]."""
        t01 = _normalize_and_validate_support(y, self.support)
        return 2.0 * t01 - 1.0

    def evaluate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Legendre design matrix at observations y.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``[support[0], support[1]]``.

        Returns
        -------
        NDArray of shape (n, order+1).

        Raises
        ------
        ValueError
            If any observation lies outside ``support``.
        """
        t = self._normalize_to_legendre(y)
        if t.size == 0:
            return np.zeros((0, self.order + 1), dtype=np.float64)
        return _legendre_matrix(t, self.order)

    def derivative(self, y: NDArray[np.float64], order: int = 1) -> NDArray[np.float64]:
        """Analytical first derivative of the Legendre design matrix.

        Uses the chain rule: d/dy P_n(t(y)) = P_n'(t) · dt/dy
        where dt/dy = 2/(b−a).

        Parameters
        ----------
        y:
            Observations, shape (n,).
        order:
            Derivative order.  Only order=1 is supported.

        Returns
        -------
        NDArray of shape (n, self.order+1).

        Raises
        ------
        ValueError
            If ``order`` is not 1.
        """
        if order != 1:
            raise ValueError(
                f"LegendreBasis.derivative only supports order=1, got {order}"
            )
        a, b = self.support
        t = self._normalize_to_legendre(y)
        if t.size == 0:
            return np.zeros((0, self.order + 1), dtype=np.float64)
        P = _legendre_matrix(t, self.order)
        dP_dt = _legendre_derivative_matrix(t, self.order, P)
        return dP_dt * (2.0 / (b - a))

    def evaluate_with_derivative(
        self, y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return Legendre design matrix and first derivative in one pass.

        Parameters
        ----------
        y:
            Observations, shape (n,).

        Returns
        -------
        B  : NDArray of shape (n, order+1)
        dB : NDArray of shape (n, order+1)
        """
        a, b = self.support
        t = self._normalize_to_legendre(y)
        if t.size == 0:
            empty = np.zeros((0, self.order + 1), dtype=np.float64)
            return empty, empty
        P = _legendre_matrix(t, self.order)
        dP_dt = _legendre_derivative_matrix(t, self.order, P)
        return P, dP_dt * (2.0 / (b - a))

    def integrate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Running integral ∫_a^y P_n(t(s)) ds for each Legendre polynomial.

        Uses the closed-form antiderivative:

            ∫_a^y P_0(t) ds = y − a
            ∫_a^y P_n(t) ds = (b−a)/2 · [P_{n+1}(t) − P_{n-1}(t)] / (2n+1)

        where the constant of integration is fixed by F_n(−1) = 0 for n ≥ 1.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``[support[0], support[1]]``.

        Returns
        -------
        NDArray of shape (n, order+1).
        """
        a, b = self.support
        t = self._normalize_to_legendre(y)
        if t.size == 0:
            return np.zeros((0, self.order + 1), dtype=np.float64)
        k = self.order
        # Need P up to degree k+1 for the antiderivative formula
        P_ext = _legendre_matrix(t, k + 1)  # shape (n, k+2)
        result = np.zeros((t.size, k + 1), dtype=np.float64)
        # n=0: ∫_{-1}^{t} 1 du = t+1 = 2(y-a)/(b-a), scaled: (b-a)/2 * (t+1) = y-a
        result[:, 0] = (b - a) / 2.0 * (t + 1.0)
        for n in range(1, k + 1):
            # ∫_{-1}^{t} P_n(u) du = (P_{n+1}(t) - P_{n-1}(t)) / (2n+1)
            antideriv = (P_ext[:, n + 1] - P_ext[:, n - 1]) / (2 * n + 1)
            result[:, n] = (b - a) / 2.0 * antideriv
        return cast(NDArray[np.float64], result)


# ---------------------------------------------------------------------------
# Log basis
# ---------------------------------------------------------------------------


@dataclass
class LogBasis:
    """Single-function log basis: evaluate(y) = log(y), shape (n, 1).

    Returns a one-column design matrix whose sole basis function is ``log(y)``.
    Useful for log-linear transformations such as the Weibull model.

    The support lower bound must be strictly positive.

    Parameters
    ----------
    support:
        Closed interval (a, b) with 0 < a < b.
    """

    support: tuple[float, float]

    def __post_init__(self) -> None:
        a, b = self.support
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f"support bounds must be finite, got {self.support}")
        if a <= 0.0:
            raise ValueError(
                f"support lower bound must be strictly positive for LogBasis, got a={a}"
            )
        if a >= b:
            raise ValueError(f"support must satisfy a < b, got {self.support}")

    @property
    def order(self) -> int:
        """One basis function → ``order = 0``."""
        return 0

    def _validate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return _normalize_and_validate_support(y, self.support)

    def evaluate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return log(y) as a column, shape (n, 1).

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``[support[0], support[1]]``.

        Returns
        -------
        NDArray of shape (n, 1).

        Raises
        ------
        ValueError
            If any observation lies outside ``support``.
        """
        self._validate(y)
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        return cast(NDArray[np.float64], np.log(y_arr)[:, None])

    def derivative(self, y: NDArray[np.float64], order: int = 1) -> NDArray[np.float64]:
        """Analytical derivative: d/dy log(y) = 1/y, shape (n, 1).

        Parameters
        ----------
        y:
            Observations, shape (n,).
        order:
            Derivative order.  Only order=1 is supported.

        Returns
        -------
        NDArray of shape (n, 1).

        Raises
        ------
        ValueError
            If ``order`` is not 1.
        """
        if order != 1:
            raise ValueError(f"LogBasis.derivative only supports order=1, got {order}")
        self._validate(y)
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        return (1.0 / y_arr)[:, None]

    def evaluate_with_derivative(
        self, y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return log(y) and 1/y in one pass, both shape (n, 1).

        Parameters
        ----------
        y:
            Observations, shape (n,).

        Returns
        -------
        B  : NDArray of shape (n, 1)  — log(y)
        dB : NDArray of shape (n, 1)  — 1/y
        """
        self._validate(y)
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        return np.log(y_arr)[:, None], (1.0 / y_arr)[:, None]

    def integrate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Running integral ∫_a^y log(s) ds, shape (n, 1).

        Closed form: ∫_a^y log(s) ds = y·log(y) − y − a·log(a) + a.

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``[support[0], support[1]]``.

        Returns
        -------
        NDArray of shape (n, 1).
        """
        a, _ = self.support
        self._validate(y)
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        result = y_arr * np.log(y_arr) - y_arr - (a * np.log(a) - a)
        return cast(NDArray[np.float64], result[:, None])


# ---------------------------------------------------------------------------
# Intercept basis
# ---------------------------------------------------------------------------


@dataclass
class InterceptBasis:
    """Constant (intercept-only) basis: evaluate(y) = ones, shape (n, 1).

    The single basis function is identically 1.  This gives a single free
    parameter — an additive intercept — in the transformation h(y) = θ₀.

    Parameters
    ----------
    support:
        Closed interval (a, b) with a < b.  Used for support validation only.
    """

    support: tuple[float, float]

    def __post_init__(self) -> None:
        if not (np.isfinite(self.support[0]) and np.isfinite(self.support[1])):
            raise ValueError(f"support bounds must be finite, got {self.support}")
        if self.support[0] >= self.support[1]:
            raise ValueError(f"support must satisfy a < b, got {self.support}")

    @property
    def order(self) -> int:
        """One basis function → ``order = 0``."""
        return 0

    def evaluate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return a column of ones, shape (n, 1).

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``[support[0], support[1]]``.

        Returns
        -------
        NDArray of shape (n, 1).

        Raises
        ------
        ValueError
            If any observation lies outside ``support``.
        """
        t = _normalize_and_validate_support(y, self.support)
        return cast(NDArray[np.float64], np.ones((t.size, 1), dtype=np.float64))

    def derivative(self, y: NDArray[np.float64], order: int = 1) -> NDArray[np.float64]:
        """Derivative of the constant basis: always zero, shape (n, 1).

        Parameters
        ----------
        y:
            Observations, shape (n,).
        order:
            Derivative order: 1 or 2.

        Returns
        -------
        NDArray of shape (n, 1) of zeros.

        Raises
        ------
        ValueError
            If ``order`` is not 1 or 2.
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        t = _normalize_and_validate_support(y, self.support)
        return np.zeros((t.size, 1), dtype=np.float64)

    def evaluate_with_derivative(
        self, y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ones and zeros, both shape (n, 1).

        Parameters
        ----------
        y:
            Observations, shape (n,).

        Returns
        -------
        B  : NDArray of shape (n, 1)  — ones
        dB : NDArray of shape (n, 1)  — zeros
        """
        t = _normalize_and_validate_support(y, self.support)
        n = t.size
        return (
            np.ones((n, 1), dtype=np.float64),
            np.zeros((n, 1), dtype=np.float64),
        )

    def integrate(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Running integral ∫_a^y 1 ds = y − a, shape (n, 1).

        Parameters
        ----------
        y:
            Observations, shape (n,).  Must lie in ``[support[0], support[1]]``.

        Returns
        -------
        NDArray of shape (n, 1).
        """
        a, _ = self.support
        _normalize_and_validate_support(y, self.support)
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        return (y_arr - a)[:, None]


# ---------------------------------------------------------------------------
# Tensor-product interaction basis (stub — implementation in slice 2)
# ---------------------------------------------------------------------------

# Supported x-basis types for closed-form column-wise monotonicity constraints.
# See ADR 0001, Decision 3.
_SUPPORTED_X_BASIS_TYPES: tuple[type, ...] = (
    BernsteinBasis,
    OrdinalBasis,
    InterceptBasis,
)


@dataclass
class InteractionBasis:
    """Tensor-product basis a(y) ⊗ b(x) for fully-interacting CTMs.

    Models the transformation

        h(y|x) = (a(y) ⊗ b(x))ᵀ vec(Θ)

    where ``a`` is the *y-basis* (response) and ``b`` is the *x-basis*
    (covariate), and ``Θ`` is a ``(p, q)`` coefficient matrix with
    ``p = y_basis.order + 1`` and ``q = x_basis.order + 1``.

    The parameter vector ``theta_`` stores ``vec_C(Θ)`` (row-major /
    C-order flattening) of length ``p * q``.  See ADR 0001 for the full
    design rationale.

    **Supported x-basis types (initial release):**
    :class:`BernsteinBasis`, :class:`OrdinalBasis`, :class:`InterceptBasis`.
    Other x-basis types raise ``ValueError`` at constraint-building time
    because the closed-form column-wise monotonicity guarantee requires the
    x-basis to be non-negative and a partition of unity.

    Parameters
    ----------
    y_basis:
        Basis for the response variable ``y``.
    x_basis:
        Basis for the covariate(s) ``x``.  Must be one of the supported
        types listed above.

    Notes
    -----
    The ``evaluate`` and ``derivative`` signatures differ from the scalar
    basis interface: they accept both ``y`` and ``X`` because the Kronecker
    product requires both.  The model layer owns this two-argument call
    convention.

    References
    ----------
    See ``docs/adr/0001-tensor-product-interaction-basis.md``.
    """

    y_basis: (
        BernsteinBasis
        | LogBernsteinBasis
        | PolynomialBasis
        | LegendreBasis
        | LogBasis
        | InterceptBasis
        | OrdinalBasis
    )
    x_basis: (
        BernsteinBasis
        | OrdinalBasis
        | InterceptBasis
    )

    def __post_init__(self) -> None:
        if not isinstance(self.x_basis, _SUPPORTED_X_BASIS_TYPES):
            raise ValueError(
                f"InteractionBasis requires an x-basis that is non-negative and "
                f"a partition of unity (BernsteinBasis, OrdinalBasis, or "
                f"InterceptBasis). Got {type(self.x_basis).__name__}. "
                f"See docs/adr/0001-tensor-product-interaction-basis.md, "
                f"Decision 3."
            )

    @property
    def order(self) -> int:
        """``y_basis.order`` — used by model layer for param-count bookkeeping."""
        return self.y_basis.order

    @property
    def support(self) -> tuple[float, float]:
        """Support of the y-basis."""
        return self.y_basis.support

    @property
    def n_y_params(self) -> int:
        """Number of y-basis functions: ``y_basis.order + 1``."""
        return self.y_basis.order + 1

    @property
    def n_x_params(self) -> int:
        """Number of x-basis functions: ``x_basis.order + 1``."""
        return self.x_basis.order + 1

    @property
    def n_params(self) -> int:
        """Total number of free parameters: ``n_y_params * n_x_params``."""
        return self.n_y_params * self.n_x_params

    def evaluate(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Row-wise Kronecker product a(y_i) ⊗ b(x_i), shape (n, p*q).

        Parameters
        ----------
        y:
            Response observations, shape (n,).
        X:
            Covariate matrix, shape (n, d) where d is the dimension of the
            x-basis input.

        Returns
        -------
        NDArray of shape (n, p*q).

        Raises
        ------
        NotImplementedError
            Until slice 2 is implemented.
        """
        raise NotImplementedError(
            "InteractionBasis.evaluate is not yet implemented. "
            "This stub exists for import compatibility. "
            "See docs/adr/0001-tensor-product-interaction-basis.md."
        )

    def derivative(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64],
        order: int = 1,
    ) -> NDArray[np.float64]:
        """Row-wise Kronecker product da(y_i)/dy ⊗ b(x_i), shape (n, p*q).

        Parameters
        ----------
        y:
            Response observations, shape (n,).
        X:
            Covariate matrix, shape (n, d).
        order:
            Derivative order.  Only order=1 is supported.

        Returns
        -------
        NDArray of shape (n, p*q).

        Raises
        ------
        NotImplementedError
            Until slice 2 is implemented.
        """
        raise NotImplementedError(
            "InteractionBasis.derivative is not yet implemented. "
            "This stub exists for import compatibility. "
            "See docs/adr/0001-tensor-product-interaction-basis.md."
        )

    def evaluate_with_derivative(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return evaluate(y, X) and derivative(y, X) in one pass.

        Parameters
        ----------
        y:
            Response observations, shape (n,).
        X:
            Covariate matrix, shape (n, d).

        Returns
        -------
        B  : NDArray of shape (n, p*q)
        dB : NDArray of shape (n, p*q)

        Raises
        ------
        NotImplementedError
            Until slice 2 is implemented.
        """
        raise NotImplementedError(
            "InteractionBasis.evaluate_with_derivative is not yet implemented. "
            "This stub exists for import compatibility. "
            "See docs/adr/0001-tensor-product-interaction-basis.md."
        )

    def integrate(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Running integral of a(y) ⊗ b(x) w.r.t. y.

        Parameters
        ----------
        y:
            Response observations, shape (n,).
        X:
            Covariate matrix, shape (n, d).

        Returns
        -------
        NDArray of shape (n, p*q).

        Raises
        ------
        NotImplementedError
            Until slice 2 is implemented.
        """
        raise NotImplementedError(
            "InteractionBasis.integrate is not yet implemented. "
            "This stub exists for import compatibility. "
            "See docs/adr/0001-tensor-product-interaction-basis.md."
        )
