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
