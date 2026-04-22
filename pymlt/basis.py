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
    y_arr = np.asarray(y, dtype=float).ravel()
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
            Derivative order: 1 (default) or 2.

        Returns
        -------
        NDArray of shape (n, self.order+1).

        Raises
        ------
        ValueError
            If ``order`` is not 1 or 2, or if any observation lies outside
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
# Module-level convenience function
# ---------------------------------------------------------------------------


def monotone_trafo(
    theta: NDArray[np.float64], basis_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the transformation h(y) = basis_matrix @ theta.

    Monotone iff `theta` is non-decreasing.  This function does **not**
    enforce that constraint — enforcement is handled by ``constraints.py``
    (Step 3).

    Parameters
    ----------
    theta:
        Coefficient vector, shape (order+1,).
    basis_matrix:
        Design matrix from ``BernsteinBasis.evaluate``, shape (n, order+1).

    Returns
    -------
    NDArray of shape (n,).
    """
    return basis_matrix @ theta
