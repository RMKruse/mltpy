"""Linear constraints for monotone Bernstein coefficient optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint

# ---------------------------------------------------------------------------
# Monotonicity constraint
# ---------------------------------------------------------------------------


@dataclass
class MonotonicityConstraint:
    """Encodes the constraint that Bernstein coefficients are non-decreasing.

    For a coefficient vector ``theta`` of length ``n_params``, monotonicity
    of the transformation h(y) = B_k(y) · theta requires::

        theta[0] <= theta[1] <= ... <= theta[n_params-1]

    This is equivalent to the linear inequality::

        D @ theta >= 0

    where ``D`` is the ``(n_params-1, n_params)`` forward-difference matrix::

        D = [[-1,  1,  0,  0, ...],
             [ 0, -1,  1,  0, ...],
             [ 0,  0, -1,  1, ...],
             ...]

    Parameters
    ----------
    n_params:
        Number of Bernstein coefficients (= polynomial degree + 1).
    """

    n_params: int
    _D: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_params < 2:
            raise ValueError(f"n_params must be >= 2, got {self.n_params}")
        # Computed once — Jacobian is constant (linear constraint).
        self._D = np.diff(np.eye(self.n_params), axis=0)

    # ------------------------------------------------------------------

    def as_matrix(self) -> NDArray[np.float64]:
        """Return the (n_params-1, n_params) difference matrix D."""
        return self._D.copy()

    def as_scipy_constraint(self) -> dict[str, Any]:
        """Return an SLSQP-compatible constraint dict.

        ``fun(theta)`` returns the vector ``D @ theta``; each element must
        be >= 0 for the constraint to be satisfied.  ``jac(theta)`` returns
        ``D`` (constant, computed once in ``__post_init__``).
        """
        D = self._D  # captured by closure — not recomputed on each call
        return {
            "type": "ineq",
            "fun": lambda theta: D @ theta,
            "jac": lambda theta: D,
        }

    def as_LinearConstraint(self) -> LinearConstraint:
        """Return a ``LinearConstraint`` for use with ``trust-constr``."""
        return LinearConstraint(self._D, lb=0.0, ub=np.inf)


# ---------------------------------------------------------------------------
# Boundary constraint
# ---------------------------------------------------------------------------


@dataclass
class BoundaryConstraint:
    """Fix one or both boundary coefficients of the Bernstein expansion.

    Enforces equality constraints::

        theta[0]  == lower   (if lower is not None)
        theta[-1] == upper   (if upper is not None)

    Parameters
    ----------
    n_params:
        Number of Bernstein coefficients.
    lower:
        Value to fix ``theta[0]`` to, or ``None`` to leave it free.
    upper:
        Value to fix ``theta[-1]`` to, or ``None`` to leave it free.
    """

    n_params: int
    lower: float | None
    upper: float | None
    _A: NDArray[np.float64] = field(init=False, repr=False)
    _rhs: NDArray[np.float64] = field(init=False, repr=False)
    _jacs: list[NDArray[np.float64]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError("At least one of lower or upper must be provided")

        rows: list[NDArray[np.float64]] = []
        rhs: list[float] = []
        jacs: list[NDArray[np.float64]] = []

        e0 = np.eye(self.n_params)[0]
        e_last = np.eye(self.n_params)[-1]

        if self.lower is not None:
            rows.append(e0)
            rhs.append(float(self.lower))
            jacs.append(e0)
        if self.upper is not None:
            rows.append(e_last)
            rhs.append(float(self.upper))
            jacs.append(e_last)

        # _A: shape (n_active, n_params); _rhs: shape (n_active,)
        self._A = np.array(rows)
        self._rhs = np.array(rhs)
        self._jacs = jacs  # per-constraint row vectors (precomputed)

    # ------------------------------------------------------------------

    def as_scipy_constraint(self) -> list[dict[str, Any]]:
        """Return SLSQP-compatible equality constraint dicts.

        Returns one dict per active boundary (one or two elements).
        """
        constraints: list[dict[str, Any]] = []
        idx = 0
        if self.lower is not None:
            lo = self.lower
            jac_row = self._jacs[idx]
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda theta, lo=lo: theta[0] - lo,
                    "jac": lambda theta, j=jac_row: j,
                }
            )
            idx += 1
        if self.upper is not None:
            up = self.upper
            jac_row = self._jacs[idx]
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda theta, up=up: theta[-1] - up,
                    "jac": lambda theta, j=jac_row: j,
                }
            )
        return constraints

    def as_LinearConstraint(self) -> LinearConstraint:
        """Return a ``LinearConstraint`` for use with ``trust-constr``.

        Both ``lb`` and ``ub`` are set to ``rhs`` (equality).
        """
        return LinearConstraint(self._A, lb=self._rhs, ub=self._rhs)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_constraints(
    n_params: int,
    lower: float | None = None,
    upper: float | None = None,
    solver: Literal["slsqp", "trust-constr"] = "slsqp",
    total_params: int | None = None,
    nonneg_lower: bool = False,
) -> list[dict[str, Any]] | list[LinearConstraint]:
    """Build all optimisation constraints for a Bernstein model.

    Always includes the monotonicity constraint (non-decreasing ``theta``).
    Optionally adds boundary equality constraints when ``lower`` or ``upper``
    are specified, and an inequality ``theta[0] >= 0`` when ``nonneg_lower``
    is set.

    ``optimizer.py`` calls this function — it does **not** instantiate the
    constraint classes directly.

    Parameters
    ----------
    n_params:
        Number of Bernstein coefficients (= ``BernsteinBasis.order + 1``).
    lower:
        If not ``None``, fix ``theta[0] == lower``.
    upper:
        If not ``None``, fix ``theta[-1] == upper``.
    solver:
        ``"slsqp"``  → returns ``list[dict]`` (for ``scipy.optimize.minimize``
                        with ``method="SLSQP"``).
        ``"trust-constr"`` → returns ``list[LinearConstraint]`` (for
                             ``method="trust-constr"``).
    total_params:
        Total length of the parameter vector passed to the optimiser, including
        any regression coefficients (``beta``).  When ``total_params > n_params``
        the constraint matrix is padded with zero columns so that it maps the
        full ``theta`` vector.  Defaults to ``n_params`` (no beta).
    nonneg_lower:
        If ``True``, add the inequality ``theta[0] >= 0``.  Used for
        ``base_distribution="exponential"``, whose support is ``[0, ∞)``:
        combined with the monotonicity constraint this guarantees
        ``h(y) = B_k(y) · theta >= 0`` for all ``y``.  Kept distinct from
        ``lower`` because ``lower`` is an *equality* that pins ``theta[0]``.

    Returns
    -------
    list[dict] for ``solver="slsqp"``, list[LinearConstraint] for
    ``solver="trust-constr"``.
    """
    total = total_params if total_params is not None else n_params
    mono = MonotonicityConstraint(n_params)
    D = mono.as_matrix()  # shape (n_params-1, n_params)

    # Pad D with zero columns for regression coefficients
    if total > n_params:
        D = np.hstack([D, np.zeros((D.shape[0], total - n_params))])

    # Row selecting theta[0] (padded to full parameter length)
    e0 = np.zeros((1, total))
    e0[0, 0] = 1.0

    if solver == "slsqp":
        result: list[dict[str, Any]] = [
            {
                "type": "ineq",
                "fun": lambda theta, _D=D: _D @ theta,
                "jac": lambda theta, _D=D: _D,
            }
        ]
        if nonneg_lower:
            result.append(
                {
                    "type": "ineq",
                    "fun": lambda theta, _e=e0: _e @ theta,
                    "jac": lambda theta, _e=e0: _e,
                }
            )
        if lower is not None or upper is not None:
            bc = BoundaryConstraint(n_params, lower=lower, upper=upper)
            result.extend(bc.as_scipy_constraint())
        return result

    else:  # trust-constr
        lcs: list[LinearConstraint] = [LinearConstraint(D, lb=0.0, ub=np.inf)]
        if nonneg_lower:
            lcs.append(LinearConstraint(e0, lb=0.0, ub=np.inf))
        if lower is not None or upper is not None:
            bc = BoundaryConstraint(n_params, lower=lower, upper=upper)
            lcs.append(bc.as_LinearConstraint())
        return lcs
