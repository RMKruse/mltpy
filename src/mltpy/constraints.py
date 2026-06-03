"""Linear constraints for monotone Bernstein coefficient optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint

if TYPE_CHECKING:
    from mltpy.basis import InteractionBasis

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
        if self.n_params < 1:
            raise ValueError(f"n_params must be >= 1, got {self.n_params}")
        # Computed once — Jacobian is constant (linear constraint).  When
        # n_params == 1 (e.g. K=2 ordinal model) this is a (0, 1) array and
        # ``D @ theta`` is the empty constraint vector — the monotonicity
        # condition is vacuous and SLSQP/trust-constr accept zero-row blocks.
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

        theta[0]          == lower   (if lower is not None)
        theta[n_params-1] == upper   (if upper is not None)

    Parameters
    ----------
    n_params:
        Number of Bernstein coefficients.
    lower:
        Value to fix ``theta[0]`` to, or ``None`` to leave it free.
    upper:
        Value to fix ``theta[n_params-1]`` to, or ``None`` to leave it free.
    total_params:
        Total length of the parameter vector passed to the optimiser, including
        any regression coefficients.  When ``total_params > n_params`` the
        constraint rows are padded with zero columns for the beta entries so
        that ``_A`` has shape ``(n_active, total_params)``.  Defaults to
        ``n_params`` (no beta).
    """

    n_params: int
    lower: float | None
    upper: float | None
    total_params: int | None = None
    _A: NDArray[np.float64] = field(init=False, repr=False)
    _rhs: NDArray[np.float64] = field(init=False, repr=False)
    _jacs: list[NDArray[np.float64]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError("At least one of lower or upper must be provided")

        total = self.total_params if self.total_params is not None else self.n_params

        rows: list[NDArray[np.float64]] = []
        rhs: list[float] = []
        jacs: list[NDArray[np.float64]] = []

        e0 = np.zeros(total)
        e0[0] = 1.0
        e_upper = np.zeros(total)
        e_upper[self.n_params - 1] = 1.0

        if self.lower is not None:
            rows.append(e0)
            rhs.append(float(self.lower))
            jacs.append(e0)
        if self.upper is not None:
            rows.append(e_upper)
            rhs.append(float(self.upper))
            jacs.append(e_upper)

        # _A: shape (n_active, total); _rhs: shape (n_active,)
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
            n = self.n_params
            jac_row = self._jacs[idx]
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda theta, up=up, n=n: theta[n - 1] - up,
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
# Constraint matrix dataclass for auglag solver
# ---------------------------------------------------------------------------


@dataclass
class ConstraintMatrices:
    """Constraint matrices in the canonical form used by :func:`auglag_minimize`.

    Represents the linear constraints as:

        A_ineq @ θ ≥ b_ineq   (inequality)
        C_eq   @ θ  = d_eq    (equality)

    Parameters
    ----------
    A_ineq:
        Inequality constraint matrix, shape (m_ineq, total_params).
    b_ineq:
        Inequality right-hand side, shape (m_ineq,).
    C_eq:
        Equality constraint matrix, shape (m_eq, total_params).
        Zero-row matrix when there are no equality constraints.
    d_eq:
        Equality right-hand side, shape (m_eq,).
        Zero-length array when there are no equality constraints.
    """

    A_ineq: NDArray[np.float64]
    b_ineq: NDArray[np.float64]
    C_eq: NDArray[np.float64]
    d_eq: NDArray[np.float64]


def _assemble_constraint_arrays(
    n_params: int,
    *,
    total_params: int | None,
    nonneg_lower: bool,
    X: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Build the inequality rows shared by both constraint builders.

    Returns the padded monotonicity matrix ``D`` (shape
    ``(n_params-1, total)``) and the optional support-feasibility rows
    ``support_rows`` (shape ``(n_rows, total)`` or ``None``).  This is the
    block that :func:`build_constraints` and :func:`build_constraint_matrices`
    previously duplicated character-for-character; both now consume its output
    — :func:`build_constraints` keeps ``D`` and ``support_rows`` as separate
    inequality blocks (SLSQP dict / ``LinearConstraint`` per block), while
    :func:`build_constraint_matrices` stacks them into a single ``A_ineq``.

    Parameters
    ----------
    n_params:
        Number of Bernstein coefficients (= ``BernsteinBasis.order + 1``).
    total_params:
        Total parameter-vector length including regression coefficients.
        When ``total_params > n_params`` the monotonicity matrix and the
        support rows are padded with zero columns for the ``beta`` block.
        Defaults to ``n_params``.
    nonneg_lower:
        If ``True``, append support-feasibility rows enforcing
        ``h(y|x) >= 0`` (exponential support).  See :func:`build_constraints`.
    X:
        Covariate matrix, shape ``(n, q)``.  Only consulted when
        ``nonneg_lower=True``.

    Returns
    -------
    D : NDArray of shape ``(n_params-1, total)``
        Padded forward-difference (monotonicity) matrix.
    support_rows : NDArray of shape ``(n_rows, total)`` or None
        Support-feasibility inequality rows, or ``None`` when
        ``nonneg_lower=False``.

    Raises
    ------
    ValueError
        If ``X`` has invalid shape, if ``X`` columns do not match
        ``total_params - n_params``, or if ``nonneg_lower=True`` with ``X``
        but ``total_params`` is omitted.
    """
    total = total_params if total_params is not None else n_params
    D = MonotonicityConstraint(n_params).as_matrix()  # (n_params-1, n_params)

    if total > n_params:
        D = np.hstack([D, np.zeros((D.shape[0], total - n_params))])

    # Support-feasibility rows.  No covariates: one row [1, 0, ..., 0].
    # With covariates: n rows [1, 0, ..., 0 | X_i].
    support_rows: NDArray[np.float64] | None = None
    if nonneg_lower:
        if X is None:
            if total > n_params:
                raise ValueError(
                    "X must be provided when nonneg_lower=True and "
                    "total_params > n_params so support-feasibility "
                    "constraints can include the regression coefficients."
                )
            support_rows = np.zeros((1, total), dtype=np.float64)
            support_rows[0, 0] = 1.0
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim != 2:
                raise ValueError(f"X must be 2-D, got shape {X_arr.shape}")
            if total_params is None:
                raise ValueError(
                    "total_params must be provided when nonneg_lower=True and "
                    "X is passed. Expected full parameter length "
                    "n_params + X.shape[1]."
                )
            if X_arr.shape[1] != total - n_params:
                raise ValueError(
                    f"X has {X_arr.shape[1]} columns but total_params - "
                    f"n_params = {total - n_params}"
                )
            if X_arr.shape[1] == 0:
                support_rows = np.zeros((1, total), dtype=np.float64)
                support_rows[0, 0] = 1.0
            else:
                n_obs = X_arr.shape[0]
                support_rows = np.zeros((n_obs, total), dtype=np.float64)
                support_rows[:, 0] = 1.0
                support_rows[:, n_params:] = X_arr

    return D, support_rows


def build_constraint_matrices_interaction(
    basis: "InteractionBasis",
) -> ConstraintMatrices:
    """Build constraint matrices for an :class:`~mltpy.basis.InteractionBasis`.

    Constructs the Kronecker inequality ``(D ⊗ I_q) @ vec(Θ) ≥ 0`` that
    enforces column-wise monotonicity: ``D @ Θ[:, j] ≥ 0`` for every
    column ``j = 0, …, q-1``.

    Parameters
    ----------
    basis:
        The :class:`~mltpy.basis.InteractionBasis` to build constraints for.

    Returns
    -------
    ConstraintMatrices
        ``A_ineq`` has shape ``((p-1)*q, p*q)``.  ``b_ineq`` is all-zeros.
        ``C_eq`` is a zero-row matrix (no equality constraints).  ``d_eq``
        is a zero-length array.

    Raises
    ------
    ValueError
        If the x-basis type is not supported for closed-form constraints.
        (This is already checked at :class:`~mltpy.basis.InteractionBasis`
        construction time, so this branch is a safety net.)
    """
    from mltpy.basis import _SUPPORTED_X_BASIS_TYPES

    if not isinstance(basis.x_basis, _SUPPORTED_X_BASIS_TYPES):
        raise ValueError(
            f"InteractionBasis requires an x-basis that is non-negative and "
            f"a partition of unity (BernsteinBasis, OrdinalBasis, "
            f"InterceptBasis, or OneHotBasis). Got "
            f"{type(basis.x_basis).__name__}. "
            f"See docs/adr/0001-tensor-product-interaction-basis.md, Decision 3."
        )
    p = basis.n_y_params
    q = basis.n_x_params
    total = p * q

    if p >= 2:
        D = np.diff(np.eye(p), axis=0)  # (p-1, p)
        A_ineq = np.kron(D, np.eye(q)).astype(np.float64)  # ((p-1)*q, p*q)
    else:
        A_ineq = np.zeros((0, total), dtype=np.float64)

    m_ineq = A_ineq.shape[0]
    return ConstraintMatrices(
        A_ineq=A_ineq,
        b_ineq=np.zeros(m_ineq, dtype=np.float64),
        C_eq=np.zeros((0, total), dtype=np.float64),
        d_eq=np.zeros(0, dtype=np.float64),
    )


def build_constraint_matrices(
    n_params: int,
    lower: float | None = None,
    upper: float | None = None,
    *,
    total_params: int | None = None,
    nonneg_lower: bool = False,
    X: NDArray[np.float64] | None = None,
) -> ConstraintMatrices:
    """Build constraint matrices for the augmented Lagrangian solver.

    Returns a :class:`ConstraintMatrices` dataclass whose fields are passed
    directly to :func:`~mltpy._auglag.auglag_minimize`.

    Monotonicity (``A_ineq @ θ ≥ 0``) is always included.  When ``lower`` or
    ``upper`` are provided, equality rows pinning ``θ[0] = lower`` and/or
    ``θ[n_params-1] = upper`` are added to ``C_eq``/``d_eq`` — mirroring
    :class:`BoundaryConstraint`.

    Parameters
    ----------
    n_params:
        Number of Bernstein coefficients (= ``BernsteinBasis.order + 1``).
    lower:
        If not ``None``, pins ``θ[0] = lower`` (equality).
    upper:
        If not ``None``, pins ``θ[n_params-1] = upper`` (equality).
    total_params:
        Total parameter-vector length including any regression coefficients.
        When ``total_params > n_params`` both the monotonicity matrix and the
        boundary rows are padded with zero columns for the ``beta`` block.
        Defaults to ``n_params``.
    nonneg_lower:
        If ``True``, append support-feasibility inequality rows so the
        transformation stays in the exponential support ``[0, ∞)``.  Mirrors
        the ``nonneg_lower`` branch of :func:`build_constraints`:

        * No covariates (``X is None``): a single row ``[1, 0, …, 0]``
          enforcing ``θ_b[0] ≥ 0``.
        * With covariates: one row ``[1, 0, …, 0 | X_i]`` per observation,
          enforcing ``θ_b[0] + X_i · β ≥ 0``.

    X:
        Covariate matrix, shape ``(n, q)``.  Only consulted when
        ``nonneg_lower=True``; ``q`` must equal ``total_params - n_params``.

    Returns
    -------
    ConstraintMatrices
        ``A_ineq`` is the padded forward-difference matrix D (shape
        ``(n_params-1, total_params)``) optionally followed by the support
        rows; ``b_ineq`` is all-zeros.  ``C_eq`` has 0, 1, or 2 rows
        depending on which of ``lower``/``upper`` are provided; ``d_eq``
        carries the corresponding right-hand-side values.

    Raises
    ------
    ValueError
        If ``X`` has invalid shape, if ``X`` columns do not match
        ``total_params - n_params``, or if ``nonneg_lower=True`` with ``X``
        but ``total_params`` is omitted.
    """
    total = total_params if total_params is not None else n_params
    D, support_rows = _assemble_constraint_arrays(
        n_params, total_params=total_params, nonneg_lower=nonneg_lower, X=X
    )

    if support_rows is not None:
        A_ineq = np.vstack([D, support_rows])
    else:
        A_ineq = D

    m_ineq = A_ineq.shape[0]

    # Boundary equality rows.  Mirrors BoundaryConstraint: a single row per
    # active side picking out θ[0] (lower) or θ[n_params-1] (upper).
    if lower is None and upper is None:
        C_eq = np.zeros((0, total), dtype=np.float64)
        d_eq = np.zeros(0, dtype=np.float64)
    else:
        rows: list[NDArray[np.float64]] = []
        rhs: list[float] = []
        if lower is not None:
            e0 = np.zeros(total, dtype=np.float64)
            e0[0] = 1.0
            rows.append(e0)
            rhs.append(float(lower))
        if upper is not None:
            e_upper = np.zeros(total, dtype=np.float64)
            e_upper[n_params - 1] = 1.0
            rows.append(e_upper)
            rhs.append(float(upper))
        C_eq = np.vstack(rows)
        d_eq = np.array(rhs, dtype=np.float64)

    return ConstraintMatrices(
        A_ineq=A_ineq,
        b_ineq=np.zeros(m_ineq, dtype=np.float64),
        C_eq=C_eq,
        d_eq=d_eq,
    )


# ---------------------------------------------------------------------------
# Public builder (SLSQP / trust-constr)
# ---------------------------------------------------------------------------


@overload
def build_constraints(
    n_params: int,
    lower: float | None = None,
    upper: float | None = None,
    *,
    solver: Literal["slsqp"] = "slsqp",
    total_params: int | None = None,
    nonneg_lower: bool = False,
    X: NDArray[np.float64] | None = None,
) -> list[dict[str, Any]]: ...


@overload
def build_constraints(
    n_params: int,
    lower: float | None = None,
    upper: float | None = None,
    *,
    solver: Literal["trust-constr"],
    total_params: int | None = None,
    nonneg_lower: bool = False,
    X: NDArray[np.float64] | None = None,
) -> list[LinearConstraint]: ...


def build_constraints(
    n_params: int,
    lower: float | None = None,
    upper: float | None = None,
    *,
    solver: Literal["slsqp", "trust-constr"] = "slsqp",
    total_params: int | None = None,
    nonneg_lower: bool = False,
    X: NDArray[np.float64] | None = None,
) -> list[dict[str, Any]] | list[LinearConstraint]:
    """Build all optimisation constraints for a Bernstein model.

    Always includes the monotonicity constraint (non-decreasing ``theta``).
    Optionally adds boundary equality constraints when ``lower`` or ``upper``
    are specified, and support-feasibility inequalities when ``nonneg_lower``
    is set (see below).

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
        full ``theta`` vector.  Defaults to ``n_params`` (no beta).  If
        ``nonneg_lower=True`` and ``X`` is passed, this must be provided as the
        full parameter length ``n_params + X.shape[1]``.
    nonneg_lower:
        If ``True``, require ``h(y|x) >= 0``.  Used for
        ``base_distribution="exponential"``, whose support is ``[0, ∞)``.

        * No covariates (``X is None``): the single inequality
          ``theta_b[0] >= 0`` is sufficient, since ``h(y) = B_k(y) · theta_b``
          and ``min_y B_k(y) · theta_b = theta_b[0]`` under monotonicity.
        * With covariates: ``h(y|x) = B_k(y) · theta_b + x'β``; the minimum
          over ``y`` is attained at ``y_min`` (because ``theta_b`` is
          non-decreasing and ``B_k(y_min) = [1, 0, ..., 0]``), giving
          ``min_y h(y|x_i) = theta_b[0] + X_i · β`` per observation ``i``.
          One inequality ``theta_b[0] + X_i · β >= 0`` is added per row of
          ``X``, making the training fit feasible under the exponential
          support.

        Kept distinct from ``lower`` because ``lower`` is an *equality* that
        pins ``theta[0]``.
    X:
        Optional covariate matrix of shape ``(n, q)``.  Only consulted when
        ``nonneg_lower=True`` — see above.  ``q`` must equal
        ``total_params - n_params``.

    Raises
    ------
    ValueError
        If ``X`` has invalid shape, if ``X`` columns do not match
        ``total_params - n_params``, or if ``nonneg_lower=True`` with ``X``
        but ``total_params`` is omitted.

    Returns
    -------
    list[dict] for ``solver="slsqp"``, list[LinearConstraint] for
    ``solver="trust-constr"``.
    """
    total = total_params if total_params is not None else n_params
    D, support_rows = _assemble_constraint_arrays(
        n_params, total_params=total_params, nonneg_lower=nonneg_lower, X=X
    )

    if solver == "slsqp":
        result: list[dict[str, Any]] = [
            {
                "type": "ineq",
                "fun": lambda theta, _D=D: _D @ theta,
                "jac": lambda theta, _D=D: _D,
            }
        ]
        if support_rows is not None:
            result.append(
                {
                    "type": "ineq",
                    "fun": lambda theta, _S=support_rows: _S @ theta,
                    "jac": lambda theta, _S=support_rows: _S,
                }
            )
        if lower is not None or upper is not None:
            bc = BoundaryConstraint(
                n_params, lower=lower, upper=upper, total_params=total
            )
            result.extend(bc.as_scipy_constraint())
        return result

    else:  # trust-constr
        lcs: list[LinearConstraint] = [LinearConstraint(D, lb=0.0, ub=np.inf)]
        if support_rows is not None:
            lcs.append(LinearConstraint(support_rows, lb=0.0, ub=np.inf))
        if lower is not None or upper is not None:
            bc = BoundaryConstraint(
                n_params, lower=lower, upper=upper, total_params=total
            )
            lcs.append(bc.as_LinearConstraint())
        return lcs
