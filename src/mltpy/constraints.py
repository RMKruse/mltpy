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
    nonlinear_ineq:
        Optional smooth inequality block ``g(θ) ≥ 0`` stacked below
        ``A_ineq`` (passed to ``auglag_minimize(nonlinear_ineq=…)``): the
        exponential support rows under a scaling term
        (:class:`ScaledSupportConstraint`, issue #115).  ``None`` when every
        constraint is linear.
    """

    A_ineq: NDArray[np.float64]
    b_ineq: NDArray[np.float64]
    C_eq: NDArray[np.float64]
    d_eq: NDArray[np.float64]
    nonlinear_ineq: ScaledSupportConstraint | None = None


@dataclass(frozen=True, eq=False)
class ScaledSupportConstraint:
    """Exponential support rows under a scaling term, as a nonlinear block.

    Callable ``θ ↦ (g(θ), J(θ))`` for the rows (ADR 0005 Decision 2)

        g_i(θ) = θ_b[0] · f_i + x_d,i · β + offset_i ≥ 0,
        f_i = exp(0.5 · x_s,i · γ),    i = 1, …, n,

    on the parameter vector ``θ = [θ_b | β | γ]``.  Build it with
    :func:`build_scaled_support_constraint`, which validates and copies the
    design.

    Parameters
    ----------
    n_params:
        Number of Bernstein coefficients ``p`` (length of ``θ_b``).
    X:
        Shift design, shape ``(n, q_d)``; ``q_d = 0`` for none.
    X_scale:
        Scaling design, shape ``(n, q_s)`` with ``q_s ≥ 1``.
    offset:
        Per-observation offset, shape ``(n,)`` (zeros for none).

    Notes
    -----
    ``h(y|x_i) = h_0(y)·f_i + x_d,i·β + offset_i`` and ``h_0`` is
    non-decreasing with ``h_0(a) = θ_b[0]``, so ``g_i(θ) = h(a|x_i)`` is the
    minimum of ``h(·|x_i)`` over the support (ADR 0005 Decision 1).  Rows of
    the Jacobian ``J = ∂g/∂θ``:

    ==========================  ===============================
    Block                       Entry
    ==========================  ===============================
    ``∂g_i/∂θ_b[0]``            ``f_i``
    ``∂g_i/∂θ_b[k]``, k ≥ 1     ``0``
    ``∂g_i/∂β``                 ``x_d,i``
    ``∂g_i/∂γ``                 ``0.5 · θ_b[0] · f_i · x_s,i``
    ==========================  ===============================

    At ``γ = 0`` (``f_i = 1``) the rows are the unscaled linear support rows
    ``θ_b[0] + x_d,i·β ≥ −offset_i`` (ADR 0005 Decision 2, simplification 2).
    ``f_i`` is evaluated as ``np.exp(0.5 · x_s,i · γ)``, exactly as the scaled
    likelihood evaluates it.

    Examples
    --------
    >>> import numpy as np
    >>> g = build_scaled_support_constraint(
    ...     2, np.array([[1.0]]), np.array([[2.0]]), np.array([0.5])
    ... )
    >>> values, jac = g(np.array([1.0, 2.0, -1.0, 0.0]))  # [θ_b | β | γ]
    >>> values.tolist()  # 1·e⁰ − 1 + 0.5
    [0.5]
    >>> jac.tolist()
    [[1.0, 0.0, 1.0, 1.0]]
    """

    n_params: int
    X: NDArray[np.float64]
    X_scale: NDArray[np.float64]
    offset: NDArray[np.float64]

    @property
    def n_rows(self) -> int:
        """Number of support rows ``n`` (training observations)."""
        return int(self.X_scale.shape[0])

    @property
    def q_d(self) -> int:
        """Number of shift columns (length of ``β``)."""
        return int(self.X.shape[1])

    @property
    def q_s(self) -> int:
        """Number of scaling columns (length of ``γ``)."""
        return int(self.X_scale.shape[1])

    @property
    def n_total(self) -> int:
        """Full parameter length ``p + q_d + q_s``."""
        return self.n_params + self.q_d + self.q_s

    def __call__(
        self, theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the support rows and their Jacobian at ``theta``.

        Parameters
        ----------
        theta:
            Parameter vector ``[θ_b | β | γ]``, shape ``(n_total,)``.

        Returns
        -------
        values : NDArray of shape ``(n,)``
            ``g_i(θ) = θ_b[0]·f_i + x_d,i·β + offset_i``.
        jac : NDArray of shape ``(n, n_total)``
            ``∂g/∂θ`` (see the class Notes).

        Raises
        ------
        ValueError
            If ``theta`` does not have shape ``(n_total,)``.
        """
        th = np.asarray(theta, dtype=np.float64)
        if th.shape != (self.n_total,):
            raise ValueError(
                f"theta must have shape ({self.n_total},) = p + q_d + q_s; got "
                f"shape {th.shape}."
            )
        p, q_d = self.n_params, self.q_d
        # θ_b[0]: lower-end coefficient, β: shift block, γ: scaling block.
        t0 = th[0]
        beta = th[p : p + q_d]
        gamma = th[p + q_d :]
        f = np.exp(0.5 * (self.X_scale @ gamma))  # f_i = exp(0.5·x_s,i·γ)
        values = t0 * f + self.X @ beta + self.offset
        jac = np.zeros((self.n_rows, self.n_total), dtype=np.float64)
        jac[:, 0] = f
        jac[:, p : p + q_d] = self.X
        jac[:, p + q_d :] = (0.5 * t0 * f)[:, None] * self.X_scale
        return np.asarray(values, dtype=np.float64), jac


def build_scaled_support_constraint(
    n_params: int,
    X: NDArray[np.float64] | None,
    X_scale: NDArray[np.float64],
    offset: NDArray[np.float64] | None = None,
) -> ScaledSupportConstraint:
    """Build the nonlinear exponential support rows under a scaling term.

    Parameters
    ----------
    n_params:
        Number of Bernstein coefficients ``p`` (``BernsteinBasis.order + 1``).
    X:
        Shift design, shape ``(n, q_d)``, or ``None`` for no shift block.
    X_scale:
        Scaling design, shape ``(n, q_s)`` with ``q_s ≥ 1``.
    offset:
        Per-observation offset, shape ``(n,)``, or ``None`` (zero).

    Returns
    -------
    ScaledSupportConstraint
        Callable ``θ ↦ (g(θ), J(θ))`` over ``θ = [θ_b | β | γ]`` of length
        ``p + q_d + q_s``; the design is copied.

    Raises
    ------
    TypeError
        If ``n_params`` is not an integer.
    ValueError
        If ``n_params < 1``; if ``X_scale`` is not 2-D with at least one
        column; if ``X`` is not 2-D or its row count differs from
        ``X_scale``'s; if ``offset`` does not have one entry per row; or if
        ``X``, ``X_scale`` or ``offset`` contain NaN or infinite entries.

    Notes
    -----
    ``g_i(θ) = θ_b[0]·exp(0.5·x_s,i·γ) + x_d,i·β + offset_i``: ADR 0005
    Decision 2's exact nonlinear constraint, offset included as in the
    #160 amendment.  See :class:`ScaledSupportConstraint` for the Jacobian.

    Examples
    --------
    >>> import numpy as np
    >>> g = build_scaled_support_constraint(
    ...     3, np.array([[0.5], [1.0]]), np.array([[0.0], [1.0]])
    ... )
    >>> g.n_rows, g.n_total
    (2, 5)
    """
    if isinstance(n_params, bool) or not isinstance(n_params, (int, np.integer)):
        raise TypeError(
            f"n_params must be an int (number of Bernstein coefficients); got "
            f"{type(n_params).__name__}."
        )
    if n_params < 1:
        raise ValueError(f"n_params must be at least 1; got n_params={n_params}.")
    xs = np.array(X_scale, dtype=np.float64, copy=True)
    if xs.ndim != 2 or xs.shape[1] == 0:
        raise ValueError(
            "X_scale must be a 2-D scaling design of shape (n, q_s) with "
            f"q_s >= 1; got shape {xs.shape}."
        )
    n = xs.shape[0]
    if X is None:
        xd = np.zeros((n, 0), dtype=np.float64)
    else:
        xd = np.array(X, dtype=np.float64, copy=True)
        if xd.ndim != 2:
            raise ValueError(f"X must be 2-D (n, q_d); got shape {xd.shape}.")
        if xd.shape[0] != n:
            raise ValueError(
                f"X and X_scale must have the same number of rows; got "
                f"{xd.shape[0]} and {n}."
            )
    if offset is None:
        off = np.zeros(n, dtype=np.float64)
    else:
        off = np.array(offset, dtype=np.float64, copy=True).ravel()
        if off.shape[0] != n:
            raise ValueError(
                f"offset must have one entry per row of X_scale ({n}); got "
                f"{off.shape[0]}."
            )
    for name, arr in (("X", xd), ("X_scale", xs), ("offset", off)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{name} must be finite to define the exponential support rows "
                "h(a|x_i) >= 0; got NaN or infinite entries."
            )
    return ScaledSupportConstraint(n_params=int(n_params), X=xd, X_scale=xs, offset=off)


def _scaled_support_is_nonlinear(
    n_params: int,
    total: int,
    q_s: int,
    X: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
) -> bool:
    """Whether the scaled exponential support rule needs the nonlinear block.

    ``θ_b[0]·f_i + x_d,i·β ≥ −offset_i`` collapses to the linear row
    ``θ_b[0] ≥ 0`` only with no shift block and a zero (or absent) offset
    (ADR 0005 Decision 2, simplification 1; issue #114).  Shape errors are
    left to the builders.
    """
    q_d = total - n_params - q_s
    if X is not None and np.ndim(X) == 2:
        q_d = max(q_d, int(np.shape(X)[1]))
    if q_d > 0:
        return True
    if offset is None:
        return False
    off = np.asarray(offset, dtype=np.float64)
    return bool(off.size > 0 and np.any(off != 0.0))


def _validate_support_offset(
    offset: NDArray[np.float64] | None, n_rows: int | None
) -> NDArray[np.float64] | None:
    """Validate the per-observation offset that shifts the support rows.

    Parameters
    ----------
    offset:
        Per-observation offset, any shape that ravels to ``(n,)``, or ``None``.
    n_rows:
        Required length (number of rows of ``X``), or ``None`` when there are
        no covariate rows to match (the single no-covariate support row), in
        which case any non-empty length is accepted.

    Returns
    -------
    NDArray of shape ``(n,)`` or None
        The offset as a flat ``float64`` array, or ``None``.

    Raises
    ------
    ValueError
        If ``offset`` has the wrong length, is empty, or is not finite.
    """
    if offset is None:
        return None
    off = np.asarray(offset, dtype=np.float64).ravel()
    if n_rows is not None and off.shape[0] != n_rows:
        raise ValueError(
            f"offset must have one entry per row of X ({n_rows}); got {off.shape[0]}."
        )
    if off.shape[0] == 0:
        raise ValueError(
            "offset must have at least one entry when nonneg_lower=True; "
            "got an empty array."
        )
    if not np.all(np.isfinite(off)):
        raise ValueError(
            "offset must be finite to define the exponential support rows "
            "h(a|x_i) >= 0; got NaN or infinite entries."
        )
    return off


def _single_row_rhs(offset: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """Right-hand side of the no-covariate row ``θ_b[0] ≥ max_i(−offset_i)``.

    Parameters
    ----------
    offset:
        Validated per-observation offset, or ``None`` (right-hand side 0).

    Returns
    -------
    NDArray of shape ``(1,)``
    """
    if offset is None:
        return np.zeros(1, dtype=np.float64)
    return np.array([float(np.max(-offset))], dtype=np.float64)


def _scaled_support_row(
    n_params: int,
    total: int,
    *,
    q_s: int,
    X: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exponential support row under a scaling term with no shift block.

    Parameters
    ----------
    n_params:
        Number of Bernstein coefficients.
    total:
        Full parameter length ``n_params + q_d + q_s``.
    q_s:
        Number of scaling columns (``q_s > 0``).
    X:
        Shift design, shape ``(n, q_d)``, or ``None``.  Must have no columns.
    offset:
        Per-observation offset, or ``None``.  Must be identically zero.

    Returns
    -------
    rows : NDArray of shape ``(1, total)``
        The row ``[1, 0, …, 0 | 0_{q_s}]``.
    rhs : NDArray of shape ``(1,)``
        Zero.

    Raises
    ------
    NotImplementedError
        If a shift block is present (``q_d > 0``) or the offset is non-zero.
    ValueError
        If ``X`` is not 2-D, or ``offset`` is empty, not finite, or does not
        match the rows of ``X``.

    Notes
    -----
    ``h(a|x_i) = θ_b[0]·f_i + x_i·β + offset_i`` with
    ``f_i = exp(0.5·x_s,i·γ) > 0``.  With ``q_d = 0`` and ``offset ≡ 0`` the
    rule ``h(a|x_i) ≥ 0`` is ``θ_b[0]·f_i ≥ 0 ⇔ θ_b[0] ≥ 0`` for every ``i``
    (ADR 0005 Decision 2, simplification 1).  Otherwise the row is non-linear
    in ``γ`` (``θ_b[0] ≥ −offset_i / f_i``, or the β term), which needs the
    nonlinear inequality block of ADR 0005 Decision 2.  The auglag builder
    :func:`build_constraint_matrices` provides it (``X_scale=``, issue #115)
    and never reaches this helper in that case; the SLSQP / trust-constr
    builder :func:`build_constraints` has no nonlinear form yet (issue #116).
    """
    q_d = total - n_params - q_s
    n_rows: int | None = None
    if X is not None:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X_arr.shape}")
        q_d = max(q_d, X_arr.shape[1])
        n_rows = X_arr.shape[0]
    if q_d > 0:
        raise NotImplementedError(
            "The exponential support row with scaling= and a shift block "
            f"(q_d={q_d}) is θ_b[0]·exp(0.5·x_s·γ) + x·β ≥ 0, which is "
            "non-linear in γ; only the auglag builder (build_constraint_matrices "
            "with X_scale=) supports it, SLSQP / trust-constr are issue #116 "
            "(see docs/adr/0005-exponential-scaling-constraint.md, Decision 2)."
        )
    off = _validate_support_offset(offset, n_rows)
    if off is not None and np.any(off != 0.0):
        raise NotImplementedError(
            "The exponential support row with scaling= and a non-zero offset "
            "is θ_b[0] ≥ −offset_i·exp(−0.5·x_s,i·γ), which is non-linear in "
            "γ; only the auglag builder (build_constraint_matrices with "
            "X_scale=) supports it, SLSQP / trust-constr are issue #116 (see "
            "docs/adr/0005-exponential-scaling-constraint.md, Decision 2)."
        )
    rows = np.zeros((1, total), dtype=np.float64)
    rows[0, 0] = 1.0
    return rows, np.zeros(1, dtype=np.float64)


def _assemble_constraint_arrays(
    n_params: int,
    *,
    total_params: int | None,
    nonneg_lower: bool,
    X: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None = None,
    q_s: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Build the inequality rows shared by both constraint builders.

    Returns the padded monotonicity matrix ``D`` (shape
    ``(n_params-1, total)``), the optional support-feasibility rows
    ``support_rows`` (shape ``(n_rows, total)`` or ``None``) and their
    right-hand side ``support_rhs`` (shape ``(n_rows,)`` or ``None``).  This is the
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
    offset:
        Per-observation offset, shape ``(n,)``, or ``None``.  Only consulted
        when ``nonneg_lower=True``; it sets the support right-hand side (see
        Notes).
    q_s:
        Number of scaling-design columns (``γ`` block, trailing in
        ``[θ_b | β | γ]``).  Only consulted when ``nonneg_lower=True``; see
        Notes.

    Returns
    -------
    D : NDArray of shape ``(n_params-1, total)``
        Padded forward-difference (monotonicity) matrix.
    support_rows : NDArray of shape ``(n_rows, total)`` or None
        Support-feasibility inequality rows, or ``None`` when
        ``nonneg_lower=False``.
    support_rhs : NDArray of shape ``(n_rows,)`` or None
        Right-hand side ``b`` of ``support_rows @ θ ≥ b``: ``−offset`` per
        observation (zeros without an offset), or ``None`` when
        ``nonneg_lower=False``.

    Raises
    ------
    TypeError
        If ``q_s`` is not an integer.
    ValueError
        If ``X`` has invalid shape, if ``X`` columns do not match
        ``total_params - n_params``, if ``nonneg_lower=True`` with ``X``
        but ``total_params`` is omitted, if ``offset`` is empty, not
        finite, or does not have one entry per row of ``X``, or if ``q_s``
        is negative or exceeds ``total_params - n_params``.
    NotImplementedError
        If ``nonneg_lower=True`` and ``q_s > 0`` with a shift block or a
        non-zero offset (the support row is then non-linear in ``γ``).

    Notes
    -----
    ADR 0005 Decision 1 requires ``h(a|x_i) ≥ 0`` for every training row,
    with ``h`` the transformation as the likelihood evaluates it, offset
    included: ``h(a|x_i) = θ_b[0] + x_i·β + offset_i`` (issue #160).  With
    covariates that is one row ``[1, 0, …, 0 | x_i]·θ ≥ −offset_i`` per
    observation.  Without covariates the rows all read ``θ_b[0] ≥ −offset_i``
    and collapse to the single row ``θ_b[0] ≥ max_i(−offset_i)``.

    With a scaling term (``q_s > 0``) the rule reads
    ``θ_b[0]·f_i + x_i·β ≥ −offset_i`` with ``f_i = exp(0.5·x_s,i·γ) > 0``,
    which is non-linear in ``γ``.  Without a shift block and with a zero
    offset it collapses to ``θ_b[0]·f_i ≥ 0 ⇔ θ_b[0] ≥ 0`` (ADR 0005
    Decision 2, simplification 1): the single row ``[1, 0, …, 0 | 0_{q_s}]``
    (issue #114).  Every other scaled case needs the nonlinear block of ADR
    0005 Decision 2 and raises ``NotImplementedError``.
    """
    total = total_params if total_params is not None else n_params
    if isinstance(q_s, bool) or not isinstance(q_s, (int, np.integer)):
        raise TypeError(
            f"q_s must be an int (number of scaling columns); got {type(q_s).__name__}."
        )
    if q_s < 0 or q_s > total - n_params:
        raise ValueError(
            f"q_s must lie in [0, total_params - n_params] = "
            f"[0, {total - n_params}]; got q_s={q_s}."
        )
    D = MonotonicityConstraint(n_params).as_matrix()  # (n_params-1, n_params)

    if total > n_params:
        D = np.hstack([D, np.zeros((D.shape[0], total - n_params))])

    # Support-feasibility rows.  No covariates: one row [1, 0, ..., 0].
    # With covariates: n rows [1, 0, ..., 0 | X_i].
    support_rows: NDArray[np.float64] | None = None
    support_rhs: NDArray[np.float64] | None = None
    if nonneg_lower and q_s > 0:
        support_rows, support_rhs = _scaled_support_row(
            n_params, total, q_s=q_s, X=X, offset=offset
        )
    elif nonneg_lower:
        if X is None:
            if total > n_params:
                raise ValueError(
                    "X must be provided when nonneg_lower=True and "
                    "total_params > n_params so support-feasibility "
                    "constraints can include the regression coefficients."
                )
            support_rows = np.zeros((1, total), dtype=np.float64)
            support_rows[0, 0] = 1.0
            support_rhs = _single_row_rhs(_validate_support_offset(offset, None))
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
            off = _validate_support_offset(offset, X_arr.shape[0])
            if X_arr.shape[1] == 0:
                support_rows = np.zeros((1, total), dtype=np.float64)
                support_rows[0, 0] = 1.0
                support_rhs = _single_row_rhs(off)
            else:
                n_obs = X_arr.shape[0]
                support_rows = np.zeros((n_obs, total), dtype=np.float64)
                support_rows[:, 0] = 1.0
                support_rows[:, n_params:] = X_arr
                support_rhs = (
                    -off if off is not None else np.zeros(n_obs, dtype=np.float64)
                )

    return D, support_rows, support_rhs


def _interaction_support_rows(
    basis: "InteractionBasis",
    q_s: int,
    X: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exponential support rows ``(a(a) ⊗ b(x_i))ᵀ vec(Θ) ≥ −offset_i``.

    See :func:`build_constraint_matrices_interaction` (Notes) for the math.

    Returns
    -------
    rows : NDArray of shape ``(n, p*q)``
        One row per training observation.
    rhs : NDArray of shape ``(n,)``
        ``−offset`` or zeros.

    Raises
    ------
    ValueError
        If ``X`` is missing, ``q_s > 0``, or ``offset`` has the wrong length.
    """
    if X is None:
        raise ValueError(
            "X must be provided when nonneg_lower=True on an InteractionBasis: "
            "the exponential support rows h(a|x_i) >= 0 need one b(x_i) per "
            "training observation."
        )
    if q_s > 0:
        raise ValueError(
            "nonneg_lower=True requires q_s=0 on an InteractionBasis; got "
            f"q_s={q_s}. The exponential support row is non-linear in γ "
            "(see docs/adr/0003-scaling-with-interaction.md, Decision 5)."
        )
    x = basis._coerce_x(np.asarray(X, dtype=np.float64))
    n = x.shape[0]
    a = basis.support[0]
    rows = basis.evaluate(np.full(n, a, dtype=np.float64), x)  # (n, p*q)
    if offset is None:
        return rows, np.zeros(n, dtype=np.float64)
    off = np.asarray(offset, dtype=np.float64).ravel()
    if off.shape[0] != n:
        raise ValueError(
            f"offset must have one entry per row of X ({n}); got {off.shape[0]}."
        )
    return rows, -off


def build_constraint_matrices_interaction(
    basis: "InteractionBasis",
    q_s: int = 0,
    *,
    nonneg_lower: bool = False,
    X: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
) -> ConstraintMatrices:
    """Build constraint matrices for an :class:`~mltpy.basis.InteractionBasis`.

    Constructs the Kronecker inequality ``(D ⊗ I_q) @ vec(Θ) ≥ 0`` that
    enforces column-wise monotonicity: ``D @ Θ[:, j] ≥ 0`` for every
    column ``j = 0, …, q-1``.  With ``nonneg_lower=True`` it appends the
    exponential-link support rows ``h(a|x_i) ≥ 0`` (ADR 0005, Decision 1).

    Parameters
    ----------
    basis:
        The :class:`~mltpy.basis.InteractionBasis` to build constraints for.
    q_s:
        Number of scaling-design columns (``γ`` block).  When non-zero, the
        parameter vector is ``[vec_C(Θ) | γ]`` and ``q_s`` zero columns are
        appended to every constraint matrix so their width matches; ``γ`` is
        unconstrained (ADR 0003 Decision 3).
    nonneg_lower:
        If ``True``, append one support row per training observation so the
        transformation stays in the exponential support ``[0, ∞)`` at the
        lower support end ``a`` of the y-basis.  Required for
        ``base_distribution="exponential"``.
    X:
        Training covariates for the x-basis, shape ``(n,)`` or ``(n, 1)``.
        Required when ``nonneg_lower=True``, ignored otherwise.
    offset:
        Optional per-observation offset, shape ``(n,)``.  Only consulted when
        ``nonneg_lower=True``; it moves the support-row right-hand side to
        ``−offset_i``.

    Returns
    -------
    ConstraintMatrices
        ``A_ineq`` has shape ``((p-1)*q + n_s, p*q + q_s)``, where ``n_s`` is
        ``n`` with ``nonneg_lower=True`` and 0 otherwise.  ``b_ineq`` is zero
        on the monotonicity rows and ``−offset`` (or zero) on the support
        rows.  ``C_eq`` is a zero-row matrix (no equality constraints).
        ``d_eq`` is a zero-length array.

    Raises
    ------
    ValueError
        If the x-basis type is not supported for closed-form constraints
        (already checked at :class:`~mltpy.basis.InteractionBasis`
        construction time, so this branch is a safety net); if
        ``nonneg_lower=True`` without ``X``, with ``q_s > 0``, or with an
        ``offset`` whose length differs from the number of rows of ``X``.

    Notes
    -----
    **Support rows (ADR 0005).**  Every column of ``Θ`` is non-decreasing
    and ``b(x)`` is non-negative, so ``h(·|x)`` is non-decreasing in ``y``
    and ``min_y h(y|x_i) = h(a|x_i)``.  The rule ``h(a|x_i) ≥ 0`` is the
    linear row

        (a(a) ⊗ b(x_i))ᵀ vec(Θ) ≥ −offset_i,

    which for a Bernstein y-basis (``a(a) = e_0``) reads
    ``Θ[0, :] · b(x_i) ≥ −offset_i``.  Because ``b(x)`` is non-linear in
    ``x``, these rows do not imply ``h(a|x) ≥ 0`` between training points;
    ``predict`` checks new rows instead.

    The rule is not available with ``q_s > 0``: under scaling the row is
    non-linear in ``γ`` (ADR 0003 Decision 5).

    Examples
    --------
    >>> import numpy as np
    >>> from mltpy.basis import BernsteinBasis, InteractionBasis
    >>> ib = InteractionBasis(
    ...     BernsteinBasis(order=2, support=(0.0, 1.0)),
    ...     BernsteinBasis(order=1, support=(0.0, 1.0)),
    ... )
    >>> cm = build_constraint_matrices_interaction(
    ...     ib, nonneg_lower=True, X=np.array([0.25, 0.75])
    ... )
    >>> cm.A_ineq[-2:]
    array([[0.75, 0.25, 0.  , 0.  , 0.  , 0.  ],
           [0.25, 0.75, 0.  , 0.  , 0.  , 0.  ]])
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
    total = p * q + q_s

    if p >= 2:
        D = np.diff(np.eye(p), axis=0)  # (p-1, p)
        A_kron = np.kron(D, np.eye(q)).astype(np.float64)  # ((p-1)*q, p*q)
        if q_s > 0:
            # γ is unconstrained: pad with q_s zero columns to match the
            # [vec_C(Θ) | γ] layout (ADR 0003 Decision 3).
            A_ineq = np.hstack(
                [A_kron, np.zeros((A_kron.shape[0], q_s), dtype=np.float64)]
            )
        else:
            A_ineq = A_kron
    else:
        A_ineq = np.zeros((0, total), dtype=np.float64)
    b_ineq = np.zeros(A_ineq.shape[0], dtype=np.float64)

    if nonneg_lower:
        support_rows, support_rhs = _interaction_support_rows(basis, q_s, X, offset)
        A_ineq = np.vstack([A_ineq, support_rows])
        b_ineq = np.concatenate([b_ineq, support_rhs])

    return ConstraintMatrices(
        A_ineq=A_ineq,
        b_ineq=b_ineq,
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
    offset: NDArray[np.float64] | None = None,
    q_s: int = 0,
    X_scale: NDArray[np.float64] | None = None,
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
          enforcing ``θ_b[0] ≥ max_i(−offset_i)`` (``θ_b[0] ≥ 0`` without an
          offset).
        * With covariates: one row ``[1, 0, …, 0 | X_i]`` per observation,
          enforcing ``θ_b[0] + X_i · β ≥ −offset_i``.

    X:
        Covariate matrix, shape ``(n, q)``.  Only consulted when
        ``nonneg_lower=True``; ``q`` must equal ``total_params - n_params``.
    offset:
        Optional per-observation offset, shape ``(n,)``.  Only consulted when
        ``nonneg_lower=True``; it moves the support-row right-hand side to
        ``−offset_i`` so the rule covers ``h(a|x_i) = θ_b[0] + X_i·β +
        offset_i`` as the likelihood evaluates it (ADR 0005 Decision 1,
        issue #160).  ``None`` leaves the right-hand side at zero.
    q_s:
        Number of scaling-design columns (the trailing ``γ`` block of
        ``[θ_b | β | γ]``), included in ``total_params``.  Monotonicity
        rows leave ``γ`` free.  With ``nonneg_lower=True`` and ``q_s > 0``
        the support rule ``θ_b[0]·exp(0.5·x_s,i·γ) + X_i·β ≥ −offset_i`` is
        linear only without a shift block and with a zero offset, where it
        collapses to the single row ``[1, 0, …, 0 | 0_{q_s}]``, i.e.
        ``θ_b[0] ≥ 0`` (ADR 0005 Decision 2, simplification 1; issue #114).
        Otherwise the rows are returned as the nonlinear block
        ``nonlinear_ineq`` built from ``X_scale`` (issue #115).
    X_scale:
        Scaling design, shape ``(n, q_s)``.  Required when
        ``nonneg_lower=True`` and ``q_s > 0`` with a shift block or a
        non-zero offset; it defines the nonlinear support rows
        ``θ_b[0]·exp(0.5·x_s,i·γ) + X_i·β ≥ −offset_i``
        (:func:`build_scaled_support_constraint`).  Ignored otherwise.

    Returns
    -------
    ConstraintMatrices
        ``A_ineq`` is the padded forward-difference matrix D (shape
        ``(n_params-1, total_params)``) optionally followed by the linear
        support rows; ``b_ineq`` is zero on the monotonicity rows and
        ``−offset`` (or zero) on the support rows.  ``C_eq`` has 0, 1, or 2
        rows depending on which of ``lower``/``upper`` are provided; ``d_eq``
        carries the corresponding right-hand-side values.  On the scaled
        nonlinear case ``A_ineq`` holds the monotonicity rows only and
        ``nonlinear_ineq`` the support rows; ``nonlinear_ineq`` is ``None``
        otherwise.

    Raises
    ------
    TypeError
        If ``q_s`` is not an integer.
    ValueError
        If ``X`` has invalid shape, if ``X`` columns do not match
        ``total_params - n_params``, if ``nonneg_lower=True`` with ``X``
        but ``total_params`` is omitted, if ``offset`` is empty, not
        finite, or does not have one entry per row of ``X``, or if ``q_s``
        is negative or exceeds ``total_params - n_params``; or, on the scaled
        nonlinear case, if ``X_scale`` is missing, does not have ``q_s``
        columns, or does not match the rows of ``X`` / ``offset`` (see
        :func:`build_scaled_support_constraint`).

    Examples
    --------
    >>> import numpy as np
    >>> cm = build_constraint_matrices(
    ...     3,
    ...     total_params=4,
    ...     nonneg_lower=True,
    ...     X=np.array([[0.0], [1.0]]),
    ...     offset=np.array([0.5, -0.25]),
    ... )
    >>> cm.b_ineq.tolist()
    [0.0, 0.0, -0.5, 0.25]
    """
    total = total_params if total_params is not None else n_params
    nonlinear: ScaledSupportConstraint | None = None
    if (
        nonneg_lower
        and isinstance(q_s, (int, np.integer))
        and not isinstance(q_s, bool)
        and q_s > 0
        and _scaled_support_is_nonlinear(n_params, total, q_s, X, offset)
    ):
        # Issue #115: θ_b[0]·f_i + X_i·β ≥ −offset_i is non-linear in γ; the
        # linear block keeps monotonicity only and the support rows become
        # the nonlinear block (ADR 0005 Decision 3).
        nonlinear = _scaled_support_block(n_params, total, q_s, X, X_scale, offset)
        D, support_rows, support_rhs = _assemble_constraint_arrays(
            n_params, total_params=total_params, nonneg_lower=False, X=None, q_s=q_s
        )
    else:
        D, support_rows, support_rhs = _assemble_constraint_arrays(
            n_params,
            total_params=total_params,
            nonneg_lower=nonneg_lower,
            X=X,
            offset=offset,
            q_s=q_s,
        )

    b_ineq = np.zeros(D.shape[0], dtype=np.float64)
    if support_rows is not None and support_rhs is not None:
        A_ineq = np.vstack([D, support_rows])
        b_ineq = np.concatenate([b_ineq, support_rhs])
    else:
        A_ineq = D

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
        b_ineq=b_ineq,
        C_eq=C_eq,
        d_eq=d_eq,
        nonlinear_ineq=nonlinear,
    )


def _scaled_support_block(
    n_params: int,
    total: int,
    q_s: int,
    X: NDArray[np.float64] | None,
    X_scale: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
) -> ScaledSupportConstraint:
    """Validate the layout and build the nonlinear scaled support block.

    Raises
    ------
    ValueError
        If ``q_s`` exceeds ``total − n_params``, ``X_scale`` is missing or
        does not have ``q_s`` columns, ``X`` columns do not match
        ``total − n_params − q_s``, or the design is otherwise invalid (see
        :func:`build_scaled_support_constraint`).
    """
    if q_s > total - n_params:
        raise ValueError(
            f"q_s must lie in [0, total_params - n_params] = "
            f"[0, {total - n_params}]; got q_s={q_s}."
        )
    if X_scale is None:
        raise ValueError(
            "X_scale must be provided when nonneg_lower=True and q_s > 0 with "
            "a shift block or a non-zero offset: the exponential support rows "
            "θ_b[0]·exp(0.5·x_s,i·γ) + X_i·β >= −offset_i depend on the scaling "
            "design (docs/adr/0005-exponential-scaling-constraint.md, Decision 2)."
        )
    xs = np.asarray(X_scale)
    if xs.ndim != 2 or xs.shape[1] != q_s:
        raise ValueError(
            f"X_scale must have shape (n, q_s) with q_s={q_s}; got shape {xs.shape}."
        )
    q_d = total - n_params - q_s
    if X is not None:
        x_arr = np.asarray(X)
        if x_arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {x_arr.shape}")
        if x_arr.shape[1] != q_d:
            raise ValueError(
                f"X has {x_arr.shape[1]} columns but total_params - n_params - "
                f"q_s = {q_d}"
            )
    elif q_d > 0:
        raise ValueError(
            "X must be provided when nonneg_lower=True and total_params > "
            "n_params + q_s so the support rows can include the shift block."
        )
    return build_scaled_support_constraint(n_params, X, X_scale, offset)


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
    offset: NDArray[np.float64] | None = None,
    q_s: int = 0,
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
    offset: NDArray[np.float64] | None = None,
    q_s: int = 0,
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
    offset: NDArray[np.float64] | None = None,
    q_s: int = 0,
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

        With an ``offset`` the rule covers ``h(a|x_i) = theta_b[0] + X_i·β
        + offset_i`` (ADR 0005 Decision 1, issue #160): the rows become
        ``theta_b[0] + X_i · β >= −offset_i``, and without covariates the
        single row becomes ``theta_b[0] >= max_i(−offset_i)``.  The SLSQP
        dict returns ``S @ theta − b`` and the ``LinearConstraint`` uses
        ``lb = b``.

        Kept distinct from ``lower`` because ``lower`` is an *equality* that
        pins ``theta[0]``.
    X:
        Optional covariate matrix of shape ``(n, q)``.  Only consulted when
        ``nonneg_lower=True`` — see above.  ``q`` must equal
        ``total_params - n_params``.
    offset:
        Optional per-observation offset, shape ``(n,)``.  Only consulted when
        ``nonneg_lower=True`` — see above.  ``None`` leaves the support
        right-hand side at zero.
    q_s:
        Number of scaling-design columns (trailing ``γ`` block), included in
        ``total_params``.  See :func:`build_constraint_matrices`: with
        ``nonneg_lower=True`` and ``q_s > 0`` the support rule collapses to
        ``θ_b[0] ≥ 0`` without a shift block and with a zero offset.

    Raises
    ------
    TypeError
        If ``q_s`` is not an integer.
    ValueError
        If ``X`` has invalid shape, if ``X`` columns do not match
        ``total_params - n_params``, if ``nonneg_lower=True`` with ``X``
        but ``total_params`` is omitted, if ``offset`` is empty, not
        finite, or does not have one entry per row of ``X``, or if ``q_s``
        is negative or exceeds ``total_params - n_params``.
    NotImplementedError
        If ``nonneg_lower=True`` and ``q_s > 0`` together with a shift block
        or a non-zero offset (ADR 0005 Decision 2).

    Returns
    -------
    list[dict] for ``solver="slsqp"``, list[LinearConstraint] for
    ``solver="trust-constr"``.

    Examples
    --------
    >>> import numpy as np
    >>> cons = build_constraints(
    ...     3,
    ...     solver="slsqp",
    ...     total_params=4,
    ...     nonneg_lower=True,
    ...     X=np.array([[0.0], [1.0]]),
    ...     offset=np.array([0.5, -0.25]),
    ... )
    >>> theta = np.array([0.25, 0.5, 1.0, 0.0])  # [θ_b | β]
    >>> cons[1]["fun"](theta).tolist()  # h(a|x_i) = θ_b[0] + x_i·β + offset_i
    [0.75, 0.0]
    """
    total = total_params if total_params is not None else n_params
    D, support_rows, support_rhs = _assemble_constraint_arrays(
        n_params,
        total_params=total_params,
        nonneg_lower=nonneg_lower,
        X=X,
        offset=offset,
        q_s=q_s,
    )

    if solver == "slsqp":
        result: list[dict[str, Any]] = [
            {
                "type": "ineq",
                "fun": lambda theta, _D=D: _D @ theta,
                "jac": lambda theta, _D=D: _D,
            }
        ]
        if support_rows is not None and support_rhs is not None:
            result.append(
                {
                    "type": "ineq",
                    "fun": lambda theta, _S=support_rows, _b=support_rhs: (
                        _S @ theta - _b
                    ),
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
        if support_rows is not None and support_rhs is not None:
            lcs.append(LinearConstraint(support_rows, lb=support_rhs, ub=np.inf))
        if lower is not None or upper is not None:
            bc = BoundaryConstraint(
                n_params, lower=lower, upper=upper, total_params=total
            )
            lcs.append(bc.as_LinearConstraint())
        return lcs
