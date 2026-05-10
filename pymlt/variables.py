"""Variable types and censoring classes for conditional transformation models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class CensoringType(Enum):
    """Censoring regime for a dataset passed to the log-likelihood."""

    NONE = auto()  # exact observations
    LEFT = auto()  # left-censored:   actual value < observed threshold
    RIGHT = auto()  # right-censored:  actual value > observed threshold
    INTERVAL = auto()  # interval-censored: lower <= actual value <= upper


@dataclass
class CensoredData:
    """Encodes n observations with optional censoring and truncation.

    For observation i exactly one censoring pattern is valid:

    * **Exact**:           ``exact[i]`` is finite
    * **Right-censored**:  ``exact[i]`` is NaN, ``lower[i]`` finite, ``upper[i]`` = +inf
    * **Left-censored**:   ``exact[i]`` is NaN, ``lower[i]`` = -inf, ``upper[i]`` finite
    * **Interval-censored**: ``exact[i]`` is NaN, both bounds finite

    Truncation bounds constrain the *observable* range: only observations
    inside ``[trunc_lower[i], trunc_upper[i]]`` can appear in the sample.

    Parameters
    ----------
    exact:
        Length-n array.  Use ``np.nan`` for censored observations.
    lower:
        Length-n array of lower bounds.  Use ``-np.inf`` for left-censored.
    upper:
        Length-n array of upper bounds.  Use ``+np.inf`` for right-censored.
    trunc_lower:
        Optional length-n array of left truncation points.
    trunc_upper:
        Optional length-n array of right truncation points.
    """

    exact: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    trunc_lower: NDArray[np.float64] | None = None
    trunc_upper: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        self.exact = np.asarray(self.exact, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)

        n = len(self.exact)
        if len(self.lower) != n or len(self.upper) != n:
            raise ValueError("exact, lower, and upper must all have the same length")
        if np.any(self.lower > self.upper):
            raise ValueError("lower must be <= upper for every observation")
        if self.trunc_lower is not None:
            self.trunc_lower = np.asarray(self.trunc_lower, dtype=float)
            if len(self.trunc_lower) != n:
                raise ValueError("trunc_lower must have the same length as exact")
        if self.trunc_upper is not None:
            self.trunc_upper = np.asarray(self.trunc_upper, dtype=float)
            if len(self.trunc_upper) != n:
                raise ValueError("trunc_upper must have the same length as exact")
        if self.trunc_lower is not None and self.trunc_upper is not None:
            if np.any(self.trunc_lower > self.trunc_upper):
                raise ValueError(
                    "trunc_lower must be <= trunc_upper for every observation"
                )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_exact(cls, y: NDArray[np.float64]) -> CensoredData:
        """All observations exact (no censoring)."""
        y = np.asarray(y, dtype=float)
        return cls(exact=y.copy(), lower=y.copy(), upper=y.copy())

    @classmethod
    def right_censored(
        cls, y: NDArray[np.float64], censored: NDArray[np.bool_]
    ) -> CensoredData:
        """Right-censored data.

        Parameters
        ----------
        y:
            Observed value (exact value or censoring threshold).
        censored:
            Boolean array.  ``True`` means the actual event time is
            *above* ``y`` (only a lower bound is known).
        """
        y = np.asarray(y, dtype=float)
        censored = np.asarray(censored, dtype=bool)
        if len(y) != len(censored):
            raise ValueError("y and censored must have the same length")
        exact = np.where(censored, np.nan, y)
        lower = y.copy()
        upper = np.where(censored, np.inf, y)
        return cls(exact=exact, lower=lower, upper=upper)

    @classmethod
    def left_censored(
        cls, y: NDArray[np.float64], censored: NDArray[np.bool_]
    ) -> CensoredData:
        """Left-censored data.

        Parameters
        ----------
        y:
            Observed value (exact value or censoring threshold).
        censored:
            Boolean array.  ``True`` means the actual value is
            *below* ``y`` (only an upper bound is known).
        """
        y = np.asarray(y, dtype=float)
        censored = np.asarray(censored, dtype=bool)
        if len(y) != len(censored):
            raise ValueError("y and censored must have the same length")
        exact = np.where(censored, np.nan, y)
        lower = np.where(censored, -np.inf, y)
        upper = y.copy()
        return cls(exact=exact, lower=lower, upper=upper)

    @classmethod
    def interval_censored(
        cls, lower: NDArray[np.float64], upper: NDArray[np.float64]
    ) -> CensoredData:
        """All observations interval-censored with known bounds [lower, upper]."""
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if len(lower) != len(upper):
            raise ValueError("lower and upper must have the same length")
        exact = np.full(len(lower), np.nan)
        return cls(exact=exact, lower=lower, upper=upper)

    @classmethod
    def left_truncated(
        cls,
        y: NDArray[np.float64],
        trunc_lower: NDArray[np.float64],
        censored: NDArray[np.bool_] | None = None,
    ) -> CensoredData:
        """Left-truncated (delayed-entry) data, optionally with right censoring.

        Mirrors R's ``Surv(start, stop, event)`` counting-process encoding
        used by the survival package: each observation is only at risk
        starting from ``trunc_lower[i]``.  When ``censored`` is given, the
        same boolean convention as :meth:`right_censored` applies — ``True``
        means the actual event time is *above* ``y[i]``.

        Parameters
        ----------
        y:
            Observed value (exact event time, or right-censoring threshold).
        trunc_lower:
            Length-n array of left-truncation points (delayed-entry times).
        censored:
            Optional boolean array of right-censoring indicators.  ``None``
            (default) treats all observations as exactly observed.
        """
        y = np.asarray(y, dtype=float)
        trunc_lower = np.asarray(trunc_lower, dtype=float)
        if len(trunc_lower) != len(y):
            raise ValueError("trunc_lower must have the same length as y")
        if censored is None:
            exact = y.copy()
            lower = y.copy()
            upper = y.copy()
        else:
            censored = np.asarray(censored, dtype=bool)
            if len(censored) != len(y):
                raise ValueError("y and censored must have the same length")
            exact = np.where(censored, np.nan, y)
            lower = y.copy()
            upper = np.where(censored, np.inf, y)
        return cls(exact=exact, lower=lower, upper=upper, trunc_lower=trunc_lower)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of observations."""
        return len(self.exact)

    @property
    def is_exact_mask(self) -> NDArray[np.bool_]:
        """Boolean mask: True where observation is exact."""
        return cast(NDArray[np.bool_], ~np.isnan(self.exact))

    @property
    def is_right_censored_mask(self) -> NDArray[np.bool_]:
        """Boolean mask: True where observation is right-censored."""
        return np.isnan(self.exact) & np.isfinite(self.lower) & ~np.isfinite(self.upper)

    @property
    def is_left_censored_mask(self) -> NDArray[np.bool_]:
        """Boolean mask: True where observation is left-censored."""
        return np.isnan(self.exact) & ~np.isfinite(self.lower) & np.isfinite(self.upper)

    @property
    def is_interval_censored_mask(self) -> NDArray[np.bool_]:
        """Boolean mask: True where observation is interval-censored."""
        return np.isnan(self.exact) & np.isfinite(self.lower) & np.isfinite(self.upper)

    @property
    def n_exact(self) -> int:
        return int(self.is_exact_mask.sum())

    @property
    def n_censored(self) -> int:
        return self.n - self.n_exact


# ---------------------------------------------------------------------------
# Ordered categorical response
# ---------------------------------------------------------------------------


@dataclass
class OrderedVariable:
    """Ordered categorical response with K levels and K-1 transformation cutpoints.

    Used by :class:`pymlt.tram.Polr` (proportional-odds ordinal regression).
    A level-``k`` observation (``1 <= k <= K``) is mapped to interval-censored
    bounds on a synthetic integer cut scale::

        level 1   → (-∞, 1]
        level k   → (k-1, k]      for 1 < k < K
        level K   → (K-1, +∞)

    Combined with :class:`pymlt.basis.OrdinalBasis`, the cut position ``k``
    selects one of ``K-1`` Bernstein-like coefficients ``θ_k`` so that
    ``h(y_k) = θ_k`` exactly.

    Parameters
    ----------
    levels:
        Tuple of ordered category labels (any hashable values).  Must contain
        at least two distinct levels.
    """

    levels: tuple[Any, ...] = field()

    def __post_init__(self) -> None:
        self.levels = tuple(self.levels)
        if len(self.levels) < 2:
            raise ValueError(
                f"OrderedVariable requires at least 2 levels, got {len(self.levels)}"
            )
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(
                f"OrderedVariable levels must be unique, got {self.levels}"
            )
        # Build a label → 1-based index lookup (1..K) for fast encoding.
        self._label_to_code = {label: i + 1 for i, label in enumerate(self.levels)}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def K(self) -> int:  # noqa: N802 — match standard ordinal-regression notation
        """Number of levels."""
        return len(self.levels)

    # ------------------------------------------------------------------
    # Constructors / conversions
    # ------------------------------------------------------------------

    @classmethod
    def from_labels(
        cls,
        y: Sequence[Any] | NDArray[Any],
        levels: Sequence[Any] | None = None,
    ) -> tuple[OrderedVariable, CensoredData]:
        """Coerce raw observations into ``(OrderedVariable, CensoredData)``.

        Level inference order:

        1. If ``levels`` is given explicitly, use it.
        2. Else if ``y`` is a pandas ordered Categorical, use ``y.cat.categories``.
        3. Else: sorted unique values (deterministic ordering).

        Validates that every observation lies in the resolved level set.

        Parameters
        ----------
        y:
            Length-n sequence of category labels (any hashable type).
        levels:
            Optional ordered tuple of all valid labels.  Overrides automatic
            inference.

        Returns
        -------
        (variable, censored_data)
            ``variable`` carries the level vocabulary; ``censored_data`` has
            one row per observation with synthetic integer-cut bounds suitable
            for :class:`~pymlt.basis.OrdinalBasis` and the interval-censored
            likelihood path.

        Raises
        ------
        ValueError
            If ``levels`` is empty or not unique, or if any observation in
            ``y`` is missing from the resolved level set.
        """
        # Resolve the ordered level vocabulary.  Look for pandas Categorical
        # (object exposes ``categories``) directly, or a pandas Series with
        # a ``.cat`` accessor.  Fall back to sorted unique values otherwise.
        if levels is not None:
            resolved_levels = tuple(levels)
        else:
            categories = None
            cat = getattr(y, "cat", None)
            if cat is not None and hasattr(cat, "categories"):
                categories = cat.categories
            elif hasattr(y, "categories"):
                categories = y.categories
            if categories is not None:
                resolved_levels = tuple(categories)
            else:
                # Avoid np.asarray inference, which coerces mixed types to a
                # common string dtype (e.g. [1, "a"] -> ["1", "a"]) and
                # silently corrupts integer / boolean labels.  Materialise as
                # a Python list of original objects, then sort-unique.  An
                # ndarray's .tolist() converts numpy scalars back to Python
                # scalars (preserving identity for object-dtype arrays).
                if isinstance(y, np.ndarray):
                    seq = y.tolist()
                else:
                    seq = list(y)
                unique = list(dict.fromkeys(seq))
                try:
                    resolved_levels = tuple(sorted(unique))
                except TypeError as exc:
                    raise ValueError(
                        "Cannot infer level order from observations of mixed "
                        f"types: {unique!r}. Pass `levels=` explicitly."
                    ) from exc

        variable = cls(levels=resolved_levels)
        codes = variable.encode(y)
        K = variable.K
        n = codes.shape[0]
        lower = np.where(codes == 1, -np.inf, codes.astype(np.float64) - 1.0)
        upper = np.where(codes == K, np.inf, codes.astype(np.float64))
        cd = CensoredData(exact=np.full(n, np.nan), lower=lower, upper=upper)
        return variable, cd

    def encode(self, y: Sequence[Any] | NDArray[Any]) -> NDArray[np.intp]:
        """Map labels to 1-based integer codes ``1..K``.

        Parameters
        ----------
        y:
            Sequence of category labels.

        Returns
        -------
        NDArray[np.intp]
            Integer codes of shape ``(n,)``.

        Raises
        ------
        ValueError
            If any label is not in :attr:`levels`.
        """
        # Materialise as a Python list of original objects.  np.asarray on
        # mixed types coerces to a string dtype (e.g. [1, "a"] -> ["1", "a"])
        # which would corrupt integer / boolean labels and break the dict
        # lookup against explicit mixed `levels`.  list() on a pandas
        # Categorical / Series yields the labels (not the underlying codes);
        # an ndarray's .tolist() preserves Python scalars.
        if hasattr(y, "cat") or hasattr(y, "categories"):
            seq = list(y)
        elif isinstance(y, np.ndarray):
            seq = y.tolist()
        else:
            seq = list(y)
        codes = np.empty(len(seq), dtype=np.intp)
        for i, label in enumerate(seq):
            try:
                codes[i] = self._label_to_code[label]
            except KeyError as exc:
                raise ValueError(
                    f"Observation {label!r} at position {i} is not in the "
                    f"OrderedVariable levels {self.levels!r}."
                ) from exc
        return codes

    def decode(self, codes: NDArray[np.intp]) -> NDArray[Any]:
        """Inverse of :meth:`encode` — map ``1..K`` codes back to labels.

        Parameters
        ----------
        codes:
            Integer codes of shape ``(n,)`` with values in ``{1, ..., K}``.

        Returns
        -------
        NDArray
            Labels in their original dtype (object array for non-numeric).

        Raises
        ------
        ValueError
            If any code is outside the valid range, or if any floating-point
            code is not integer-valued (e.g. ``1.7``).
        TypeError
            If ``codes`` has a non-numeric dtype (object, complex, …).
        """
        raw = np.asarray(codes)
        if np.issubdtype(raw.dtype, np.integer):
            codes_arr = raw.astype(np.intp)
        elif np.issubdtype(raw.dtype, np.floating):
            fractional = raw[raw != np.floor(raw)]
            if fractional.size:
                raise ValueError(
                    f"decode() received non-integer float codes: "
                    f"{fractional.tolist()!r}. Codes must be whole-number "
                    f"values in {{1, ..., {self.K}}}."
                )
            codes_arr = raw.astype(np.intp)
        else:
            raise TypeError(
                f"codes must have an integer or floating dtype, got {raw.dtype!r}."
            )
        if codes_arr.size and (codes_arr.min() < 1 or codes_arr.max() > self.K):
            raise ValueError(
                f"Codes must be in [1, {self.K}], got "
                f"min={int(codes_arr.min())}, max={int(codes_arr.max())}."
            )
        labels = np.array(self.levels, dtype=object)
        return labels[codes_arr - 1]
