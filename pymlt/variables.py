"""Variable types and censoring classes for conditional transformation models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray


class CensoringType(Enum):
    """Censoring regime for a dataset passed to the log-likelihood."""
    NONE     = auto()   # exact observations
    LEFT     = auto()   # left-censored:   actual value < observed threshold
    RIGHT    = auto()   # right-censored:  actual value > observed threshold
    INTERVAL = auto()   # interval-censored: lower <= actual value <= upper


@dataclass
class NumericVariable:
    """Continuous numeric variable with a defined support interval.

    Parameters
    ----------
    name:
        Variable name.
    support:
        Closed interval (a, b) defining the domain for the Bernstein basis.
        Must satisfy a < b.
    bounds:
        Optional narrower observation bounds (a, b) with a < b.
        If None, `support` is used as bounds.
    log_first:
        If True, apply log-transform before Bernstein evaluation.
    """

    name: str
    support: tuple[float, float]
    bounds: Optional[tuple[float, float]] = None
    log_first: bool = False

    def __post_init__(self) -> None:
        if self.support[0] >= self.support[1]:
            raise ValueError(
                f"support must satisfy a < b, got {self.support}"
            )
        if self.bounds is not None and self.bounds[0] >= self.bounds[1]:
            raise ValueError(
                f"bounds must satisfy a < b, got {self.bounds}"
            )


@dataclass
class OrderedVariable:
    """Ordinal categorical variable.

    Parameters
    ----------
    name:
        Variable name.
    levels:
        Ordered list of level labels, from lowest to highest.
        At least two levels are required.
    """

    name: str
    levels: list[str]

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError(
                f"OrderedVariable requires at least 2 levels, got {len(self.levels)}"
            )
        if len(self.levels) != len(set(self.levels)):
            raise ValueError("OrderedVariable levels must be unique")

    @property
    def n_levels(self) -> int:
        return len(self.levels)


@dataclass
class SurvivalVariable:
    """Non-negative continuous variable for survival/time-to-event data.

    Parameters
    ----------
    name:
        Variable name.
    support:
        Interval (a, b) with a >= 0 and a < b.  Defaults to (0, inf).
    """

    name: str
    support: tuple[float, float] = field(default_factory=lambda: (0.0, float("inf")))

    def __post_init__(self) -> None:
        if self.support[0] < 0.0:
            raise ValueError(
                f"SurvivalVariable support must start at >= 0, got {self.support[0]}"
            )
        if self.support[0] >= self.support[1]:
            raise ValueError(
                f"support must satisfy a < b, got {self.support}"
            )


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
    trunc_lower: Optional[NDArray[np.float64]] = None
    trunc_upper: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        self.exact = np.asarray(self.exact, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)

        n = len(self.exact)
        if len(self.lower) != n or len(self.upper) != n:
            raise ValueError(
                "exact, lower, and upper must all have the same length"
            )
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
        return (
            np.isnan(self.exact)
            & np.isfinite(self.lower)
            & ~np.isfinite(self.upper)
        )

    @property
    def is_left_censored_mask(self) -> NDArray[np.bool_]:
        """Boolean mask: True where observation is left-censored."""
        return (
            np.isnan(self.exact)
            & ~np.isfinite(self.lower)
            & np.isfinite(self.upper)
        )

    @property
    def is_interval_censored_mask(self) -> NDArray[np.bool_]:
        """Boolean mask: True where observation is interval-censored."""
        return (
            np.isnan(self.exact)
            & np.isfinite(self.lower)
            & np.isfinite(self.upper)
        )

    @property
    def n_exact(self) -> int:
        return int(self.is_exact_mask.sum())

    @property
    def n_censored(self) -> int:
        return self.n - self.n_exact
