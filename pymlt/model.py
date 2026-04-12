"""Public API for conditional transformation models.

Users import exclusively from this module::

    import pymlt
    model = pymlt.MLT(order=6, support=(0, 100))
    model.fit(y)
    cdf = model.predict(y_new, what="distribution")

Classes
-------
ConditionalTransformationModel
    Base class for all transformation models.
MLT
    Most Likely Transformation — convenience subclass with sensible defaults.
"""

from __future__ import annotations

import warnings
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from pymlt.basis import BernsteinBasis
from pymlt.likelihood import BaseDistribution, _get_dist, log_likelihood
from pymlt.optimizer import OptimizationResult, OptimizerConfig, optimize
from pymlt.variables import CensoredData, CensoringType

# ---------------------------------------------------------------------------
# Exceptions and warnings
# ---------------------------------------------------------------------------


class NotFittedError(ValueError):
    """Raised when a method that requires a fitted model is called before fit()."""


class ConvergenceWarning(UserWarning):
    """Raised when the optimiser fails to converge within the allowed restarts."""


# ---------------------------------------------------------------------------
# Valid predict targets
# ---------------------------------------------------------------------------

_VALID_WHAT = ("distribution", "density", "quantile", "hazard")

# Small epsilon used for bracket safety in brentq
_BRENTQ_EPS = 1e-10


# ---------------------------------------------------------------------------
# ConditionalTransformationModel
# ---------------------------------------------------------------------------


class ConditionalTransformationModel:
    """Base class for conditional transformation models.

    Fits a monotone transformation h(y|x) parametrised as a Bernstein
    polynomial such that h(y|x) follows a standard distribution.

    Parameters
    ----------
    basis:
        :class:`~pymlt.basis.BernsteinBasis` defining the response
        transformation.
    censoring:
        Censoring type of the response data.  Defaults to
        :attr:`~pymlt.variables.CensoringType.NONE`.
    optimizer_config:
        Optimisation settings.  If ``None``, defaults from
        :class:`~pymlt.optimizer.OptimizerConfig` are used.

    Attributes
    ----------
    theta_ : NDArray or None
        Fitted parameter vector ``[theta_basis | beta]``.  ``None`` before
        :meth:`fit`.
    result_ : OptimizationResult or None
        Full result object from the last :meth:`fit` call.  ``None`` before
        :meth:`fit`.
    is_fitted_ : bool
        Whether :meth:`fit` has been called successfully.
    """

    def __init__(
        self,
        basis: BernsteinBasis,
        censoring: CensoringType = CensoringType.NONE,
        optimizer_config: OptimizerConfig | None = None,
        base_distribution: BaseDistribution = "normal",
    ) -> None:
        _get_dist(base_distribution)  # raises ValueError for unsupported values
        self.basis = basis
        self.censoring = censoring
        self.optimizer_config = optimizer_config
        self.base_distribution = base_distribution

        # State — set by fit()
        self.theta_: NDArray[np.float64] | None = None
        self.result_: OptimizationResult | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Raise :exc:`NotFittedError` if the model has not been fitted yet."""
        if not self.is_fitted_:
            raise NotFittedError("Modell wurde noch nicht gefittet. Rufe fit(y) auf.")

    def _validate_input(
        self,
        y: NDArray[np.float64] | CensoredData,
        X: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64] | CensoredData, NDArray[np.float64] | None]:
        """Coerce and validate ``y`` and ``X`` before fitting.

        Parameters
        ----------
        y:
            Response.  ``pd.Series`` and array-likes are coerced to
            ``np.ndarray``.  ``CensoredData`` is passed through.
        X:
            Optional covariate matrix.

        Returns
        -------
        (y_clean, X_clean)

        Raises
        ------
        ValueError
            If observations fall outside ``basis.support``, if ``X`` shape
            does not match ``y``, or if censored bounds are inconsistent.
        """
        a, b = self.basis.support

        if isinstance(y, CensoredData):
            # Check finite lower/upper bounds lie within support
            fin_lo = y.lower[np.isfinite(y.lower)]
            fin_hi = y.upper[np.isfinite(y.upper)]
            for vals, name in [(fin_lo, "lower"), (fin_hi, "upper")]:
                if len(vals) and (vals.min() < a or vals.max() > b):
                    raise ValueError(
                        f"CensoredData.{name} enthält Werte außerhalb support "
                        f"[{a}, {b}]. Passe BernsteinBasis(support=...) an."
                    )
            n = y.n
            y_clean: NDArray[np.float64] | CensoredData = y
        else:
            # Coerce without importing pandas: np.asarray handles pd.Series
            y_arr = np.asarray(y, dtype=float).ravel()
            if y_arr.min() < a or y_arr.max() > b:
                raise ValueError(
                    f"y enthält Werte außerhalb support [{a}, {b}]. "
                    f"Passe BernsteinBasis(support=...) an. "
                    f"(min={y_arr.min():.4g}, max={y_arr.max():.4g})"
                )
            n = len(y_arr)
            y_clean = y_arr

        X_clean: NDArray[np.float64] | None = None
        if X is not None:
            X_clean = np.asarray(X, dtype=float)
            if X_clean.ndim == 1:
                X_clean = X_clean[:, None]
            if X_clean.shape[0] != n:
                raise ValueError(
                    f"X hat {X_clean.shape[0]} Zeilen, y hat aber {n} "
                    "Beobachtungen. Beide müssen übereinstimmen."
                )

        return y_clean, X_clean

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y: NDArray[np.float64] | CensoredData,
        X: NDArray[np.float64] | None = None,
    ) -> "ConditionalTransformationModel":
        """Fit the transformation model by maximum likelihood.

        Parameters
        ----------
        y:
            Response observations.  Must lie within ``basis.support``.
            Accepts ``np.ndarray``, ``pd.Series``, or
            :class:`~pymlt.variables.CensoredData`.
        X:
            Optional covariate matrix of shape ``(n, q)``.  If given, the
            last ``q`` entries of ``theta_`` are regression coefficients.

        Returns
        -------
        self
            Returns itself for method chaining::

                cdf = model.fit(y).predict(y, what="distribution")

        Raises
        ------
        ValueError
            If ``y`` contains values outside ``basis.support``.
        """
        y_clean, X_clean = self._validate_input(y, X)

        result = optimize(
            self.basis,
            y_clean,
            X=X_clean,
            censoring=self.censoring,
            config=self.optimizer_config,
            base_distribution=self.base_distribution,
        )

        if not result.converged:
            warnings.warn(
                f"Optimierung nicht konvergiert nach {result.n_restarts} "
                f"Restarts. Solver-Meldung: {result.solver_message}. "
                "Ergebnis ist das beste gefundene, aber möglicherweise "
                "nicht das MLE.",
                ConvergenceWarning,
                stacklevel=2,
            )

        self.theta_ = result.theta
        self.result_ = result
        self.is_fitted_ = True
        return self

    def predict(
        self,
        y_new: NDArray[np.float64],
        X_new: NDArray[np.float64] | None = None,
        what: Literal["distribution", "density", "quantile", "hazard"] = "distribution",
    ) -> NDArray[np.float64]:
        """Compute model predictions at new observations.

        Parameters
        ----------
        y_new:
            For ``what ∈ {"distribution", "density", "hazard"}``: response
            values in ``basis.support``.
            For ``what="quantile"``: probabilities in ``(0, 1)``.
        X_new:
            Optional covariate matrix of shape ``(m, q)``.
        what:
            Type of prediction:

            * ``"distribution"`` — CDF: F(h(y|x))
            * ``"density"``      — PDF: f(h(y|x)) · h'(y|x)
            * ``"quantile"``     — Quantile via numerical inversion (brentq)
            * ``"hazard"``       — Hazard: f(h)/F̄(h); only for
              ``censoring=RIGHT``

        Returns
        -------
        NDArray of shape ``(m,)``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``what`` is not one of the valid options.
        NotImplementedError
            If ``what="hazard"`` and ``censoring`` is not ``RIGHT``.

        Examples
        --------
        >>> model = MLT(order=4, support=(0, 1)).fit(y)
        >>> cdf = model.predict(y_new, what="distribution")
        >>> q50 = model.predict(np.array([0.5]), what="quantile")
        """
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError(
                "Modellparameter (theta_) fehlen unerwartet nach dem Fitten."
            )

        if what not in _VALID_WHAT:
            raise ValueError(f"what={what!r} ist ungültig. Erlaubt: {_VALID_WHAT}")

        p = self.basis.order + 1
        theta_b = self.theta_[:p]

        y_arr = np.asarray(y_new, dtype=float).ravel()
        X_arr: NDArray[np.float64] | None = None
        if X_new is not None:
            X_arr = np.asarray(X_new, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]

        if what == "quantile":
            return self._predict_quantile(y_arr, theta_b)

        if what == "hazard":
            if self.censoring is not CensoringType.RIGHT:
                raise NotImplementedError(
                    "what='hazard' ist nur für censoring=CensoringType.RIGHT "
                    "implementiert."
                )

        # Evaluate transformation and its derivative
        B = self.basis.evaluate(y_arr)  # (m, p)
        D = self.basis.derivative(y_arr, order=1)  # (m, p)
        h = B @ theta_b  # (m,)
        hp = D @ theta_b  # (m,)

        if X_arr is not None and len(self.theta_) > p:
            beta = self.theta_[p:]
            h = h + X_arr @ beta

        dist = _get_dist(self.base_distribution)
        if what == "distribution":
            return cast(NDArray[np.float64], dist.cdf(h))
        elif what == "density":
            return cast(NDArray[np.float64], dist.pdf(h) * np.maximum(hp, 0.0))
        else:  # hazard
            return cast(
                NDArray[np.float64], dist.pdf(h) / np.maximum(dist.sf(h), 1e-300)
            )

    def _predict_quantile(
        self, probs: NDArray[np.float64], theta_b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Numerically invert h(q) = F⁻¹(p) via brentq for each p.

        F⁻¹ is the quantile function (``dist.ppf``) of the base distribution
        (normal or logistic depending on ``self.base_distribution``).

        Parameters
        ----------
        probs:
            Probabilities in (0, 1).
        theta_b:
            Bernstein coefficient vector of length ``order + 1``.

        Returns
        -------
        NDArray of same length as ``probs``.
        """
        a, b = self.basis.support
        # Clip z into the range that brentq can bracket
        z_min = float(theta_b[0]) + _BRENTQ_EPS
        z_max = float(theta_b[-1]) - _BRENTQ_EPS

        def _h_scalar(q: float) -> float:
            return float(self.basis.evaluate(np.array([q]))[0] @ theta_b)

        dist = _get_dist(self.base_distribution)
        quantiles = np.empty(len(probs))
        for i, p in enumerate(probs):
            z = float(np.clip(dist.ppf(p), z_min, z_max))
            quantiles[i] = brentq(
                lambda q, z=z: _h_scalar(q) - z,
                a,
                b,
                xtol=1e-6,
                full_output=False,
            )
        return cast(NDArray[np.float64], quantiles)

    def score(
        self,
        y: NDArray[np.float64] | CensoredData,
        X: NDArray[np.float64] | None = None,
    ) -> float:
        """Log-likelihood at the fitted parameters (sklearn-compatible).

        Higher is better; this is NOT the negative log-likelihood.

        Parameters
        ----------
        y:
            Response observations.
        X:
            Optional covariate matrix.

        Returns
        -------
        float  (log-likelihood value)

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError(
                "Modellparameter (theta_) fehlen unerwartet nach dem Fitten."
            )
        y_clean, X_clean = self._validate_input(y, X)
        return log_likelihood(
            self.theta_,
            self.basis,
            y_clean,
            X_clean,
            self.censoring,
            base_distribution=self.base_distribution,
        )

    def simulate(
        self,
        n: int,
        X: NDArray[np.float64] | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Draw samples from the fitted model via the quantile transformation.

        Samples ``u ~ Uniform(0, 1)`` and returns
        ``predict(u, X, what="quantile")``.

        Parameters
        ----------
        n:
            Number of samples to draw.
        X:
            Optional covariate matrix of shape ``(n, q)``.
        random_state:
            Seed or :class:`numpy.random.Generator` for reproducibility.

        Returns
        -------
        NDArray of shape ``(n,)`` with values in ``basis.support``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        self._check_is_fitted()

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        # Clip away from 0/1 to avoid Φ⁻¹(0) = -inf and Φ⁻¹(1) = +inf
        u = np.clip(rng.uniform(size=n), 1e-10, 1 - 1e-10)
        return self.predict(u, X_new=X, what="quantile")

    def __repr__(self) -> str:
        name = type(self).__name__
        order = self.basis.order
        censoring = self.censoring.name
        if self.is_fitted_:
            if self.result_ is None:
                raise RuntimeError(
                    "Ergebnis (result_) fehlt unerwartet nach dem Fitten."
                )
            ll = self.result_.log_likelihood
            return (
                f"{name}(order={order}, censoring={censoring}, "
                f"fitted=True, ll={ll:.2f})"
            )
        return f"{name}(order={order}, censoring={censoring}, fitted=False)"


# ---------------------------------------------------------------------------
# MLT — convenience subclass
# ---------------------------------------------------------------------------


class MLT(ConditionalTransformationModel):
    """Most Likely Transformation — convenience interface.

    A :class:`ConditionalTransformationModel` with an explicit ``order`` and
    ``support`` parameter instead of a pre-built ``BernsteinBasis``.

    Parameters
    ----------
    order:
        Polynomial degree of the Bernstein basis.  Defaults to 6.
    support:
        Closed interval ``(a, b)`` with ``a < b``.  Defaults to ``(0, 1)``.
    censoring:
        Censoring type of the response data.
    optimizer_config:
        Optimisation settings.

    Examples
    --------
    >>> model = MLT(order=6, support=(0, 100))
    >>> model.fit(y)
    >>> cdf = model.predict(y_new, what="distribution")
    """

    def __init__(
        self,
        order: int = 6,
        support: tuple[float, float] = (0.0, 1.0),
        censoring: CensoringType = CensoringType.NONE,
        optimizer_config: OptimizerConfig | None = None,
        base_distribution: BaseDistribution = "normal",
    ) -> None:
        basis = BernsteinBasis(order=order, support=support)
        super().__init__(
            basis=basis,
            censoring=censoring,
            optimizer_config=optimizer_config,
            base_distribution=base_distribution,
        )
        # Store for repr
        self._order = order
        self._support = support

    def __repr__(self) -> str:
        censoring = self.censoring.name
        if self.is_fitted_:
            if self.result_ is None:
                raise RuntimeError(
                    "Ergebnis (result_) fehlt unerwartet nach dem Fitten."
                )
            ll = self.result_.log_likelihood
            return (
                f"MLT(order={self._order}, support={self._support}, "
                f"censoring={censoring}, fitted=True, ll={ll:.2f})"
            )
        return (
            f"MLT(order={self._order}, support={self._support}, "
            f"censoring={censoring}, fitted=False)"
        )
