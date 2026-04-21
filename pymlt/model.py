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

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import chi2, norm

from pymlt.basis import BernsteinBasis
from pymlt.likelihood import (
    _H_CLIP,
    BaseDistribution,
    _get_dist,
    _neg_score,
    log_likelihood,
)
from pymlt.likelihood import (
    hessian as _hessian,
)
from pymlt.likelihood import (
    score_matrix as _score_matrix,
)
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

_VALID_WHAT = (
    "trafo",
    "distribution",
    "logdistribution",
    "survivor",
    "logsurvivor",
    "density",
    "logdensity",
    "hazard",
    "loghazard",
    "cumhazard",
    "logcumhazard",
    "odds",
    "logodds",
    "quantile",
)

_VALID_CONFBAND_WHAT = (
    "trafo",
    "distribution",
    "survivor",
    "density",
    "hazard",
)

# Small epsilon used for bracket safety in brentq
_BRENTQ_EPS = 1e-10

# Floor for log(h') to avoid log(0) at boundaries where monotonicity is marginal
_LOG_HP_FLOOR = np.finfo(np.float64).tiny


def _extract_feature_names(X: object) -> list[str] | None:
    """Extract column names from a pandas DataFrame, else ``None``.

    Kept as a free function so no pandas import is required at module load
    time — we only touch pandas attributes by duck-typing.
    """
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    try:
        return [str(c) for c in columns]
    except Exception:
        return None


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
        """Fitted parameter vector ``[theta_basis | beta]``.  ``None`` before
        :meth:`fit`."""

        self.result_: OptimizationResult | None = None
        """Full result object from the last :meth:`fit` call.  ``None`` before
        :meth:`fit`."""

        self.is_fitted_: bool = False
        """Whether :meth:`fit` has been called successfully."""

        self.n_obs_: int | None = None
        """Number of observations used in :meth:`fit`.  For
        :class:`~pymlt.variables.CensoredData`, this is ``y.n``; otherwise
        ``len(y)``.  ``None`` before :meth:`fit`."""

        self.n_free_params_: int | None = None
        """Number of free parameters in the fitted model — equal to
        ``len(theta_)`` (Bernstein coefficients plus optional regression
        coefficients).  The monotonicity constraint ``D @ theta_b >= 0`` is
        an inequality and does not reduce the parameter count.  ``None``
        before :meth:`fit`."""

        self.hessian_: NDArray[np.float64] | None = None
        """Observed information matrix — analytical Hessian of the *negative*
        log-likelihood evaluated at :attr:`theta_`.  Shape ``(p+q, p+q)``.
        Computed eagerly at the end of :meth:`fit`.  ``None`` before
        :meth:`fit`."""

        self.feature_names_in_: list[str] | None = None
        """Names of the covariate columns supplied to :meth:`fit`, if any.
        Populated from a ``pandas.DataFrame`` column index when available,
        otherwise ``["X1", "X2", ...]``.  ``None`` when the model was fit
        without covariates."""

        # Score matrix — computed eagerly at the end of fit().
        self._estfun_cache_: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Raise :exc:`NotFittedError` if the model has not been fitted yet."""
        if not self.is_fitted_:
            raise NotFittedError("Model has not been fitted yet. Call fit(y) first.")

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
                        f"CensoredData.{name} contains values outside support "
                        f"[{a}, {b}]. Adjust BernsteinBasis(support=...) accordingly."
                    )
            n = y.n
            y_clean: NDArray[np.float64] | CensoredData = y
        else:
            # Coerce without importing pandas: np.asarray handles pd.Series
            y_arr = np.asarray(y, dtype=float).ravel()
            if y_arr.min() < a or y_arr.max() > b:
                raise ValueError(
                    f"y contains values outside support [{a}, {b}]. "
                    f"Adjust BernsteinBasis(support=...) accordingly. "
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
                    f"X has {X_clean.shape[0]} rows, but y has {n} "
                    "observations. Both must match."
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
        feature_names = _extract_feature_names(X)
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
                f"Optimization did not converge after {result.n_restarts}"
                f"restarts. Solver message: {result.solver_message}."
                "The result is the best found, but may not be the MLE.",
                ConvergenceWarning,
                stacklevel=2,
            )

        self.theta_ = result.theta
        self.result_ = result
        self.is_fitted_ = True
        self.n_obs_ = (
            int(y_clean.n) if isinstance(y_clean, CensoredData) else len(y_clean)
        )
        self.n_free_params_ = int(result.theta.size)

        # Feature names for the covariate block of theta_.
        if X_clean is not None:
            q = X_clean.shape[1]
            if feature_names is None or len(feature_names) != q:
                feature_names = [f"X{j + 1}" for j in range(q)]
            self.feature_names_in_ = feature_names
        else:
            self.feature_names_in_ = None

        # Observed information and score matrix — computed eagerly so that
        # later mutations of the caller's ``y``/``X`` cannot affect
        # ``vcov()`` or ``estfun()`` results.  Failures here indicate a real
        # modelling problem (degenerate basis, constraint-binding fit);
        # surface them.
        self.hessian_ = _hessian(
            self.theta_,
            self.basis,
            y_clean,
            X_clean,
            self.censoring,
            base_distribution=self.base_distribution,
        )
        self._estfun_cache_ = _score_matrix(
            self.theta_,
            self.basis,
            y_clean,
            X_clean,
            self.censoring,
            base_distribution=self.base_distribution,
        )
        return self

    def predict(
        self,
        y_new: NDArray[np.float64],
        X_new: NDArray[np.float64] | None = None,
        what: Literal[
            "trafo",
            "distribution",
            "logdistribution",
            "survivor",
            "logsurvivor",
            "density",
            "logdensity",
            "hazard",
            "loghazard",
            "cumhazard",
            "logcumhazard",
            "odds",
            "logodds",
            "quantile",
        ] = "distribution",
    ) -> NDArray[np.float64]:
        """Compute model predictions at new observations.

        Parameters
        ----------
        y_new:
            For ``what="quantile"``: probabilities in ``(0, 1)``.
            For all other ``what``: response values in ``basis.support``.
        X_new:
            Optional covariate matrix of shape ``(m, q)``.
        what:
            Type of prediction.  Let ``h = h(y|x)`` and ``h' = ∂h/∂y``; ``F``,
            ``S``, ``f`` denote the base distribution's CDF, survivor, and PDF.

            * ``"trafo"``           — Transformation ``h(y|x)``
            * ``"distribution"``    — CDF: ``F(h)``
            * ``"logdistribution"`` — ``log F(h)``
            * ``"survivor"``        — Survivor: ``S(h) = 1 − F(h)``
            * ``"logsurvivor"``     — ``log S(h)``
            * ``"density"``         — PDF: ``f(h) · h'``
            * ``"logdensity"``      — ``log f(h) + log h'``
            * ``"hazard"``          — Hazard: ``f(h) · h' / S(h)``
            * ``"loghazard"``       — ``log f(h) + log h' − log S(h)``
            * ``"cumhazard"``       — Cumulative hazard: ``−log S(h)``
            * ``"logcumhazard"``    — ``log(−log S(h))``
            * ``"odds"``            — ``F(h) / S(h)``
            * ``"logodds"``         — ``log F(h) − log S(h)``
            * ``"quantile"``        — Quantile via numerical inversion (brentq)

        Returns
        -------
        NDArray of shape ``(m,)``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``what`` is not one of the valid options.

        Notes
        -----
        Log-scale variants use ``dist.logcdf``/``logsf``/``logpdf`` directly
        and are numerically stable in the tails where the primal quantities
        would under- or overflow.

        Examples
        --------
        >>> model = MLT(order=4, support=(0, 1)).fit(y)
        >>> cdf = model.predict(y_new, what="distribution")
        >>> q50 = model.predict(np.array([0.5]), what="quantile")
        """
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError(
                "Model parameters (theta_) are unexpectedly missing after fitting."
            )

        if what not in _VALID_WHAT:
            raise ValueError(f"what={what!r} is invalid. Allowed: {_VALID_WHAT}")

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

        # Evaluate transformation and its derivative
        B = self.basis.evaluate(y_arr)  # (m, p)
        D = self.basis.derivative(y_arr, order=1)  # (m, p)
        h = B @ theta_b  # (m,)
        hp = D @ theta_b  # (m,)

        if X_arr is not None and len(self.theta_) > p:
            beta = self.theta_[p:]
            h = h + X_arr @ beta

        dist = _get_dist(self.base_distribution)
        hp_pos = np.maximum(hp, 0.0)
        log_hp = np.log(np.maximum(hp, _LOG_HP_FLOOR))

        if what == "trafo":
            return h
        if what == "distribution":
            return cast(NDArray[np.float64], dist.cdf(h))
        if what == "logdistribution":
            return cast(NDArray[np.float64], dist.logcdf(h))
        if what == "survivor":
            return cast(NDArray[np.float64], dist.sf(h))
        if what == "logsurvivor":
            return cast(NDArray[np.float64], dist.logsf(h))
        if what == "density":
            return cast(NDArray[np.float64], dist.pdf(h) * hp_pos)
        if what == "logdensity":
            return cast(NDArray[np.float64], dist.logpdf(h) + log_hp)
        if what == "hazard":
            return cast(
                NDArray[np.float64],
                dist.pdf(h) * hp_pos / np.maximum(dist.sf(h), 1e-300),
            )
        if what == "loghazard":
            return cast(NDArray[np.float64], dist.logpdf(h) + log_hp - dist.logsf(h))
        if what == "cumhazard":
            return cast(NDArray[np.float64], -dist.logsf(h))
        if what == "logcumhazard":
            return cast(NDArray[np.float64], np.log(-dist.logsf(h)))
        if what == "odds":
            return cast(NDArray[np.float64], np.exp(dist.logcdf(h) - dist.logsf(h)))
        # logodds
        return cast(NDArray[np.float64], dist.logcdf(h) - dist.logsf(h))

    def _predict_quantile(
        self, probs: NDArray[np.float64], theta_b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Numerically invert h(q) = F⁻¹(p) via brentq for each p.

        F⁻¹ is the quantile function (``dist.ppf``) of the base distribution
        selected by ``self.base_distribution`` (see ``likelihood._get_dist``
        for the full list of supported distributions).

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
                "Model parameters (theta_) are unexpectedly missing after fitting."
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

    def vcov(self) -> NDArray[np.float64]:
        """Asymptotic variance–covariance matrix of :attr:`theta_`.

        Returns the inverse of the observed information matrix
        :attr:`hessian_` (Hessian of the *negative* log-likelihood at the
        MLE).  Under standard regularity conditions, this is a consistent
        estimator of the asymptotic covariance of the maximum-likelihood
        estimator.

        Returns
        -------
        NDArray[np.float64]
            Symmetric ``(p+q, p+q)`` matrix.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        RuntimeError
            If the Hessian is singular or not positive definite (e.g. a
            constraint is active at the MLE, or the basis is degenerate for
            the given data).  ``np.linalg.LinAlgError`` is wrapped so callers
            do not have to special-case the linalg module.
        """
        self._check_is_fitted()
        if self.hessian_ is None:
            raise RuntimeError(
                "hessian_ is unexpectedly missing after fitting. "
                "Please call fit(y) again."
            )
        try:
            return cast(NDArray[np.float64], np.linalg.inv(self.hessian_))
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "vcov() could not be computed: the Hessian matrix is singular "
                "or ill-conditioned.  Possible causes: active monotonicity "
                "constraint at the MLE, basis order too high relative to "
                "sample size, or collinear covariates."
            ) from exc

    def estfun(self) -> NDArray[np.float64]:
        """Per-observation score contributions, ``(n, p+q)``.

        Equivalent to R's ``sandwich::estfun(mlt_fit)``: row ``i`` is
        ``∂ℓ_i/∂θ`` evaluated at :attr:`theta_`.  At the MLE the column sums
        are zero up to optimiser tolerance.

        Returns
        -------
        NDArray[np.float64]
            Matrix of shape ``(n_obs_, p+q)``.  Computed eagerly in
            :meth:`fit` and cached; subsequent mutations of the original
            ``y``/``X`` cannot affect the returned matrix.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        self._check_is_fitted()
        assert self._estfun_cache_ is not None  # guaranteed by fit()
        return self._estfun_cache_

    # R/sandwich-style alias.  Kept as a method (not a bare attribute) so it
    # dispatches on subclass overrides if any.
    def score_contributions(self) -> NDArray[np.float64]:
        """Alias for :meth:`estfun`.  See that method for details."""
        return self.estfun()

    def standard_errors(self) -> NDArray[np.float64]:
        """Vector of asymptotic standard errors for :attr:`theta_`.

        Computed as ``sqrt(diag(vcov()))``.  Length equals ``len(theta_)``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        RuntimeError
            Propagated from :meth:`vcov` if the Hessian is singular.
        """
        diag = np.diag(self.vcov())
        if np.any(diag < 0):
            raise RuntimeError(
                "vcov() contains negative diagonal entries — the Hessian "
                "matrix is not positive definite.  The model may not be "
                "identified or the optimisation may have stalled at a "
                "saddle point."
            )
        return cast(NDArray[np.float64], np.sqrt(diag))

    def confint(
        self,
        level: float = 0.95,
        parm: Sequence[int] | None = None,
    ) -> NDArray[np.float64]:
        """Wald confidence intervals for :attr:`theta_`.

        Computes the symmetric normal-approximation interval

        .. math::
            \\hat\\theta_j \\pm z_{1-\\alpha/2}\\,\\sqrt{V_{jj}},

        where :math:`V = \\mathrm{vcov}()` is the inverse observed information
        matrix and :math:`z_{1-\\alpha/2}` is the standard normal quantile for
        confidence ``level`` :math:`= 1-\\alpha`.  Matches R
        ``confint.default(mlt_fit, level=level)``.

        Parameters
        ----------
        level:
            Confidence level in ``(0, 1)``.  Defaults to ``0.95``.
        parm:
            Optional sequence of integer indices selecting a subset of
            parameters.  ``None`` returns intervals for all entries of
            :attr:`theta_`.

        Returns
        -------
        NDArray[np.float64]
            Array of shape ``(k, 2)`` with columns ``[lower, upper]``; ``k``
            equals ``len(theta_)`` when ``parm is None`` else ``len(parm)``.
            Row order matches the requested index order.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``level`` is outside ``(0, 1)`` or ``parm`` contains indices
            outside ``[0, len(theta_))``.
        RuntimeError
            Propagated from :meth:`vcov` on singular Hessians.

        Examples
        --------
        >>> model = MLT(order=4, support=(0, 1)).fit(y)
        >>> ci = model.confint(level=0.95)  # shape (p, 2)
        """
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError(
                "Model parameters (theta_) are unexpectedly missing after fitting."
            )
        if not (0.0 < level < 1.0):
            raise ValueError(f"level={level!r} is invalid. Expected: 0 < level < 1.")

        se = self.standard_errors()
        k = self.theta_.size
        if parm is None:
            idx = np.arange(k)
        else:
            idx = np.asarray(parm, dtype=int).ravel()
            if idx.size and (idx.min() < 0 or idx.max() >= k):
                raise ValueError(
                    f"parm contains indices outside [0, {k}). Received: "
                    f"min={int(idx.min())}, max={int(idx.max())}."
                )

        z = float(norm.ppf(0.5 * (1.0 + level)))
        est = self.theta_[idx]
        half = z * se[idx]
        return np.column_stack((est - half, est + half))

    def confband(
        self,
        y_grid: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        level: float = 0.95,
        what: Literal[
            "trafo", "distribution", "survivor", "density", "hazard"
        ] = "distribution",
    ) -> NDArray[np.float64]:
        """Pointwise delta-method confidence band for a predicted curve.

        For each grid point ``y_i`` (with an optional covariate profile
        ``x``), compute a "linear-predictor" scale ``η_i`` together with its
        asymptotic variance via the delta method

        .. math::
            \\eta_i = g(y_i, x;\\,\\theta),\\qquad
            \\mathrm{Var}(\\eta_i) = J_i\\,V\\,J_i^\\top,\\quad
            J_i = \\partial\\eta_i/\\partial\\theta,

        form the Wald interval ``η_i ± z · sqrt(Var(η_i))``, and
        back-transform the endpoints to the requested ``what`` scale.  The
        intervals are *pointwise*, not simultaneous.

        The linear predictor and back-transform depend on ``what``:

        * ``"trafo"``        — ``η = h``; back-transform = identity
        * ``"distribution"`` — ``η = h``; back-transform = ``F_base(·)``
        * ``"survivor"``     — ``η = h``; back-transform = ``1 − F_base(·)``
          (endpoints swapped, since ``1 − F`` is decreasing)
        * ``"density"``      — ``η = log f(h) + log h'``; back-transform = ``exp(·)``
        * ``"hazard"``       — ``η = log f(h) + log h' − log S(h)``;
          back-transform = ``exp(·)``

        Parameters
        ----------
        y_grid:
            Response values at which to evaluate the band.  Must lie within
            ``basis.support``.
        X:
            Covariate profile for a single curve.  Accepts a 1D array of
            length ``q`` or a 2D ``(1, q)`` array; broadcast across
            ``y_grid``.  Required when the model was fit with covariates;
            must be ``None`` when it was not.
        level:
            Confidence level in ``(0, 1)``.  Defaults to ``0.95``.
        what:
            One of ``"trafo"``, ``"distribution"``, ``"survivor"``,
            ``"density"``, ``"hazard"``.  Defaults to ``"distribution"``.

        Returns
        -------
        NDArray[np.float64]
            Array of shape ``(len(y_grid), 3)`` with columns
            ``[estimate, lower, upper]`` on the ``what`` scale.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``level`` is outside ``(0, 1)``, ``what`` is not supported,
            or the shape/presence of ``X`` is inconsistent with the fitted
            model.
        RuntimeError
            Propagated from :meth:`vcov` on singular Hessians, or if the
            fitted basis violates monotonicity at a grid point
            (``h'(y) ≤ 0``), which would make the ``density``/``hazard``
            linear predictor ill-defined.

        Notes
        -----
        Working on the transformation scale before back-transforming keeps
        probability bands in ``[0, 1]`` and density/hazard bands positive.
        The reference R routine ``mlt::confband`` builds *simultaneous*
        bands via multivariate-normal quantiles; this implementation is
        pointwise to match the Wald construction used in most applied
        survival plots.

        Examples
        --------
        >>> model = Coxph(support=(0.01, t.max())).fit(cd, X=X)
        >>> grid = np.linspace(0.1, t.max(), 100)
        >>> band = model.confband(grid, X=X[:1], what="survivor")
        >>> ax.fill_between(grid, band[:, 1], band[:, 2], alpha=0.2)
        >>> ax.plot(grid, band[:, 0])
        """
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError(
                "Model parameters (theta_) are unexpectedly missing after fitting."
            )
        if not (0.0 < level < 1.0):
            raise ValueError(f"level={level!r} is invalid. Expected: 0 < level < 1.")
        if what not in _VALID_CONFBAND_WHAT:
            raise ValueError(
                f"what={what!r} is invalid. Allowed: {_VALID_CONFBAND_WHAT}."
            )

        p = self.basis.order + 1
        q = self.theta_.size - p
        theta_b = self.theta_[:p]
        beta = self.theta_[p:] if q > 0 else None

        # Validate X versus the fitted parameter layout
        if q == 0:
            if X is not None:
                raise ValueError(
                    "The model was fitted without covariates; X must be None."
                )
            x_row: NDArray[np.float64] | None = None
        else:
            if X is None:
                raise ValueError(
                    f"The model was fitted with {q} covariates; X is "
                    "required (shape (q,) or (1, q))."
                )
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[None, :]
            if X_arr.shape != (1, q):
                raise ValueError(
                    f"X has shape {X_arr.shape}, expected (q,) or (1, q) with q={q}."
                )
            x_row = X_arr[0]

        y_arr = np.asarray(y_grid, dtype=float).ravel()
        m = y_arr.size
        V = self.vcov()

        B = self.basis.evaluate(y_arr)  # (m, p)
        D = self.basis.derivative(y_arr, order=1)  # (m, p)
        h = B @ theta_b
        hp = D @ theta_b
        if x_row is not None and beta is not None:
            h = h + float(x_row @ beta)

        # Assemble per-grid-point Jacobian J of shape (m, p+q).
        # For scales whose η involves log h', also validate h' > 0.
        if what in ("density", "hazard") and np.any(hp <= 0.0):
            raise RuntimeError(
                "h'(y) <= 0 at at least one grid point — the "
                f"{what} target involves a log h' term and the "
                "confidence-band formula is undefined. Check monotonicity "
                "of the fit or choose a different what."
            )

        dist = _get_dist(self.base_distribution)

        if what in ("trafo", "distribution", "survivor"):
            # η = h;  J_b = B(y),  J_β = x_row  (broadcast)
            # q == 0 ⇒ J is (m, p), no β columns; the branch below is skipped.
            J = np.empty((m, p + q), dtype=np.float64)
            J[:, :p] = B
            if q > 0 and x_row is not None:
                J[:, p:] = x_row[None, :]
            eta = h
        else:
            # "density" or "hazard":
            #   density:  η = log f(h) + log h'
            #   hazard :  η = log f(h) + log h' - log S(h)
            #   ∂/∂h  of log f(h)   = ψ(h)
            #   ∂/∂h  of (-log S(h)) = λ(h) = f(h)/S(h)
            # So:
            #   dη/dθ_b = coeff * B(y) + D(y) / h'
            #   dη/dβ   = coeff * x_row
            # with coeff = ψ(h)         (density)
            #      coeff = ψ(h) + λ(h)  (hazard)
            # At extreme |h|, logsf(h) → -∞ for light-tailed bases (normal,
            # min/max extreme value), so ψ and λ = exp(logpdf − logsf) can
            # overflow and propagate inf into J and η.  Clip to ±_H_CLIP
            # (the same bound likelihood.py uses on every h) and warn when
            # the clip actually bites so the caller knows the band tails
            # are saturated rather than silently infinite.
            if np.any(np.abs(h) > _H_CLIP):
                warnings.warn(
                    f"confband(what={what!r}): |h(y|x)| exceeds ±{_H_CLIP} "
                    "at one or more grid points; clipping for numerical "
                    "stability. The band at these points is a floor/ceiling, "
                    "not a true asymptotic interval. Consider restricting "
                    "y_grid to values where the CDF is not saturated.",
                    stacklevel=2,
                )
            h_c = np.clip(h, -_H_CLIP, _H_CLIP)
            psi = -_neg_score(h_c, dist)  # ψ(h) = d log f(h)/dh
            if what == "hazard":
                # λ(h) = f(h)/S(h); compute in log-space for tail stability.
                lam = np.exp(dist.logpdf(h_c) - dist.logsf(h_c))
                coeff = psi + lam
            else:
                coeff = psi

            # q == 0 ⇒ J is (m, p), no β columns; the branch below is skipped.
            J = np.empty((m, p + q), dtype=np.float64)
            J[:, :p] = coeff[:, None] * B + D / hp[:, None]
            if q > 0 and x_row is not None:
                J[:, p:] = coeff[:, None] * x_row[None, :]

            if what == "density":
                eta = dist.logpdf(h_c) + np.log(hp)
            else:  # hazard
                eta = dist.logpdf(h_c) + np.log(hp) - dist.logsf(h_c)

        # Var(η_i) = J_i · V · J_i^T, vectorised across grid points.
        var_eta = np.einsum("ij,jk,ik->i", J, V, J)
        var_eta = np.maximum(var_eta, 0.0)
        se_eta = np.sqrt(var_eta)

        z = float(norm.ppf(0.5 * (1.0 + level)))
        lo_eta = eta - z * se_eta
        hi_eta = eta + z * se_eta

        if what == "trafo":
            est, lo, hi = eta, lo_eta, hi_eta
        elif what == "distribution":
            est = dist.cdf(h)
            lo = dist.cdf(lo_eta)
            hi = dist.cdf(hi_eta)
        elif what == "survivor":
            est = dist.sf(h)
            # 1 − F is monotone decreasing → swap endpoints
            lo = dist.sf(hi_eta)
            hi = dist.sf(lo_eta)
        else:  # density or hazard: both back-transformed with exp
            est = np.exp(eta)
            lo = np.exp(lo_eta)
            hi = np.exp(hi_eta)

        return cast(
            NDArray[np.float64],
            np.column_stack(
                (
                    np.asarray(est, dtype=np.float64),
                    np.asarray(lo, dtype=np.float64),
                    np.asarray(hi, dtype=np.float64),
                )
            ),
        )

    def aic(self) -> float:
        """Akaike Information Criterion of the fitted model.

        Returns
        -------
        float
            ``AIC = -2 · loglik + 2 · k`` where ``k`` is the number of free
            parameters (``n_free_params_``) and ``loglik`` is the maximised
            log-likelihood.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.

        Notes
        -----
        Lower is better.  The monotonicity inequality ``D @ theta_b >= 0``
        is not counted as a binding equality constraint, so ``k`` equals the
        full length of ``theta_`` — matching R ``mlt::AIC.mlt``, which uses
        ``length(coef(fit))``.

        Examples
        --------
        >>> model = MLT(order=4, support=(0, 1)).fit(y)
        >>> model.aic()
        """
        self._check_is_fitted()
        if self.result_ is None or self.n_free_params_ is None:
            raise RuntimeError("Model state is unexpectedly missing after fitting.")
        return -2.0 * self.result_.log_likelihood + 2.0 * self.n_free_params_

    def bic(self) -> float:
        """Bayesian Information Criterion of the fitted model.

        Returns
        -------
        float
            ``BIC = -2 · loglik + log(n) · k`` where ``n`` is the number of
            observations (``n_obs_``), ``k`` the number of free parameters
            (``n_free_params_``), and ``loglik`` the maximised log-likelihood.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.

        Notes
        -----
        Lower is better.  Penalises additional parameters more heavily than
        :meth:`aic` for ``n > 7``.  Matches R ``mlt::BIC.mlt`` which uses
        ``length(coef(fit))`` for ``k``.

        Examples
        --------
        >>> model = MLT(order=4, support=(0, 1)).fit(y)
        >>> model.bic()
        """
        self._check_is_fitted()
        if self.result_ is None or self.n_free_params_ is None or self.n_obs_ is None:
            raise RuntimeError("Model state is unexpectedly missing after fitting.")
        return (
            -2.0 * self.result_.log_likelihood
            + math.log(self.n_obs_) * self.n_free_params_
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
                    "Result (result_) is unexpectedly missing after fitting."
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
                    "Result (result_) is unexpectedly missing after fitting."
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


# ---------------------------------------------------------------------------
# Likelihood-ratio test (anova)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnovaResult:
    """Result of a likelihood-ratio test comparing nested models.

    Models are sorted by ``n_params`` ascending (reduced → full).  For each
    model after the first, the entry at the same index gives the LR
    statistic comparing it to the *previous* model in the sequence.

    Parameters
    ----------
    model_names : tuple[str, ...]
        Display names of the compared models, in the same order as the rows.
    n_params : tuple[int, ...]
        Number of free parameters per model.
    log_lik : tuple[float, ...]
        Maximised log-likelihood per model.
    df : tuple[int | None, ...]
        Degrees of freedom for each pairwise test (``None`` for the first
        row, which has no predecessor).
    deviance : tuple[float | None, ...]
        Likelihood-ratio statistic ``D = 2·(loglik_full − loglik_reduced)``
        for each pairwise test (``None`` for the first row).
    p_value : tuple[float | None, ...]
        Right-tail probability of the chi-squared distribution with the
        corresponding degrees of freedom (``None`` for the first row).
    """

    model_names: tuple[str, ...]
    n_params: tuple[int, ...]
    log_lik: tuple[float, ...]
    df: tuple[int | None, ...]
    deviance: tuple[float | None, ...]
    p_value: tuple[float | None, ...]

    def __repr__(self) -> str:
        header = (
            f"{'Model':<24} {'n_par':>5} {'logLik':>12} "
            f"{'df':>4} {'Deviance':>12} {'Pr(>Chisq)':>12}"
        )
        rows = [header, "-" * len(header)]
        for i in range(len(self.model_names)):
            df_str = "" if self.df[i] is None else str(self.df[i])
            dev_str = "" if self.deviance[i] is None else f"{self.deviance[i]:>12.4f}"
            p_str = "" if self.p_value[i] is None else f"{self.p_value[i]:>12.4g}"
            rows.append(
                f"{self.model_names[i]:<24} "
                f"{self.n_params[i]:>5} "
                f"{self.log_lik[i]:>12.4f} "
                f"{df_str:>4} "
                f"{dev_str:>12} "
                f"{p_str:>12}"
            )
        return "\n".join(rows)


def anova(*models: ConditionalTransformationModel) -> AnovaResult:
    """Likelihood-ratio test for a sequence of nested transformation models.

    Models are sorted internally by their number of free parameters
    (``n_free_params_``) in ascending order, and pairwise LR statistics are
    computed against the immediately smaller model.  The user is responsible
    for ensuring the models are *actually* nested (fitted on the same data
    with the smaller's parameter space contained in the larger's).  Sample
    size is checked; structural nesting is not.

    Parameters
    ----------
    *models:
        Two or more fitted :class:`ConditionalTransformationModel` instances.

    Returns
    -------
    AnovaResult
        See :class:`AnovaResult` for the column layout.

    Raises
    ------
    ValueError
        If fewer than two models are passed; if any model is not fitted; if
        the models were fitted on different sample sizes; or if two
        consecutive models (after sorting) have the same number of free
        parameters (cannot be nested).

    Notes
    -----
    The test statistic is ``D = 2·(loglik_full − loglik_reduced)``, which is
    asymptotically ``χ²_df`` with ``df = k_full − k_reduced`` under the null
    hypothesis that the reduced model is correct.  Mirrors R's
    ``anova.mlt``.

    Examples
    --------
    >>> small = MLT(order=3, support=(0, 1)).fit(y)
    >>> large = MLT(order=6, support=(0, 1)).fit(y)
    >>> print(anova(small, large))
    """
    if len(models) < 2:
        raise ValueError(
            f"anova() requires at least 2 models, received: {len(models)}."
        )
    for i, m in enumerate(models):
        if not m.is_fitted_:
            raise ValueError(
                f"Model #{i} is not fitted. Call fit() before anova()."
            )

    n_obs_ref = models[0].n_obs_
    for i, m in enumerate(models):
        if m.n_obs_ != n_obs_ref:
            raise ValueError(
                f"Models must be fitted on the same sample size. "
                f"Model #0 has n={n_obs_ref}, model #{i} has n={m.n_obs_}."
            )

    # Label each model by its caller-input position *before* sorting, so that
    # row labels in the output table point back to the argument the user
    # actually passed. Without this, anova(large, small) would print the
    # smaller model as "#1" because sort moves it to row 0.
    indexed = list(enumerate(models))
    ordered_indexed = sorted(indexed, key=lambda im: cast(int, im[1].n_free_params_))
    ordered = [m for _, m in ordered_indexed]

    names = tuple(f"{type(m).__name__}#{i}" for i, m in ordered_indexed)
    n_params = tuple(cast(int, m.n_free_params_) for m in ordered)
    log_lik = tuple(cast(OptimizationResult, m.result_).log_likelihood for m in ordered)

    df: list[int | None] = [None]
    deviance: list[float | None] = [None]
    p_value: list[float | None] = [None]
    for i in range(1, len(ordered)):
        ddf = n_params[i] - n_params[i - 1]
        if ddf <= 0:
            raise ValueError(
                "Consecutive models must have strictly different parameter "
                f"counts (model {i - 1}: k={n_params[i - 1]}, "
                f"model {i}: k={n_params[i]})."
            )
        d = 2.0 * (log_lik[i] - log_lik[i - 1])
        # Negative deviance can occur if the larger model failed to converge
        # to a strictly higher likelihood; clamp at 0 for the chi2 tail.
        d_clamped = max(d, 0.0)
        df.append(ddf)
        deviance.append(d)
        p_value.append(float(chi2.sf(d_clamped, ddf)))

    return AnovaResult(
        model_names=names,
        n_params=n_params,
        log_lik=log_lik,
        df=tuple(df),
        deviance=tuple(deviance),
        p_value=tuple(p_value),
    )
