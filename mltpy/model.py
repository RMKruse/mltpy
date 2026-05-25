"""Public API for conditional transformation models.

Users import exclusively from this module::

    import mltpy
    model = mltpy.MLT(order=6, support=(0, 100))
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import comb, log_ndtr
from scipy.stats import chi2, norm

from mltpy.basis import BernsteinBasis, InteractionBasis
from mltpy.likelihood import (
    _H_CLIP,
    BaseDistribution,
    InfeasibleParameterError,
    _get_dist,
    _neg_score,
    _validate_offset,
    _validate_weights_offset,
    log_likelihood,
)
from mltpy.likelihood import (
    hessian as _hessian,
)
from mltpy.likelihood import (
    intercept_score as _intercept_score,
)
from mltpy.likelihood import (
    score_matrix as _score_matrix,
)
from mltpy.optimizer import OptimizationResult, OptimizerConfig, optimize
from mltpy.variables import CensoredData, CensoringType

# ---------------------------------------------------------------------------
# Exceptions and warnings
# ---------------------------------------------------------------------------


class NotFittedError(ValueError):
    """Raised when a method that requires a fitted model is called before fit()."""


class ConvergenceWarning(UserWarning):
    """Raised when the optimiser fails to converge within the allowed restarts."""


class _ProfileInnerFailure(Exception):
    """Internal — signals that one inner refit inside profile-CI failed.

    Raised by :meth:`ConditionalTransformationModel._profile_loglik_at`
    when the pinned refit either lands on a degenerate monotonicity
    active set (``kind="boundary"`` — the equality ``theta_[j]=v`` could
    not be honoured while preserving the inequality constraints, detected
    via ``theta[j]`` drift from the pin) or fails to converge within
    tolerance (``kind="convergence"`` — non-negligible KKT residual with
    no theta drift).

    Caught in :meth:`_profile_root` and translated to the user-facing
    behaviour documented on :meth:`confint` (``ConvergenceWarning`` +
    ``±inf`` / ``NaN`` endpoint under ``parm=None``, or ``RuntimeError``
    under explicit ``parm=[j]``).  Never reaches end users.
    """

    def __init__(
        self,
        *,
        j: int,
        kind: Literal["boundary", "convergence"],
        diagnostic: str,
    ) -> None:
        self.j = j
        self.kind = kind
        self.diagnostic = diagnostic
        super().__init__(diagnostic)


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

# Small epsilon used for bracket safety in the quantile bisection
_BRENTQ_EPS = 1e-10

# Grid size used by R's qmlt() default quantile inversion.
_QMLT_GRID_POINTS = 50

# simulate(): clip Uniform(0,1) draws to [_SIMULATE_U_EPS, 1 − _SIMULATE_U_EPS]
# before inverting through F⁻¹ in _predict_quantile.  At u = 0 / u = 1 the
# base CDF inverse returns ∓∞, which would propagate as NaN through the
# Bernstein bisection.  1e-10 keeps the saturation tail to ppf(1e-10) ≈ ±6.4
# (normal) — well inside _H_CLIP — while losing < 0.0000001% of the
# distribution at each end.  Bracket saturation beyond this is reported by
# _predict_quantile's own warning, not silenced here.
_SIMULATE_U_EPS = 1e-10

# `what` values whose formula involves h'(y) and therefore require hp > 0.
_HP_REQUIRING_WHAT = frozenset({"density", "logdensity", "hazard", "loghazard"})


def _extract_feature_names(X: object) -> list[str] | None:
    """Extract column names from DataFrame-like ``X``, else ``None``.

    Kept as a free function so no pandas import is required at module load
    time — we only touch DataFrame-like attributes by duck-typing.

    Raises
    ------
    ValueError
        If ``X`` exposes both ``columns`` and a 2-D ``shape``, but the column
        name count does not match ``shape[1]``.
    """
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    feature_names = [str(c) for c in columns]
    shape = getattr(X, "shape", None)
    if (
        isinstance(shape, tuple)
        and len(shape) >= 2
        and isinstance(shape[1], int | np.integer)
        and len(feature_names) != int(shape[1])
    ):
        raise ValueError(
            f"X columns metadata has length {len(feature_names)} but X has "
            f"{int(shape[1])} columns."
        )
    return feature_names


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
        :class:`~mltpy.basis.BernsteinBasis` defining the response
        transformation.
    censoring:
        Censoring type of the response data.  Defaults to
        :attr:`~mltpy.variables.CensoringType.NONE`.
    optimizer_config:
        Optimisation settings.  If ``None``, defaults from
        :class:`~mltpy.optimizer.OptimizerConfig` are used.
    """

    def __init__(
        self,
        basis: BernsteinBasis | InteractionBasis,
        censoring: CensoringType | None = CensoringType.NONE,
        optimizer_config: OptimizerConfig | None = None,
        base_distribution: BaseDistribution = "normal",
        scaling: NDArray[np.float64] | None = None,
    ) -> None:
        _get_dist(base_distribution)  # raises ValueError for unsupported values
        scaling_arr: NDArray[np.float64] | None = None
        scaling_feature_names: list[str] | None = None
        if scaling is not None:
            # ADR 0002 — the non-interaction shift + scaled path supports all
            # four censoring types (#71).  ADR 0003 (#103) integrates scaling
            # with ``InteractionBasis`` (exact data, normal / min-extreme-value
            # base only).  Both paths reject ``"exponential"`` because its
            # support-feasibility row becomes non-linear in γ (ADR 0002
            # Decision 3 / ADR 0003 Decision 5).
            if base_distribution == "exponential":
                raise ValueError(
                    "scaling= is not supported with base_distribution="
                    "'exponential' (see docs/adr/0002-scaling-terms.md, "
                    "Decision 3, and docs/adr/0003-scaling-with-interaction.md, "
                    "Decision 5)."
                )
            scaling_feature_names = _extract_feature_names(scaling)
            scaling_arr = np.asarray(scaling, dtype=float)
            if scaling_arr.ndim == 1:
                scaling_arr = scaling_arr[:, None]
            if scaling_arr.ndim != 2:
                raise ValueError(
                    "scaling must be a 2-D array of shape (n, q_s); got shape "
                    f"{scaling_arr.shape}."
                )
            if not np.all(np.isfinite(scaling_arr)):
                raise ValueError("scaling must be finite (no NaN or inf).")
        self.basis = basis
        self.censoring = censoring
        self.optimizer_config = optimizer_config
        self.base_distribution = base_distribution
        self.scaling = scaling_arr
        self.scaling_feature_names_in_: list[str] | None = scaling_feature_names

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
        :class:`~mltpy.variables.CensoredData`, this is ``y.n``; otherwise
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

        self._A_ineq_: NDArray[np.float64] | None = None
        """Inequality constraint matrix from the last auglag fit, shape
        ``(m_ineq, total_params)``.  ``None`` before :meth:`fit` or when
        the solver is not auglag."""

        self._C_eq_: NDArray[np.float64] | None = None
        """Equality constraint matrix from the last auglag fit when
        ``lower`` / ``upper`` are pinned.  ``None`` when no equality
        constraints were imposed, or when the solver is not auglag."""

        self.weights_: NDArray[np.float64] | None = None
        """Observation weights supplied to the last :meth:`fit` call.
        ``None`` when no weights were used."""

        self.offset_: NDArray[np.float64] | None = None
        """Per-observation offset supplied to the last :meth:`fit` call.
        ``None`` when no offset was used."""

        # Training response/covariates retained for residuals().  Stored as
        # copies so subsequent caller mutations cannot affect diagnostics.
        self._y_train_: NDArray[np.float64] | CensoredData | None = None
        self._X_train_: NDArray[np.float64] | None = None
        self._weights_train_: NDArray[np.float64] | None = None
        self._offset_train_: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def Theta_(self) -> NDArray[np.float64] | None:
        """Coefficient matrix ``Θ`` of shape ``(p, q)`` for interaction models.

        ``None`` before :meth:`fit` or for non-interaction models.
        ``theta_[i*q + j] = Θ[i, j]`` (row-major layout).
        """
        if self.theta_ is None or not isinstance(self.basis, InteractionBasis):
            return None
        p = self.basis.n_y_params
        q = self.basis.n_x_params
        # On the scaled-interaction path theta_ = [vec_C(Θ) | γ]; reshape only
        # the leading p·q block (ADR 0003 Decision 1).
        return self.theta_[: p * q].reshape(p, q)

    @property
    def gamma_coef_(self) -> NDArray[np.float64] | None:
        """Scaling-block coefficients ``γ`` (length ``q_s``).

        ``None`` before :meth:`fit` or when the model was constructed without
        ``scaling=``.  Sign-aligned with R ``tram::*(scale=...)``'s scaling
        block (no flip needed for parity comparisons; see
        ``docs/adr/0002-scaling-terms.md``, Decision 5).
        """
        if self.theta_ is None or self.scaling is None:
            return None
        if isinstance(self.basis, InteractionBasis):
            # Interaction layout is [vec_C(Θ) | γ] — no β shift block
            # (ADR 0003 Decision 1).  γ is the trailing q_s entries.
            pq = self.basis.n_y_params * self.basis.n_x_params
            return self.theta_[pq:]
        p = self.basis.order + 1
        q_d = 0 if self._X_train_ is None else self._X_train_.shape[1]
        return self.theta_[p + q_d :]

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
            if y_arr.size == 0:
                raise ValueError(
                    "y must contain at least one observation, got empty array"
                )
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
        weights: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> "ConditionalTransformationModel":
        """Fit the transformation model by maximum likelihood.

        Parameters
        ----------
        y:
            Response observations.  Must lie within ``basis.support``.
            Accepts ``np.ndarray``, ``pd.Series``, or
            :class:`~mltpy.variables.CensoredData`.
        X:
            Optional covariate matrix of shape ``(n, q)``.  If given, the
            last ``q`` entries of ``theta_`` are regression coefficients.
        weights:
            Optional non-negative per-observation weights of shape ``(n,)``.
            The weighted log-likelihood ``Σ w_i · ℓ_i`` is maximised; no
            normalisation is applied.  ``None`` is equivalent to all-ones.
        offset:
            Optional per-observation offset of shape ``(n,)``.  Added to
            ``h(y|x)`` before distribution calls on every likelihood
            evaluation: ``h_eff = B·θ_b + X·β + offset``.  ``None`` is
            equivalent to all-zeros.

        Returns
        -------
        self
            Returns itself for method chaining::

                cdf = model.fit(y).predict(y, what="distribution")

        Raises
        ------
        ValueError
            If ``y`` contains values outside ``basis.support``, or if
            ``weights``/``offset`` have the wrong shape or invalid values.
        """
        feature_names = _extract_feature_names(X)
        y_clean, X_clean = self._validate_input(y, X)
        n = int(y_clean.n) if isinstance(y_clean, CensoredData) else len(y_clean)
        weights_clean, offset_clean = _validate_weights_offset(weights, offset, n)
        if self.scaling is not None and self.scaling.shape[0] != n:
            raise ValueError(
                f"scaling has {self.scaling.shape[0]} rows but y has {n} "
                "observations; both must match."
            )

        if self.scaling is not None and isinstance(self.basis, InteractionBasis):
            # ADR 0003 Decision 7 — mirror R tram's stram "highly experimental"
            # warning for the combined non-proportional, heteroskedastic CTM.
            warnings.warn(
                "scaling= with InteractionBasis is an experimental path; "
                "validate against your use case (mirrors R tram's stram "
                "warning).",
                UserWarning,
                stacklevel=2,
            )

        censoring_arg = CensoringType.NONE if self.censoring is None else self.censoring
        result = optimize(
            self.basis,
            y_clean,
            X=X_clean,
            censoring=censoring_arg,
            config=self.optimizer_config,
            base_distribution=self.base_distribution,
            weights=weights_clean,
            offset=offset_clean,
            scaling=self.scaling,
        )

        if not result.converged:
            warnings.warn(
                f"Optimization did not converge after {result.n_restarts} "
                f"restarts. Solver message: {result.solver_message}. "
                "The result is the best found, but may not be the MLE.",
                ConvergenceWarning,
                stacklevel=2,
            )

        self.theta_ = result.theta
        self.result_ = result
        self.is_fitted_ = True
        self._A_ineq_ = result.constraint_A_ineq
        self._C_eq_ = result.constraint_C_eq
        self.n_obs_ = (
            int(y_clean.n) if isinstance(y_clean, CensoredData) else len(y_clean)
        )
        self.n_free_params_ = int(result.theta.size)

        # Feature names for the covariate block of theta_.
        if X_clean is not None:
            q = X_clean.shape[1]
            if feature_names is None:
                feature_names = [f"X{j + 1}" for j in range(q)]
            self.feature_names_in_ = feature_names
        else:
            self.feature_names_in_ = None

        # Observed information and score matrix — computed eagerly so that
        # later mutations of the caller's ``y``/``X`` cannot affect
        # ``vcov()`` or ``estfun()`` results.  Failures here indicate a real
        # modelling problem (degenerate basis, constraint-binding fit);
        # surface them.
        self.weights_ = weights_clean
        self.offset_ = offset_clean

        self.hessian_ = _hessian(
            self.theta_,
            self.basis,
            y_clean,
            X_clean,
            censoring_arg,
            base_distribution=self.base_distribution,
            weights=weights_clean,
            offset=offset_clean,
            scaling=self.scaling,
        )
        self._estfun_cache_ = _score_matrix(
            self.theta_,
            self.basis,
            y_clean,
            X_clean,
            censoring_arg,
            base_distribution=self.base_distribution,
            weights=weights_clean,
            offset=offset_clean,
            scaling=self.scaling,
        )

        # Snapshot the training response and covariates for diagnostics
        # (residuals()).  Defensive copies so caller mutations of the
        # original ``y``/``X`` cannot leak into later diagnostic calls.
        if isinstance(y_clean, CensoredData):
            self._y_train_ = CensoredData(
                exact=y_clean.exact.copy(),
                lower=y_clean.lower.copy(),
                upper=y_clean.upper.copy(),
                trunc_lower=(
                    y_clean.trunc_lower.copy()
                    if y_clean.trunc_lower is not None
                    else None
                ),
                trunc_upper=(
                    y_clean.trunc_upper.copy()
                    if y_clean.trunc_upper is not None
                    else None
                ),
            )
        else:
            self._y_train_ = y_clean.copy()
        self._X_train_ = X_clean.copy() if X_clean is not None else None
        self._weights_train_ = (
            weights_clean.copy() if weights_clean is not None else None
        )
        self._offset_train_ = offset_clean.copy() if offset_clean is not None else None
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
        offset_new: NDArray[np.float64] | None = None,
        X_scale_new: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute model predictions at new observations.

        Parameters
        ----------
        y_new:
            For ``what="quantile"``: probabilities in ``(0, 1)``.
            For all other ``what``: response values in ``basis.support``.
        X_new:
            Optional covariate matrix of shape ``(m, q)``.
        offset_new:
            Optional per-observation offset of shape ``(m,)``.  Added to
            ``h(y|x)`` before distribution calls.
        X_scale_new:
            New-data scaling-design matrix of shape ``(m, q_s)``, required
            when the model was fitted with ``scaling=``.  Enters via
            ``h(y|x_d, x_s) = h_0(y) · exp(0.5 · x_s · γ) + x_d · β`` —
            same parameterisation as :meth:`fit`.  Pass ``None`` for
            non-scaling fits.
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
            * ``"quantile"``        — Quantile via inversion; right-censored
              models use an R-compatible grid+spline inversion.

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
        Log-scale variants use ``scipy.special.log_ndtr`` for the normal
        distribution's log-CDF (more accurate in the tails) and
        ``dist.logcdf``/``logsf``/``logpdf`` otherwise.

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

        y_arr = np.asarray(y_new, dtype=float).ravel()
        m = y_arr.shape[0]
        offset_arr: NDArray[np.float64] | None = (
            _validate_offset(offset_new, m) if offset_new is not None else None
        )
        X_arr: NDArray[np.float64] | None = None
        if X_new is not None:
            X_arr = np.asarray(X_new, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]

        # ------------------------------------------------------------------
        # Scaling-design validation (ADR 0002; required when the model was
        # fit with ``scaling=``).  Validated here so the same check fires
        # before either the interaction or shift dispatch.
        # ------------------------------------------------------------------
        X_scale_arr: NDArray[np.float64] | None = None
        if self.scaling is not None:
            if X_scale_new is None:
                raise ValueError(
                    "Model was fitted with scaling=; X_scale_new must be "
                    f"provided (shape (m, {self.scaling.shape[1]}))."
                )
            X_scale_arr = np.asarray(X_scale_new, dtype=float)
            if X_scale_arr.ndim == 1:
                X_scale_arr = X_scale_arr[:, None]
            if X_scale_arr.ndim != 2:
                raise ValueError(
                    "X_scale_new must be a 2-D array of shape (m, q_s); "
                    f"got shape {X_scale_arr.shape}."
                )
            if X_scale_arr.shape[0] != m:
                raise ValueError(
                    f"X_scale_new has {X_scale_arr.shape[0]} rows but y_new "
                    f"has {m} elements; both must match."
                )
            if X_scale_arr.shape[1] != self.scaling.shape[1]:
                raise ValueError(
                    f"X_scale_new has {X_scale_arr.shape[1]} columns but the "
                    f"fitted model has q_s={self.scaling.shape[1]} scaling "
                    "coefficients."
                )
        elif X_scale_new is not None:
            raise ValueError(
                "Model was not fitted with scaling=; X_scale_new must be None."
            )

        # ------------------------------------------------------------------
        # Interaction basis path
        # ------------------------------------------------------------------
        if isinstance(self.basis, InteractionBasis):
            if X_arr is None:
                raise ValueError(
                    "InteractionBasis model requires X_new for prediction."
                )
            if X_arr.shape[0] != y_arr.shape[0]:
                raise ValueError(
                    f"X_new has {X_arr.shape[0]} rows but y_new has "
                    f"{y_arr.shape[0]} elements; they must match for "
                    "InteractionBasis prediction."
                )
            if what == "quantile":
                return self._predict_quantile_interaction(
                    y_arr, X_arr, offset=offset_arr
                )
            x_1d = self.basis._coerce_x(X_arr)
            design = self.basis.evaluate(y_arr, x_1d)  # (m, p*q)
            d_design = self.basis.derivative(y_arr, x_1d)  # (m, p*q)
            h = design @ self.theta_
            hp = d_design @ self.theta_
            if offset_arr is not None:
                h = h + offset_arr

            dist = _get_dist(self.base_distribution)
            _logcdf = log_ndtr if dist.kind == "normal" else dist.logcdf

            if what in _HP_REQUIRING_WHAT and np.any(hp <= 0.0):
                raise InfeasibleParameterError(
                    f"predict(what={what!r}) requires h'(y) > 0."
                )
            if what == "trafo":
                return h
            if np.any(np.abs(h) > _H_CLIP):
                warnings.warn(
                    f"predict(what={what!r}): |h(y|x)| exceeds ±{_H_CLIP} at "
                    "one or more points; clipping for numerical stability.",
                    stacklevel=2,
                )
            h_c = np.clip(h, -_H_CLIP, _H_CLIP)
            if what == "distribution":
                return dist.cdf(h_c)
            if what == "logdistribution":
                return cast(NDArray[np.float64], _logcdf(h_c))
            if what == "survivor":
                return cast(NDArray[np.float64], dist.sf(h_c))
            if what == "logsurvivor":
                return dist.logsf(h_c)
            if what == "density":
                return dist.pdf(h_c) * hp
            if what == "logdensity":
                with np.errstate(divide="ignore"):
                    return dist.logpdf(h_c) + np.log(hp)
            if what == "hazard":
                return cast(NDArray[np.float64], dist.pdf(h_c) * hp / dist.sf(h_c))
            if what == "loghazard":
                return dist.logpdf(h_c) + np.log(hp) - dist.logsf(h_c)
            if what == "cumhazard":
                return -dist.logsf(h_c)
            if what == "logcumhazard":
                return cast(NDArray[np.float64], np.log(-dist.logsf(h_c)))
            if what == "odds":
                return cast(NDArray[np.float64], dist.cdf(h_c) / dist.sf(h_c))
            if what == "logodds":
                return cast(NDArray[np.float64], _logcdf(h_c) - dist.logsf(h_c))
            raise ValueError(
                f"what={what!r} is not supported for InteractionBasis predict."
            )

        # ------------------------------------------------------------------
        # Standard (shift) basis path
        # ------------------------------------------------------------------
        p = self.basis.order + 1
        theta_b = self.theta_[:p]
        q_s = 0 if self.scaling is None else self.scaling.shape[1]
        q_d = self.theta_.size - p - q_s
        beta_fit: NDArray[np.float64] | None = (
            self.theta_[p : p + q_d] if q_d > 0 else None
        )
        gamma_fit: NDArray[np.float64] | None = (
            self.theta_[p + q_d :] if q_s > 0 else None
        )

        if what == "quantile":
            xbeta: NDArray[np.float64] | None = None
            if q_d > 0:
                if X_arr is None:
                    raise ValueError(
                        "Model was fitted with covariates; X_new must be "
                        "provided for conditional quantile prediction."
                    )
                if X_arr.shape[0] != y_arr.shape[0]:
                    raise ValueError(
                        f"X_new has {X_arr.shape[0]} rows but y_new has "
                        f"{y_arr.shape[0]} elements; they must match for "
                        "quantile prediction."
                    )
                assert beta_fit is not None
                if X_arr.shape[1] != beta_fit.shape[0]:
                    raise ValueError(
                        f"X_new has {X_arr.shape[1]} columns but the fitted "
                        f"model has {beta_fit.shape[0]} covariate coefficients."
                    )
                xbeta = X_arr @ beta_fit
            if gamma_fit is not None and X_scale_arr is not None:
                return self._predict_quantile_scaling(
                    y_arr,
                    theta_b,
                    gamma_fit,
                    X_scale_arr,
                    xbeta=xbeta,
                    offset=offset_arr,
                )
            return self._predict_quantile(
                y_arr, theta_b, xbeta=xbeta, offset=offset_arr
            )

        # Evaluate transformation and its derivative
        B = self.basis.evaluate(y_arr)  # (m, p)
        D = self.basis.derivative(y_arr, order=1)  # (m, p)
        h0 = B @ theta_b  # (m,)
        hp0 = D @ theta_b  # (m,)
        # Scaling factor f_i = exp(0.5 · x_s,i · γ).  Same convention as
        # likelihood._ll_none (and R `mlt:::tmlt`), so γ is sign- and
        # magnitude-aligned with R `tram`'s scaling block.
        if gamma_fit is not None and X_scale_arr is not None:
            f_scale = np.exp(0.5 * (X_scale_arr @ gamma_fit))
            h = h0 * f_scale
            hp = hp0 * f_scale
        else:
            h = h0
            hp = hp0

        if X_arr is not None and beta_fit is not None:
            h = h + X_arr @ beta_fit

        if offset_arr is not None:
            h = h + offset_arr

        dist = _get_dist(self.base_distribution)
        _logcdf = log_ndtr if dist.kind == "normal" else dist.logcdf
        if what in _HP_REQUIRING_WHAT and np.any(hp <= 0.0):
            raise InfeasibleParameterError(
                f"predict(what={what!r}) requires h'(y) > 0, but the fitted "
                f"theta_ yields min(h'(y)) = {float(np.min(hp)):.4g} ≤ 0 at "
                "one or more requested points.  This indicates a non-monotone "
                "transformation — fit() should have rejected this parameter; "
                "if you see this, the model state is inconsistent."
            )

        if what == "trafo":
            return h

        # Clip h to ±_H_CLIP before distribution calls — the same bound
        # likelihood.py and confband() use everywhere else — and warn when
        # the clip actually bites so the caller knows the returned values
        # at those points are saturated at a floor/ceiling rather than the
        # true asymptotic limit.
        if np.any(np.abs(h) > _H_CLIP):
            warnings.warn(
                f"predict(what={what!r}): |h(y|x)| exceeds ±{_H_CLIP} at "
                "one or more points; clipping for numerical stability. "
                "Values at these points are saturated, not the true "
                "asymptotic limit.",
                stacklevel=2,
            )
        h_c = np.clip(h, -_H_CLIP, _H_CLIP)

        if what == "distribution":
            return dist.cdf(h_c)
        if what == "logdistribution":
            return cast(NDArray[np.float64], _logcdf(h_c))
        if what == "survivor":
            return cast(NDArray[np.float64], dist.sf(h_c))
        if what == "logsurvivor":
            return dist.logsf(h_c)
        if what == "density":
            return dist.pdf(h_c) * hp
        if what == "logdensity":
            return dist.logpdf(h_c) + np.log(hp)
        if what == "hazard":
            return cast(
                NDArray[np.float64],
                np.exp(dist.logpdf(h_c) - dist.logsf(h_c)) * hp,
            )
        if what == "loghazard":
            return dist.logpdf(h_c) + np.log(hp) - dist.logsf(h_c)
        if what == "cumhazard":
            return -dist.logsf(h_c)
        if what == "logcumhazard":
            return cast(NDArray[np.float64], np.log(-dist.logsf(h_c)))
        if what == "odds":
            return cast(NDArray[np.float64], np.exp(_logcdf(h_c) - dist.logsf(h_c)))
        # logodds
        return cast(NDArray[np.float64], _logcdf(h_c) - dist.logsf(h_c))

    def _predict_quantile(
        self,
        probs: NDArray[np.float64],
        theta_b: NDArray[np.float64],
        xbeta: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Numerically invert the fitted distribution for quantile prediction.

        For right-censored models, follow R ``mlt::qmlt`` semantics:
        evaluate a CDF curve on a fixed grid and invert that curve via
        spline/interpolation.

        For other censoring types, solve
        ``h_baseline(q_i) = F⁻¹(p_i) − xbeta[i] − offset[i]`` directly via
        vectorised bisection, where ``h_baseline(y) = B_k(y) · theta_b`` and
        ``F⁻¹`` is the base-distribution quantile function.

        Parameters
        ----------
        probs:
            Probabilities in (0, 1).
        theta_b:
            Bernstein coefficient vector of length ``order + 1``.
        xbeta:
            Per-row linear predictor ``X @ beta`` of shape ``(len(probs),)``,
            or ``None`` for the baseline (no-covariate) case.
        offset:
            Per-row offset of shape ``(len(probs),)``, or ``None``.

        Returns
        -------
        NDArray of same length as ``probs``.
        """
        a, b = self.basis.support
        k = self.basis.order
        theta_b = np.asarray(theta_b, dtype=float)
        probs_arr = np.asarray(probs, dtype=np.float64)

        n_probs = len(probs_arr)
        if n_probs == 0:
            return np.empty(0, dtype=float)

        dist = _get_dist(self.base_distribution)
        shift: NDArray[np.float64] = (
            np.zeros(n_probs, dtype=np.float64)
            if xbeta is None
            else np.asarray(xbeta, dtype=np.float64)
        )
        if offset is not None:
            shift = shift + np.asarray(offset, dtype=np.float64)

        # R-compatible quantile inversion path for right-censored models:
        # qmlt() evaluates CDF on a K-point grid and .invf() inverts with
        # spline+approx. This avoids support-bracket clipping artifacts in
        # Coxph quantiles with covariate shifts.
        if self.censoring is CensoringType.RIGHT:
            q_grid = np.linspace(0.0, b, _QMLT_GRID_POINTS, dtype=np.float64)

            i_arr = np.arange(k + 1, dtype=float)
            j_arr = k - i_arr
            binom_theta = comb(k, i_arr, exact=False) * theta_b
            inv_width = 1.0 / (b - a)
            t_grid = (q_grid - a) * inv_width
            T_grid = (t_grid[:, None] ** i_arr) * ((1.0 - t_grid)[:, None] ** j_arr)
            h_base_grid = cast(NDArray[np.float64], T_grid @ binom_theta)

            eps = float(np.sqrt(np.finfo(float).eps))
            out = np.empty(n_probs, dtype=np.float64)
            saturated = False

            for i, p in enumerate(probs_arr):
                cdf_grid = dist.cdf(h_base_grid + shift[i])
                # For survival-time responses, R's grid starts at 0 and the
                # CDF at time zero is treated as zero.
                cdf_grid[0] = 0.0

                finite = np.isfinite(cdf_grid)
                if not np.any(finite):
                    out[i] = q_grid[0]
                    saturated = True
                    continue

                cmin = float(np.min(cdf_grid[finite]))
                cmax = float(np.max(cdf_grid[finite]))
                remove = ~finite
                flat_low = np.where(cdf_grid < cmin + eps)[0]
                flat_high = np.where(cdf_grid > cmax - eps)[0]
                if flat_low.size > 1:
                    remove[flat_low[:-1]] = True
                if flat_high.size > 1:
                    remove[flat_high[1:]] = True

                keep = ~remove
                qk = q_grid[keep]
                ck = cdf_grid[keep]
                if qk.size < 2:
                    out[i] = q_grid[0]
                    saturated = True
                    continue

                n_spline = max(3 * qk.size, 3)
                q_s = np.linspace(
                    float(qk[0]),
                    float(qk[-1]),
                    n_spline,
                    dtype=np.float64,
                )
                c_s = cast(
                    NDArray[np.float64],
                    CubicSpline(qk, ck, bc_type="not-a-knot", extrapolate=False)(q_s),
                )

                finite_s = np.isfinite(c_s)
                if not np.any(finite_s):
                    out[i] = q_grid[0]
                    saturated = True
                    continue
                q_s = q_s[finite_s]
                c_s = c_s[finite_s]

                order = np.argsort(c_s)
                c_s = c_s[order]
                q_s = q_s[order]
                uniq = np.concatenate(([True], np.diff(c_s) > 0.0))
                c_s = c_s[uniq]
                q_s = q_s[uniq]
                if c_s.size == 0:
                    out[i] = q_grid[0]
                    saturated = True
                    continue

                if p < c_s[0]:
                    out[i] = q_grid[0]
                    saturated = True
                elif p > c_s[-1]:
                    out[i] = q_grid[-1]
                    saturated = True
                else:
                    out[i] = float(np.interp(p, c_s, q_s))

            if saturated:
                warnings.warn(
                    "predict(what='quantile'): probability target lies outside "
                    "the finite R-style inversion grid at one or more points; "
                    "returning boundary-saturated quantiles.",
                    stacklevel=2,
                )
            return out

        # Bracket-clip range for h_baseline — independent of xbeta.
        z_min = float(theta_b[0]) + _BRENTQ_EPS
        z_max = float(theta_b[-1]) - _BRENTQ_EPS

        z_raw = dist.ppf(probs_arr) - shift
        if np.any((z_raw < z_min) | (z_raw > z_max)):
            warnings.warn(
                "predict(what='quantile'): F⁻¹(p) − xβ exceeds the basis "
                f"bracket [θ_b[0]+ε, θ_b[-1]−ε] = [{z_min:.4g}, {z_max:.4g}] "
                "at one or more points; clipping for numerical stability. "
                "Quantiles at these points are saturated at the basis "
                "endpoints, not the true asymptotic limit. Consider widening "
                "the basis support or restricting probs away from 0/1.",
                stacklevel=2,
            )
        z = np.clip(z_raw, z_min, z_max)

        # Precomputed Bernstein constants: h(q) = sum_i binom(k,i) t^i (1-t)^(k-i) θ_i
        # with t = (q - a) / (b - a). Folding binom·θ once avoids recomputation.
        i_arr = np.arange(k + 1, dtype=float)
        j_arr = k - i_arr
        binom_theta = comb(k, i_arr, exact=False) * theta_b
        inv_width = 1.0 / (b - a)

        def _h_vec(q: NDArray[np.float64]) -> NDArray[np.float64]:
            t = (q - a) * inv_width
            T = (t[:, None] ** i_arr) * ((1.0 - t)[:, None] ** j_arr)
            return cast(NDArray[np.float64], T @ binom_theta)

        # Vectorised bisection. At most 60 iterations; breaks early once the
        # widest remaining bracket is < _BRENTQ_EPS (same tolerance used for
        # bracket endpoints).  For a width-100 support this exits after ~40
        # iters; for width-1 after ~33.
        lo = np.full(n_probs, a, dtype=float)
        hi = np.full(n_probs, b, dtype=float)
        mid = 0.5 * (lo + hi)
        for _ in range(60):
            if np.max(hi - lo) < _BRENTQ_EPS:
                break
            below = _h_vec(mid) < z
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
            mid = 0.5 * (lo + hi)
        return cast(NDArray[np.float64], mid)

    def _predict_quantile_scaling(
        self,
        probs: NDArray[np.float64],
        theta_b: NDArray[np.float64],
        gamma: NDArray[np.float64],
        X_scale: NDArray[np.float64],
        xbeta: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Row-wise quantile inversion for scaled-baseline models (ADR 0002).

        For each row ``i``, solve

            h_0(q_i) · exp(0.5 · x_s,i · γ) + x_d,i · β + offset_i = F⁻¹(p_i)

        for ``q_i`` via vectorised bisection.  Equivalently,

            h_0(q_i) = (F⁻¹(p_i) − x_d,i·β − offset_i) / exp(0.5 · x_s,i · γ)
                     =: z_i,

        and ``z_i`` is bracketed by ``[θ_b[0]+ε, θ_b[-1]−ε]`` — *the same*
        bracket as the shift-only path, because the scaling factor is folded
        into ``z`` rather than into the bracket itself.  Saturation outside
        that bracket triggers the same warning text used by the shift path.

        Parameters
        ----------
        probs:
            Probabilities in ``(0, 1)``, shape ``(m,)``.
        theta_b:
            Bernstein coefficient vector, length ``order + 1``.
        gamma:
            Scaling coefficients ``γ`` of length ``q_s``.
        X_scale:
            Scaling-design matrix of shape ``(m, q_s)``.
        xbeta:
            Optional per-row linear predictor ``X · β``, shape ``(m,)``.
        offset:
            Optional per-row offset, shape ``(m,)``.

        Returns
        -------
        NDArray of shape ``(m,)`` with values in ``basis.support``.
        """
        a, b = self.basis.support
        k = self.basis.order
        probs_arr = np.asarray(probs, dtype=np.float64)
        m = probs_arr.shape[0]
        if m == 0:
            return np.empty(0, dtype=float)

        dist = _get_dist(self.base_distribution)
        shift: NDArray[np.float64] = (
            np.zeros(m, dtype=np.float64)
            if xbeta is None
            else np.asarray(xbeta, dtype=np.float64)
        )
        if offset is not None:
            shift = shift + np.asarray(offset, dtype=np.float64)

        # f_i = exp(0.5 · x_s,i · γ); strictly positive.  Re-scale the
        # F⁻¹(p)−xβ target into the baseline-h scale so the existing bracket
        # [θ_b[0]+ε, θ_b[-1]−ε] applies row-wise without change.
        f_scale = np.exp(0.5 * (X_scale @ gamma))
        z_min = float(theta_b[0]) + _BRENTQ_EPS
        z_max = float(theta_b[-1]) - _BRENTQ_EPS
        z_raw = (dist.ppf(probs_arr) - shift) / f_scale
        if np.any((z_raw < z_min) | (z_raw > z_max)):
            warnings.warn(
                "predict(what='quantile'): (F⁻¹(p) − xβ − offset) / "
                "exp(0.5·x_s·γ) exceeds the basis bracket "
                f"[θ_b[0]+ε, θ_b[-1]−ε] = [{z_min:.4g}, {z_max:.4g}] at "
                "one or more points; clipping for numerical stability. "
                "Quantiles at these points are saturated at the basis "
                "endpoints, not the true asymptotic limit. Consider widening "
                "the basis support or restricting probs away from 0/1.",
                stacklevel=2,
            )
        z = np.clip(z_raw, z_min, z_max)

        # Reuse the same Bernstein bisection scaffolding as ``_predict_quantile``.
        i_arr = np.arange(k + 1, dtype=float)
        j_arr = k - i_arr
        binom_theta = comb(k, i_arr, exact=False) * theta_b
        inv_width = 1.0 / (b - a)

        def _h_vec(q: NDArray[np.float64]) -> NDArray[np.float64]:
            t = (q - a) * inv_width
            T = (t[:, None] ** i_arr) * ((1.0 - t)[:, None] ** j_arr)
            return cast(NDArray[np.float64], T @ binom_theta)

        lo = np.full(m, a, dtype=float)
        hi = np.full(m, b, dtype=float)
        mid = 0.5 * (lo + hi)
        for _ in range(60):
            if np.max(hi - lo) < _BRENTQ_EPS:
                break
            below = _h_vec(mid) < z
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
            mid = 0.5 * (lo + hi)
        return cast(NDArray[np.float64], mid)

    def _predict_quantile_interaction(
        self,
        probs: NDArray[np.float64],
        X: NDArray[np.float64],
        offset: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Row-wise quantile inversion for ``InteractionBasis`` models.

        For each row ``i``, solve ``h(q_i | x_i) = F⁻¹(probs_i) − offset_i``
        where ``h(y|x) = a(y)ᵀ Θ b(x)`` and ``F⁻¹`` is the base-distribution
        quantile.  Per-row bracket: ``[h(a, x_i) + ε, h(b, x_i) − ε]``.

        Parameters
        ----------
        probs:
            Probabilities in ``(0, 1)``, shape ``(m,)``.
        X:
            Covariate matrix, shape ``(m, ?)`` — already validated to match
            the y-length and to be 2-D by the caller.
        offset:
            Optional per-row offset added to ``h`` before inversion,
            shape ``(m,)`` or ``None``.

        Returns
        -------
        NDArray of shape ``(m,)`` with values in ``basis.support``.
        """
        assert isinstance(self.basis, InteractionBasis)
        assert self.theta_ is not None
        a, b = self.basis.support
        probs_arr = np.asarray(probs, dtype=np.float64)
        m = probs_arr.shape[0]
        if m == 0:
            return np.empty(0, dtype=float)

        p = self.basis.n_y_params
        q = self.basis.n_x_params
        Theta = self.theta_.reshape(p, q)  # (p, q)

        x_1d = self.basis._coerce_x(X)
        B_x = self.basis.x_basis.evaluate(x_1d)  # (m, q)
        # Per-row baseline-θ vector along the y-axis: theta_rows[i] = Θ · b(x_i).
        theta_rows = B_x @ Theta.T  # (m, p)

        # Endpoint values h(a, x_i) and h(b, x_i) define the per-row bracket.
        A_a = self.basis.y_basis.evaluate(np.full(1, a, dtype=float))[0]  # (p,)
        A_b = self.basis.y_basis.evaluate(np.full(1, b, dtype=float))[0]  # (p,)
        h_a = theta_rows @ A_a  # (m,)
        h_b = theta_rows @ A_b  # (m,)
        z_min = h_a + _BRENTQ_EPS
        z_max = h_b - _BRENTQ_EPS

        dist = _get_dist(self.base_distribution)
        shift: NDArray[np.float64] = (
            np.zeros(m, dtype=np.float64)
            if offset is None
            else np.asarray(offset, dtype=np.float64)
        )
        z_raw = dist.ppf(probs_arr) - shift
        if np.any((z_raw < z_min) | (z_raw > z_max)):
            warnings.warn(
                "predict(what='quantile'): F⁻¹(p) − offset exceeds the per-row "
                "bracket [h(a, x_i)+ε, h(b, x_i)−ε] at one or more points; "
                "clipping for numerical stability.  Quantiles at these points "
                "are saturated at the basis endpoints, not the true asymptotic "
                "limit. Consider widening the y-basis support or restricting "
                "probs away from 0/1.",
                stacklevel=3,
            )
        z = np.clip(z_raw, z_min, z_max)

        # Vectorised bisection: at each midpoint evaluate the y-basis once and
        # take the row-wise inner product with the per-row baseline-θ.
        lo = np.full(m, a, dtype=float)
        hi = np.full(m, b, dtype=float)
        mid = 0.5 * (lo + hi)
        for _ in range(60):
            if np.max(hi - lo) < _BRENTQ_EPS:
                break
            A_mid = self.basis.y_basis.evaluate(mid)  # (m, p)
            h_mid = np.einsum("ij,ij->i", A_mid, theta_rows)
            below = h_mid < z
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
            mid = 0.5 * (lo + hi)
        return cast(NDArray[np.float64], mid)

    def score(
        self,
        y: NDArray[np.float64] | CensoredData,
        X: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> float:
        """Log-likelihood at the fitted parameters (sklearn-compatible).

        Higher is better; this is NOT the negative log-likelihood.

        Parameters
        ----------
        y:
            Response observations.
        X:
            Optional covariate matrix.
        weights:
            Optional per-observation weights.
        offset:
            Optional per-observation offset added to ``h``.

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
        n = int(y_clean.n) if isinstance(y_clean, CensoredData) else len(y_clean)
        weights_clean, offset_clean = _validate_weights_offset(weights, offset, n)
        cens = CensoringType.NONE if self.censoring is None else self.censoring
        return log_likelihood(
            self.theta_,
            self.basis,
            y_clean,
            X_clean,
            cens,
            base_distribution=self.base_distribution,
            weights=weights_clean,
            offset=offset_clean,
        )

    _ACTIVE_CONSTRAINT_TOL: float = 1e-8

    def vcov(self, regularize: str | None = "active") -> NDArray[np.float64]:
        """Asymptotic variance–covariance matrix of :attr:`theta_`.

        Returns the inverse of the observed information matrix
        :attr:`hessian_` (Hessian of the *negative* log-likelihood at the
        MLE).  Under standard regularity conditions, this is a consistent
        estimator of the asymptotic covariance of the maximum-likelihood
        estimator.

        Parameters
        ----------
        regularize : {'active', 'auglag', None}, default 'active'
            Regularization strategy for near-singular Hessians.

            * ``'active'`` — if direct inversion fails, recover a finite
              covariance via the *active-set-constrained* form: the top-left
              block of the inverse of the bordered KKT matrix ``[[H, A_aᵀ],
              [A_a, 0]]``, where ``A_a`` is the sub-matrix of rows of
              ``_A_ineq_`` whose KKT multiplier exceeds
              :attr:`_ACTIVE_CONSTRAINT_TOL` (see
              :meth:`_constrained_vcov_active`).  When auglag data are not
              available (SLSQP / trust-constr fits) the pseudoinverse is used
              as a fallback.  This is the default because the Hessian can be
              singular at constrained MLEs, and on well-conditioned fits it
              reduces to bare ``H⁻¹`` (R ``mlt::vcov.mlt`` behaves the same
              way in the cases where mltpy's bare ``inv(H)`` already matches R
              — see ``tests/test_confidence.py``).
            * ``'auglag'`` — *always* return the active-set-constrained
              covariance when active monotonicity rows exist, rather than
              waiting for bare inversion to fail.  This is the ρ→∞ limit of
              the penalty form ``(H + ρ·A_aᵀA_a)⁻¹`` and mirrors R
              ``mlt::vcov.mlt`` on the constrained branches that bare
              ``inv(H)`` misses (notably the scaled-baseline Coxph path, where
              bare ``inv(H)`` diverges from R's ``vcov(as.mlt(fit))`` by ~37×
              on the binding rows while the constrained form matches at
              ``rtol≈1e-4``).  Unlike the earlier penalty implementation it
              does not depend on the optimiser's final penalty ``ρ`` (which
              the augmented-Lagrangian now freezes once feasible).  Falls back
              to bare ``H`` when no constraint binds or auglag data are
              unavailable.  Opt-in because it inflates standard errors along
              tied rows and consequently widens ``confint`` / ``confband``
              outputs in cases where mltpy's bare ``inv(H)`` already matches R.
            * ``None`` — raise ``RuntimeError`` on singular Hessian (original
              behaviour; useful when you need a diagnostic failure).

        Returns
        -------
        NDArray[np.float64]
            Symmetric ``(p+q, p+q)`` matrix.

        Raises
        ------
        ValueError
            If *regularize* is not ``'active'``, ``'auglag'``, or ``None``.
        NotFittedError
            If called before :meth:`fit`.
        RuntimeError
            If the Hessian is singular and ``regularize=None``, or if
            ``hessian_`` is unexpectedly missing after fitting.
        """
        self._check_is_fitted()
        if self.hessian_ is None:
            raise RuntimeError(
                "hessian_ is unexpectedly missing after fitting. "
                "Please call fit(y) again."
            )
        if regularize not in ("active", "auglag", None):
            raise ValueError(
                f"regularize must be 'active', 'auglag', or None, got {regularize!r}"
            )

        # 'auglag': pre-augment unconditionally when active constraints exist.
        # The other two modes go through bare inv first.
        if regularize == "auglag":
            V_constrained = self._constrained_vcov_active(self.hessian_)
            if V_constrained is not None:
                return V_constrained
            # No binding constraint / no auglag artefacts: bare inverse, with a
            # graceful pseudoinverse rather than raising — 'auglag' is the
            # opt-in regularising mode and shouldn't fail loudly.
            try:
                return cast(NDArray[np.float64], np.linalg.inv(self.hessian_))
            except np.linalg.LinAlgError:
                return cast(NDArray[np.float64], np.linalg.pinv(self.hessian_))

        try:
            return cast(NDArray[np.float64], np.linalg.inv(self.hessian_))
        except np.linalg.LinAlgError as exc:
            if regularize != "active":
                raise RuntimeError(
                    "vcov() could not be computed: the Hessian matrix is singular "
                    "or ill-conditioned.  Possible causes: active monotonicity "
                    "constraint at the MLE, basis order too high relative to "
                    "sample size, or collinear covariates.  "
                    "Pass regularize='active' (penalty fallback) or 'auglag' "
                    "(unconditional augmentation) to recover a finite vcov."
                ) from exc

        # regularize='active' fallback: bare inv(H) failed, so recover a finite
        # covariance via the active-set-constrained (bordered KKT) form.
        V_constrained = self._constrained_vcov_active(self.hessian_)
        if V_constrained is not None:
            return V_constrained
        return cast(NDArray[np.float64], np.linalg.pinv(self.hessian_))

    def _constrained_vcov_active(
        self, H: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        """Constrained covariance under the active monotonicity constraints.

        Returns the ``(p+q, p+q)`` covariance of the constrained MLE — the
        top-left block of the inverse of the bordered KKT matrix

        .. math::
            K = \\begin{bmatrix} H & A_a^\\top \\\\ A_a & 0 \\end{bmatrix},

        where ``A_a`` collects the rows of ``_A_ineq_`` whose KKT multiplier
        (``result_.mu_ineq``) exceeds :attr:`_ACTIVE_CONSTRAINT_TOL`.  For a
        non-singular ``H`` this equals the standard projection
        ``H⁻¹ − H⁻¹A_aᵀ(A_a H⁻¹A_aᵀ)⁻¹A_a H⁻¹``; it is the ρ→∞ limit of the
        penalty form ``(H + ρ·A_aᵀA_a)⁻¹`` and matches R ``mlt::vcov.mlt`` on
        binding rows (where pinned coefficients share identical covariance
        rows).  Crucially it is *independent* of the optimiser's final penalty
        ``ρ`` — the earlier penalty form silently depended on ``rho_final``
        being driven very large, which is no longer the case now that the
        augmented-Lagrangian freezes ρ once feasible.

        Returns ``None`` when the auglag active-set metadata is unavailable
        (SLSQP / trust-constr fits) or no constraint binds, so the caller can
        fall back to bare ``H⁻¹``.  Uses ``numpy.linalg.pinv`` if the bordered
        matrix is itself singular (e.g. an injected rank-deficient ``H``).
        """
        if (
            self._A_ineq_ is None
            or self.result_ is None
            or self.result_.mu_ineq is None
        ):
            return None
        active_mask = self.result_.mu_ineq > self._ACTIVE_CONSTRAINT_TOL
        if not np.any(active_mask):
            return None
        A_active = self._A_ineq_[active_mask, :]
        n = H.shape[0]
        m = A_active.shape[0]
        K = np.zeros((n + m, n + m), dtype=np.float64)
        K[:n, :n] = H
        K[:n, n:] = A_active.T
        K[n:, :n] = A_active
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)
        V: NDArray[np.float64] = K_inv[:n, :n]
        return V

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
        RuntimeError
            If the cached score matrix is unexpectedly missing after
            fitting (e.g. a prior ``fit()`` call failed partway through).
        """
        self._check_is_fitted()
        if self._estfun_cache_ is None:
            raise RuntimeError(
                "_estfun_cache_ is unexpectedly missing after fitting. "
                "Please call fit(y) again."
            )
        return self._estfun_cache_

    # R/sandwich-style alias.  Kept as a method (not a bare attribute) so it
    # dispatches on subclass overrides if any.
    def score_contributions(self) -> NDArray[np.float64]:
        """Alias for :meth:`estfun`.  See that method for details."""
        return self.estfun()

    def residuals(
        self,
        type: Literal["score", "cox-snell", "deviance"] = "score",
    ) -> NDArray[np.float64]:
        """Per-observation residuals for model diagnostics.

        Computed at the training data passed to :meth:`fit`.  Mirrors R
        ``mlt::residuals`` for ``type="score"``; the Cox-Snell and deviance
        forms are derived from the fitted survivor function.

        Parameters
        ----------
        type:
            Which residual to compute.

            * ``"score"`` (default) — score residual w.r.t. an artificial
              intercept added to ``h(y|x)``: for exact ``-ψ(h_i)``; for
              right-censored ``f(h)/S(h)``; for left-censored
              ``-f(h)/F(h)``; for interval
              ``-(f(h_b) - f(h_a)) / (F(h_b) - F(h_a))``.  Sign matches R
              ``mlt::residuals`` (the negative of the positive-log-likelihood
              score).  At the MLE the sum is zero up to optimiser tolerance.
            * ``"cox-snell"`` — ``r_i = -log S(y_i|x_i)``.  Under a correctly
              specified model these are approximately ``Exp(1)``.  For
              censored observations ``y_i`` is the censoring threshold
              (``lower`` for right-censored, ``upper`` for left-censored,
              the midpoint for interval-censored); the resulting residuals
              for those observations are themselves censored ``Exp(1)``
              variates.
            * ``"deviance"`` — ``sign(r_i - 1) · sqrt(2·|r_i - log(r_i) - 1|)``
              where ``r_i`` is the Cox-Snell residual.  Under a correctly
              specified model these are approximately standard normal.

        Returns
        -------
        NDArray[np.float64]
            Vector of length :attr:`n_obs_`.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``type`` is not one of the supported residual kinds.

        Notes
        -----
        - ``type="score"`` matches R ``mlt::residuals(mlt_fit)`` exactly,
          element-wise to ``rtol=1e-6``.
        - ``type="cox-snell"`` uses the same evaluation point convention as
          R's ``-log(predict(mlt_fit, type = "survivor"))``.
        - ``r_i`` is clipped at ``np.finfo(float).tiny`` before
          ``log(r_i)`` in the deviance formula to avoid ``-inf``.
        """
        self._check_is_fitted()
        if isinstance(self.basis, InteractionBasis):
            raise NotImplementedError(
                "residuals() is not supported for InteractionBasis models."
            )
        if type not in {"score", "cox-snell", "deviance"}:
            raise ValueError(
                f"type={type!r} is invalid. Allowed: 'score', 'cox-snell', 'deviance'."
            )
        if self.theta_ is None or self._y_train_ is None:
            raise RuntimeError(
                "Training data is unexpectedly missing after fitting. "
                "Please call fit(y) again."
            )

        if type == "score":
            # R's ``mlt::residuals`` returns the negative of the
            # positive-log-likelihood intercept score (so the residual is
            # interpretable as ``∂(-ℓ_i)/∂α``).  Negate to match.
            # Under scaling (ADR 0002) the hypothetical intercept α is
            # added to the *final* h (post-scaling, post-shift), so the
            # closed-form score formulas apply unchanged once h is
            # evaluated at the scaled value h_0(y)·exp(0.5·X_s·γ)+Xβ.
            assert isinstance(self.basis, BernsteinBasis)
            cens_r = CensoringType.NONE if self.censoring is None else self.censoring
            return -_intercept_score(
                self.theta_,
                self.basis,
                self._y_train_,
                self._X_train_,
                cens_r,
                base_distribution=self.base_distribution,
                weights=self._weights_train_,
                offset=self._offset_train_,
                scaling=self.scaling,
            )

        # Cox-Snell / deviance: evaluate -log S(y|x) at a single point per
        # observation.  Pick the point per censoring status of each row.
        y_eval = self._cox_snell_eval_points()
        p = self.basis.order + 1
        theta_b = self.theta_[:p]
        B = self.basis.evaluate(y_eval)
        h = B @ theta_b
        if self._X_train_ is not None and len(self.theta_) > p:
            beta = self.theta_[p:]
            h = h + self._X_train_ @ beta
        if self._offset_train_ is not None:
            h = h + self._offset_train_
        h_c = np.clip(h, -_H_CLIP, _H_CLIP)
        dist = _get_dist(self.base_distribution)
        r = -dist.logsf(h_c)

        if type == "cox-snell":
            return r

        # deviance
        r_safe = np.clip(r, np.finfo(float).tiny, None)
        sign = np.sign(r - 1.0)
        return cast(
            NDArray[np.float64],
            sign * np.sqrt(2.0 * np.abs(r - np.log(r_safe) - 1.0)),
        )

    def _cox_snell_eval_points(self) -> NDArray[np.float64]:
        """Per-observation evaluation point for ``-log S`` residuals.

        Exact obs: the observed value.  Right-censored: ``lower`` (censoring
        time).  Left-censored: ``upper`` (censoring time).  Interval-censored:
        midpoint of ``[lower, upper]``.
        """
        y = self._y_train_
        if isinstance(y, np.ndarray):
            return y
        if not isinstance(y, CensoredData):
            raise RuntimeError(
                f"Unexpected stored training response type: {type(y).__name__!r}."
            )
        out = np.empty(y.n, dtype=np.float64)
        mask_e = y.is_exact_mask
        out[mask_e] = y.exact[mask_e]
        mask_r = y.is_right_censored_mask
        out[mask_r] = y.lower[mask_r]
        mask_l = y.is_left_censored_mask
        out[mask_l] = y.upper[mask_l]
        mask_i = y.is_interval_censored_mask
        out[mask_i] = 0.5 * (y.lower[mask_i] + y.upper[mask_i])
        return out

    def standard_errors(self, regularize: str | None = "active") -> NDArray[np.float64]:
        """Vector of asymptotic standard errors for :attr:`theta_`.

        Computed as ``sqrt(diag(vcov(regularize=regularize)))``.  Length
        equals ``len(theta_)``.

        Parameters
        ----------
        regularize : {'active', None}, default 'active'
            Passed directly to :meth:`vcov`.  See that method's documentation
            for details.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        RuntimeError
            Propagated from :meth:`vcov` if the Hessian is singular and
            ``regularize=None``.
        """
        diag = np.diag(self.vcov(regularize=regularize))
        if np.any(diag < 0):
            raise RuntimeError(
                "vcov() contains negative diagonal entries — the Hessian "
                "matrix is not positive definite.  The model may not be "
                "identified or the optimisation may have stalled at a "
                "saddle point."
            )
        return cast(NDArray[np.float64], np.sqrt(diag))

    def sandwich_vcov(self, regularize: str | None = "active") -> NDArray[np.float64]:
        """Sandwich (robust) variance–covariance matrix of :attr:`theta_`.

        Computes the HC0 sandwich estimator

        .. math::
            V_{\\text{sand}} = B M B,
            \\quad B = \\mathrm{vcov}(\\mathrm{regularize}),
            \\quad M = \\sum_i s_i s_i^\\top,

        where :math:`B` is the *bread* — the inverse observed information
        computed by :meth:`vcov` — and :math:`s_i` is the per-observation
        score (row :math:`i` of :meth:`estfun`).

        The ``regularize`` parameter is forwarded to :meth:`vcov`, so the
        bread inherits the same penalty-augmented Hessian recovery as
        ``vcov(regularize='active')`` (the default).

        Parameters
        ----------
        regularize : {'active', None}, default 'active'
            Passed directly to :meth:`vcov`.  See that method's documentation
            for details.

        Returns
        -------
        NDArray[np.float64]
            Symmetric ``(p+q, p+q)`` matrix.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If *regularize* is not ``'active'`` or ``None``.
        RuntimeError
            If the Hessian is singular and ``regularize=None``.
        """
        self._check_is_fitted()
        bread = self.vcov(regularize=regularize)
        ef = self.estfun()
        meat = ef.T @ ef
        return bread @ meat @ bread

    def sandwich_se(self, regularize: str | None = "active") -> NDArray[np.float64]:
        """Sandwich (robust) standard errors for :attr:`theta_`.

        Computed as ``sqrt(diag(sandwich_vcov(regularize=regularize)))``.

        Parameters
        ----------
        regularize : {'active', None}, default 'active'
            Passed directly to :meth:`sandwich_vcov`.  See :meth:`vcov` for
            details.

        Returns
        -------
        NDArray[np.float64]
            Vector of length ``len(theta_)``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        RuntimeError
            Propagated from :meth:`sandwich_vcov` on singular Hessians, or if
            the sandwich variance matrix has negative diagonal entries.
        """
        diag = np.diag(self.sandwich_vcov(regularize=regularize))
        if np.any(diag < 0):
            raise RuntimeError(
                "sandwich_vcov() contains negative diagonal entries — the "
                "sandwich matrix is not positive definite."
            )
        return cast(NDArray[np.float64], np.sqrt(diag))

    def wald_test(
        self,
        R: NDArray[np.float64],
        r: NDArray[np.float64] | None = None,
        vcov: Literal["information", "sandwich"] = "information",
        regularize: str | None = "active",
    ) -> "WaldTestResult":
        """Wald test for linear restrictions ``Rθ = r``.

        Computes the chi-squared Wald statistic

        .. math::
            W = (R\\hat\\theta - r)^\\top
                \\bigl[R\\,V\\,R^\\top\\bigr]^{-1}
                (R\\hat\\theta - r)
            \\;\\sim\\; \\chi^2(k),

        where :math:`k` is the number of rows in :math:`R` and :math:`V` is
        either the inverse-information :meth:`vcov` or the sandwich estimator
        :meth:`sandwich_vcov`.

        Parameters
        ----------
        R : NDArray[np.float64]
            Contrast matrix of shape ``(k, p+q)``.  Each row encodes one
            linear restriction on :attr:`theta_`.
        r : NDArray[np.float64] | None
            Null-hypothesis value vector of length ``k``.  Defaults to the
            zero vector (i.e. ``Rθ = 0``).
        vcov : ``"information"`` | ``"sandwich"``
            Which variance–covariance matrix to use.  ``"information"`` (the
            default) uses the observed Fisher information :meth:`vcov`;
            ``"sandwich"`` uses the HC0 sandwich estimator
            :meth:`sandwich_vcov`.
        regularize : str | None
            Passed directly to :meth:`vcov` (or :meth:`sandwich_vcov`).  See
            :meth:`vcov` for the accepted values and their effect.  Default
            ``"active"`` applies penalty-augmented Hessian recovery when
            inversion fails.

        Returns
        -------
        WaldTestResult
            Dataclass with fields ``statistic``, ``df``, ``p_value``, and
            ``vcov_type``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``R`` does not have ``len(theta_)`` columns or ``r`` has the
            wrong length.
        RuntimeError
            If ``R V R^T`` is singular (the restriction is degenerate or
            collinear).
        """
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError(
                "Model parameters (theta_) are unexpectedly missing after fitting."
            )
        R = np.atleast_2d(np.asarray(R, dtype=np.float64))
        k, p = R.shape
        if p != self.theta_.size:
            raise ValueError(
                f"R has {p} columns but model has {self.theta_.size} parameters."
            )
        if r is None:
            r_vec = np.zeros(k, dtype=np.float64)
        else:
            r_vec = np.asarray(r, dtype=np.float64).ravel()
            if r_vec.size != k:
                raise ValueError(f"r has {r_vec.size} elements but R has {k} rows.")
        if vcov == "information":
            V = self.vcov(regularize=regularize)
        else:
            V = self.sandwich_vcov(regularize=regularize)
        diff = R @ self.theta_ - r_vec
        RVR = R @ V @ R.T
        try:
            RVR_inv = np.linalg.inv(RVR)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "wald_test(): R V R^T is singular — the restrictions may be "
                "collinear or the model is not identified."
            ) from exc
        W = float(diff @ RVR_inv @ diff)
        p_value = float(chi2.sf(W, df=k))
        return WaldTestResult(statistic=W, df=k, p_value=p_value, vcov_type=vcov)

    def confint(
        self,
        level: float = 0.95,
        parm: Sequence[int] | None = None,
        type: Literal["wald", "profile"] = "wald",
    ) -> NDArray[np.float64]:
        """Confidence intervals for :attr:`theta_`.

        Two interval types are supported:

        * ``type="wald"`` (default) — symmetric normal-approximation
          interval

          .. math::
              \\hat\\theta_j \\pm z_{1-\\alpha/2}\\,\\sqrt{V_{jj}},

          where :math:`V = \\mathrm{vcov}()` is the inverse observed
          information matrix and :math:`z_{1-\\alpha/2}` is the standard
          normal quantile for confidence ``level`` :math:`= 1-\\alpha`.
          Matches R ``confint.default(mlt_fit, level=level)``.

        * ``type="profile"`` — profile-likelihood interval obtained by
          inverting the :math:`\\chi^2_1` likelihood-ratio test.  For each
          requested parameter index :math:`j` we solve

          .. math::
              2\\,(\\hat\\ell - \\ell_p(v)) = \\chi^2_{1,1-\\alpha},

          where :math:`\\ell_p(v)` is the maximised log-likelihood with
          :math:`\\theta_j` pinned to :math:`v` and the remaining
          parameters re-optimised under the model constraints.  Each
          parameter costs roughly ten constrained refits, so always pass
          ``parm=`` to restrict the work on larger models.

          Robustness (issue #89): three inner-fit failure modes can occur
          per parameter — (i) the adaptive bracket fails to span a sign
          change, (ii) the pinned refit lands on a degenerate monotonicity
          active set so the equality ``theta[j] = v`` cannot be honoured
          ("boundary"), or (iii) the pinned refit does not converge to
          tolerance ("convergence", KKT residual ≥
          ``_PROFILE_INNER_KKT_THRESHOLD``).  When ``parm is None`` (you
          asked for every parameter) each failure emits a
          :class:`ConvergenceWarning` naming the parameter and writes
          ``±np.inf`` (bracket / boundary) or ``np.nan`` (convergence) to
          that row, so one un-identified parameter does not abort the
          whole call.  When ``parm`` is an explicit sequence (you asked
          for those parameters specifically) the same failures re-raise
          as :class:`RuntimeError` so you can debug the request.

        Parameters
        ----------
        level:
            Confidence level in ``(0, 1)``.  Defaults to ``0.95``.
        parm:
            Optional sequence of integer indices selecting a subset of
            parameters.  ``None`` returns intervals for all entries of
            :attr:`theta_`.
        type:
            Interval type.  ``"wald"`` (default) preserves the existing
            normal-approximation behaviour; ``"profile"`` returns the
            likelihood-ratio interval.

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
            If ``level`` is outside ``(0, 1)``, ``parm`` contains indices
            outside ``[0, len(theta_))``, or ``type`` is not one of
            ``{"wald", "profile"}``.
        RuntimeError
            Propagated from :meth:`vcov` on singular Hessians (Wald), or
            from the profile-CI bracket search / inner-fit failure when
            an explicit ``parm`` was provided (profile).  Under
            ``parm=None`` the same failures become
            :class:`ConvergenceWarning` instead.

        Examples
        --------
        >>> model = MLT(order=4, support=(0, 1)).fit(y)
        >>> ci = model.confint(level=0.95)  # shape (p, 2)
        >>> ci_prof = model.confint(level=0.95, parm=[0], type="profile")
        """
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError(
                "Model parameters (theta_) are unexpectedly missing after fitting."
            )
        if not (0.0 < level < 1.0):
            raise ValueError(f"level={level!r} is invalid. Expected: 0 < level < 1.")
        if type not in ("wald", "profile"):
            raise ValueError(
                f"type={type!r} is invalid. Expected one of {{'wald', 'profile'}}."
            )

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

        if type == "profile":
            return self._confint_profile(level=level, idx=idx, strict=parm is not None)

        se = self.standard_errors()
        z = float(norm.ppf(0.5 * (1.0 + level)))
        est = self.theta_[idx]
        half = z * se[idx]
        return np.column_stack((est - half, est + half))

    # Adaptive-bracket parameters for the profile-CI root search.  See the
    # docstring on :meth:`_confint_profile` for the algorithm; constants are
    # exposed as class attributes so that downstream tests / advanced users
    # can override them on a subclass without monkey-patching the method.
    _PROFILE_BRACKET_INIT: float = 3.0
    _PROFILE_BRACKET_MAX_DOUBLINGS: int = 24
    _PROFILE_BRENTQ_XTOL: float = 1e-6
    # Detection thresholds for inner-fit failure inside profile-CI; see
    # :meth:`_profile_loglik_at` and #89.  ``_PROFILE_INNER_KKT_THRESHOLD``
    # is intentionally four orders of magnitude above auglag's nominal
    # 1e-5 KKT target — CLAUDE.md notes that pinned refits on
    # stacked-active-set fixtures (BoxCox + small N, Coxph with
    # neighbouring Bernstein boundary, etc.) routinely emit
    # ``converged=False`` with KKT ~ 1e-3 yet still produce a meaningful
    # ``log_likelihood`` that R agrees with.  Only flag as a true
    # convergence failure when the residual is *clearly* off (≥ 0.1).
    # ``_PROFILE_INNER_PIN_TOL`` is the equality-violation tolerance — a
    # post-fit ``|theta[j] - v| > 1e-6`` means the pin couldn't be
    # honoured under the constraints, which is the unambiguous
    # boundary-conflict signal.
    _PROFILE_INNER_KKT_THRESHOLD: float = 1e-1
    _PROFILE_INNER_PIN_TOL: float = 1e-6

    def _profile_loglik_at(self, j: int, v: float) -> float:
        """Maximised log-likelihood with ``theta_[j]`` pinned to ``v``.

        Refits the model on the cached training data (``_y_train_`` etc.)
        using a fresh :class:`OptimizerConfig` that copies every field of
        ``self.optimizer_config`` and replaces ``fixed_params`` with
        ``{j: v}``.  ``ConvergenceWarning`` from the inner fit is
        suppressed — failed-to-converge inner fits show up as a larger
        residual on the outer bracket search and are surfaced from
        :meth:`_confint_profile` if they prevent a sign change.

        Parameters
        ----------
        j:
            Index into :attr:`theta_` to pin.  Must lie in
            ``[0, len(theta_))``; the underlying call validates this.
        v:
            Pinned value for ``theta_[j]``.

        Returns
        -------
        float
            ``ℓ_p(v)`` — the maximised log-likelihood subject to
            ``theta_[j] = v`` and the model's existing monotonicity /
            boundary / equality constraints.
        """
        # ``_y_train_`` is populated by ``fit()``; callers reach this helper
        # only via ``confint(type="profile")``, which has already passed
        # ``_check_is_fitted()``.  The guard is here purely to satisfy
        # type-narrowing on the cached attributes.
        if self._y_train_ is None:
            raise RuntimeError(
                "Training data cache is unavailable; profile-CI requires "
                "the model to have been fit on this instance."
            )

        base_config = (
            self.optimizer_config
            if self.optimizer_config is not None
            else OptimizerConfig()
        )
        pinned_config = replace(
            base_config,
            fixed_params={int(j): float(v)},
        )
        censoring_arg = CensoringType.NONE if self.censoring is None else self.censoring
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = optimize(
                self.basis,
                self._y_train_,
                X=self._X_train_,
                censoring=censoring_arg,
                config=pinned_config,
                base_distribution=self.base_distribution,
                weights=self._weights_train_,
                offset=self._offset_train_,
                scaling=self.scaling,
            )

        # Detect the two failure modes #89 distinguishes.  Theta drift past
        # ``_PROFILE_INNER_PIN_TOL`` from the pin means the equality
        # ``theta[j] = v`` could not be honoured under the active
        # monotonicity constraints — flag as "boundary".  A high KKT
        # residual with the pin still honoured means the inner auglag
        # simply couldn't reach its tolerance budget — flag as
        # "convergence".  A successful run with ``converged=True`` and a
        # moderate KKT residual passes through unmodified.
        pin_drift = float(abs(result.theta[int(j)] - float(v)))
        kkt = result.kkt_residual
        if pin_drift > self._PROFILE_INNER_PIN_TOL:
            raise _ProfileInnerFailure(
                j=int(j),
                kind="boundary",
                diagnostic=(
                    f"pinned refit returned theta[{int(j)}]="
                    f"{float(result.theta[int(j)]):.6g} but was pinned to "
                    f"{float(v):.6g} (drift {pin_drift:.3g} > "
                    f"{self._PROFILE_INNER_PIN_TOL:.0e}); the equality "
                    "conflicts with the monotonicity active set"
                ),
            )
        if (
            not result.converged
            and kkt is not None
            and kkt > self._PROFILE_INNER_KKT_THRESHOLD
        ):
            raise _ProfileInnerFailure(
                j=int(j),
                kind="convergence",
                diagnostic=(
                    f"pinned refit did not converge (kkt_residual={kkt:.3g} "
                    f"> {self._PROFILE_INNER_KKT_THRESHOLD:.0e}); solver "
                    f"message: {result.solver_message!r}"
                ),
            )
        return float(result.log_likelihood)

    def _confint_profile(
        self,
        level: float,
        idx: NDArray[np.intp],
        strict: bool = True,
    ) -> NDArray[np.float64]:
        """Profile-likelihood confidence intervals for ``self.theta_[idx]``.

        For each requested index ``j``, finds the two roots of

        .. math::
            f(v) = 2\\,(\\hat\\ell - \\ell_p(v)) - \\chi^2_{1, 1-\\alpha}

        by :func:`scipy.optimize.brentq`.  The bracket is grown adaptively:
        starting at ``θ̂_j ± 3·se_j``, the multiplier doubles up to 24
        times until ``f`` changes sign.  Failure to bracket within the
        widest multiplier raises :class:`RuntimeError` naming the
        parameter and the largest ``|f|`` value observed.

        Parameters
        ----------
        level:
            Confidence level, already validated by :meth:`confint`.
        idx:
            Indices into :attr:`theta_` to compute intervals for.

        Returns
        -------
        NDArray[np.float64]
            ``(len(idx), 2)`` array with columns ``[lower, upper]``.
        """
        if self.theta_ is None or self.result_ is None:
            raise RuntimeError(
                "Model parameters are unexpectedly missing after fitting."
            )

        se = self.standard_errors()
        theta_hat = self.theta_
        ll_hat = float(self.result_.log_likelihood)
        crit = float(chi2.ppf(level, df=1))

        out = np.empty((idx.size, 2), dtype=np.float64)
        for row, raw_j in enumerate(idx):
            j = int(raw_j)
            th = float(theta_hat[j])
            s = float(se[j])
            if not np.isfinite(s) or s <= 0.0:
                raise RuntimeError(
                    f"Profile-CI requires a finite, positive Wald SE to "
                    f"seed the bracket search; se[{j}] = {s!r}. The "
                    f"Hessian may be singular at this parameter — try "
                    f"confint(type='wald') with regularize='active' "
                    f"first to diagnose."
                )

            def f(v: float, _j: int = j) -> float:
                return 2.0 * (ll_hat - self._profile_loglik_at(_j, v)) - crit

            out[row, 0] = self._profile_root(
                f, anchor=th, step=s, j=j, side="lower", strict=strict
            )
            out[row, 1] = self._profile_root(
                f, anchor=th, step=s, j=j, side="upper", strict=strict
            )
        return out

    def _profile_root(
        self,
        f: Callable[[float], float],
        *,
        anchor: float,
        step: float,
        j: int,
        side: Literal["lower", "upper"],
        strict: bool = True,
    ) -> float:
        """Adaptive bracket then ``brentq`` for one side of one parameter.

        Returns the root ``v`` such that ``f(v) ≈ 0``.  If no sign change
        is observed within the widest bracket multiplier the behaviour
        splits on ``strict``:

        * ``strict=True`` (caller passed ``parm=[j]``) — raise
          :class:`RuntimeError` with the largest ``|f|`` observed and the
          widest bracket multiplier tried, so the caller can decide
          whether to widen further.
        * ``strict=False`` (caller passed ``parm=None``) — emit a
          :class:`ConvergenceWarning` naming the parameter and return
          ``-np.inf`` (lower side) or ``np.inf`` (upper side), so a
          single un-bracketable parameter does not abort the whole call.
        """
        sign = -1.0 if side == "lower" else 1.0
        best_abs_f = -np.inf
        endpoint = anchor
        widest_mult = 0.0
        for k in range(1, self._PROFILE_BRACKET_MAX_DOUBLINGS + 1):
            mult = self._PROFILE_BRACKET_INIT * (2.0 ** (k - 1))
            widest_mult = mult
            endpoint = anchor + sign * mult * step
            try:
                f_end = f(endpoint)
            except _ProfileInnerFailure as exc:
                return self._handle_inner_failure(exc, side=side, strict=strict)
            if abs(f_end) > best_abs_f:
                best_abs_f = abs(f_end)
            if f_end > 0.0:
                break
        else:
            msg = (
                f"Profile-CI bracket search for parameter {j} on the "
                f"{side} side did not change sign after "
                f"{self._PROFILE_BRACKET_MAX_DOUBLINGS} doublings (widest "
                f"bracket multiplier {widest_mult:.6g}·se); the largest "
                f"|f| observed was {best_abs_f:.6g}. The Wald SE may be "
                f"too small to span the LR critical value, or theta_[{j}] "
                f"may lie on a constraint boundary — try checking "
                f"model.standard_errors()[{j}] and "
                f"model.confint(type='wald')."
            )
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg, ConvergenceWarning, stacklevel=2)
            return -np.inf if side == "lower" else np.inf
        lo, hi = (endpoint, anchor) if side == "lower" else (anchor, endpoint)
        try:
            return float(brentq(f, lo, hi, xtol=self._PROFILE_BRENTQ_XTOL))
        except _ProfileInnerFailure as exc:
            return self._handle_inner_failure(exc, side=side, strict=strict)

    def _handle_inner_failure(
        self,
        exc: _ProfileInnerFailure,
        *,
        side: Literal["lower", "upper"],
        strict: bool,
    ) -> float:
        """Translate a :class:`_ProfileInnerFailure` into the user-facing
        return value (or ``RuntimeError``) for the current side.

        The mapping per #89:

        =====================  ===========================  =====================
        ``kind``               ``strict=False`` (parm=None)  ``strict=True``
        =====================  ===========================  =====================
        ``"boundary"``         warn + ``±np.inf`` (signed)   ``RuntimeError``
        ``"convergence"``      warn + ``np.nan``             ``RuntimeError``
        =====================  ===========================  =====================
        """
        msg = (
            f"Profile-CI {exc.kind} failure for parameter {exc.j} on the "
            f"{side} side: {exc.diagnostic}"
        )
        if strict:
            raise RuntimeError(msg) from exc
        warnings.warn(msg, ConvergenceWarning, stacklevel=3)
        if exc.kind == "boundary":
            return -np.inf if side == "lower" else np.inf
        return float("nan")

    def confband(
        self,
        y_grid: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        level: float = 0.95,
        what: Literal[
            "trafo", "distribution", "survivor", "density", "hazard"
        ] = "distribution",
        offset: NDArray[np.float64] | None = None,
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
        offset:
            Optional per-grid-point offset of shape ``(len(y_grid),)``.
            Added to ``h`` before computing the band; does not affect the
            delta-method Jacobian (offset is constant w.r.t. ``theta``).

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
        if isinstance(self.basis, InteractionBasis):
            raise NotImplementedError(
                "confband() is not supported for InteractionBasis models."
            )
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
        offset_arr: NDArray[np.float64] | None = (
            _validate_offset(offset, m) if offset is not None else None
        )
        V = self.vcov()

        B = self.basis.evaluate(y_arr)  # (m, p)
        D = self.basis.derivative(y_arr, order=1)  # (m, p)
        h = B @ theta_b
        hp = D @ theta_b
        if x_row is not None and beta is not None:
            h = h + float(x_row @ beta)

        if offset_arr is not None:
            h = h + offset_arr

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
        X_scale: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Draw samples from the fitted model via the quantile transformation.

        Samples ``u ~ Uniform(0, 1)`` and returns
        ``predict(u, X, X_scale_new=X_scale, what="quantile")``.

        Parameters
        ----------
        n:
            Number of samples to draw.
        X:
            Covariate matrix of shape ``(n, q)``.  Each row yields one
            conditional draw; must be supplied when the model was fitted
            with covariates.  Pass ``None`` only for covariate-free fits.
        random_state:
            Seed or :class:`numpy.random.Generator` for reproducibility.
        X_scale:
            Scaling-design matrix of shape ``(n, q_s)``.  Required when the
            model was fitted with ``scaling=``; ignored otherwise.  Each
            row yields one heteroskedastic conditional draw via
            ``q_i = h_0⁻¹((Φ⁻¹(u_i) − x_d,i·β) / exp(0.5·x_s,i·γ))``.

        Returns
        -------
        NDArray of shape ``(n,)`` with values in ``basis.support``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` is provided but its number of rows does not equal ``n``,
            or if ``X_scale`` shape is inconsistent with the fit.
        """
        self._check_is_fitted()

        if X is not None:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]
            if X_arr.shape[0] != n:
                raise ValueError(
                    f"X has {X_arr.shape[0]} rows but n={n}; simulate() "
                    "draws one observation per row of X, so the counts "
                    "must match."
                )
        else:
            X_arr = None

        X_scale_arr: NDArray[np.float64] | None = None
        if X_scale is not None:
            X_scale_arr = np.asarray(X_scale, dtype=float)
            if X_scale_arr.ndim == 1:
                X_scale_arr = X_scale_arr[:, None]
            if X_scale_arr.shape[0] != n:
                raise ValueError(
                    f"X_scale has {X_scale_arr.shape[0]} rows but n={n}; "
                    "simulate() draws one observation per row, so the "
                    "counts must match."
                )

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        # Clip away from 0/1 to avoid Φ⁻¹(0) = -inf and Φ⁻¹(1) = +inf.
        u = np.clip(rng.uniform(size=n), _SIMULATE_U_EPS, 1.0 - _SIMULATE_U_EPS)
        return self.predict(u, X_new=X_arr, X_scale_new=X_scale_arr, what="quantile")

    def plot(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        ax: object = None,
    ) -> object:
        """Plot the estimated CDF and density.

        For non-interacting models, draws a single CDF/density curve over
        ``y`` (covariates are ignored — the unconditional baseline is shown).
        For :class:`InteractionBasis` models, draws one CDF curve and one
        density curve per row of ``X`` on a shared y-axis.

        Parameters
        ----------
        y:
            Response values at which to evaluate the model.  Must lie within
            ``basis.support``.
        X:
            Required for :class:`InteractionBasis` models: a 2-D matrix whose
            rows are the representative covariate values at which to draw the
            conditional curves.  Ignored for non-interacting models.
        ax:
            Optional 2-tuple ``(ax_cdf, ax_pdf)`` of ``matplotlib.axes.Axes``,
            or a single ``matplotlib.axes.Axes`` instance.  If a single axes
            is given, only the CDF is plotted.  If ``None``, a new figure
            with two subplots is created automatically.

        Returns
        -------
        list of matplotlib.axes.Axes, or matplotlib.axes.Axes
            ``[ax_cdf, ax_pdf]`` if two panels are plotted, otherwise the
            single ``ax_cdf``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ImportError
            If matplotlib is not installed.
        ValueError
            If ``X`` is not provided for an interacting model, or if it
            cannot be interpreted as a 2-D array.
        TypeError
            If ``ax`` is provided but cannot be unpacked into two axes nor
            used as a single axes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot(). "
                "Install with: pip install 'mltpy[plots]'"
            ) from exc

        self._check_is_fitted()
        y_arr = np.asarray(y, dtype=float).ravel()
        y_sorted = np.sort(y_arr)

        is_interaction = isinstance(self.basis, InteractionBasis)
        X_rows: NDArray[np.float64] | None = None
        if is_interaction:
            if X is None:
                raise ValueError(
                    "plot() on an InteractionBasis model requires X — a 2-D "
                    "matrix whose rows are the representative covariate "
                    "values at which to draw conditional curves."
                )
            X_rows = np.asarray(X, dtype=float)
            if X_rows.ndim == 1:
                X_rows = X_rows[:, None]
            if X_rows.ndim != 2:
                raise ValueError(
                    f"plot(): X must be 1-D or 2-D; got {X_rows.ndim}-D array."
                )

        if ax is not None:
            if isinstance(ax, tuple) and len(ax) == 2:
                ax_cdf, ax_pdf = ax
            elif hasattr(ax, "plot"):
                ax_cdf = ax
                ax_pdf = None
            else:
                raise TypeError(
                    "ax must be a 2-tuple (ax_cdf, ax_pdf) or a single Axes"
                )
            fig = None
        else:
            fig, (ax_cdf, ax_pdf) = plt.subplots(1, 2, figsize=(10, 4))

        if is_interaction:
            assert X_rows is not None
            m = y_sorted.shape[0]
            for x_row in X_rows:
                X_rep = np.broadcast_to(x_row, (m, x_row.shape[0])).astype(float)
                cdf = self.predict(y_sorted, X_new=X_rep, what="distribution")
                ax_cdf.plot(y_sorted, cdf, label=f"x={np.round(x_row, 3).tolist()}")
                if ax_pdf is not None:
                    pdf = self.predict(y_sorted, X_new=X_rep, what="density")
                    ax_pdf.plot(y_sorted, pdf, label=f"x={np.round(x_row, 3).tolist()}")
            ax_cdf.legend(fontsize="small")
            if ax_pdf is not None:
                ax_pdf.legend(fontsize="small")
        else:
            cdf = self.predict(y_sorted, what="distribution")
            ax_cdf.plot(y_sorted, cdf)
            if ax_pdf is not None:
                pdf = self.predict(y_sorted, what="density")
                ax_pdf.plot(y_sorted, pdf)

        ax_cdf.set_xlabel("y")
        ax_cdf.set_ylabel("F(y|x)" if is_interaction else "F(y)")
        ax_cdf.set_title(f"{type(self).__name__} — CDF")
        if ax_pdf is not None:
            ax_pdf.set_xlabel("y")
            ax_pdf.set_ylabel("f(y|x)" if is_interaction else "f(y)")
            ax_pdf.set_title(f"{type(self).__name__} — Density")

        if fig is not None:
            fig.tight_layout()

        if ax_pdf is None:
            return ax_cdf
        return [ax_cdf, ax_pdf]

    def __repr__(self) -> str:
        name = type(self).__name__
        order = self.basis.order
        cens = self.censoring if self.censoring is not None else CensoringType.NONE
        censoring = cens.name
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
        scaling: NDArray[np.float64] | None = None,
    ) -> None:
        basis = BernsteinBasis(order=order, support=support)
        super().__init__(
            basis=basis,
            censoring=censoring,
            optimizer_config=optimizer_config,
            base_distribution=base_distribution,
            scaling=scaling,
        )
        # Store for repr
        self._order = order
        self._support = support

    def __repr__(self) -> str:
        cens = self.censoring if self.censoring is not None else CensoringType.NONE
        censoring = cens.name
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
        w = max(max(len(n) for n in self.model_names), len("Model"))
        header = (
            f"{'Model':<{w}} {'n_par':>5} {'logLik':>12} "
            f"{'df':>4} {'Deviance':>12} {'Pr(>Chisq)':>12}"
        )
        rows = [header, "-" * len(header)]
        for i in range(len(self.model_names)):
            df_str = "" if self.df[i] is None else str(self.df[i])
            dev_str = "" if self.deviance[i] is None else f"{self.deviance[i]:>12.4f}"
            p_str = "" if self.p_value[i] is None else f"{self.p_value[i]:>12.4g}"
            rows.append(
                f"{self.model_names[i]:<{w}} "
                f"{self.n_params[i]:>5} "
                f"{self.log_lik[i]:>12.4f} "
                f"{df_str:>4} "
                f"{dev_str:>12} "
                f"{p_str:>12}"
            )
        return "\n".join(rows)


@dataclass(frozen=True)
class WaldTestResult:
    """Result of a Wald test for linear restrictions on model parameters.

    Parameters
    ----------
    statistic : float
        Wald chi-squared statistic ``W = (Rθ - r)^T [R V R^T]^{-1} (Rθ - r)``.
    df : int
        Degrees of freedom (number of restrictions, i.e. number of rows in ``R``).
    p_value : float
        Right-tail probability ``Pr(χ²(df) > W)``.
    vcov_type : str
        Which variance–covariance matrix was used: ``"information"`` or
        ``"sandwich"``.
    """

    statistic: float
    df: int
    p_value: float
    vcov_type: str

    def __repr__(self) -> str:
        return (
            f"Wald test ({self.vcov_type} vcov)\n"
            f"  W = {self.statistic:.4f}, df = {self.df}, "
            f"p-value = {self.p_value:.4g}"
        )


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
            raise ValueError(f"Model #{i} is not fitted. Call fit() before anova().")

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
        if d < 0.0:
            warnings.warn(
                f"anova: deviance between model {i - 1} "
                f"(k={n_params[i - 1]}, ll={log_lik[i - 1]:.4f}) and "
                f"model {i} (k={n_params[i]}, ll={log_lik[i]:.4f}) is "
                f"negative (D={d:.4g}).  The larger model fits worse — "
                "possible causes: models are not nested, or the larger "
                "model failed to converge.  The p-value is computed with "
                "D clamped to 0 and is not meaningful.",
                UserWarning,
                stacklevel=2,
            )
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
