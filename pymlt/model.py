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
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import chi2

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

# Small epsilon used for bracket safety in brentq
_BRENTQ_EPS = 1e-10

# Floor for log(h') to avoid log(0) at boundaries where monotonicity is marginal
_LOG_HP_FLOOR = np.finfo(np.float64).tiny


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
        self.n_obs_ = (
            int(y_clean.n) if isinstance(y_clean, CensoredData) else len(y_clean)
        )
        self.n_free_params_ = int(result.theta.size)
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
            raise RuntimeError("Modellzustand fehlt unerwartet nach dem Fitten.")
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
            raise RuntimeError("Modellzustand fehlt unerwartet nach dem Fitten.")
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
            f"anova() benötigt mindestens 2 Modelle, erhalten: {len(models)}."
        )
    for i, m in enumerate(models):
        if not m.is_fitted_:
            raise ValueError(
                f"Modell #{i} ist nicht gefittet. Rufe fit() vor anova() auf."
            )

    n_obs_ref = models[0].n_obs_
    for i, m in enumerate(models):
        if m.n_obs_ != n_obs_ref:
            raise ValueError(
                f"Modelle müssen auf derselben Stichprobengröße gefittet sein. "
                f"Modell #0 hat n={n_obs_ref}, Modell #{i} hat n={m.n_obs_}."
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
                "Aufeinanderfolgende Modelle müssen eine echt unterschiedliche "
                f"Parameterzahl haben (Modell {i - 1}: k={n_params[i - 1]}, "
                f"Modell {i}: k={n_params[i]})."
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
