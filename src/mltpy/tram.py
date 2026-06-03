"""Convenience layer for common conditional transformation models.

This module provides pre-configured wrappers around
:class:`~mltpy.model.ConditionalTransformationModel` / :class:`~mltpy.model.MLT`
that mirror the R ``tram`` package (Hothorn).  Users working with these classes
never need to import :class:`~mltpy.basis.BernsteinBasis`,
:class:`~mltpy.variables.CensoringType`, or
:class:`~mltpy.optimizer.OptimizerConfig` directly.

Classes
-------
BoxCox
    Box-Cox transformation model for continuous outcomes with exact observations.
Coxph
    Cox proportional hazards model for right-censored survival data.
Colr
    Continuous outcome logistic regression — uses a logistic base distribution.
Lm
    Normal linear regression as a CTM (order=1 Bernstein, normal base).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as _norm

from mltpy.basis import (
    BernsteinBasis,
    InteractionBasis,
    LogBernsteinBasis,
    OrdinalBasis,
)
from mltpy.likelihood import _H_CLIP, BaseDistribution, _get_dist
from mltpy.model import MLT, ConditionalTransformationModel
from mltpy.optimizer import OptimizerConfig
from mltpy.variables import CensoringType, OrderedVariable

# ---------------------------------------------------------------------------
# Shared Wald-table helper
# ---------------------------------------------------------------------------


def _format_wald_table(
    names: Sequence[str],
    estimates: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
) -> str:
    """Format a Wald-style coefficient table.

    Columns: ``Estimate``, ``Std. Error``, ``z value``, ``Pr(>|z|)``.  Used by
    both :class:`_TramModel.summary` and :class:`Polr.summary`.
    """
    name_width = max((len(n) for n in names), default=4)
    name_width = max(name_width, 4)
    valid = np.isfinite(standard_errors) & (standard_errors > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = estimates / standard_errors
        pvals = 2.0 * _norm.sf(np.abs(z))
    header = (
        f"  {'':<{name_width}}  {'Estimate':>10}  {'Std. Error':>10}  "
        f"{'z value':>8}  {'Pr(>|z|)':>9}"
    )
    rows = [header]
    for name, b, s, zv, pv, ok in zip(
        names, estimates, standard_errors, z, pvals, valid
    ):
        if ok:
            zv_str = f"{zv:>8.3f}"
            pv_str = f"{pv:>9.4g}"
        else:
            zv_str = f"{'NA':>8}"
            pv_str = f"{'NA':>9}"
        rows.append(
            f"  {name:<{name_width}}  {b:>10.4f}  {s:>10.4f}  {zv_str}  {pv_str}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Internal base class
# ---------------------------------------------------------------------------


class _TramModel(MLT):
    """Base class for all tram convenience models.

    Extends :class:`~mltpy.model.MLT` with a diagnostic summary and an
    optional matplotlib plot.  Not part of the public API.
    """

    def __repr__(self) -> str:
        name = type(self).__name__
        cens = self.censoring if self.censoring is not None else CensoringType.NONE
        censoring = cens.name
        if self.is_fitted_:
            if self.result_ is None:
                raise RuntimeError("Unexpected None result_ for fitted model")
            ll = self.result_.log_likelihood
            return (
                f"{name}(order={self._order}, support={self._support}, "
                f"censoring={censoring}, fitted=True, ll={ll:.2f})"
            )
        return (
            f"{name}(order={self._order}, support={self._support}, "
            f"censoring={censoring}, fitted=False)"
        )

    def summary(self) -> str:
        """Return a formatted diagnostic string.

        When covariates are present the output includes a Wald coefficient
        table for the shift terms (``beta``), matching R ``tram::summary``:
        ``Estimate``, ``Std. Error``, ``z value``, ``Pr(>|z|)``.  Baseline
        Bernstein coefficients ``theta_b`` are treated as nuisance and not
        tabulated — they are constrained (monotone) and the corresponding
        Wald z-tests are not meaningful.

        Returns
        -------
        str
            Multi-line summary suitable for printing.

        Examples
        --------
        >>> model = BoxCox(support=(0.0, 10.0)).fit(y)
        >>> print(model.summary())
        """
        lines = [
            f"Model:        {type(self).__name__}",
            f"Support:      {self._support}",
            f"Basis order:  {self._order}",
            f"Fitted:       {'Yes' if self.is_fitted_ else 'No'}",
        ]
        if self.is_fitted_:
            if self.result_ is None:
                raise RuntimeError("Unexpected None result_ for fitted model")
            lines += [
                f"Log-lik:      {self.result_.log_likelihood:.4f}",
                f"AIC:          {self.aic():.4f}",
                f"BIC:          {self.bic():.4f}",
                f"Converged:    {'Yes' if self.result_.converged else 'No'}",
                f"n_restarts:   {self.result_.n_restarts}",
            ]
            table = self._coef_table()
            if table is not None:
                lines += ["", "Coefficients:", table]
        return "\n".join(lines)

    def _coef_table(self) -> str | None:
        """Format Wald coefficient tables for the ``β`` (and ``γ``) blocks.

        Returns ``None`` when the fitted model has no covariates, when the
        model uses an :class:`~mltpy.basis.InteractionBasis` (the tensor
        product has no flat ``beta`` block to tabulate), or if the Hessian
        is singular so that standard errors cannot be computed.  When the
        model was fitted with ``scaling=`` (γ block present), the scaling
        coefficients are tabulated below the shift coefficients under a
        ``Scaling coefficients`` heading.
        """
        if self.theta_ is None:
            return None
        if isinstance(self.basis, InteractionBasis):
            return None
        _, beta, gamma, p, q_d, q_s = self._split_fitted_theta()
        if q_d <= 0 and q_s <= 0:
            return None

        try:
            se = self.standard_errors()
        except RuntimeError:
            return "  [Standard errors not available: Hessian matrix is singular.]"

        sections: list[str] = []
        if q_d > 0 and beta is not None:
            names = self.feature_names_in_ or [f"X{j + 1}" for j in range(q_d)]
            sections.append(_format_wald_table(names, beta, se[p : p + q_d]))
        if q_s > 0 and gamma is not None:
            s_names = self.scaling_feature_names_in_ or [
                f"X{j + 1}" for j in range(q_s)
            ]
            sections.append("Scaling coefficients:")
            sections.append(_format_wald_table(s_names, gamma, se[p + q_d :]))
        return "\n".join(sections)

    def plot(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        ax: Any = None,
    ) -> Any | list[Any]:
        """Plot the estimated CDF and density side by side.

        Parameters
        ----------
        y:
            Response values at which to evaluate the model.  Must lie within
            ``basis.support``.
        X:
            Ignored — TRAM models are not interacting, so the plotted
            baseline does not depend on covariates.  Accepted for signature
            compatibility with the base class.
        ax:
            Optional 2-tuple ``(ax_cdf, ax_pdf)`` of ``matplotlib.axes.Axes``,
            or a single ``matplotlib.axes.Axes`` instance. If a single axes is
            provided, only the CDF is plotted. If ``None``, a new figure with
            two subplots is created automatically.

        Returns
        -------
        list of matplotlib.axes.Axes, or matplotlib.axes.Axes
            Returns ``[ax_cdf, ax_pdf]`` if two panels are plotted, otherwise
            returns the single ``ax_cdf``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ImportError
            If matplotlib is not installed.
        TypeError
            If ``ax`` is provided but cannot be unpacked into two axes, nor used
            as a single axes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot(). "
                "Install with: pip install 'mltpy[plots]'"
            )

        self._check_is_fitted()
        y_arr = np.asarray(y, dtype=float).ravel()
        y_sorted = np.sort(y_arr)

        cdf = self.predict(y_sorted, what="distribution")
        pdf = self.predict(y_sorted, what="density")

        if ax is not None:
            if isinstance(ax, tuple) and len(ax) == 2:
                ax_cdf, ax_pdf = ax
            elif hasattr(ax, "plot"):
                ax_cdf = ax
                ax_pdf = None
            else:
                raise TypeError(
                    "ax must be a 2-tuple (ax_cdf, ax_pdf) or a single Axes"
                ) from None
            fig = None
        else:
            fig, (ax_cdf, ax_pdf) = plt.subplots(1, 2, figsize=(10, 4))

        ax_cdf.plot(y_sorted, cdf)
        ax_cdf.set_xlabel("y")
        ax_cdf.set_ylabel("F(y)")
        ax_cdf.set_title(f"{type(self).__name__} — CDF")

        if ax_pdf is not None:
            ax_pdf.plot(y_sorted, pdf)
            ax_pdf.set_xlabel("y")
            ax_pdf.set_ylabel("f(y)")
            ax_pdf.set_title(f"{type(self).__name__} — Density")

        if fig is not None:
            fig.tight_layout()

        if ax_pdf is None:
            return ax_cdf
        return [ax_cdf, ax_pdf]


# ---------------------------------------------------------------------------
# BoxCox
# ---------------------------------------------------------------------------


class BoxCox(_TramModel):
    """Box-Cox transformation model for continuous outcomes.

    Fits a flexible, monotone transformation h(y) that maps the response
    distribution to a standard normal.  Useful as a non-parametric
    generalisation of the classical Box-Cox power transform when the
    normality assumption for linear regression is violated.

    Parameters
    ----------
    support:
        Closed interval ``(a, b)`` covering all observed values.
    order:
        Polynomial degree of the Bernstein basis.  Defaults to 6.
    optimizer_config:
        Optimisation settings.  If ``None``, library defaults are used.
    censoring:
        Censoring type of the response data.  Defaults to
        :attr:`~mltpy.variables.CensoringType.NONE`.  Pass
        :attr:`~mltpy.variables.CensoringType.RIGHT`,
        :attr:`~mltpy.variables.CensoringType.LEFT`, or
        :attr:`~mltpy.variables.CensoringType.INTERVAL` together with a
        :class:`~mltpy.variables.CensoredData` ``y`` to fit the censored
        Box-Cox likelihood.
    scaling:
        Optional scaling-design matrix of shape ``(n, q_s)`` mirroring
        R ``tram::BoxCox(..., scale=~x_s)``.  Threads through to the
        scaled-baseline likelihood (issue #71) and the scaled-predict
        path (issue #72).  When supplied, the fitted parameter vector
        gains a γ block (length ``q_s``) exposed as :attr:`gamma_`, and
        :meth:`predict` requires ``X_scale_new``.  Sign-aligned with the
        R ``scale=`` block (ADR 0002, Decision 5); see
        ``docs/adr/0002-scaling-terms.md``.

    Examples
    --------
    >>> from mltpy.tram import BoxCox
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.lognormal(size=200)
    >>> model = BoxCox(support=(y.min(), y.max()))
    >>> model.fit(y)
    >>> cdf   = model.predict(y, what="distribution")
    >>> trafo = model.fitted_transformation(y)
    """

    def __init__(
        self,
        support: tuple[float, float],
        order: int = 6,
        optimizer_config: OptimizerConfig | None = None,
        censoring: CensoringType = CensoringType.NONE,
        scaling: NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(
            order=order,
            support=support,
            censoring=censoring,
            optimizer_config=optimizer_config,
            base_distribution="normal",
            scaling=scaling,
        )

    @property
    def gamma_(self) -> NDArray[np.float64]:
        """Fitted scaling-block coefficients ``γ`` (length ``q_s``).

        Sign-aligned with R ``tram::BoxCox(..., scale=~x_s)``'s scaling
        block (ADR 0002, Decision 5).

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        ValueError
            If the model was constructed without ``scaling=``.
        """
        self._check_is_fitted()
        if self.scaling is None:
            raise ValueError("Model was not fitted with scaling=; gamma_ is undefined.")
        gamma = self.gamma_coef_
        if gamma is None:
            raise RuntimeError("Unexpected None gamma_coef_ for fitted model")
        return gamma

    @property
    def feature_names_scaling_(self) -> list[str]:
        """Column names of the scaling-design matrix supplied at fit time.

        Populated from a ``pandas.DataFrame`` column index when available,
        otherwise ``["X1", "X2", ...]``.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        ValueError
            If the model was constructed without ``scaling=``.
        """
        self._check_is_fitted()
        if self.scaling is None:
            raise ValueError(
                "Model was not fitted with scaling=; "
                "feature_names_scaling_ is undefined."
            )
        names = self.scaling_feature_names_in_
        if names is None:
            q_s = self.scaling.shape[1]
            return [f"X{j + 1}" for j in range(q_s)]
        return names

    def fitted_transformation(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the raw fitted transformation h(y) = B_k(y) @ theta_b.

        This is the monotone function that maps the observed response scale
        to the latent standard-normal scale.  Useful for visualising the
        shape of the estimated transformation.

        Parameters
        ----------
        y:
            Response values within ``basis.support``.

        Returns
        -------
        NDArray of shape ``(len(y),)``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError("Unexpected None theta_ for fitted model")
        if not isinstance(self.basis, BernsteinBasis):
            raise NotImplementedError(
                "fitted_transformation() is only defined for the shift-basis "
                "(BernsteinBasis) path; it is not available for "
                "InteractionBasis models."
            )
        p = self.basis.order + 1
        theta_b = self.theta_[:p]
        y_arr = np.asarray(y, dtype=float).ravel()
        B = self.basis.evaluate(y_arr)  # (m, p)
        return B @ theta_b


# ---------------------------------------------------------------------------
# Coxph
# ---------------------------------------------------------------------------


class Coxph(_TramModel):
    """Cox proportional hazards model for right-censored survival data.

    Fits a monotone transformation h(t) under right-censoring using the
    minimum extreme value (``"min_extreme_value"``) base distribution, also
    known as the reversed Gumbel link. With covariates entering linearly,
    this parameterisation is equivalent to the classical Cox proportional
    hazards model. The baseline distribution is estimated
    non-parametrically via a Bernstein polynomial.

    Pass ``interacting`` to fit a *non-proportional* (stratified or
    fully-interacting) Cox model where the transformation itself depends
    on the covariate via the tensor product
    ``h(t | x) = (a(t) ⊗ b(x))ᵀ vec(Θ)``.  See ADR 0001 and
    :class:`~mltpy.basis.InteractionBasis` for the parameter-vector layout
    and the column-wise monotonicity strategy.

    Parameters
    ----------
    support:
        Closed interval ``(a, b)`` with ``a > 0`` and ``b`` at least as
        large as the longest observed follow-up time.
    order:
        Polynomial degree of the Bernstein basis on the response.  Defaults
        to 6.
    optimizer_config:
        Optimisation settings.  If ``None``, library defaults are used.
    interacting:
        Optional x-basis (:class:`~mltpy.basis.BernsteinBasis`,
        :class:`~mltpy.basis.OrdinalBasis`, or
        :class:`~mltpy.basis.InterceptBasis`).  When provided, the model
        is fit as ``MLT(InteractionBasis(BernsteinBasis(...), interacting))``
        instead of the standard shift model.  Only exact (non-censored)
        time data is currently supported on this path; censoring with an
        interacting basis is not yet implemented in the likelihood path.
    scaling:
        Optional scaling-design matrix of shape ``(n, q_s)`` mirroring
        R ``tram::Coxph(Surv(y, event) ~ x_d | x_s)``.  Routes through to
        the scaled-baseline likelihood (#71) and the scaled-predict path
        (#72).  When supplied, the fit becomes a heteroskedastic /
        non-proportional-hazards Cox model
        ``log[-log S(t | x)] = h_0(t) · exp(x_s · γ) + x_d · β``: the
        hazard ratio between two ``x_s`` values varies with ``t`` (the
        proportional-hazards assumption is relaxed).  The fitted parameter
        vector gains a γ block exposed as :attr:`gamma_`, and the
        :meth:`survival`, :meth:`hazard`, and :meth:`predict` methods
        require ``X_scale`` / ``X_scale_new``.  Sign-aligned with R
        (ADR 0002, Decision 5).  Not supported together with
        ``interacting=`` (ADR 0002, Decision 2).

    Examples
    --------
    >>> from mltpy.tram import Coxph
    >>> from mltpy.variables import CensoredData
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y_time   = rng.exponential(scale=2.0, size=200)
    >>> y_status = rng.binomial(1, 0.7, size=200).astype(bool)
    >>> cd = CensoredData.right_censored(y_time, censored=~y_status)
    >>> model = Coxph(support=(0.01, y_time.max()))
    >>> model.fit(cd)
    >>> surv = model.survival(y_time)
    """

    def __init__(
        self,
        support: tuple[float, float],
        order: int = 6,
        optimizer_config: OptimizerConfig | None = None,
        interacting: BernsteinBasis | OrdinalBasis | None = None,
        scaling: NDArray[np.float64] | None = None,
    ) -> None:
        if scaling is not None and interacting is not None:
            raise ValueError(
                "scaling= and interacting= cannot be combined "
                "(see docs/adr/0002-scaling-terms.md, Decision 2)."
            )

        if interacting is None:
            super().__init__(
                order=order,
                support=support,
                censoring=CensoringType.RIGHT,
                optimizer_config=optimizer_config,
                base_distribution="min_extreme_value",
                scaling=scaling,
            )
            return

        y_basis = BernsteinBasis(order=order, support=support)
        ib = InteractionBasis(y_basis=y_basis, x_basis=interacting)
        ConditionalTransformationModel.__init__(
            self,
            basis=ib,
            censoring=CensoringType.RIGHT,
            optimizer_config=optimizer_config,
            base_distribution="min_extreme_value",
        )
        self._order = order
        self._support = support

    @property
    def gamma_(self) -> NDArray[np.float64]:
        """Fitted scaling-block coefficients ``γ`` (length ``q_s``).

        Sign-aligned with R ``tram::Coxph(..., scale=~x_s)``'s scaling
        block (ADR 0002, Decision 5).

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        ValueError
            If the model was constructed without ``scaling=``.
        """
        self._check_is_fitted()
        if self.scaling is None:
            raise ValueError("Model was not fitted with scaling=; gamma_ is undefined.")
        gamma = self.gamma_coef_
        if gamma is None:
            raise RuntimeError("Unexpected None gamma_coef_ for fitted model")
        return gamma

    @property
    def feature_names_scaling_(self) -> list[str]:
        """Column names of the scaling-design matrix supplied at fit time.

        Populated from a ``pandas.DataFrame`` column index when available,
        otherwise ``["X1", "X2", ...]``.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        ValueError
            If the model was constructed without ``scaling=``.
        """
        self._check_is_fitted()
        if self.scaling is None:
            raise ValueError(
                "Model was not fitted with scaling=; "
                "feature_names_scaling_ is undefined."
            )
        names = self.scaling_feature_names_in_
        if names is None:
            q_s = self.scaling.shape[1]
            return [f"X{j + 1}" for j in range(q_s)]
        return names

    def survival(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
        X_scale: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Estimate the survival function S(y) = 1 − F(y|x).

        Parameters
        ----------
        y:
            Time points within ``basis.support``.
        X:
            Optional covariate matrix of shape ``(m, q_d)``.
        offset:
            Optional per-observation offset added to ``h``.
        X_scale:
            New-data scaling-design matrix of shape ``(m, q_s)``, required
            when the model was fitted with ``scaling=``.  Threaded through
            to :meth:`predict` as ``X_scale_new``.

        Returns
        -------
        NDArray of shape ``(m,)`` with values in ``[0, 1]``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        return 1.0 - self.predict(
            y,
            X_new=X,
            what="distribution",
            offset_new=offset,
            X_scale_new=X_scale,
        )

    def hazard(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
        X_scale: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Estimate the hazard rate h(y) = f(y|x) / S(y|x).

        Parameters
        ----------
        y:
            Time points within ``basis.support``.
        X:
            Optional covariate matrix of shape ``(m, q_d)``.
        offset:
            Optional per-observation offset added to ``h``.
        X_scale:
            New-data scaling-design matrix of shape ``(m, q_s)``, required
            when the model was fitted with ``scaling=``.  Threaded through
            to :meth:`predict` as ``X_scale_new``.

        Returns
        -------
        NDArray of shape ``(m,)`` with non-negative values.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        return self.predict(
            y,
            X_new=X,
            what="hazard",
            offset_new=offset,
            X_scale_new=X_scale,
        )


# ---------------------------------------------------------------------------
# Lehmann
# ---------------------------------------------------------------------------


class Lehmann(_TramModel):
    """Lehmann (proportional reverse-time hazards) model for right-censored data.

    Dual of :class:`Coxph`.  Fits a monotone transformation h(t) under
    right-censoring using the maximum extreme value (``"max_extreme_value"``)
    base distribution — the standard Gumbel distribution.  With covariates
    entering linearly this parameterisation satisfies
    ``-log F(t | x) = h(t) + x'β``, which is the Lehmann alternative
    (proportional reverse-time hazards) model.

    Parameters
    ----------
    support:
        Closed interval ``(a, b)`` with ``a > 0`` and ``b`` at least as
        large as the longest observed follow-up time.
    order:
        Polynomial degree of the Bernstein basis.  Defaults to 6.
    optimizer_config:
        Optimisation settings.  If ``None``, library defaults are used.

    Examples
    --------
    >>> from mltpy.tram import Lehmann
    >>> from mltpy.variables import CensoredData
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y_time   = rng.exponential(scale=2.0, size=200)
    >>> y_status = rng.binomial(1, 0.7, size=200).astype(bool)
    >>> cd = CensoredData.right_censored(y_time, censored=~y_status)
    >>> model = Lehmann(support=(0.01, y_time.max()))
    >>> model.fit(cd)
    >>> surv = model.survival(y_time)
    """

    def __init__(
        self,
        support: tuple[float, float],
        order: int = 6,
        optimizer_config: OptimizerConfig | None = None,
    ) -> None:
        super().__init__(
            order=order,
            support=support,
            censoring=CensoringType.RIGHT,
            optimizer_config=optimizer_config,
            base_distribution="max_extreme_value",
        )

    def survival(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Estimate the survival function S(y) = 1 − F(y|x).

        Parameters
        ----------
        y:
            Time points within ``basis.support``.
        X:
            Optional covariate matrix of shape ``(m, q)``.
        offset:
            Optional per-observation offset added to ``h``.

        Returns
        -------
        NDArray of shape ``(m,)`` with values in ``[0, 1]``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        return 1.0 - self.predict(y, X_new=X, what="distribution", offset_new=offset)

    def hazard(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Estimate the hazard rate h(y) = f(y|x) / S(y|x).

        Parameters
        ----------
        y:
            Time points within ``basis.support``.
        X:
            Optional covariate matrix of shape ``(m, q)``.
        offset:
            Optional per-observation offset added to ``h``.

        Returns
        -------
        NDArray of shape ``(m,)`` with non-negative values.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        return self.predict(y, X_new=X, what="hazard", offset_new=offset)


# ---------------------------------------------------------------------------
# Colr
# ---------------------------------------------------------------------------


class Colr(_TramModel):
    """Continuous outcome logistic regression.

    Fits a monotone transformation h(y) such that h(Y|X) follows a standard
    logistic distribution.  Analogous to ordinal logistic regression but for
    continuous, fully observed outcomes.  Produces proportional-odds model
    when covariates are included.

    Parameters
    ----------
    support:
        Closed interval ``(a, b)`` covering all observed values.
    order:
        Polynomial degree of the Bernstein basis.  Defaults to 6.
    optimizer_config:
        Optimisation settings.  If ``None``, library defaults are used.
    scaling:
        Optional scaling-design matrix of shape ``(n, q_s)`` mirroring
        R ``tram::Colr(y ~ x_d | x_s)``.  Threads through to the
        scaled-baseline likelihood (#71) and the scaled-predict path
        (#72).  When supplied, the fit becomes a *heteroskedastic*
        continuous-outcome logistic regression with non-proportional
        log-odds — the log-odds gap between two ``x_s`` values varies
        with ``y`` (the proportional-odds assumption is relaxed).  The
        fitted parameter vector gains a γ block exposed as
        :attr:`gamma_`, and :meth:`predict` requires ``X_scale_new``.
        Sign-aligned with R (ADR 0002, Decision 5); see
        ``docs/adr/0002-scaling-terms.md``.

    Examples
    --------
    >>> from mltpy.tram import Colr
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.logistic(loc=2.0, scale=0.5, size=200)
    >>> model = Colr(support=(y.min(), y.max()))
    >>> model.fit(y)
    >>> cdf = model.predict(y, what="distribution")
    """

    def __init__(
        self,
        support: tuple[float, float],
        order: int = 6,
        optimizer_config: OptimizerConfig | None = None,
        scaling: NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(
            order=order,
            support=support,
            censoring=CensoringType.NONE,
            optimizer_config=optimizer_config,
            base_distribution="logistic",
            scaling=scaling,
        )

    @property
    def gamma_(self) -> NDArray[np.float64]:
        """Fitted scaling-block coefficients ``γ`` (length ``q_s``).

        Sign-aligned with R ``tram::Colr(..., scale=~x_s)``'s scaling
        block (ADR 0002, Decision 5).

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        ValueError
            If the model was constructed without ``scaling=``.
        """
        self._check_is_fitted()
        if self.scaling is None:
            raise ValueError("Model was not fitted with scaling=; gamma_ is undefined.")
        gamma = self.gamma_coef_
        if gamma is None:
            raise RuntimeError("Unexpected None gamma_coef_ for fitted model")
        return gamma

    @property
    def feature_names_scaling_(self) -> list[str]:
        """Column names of the scaling-design matrix supplied at fit time.

        Populated from a ``pandas.DataFrame`` column index when available,
        otherwise ``["X1", "X2", ...]``.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        ValueError
            If the model was constructed without ``scaling=``.
        """
        self._check_is_fitted()
        if self.scaling is None:
            raise ValueError(
                "Model was not fitted with scaling=; "
                "feature_names_scaling_ is undefined."
            )
        names = self.scaling_feature_names_in_
        if names is None:
            q_s = self.scaling.shape[1]
            return [f"X{j + 1}" for j in range(q_s)]
        return names


# ---------------------------------------------------------------------------
# Lm
# ---------------------------------------------------------------------------


class Lm(_TramModel):
    r"""Normal linear regression expressed as a CTM.

    Fixes the Bernstein basis to ``order=1`` and the base distribution to
    standard normal.  With these constraints the transformation
    :math:`h(y) = \theta_0 (1-u) + \theta_1 u`, where
    :math:`u = (y-a)/(b-a)`, is affine, so the CTM
    :math:`h(Y) - \beta^\top X \sim \mathcal{N}(0,1)` is exactly equivalent
    to the classical normal linear model
    :math:`Y = \mu + \gamma^\top X + \varepsilon`,
    :math:`\varepsilon \sim \mathcal{N}(0, \sigma^2)`.

    The mapping between CTM and lm parameters is

    .. math::

        \hat{\sigma} &= (b - a) / (\theta_1 - \theta_0), \\
        \hat{\mu}    &= a - \theta_0 \hat{\sigma}, \\
        \hat{\gamma} &= -\hat{\sigma} \, \beta_{\mathrm{ctm}}.

    The minus sign on :math:`\hat{\gamma}` reflects mltpy's internal shift
    convention ``h(y) + X @ beta = z`` (the R ``tram`` package uses
    ``h(y) - X @ beta = z``, hence R's :math:`\beta` equals
    :math:`-\beta_{\mathrm{ctm}}`).

    Note that :math:`\hat{\sigma}` is the MLE, which differs from the
    unbiased OLS estimator returned by ``lm()`` by a factor
    :math:`\sqrt{(n-p)/n}`.

    These are exposed via :attr:`sigma_`, :attr:`intercept_`, and
    :attr:`coef_` (sklearn-style fitted attributes).

    Parameters
    ----------
    support:
        Closed interval ``(a, b)`` covering all observed response values.
    optimizer_config:
        Optimisation settings.  If ``None``, library defaults are used.
    scaling:
        Optional scaling-design matrix of shape ``(n, q_s)`` mirroring
        R ``tram::Lm(y ~ x_d | x_s, ..., scale = ~x_s)``.  When supplied,
        the fitted model is heteroskedastic — the constant-variance
        closed-form mapping to :attr:`sigma_` / :attr:`intercept_` /
        :attr:`coef_` no longer applies, and those properties raise
        ``NotImplementedError`` pointing at :attr:`gamma_`.  Use
        :meth:`predict` with ``X_scale_new`` for inference, and access
        the scaling-block coefficients via :attr:`gamma_`.  Sign-aligned
        with R (ADR 0002, Decision 5).

    Notes
    -----
    The Bernstein order is fixed at ``1`` by construction; passing an
    ``order`` keyword raises ``TypeError``.

    Examples
    --------
    >>> import numpy as np
    >>> from mltpy.tram import Lm
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=200)
    >>> y = 2.0 + 3.0 * x + rng.normal(scale=0.5, size=200)
    >>> model = Lm(support=(y.min() - 0.1, y.max() + 0.1))
    >>> model.fit(y, X=x.reshape(-1, 1))
    >>> # OLS cross-check
    >>> A = np.c_[np.ones_like(x), x]
    >>> beta_ols, *_ = np.linalg.lstsq(A, y, rcond=None)
    >>> np.allclose([model.intercept_, model.coef_[0]], beta_ols, atol=0.05)
    True
    """

    def __init__(
        self,
        support: tuple[float, float],
        optimizer_config: OptimizerConfig | None = None,
        scaling: NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(
            order=1,
            support=support,
            censoring=CensoringType.NONE,
            optimizer_config=optimizer_config,
            base_distribution="normal",
            scaling=scaling,
        )

    def _baseline(self) -> tuple[float, float]:
        """Return the two baseline Bernstein coefficients (theta_0, theta_1)."""
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError("Unexpected None theta_ for fitted model")
        return float(self.theta_[0]), float(self.theta_[1])

    _SCALING_NOT_IMPLEMENTED_MSG = (
        "{prop} is undefined under scaling=: the constant-variance "
        "closed-form mapping from CTM to lm parameters no longer applies "
        "when sigma depends on x_s via gamma. Use gamma_, predict(), or "
        "the raw theta_ / beta_coef_ block instead "
        "(see docs/adr/0002-scaling-terms.md)."
    )

    @property
    def sigma_(self) -> float:
        """Estimated residual standard deviation of the equivalent lm.

        Computed as ``(b - a) / (theta_1 - theta_0)``.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        NotImplementedError
            If the model was fitted with ``scaling=`` — the residual
            standard deviation is no longer constant in ``x_s`` and the
            scalar closed-form mapping is undefined.  Use :attr:`gamma_`
            and :meth:`predict` instead.
        RuntimeError
            If the fit is degenerate with ``theta_[1] == theta_[0]``, which
            is feasible under the non-strict monotonicity constraint but
            leaves the lm-equivalence mapping undefined.
        """
        self._check_is_fitted()
        if self.scaling is not None:
            raise NotImplementedError(
                self._SCALING_NOT_IMPLEMENTED_MSG.format(prop="sigma_")
            )
        t0, t1 = self._baseline()
        a, b = self._support
        denom = t1 - t0
        if denom == 0.0:
            raise RuntimeError(
                "Degenerate Lm fit: theta_[1] == theta_[0] (denom = 0). "
                "The fitted transformation is constant "
                "on the support, so sigma_, intercept_, and coef_ are "
                "undefined. Check the fit diagnostics (support, data "
                "scale, optimiser convergence)."
            )
        return (b - a) / denom

    @property
    def intercept_(self) -> float:
        """Estimated intercept of the equivalent lm.

        Computed as ``a - theta_0 * sigma_``.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        NotImplementedError
            If the model was fitted with ``scaling=`` — see :attr:`sigma_`.
        """
        self._check_is_fitted()
        if self.scaling is not None:
            raise NotImplementedError(
                self._SCALING_NOT_IMPLEMENTED_MSG.format(prop="intercept_")
            )
        t0, _ = self._baseline()
        a, _ = self._support
        return a - t0 * self.sigma_

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Estimated regression coefficients of the equivalent lm.

        Computed as ``-sigma_ * beta_ctm``, where ``beta_ctm`` is the
        covariate part of ``theta_``.  Has shape ``(0,)`` when no
        covariates were supplied at fit time.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        NotImplementedError
            If the model was fitted with ``scaling=`` — see :attr:`sigma_`.
        """
        self._check_is_fitted()
        if self.scaling is not None:
            raise NotImplementedError(
                self._SCALING_NOT_IMPLEMENTED_MSG.format(prop="coef_")
            )
        if self.theta_ is None:
            raise RuntimeError("Unexpected None theta_ for fitted model")
        beta_ctm = self.theta_[2:]
        return -self.sigma_ * beta_ctm

    @property
    def gamma_(self) -> NDArray[np.float64]:
        """Fitted scaling-block coefficients ``γ`` (length ``q_s``).

        Sign-aligned with R ``tram::Lm(..., scale=~x_s)``'s scaling block
        (ADR 0002, Decision 5).

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        ValueError
            If the model was constructed without ``scaling=``.
        """
        self._check_is_fitted()
        if self.scaling is None:
            raise ValueError("Model was not fitted with scaling=; gamma_ is undefined.")
        gamma = self.gamma_coef_
        if gamma is None:
            raise RuntimeError("Unexpected None gamma_coef_ for fitted model")
        return gamma

    @property
    def feature_names_scaling_(self) -> list[str]:
        """Column names of the scaling-design matrix supplied at fit time.

        Populated from a ``pandas.DataFrame`` column index when available,
        otherwise ``["X1", "X2", ...]``.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        ValueError
            If the model was constructed without ``scaling=``.
        """
        self._check_is_fitted()
        if self.scaling is None:
            raise ValueError(
                "Model was not fitted with scaling=; "
                "feature_names_scaling_ is undefined."
            )
        names = self.scaling_feature_names_in_
        if names is None:
            q_s = self.scaling.shape[1]
            return [f"X{j + 1}" for j in range(q_s)]
        return names

    def fitted_transformation(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the fitted affine transformation h(y) = B(y) @ theta_b.

        Parameters
        ----------
        y:
            Response values within ``basis.support``.

        Returns
        -------
        NDArray of shape ``(len(y),)``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError("Unexpected None theta_ for fitted model")
        if not isinstance(self.basis, BernsteinBasis):
            raise NotImplementedError(
                "fitted_transformation() is only defined for the shift-basis "
                "(BernsteinBasis) path; it is not available for "
                "InteractionBasis models."
            )
        p = self.basis.order + 1
        theta_b = self.theta_[:p]
        y_arr = np.asarray(y, dtype=float).ravel()
        B = self.basis.evaluate(y_arr)
        return B @ theta_b


# ---------------------------------------------------------------------------
# Survreg — parametric survival (log-scale models)
# ---------------------------------------------------------------------------

SurvregDistribution = Literal["weibull", "lognormal", "loglogistic"]

_SURVREG_DIST_MAP: dict[str, str] = {
    "weibull": "min_extreme_value",
    "lognormal": "normal",
    "loglogistic": "logistic",
}


class Survreg(_TramModel):
    """Parametric survival model on the log-time scale (R ``tram::Survreg``).

    Fits a monotone transformation h(log t) such that h(log T | X) follows a
    standard distribution.  This is equivalent to fitting a TRAM on
    ``Y = log(T)`` — hence the name *Survreg*.  Supported distributions:

    * ``"weibull"``     — Weibull (minimum extreme value / reversed Gumbel link)
    * ``"lognormal"``   — Log-normal (standard normal link)
    * ``"loglogistic"`` — Log-logistic (standard logistic link)

    With covariates X the model is proportional (Weibull / log-logistic) or
    additive (log-normal) on the log-time scale.

    Parameters
    ----------
    support:
        Closed interval ``(a, b)`` with ``0 < a < b`` on the *original*
        positive time scale (not log-scale).  Should bracket all observed
        survival times.
    distribution:
        Parametric family: ``"weibull"`` (default), ``"lognormal"``, or
        ``"loglogistic"``.
    order:
        Polynomial degree of the Bernstein basis on the log scale.  Defaults
        to 6.  Note that ``tram::Survreg`` itself fits a strictly affine
        (two-parameter) baseline on ``log(t)`` regardless of ``order``;
        ``order = 1`` on the mltpy side reproduces that parameterisation
        and is required for R parity comparisons.
    optimizer_config:
        Optimisation settings.  If ``None``, library defaults are used.
    scaling:
        Optional scaling-design matrix of shape ``(n, q_s)`` mirroring
        R ``tram::Survreg(Surv(y, event) ~ x_d | x_s, ..., scale=~x_s)``.
        When supplied, the model becomes heteroskedastic on the log-time
        scale: ``h(log t | x) = h_0(log t) · exp(0.5 · x_s · γ) + x_d · β``.
        The fitted parameter vector gains a γ block exposed as
        :attr:`gamma_`, and the :meth:`survival`, :meth:`hazard`, and
        :meth:`predict` methods require ``X_scale`` / ``X_scale_new``.
        Sign-aligned with R (ADR 0002, Decision 5).

    Examples
    --------
    >>> from mltpy.tram import Survreg
    >>> from mltpy.variables import CensoredData
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = rng.lognormal(mean=1.0, sigma=0.5, size=200)
    >>> status = rng.binomial(1, 0.7, size=200).astype(bool)
    >>> cd = CensoredData.right_censored(t, censored=~status)
    >>> model = Survreg(support=(t.min() * 0.9, t.max() * 1.1)).fit(cd)
    >>> surv = model.survival(t)
    """

    def __init__(
        self,
        support: tuple[float, float],
        distribution: SurvregDistribution = "weibull",
        order: int = 6,
        optimizer_config: OptimizerConfig | None = None,
        scaling: NDArray[np.float64] | None = None,
    ) -> None:
        if distribution not in _SURVREG_DIST_MAP:
            raise ValueError(
                f"distribution={distribution!r} is not supported. "
                f"Valid options: {sorted(_SURVREG_DIST_MAP)}."
            )
        base_dist = cast(BaseDistribution, _SURVREG_DIST_MAP[distribution])
        log_basis = LogBernsteinBasis(order=order, support=support)

        # Bypass MLT.__init__ and call ConditionalTransformationModel directly,
        # because MLT builds a BernsteinBasis internally.  We store _order/_support
        # for _TramModel.__repr__.
        ConditionalTransformationModel.__init__(
            self,
            basis=log_basis,  # type: ignore[arg-type]
            censoring=CensoringType.RIGHT,
            optimizer_config=optimizer_config,
            base_distribution=base_dist,
            scaling=scaling,
        )
        self._order = order
        self._support = support
        self._distribution = distribution

    @property
    def gamma_(self) -> NDArray[np.float64]:
        """Fitted scaling-block coefficients ``γ`` (length ``q_s``).

        Sign-aligned with R ``tram::Survreg(..., scale=~x_s)``'s scaling
        block (ADR 0002, Decision 5).

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        ValueError
            If the model was constructed without ``scaling=``.
        """
        self._check_is_fitted()
        if self.scaling is None:
            raise ValueError("Model was not fitted with scaling=; gamma_ is undefined.")
        gamma = self.gamma_coef_
        if gamma is None:
            raise RuntimeError("Unexpected None gamma_coef_ for fitted model")
        return gamma

    @property
    def feature_names_scaling_(self) -> list[str]:
        """Column names of the scaling-design matrix supplied at fit time.

        Populated from a ``pandas.DataFrame`` column index when available,
        otherwise ``["X1", "X2", ...]``.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        ValueError
            If the model was constructed without ``scaling=``.
        """
        self._check_is_fitted()
        if self.scaling is None:
            raise ValueError(
                "Model was not fitted with scaling=; "
                "feature_names_scaling_ is undefined."
            )
        names = self.scaling_feature_names_in_
        if names is None:
            q_s = self.scaling.shape[1]
            return [f"X{j + 1}" for j in range(q_s)]
        return names

    # ------------------------------------------------------------------
    # Survival / hazard convenience methods (mirrors Coxph)
    # ------------------------------------------------------------------

    def survival(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
        X_scale: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Estimate the survival function S(t) = 1 − F(t | x).

        Parameters
        ----------
        y:
            Time points within ``support``.
        X:
            Optional covariate matrix of shape ``(m, q)``.
        offset:
            Optional per-observation offset.
        X_scale:
            New-data scaling-design matrix of shape ``(m, q_s)``, required
            when the model was fitted with ``scaling=``.  Threaded through
            to :meth:`predict` as ``X_scale_new``.

        Returns
        -------
        NDArray of shape ``(m,)`` with values in ``[0, 1]``.
        """
        return 1.0 - self.predict(
            y,
            X_new=X,
            what="distribution",
            offset_new=offset,
            X_scale_new=X_scale,
        )

    def hazard(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
        X_scale: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Estimate the hazard rate h_T(t) = f(t | x) / S(t | x).

        Parameters
        ----------
        y:
            Time points within ``support``.
        X:
            Optional covariate matrix.
        offset:
            Optional per-observation offset.
        X_scale:
            New-data scaling-design matrix of shape ``(m, q_s)``, required
            when the model was fitted with ``scaling=``.  Threaded through
            to :meth:`predict` as ``X_scale_new``.

        Returns
        -------
        NDArray of shape ``(m,)`` with non-negative values.
        """
        return self.predict(
            y,
            X_new=X,
            what="hazard",
            offset_new=offset,
            X_scale_new=X_scale,
        )

    # ------------------------------------------------------------------
    # Quantile prediction override
    # ------------------------------------------------------------------

    def _predict_quantile(
        self,
        probs: NDArray[np.float64],
        theta_b: NDArray[np.float64],
        xbeta: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Quantile inversion using basis.evaluate (handles log-transform).

        Overrides the parent to avoid the inlined Bernstein polynomial that
        bypasses the log-transform in LogBernsteinBasis.
        """
        import warnings

        from scipy.interpolate import CubicSpline

        from mltpy.likelihood import _get_dist
        from mltpy.model import _QMLT_GRID_POINTS

        a, b = self.basis.support
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

        # Right-censored: R-compatible grid+spline inversion on positive time grid.
        # Start at a (not 0) since log(0) = -inf for LogBernsteinBasis.
        if isinstance(self.basis, InteractionBasis):
            raise RuntimeError(
                "Survreg quantile inversion does not support InteractionBasis; "
                "this is an internal invariant violation."
            )
        q_grid = np.linspace(a, b, _QMLT_GRID_POINTS, dtype=np.float64)
        h_base_grid: NDArray[np.float64] = self.basis.evaluate(q_grid) @ theta_b

        eps = float(np.sqrt(np.finfo(float).eps))
        out = np.empty(n_probs, dtype=np.float64)
        saturated = False

        for i, p in enumerate(probs_arr):
            cdf_grid = dist.cdf(h_base_grid + shift[i])

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
            q_s = np.linspace(float(qk[0]), float(qk[-1]), n_spline, dtype=np.float64)
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
                "the finite inversion grid at one or more points; "
                "returning boundary-saturated quantiles.",
                stacklevel=3,
            )
        return out


# ---------------------------------------------------------------------------
# Polr — proportional-odds ordinal regression
# ---------------------------------------------------------------------------


PolrDistribution = Literal["logistic", "normal", "min_extreme_value"]


class Polr(ConditionalTransformationModel):
    """Proportional-odds ordinal regression (R ``tram::Polr``).

    For an ordered response ``Y ∈ {1, ..., K}`` and covariates ``x``, models

    .. math::

        P(Y \\leq k \\mid x) = F(\\theta_k + x^\\top \\beta),
        \\quad k = 1, \\ldots, K-1,

    where ``F`` is the CDF of the chosen base distribution and
    ``θ_1 ≤ ... ≤ θ_{K-1}`` are the cutpoints.  Internally implemented as a
    CTM with a degenerate :class:`~mltpy.basis.OrdinalBasis` plus the
    standard interval-censored likelihood path — the integer cut positions
    select the right ``θ_k`` per observation.

    .. note::
       **Sign convention.** mltpy parameterises ``h(y|x) = h(y) + x'β``, so
       the fitted ``β`` has the *opposite* sign of R's ``tram::Polr`` (which
       uses ``h(y) - x'β``).  Negate ``coef_`` to compare with R output.

    Parameters
    ----------
    levels:
        Optional explicit ordered tuple of category labels.  When ``None``
        (default), levels are inferred at :meth:`fit` time from a pandas
        ordered ``Categorical`` (uses ``cat.categories``) or, failing that,
        from sorted unique values of ``y``.
    distribution:
        Base distribution / link.

        * ``"logistic"`` (default) — proportional-odds model (R default).
        * ``"normal"`` — ordered probit.
        * ``"min_extreme_value"`` — proportional-hazards / cloglog link.
    optimizer_config:
        Optimisation settings.  If ``None``, library defaults are used.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from mltpy import Polr
    >>> rng = np.random.default_rng(0)
    >>> y = pd.Categorical(
    ...     rng.choice(["low", "mid", "high"], size=200),
    ...     categories=["low", "mid", "high"],
    ...     ordered=True,
    ... )
    >>> X = rng.standard_normal((200, 2))
    >>> m = Polr().fit(y, X)
    >>> probs = m.predict_proba(X[:5])
    """

    def __init__(
        self,
        levels: Sequence[Any] | None = None,
        distribution: PolrDistribution = "logistic",
        optimizer_config: OptimizerConfig | None = None,
    ) -> None:
        # Validate distribution eagerly (fast-fail before fit()).
        _get_dist(distribution)
        self._levels_arg = tuple(levels) if levels is not None else None
        self._distribution: PolrDistribution = distribution
        self._user_optimizer_config = optimizer_config
        self._ordvar: OrderedVariable | None = None

        # Defer ConditionalTransformationModel.__init__ until fit() — basis
        # depends on K which is only known once y is observed.  Initialise
        # the public attributes that callers of __repr__ may read pre-fit.
        self.is_fitted_: bool = False
        self.theta_: NDArray[np.float64] | None = None
        self.result_ = None
        self.basis: OrdinalBasis | None = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def levels_(self) -> tuple[Any, ...]:
        """Ordered category labels resolved at fit time."""
        if self._ordvar is None:
            from mltpy.model import NotFittedError

            raise NotFittedError("Model has not been fitted yet. Call fit(y) first.")
        return self._ordvar.levels

    @property
    def K_(self) -> int:  # noqa: N802 — match standard ordinal-regression notation
        """Number of ordered levels."""
        return len(self.levels_)

    @property
    def cutpoints_(self) -> NDArray[np.float64]:
        """Estimated cutpoints ``θ_1, ..., θ_{K-1}``."""
        self._check_is_fitted()
        if self.theta_ is None or self._ordvar is None:
            raise RuntimeError("Unexpected None theta_/_ordvar for fitted model")
        return self.theta_[: self._ordvar.K - 1].copy()

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Estimated regression coefficients ``β``.

        Length equals ``X.shape[1]`` from :meth:`fit`; empty array when
        no ``X`` was supplied.  mltpy's sign convention (``h + Xβ``) flips
        the sign relative to R ``tram::Polr`` (``h − Xβ``) — negate to
        compare.
        """
        self._check_is_fitted()
        if self.theta_ is None or self._ordvar is None:
            raise RuntimeError("Unexpected None theta_/_ordvar for fitted model")
        return self.theta_[self._ordvar.K - 1 :].copy()

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(  # type: ignore[override]
        self,
        y: Sequence[Any] | NDArray[Any],
        X: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> Polr:
        """Fit the Polr model by maximum likelihood.

        Parameters
        ----------
        y:
            Ordered categorical response.  Accepts a pandas ordered
            ``Categorical``/``Series``, or any sequence of hashable labels
            (whose sorted unique values determine the level order).
        X:
            Optional covariate matrix of shape ``(n, q)``.
        weights:
            Optional non-negative per-observation weights of shape ``(n,)``.
        offset:
            Optional fixed linear predictor offset of shape ``(n,)``.
        """
        ordvar, cd = OrderedVariable.from_labels(y, self._levels_arg)
        basis = OrdinalBasis(K=ordvar.K)

        ConditionalTransformationModel.__init__(
            self,
            basis=basis,  # type: ignore[arg-type]
            censoring=CensoringType.INTERVAL,
            optimizer_config=self._user_optimizer_config,
            base_distribution=self._distribution,
        )
        self._ordvar = ordvar
        super().fit(cd, X=X, weights=weights, offset=offset)
        return self

    def predict(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Disabled for ``Polr`` — use :meth:`predict_proba` or :meth:`predict_class`.

        Continuous CDF/density predictions are not meaningful for an ordinal
        response.
        """
        raise NotImplementedError(
            "Polr does not support predict(); use predict_proba(X) for level "
            "probabilities or predict_class(X) for the modal class."
        )

    def predict_proba(
        self,
        X: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute per-row level probabilities ``P(Y = level_k | x)``.

        Parameters
        ----------
        X:
            Optional covariate matrix of shape ``(m, q)``.  Must be supplied
            iff the model was fitted with covariates.
        offset:
            Optional offset of shape ``(m,)`` added to the linear predictor.

        Returns
        -------
        NDArray of shape ``(m, K)``.  Rows sum to 1.
        """
        self._check_is_fitted()
        if self.theta_ is None or self._ordvar is None:
            raise RuntimeError("Unexpected None theta_/_ordvar for fitted model")

        K = self._ordvar.K
        cutpoints = self.theta_[: K - 1]
        beta = self.theta_[K - 1 :]

        if X is None:
            if beta.size > 0:
                raise ValueError(
                    f"Model was fitted with {beta.size} covariates; "
                    "X must be provided to predict_proba()."
                )
            xbeta = np.zeros(1, dtype=np.float64)
            m = 1
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]
            if X_arr.shape[1] != beta.size:
                raise ValueError(
                    f"X has {X_arr.shape[1]} columns but the fitted model "
                    f"has {beta.size} covariate coefficients."
                )
            xbeta = X_arr @ beta if beta.size > 0 else np.zeros(X_arr.shape[0])
            m = X_arr.shape[0]

        if offset is not None:
            offset_arr = np.asarray(offset, dtype=float).ravel()
            if offset_arr.shape != (m,):
                raise ValueError(
                    f"offset must have shape ({m},), got {offset_arr.shape}."
                )
            xbeta = xbeta + offset_arr

        # Linear predictors at each cutpoint, shape (m, K-1).
        eta = cutpoints[None, :] + xbeta[:, None]
        eta = np.clip(eta, -_H_CLIP, _H_CLIP)
        dist = _get_dist(self._distribution)
        cdf = dist.cdf(eta)  # F(θ_k + x'β)

        # P(Y = level_k) = F(θ_k) − F(θ_{k-1}); θ_0=-∞ ⇒ F=0; θ_K=+∞ ⇒ F=1.
        probs = np.empty((m, K), dtype=np.float64)
        probs[:, 0] = cdf[:, 0]
        if K > 2:
            probs[:, 1 : K - 1] = np.diff(cdf, axis=1)
        probs[:, K - 1] = 1.0 - cdf[:, K - 2]
        # Guard against tiny negatives from floating-point rounding.
        return cast(NDArray[np.float64], np.clip(probs, 0.0, 1.0))

    def predict_class(
        self,
        X: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> NDArray[Any]:
        """Predict the modal level (argmax over ``predict_proba``).

        Returns the original level labels (decoded back from internal codes).
        """
        probs = self.predict_proba(X=X, offset=offset)
        codes = np.argmax(probs, axis=1) + 1
        if self._ordvar is None:
            raise RuntimeError("Unexpected None _ordvar for fitted model")
        return self._ordvar.decode(codes.astype(np.intp))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Multi-line summary with cutpoints and a Wald table for ``β``."""
        if not self.is_fitted_ or self._ordvar is None or self.theta_ is None:
            return f"Polr(distribution={self._distribution!r}, fitted=False)"
        K = self._ordvar.K
        lines = [
            "Model:        Polr",
            f"Distribution: {self._distribution}",
            f"Levels:       {list(self._ordvar.levels)}",
            "Fitted:       Yes",
        ]
        if self.result_ is not None:
            lines += [
                f"Log-lik:      {self.result_.log_likelihood:.4f}",
                f"AIC:          {self.aic():.4f}",
                f"BIC:          {self.bic():.4f}",
                f"Converged:    {'Yes' if self.result_.converged else 'No'}",
            ]
        cuts = self.theta_[: K - 1]
        cut_names = [
            f"{self._ordvar.levels[i]}|{self._ordvar.levels[i + 1]}"
            for i in range(K - 1)
        ]
        cut_lines = ["", "Cutpoints:"]
        cut_width = max(len(n) for n in cut_names)
        for name, val in zip(cut_names, cuts):
            cut_lines.append(f"  {name:<{cut_width}}  {val:>10.4f}")
        lines += cut_lines

        beta = self.theta_[K - 1 :]
        if beta.size:
            try:
                se = self.standard_errors()
                names = self.feature_names_in_ or [
                    f"X{j + 1}" for j in range(beta.size)
                ]
                lines += ["", "Coefficients (note: mltpy sign — negate for R):"]
                lines.append(_format_wald_table(names, beta, se[K - 1 :]))
            except RuntimeError:
                lines += [
                    "",
                    "  [Standard errors not available: Hessian matrix is singular.]",
                ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        if self.is_fitted_ and self.result_ is not None and self._ordvar is not None:
            return (
                f"Polr(distribution={self._distribution!r}, "
                f"K={self._ordvar.K}, fitted=True, "
                f"ll={self.result_.log_likelihood:.2f})"
            )
        return f"Polr(distribution={self._distribution!r}, fitted=False)"
