"""Convenience layer for common conditional transformation models.

This module provides pre-configured wrappers around
:class:`~pymlt.model.ConditionalTransformationModel` / :class:`~pymlt.model.MLT`
that mirror the R ``tram`` package (Hothorn).  Users working with these classes
never need to import :class:`~pymlt.basis.BernsteinBasis`,
:class:`~pymlt.variables.CensoringType`, or
:class:`~pymlt.optimizer.OptimizerConfig` directly.

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

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as _norm

from pymlt.model import MLT
from pymlt.optimizer import OptimizerConfig
from pymlt.variables import CensoringType

# ---------------------------------------------------------------------------
# Internal base class
# ---------------------------------------------------------------------------


class _TramModel(MLT):
    """Base class for all tram convenience models.

    Extends :class:`~pymlt.model.MLT` with a diagnostic summary and an
    optional matplotlib plot.  Not part of the public API.
    """

    def __repr__(self) -> str:
        name = type(self).__name__
        censoring = self.censoring.name
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
        """Format a Wald coefficient table for the ``beta`` block.

        Returns ``None`` when the fitted model has no covariates or if the
        Hessian is singular so that standard errors cannot be computed.
        """
        if self.theta_ is None:
            return None
        p = self.basis.order + 1
        q = self.theta_.size - p
        if q <= 0:
            return None

        try:
            se = self.standard_errors()
        except RuntimeError:
            return "  [Standard errors not available: Hessian matrix is singular.]"

        beta = self.theta_[p:]
        beta_se = se[p:]
        z = beta / beta_se
        pvals = 2.0 * _norm.sf(np.abs(z))

        names = self.feature_names_in_ or [f"X{j + 1}" for j in range(q)]
        name_width = max(len(n) for n in names)
        name_width = max(name_width, 4)

        header = (
            f"  {'':<{name_width}}  {'Estimate':>10}  {'Std. Error':>10}  "
            f"{'z value':>8}  {'Pr(>|z|)':>9}"
        )
        rows = [header]
        for name, b, s, zv, pv in zip(names, beta, beta_se, z, pvals):
            rows.append(
                f"  {name:<{name_width}}  {b:>10.4f}  {s:>10.4f}  "
                f"{zv:>8.3f}  {pv:>9.4g}"
            )
        return "\n".join(rows)

    def plot(self, y: NDArray[np.float64], ax: Any = None) -> Any | list[Any]:
        """Plot the estimated CDF and density side by side.

        Parameters
        ----------
        y:
            Response values at which to evaluate the model.  Must lie within
            ``basis.support``.
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
                "Install with: pip install 'pymlt[plots]'"
            )

        self._check_is_fitted()
        y_arr = np.asarray(y, dtype=float).ravel()
        y_sorted = np.sort(y_arr)

        cdf = self.predict(y_sorted, what="distribution")
        pdf = self.predict(y_sorted, what="density")

        if ax is not None:
            try:
                ax_cdf, ax_pdf = ax
            except (TypeError, ValueError):
                # If we cannot unpack, assume it's a single Axes object
                if hasattr(ax, "plot"):
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

    Examples
    --------
    >>> from pymlt.tram import BoxCox
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
    ) -> None:
        super().__init__(
            order=order,
            support=support,
            censoring=CensoringType.NONE,
            optimizer_config=optimizer_config,
            base_distribution="normal",
        )

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
    >>> from pymlt.tram import Coxph
    >>> from pymlt.variables import CensoredData
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
    ) -> None:
        super().__init__(
            order=order,
            support=support,
            censoring=CensoringType.RIGHT,
            optimizer_config=optimizer_config,
            base_distribution="min_extreme_value",
        )

    def survival(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Estimate the survival function S(y) = 1 − F(y|x).

        Parameters
        ----------
        y:
            Time points within ``basis.support``.
        X:
            Optional covariate matrix of shape ``(m, q)``.

        Returns
        -------
        NDArray of shape ``(m,)`` with values in ``[0, 1]``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        return 1.0 - self.predict(y, X_new=X, what="distribution")

    def hazard(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Estimate the hazard rate h(y) = f(y|x) / S(y|x).

        Parameters
        ----------
        y:
            Time points within ``basis.support``.
        X:
            Optional covariate matrix of shape ``(m, q)``.

        Returns
        -------
        NDArray of shape ``(m,)`` with non-negative values.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        return self.predict(y, X_new=X, what="hazard")


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

    Examples
    --------
    >>> from pymlt.tram import Colr
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
    ) -> None:
        super().__init__(
            order=order,
            support=support,
            censoring=CensoringType.NONE,
            optimizer_config=optimizer_config,
            base_distribution="logistic",
        )


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

    The minus sign on :math:`\hat{\gamma}` reflects pymlt's internal shift
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

    Notes
    -----
    The Bernstein order is fixed at ``1`` by construction; passing an
    ``order`` keyword raises ``TypeError``.

    Examples
    --------
    >>> import numpy as np
    >>> from pymlt.tram import Lm
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
    ) -> None:
        super().__init__(
            order=1,
            support=support,
            censoring=CensoringType.NONE,
            optimizer_config=optimizer_config,
            base_distribution="normal",
        )

    def _baseline(self) -> tuple[float, float]:
        """Return the two baseline Bernstein coefficients (theta_0, theta_1)."""
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError("Unexpected None theta_ for fitted model")
        return float(self.theta_[0]), float(self.theta_[1])

    @property
    def sigma_(self) -> float:
        """Estimated residual standard deviation of the equivalent lm.

        Computed as ``(b - a) / (theta_1 - theta_0)``.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`fit`.
        RuntimeError
            If the fit is degenerate with ``theta_[1] == theta_[0]``, which
            is feasible under the non-strict monotonicity constraint but
            leaves the lm-equivalence mapping undefined.
        """
        t0, t1 = self._baseline()
        a, b = self._support
        denom = t1 - t0
        if denom <= 0.0:
            raise RuntimeError(
                f"Degenerate Lm fit: theta_[1] - theta_[0] = {denom!r} "
                "(expected > 0). The fitted transformation is constant "
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
        """
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
        """
        self._check_is_fitted()
        if self.theta_ is None:
            raise RuntimeError("Unexpected None theta_ for fitted model")
        beta_ctm = self.theta_[2:]
        return -self.sigma_ * beta_ctm

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
        p = self.basis.order + 1
        theta_b = self.theta_[:p]
        y_arr = np.asarray(y, dtype=float).ravel()
        B = self.basis.evaluate(y_arr)
        return B @ theta_b
