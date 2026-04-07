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
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

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
            lines += [
                f"Log-lik:      {self.result_.log_likelihood:.4f}",
                f"Converged:    {'Yes' if self.result_.converged else 'No'}",
                f"n_restarts:   {self.result_.n_restarts}",
            ]
        return "\n".join(lines)

    def plot(self, y: NDArray, ax=None):
        """Plot the estimated CDF and density side by side.

        Parameters
        ----------
        y:
            Response values at which to evaluate the model.  Must lie within
            ``basis.support``.
        ax:
            Optional 2-tuple ``(ax_cdf, ax_pdf)`` of ``matplotlib.axes.Axes``.
            If ``None``, a new figure with two subplots is created automatically.

        Returns
        -------
        list of matplotlib.axes.Axes
            Always ``[ax_cdf, ax_pdf]``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ImportError
            If matplotlib is not installed.
        TypeError
            If ``ax`` is provided but cannot be unpacked into two axes.
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
            ax_cdf, ax_pdf = ax  # TypeError for wrong shape (e.g. bare single Axes)
            fig = None
        else:
            fig, (ax_cdf, ax_pdf) = plt.subplots(1, 2, figsize=(10, 4))

        ax_cdf.plot(y_sorted, cdf)
        ax_cdf.set_xlabel("y")
        ax_cdf.set_ylabel("F(y)")
        ax_cdf.set_title(f"{type(self).__name__} — CDF")

        ax_pdf.plot(y_sorted, pdf)
        ax_pdf.set_xlabel("y")
        ax_pdf.set_ylabel("f(y)")
        ax_pdf.set_title(f"{type(self).__name__} — Density")

        if fig is not None:
            fig.tight_layout()

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
        optimizer_config: Optional[OptimizerConfig] = None,
    ) -> None:
        super().__init__(
            order=order,
            support=support,
            censoring=CensoringType.NONE,
            optimizer_config=optimizer_config,
            base_distribution="normal",
        )

    def fitted_transformation(self, y: NDArray) -> NDArray:
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
        p = self.basis.order + 1
        theta_b = self.theta_[:p]
        y_arr = np.asarray(y, dtype=float).ravel()
        B = self.basis.evaluate(y_arr)   # (m, p)
        return B @ theta_b               # h(y)


# ---------------------------------------------------------------------------
# Coxph
# ---------------------------------------------------------------------------

class Coxph(_TramModel):
    """Cox proportional hazards model for right-censored survival data.

    Fits a monotone transformation h(t) such that h(T) ~ N(0, 1) under
    right-censoring, which is equivalent to the Cox PH model when covariates
    enter linearly.  Extends the classical Cox model by estimating the full
    baseline distribution non-parametrically via a Bernstein polynomial.

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
        optimizer_config: Optional[OptimizerConfig] = None,
    ) -> None:
        super().__init__(
            order=order,
            support=support,
            censoring=CensoringType.RIGHT,
            optimizer_config=optimizer_config,
            base_distribution="normal",
        )

    def survival(
        self,
        y: NDArray,
        X: Optional[NDArray] = None,
    ) -> NDArray:
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
        y: NDArray,
        X: Optional[NDArray] = None,
    ) -> NDArray:
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
        optimizer_config: Optional[OptimizerConfig] = None,
    ) -> None:
        super().__init__(
            order=order,
            support=support,
            censoring=CensoringType.NONE,
            optimizer_config=optimizer_config,
            base_distribution="logistic",
        )
