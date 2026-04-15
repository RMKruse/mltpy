Quick start
===========

pymlt fits a monotone transformation ``h(y|x)`` such that ``h(Y|X)`` follows
a known reference distribution (standard normal by default). Once fitted,
the model answers distributional queries — CDF, density, quantile, hazard —
through a single ``predict`` call.

The snippet below fits the unconditional model to synthetic log-normal data
and prints the estimated median.

.. code-block:: python

   import numpy as np
   import pymlt

   rng = np.random.default_rng(0)
   y = rng.lognormal(mean=3.5, sigma=0.8, size=200).clip(0, 200)

   model = pymlt.MLT(order=6, support=(0, 200))
   model.fit(y)

   grid   = np.linspace(10, 180, 100)
   cdf    = model.predict(grid, what="distribution")
   median = model.predict(np.array([0.5]), what="quantile")[0]
   print(f"Estimated median: {median:.1f}")

Next, work through the :doc:`vignettes <examples/01_boxcox_regression>` for
three canonical use cases from the Hothorn papers: flexible Box-Cox
regression, survival analysis under right-censoring, and conditional
regression with covariates.

For the full list of prediction modes (``distribution``, ``density``,
``quantile``, ``hazard``) and the ``CensoredData`` container, see the
:doc:`API reference <api/model>`.
