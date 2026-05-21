pymlt — Conditional Transformation Models
==========================================

Python port of Hothorn's R ``mlt`` package. Fit flexible conditional
distributions to continuous, censored, or covariate-dependent data using
monotone Bernstein-polynomial transformations.

From a single fitted model, pymlt exposes the cumulative distribution,
density, quantile, and hazard functions — and can simulate synthetic
observations. The methodology is described in Hothorn, Kneib & Bühlmann
(2014) and Hothorn (2020).

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Vignettes

   examples/01_boxcox_regression
   examples/02_survival_analysis
   examples/03_regression_covariates
   examples/04_interacting_terms

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/variables
   api/basis
   api/constraints
   api/likelihood
   api/optimizer
   api/model
   api/tram

.. toctree::
   :maxdepth: 1
   :caption: Project

   citation
