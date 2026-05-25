# pymlt — Conditional Transformation Models in Python

[![CI](https://github.com/RMKruse/pymlt/actions/workflows/ci.yml/badge.svg)](https://github.com/RMKruse/pymlt/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://rmkruse.github.io/pymlt/)
[![codecov](https://img.shields.io/badge/coverage-tracked-informational)](https://github.com/RMKruse/pymlt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fit flexible conditional distributions to continuous, censored, or covariate-dependent data using monotone Bernstein polynomial transformations.

**Documentation:** <https://rmkruse.github.io/pymlt/> · **Paper:** [Hothorn (2020), JSS 92(1)](https://doi.org/10.18637/jss.v092.i01)

---

## Overview

pymlt estimates the full conditional distribution of a response variable — not just its mean. The core model fits a monotone transformation h(y|x) that maps observations to a standard normal distribution via maximum likelihood. Once fitted, the model yields CDFs, densities, quantile functions, hazard rates, and synthetic samples from a single object.

The package supports exact, right-censored, left-censored, and interval-censored data, with optional covariate matrices for conditional (regression) inference. It is a Python port of Hothorn (2020) `mlt` R package.

---

## Installation

```bash
pip install pymlt
```

Optional extras:

```bash
pip install "pymlt[plots]"      # matplotlib-backed .plot() helpers
pip install "pymlt[pandas]"     # pd.Series inputs
pip install "pymlt[examples]"   # lifelines, jupyter, matplotlib — run the vignettes
pip install "pymlt[docs]"       # sphinx, nbsphinx, pydata-sphinx-theme
```

**Requirements:** Python ≥ 3.12, numpy ≥ 1.24, scipy ≥ 1.10.

---

## Quick start

```python
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
```

---

## Features

- Fourteen prediction types from one fitted model: transformation, CDF, PDF, survivor, hazard, cumulative hazard, odds, quantile, and log-scale variants of each
- Full censoring support: exact, right-, left-, and interval-censored observations
- Conditional distributions via optional covariate matrix `X`
- Ready-made `tram` regression models mirroring R's `tram` package:
  - `BoxCox` — Box-Cox transformation for continuous outcomes
  - `Lm` — normal linear regression expressed as a CTM
  - `Coxph` — Cox proportional hazards for right-censored survival data
  - `Lehmann` — Lehmann / proportional reverse-time hazards (dual of `Coxph`)
  - `Colr` — continuous outcome logistic regression
  - `Polr` — proportional-odds ordinal regression
  - `Survreg` — parametric survival on the log-time scale (Weibull / log-normal / log-logistic)
- Seven selectable base distributions: `normal`, `logistic`, `min_extreme_value` (Cox link), `max_extreme_value` (Lehmann link), `exponential`, `laplace` (median regression), and `cauchy`
- Non-proportional / stratified-baseline models via tensor-product `InteractionBasis(y_basis, x_basis)` — see the [interacting-terms vignette](docs/examples/04_interacting_terms.ipynb)
- Heteroskedastic / scaled-baseline models via `scaling=X_s` on `BoxCox`, `Coxph`, `Colr`, `Lm`, `Survreg` — `h(y|x) = h_0(y)·exp(0.5·x_s·γ) + x_d·β`, mirroring R `tram::*(scale=~x_s)`; see the [scaling-terms vignette](docs/examples/05_scaling_terms.ipynb)
- Profile-likelihood confidence intervals via `confint(type="profile")` — inverts the χ²₁ LR test for asymmetric / boundary-bound parameters where the Wald approximation breaks down; see the [profile-likelihood vignette](docs/examples/06_profile_likelihood.ipynb)
- Full inference suite: variance–covariance (`vcov`), Wald & HC0 sandwich standard errors (`standard_errors` / `sandwich_se`), Wald confidence intervals & delta-method confidence bands (`confint` / `confband`), score / Cox–Snell / deviance residuals (`residuals`), likelihood-ratio model comparison (`anova`), and linear-restriction Wald tests (`wald_test`)
- Observation weights and offsets threaded through `fit` / `predict` / `score` / `confband` / `residuals`
- Analytical gradients for fast, stable MLE with automatic restarts on non-convergence
- scikit-learn-compatible API: `fit` / `predict` / `score` / `simulate`
- Lightweight: only numpy and scipy required
- Numerically stable: log-space likelihood, h-clipping, Taylor fallback for narrow intervals

---

## Performance

`pymlt.MLT.fit()` is on geometric mean **2.52× the speed of R `mlt::mlt()`** across the 24-cell grid `n ∈ {100, 500, 1000, 5000} × order ∈ {4, 6, 8} × censoring ∈ {none, right}` (median per cell over the converged reps). pymlt is the faster backend in all 24 cells (none 2.38×, right 2.68×). Representative slice at **`order = 6`** (full grid in the report linked below):

| n | Censoring | Python (median) | R (median) | Speedup |
|---:|:---|---:|---:|---:|
|  100 | none  | 3.67 ms  | 5.57 ms  | 1.52× |
|  500 | none  | 5.46 ms  | 10.10 ms | 1.85× |
| 1000 | none  | 4.93 ms  | 15.56 ms | 3.16× |
| 5000 | none  | 16.31 ms | 69.84 ms | 4.28× |
|  100 | right | 6.69 ms  | 12.19 ms | 1.82× |
|  500 | right | 6.77 ms  | 18.72 ms | 2.77× |
| 1000 | right | 16.28 ms | 42.38 ms | 2.60× |
| 5000 | right | 20.90 ms | 62.97 ms | 3.01× |

Hardware: Apple M5 Pro, R 4.5.3 + mlt 1.7.4, Python 3.12.13 + numpy 2.4.4 + scipy 1.17.1. Numbers depend on hardware and R/Python versions; the speedup ratio is the meaningful comparison.

**Reproduce:** `make benchmark` (requires R with `mlt`, `basefun`, `variables`, `survival` installed). The full grid, environment metadata, and IQR per cell live in [`benchmarks/results/benchmark_report.md`](benchmarks/results/benchmark_report.md).

---

## Usage

### Survival analysis with right-censored data

```python
import numpy as np
import pymlt

times    = np.array([12.5, 45.2, 23.1, 89.3, 55.0, 31.7, 78.4])
censored = np.array([False, True, False, False, True, False, True])

cd = pymlt.CensoredData.right_censored(times, censored)

model = pymlt.MLT(
    order=5,
    support=(0, 365),
    censoring=pymlt.CensoringType.RIGHT,
)
model.fit(cd)

t_grid = np.linspace(1, 360, 200)
hazard = model.predict(t_grid, what="hazard")
cdf    = model.predict(t_grid, what="distribution")

q25, q50, q75 = model.predict(np.array([0.25, 0.50, 0.75]), what="quantile")
print(f"Q1={q25:.1f}  Median={q50:.1f}  Q3={q75:.1f}")
```

### Conditional distributions with covariates

Passing a covariate matrix `X` of shape `(n, q)` fits a conditional model
P(Y ≤ y | X = x). The last `q` entries of `theta_` are regression coefficients.

```python
rng = np.random.default_rng(1)
n   = 300
X   = rng.standard_normal((n, 2))
y   = rng.uniform(0.05, 0.95, n)

model = pymlt.MLT(order=4, support=(0, 1))
model.fit(y, X=X)

X_new = np.array([[0.0, 1.0], [-1.0, 0.5]])
y_new = np.array([0.5, 0.5])
cdf   = model.predict(y_new, X_new=X_new, what="distribution")
```

### Interval-censored data

```python
centers = np.linspace(0.1, 0.9, 50)
cd = pymlt.CensoredData.interval_censored(
    lower=centers - 0.05,
    upper=centers + 0.05,
)
model = pymlt.MLT(order=4, support=(0, 1), censoring=pymlt.CensoringType.INTERVAL)
model.fit(cd)
```

### Sampling synthetic data

```python
# Simulate 1000 observations from the fitted distribution
samples = model.simulate(n=1000, random_state=42)
```

### Custom optimizer settings

The default solver is `"auglag"` — a Powell–Hestenes–Rockafellar augmented
Lagrangian that mirrors R `mlt`'s `alabama::auglag` and gives the closest
parity with the reference R implementation. `"slsqp"` and `"trust-constr"`
remain as opt-in alternatives; SLSQP is faster on small unconstrained-like
problems, trust-constr handles ill-conditioned ones better.

```python
cfg = pymlt.OptimizerConfig(
    solver="slsqp",          # opt-in alternative to the auglag default
    max_iter=2000,
    max_restarts=5,
    verbose=True,
)
model = pymlt.MLT(order=6, support=(0, 1), optimizer_config=cfg)
```

---

## API reference

**Models**

| Symbol | Description |
|---|---|
| `MLT(order, support, censoring, base_distribution, optimizer_config)` | Main entry point — Bernstein basis model with sensible defaults |
| `ConditionalTransformationModel(basis, censoring, base_distribution, optimizer_config)` | Base class for models with a custom basis |
| `BoxCox` · `Lm` · `Coxph` · `Lehmann` · `Colr` · `Polr` · `Survreg` | `tram`-style regression models (see [Features](#features)) |

**Bases** (pass to `ConditionalTransformationModel(basis=...)`)

| Symbol | Description |
|---|---|
| `OrdinalBasis(K)` | Degenerate one-hot cutpoint basis for ordinal responses |
| `OneHotBasis` · `InterceptBasis` | Non-negative partition-of-unity x-bases for stratified / interaction terms |
| `PolynomialBasis` · `LegendreBasis` · `LogBasis` | Alternative response-basis families |
| `InteractionBasis(y_basis, x_basis)` | Tensor-product basis for fully-interacting CTMs |

**Data & configuration**

| Symbol | Description |
|---|---|
| `CensoredData.right_censored(y, censored)` | Build a right-censored data container |
| `CensoredData.left_censored(y, censored)` | Build a left-censored data container |
| `CensoredData.interval_censored(lower, upper)` | Build an interval-censored data container |
| `CensoredData.from_exact(y)` | Wrap an exact (uncensored) array |
| `CensoringType` | Enum: `NONE` · `LEFT` · `RIGHT` · `INTERVAL` |
| `OrderedVariable` | Ordered-factor variable for ordinal responses |
| `OptimizerConfig` | Tune solver, iteration limit, restarts, tolerance, gradient use, `fixed_params` |
| `AugLagOptions` · `AugLagResult` | Augmented-Lagrangian solver options and result |

**Inference & primitives**

| Symbol | Description |
|---|---|
| `anova(*models)` → `AnovaResult` | Likelihood-ratio test for nested models |
| `WaldTestResult` | Result of `model.wald_test(R, r)` for linear restrictions |
| `log_likelihood` · `negative_log_likelihood` | Log-likelihood with analytical gradients |
| `hessian` · `score_matrix` | Observed information and per-observation scores |

**Exceptions & warnings**

| Symbol | Description |
|---|---|
| `NotFittedError` | Raised by `predict` / `score` / `simulate` before `fit` |
| `ConvergenceWarning` | Issued when MLE does not fully converge across all restarts |
| `InfeasibleParameterError` | Raised when a parameter vector violates the monotonicity constraint |

### Inference & diagnostics methods

Available on every fitted `ConditionalTransformationModel` (and its `tram` subclasses):

| Method | Description |
|---|---|
| `vcov(regularize="active")` | Variance–covariance matrix (active-set-constrained bordered-KKT default) |
| `standard_errors()` · `sandwich_se()` | Wald and HC0 sandwich standard errors |
| `confint(level, parm, type="wald"\|"profile")` | Wald or profile-likelihood confidence intervals |
| `confband(y_grid, X, level, what)` | Pointwise delta-method confidence bands |
| `residuals(type="score"\|"cox-snell"\|"deviance")` | Per-observation diagnostics |
| `estfun()` / `score_contributions()` | Per-observation score contributions |
| `wald_test(R, r)` | Wald test for linear restrictions `Rθ = r` |

### Prediction modes

`predict(y_new, X_new=None, what=...)` exposes fourteen output types from
a single fit. Let `h = h(y|x)`, `h' = ∂h/∂y`, and let `F`, `S`, `f`
denote the base distribution's CDF, survivor, and PDF.

| `what=` | Input | Output |
|---|---|---|
| `"trafo"` | y values in support | Transformation `h(y\|x)` |
| `"distribution"` | y values in support | CDF: `F(h)` ∈ [0, 1] |
| `"logdistribution"` | y values in support | `log F(h)` |
| `"survivor"` | y values in support | `S(h) = 1 − F(h)` |
| `"logsurvivor"` | y values in support | `log S(h)` |
| `"density"` | y values in support | PDF: `f(h) · h'` ≥ 0 |
| `"logdensity"` | y values in support | `log f(h) + log h'` |
| `"hazard"` | y values in support | `f(h) · h' / S(h)` |
| `"loghazard"` | y values in support | `log f(h) + log h' − log S(h)` |
| `"cumhazard"` | y values in support | Cumulative hazard: `−log S(h)` |
| `"logcumhazard"` | y values in support | `log(−log S(h))` |
| `"odds"` | y values in support | `F(h) / S(h)` |
| `"logodds"` | y values in support | `log F(h) − log S(h)` |
| `"quantile"` | probabilities p ∈ (0, 1) | y such that P(Y ≤ y) = p (numerical inversion) |

Log-scale variants use `dist.logcdf`/`logsf`/`logpdf` directly and stay
finite in tails where the primal quantities would under- or overflow.

---

## Background

<details>
<summary>Mathematical formulation</summary>

A conditional transformation model specifies:

```
h(y | x) = B_k(y) @ θ + x @ β
```

where B_k(y) is a Bernstein polynomial basis of degree k evaluated at y,
θ is a non-decreasing coefficient vector (monotonicity enforced via
D @ θ ≥ 0 where D is the forward-difference matrix), and β are optional
regression coefficients for covariates x.

The model assumes h(Y | X) ~ N(0, 1), so the log-likelihood for exact
observations is:

```
ℓ(θ, β) = Σᵢ [ log φ(hᵢ) + log h′(yᵢ) ]
```

with analogous terms for censored observations (log Φ, log(1 − Φ), or
log(Φ(hᵢ_upper) − Φ(hᵢ_lower))). MLE is solved with a Powell–Hestenes–
Rockafellar augmented Lagrangian (mirrors R `mlt`'s `alabama::auglag`) using
analytical gradients; scipy SLSQP and trust-constr are available opt-in.

</details>

---

## Reference
Hothorn, T., Kneib, T. and Bühlmann, P. (2014), Conditional transformation models. 
*Journal of the Royal Statistical Society: Series B (Statistical Methodology).*, 76: 3-27. 
https://doi.org/10.1111/rssb.12017

Hothorn, T. (2020). Most Likely Transformations: The mlt Package.
*Journal of Statistical Software*, 92(1), 1–68.
https://doi.org/10.18637/jss.v092.i01

Hothorn, T., Möst, L., and Bühlmann, P. (2018) Most Likely Transformations. 
*Scandinavian Journal of Statistics*, 45: 110–134. 
https://doi.org/10.1111/sjos.12291


---

## Citation

If you use pymlt in scientific work, please cite the package alongside
the methodological papers above:

```bibtex
@software{pymlt,
  author  = {Kruse, Ren{\'e}-Marcel},
  title   = {pymlt: Conditional Transformation Models in Python},
  year    = {2026},
  url     = {https://github.com/RMKruse/pymlt},
  version = {0.4.0}
}
```

Full BibTeX entries for the underlying methodology live in the
[documentation](https://rmkruse.github.io/pymlt/citation.html).

---

## License

MIT © RMKruse
