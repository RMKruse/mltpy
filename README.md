# pymlt — Conditional Transformation Models in Python

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)

Fit flexible conditional distributions to continuous, censored, or covariate-dependent data using monotone Bernstein polynomial transformations.

---

## Overview

pymlt estimates the full conditional distribution of a response variable — not just its mean. The core model fits a monotone transformation h(y|x) that maps observations to a standard normal distribution via maximum likelihood. Once fitted, the model yields CDFs, densities, quantile functions, hazard rates, and synthetic samples from a single object.

The package supports exact, right-censored, left-censored, and interval-censored data, with optional covariate matrices for conditional (regression) inference. It is a Python port of Hothorn (2020) `mlt` R package.

---

## Installation

```bash
pip install pymlt
```

Optional pandas support (for `pd.Series` inputs):

```bash
pip install "pymlt[pandas]"
```

**Requirements:** Python ≥ 3.10, numpy ≥ 1.24, scipy ≥ 1.10.

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

- Four prediction types from one fitted model: CDF, PDF, quantile, hazard rate
- Full censoring support: exact, right-, left-, and interval-censored observations
- Conditional distributions via optional covariate matrix `X`
- Analytical gradients for fast, stable MLE with automatic restarts on non-convergence
- scikit-learn-compatible API: `fit` / `predict` / `score` / `simulate`
- Lightweight: only numpy and scipy required
- Numerically stable: log-space likelihood, h-clipping, Taylor fallback for narrow intervals

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

```python
cfg = pymlt.OptimizerConfig(
    solver="trust-constr",
    max_iter=2000,
    max_restarts=5,
    verbose=True,
)
model = pymlt.MLT(order=6, support=(0, 1), optimizer_config=cfg)
```

---

## API reference

| Symbol | Description |
|---|---|
| `MLT(order, support, censoring, optimizer_config)` | Main entry point — Bernstein basis model with sensible defaults |
| `ConditionalTransformationModel(basis, censoring, optimizer_config)` | Base class for models with a custom `BernsteinBasis` |
| `CensoredData.right_censored(y, censored)` | Build a right-censored data container |
| `CensoredData.left_censored(y, censored)` | Build a left-censored data container |
| `CensoredData.interval_censored(lower, upper)` | Build an interval-censored data container |
| `CensoredData.from_exact(y)` | Wrap an exact (uncensored) array |
| `CensoringType` | Enum: `NONE` · `LEFT` · `RIGHT` · `INTERVAL` |
| `OptimizerConfig` | Tune solver, iteration limit, restarts, tolerance, gradient use |
| `NotFittedError` | Raised by `predict` / `score` / `simulate` before `fit` |
| `ConvergenceWarning` | Issued when MLE does not fully converge across all restarts |

### Prediction modes

| `what=` | Input | Output |
|---|---|---|
| `"distribution"` | y values in support | CDF: Φ(h(y\|x)) ∈ [0, 1] |
| `"density"` | y values in support | PDF: φ(h(y\|x)) · h′(y\|x) ≥ 0 |
| `"quantile"` | probabilities p ∈ (0, 1) | y such that P(Y ≤ y) = p |
| `"hazard"` | y values in support | φ(h) / (1 − Φ(h)) — RIGHT censoring only |

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
log(Φ(hᵢ_upper) − Φ(hᵢ_lower))). MLE is solved via scipy's SLSQP or
trust-constr solvers with analytical gradients.

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

## License

MIT © RMKruse
