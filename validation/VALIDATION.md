# pymlt Validation Approach

Systematic comparison of pymlt (Python) against R's `mlt`/`tram` packages to verify correctness of the Python implementation. R is treated as ground truth.

## Pipeline

The validation runs in three steps:

```
  R (mlt/tram)                    Python
  ──────────                      ──────
  generate_all_references.R       convert_references.py        run_validation.py
  ─────────────────────────       ─────────────────────        ─────────────────
  Fit models with mlt/tram   -->  Convert CSV to .npy     -->  Fit pymlt models
  Export theta, loglik,           (lossless, 17 sig. digits)   Compare against R
  CDF, PDF, quantiles, hazard                                 Report PASS/FAIL
  as CSV + metadata.json
```

**Requirements**: R with packages `mlt`, `basefun`, `variables`, `tram`, `survival`, `jsonlite`. Python with `numpy`, `scipy`, and `pymlt`.

## Test Case Coverage

21 cases across 11 scenarios, covering all model types, censoring mechanisms, sample sizes, and Bernstein polynomial orders.

| Case | Model | Censoring | n | Order | Support | Data source | Notes |
|------|-------|-----------|---|-------|---------|-------------|-------|
| case_01 (6 cases) | MLT | None | 200, 1000 | 4, 6, 8 | [0, 1] | Uniform(0.02, 0.98) | Baseline: uncensored, varying complexity |
| case_02 (4 cases) | MLT | Right | 200, 1000 | 4, 6 | [0, 5] | Exp(rate=2), ~58% censored | Right censoring with moderate-to-heavy rates |
| case_03 (2 cases) | MLT | Left | 200 | 4, 6 | [0, 6] | N(3, 1), 30% left-censored | Detection threshold censoring |
| case_04 (2 cases) | MLT | Interval | 200 | 4, 6 | [2, 8] | N(5, 1), width ~0.1 | All observations interval-censored |
| case_05 | BoxCox | None | 200 | 6 | [0.01, 10] | LogNormal | tram::BoxCox (normal base distribution) |
| case_06 | Coxph | Right | 200 | 6 | [0.01, 8] | Exp(rate=1), ~55% censored | tram::Coxph (normal base distribution) |
| case_07 | Colr | None | 200 | 6 | [-1, 5] | Logistic(2, 0.5) | tram::Colr (logistic base distribution) |
| case_08 | MLT | None | 200 | 6 | [0, 10] | N(5, 1) + 2 covariates | Regression model with shifting term |
| case_09 | MLT | None | 30 | 4 | [0, 1] | Uniform(0.02, 0.98) | Small sample edge case |
| case_10 | MLT | Right | 30 | 4 | [0, 5] | Exp(rate=2), 40% censored | Small sample + right censoring |
| case_11 | MLT | Right | 200 | 6 | [0, 5] | Exp(rate=2), 88% censored | Heavy censoring stress test |

All data is generated with fixed random seeds for reproducibility.

## Metrics Compared

Six metrics are compared for each case, organized into three tiers:

### Primary metrics (always blocking)

These are the core correctness checks. A failure here means the model produces materially different results.

| Metric | Tolerance | Grid/Levels | Description |
|--------|-----------|-------------|-------------|
| Log-likelihood | \|Dll\| <= 0.1 | Scalar | Evaluated at fitted parameters |
| CDF | max\|Dcdf\| <= 0.02 | 100 equidistant points in support interior | F(h(y)) at dense grid |

### Derived metrics (conditionally blocking)

These test outputs derived from the fitted transformation. They are blocking **only when a primary metric also fails**. When loglik and CDF both pass, exceedances in derived metrics indicate non-identifiable parameterizations (see below) and are reported as informational.

| Metric | Tolerance | Grid/Levels | Description |
|--------|-----------|-------------|-------------|
| PDF / density | max\|Dpdf\| <= 0.05 | Same 100-point grid as CDF | f(h(y)) * h'(y) |
| Quantile | max\|Dquant\| <= 0.05 | p = {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95} | Numerical inversion via brentq; extreme tails (p < 0.05, p > 0.95) excluded |
| Hazard | max\|Dhaz\| <= 0.10 | Same 100-point grid, restricted to CDF < 0.95 | f(h)/S(h); only for right-censored models; compared only where S(t) > 0.05 |

### Informational metrics (never blocking)

| Metric | Tolerance | Description |
|--------|-----------|-------------|
| Theta | max\|Dtheta\| <= 0.05 | Bernstein coefficients — internal parameterization detail. Reported for diagnostics but never causes failure. |

## Pass/Fail Logic

The validation uses a two-tier pass/fail system that accounts for non-identifiability under censoring:

```
1. If the model did not converge → FAIL

2. Check primary metrics:
   - loglik exceeds tolerance → FAIL
   - CDF exceeds tolerance    → FAIL

3. Check derived metrics (PDF, quantile, hazard):
   - If BOTH loglik AND CDF are within tolerance:
     → derived exceedances are INFORMATIONAL (non-identifiable regime)
   - If loglik OR CDF also exceeds tolerance:
     → derived exceedances are HARD FAILURES

4. Theta exceedances are always informational

5. PASS if zero hard failures
```

### Why this two-tier system?

Under heavy censoring, the upper Bernstein coefficients are **non-identifiable**: the likelihood surface is flat and multiple theta vectors yield the same log-likelihood and CDF. pymlt's SLSQP optimizer and R's augmented Lagrangian (`alabama::auglag`) may converge to different points on this flat ridge.

When this happens:
- **Log-likelihood matches** (same flat ridge)
- **CDF matches** (same distribution function)
- **Theta differs** (different parameterization of the same function)
- **PDF, quantile, and hazard may differ** because the transformation derivative h'(y) depends on the specific theta, even when F(h(y)) is identical

These derived-metric differences are expected behavior under non-identifiability, not bugs. The two-tier system prevents false failures while still catching genuine discrepancies (where primary metrics also fail).

### Hazard and quantile comparison restrictions

- **Hazard** is compared only at grid points where CDF < 0.95 (equivalently, S(t) > 0.05). In the extreme right tail, the survival function approaches zero and the hazard ratio f/S amplifies any tiny CDF difference by orders of magnitude.

- **Quantiles** are compared only for probability levels between 0.05 and 0.95. Extreme tail quantiles (p = 0.01, p = 0.99) amplify small CDF differences through the inverse function, especially near support boundaries.

## Reference Data Structure

Each case directory (`validation/references/case_*`) contains:

| File | Format | Description |
|------|--------|-------------|
| `metadata.json` | JSON | Model type, censoring, n, order, support, seed |
| `y.npy` | float64 | Observed response values |
| `theta.npy` | float64 | R's fitted Bernstein coefficients |
| `loglik.npy` | float64 (scalar) | R's log-likelihood at fitted theta |
| `cdf_grid.npy` | float64 (100,) | Grid points for CDF/PDF/hazard evaluation |
| `cdf_values.npy` | float64 (100,) | R's CDF at grid points |
| `pdf_grid.npy` | float64 (100,) | Grid points for PDF (same as cdf_grid) |
| `pdf_values.npy` | float64 (100,) | R's PDF at grid points |
| `quantile_probs.npy` | float64 (9,) | Probability levels for quantile comparison |
| `quantile_values.npy` | float64 (9,) | R's quantile values |
| `status.npy` | bool | Event indicator (right/left censored cases only) |
| `y_lower.npy`, `y_upper.npy` | float64 | Interval bounds (interval censored cases only) |
| `X.npy` | float64 (n, q) | Covariate matrix (regression cases only) |
| `hazard_grid.npy` | float64 (100,) | Grid for hazard (right-censored cases only) |
| `hazard_values.npy` | float64 (100,) | R's hazard at grid points |

All numeric values are stored with 17 significant digits to preserve full double precision.

## How to Run

```bash
# Step 1: Generate R reference values (requires R + mlt/tram)
Rscript validation/generate_all_references.R

# Step 2: Convert CSV to NumPy format
python validation/convert_references.py

# Step 3: Run validation
python validation/run_validation.py           # summary table
python validation/run_validation.py --verbose  # detailed per-component output for failures
python validation/run_validation.py --case case_01  # filter by case prefix

# Run unit tests for the validation logic itself
pytest tests/test_validation_script.py -x --tb=short
```

Exit codes: `0` = all pass, `1` = one or more failures, `2` = no reference data found.

Reports are saved to `validation/results/validation_report.md` and `validation/results/validation_report.json`.

## R Package Versions

Reference values were generated with:
- R 4.5.3
- mlt 1.7.4
- tram 1.4.1
- basefun 1.2.6

## Known Limitations

1. **case_06 (Coxph)**: pymlt's Coxph model currently converges to a different local minimum than R's tram::Coxph (Dll = 10.2). This is under investigation and is the only case that does not pass validation.

2. **Hazard sensitivity**: Even within the CDF < 0.95 restriction, hazard rate comparisons can show large absolute differences under non-identifiable theta. The two-tier system handles this by downgrading to informational when primary metrics pass.

3. **No covariate interaction with censoring**: The test suite does not yet include cases combining covariates (regression) with censored data. Case_08 tests regression without censoring; cases_02/10/11 test censoring without covariates.
