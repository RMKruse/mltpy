# Gradient Validation

Systematic verification that pymlt's analytical gradients match finite-difference approximations. This is independent of R — it catches bugs that R comparison would miss if both implementations share the same error.

## Why This Matters

The validation against R (`validation/run_validation.py`) compares **outputs** (theta, loglik, CDF). If both R and pymlt compute the same wrong gradient, they could converge to the same wrong answer and the R comparison would pass. Finite-difference verification is the only way to confirm that the analytical gradient code in `pymlt/likelihood.py` is mathematically correct.

## What Is Tested

The test verifies all four gradient functions in `likelihood.py`:

| Function | Censoring Type | Log-Likelihood Term |
|----------|---------------|---------------------|
| `_grad_none` | Exact observations | log f(h) + log h' |
| `_grad_right` | Right-censored | log S(h) for censored obs |
| `_grad_left` | Left-censored | log F(h) for censored obs |
| `_grad_interval` | Interval-censored | log(F(h_upper) - F(h_lower)) |

### Test Matrix

**Full cross-product:** 4 censoring types x 2 base distributions x 3 theta positions x 2 covariate modes = 48 test configurations.

| Dimension | Values | Rationale |
|-----------|--------|-----------|
| Censoring | none, right, left, interval | Each has a distinct gradient function |
| Base distribution | normal, logistic | Different score functions: s(h)=h vs s(h)=2F(h)-1 |
| Theta position | initial, perturbed, converged | Catches bugs that only appear at specific parameter values |
| Covariates | without X, with X (q=2) | Verifies gradient w.r.t. both theta_b and beta |

### Theta Positions

1. **Initial**: `np.linspace(0, 1, p)` — the optimizer's starting point. Widely spaced coefficients.
2. **Perturbed**: `np.cumsum(rng.uniform(0.1, 0.5, p))` — random monotone vector in the interior of the feasible region. Tests a "mid-optimization" point.
3. **Converged**: Fit the model on the test data and use the converged `theta_`. Tests at the optimum where gradients should be near zero — this is where sign errors are hardest to detect.

All theta vectors satisfy the monotonicity constraint (`D @ theta_b >= 0`) by construction.

## Method

For each configuration:

1. **Compute analytical gradient** via `negative_log_likelihood(theta, ..., gradient=True)` which returns `(nll, grad)`.
2. **Compute finite-difference gradient** via `scipy.optimize.approx_fprime(theta, f, epsilon)` using central differences.
3. **Compare** with `np.testing.assert_allclose(analytical, finite_diff, rtol=1e-5, atol=1e-7)`.

### Why `approx_fprime` Instead of `check_grad`

`check_grad` returns a single scalar error norm, which can mask component-wise failures. `approx_fprime` returns the full gradient vector, allowing per-component comparison and more informative error messages on failure.

## Numerical Subtleties

### H-clipping discontinuity

The transformation value `h` is clipped to `[-30, 30]` before distribution calls (`_H_CLIP = 30.0`). At the clipping boundary, the gradient is technically undefined. Test theta values are constructed to keep `h` well within the clipped range.

### Hazard clamping in `_grad_right`

The hazard ratio `f(h)/S(h)` is clamped via `np.minimum(hazard, _H_CLIP)`. This introduces a non-differentiable point. Test data is chosen so the hazard stays below the clamp threshold.

### Taylor fallback in `_log_diff_ndtr`

For narrow intervals where `F(upper) - F(lower)` is very small, `_log_diff_ndtr` switches to a Taylor approximation `f(mid) * width`. The gradient through this branch differs subtly from the standard branch. The interval-censored tests include both wide and narrow intervals to exercise both code paths.

### Monotonicity of h'

The derivative `h' = D @ theta_b` must be positive everywhere for the log-likelihood to be finite (it contains `log(h')`). All test theta vectors are strictly ascending to ensure this.

## How to Run

```bash
pytest gradient_validation/test_gradient_verification.py -x --tb=short -v
```

## Files

| File | Description |
|------|-------------|
| `GRADIENT_VALIDATION.md` | This document |
| `test_gradient_verification.py` | Parametrized pytest tests (48 configurations) |
