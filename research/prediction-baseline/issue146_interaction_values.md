# Interaction quantile value-only evaluation (#146)

Compared with `860d664a0dd697a426735df60dcd75f5852ed0df` on the same fitted models from the #140 prediction harness. The original harness is preserved on the `research/prediction-baseline` branch. Seven alternating warm repetitions per cell; medians exclude fitting and tracing. Prediction inputs use the original harness seeds (`1000 + m`), with `random_state=0` for fits. Every fit was independently checked against SLSQP (maximum |Δℓ| < 8e-9). Thread count and system load were not controlled.

All 16 original model × batch-size cells are reported below for both operations, plus eight Survreg cells. All 48 operation measurements are bitwise equal to the original implementation, with identical warning categories, messages and counts. Simulation measurements call public `simulate()` with a fixed seed, including saturated tail draws.

Bernstein interaction predictions are 2.00–2.28× faster. At 100,000 rows, retained traced allocation (including the output) drops from 343–562 MiB to 6.9–11.5 MiB; peak drops from 367–598 MiB to 28.5–43.0 MiB. The remaining shared-cache entries are the x-design and two support endpoints, with no midpoint entries. Survreg uses an order-1 log-time y-basis with a Weibull link, with and without scaling: 1.41–2.00× faster, at most 6.9 MiB retained and 23.9 MiB peak. Other Survreg links retain their existing R quantile coverage.

Environment: macOS arm64, Python 3.13.13, NumPy 2.5.4, SciPy 1.18.1, Accelerate BLAS. Results are machine-specific; they are not a portable timing threshold.

Reproduce from the repository root:

```sh
.venv/bin/python research/prediction-baseline/compare_interaction_values.py \
  --base 860d664a0dd697a426735df60dcd75f5852ed0df \
  --out research/prediction-baseline/results/issue146_interaction_values.json
```

Individual timings, fit validation, warnings, output sizes and cache byte counts are in [the raw JSON](results/issue146_interaction_values.json). Memory is measured once per variant after clearing both shared caches; retained and peak include the live output. The timing comparison loads only the original interaction quantile method from the base ref, so both versions use identical fitted parameters and current public basis evaluation.

## quantile

| Model | Rows | Before ms | After ms | Speedup | Retained MiB before → after | Peak MiB before → after |
|---|---:|---:|---:|---:|---:|---:|
| ia_bs3x1_m8_n1000 | 100 | 1.287 | 0.592 | 2.17× | 0.385 → 0.014 | 0.418 → 0.045 |
| ia_bs3x1_m8_n1000 | 1,000 | 4.406 | 2.083 | 2.12× | 3.467 → 0.075 | 3.770 → 0.355 |
| ia_bs3x1_m8_n1000 | 10,000 | 34.236 | 16.820 | 2.04× | 34.298 → 0.694 | 36.802 → 2.969 |
| ia_bs3x1_m8_n1000 | 100,000 | 322.033 | 161.237 | 2.00× | 342.602 → 6.874 | 366.477 → 28.460 |
| ia_bs4x2_m15_n1000 | 100 | 1.368 | 0.653 | 2.10× | 0.456 → 0.016 | 0.495 → 0.052 |
| ia_bs4x2_m15_n1000 | 1,000 | 5.229 | 2.448 | 2.14× | 4.199 → 0.100 | 4.555 → 0.425 |
| ia_bs4x2_m15_n1000 | 10,000 | 41.581 | 20.576 | 2.02× | 41.622 → 0.923 | 44.507 → 3.503 |
| ia_bs4x2_m15_n1000 | 100,000 | 404.504 | 199.596 | 2.03× | 415.844 → 9.162 | 443.534 → 33.801 |
| ia_bs6x3_m28_n250 | 100 | 1.648 | 0.724 | 2.28× | 0.602 → 0.020 | 0.651 → 0.065 |
| ia_bs6x3_m28_n250 | 1,000 | 6.822 | 3.279 | 2.08× | 5.664 → 0.123 | 6.119 → 0.540 |
| ia_bs6x3_m28_n250 | 10,000 | 56.828 | 28.093 | 2.02× | 56.270 → 1.152 | 59.842 → 4.342 |
| ia_bs6x3_m28_n250 | 100,000 | 565.106 | 276.691 | 2.04× | 562.327 → 11.452 | 596.884 → 42.194 |
| ia_scaled_bs3x1_m9_n1000 | 100 | 1.343 | 0.617 | 2.18× | 0.384 → 0.015 | 0.417 → 0.047 |
| ia_scaled_bs3x1_m9_n1000 | 1,000 | 4.475 | 2.152 | 2.08× | 3.467 → 0.077 | 3.769 → 0.357 |
| ia_scaled_bs3x1_m9_n1000 | 10,000 | 34.348 | 16.984 | 2.02× | 34.297 → 0.695 | 36.801 → 2.970 |
| ia_scaled_bs3x1_m9_n1000 | 100,000 | 325.772 | 162.273 | 2.01× | 342.601 → 6.874 | 366.476 → 28.460 |
| survreg_interacting | 100 | 1.216 | 0.612 | 1.99× | 0.237 → 0.016 | 0.262 → 0.039 |
| survreg_interacting | 1,000 | 2.674 | 1.604 | 1.67× | 2.001 → 0.077 | 2.220 → 0.273 |
| survreg_interacting | 10,000 | 16.365 | 11.082 | 1.48× | 19.648 → 0.695 | 21.644 → 2.436 |
| survreg_interacting | 100,000 | 154.531 | 108.328 | 1.43× | 196.116 → 6.875 | 214.651 → 23.121 |
| survreg_interacting_scaled | 100 | 1.247 | 0.624 | 2.00× | 0.239 → 0.015 | 0.264 → 0.038 |
| survreg_interacting_scaled | 1,000 | 2.629 | 1.577 | 1.67× | 2.003 → 0.077 | 2.221 → 0.273 |
| survreg_interacting_scaled | 10,000 | 16.334 | 11.239 | 1.45× | 19.650 → 0.694 | 21.646 → 2.435 |
| survreg_interacting_scaled | 100,000 | 154.824 | 109.500 | 1.41× | 196.118 → 6.875 | 214.653 → 23.121 |

## simulate

| Model | Rows | Before ms | After ms | Speedup | Retained MiB before → after | Peak MiB before → after |
|---|---:|---:|---:|---:|---:|---:|
| ia_bs3x1_m8_n1000 | 100 | 1.313 | 0.603 | 2.18× | 0.384 → 0.014 | 0.419 → 0.047 |
| ia_bs3x1_m8_n1000 | 1,000 | 4.466 | 2.161 | 2.07× | 3.467 → 0.076 | 3.778 → 0.364 |
| ia_bs3x1_m8_n1000 | 10,000 | 33.885 | 16.547 | 2.05× | 34.298 → 0.694 | 36.880 → 3.047 |
| ia_bs3x1_m8_n1000 | 100,000 | 324.551 | 161.570 | 2.01× | 342.601 → 6.873 | 367.242 → 29.225 |
| ia_bs4x2_m15_n1000 | 100 | 1.388 | 0.650 | 2.14× | 0.457 → 0.016 | 0.498 → 0.054 |
| ia_bs4x2_m15_n1000 | 1,000 | 5.147 | 2.516 | 2.05× | 4.199 → 0.099 | 4.565 → 0.434 |
| ia_bs4x2_m15_n1000 | 10,000 | 41.637 | 20.647 | 2.02× | 41.620 → 0.924 | 44.583 → 3.582 |
| ia_bs4x2_m15_n1000 | 100,000 | 404.768 | 200.956 | 2.01× | 415.843 → 9.164 | 444.298 → 34.567 |
| ia_bs6x3_m28_n250 | 100 | 1.552 | 0.732 | 2.12× | 0.604 → 0.019 | 0.654 → 0.066 |
| ia_bs6x3_m28_n250 | 1,000 | 6.788 | 3.335 | 2.04× | 5.663 → 0.122 | 6.127 → 0.547 |
| ia_bs6x3_m28_n250 | 10,000 | 57.884 | 28.557 | 2.03× | 56.269 → 1.152 | 59.919 → 4.420 |
| ia_bs6x3_m28_n250 | 100,000 | 565.687 | 278.088 | 2.03× | 562.326 → 11.452 | 597.648 → 42.959 |
| ia_scaled_bs3x1_m9_n1000 | 100 | 1.343 | 0.625 | 2.15× | 0.384 → 0.016 | 0.419 → 0.049 |
| ia_scaled_bs3x1_m9_n1000 | 1,000 | 4.416 | 2.174 | 2.03× | 3.467 → 0.077 | 3.778 → 0.365 |
| ia_scaled_bs3x1_m9_n1000 | 10,000 | 34.349 | 17.118 | 2.01× | 34.297 → 0.695 | 36.879 → 3.048 |
| ia_scaled_bs3x1_m9_n1000 | 100,000 | 328.771 | 164.078 | 2.00× | 342.602 → 6.875 | 367.242 → 29.227 |
| survreg_interacting | 100 | 1.247 | 0.625 | 2.00× | 0.238 → 0.016 | 0.266 → 0.041 |
| survreg_interacting | 1,000 | 2.662 | 1.618 | 1.65× | 2.003 → 0.076 | 2.231 → 0.281 |
| survreg_interacting | 10,000 | 16.207 | 11.178 | 1.45× | 19.651 → 0.695 | 21.725 → 2.514 |
| survreg_interacting | 100,000 | 152.201 | 107.766 | 1.41× | 196.116 → 6.875 | 215.416 → 23.886 |
| survreg_interacting_scaled | 100 | 1.210 | 0.626 | 1.93× | 0.237 → 0.015 | 0.265 → 0.041 |
| survreg_interacting_scaled | 1,000 | 2.636 | 1.601 | 1.65× | 2.003 → 0.077 | 2.232 → 0.282 |
| survreg_interacting_scaled | 10,000 | 16.636 | 11.369 | 1.46× | 19.650 → 0.695 | 21.725 → 2.514 |
| survreg_interacting_scaled | 100,000 | 164.318 | 113.293 | 1.45× | 196.116 → 6.875 | 215.416 → 23.886 |

## Verification

- The cache regression first failed for Bernstein and log-Bernstein predictions and simulations; it now passes, including the polynomial-basis fallback.
- Existing R quantile fixtures, scaling and public cache/read-only contracts pass. R reference regeneration was attempted but stopped because the installed `tram` dependency `TH.data` is missing; partial generated changes were discarded and the committed fixtures were used.
- Strict mypy reports the same three pre-existing redundant-cast errors (two in `basis.py`, one in `optimizer.py`). With only that diagnostic disabled, source typechecking passes.
