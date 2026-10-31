# Right-censored quantile curve reuse (#145)

Compared with `0e7c5f23665498cb438f7d5ca88dbee9e01a1125` on the same fitted models from the #140 prediction harness. Nine alternating warm repetitions per cell; medians below exclude fitting and tracing. Inputs use the harness seeds (`1000 + m`), with `random_state=0` for fits. Each fit was checked against SLSQP by the existing harness. Thread count and system load were not controlled.

All 36 cells are bitwise equal to the original batch output, with identical warning categories, messages and counts. This includes the cells where simulation saturates. Peak traced allocation is at most 0.461 MiB; no distinct-shift cell regressed beyond 5%.

Reproduce from the repository root:

```sh
.venv/bin/python research/prediction-baseline/compare_shift_reuse.py \
  --base 0e7c5f23665498cb438f7d5ca88dbee9e01a1125 \
  --out research/prediction-baseline/results/issue145_shift_reuse.json
```

Environment and individual timings are in [the raw JSON](results/issue145_shift_reuse.json). `simulate` cells use the same clipped uniform inputs and prediction dispatch as the public method, as in the original harness.

| Model | Operation / shifts | Rows | Before ms | After ms | Speedup | Peak MiB |
|---|---|---:|---:|---:|---:|---:|
| right_baseline | quantile | 100 | 6.439 | 0.193 | 33.37× | 0.026 |
| right_baseline | quantile | 1,000 | 59.657 | 0.746 | 79.96× | 0.046 |
| right_baseline | quantile | 10,000 | 604.734 | 6.427 | 94.09× | 0.252 |
| right_baseline | simulate | 100 | 5.914 | 0.163 | 36.34× | 0.026 |
| right_baseline | simulate | 1,000 | 61.464 | 0.784 | 78.44× | 0.046 |
| right_baseline | simulate | 10,000 | 594.086 | 6.555 | 90.63× | 0.252 |
| coxph_x3 | quantile_varying | 100 | 5.940 | 6.091 | 0.98× | 0.083 |
| coxph_x3 | quantile_varying | 1,000 | 63.004 | 64.032 | 0.98× | 0.173 |
| coxph_x3 | quantile_varying | 10,000 | 583.488 | 601.151 | 0.97× | 0.379 |
| coxph_x3 | quantile_equal | 100 | 5.990 | 0.170 | 35.25× | 0.026 |
| coxph_x3 | quantile_equal | 1,000 | 60.007 | 0.757 | 79.26× | 0.047 |
| coxph_x3 | quantile_equal | 10,000 | 622.761 | 6.619 | 94.09× | 0.253 |
| coxph_x3 | quantile_10prof | 100 | 5.922 | 0.718 | 8.25× | 0.037 |
| coxph_x3 | quantile_10prof | 1,000 | 60.042 | 1.364 | 44.02× | 0.058 |
| coxph_x3 | quantile_10prof | 10,000 | 597.571 | 7.552 | 79.13× | 0.276 |
| coxph_x3 | quantile_equal_offset | 100 | 5.970 | 5.972 | 1.00× | 0.081 |
| coxph_x3 | quantile_equal_offset | 1,000 | 58.371 | 59.045 | 0.99× | 0.181 |
| coxph_x3 | quantile_equal_offset | 10,000 | 631.193 | 606.738 | 1.04× | 0.456 |
| coxph_x3 | simulate_equal | 100 | 5.991 | 0.173 | 34.61× | 0.026 |
| coxph_x3 | simulate_equal | 1,000 | 59.503 | 0.755 | 78.76× | 0.047 |
| coxph_x3 | simulate_equal | 10,000 | 584.307 | 6.596 | 88.59× | 0.253 |
| survreg_x3 | quantile_varying | 100 | 5.935 | 6.041 | 0.98× | 0.083 |
| survreg_x3 | quantile_varying | 1,000 | 58.501 | 59.521 | 0.98× | 0.180 |
| survreg_x3 | quantile_varying | 10,000 | 582.316 | 600.303 | 0.97× | 0.386 |
| survreg_x3 | quantile_equal | 100 | 6.238 | 0.189 | 32.92× | 0.033 |
| survreg_x3 | quantile_equal | 1,000 | 59.692 | 0.757 | 78.81× | 0.053 |
| survreg_x3 | quantile_equal | 10,000 | 610.783 | 6.504 | 93.92× | 0.259 |
| survreg_x3 | quantile_10prof | 100 | 5.980 | 0.692 | 8.64× | 0.044 |
| survreg_x3 | quantile_10prof | 1,000 | 62.098 | 1.394 | 44.55× | 0.065 |
| survreg_x3 | quantile_10prof | 10,000 | 604.435 | 7.646 | 79.05× | 0.283 |
| survreg_x3 | quantile_equal_offset | 100 | 5.929 | 5.955 | 1.00× | 0.083 |
| survreg_x3 | quantile_equal_offset | 1,000 | 58.824 | 59.914 | 0.98× | 0.187 |
| survreg_x3 | quantile_equal_offset | 10,000 | 585.908 | 600.123 | 0.98× | 0.461 |
| survreg_x3 | simulate_equal | 100 | 6.007 | 0.156 | 38.58× | 0.033 |
| survreg_x3 | simulate_equal | 1,000 | 60.244 | 0.767 | 78.50× | 0.053 |
| survreg_x3 | simulate_equal | 10,000 | 590.012 | 6.497 | 90.81× | 0.259 |
