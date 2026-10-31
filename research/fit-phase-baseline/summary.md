### Environment

Apple M5 Pro (15 logical CPUs), macOS-26.6.2-arm64-arm-64bit; Python 3.12.16, NumPy 2.4.6, SciPy 1.17.1, mltpy 0.4.0; BLAS `accelerate` (system); thread env {'OMP_NUM_THREADS': None, 'VECLIB_MAXIMUM_THREADS': None, 'OPENBLAS_NUM_THREADS': None}; HEAD `b906968174`, src dirty=False; restart `random_state=0`. Load average at case start ranged 1.75–2.12 (not controlled).

### A. Timing distributions (all runs, nothing filtered)

Times in ms unless marked. *Cold* = first fit in a fresh process. *Warm* = subsequent fresh-model fits. *Refit* = same instance refitted.

| Case | n | m | Cold | Warm median [min–max] (reps) | Refit median (reps) | Raw `converged` | Restarts | Stored KKT (max) | Warnings |
|---|---:|---:|---:|---|---|---|---:|---:|---|
| `shift_none_n100` | 100 | 7 | 5.1 | 4.4 [4.2–4.6] (9) | 4.2 (5) | 10/10 | 0 | 9.3e-06 | — |
| `shift_right_n100` | 100 | 7 | 7.4 | 6.4 [6.3–6.7] (9) | 6.3 (5) | 10/10 | 0 | 8.8e-07 | — |
| `shift_interval_n1000` | 1000 | 7 | 16.0 | 14.3 [14.2–15.0] (9) | 14.5 (5) | 10/10 | 0 | 4.7e-06 | — |
| `shift_none_n1000` | 1000 | 7 | 8.1 | 7.2 [7.0–7.3] (9) | 7.1 (5) | 10/10 | 0 | 1.6e-06 | — |
| `shift_right_n1000` | 1000 | 7 | 21.2 | 19.9 [19.6–20.5] (9) | 19.7 (5) | 10/10 | 0 | 2.9e-06 | — |
| `shift_none_n5000` | 5000 | 7 | 29.2 | 23.0 [22.9–23.5] (9) | 23.0 (5) | 10/10 | 0 | 2.9e-05 | — |
| `shift_right_n5000` | 5000 | 7 | 37.5 | 34.9 [34.7–35.5] (9) | 34.6 (5) | 10/10 | 0 | 8.1e-05 | — |
| `boxcox_x3_n1000` | 1000 | 10 | 13.6 | 12.5 [12.3–12.7] (9) | 12.5 (5) | 10/10 | 0 | 6.8e-06 | — |
| `boxcox_x3_n5000` | 5000 | 10 | 38.9 | 36.2 [36.1–36.7] (9) | 36.6 (5) | 10/10 | 0 | 1.4e-05 | — |
| `coxph_x3_right_n1000` | 1000 | 10 | 20.1 | 18.8 [17.7–19.1] (9) | 18.9 (5) | 10/10 | 0 | 1.8e-05 | — |
| `coxph_x3_right_n5000` | 5000 | 10 | 52.5 | 50.4 [50.0–51.2] (9) | 50.3 (5) | 10/10 | 0 | 2.8e-05 | — |
| `boxcox_scaled_n1000` | 1000 | 9 | 17.6 | 16.3 [16.2–16.7] (9) | 16.1 (5) | 10/10 | 0 | 1.8e-05 | — |
| `boxcox_scaled_n5000` | 5000 | 9 | 41.9 | 39.7 [39.5–40.2] (9) | 39.6 (5) | 10/10 | 0 | 2.4e-05 | — |
| `ia_bs3x1_m8_n250` | 250 | 8 | 11.2 | 10.4 [10.3–10.5] (5) | 10.5 (3) | 6/6 | 0 | 7.8e-07 | — |
| `ia_bs3x1_m8_n1000` | 1000 | 8 | 25.3 | 24.2 [24.1–24.5] (5) | 24.2 (3) | 6/6 | 0 | 4.3e-07 | — |
| `ia_bs3x1_m8_n4000` | 4000 | 8 | 71.4 | 69.7 [69.3–70.4] (5) | 69.8 (3) | 6/6 | 0 | 2.7e-06 | — |
| `ia_R_logistic_3x3` | 300 | 9 | 18.0 | 17.2 [16.9–17.4] (9) | 16.8 (5) | 10/10 | 0 | 1.8e-07 | — |
| `ia_R_normal_3x3` | 300 | 9 | 13.4 | 12.6 [12.5–12.8] (9) | 12.7 (5) | 10/10 | 0 | 6.6e-07 | — |
| `ia_bs4x2_m15_n1000` | 1000 | 15 | 72.8 | 71.6 [70.0–72.6] (5) | 71.1 (3) | 6/6 | 0 | 3.0e-06 | — |
| `ia_bs6x3_m28_n250` | 250 | 28 | 121.9 | 120.4 [120.3–120.5] (2) | 121.1 (1) | 3/3 | 0 | 1.5e-06 | — |
| `ia_bs6x3_m28_n1000` | 1000 | 28 | 17.9 s | 17.9 s [17.9 s–17.9 s] (2) | 17.9 s (1) | 3/3 | 0 | 1.1e+02 | — |
| `ia_scaled_bs3x1_m9_n1000` | 1000 | 9 | 31.4 | 30.3 [30.1–30.4] (5) | 30.4 (3) | 6/6 | 0 | 4.8e-07 | UserWarning |
| `ia_scaled_R_stram_3x3+1` | 300 | 10 | 16.6 | 15.9 [15.5–16.0] (9) | 15.6 (5) | 10/10 | 0 | 5.9e-07 | UserWarning |
| `profile_ci_boxcox` | 150 | 6 | 628.0 | 625.9 [625.7–626.1] (2) | — | 3/3 | 0 | 9.1e-08 | — |
| `profile_ci_coxph` | 200 | 6 | 1,266.4 | 1,257.1 [1,253.2–1,261.0] (2) | — | 3/3 | 0 | 2.0e-06 | — |

### B. Phase attribution (one instrumented fit per case)

Exclusive phases: *solver* = all auglag attempts; *polish* = trust-ncg excluding its Hessians; *polish H* = Hessians requested inside polish; *final H* = eager post-fit Hessian; *score* = eager score matrix; *rest* = fit() minus those (validation, copies, constraint build). Percent of the instrumented fit. Obj = likelihood+gradient evaluations by phase; FD = scalar interaction log-likelihood calls made by the FD Hessian.

| Case | Instr. fit | Solver | Polish | Polish H (calls) | Final H | Score | Rest | Obj solver/polish | FD calls |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `shift_none_n100` | 4.2 | 98% | 0% | 0% (0) | 0% | 0% | 1% | 211/0 | 0 |
| `shift_right_n100` | 6.6 | 98% | 0% | 0% (0) | 0% | 0% | 1% | 194/0 | 0 |
| `shift_interval_n1000` | 14.3 | 98% | 0% | 0% (0) | 1% | 1% | 0% | 134/0 | 0 |
| `shift_none_n1000` | 7.0 | 98% | 0% | 0% (0) | 0% | 0% | 1% | 148/0 | 0 |
| `shift_right_n1000` | 19.6 | 99% | 0% | 0% (0) | 0% | 0% | 0% | 316/0 | 0 |
| `shift_none_n5000` | 23.3 | 99% | 0% | 0% (0) | 0% | 0% | 0% | 196/0 | 0 |
| `shift_right_n5000` | 34.4 | 99% | 0% | 0% (0) | 0% | 1% | 0% | 222/0 | 0 |
| `boxcox_x3_n1000` | 12.3 | 99% | 0% | 0% (0) | 0% | 0% | 0% | 235/0 | 0 |
| `boxcox_x3_n5000` | 37.0 | 99% | 0% | 0% (0) | 0% | 0% | 0% | 246/0 | 0 |
| `coxph_x3_right_n1000` | 18.5 | 99% | 0% | 0% (0) | 1% | 0% | 0% | 205/0 | 0 |
| `coxph_x3_right_n5000` | 50.6 | 99% | 0% | 0% (0) | 0% | 1% | 0% | 216/0 | 0 |
| `boxcox_scaled_n1000` | 16.4 | 99% | 0% | 0% (0) | 1% | 0% | 1% | 265/0 | 0 |
| `boxcox_scaled_n5000` | 39.4 | 99% | 0% | 0% (0) | 1% | 0% | 0% | 228/0 | 0 |
| `ia_bs3x1_m8_n250` | 10.2 | 64% | 0% | 0% (0) | 35% | 0% | 1% | 121/0 | 144 |
| `ia_bs3x1_m8_n1000` | 24.4 | 25% | 1% | 36% (1) | 36% | 0% | 1% | 59/3 | 288 |
| `ia_bs3x1_m8_n4000` | 70.4 | 26% | 1% | 36% (1) | 36% | 0% | 1% | 75/3 | 288 |
| `ia_R_logistic_3x3` | 16.9 | 32% | 1% | 33% (1) | 32% | 0% | 1% | 84/3 | 360 |
| `ia_R_normal_3x3` | 12.6 | 22% | 1% | 38% (1) | 37% | 0% | 2% | 49/3 | 360 |
| `ia_bs4x2_m15_n1000` | 73.5 | 49% | 0% | 0% (0) | 51% | 0% | 0% | 317/0 | 480 |
| `ia_bs6x3_m28_n250` | 120.3 | 45% | 0% | 0% (0) | 55% | 0% | 0% | 815/0 | 1,624 |
| `ia_bs6x3_m28_n1000` | 17.8 s | 0% | 0% | 99% (111) | 1% | 0% | 0% | 10/141 | 181,888 |
| `ia_scaled_bs3x1_m9_n1000` | 30.5 | 20% | 1% | 38% (1) | 39% | 0% | 1% | 64/3 | 360 |
| `ia_scaled_R_stram_3x3+1` | 15.5 | 58% | 0% | 0% (0) | 41% | 0% | 1% | 161/0 | 220 |

### C. Outcome validation (instrumented fit; independent of solver flags)

*Stored KKT* is auglag's pre-polish value on `result_`. *Pre/Final stat.* = ‖∇NLL − A_actᵀμ‖∞ with μ ≥ 0 by NNLS over rows with slack ≤ 1e-6 (recomputed from the public `negative_log_likelihood`). ΔNLL polish = after − before. SLSQP = `optimize(solver='slsqp', tol=1e-12)` from the default start; Δℓ = auglag − SLSQP (positive = auglag better).

| Case | Raw conv. | Stored KKT | Polish | ΔNLL polish | Pre stat. | Final stat. | Active rows | Min slack | SLSQP Δℓ | SLSQP max|Δθ| | R reference |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `shift_none_n100` | True | 9.3e-06 | skipped | — | — | 5.2e-06 | 3 | -3.8e-10 | 7.0e-10 | 1.8e-06 | — |
| `shift_right_n100` | True | 8.8e-07 | skipped | — | — | 8.8e-07 | 3 | -7.4e-08 | 3.2e-08 | 1.1e-05 | — |
| `shift_interval_n1000` | True | 4.7e-06 | skipped | — | — | 4.6e-06 | 2 | -7.4e-09 | -8.8e-08 | 8.5e-07 | — |
| `shift_none_n1000` | True | 1.6e-06 | skipped | — | — | 1.6e-06 | 2 | -6.9e-09 | 3.0e-08 | 3.1e-07 | — |
| `shift_right_n1000` | True | 2.9e-06 | skipped | — | — | 2.0e-06 | 3 | -1.6e-08 | -1.8e-08 | 2.4e-06 | — |
| `shift_none_n5000` | True | 2.9e-05 | skipped | — | — | 1.1e-05 | 3 | -1.1e-08 | -2.4e-07 | 1.4e-07 | — |
| `shift_right_n5000` | True | 8.1e-05 | skipped | — | — | 3.4e-05 | 2 | -1.6e-08 | -4.1e-08 | 1.1e-06 | — |
| `boxcox_x3_n1000` | True | 6.8e-06 | skipped | — | — | 5.3e-06 | 2 | -1.3e-08 | -6.5e-08 | 8.5e-07 | — |
| `boxcox_x3_n5000` | True | 1.4e-05 | skipped | — | — | 8.5e-06 | 2 | -2.7e-08 | -4.8e-07 | 2.3e-07 | — |
| `coxph_x3_right_n1000` | True | 1.8e-05 | skipped | — | — | 5.7e-06 | 3 | -2.1e-08 | 2.2e-07 | 8.7e-07 | — |
| `coxph_x3_right_n5000` | True | 2.8e-05 | skipped | — | — | 2.4e-05 | 2 | -3.3e-08 | 8.0e-07 | 2.5e-07 | — |
| `boxcox_scaled_n1000` | True | 1.8e-05 | skipped | — | — | 4.0e-06 | 3 | -1.1e-08 | 3.4e-09 | 2.3e-06 | — |
| `boxcox_scaled_n5000` | True | 2.4e-05 | skipped | — | — | 9.8e-06 | 2 | -2.3e-08 | 5.1e-07 | 1.1e-07 | — |
| `ia_bs3x1_m8_n250` | True | 7.8e-07 | skipped | — | — | 7.8e-07 | 1 | 3.6e-09 | -2.2e-10 | 5.2e-07 | — |
| `ia_bs3x1_m8_n1000` | True | 4.3e-07 | accepted | 0.0e+00 | 4.3e-07 | 4.3e-07 | 0 | 1.6e+00 | 2.3e-13 | 4.9e-07 | — |
| `ia_bs3x1_m8_n4000` | True | 2.7e-06 | accepted | 0.0e+00 | 2.7e-06 | 2.7e-06 | 0 | 1.6e+00 | 9.1e-13 | 1.9e-07 | — |
| `ia_R_logistic_3x3` | True | 1.8e-07 | accepted | 0.0e+00 | 1.8e-07 | 1.8e-07 | 0 | 4.2e+00 | 1.1e-13 | 1.0e-06 | max|ΔΘ| 1.7e-04, |Δℓ| 1.1e-08 |
| `ia_R_normal_3x3` | True | 6.6e-07 | accepted | 0.0e+00 | 6.6e-07 | 6.6e-07 | 0 | 2.2e+00 | -2.3e-13 | 7.2e-07 | max|ΔΘ| 8.4e-05, |Δℓ| 5.9e-09 |
| `ia_bs4x2_m15_n1000` | True | 3.0e-06 | skipped | — | — | 2.0e-06 | 2 | -3.4e-08 | 7.5e-09 | 5.2e-06 | — |
| `ia_bs6x3_m28_n250` | True | 1.5e-06 | skipped | — | — | 4.1e-07 | 13 | -2.0e-08 | 1.5e-09 | 2.8e-05 | — |
| `ia_bs6x3_m28_n1000` | True | 1.1e+02 | rejected | — | 1.1e+02 | 1.1e+02 | 0 | 1.2e-02 | -1.2e+03 | 7.3e+00 | — |
| `ia_scaled_bs3x1_m9_n1000` | True | 4.8e-07 | accepted | 0.0e+00 | 4.8e-07 | 4.8e-07 | 0 | 1.4e+00 | 2.3e-13 | 2.1e-07 | — |
| `ia_scaled_R_stram_3x3+1` | True | 5.9e-07 | skipped | — | — | 5.9e-07 | 1 | -1.2e-08 | 4.4e-09 | 1.7e-07 | |Δℓ| 4.4e-09, |Δγ| 6.5e-09 |
| `profile_ci_boxcox` | True | 9.1e-08 | — | — | — | — | — | — | — | — | max|ΔCI| 2.4e-07, non-finite 0 |
| `profile_ci_coxph` | True | 2.0e-06 | — | — | — | — | — | — | — | — | max|ΔCI| 2.2e-07, non-finite 0 |

### D. Memory (one traced fit; tracemalloc + process high-water)

| Case | Traced peak MiB | Retained w/ basis caches MiB | Retained model only MiB | Score matrix MiB | maxRSS after cold fit MiB |
|---|---:|---:|---:|---:|---:|
| `shift_none_n100` | 0.08 | 0.03 | 0.01 | 0.005 | 99 |
| `shift_right_n100` | 0.08 | 0.03 | 0.01 | 0.005 | 99 |
| `shift_interval_n1000` | 0.68 | 0.43 | 0.08 | 0.053 | 100 |
| `shift_none_n1000` | 0.39 | 0.24 | 0.07 | 0.053 | 99 |
| `shift_right_n1000` | 0.40 | 0.26 | 0.08 | 0.053 | 99 |
| `shift_none_n5000` | 1.65 | 1.19 | 0.31 | 0.267 | 100 |
| `shift_right_n5000` | 1.87 | 1.27 | 0.39 | 0.267 | 100 |
| `boxcox_x3_n1000` | 0.41 | 0.29 | 0.11 | 0.076 | 99 |
| `boxcox_x3_n5000` | 1.92 | 1.42 | 0.54 | 0.381 | 101 |
| `coxph_x3_right_n1000` | 0.45 | 0.31 | 0.13 | 0.076 | 99 |
| `coxph_x3_right_n5000` | 2.12 | 1.50 | 0.62 | 0.381 | 101 |
| `boxcox_scaled_n1000` | 0.50 | 0.27 | 0.09 | 0.069 | 99 |
| `boxcox_scaled_n5000` | 2.18 | 1.30 | 0.42 | 0.343 | 101 |
| `ia_bs3x1_m8_n250` | 0.16 | 0.07 | 0.03 | 0.015 | 99 |
| `ia_bs3x1_m8_n1000` | 0.53 | 0.25 | 0.08 | 0.061 | 99 |
| `ia_bs3x1_m8_n4000` | 2.04 | 0.98 | 0.31 | 0.244 | 100 |
| `ia_R_logistic_3x3` | 0.19 | 0.08 | 0.03 | 0.021 | 99 |
| `ia_R_normal_3x3` | 0.19 | 0.08 | 0.03 | 0.021 | 99 |
| `ia_bs4x2_m15_n1000` | 0.84 | 0.35 | 0.14 | 0.114 | 99 |
| `ia_bs6x3_m28_n250` | 0.38 | 0.15 | 0.07 | 0.053 | 99 |
| `ia_bs6x3_m28_n1000` | 1.41 | 0.53 | 0.24 | 0.214 | 100 |
| `ia_scaled_bs3x1_m9_n1000` | 0.61 | 0.26 | 0.09 | 0.069 | 99 |
| `ia_scaled_R_stram_3x3+1` | 0.21 | 0.09 | 0.03 | 0.023 | 99 |
| `profile_ci_boxcox` | 0.11 | 0.04 | 0.02 | 0.007 | 99 |
| `profile_ci_coxph` | 0.13 | 0.05 | 0.03 | 0.009 | 99 |

### E. Interaction Hessian: scalar FD (current) vs central difference of the analytic gradient

| Case | m | max|ΔH| | rel. | max rel. SE diff | min eig H | cond H | grad-FD time | current final-H time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ia_bs3x1_m8_n250` | 8 | 3.5e-04 | 2.9e-05 | 6.6e-04 | 7.80e-02 | 4.7e+02 | 0.5 | 3.6 |
| `ia_bs3x1_m8_n1000` | 8 | 1.1e-03 | 2.3e-05 | 9.9e-04 | 2.71e-01 | 5.4e+02 | 1.1 | 8.8 |
| `ia_bs3x1_m8_n4000` | 8 | 4.8e-03 | 2.4e-05 | 4.0e-04 | 8.18e-01 | 7.2e+02 | 3.1 | 25.4 |
| `ia_R_logistic_3x3` | 9 | 3.3e-04 | 6.1e-05 | 1.8e-03 | 5.57e-02 | 2.2e+02 | 0.8 | 5.5 |
| `ia_R_normal_3x3` | 9 | 4.0e-04 | 2.6e-05 | 4.0e-04 | 2.25e-01 | 1.6e+02 | 0.6 | 4.7 |
| `ia_bs4x2_m15_n1000` | 15 | 1.5e-03 | 6.4e-05 | 1.5e-02 | 9.70e-03 | 8.6e+03 | 2.5 | 37.3 |
| `ia_bs6x3_m28_n250` | 28 | 3.5e-04 | 1.2e-04 | nan | -5.06e-04 | 1.2e+301 | 2.6 | 66.2 |
| `ia_bs6x3_m28_n1000` | 28 | 2.6e-03 | 2.3e-05 | 7.3e-02 | 3.17e-03 | 7.3e+04 | 5.7 | 159.0 |
| `ia_scaled_bs3x1_m9_n1000` | 9 | 1.4e-03 | 8.6e-06 | 2.0e-03 | 2.67e-01 | 6.3e+02 | 1.4 | 11.9 |
| `ia_scaled_R_stram_3x3+1` | 10 | 2.4e-04 | 1.4e-06 | 3.7e-04 | 1.48e-01 | 1.2e+03 | 0.9 | 6.4 |

### F. Solver attempts (instrumented fit)

| Case | Attempt | Seconds | Converged | KKT | Outer | Inner | Message |
|---|---:|---:|---|---:|---:|---:|---|
| `shift_none_n100` | 0 | 4.1 | True | 9.3e-06 | 11 | 111 | Converged: feasible and parameter change below tolerance (KKT residual |
| `shift_right_n100` | 0 | 6.4 | True | 8.8e-07 | 6 | 130 | Converged: feasible and parameter change below tolerance (KKT residual |
| `shift_interval_n1000` | 0 | 14.0 | True | 4.7e-06 | 10 | 101 | Converged: feasible and parameter change below tolerance (KKT residual |
| `shift_none_n1000` | 0 | 6.9 | True | 1.6e-06 | 10 | 105 | Converged: feasible and parameter change below tolerance (KKT residual |
| `shift_right_n1000` | 0 | 19.4 | True | 2.9e-06 | 13 | 222 | Converged: feasible and parameter change below tolerance (KKT residual |
| `shift_none_n5000` | 0 | 23.0 | True | 2.9e-05 | 10 | 155 | Converged: feasible and parameter change below tolerance (KKT residual |
| `shift_right_n5000` | 0 | 34.0 | True | 8.1e-05 | 10 | 136 | Converged: feasible and parameter change below tolerance (KKT residual |
| `boxcox_x3_n1000` | 0 | 12.1 | True | 6.8e-06 | 12 | 186 | Converged: feasible and parameter change below tolerance (KKT residual |
| `boxcox_x3_n5000` | 0 | 36.5 | True | 1.4e-05 | 10 | 187 | Converged: feasible and parameter change below tolerance (KKT residual |
| `coxph_x3_right_n1000` | 0 | 18.3 | True | 1.8e-05 | 10 | 158 | Converged: feasible and parameter change below tolerance (KKT residual |
| `coxph_x3_right_n5000` | 0 | 49.9 | True | 2.8e-05 | 9 | 171 | Converged: feasible and parameter change below tolerance (KKT residual |
| `boxcox_scaled_n1000` | 0 | 16.2 | True | 1.8e-05 | 10 | 209 | Converged: feasible and parameter change below tolerance (KKT residual |
| `boxcox_scaled_n5000` | 0 | 38.9 | True | 2.4e-05 | 10 | 178 | Converged: feasible and parameter change below tolerance (KKT residual |
| `ia_bs3x1_m8_n250` | 0 | 6.5 | True | 7.8e-07 | 5 | 89 | Converged: feasible and parameter change below tolerance (KKT residual |
| `ia_bs3x1_m8_n1000` | 0 | 6.2 | True | 4.3e-07 | 2 | 45 | Converged: feasible and parameter change below tolerance (KKT residual |
| `ia_bs3x1_m8_n4000` | 0 | 18.0 | True | 2.7e-06 | 2 | 51 | Converged: feasible and parameter change below tolerance (KKT residual |
| `ia_R_logistic_3x3` | 0 | 5.4 | True | 1.8e-07 | 2 | 41 | Converged: feasible and parameter change below tolerance (KKT residual |
| `ia_R_normal_3x3` | 0 | 2.8 | True | 6.6e-07 | 2 | 40 | Converged: feasible and parameter change below tolerance (KKT residual |
| `ia_bs4x2_m15_n1000` | 0 | 36.0 | True | 3.0e-06 | 8 | 265 | Converged: feasible and parameter change below tolerance (KKT residual |
| `ia_bs6x3_m28_n250` | 0 | 53.9 | True | 1.5e-06 | 6 | 712 | Converged: feasible and parameter change below tolerance (KKT residual |
| `ia_bs6x3_m28_n1000` | 0 | 1.3 | True | 1.1e+02 | 2 | 3 | Converged: feasible and parameter change below tolerance (KKT residual |
| `ia_scaled_bs3x1_m9_n1000` | 0 | 6.1 | True | 4.8e-07 | 2 | 52 | Converged: feasible and parameter change below tolerance (KKT residual |
| `ia_scaled_R_stram_3x3+1` | 0 | 9.0 | True | 5.9e-07 | 6 | 122 | Converged: feasible and parameter change below tolerance (KKT residual |

### G. Profile-CI workload (whole fit + `confint(type='profile')`)

| Case | n | m | Cold | Warm | optimize() calls | solver attempts | time in optimize() |
|---|---:|---:|---:|---|---:|---:|---:|
| `profile_ci_boxcox` | 150 | 6 | 628.0 | 626.1, 625.7 | 128 | 128 | 100% |
| `profile_ci_coxph` | 200 | 6 | 1,266.4 | 1,261.0, 1,253.2 | 132 | 132 | 100% |
