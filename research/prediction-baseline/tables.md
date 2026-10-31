## A. Fitted models (independently validated)

| case | path | θ length | converged | ℓ | ℓ − ℓ_SLSQP | max abs Δθ vs SLSQP | valid |
|---|---|---|---|---|---|---|---|
| boxcox_scaled | shift-scaling | 9 | True | -1388.5869 | 3.4e-09 | 2.3e-06 | True |
| boxcox_x3 | shift-bisection | 10 | True | -1470.3644 | -6.5e-08 | 8.5e-07 | True |
| coxph_x3 | rstyle-parent | 10 | True | -1304.9444 | 2.2e-07 | 8.7e-07 | True |
| ia_bs3x1_m8_n1000 | interaction | 8 | True | -1408.9572 | 2.3e-13 | 4.9e-07 | True |
| ia_bs4x2_m15_n1000 | interaction | 15 | True | -1406.7845 | 7.5e-09 | 5.2e-06 | True |
| ia_bs6x3_m28_n250 | interaction | 28 | True | -347.8531 | 1.5e-09 | 2.8e-05 | True |
| ia_scaled_bs3x1_m9_n1000 | interaction | 9 | True | -1406.9554 | 2.3e-13 | 2.1e-07 | True |
| right_baseline | rstyle-parent | 7 | True | -1228.9515 | -1.8e-08 | 2.4e-06 | True |
| survreg_x3 | rstyle-survreg | 10 | True | -1281.0189 | 4.0e-09 | 3.3e-04 | True |

## B. Runtime and memory per operation

Times in ms (median). *first* = first call on new data with cleared caches; *same* = repeated call on identical inputs; *changing* = new inputs every call. Memory in MiB: traced peak and retained after one call (output excluded), global-cache bytes after one call (raw/assembled entries), then cache bytes and process RSS (before → after burst → after clearing caches) after a burst of changing-data calls (40 calls; 12 at m = 100,000). Model-instance array bytes never changed during prediction (Δ = 0 in every cell).

| case | op | m | first | same | changing | peak | retained | cache after 1 call | burst cache | RSS burst |
|---|---|---|---|---|---|---|---|---|---|---|
| boxcox_scaled | distribution | 100 | 0.17 | 0.03 | 0.08 | 0.05 | 0.02 | 0.02 (2/1) | 0.6 (64/32, 40 calls) | 107 → 108 → 108 |
| boxcox_scaled | distribution | 1,000 | 0.35 | 0.07 | 0.25 | 0.41 | 0.18 | 0.18 (2/1) | 5.6 (64/32, 40 calls) | 108 → 108 → 108 |
| boxcox_scaled | distribution | 10,000 | 2.15 | 0.44 | 2.20 | 3.51 | 1.76 | 1.75 (2/1) | 56.2 (64/32, 40 calls) | 116 → 164 → 164 |
| boxcox_scaled | distribution | 100,000 | 22.18 | 4.42 | 20.71 | 35.10 | 17.55 | 17.55 (2/1) | 210.6 (24/12, 12 calls) | 234 → 379 → 379 |
| boxcox_scaled | quantile | 100 | 0.71 | 0.55 | 0.60 | 0.03 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 379 → 379 → 379 |
| boxcox_scaled | quantile | 1,000 | 3.07 | 3.03 | 3.00 | 0.29 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 379 → 379 → 379 |
| boxcox_scaled | quantile | 10,000 | 26.28 | 25.76 | 25.82 | 1.90 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 380 → 322 → 322 |
| boxcox_scaled | quantile | 100,000 | 256.48 | 254.41 | 254.81 | 17.77 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 12 calls) | 180 → 180 → 180 |
| boxcox_x3 | distribution | 100 | 0.17 | 0.02 | 0.09 | 0.05 | 0.02 | 0.02 (2/1) | 0.6 (64/32, 40 calls) | 107 → 108 → 108 |
| boxcox_x3 | distribution | 1,000 | 0.37 | 0.07 | 0.26 | 0.41 | 0.18 | 0.18 (2/1) | 5.6 (64/32, 40 calls) | 108 → 108 → 108 |
| boxcox_x3 | distribution | 10,000 | 2.02 | 0.36 | 1.91 | 3.51 | 1.76 | 1.75 (2/1) | 56.2 (64/32, 40 calls) | 115 → 165 → 165 |
| boxcox_x3 | distribution | 100,000 | 19.13 | 3.31 | 18.17 | 35.10 | 17.55 | 17.55 (2/1) | 210.6 (24/12, 12 calls) | 243 → 387 → 387 |
| boxcox_x3 | density | 100 | 0.18 | 0.03 | 0.07 | 0.05 | 0.02 | 0.02 (2/1) | 0.6 (64/32, 40 calls) | 387 → 387 → 387 |
| boxcox_x3 | density | 1,000 | 0.31 | 0.06 | 0.23 | 0.41 | 0.18 | 0.18 (2/1) | 5.6 (64/32, 40 calls) | 387 → 387 → 387 |
| boxcox_x3 | density | 10,000 | 1.73 | 0.32 | 1.70 | 3.51 | 1.76 | 1.75 (2/1) | 56.2 (64/32, 40 calls) | 387 → 387 → 387 |
| boxcox_x3 | density | 100,000 | 18.67 | 2.85 | 17.58 | 35.10 | 17.55 | 17.55 (2/1) | 210.6 (24/12, 12 calls) | 387 → 388 → 388 |
| boxcox_x3 | quantile | 100 | 0.62 | 0.48 | 0.50 | 0.03 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 388 → 388 → 388 |
| boxcox_x3 | quantile | 1,000 | 3.01 | 3.00 | 2.93 | 0.28 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 388 → 388 → 388 |
| boxcox_x3 | quantile | 10,000 | 24.32 | 24.21 | 24.09 | 1.82 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 388 → 391 → 391 |
| boxcox_x3 | quantile | 100,000 | 239.15 | 240.56 | 237.53 | 17.01 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 12 calls) | 391 → 391 → 391 |
| boxcox_x3 | simulate | 100 | 0.59 | 0.54 | 0.48 | 0.03 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 391 → 391 → 391 |
| boxcox_x3 | simulate | 1,000 | 3.03 | 2.98 | 3.02 | 0.28 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 391 → 391 → 391 |
| boxcox_x3 | simulate | 10,000 | 24.17 | 24.37 | 24.16 | 1.82 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 391 → 391 → 391 |
| boxcox_x3 | simulate | 100,000 | 236.53 | 236.86 | 237.13 | 17.01 | 0.00 | 0.00 (0/0) | 0.0 (0/0, 12 calls) | 391 → 391 → 391 |
| coxph_x3 | distribution | 100 | 0.14 | 0.02 | 0.07 | 0.05 | 0.02 | 0.02 (2/1) | 0.6 (64/32, 40 calls) | 108 → 108 → 108 |
| coxph_x3 | distribution | 1,000 | 0.30 | 0.06 | 0.24 | 0.41 | 0.18 | 0.18 (2/1) | 5.6 (64/32, 40 calls) | 108 → 109 → 109 |
| coxph_x3 | distribution | 10,000 | 1.96 | 0.36 | 1.92 | 3.51 | 1.76 | 1.75 (2/1) | 56.2 (64/32, 40 calls) | 116 → 165 → 165 |
| coxph_x3 | distribution | 100,000 | 19.35 | 3.20 | 19.15 | 35.10 | 17.55 | 17.55 (2/1) | 210.6 (24/12, 12 calls) | 243 → 388 → 388 |
| coxph_x3 | quantile_varying | 100 | 6.59 | 6.32 | 6.29 | 0.08 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 388 → 388 → 388 |
| coxph_x3 | quantile_varying | 1,000 | 61.95 | 62.77 | 62.72 | 0.17 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 388 → 388 → 388 |
| coxph_x3 | quantile_varying | 10,000 | 618.96 | 607.48 | 612.03 | 0.31 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 388 → 259 → 259 |
| coxph_x3 | quantile_equal | 100 | 6.46 | 6.16 | 6.29 | 0.08 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 259 → 259 → 259 |
| coxph_x3 | quantile_equal | 1,000 | 61.60 | 60.93 | 61.46 | 0.17 | 0.03 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 259 → 259 → 259 |
| coxph_x3 | quantile_equal | 10,000 | 603.21 | 602.20 | 604.58 | 0.31 | 0.03 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 226 → 172 → 172 |
| coxph_x3 | quantile_10prof | 100 | 6.24 | 6.15 | 6.23 | 0.08 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 172 → 172 → 172 |
| coxph_x3 | quantile_10prof | 1,000 | 61.32 | 61.54 | 61.37 | 0.17 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 172 → 172 → 172 |
| coxph_x3 | quantile_10prof | 10,000 | 605.12 | 599.90 | 625.24 | 0.30 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 172 → 172 → 172 |
| coxph_x3 | quantile_equal_offset | 100 | 6.15 | 6.09 | 6.02 | 0.09 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 172 → 172 → 172 |
| coxph_x3 | quantile_equal_offset | 1,000 | 60.46 | 59.14 | 59.48 | 0.18 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 172 → 172 → 172 |
| coxph_x3 | quantile_equal_offset | 10,000 | 595.08 | 591.90 | 591.39 | 0.38 | 0.03 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 172 → 173 → 173 |
| coxph_x3 | simulate_equal | 100 | 6.00 | 6.09 | 6.01 | 0.09 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 173 → 173 → 173 |
| coxph_x3 | simulate_equal | 1,000 | 59.53 | 58.73 | 58.88 | 0.17 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 173 → 173 → 173 |
| coxph_x3 | simulate_equal | 10,000 | 600.26 | 588.09 | 594.57 | 0.31 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 173 → 173 → 173 |
| ia_bs3x1_m8_n1000 | distribution | 100 | 0.15 | 0.03 | 0.10 | 0.05 | 0.02 | 0.02 (4/2) | 0.3 (64/32, 40 calls) | 109 → 109 → 109 |
| ia_bs3x1_m8_n1000 | distribution | 1,000 | 0.34 | 0.09 | 0.31 | 0.45 | 0.17 | 0.17 (4/2) | 2.7 (64/32, 40 calls) | 110 → 112 → 112 |
| ia_bs3x1_m8_n1000 | distribution | 10,000 | 6.33 | 0.65 | 1.90 | 3.36 | 1.68 | 1.68 (4/2) | 26.9 (64/32, 40 calls) | 123 → 143 → 143 |
| ia_bs3x1_m8_n1000 | distribution | 100,000 | 19.80 | 6.41 | 17.57 | 33.58 | 16.79 | 16.78 (4/2) | 201.4 (48/24, 12 calls) | 207 → 391 → 391 |
| ia_bs3x1_m8_n1000 | density | 100 | 0.26 | 0.03 | 0.11 | 0.05 | 0.02 | 0.02 (4/2) | 0.3 (64/32, 40 calls) | 391 → 391 → 391 |
| ia_bs3x1_m8_n1000 | density | 1,000 | 0.45 | 0.10 | 0.30 | 0.45 | 0.17 | 0.17 (4/2) | 2.7 (64/32, 40 calls) | 391 → 391 → 391 |
| ia_bs3x1_m8_n1000 | density | 10,000 | 2.14 | 0.61 | 2.00 | 3.36 | 1.68 | 1.68 (4/2) | 26.9 (64/32, 40 calls) | 391 → 392 → 392 |
| ia_bs3x1_m8_n1000 | density | 100,000 | 18.20 | 5.83 | 17.60 | 33.58 | 16.79 | 16.78 (4/2) | 201.4 (48/24, 12 calls) | 398 → 403 → 403 |
| ia_bs3x1_m8_n1000 | quantile | 100 | 2.12 | 1.65 | 1.63 | 0.42 | 0.38 | 0.34 (64/32) | 0.3 (64/32, 40 calls) | 403 → 403 → 403 |
| ia_bs3x1_m8_n1000 | quantile | 1,000 | 5.35 | 4.48 | 4.54 | 3.77 | 3.45 | 3.42 (64/32) | 3.4 (64/32, 40 calls) | 404 → 404 → 404 |
| ia_bs3x1_m8_n1000 | quantile | 10,000 | 33.13 | 32.79 | 32.70 | 36.80 | 34.21 | 34.18 (64/32) | 34.2 (64/32, 40 calls) | 405 → 405 → 405 |
| ia_bs3x1_m8_n1000 | quantile | 100,000 | 321.22 | 312.51 | 316.46 | 366.48 | 341.83 | 341.80 (64/32) | 341.8 (64/32, 12 calls) | 439 → 544 → 436 |
| ia_bs3x1_m8_n1000 | simulate | 100 | 1.77 | 1.52 | 1.65 | 0.42 | 0.38 | 0.34 (64/32) | 0.3 (64/32, 40 calls) | 436 → 436 → 436 |
| ia_bs3x1_m8_n1000 | simulate | 1,000 | 4.76 | 4.85 | 4.51 | 3.77 | 3.45 | 3.42 (64/32) | 3.4 (64/32, 40 calls) | 436 → 436 → 436 |
| ia_bs3x1_m8_n1000 | simulate | 10,000 | 34.07 | 33.43 | 32.85 | 36.80 | 34.22 | 34.18 (64/32) | 34.2 (64/32, 40 calls) | 436 → 436 → 436 |
| ia_bs3x1_m8_n1000 | simulate | 100,000 | 318.93 | 312.92 | 313.33 | 366.48 | 341.83 | 341.80 (64/32) | 341.8 (64/32, 12 calls) | 432 → 541 → 432 |
| ia_bs4x2_m15_n1000 | distribution | 100 | 0.23 | 0.04 | 0.12 | 0.08 | 0.02 | 0.02 (4/2) | 0.3 (64/32, 40 calls) | 103 → 103 → 103 |
| ia_bs4x2_m15_n1000 | distribution | 1,000 | 0.50 | 0.11 | 0.36 | 0.61 | 0.22 | 0.21 (4/2) | 3.4 (64/32, 40 calls) | 104 → 107 → 107 |
| ia_bs4x2_m15_n1000 | distribution | 10,000 | 3.18 | 0.87 | 2.92 | 4.94 | 2.14 | 2.14 (4/2) | 34.2 (64/32, 40 calls) | 122 → 147 → 147 |
| ia_bs4x2_m15_n1000 | distribution | 100,000 | 30.38 | 8.51 | 27.79 | 48.20 | 21.36 | 21.36 (4/2) | 256.3 (48/24, 12 calls) | 240 → 451 → 443 |
| ia_bs4x2_m15_n1000 | density | 100 | 0.26 | 0.04 | 0.12 | 0.08 | 0.02 | 0.02 (4/2) | 0.3 (64/32, 40 calls) | 443 → 443 → 443 |
| ia_bs4x2_m15_n1000 | density | 1,000 | 0.48 | 0.11 | 0.35 | 0.61 | 0.22 | 0.21 (4/2) | 3.4 (64/32, 40 calls) | 443 → 443 → 443 |
| ia_bs4x2_m15_n1000 | density | 10,000 | 2.77 | 0.80 | 3.22 | 4.94 | 2.14 | 2.14 (4/2) | 34.2 (64/32, 40 calls) | 447 → 449 → 449 |
| ia_bs4x2_m15_n1000 | density | 100,000 | 25.60 | 7.60 | 24.96 | 48.20 | 21.36 | 21.36 (4/2) | 256.3 (48/24, 12 calls) | 449 → 450 → 445 |
| ia_bs4x2_m15_n1000 | quantile | 100 | 2.48 | 1.79 | 1.72 | 0.50 | 0.45 | 0.42 (64/32) | 0.4 (64/32, 40 calls) | 446 → 446 → 446 |
| ia_bs4x2_m15_n1000 | quantile | 1,000 | 6.31 | 5.80 | 5.94 | 4.56 | 4.19 | 4.15 (64/32) | 4.2 (64/32, 40 calls) | 446 → 426 → 426 |
| ia_bs4x2_m15_n1000 | quantile | 10,000 | 43.87 | 44.17 | 44.47 | 44.51 | 41.54 | 41.50 (64/32) | 41.5 (64/32, 40 calls) | 351 → 201 → 201 |
| ia_bs4x2_m15_n1000 | quantile | 100,000 | 466.10 | 423.34 | 435.13 | 443.54 | 415.07 | 415.04 (64/32) | 415.0 (64/32, 12 calls) | 467 → 608 → 466 |
| ia_bs4x2_m15_n1000 | simulate | 100 | 1.92 | 1.72 | 1.75 | 0.50 | 0.45 | 0.42 (64/32) | 0.4 (64/32, 40 calls) | 466 → 466 → 466 |
| ia_bs4x2_m15_n1000 | simulate | 1,000 | 6.23 | 5.74 | 5.70 | 4.56 | 4.19 | 4.15 (64/32) | 4.2 (64/32, 40 calls) | 466 → 466 → 466 |
| ia_bs4x2_m15_n1000 | simulate | 10,000 | 43.68 | 44.03 | 44.00 | 44.51 | 41.54 | 41.50 (64/32) | 41.5 (64/32, 40 calls) | 466 → 292 → 292 |
| ia_bs4x2_m15_n1000 | simulate | 100,000 | 451.43 | 433.84 | 432.23 | 443.54 | 415.08 | 415.04 (64/32) | 415.0 (64/32, 12 calls) | 460 → 601 → 460 |
| ia_bs6x3_m28_n250 | distribution | 100 | 0.21 | 0.04 | 0.16 | 0.13 | 0.03 | 0.03 (4/2) | 0.5 (64/32, 40 calls) | 103 → 103 → 103 |
| ia_bs6x3_m28_n250 | distribution | 1,000 | 0.61 | 0.14 | 0.54 | 0.89 | 0.28 | 0.28 (4/2) | 4.5 (64/32, 40 calls) | 104 → 109 → 109 |
| ia_bs6x3_m28_n250 | distribution | 10,000 | 4.39 | 1.13 | 3.80 | 7.76 | 2.83 | 2.82 (4/2) | 45.2 (64/32, 40 calls) | 128 → 163 → 163 |
| ia_bs6x3_m28_n250 | distribution | 100,000 | 42.63 | 11.51 | 36.73 | 76.43 | 28.23 | 28.23 (4/2) | 338.7 (48/24, 12 calls) | 313 → 547 → 448 |
| ia_bs6x3_m28_n250 | density | 100 | 0.30 | 0.04 | 0.13 | 0.13 | 0.03 | 0.03 (4/2) | 0.5 (64/32, 40 calls) | 448 → 448 → 448 |
| ia_bs6x3_m28_n250 | density | 1,000 | 0.62 | 0.15 | 0.48 | 0.89 | 0.28 | 0.28 (4/2) | 4.5 (64/32, 40 calls) | 448 → 448 → 448 |
| ia_bs6x3_m28_n250 | density | 10,000 | 3.95 | 1.09 | 3.59 | 7.76 | 2.83 | 2.82 (4/2) | 45.2 (64/32, 40 calls) | 448 → 448 → 448 |
| ia_bs6x3_m28_n250 | density | 100,000 | 38.33 | 10.37 | 34.38 | 76.43 | 28.23 | 28.23 (4/2) | 338.7 (48/24, 12 calls) | 483 → 557 → 461 |
| ia_bs6x3_m28_n250 | quantile | 100 | 2.19 | 1.93 | 1.97 | 0.66 | 0.60 | 0.56 (64/32) | 0.6 (64/32, 40 calls) | 461 → 461 → 461 |
| ia_bs6x3_m28_n250 | quantile | 1,000 | 7.66 | 7.49 | 7.45 | 6.12 | 5.65 | 5.62 (64/32) | 5.6 (64/32, 40 calls) | 461 → 461 → 461 |
| ia_bs6x3_m28_n250 | quantile | 10,000 | 60.01 | 60.07 | 59.73 | 59.85 | 56.19 | 56.15 (64/32) | 56.2 (64/32, 40 calls) | 385 → 177 → 177 |
| ia_bs6x3_m28_n250 | quantile | 100,000 | 633.73 | 585.60 | 579.62 | 596.89 | 561.56 | 561.52 (64/32) | 561.5 (64/32, 12 calls) | 551 → 756 → 551 |
| ia_bs6x3_m28_n250 | simulate | 100 | 2.11 | 1.90 | 1.86 | 0.66 | 0.60 | 0.56 (64/32) | 0.6 (64/32, 40 calls) | 551 → 551 → 551 |
| ia_bs6x3_m28_n250 | simulate | 1,000 | 7.56 | 7.43 | 7.44 | 6.12 | 5.65 | 5.62 (64/32) | 5.6 (64/32, 40 calls) | 551 → 551 → 551 |
| ia_bs6x3_m28_n250 | simulate | 10,000 | 59.19 | 58.68 | 59.17 | 59.85 | 56.19 | 56.15 (64/32) | 56.2 (64/32, 40 calls) | 450 → 208 → 208 |
| ia_bs6x3_m28_n250 | simulate | 100,000 | 658.94 | 596.71 | 613.74 | 596.89 | 561.56 | 561.52 (64/32) | 561.5 (64/32, 12 calls) | 553 → 760 → 556 |
| ia_scaled_bs3x1_m9_n1000 | distribution | 100 | 0.22 | 0.04 | 0.12 | 0.05 | 0.02 | 0.02 (4/2) | 0.3 (64/32, 40 calls) | 102 → 102 → 102 |
| ia_scaled_bs3x1_m9_n1000 | distribution | 1,000 | 0.40 | 0.10 | 0.32 | 0.45 | 0.17 | 0.17 (4/2) | 2.7 (64/32, 40 calls) | 103 → 105 → 105 |
| ia_scaled_bs3x1_m9_n1000 | distribution | 10,000 | 2.56 | 0.75 | 2.18 | 3.36 | 1.68 | 1.68 (4/2) | 26.9 (64/32, 40 calls) | 117 → 137 → 137 |
| ia_scaled_bs3x1_m9_n1000 | distribution | 100,000 | 21.71 | 7.82 | 19.80 | 33.58 | 16.79 | 16.78 (4/2) | 201.4 (48/24, 12 calls) | 205 → 387 → 387 |
| ia_scaled_bs3x1_m9_n1000 | density | 100 | 0.22 | 0.04 | 0.11 | 0.05 | 0.02 | 0.02 (4/2) | 0.3 (64/32, 40 calls) | 387 → 387 → 387 |
| ia_scaled_bs3x1_m9_n1000 | density | 1,000 | 0.40 | 0.11 | 0.31 | 0.45 | 0.17 | 0.17 (4/2) | 2.7 (64/32, 40 calls) | 387 → 387 → 387 |
| ia_scaled_bs3x1_m9_n1000 | density | 10,000 | 2.01 | 0.71 | 1.97 | 3.36 | 1.68 | 1.68 (4/2) | 26.9 (64/32, 40 calls) | 388 → 388 → 388 |
| ia_scaled_bs3x1_m9_n1000 | density | 100,000 | 19.47 | 6.66 | 18.28 | 33.58 | 16.79 | 16.78 (4/2) | 201.4 (48/24, 12 calls) | 390 → 395 → 395 |
| ia_scaled_bs3x1_m9_n1000 | quantile | 100 | 1.80 | 1.71 | 1.70 | 0.42 | 0.37 | 0.34 (64/32) | 0.3 (64/32, 40 calls) | 395 → 395 → 395 |
| ia_scaled_bs3x1_m9_n1000 | quantile | 1,000 | 5.18 | 5.02 | 4.87 | 3.77 | 3.45 | 3.42 (64/32) | 3.4 (64/32, 40 calls) | 397 → 397 → 397 |
| ia_scaled_bs3x1_m9_n1000 | quantile | 10,000 | 35.94 | 35.79 | 36.11 | 36.80 | 34.21 | 34.18 (64/32) | 34.2 (64/32, 40 calls) | 397 → 317 → 317 |
| ia_scaled_bs3x1_m9_n1000 | quantile | 100,000 | 357.93 | 344.61 | 347.37 | 366.48 | 341.83 | 341.80 (64/32) | 341.8 (64/32, 12 calls) | 435 → 540 → 433 |
| ia_scaled_bs3x1_m9_n1000 | simulate | 100 | 1.91 | 1.78 | 1.71 | 0.42 | 0.37 | 0.34 (64/32) | 0.3 (64/32, 40 calls) | 433 → 433 → 433 |
| ia_scaled_bs3x1_m9_n1000 | simulate | 1,000 | 5.12 | 4.85 | 4.93 | 3.77 | 3.45 | 3.42 (64/32) | 3.4 (64/32, 40 calls) | 433 → 433 → 433 |
| ia_scaled_bs3x1_m9_n1000 | simulate | 10,000 | 35.85 | 36.36 | 35.84 | 36.80 | 34.21 | 34.18 (64/32) | 34.2 (64/32, 40 calls) | 365 → 301 → 301 |
| ia_scaled_bs3x1_m9_n1000 | simulate | 100,000 | 359.26 | 341.26 | 348.73 | 366.47 | 341.83 | 341.80 (64/32) | 341.8 (64/32, 12 calls) | 434 → 543 → 434 |
| right_baseline | distribution | 100 | 0.13 | 0.02 | 0.07 | 0.05 | 0.02 | 0.02 (2/1) | 0.6 (64/32, 40 calls) | 108 → 108 → 108 |
| right_baseline | distribution | 1,000 | 0.31 | 0.06 | 0.23 | 0.41 | 0.18 | 0.18 (2/1) | 5.6 (64/32, 40 calls) | 108 → 108 → 108 |
| right_baseline | distribution | 10,000 | 1.92 | 0.33 | 1.89 | 3.51 | 1.76 | 1.75 (2/1) | 56.2 (64/32, 40 calls) | 114 → 165 → 165 |
| right_baseline | distribution | 100,000 | 19.66 | 3.21 | 18.19 | 35.10 | 17.55 | 17.55 (2/1) | 210.6 (24/12, 12 calls) | 234 → 379 → 379 |
| right_baseline | quantile | 100 | 6.48 | 6.03 | 6.03 | 0.09 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 379 → 379 → 379 |
| right_baseline | quantile | 1,000 | 59.62 | 58.60 | 58.28 | 0.17 | 0.03 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 379 → 379 → 379 |
| right_baseline | quantile | 10,000 | 596.33 | 593.86 | 592.30 | 0.31 | 0.03 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 379 → 240 → 240 |
| right_baseline | simulate | 100 | 5.99 | 5.98 | 6.01 | 0.09 | 0.02 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 240 → 240 → 240 |
| right_baseline | simulate | 1,000 | 60.37 | 59.96 | 60.06 | 0.17 | 0.03 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 240 → 240 → 240 |
| right_baseline | simulate | 10,000 | 627.97 | 611.40 | 611.27 | 0.31 | 0.03 | 0.00 (0/0) | 0.0 (0/0, 40 calls) | 240 → 171 → 171 |
| survreg_x3 | distribution | 100 | 0.18 | 0.03 | 0.08 | 0.05 | 0.02 | 0.02 (2/1) | 0.6 (64/32, 40 calls) | 108 → 109 → 109 |
| survreg_x3 | distribution | 1,000 | 0.32 | 0.08 | 0.26 | 0.42 | 0.18 | 0.18 (2/1) | 5.6 (64/32, 40 calls) | 109 → 109 → 109 |
| survreg_x3 | distribution | 10,000 | 2.11 | 0.41 | 2.00 | 3.59 | 1.76 | 1.75 (2/1) | 56.2 (64/32, 40 calls) | 116 → 168 → 168 |
| survreg_x3 | distribution | 100,000 | 20.17 | 4.03 | 19.32 | 35.86 | 17.55 | 17.55 (2/1) | 210.6 (24/12, 12 calls) | 246 → 391 → 391 |
| survreg_x3 | quantile_varying | 100 | 6.43 | 6.19 | 6.08 | 0.08 | 0.01 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 391 → 391 → 391 |
| survreg_x3 | quantile_varying | 1,000 | 61.07 | 60.84 | 60.80 | 0.15 | 0.01 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 391 → 391 → 391 |
| survreg_x3 | quantile_varying | 10,000 | 602.68 | 602.69 | 606.39 | 0.29 | 0.01 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 391 → 191 → 191 |
| survreg_x3 | quantile_equal | 100 | 6.99 | 6.49 | 6.42 | 0.08 | 0.01 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 191 → 191 → 191 |
| survreg_x3 | quantile_equal | 1,000 | 60.80 | 60.42 | 59.92 | 0.16 | 0.02 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 191 → 191 → 191 |
| survreg_x3 | quantile_equal | 10,000 | 610.18 | 632.61 | 622.08 | 0.31 | 0.04 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 95 → 95 → 95 |
| survreg_x3 | quantile_10prof | 100 | 7.34 | 7.02 | 7.34 | 0.09 | 0.03 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 95 → 95 → 95 |
| survreg_x3 | quantile_10prof | 1,000 | 67.92 | 65.79 | 67.14 | 0.17 | 0.03 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 95 → 95 → 95 |
| survreg_x3 | quantile_10prof | 10,000 | 652.83 | 651.12 | 662.98 | 0.30 | 0.02 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 90 → 90 → 90 |
| survreg_x3 | quantile_equal_offset | 100 | 7.03 | 6.81 | 6.96 | 0.09 | 0.03 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 90 → 90 → 90 |
| survreg_x3 | quantile_equal_offset | 1,000 | 68.16 | 66.65 | 67.30 | 0.16 | 0.01 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 90 → 90 → 90 |
| survreg_x3 | quantile_equal_offset | 10,000 | 664.34 | 665.46 | 665.85 | 0.37 | 0.01 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 91 → 92 → 92 |
| survreg_x3 | simulate_equal | 100 | 6.86 | 6.77 | 6.60 | 0.09 | 0.02 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 92 → 92 → 92 |
| survreg_x3 | simulate_equal | 1,000 | 66.19 | 65.89 | 66.06 | 0.16 | 0.02 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 92 → 92 → 92 |
| survreg_x3 | simulate_equal | 10,000 | 655.48 | 659.02 | 659.37 | 0.30 | 0.02 | 0.01 (2/1) | 0.0 (2/1, 40 calls) | 92 → 91 → 91 |

Max |model-instance Δ| over all cells: 0 MiB.


## C. Cache traffic per call (m = 1,000; instrumented run)

Counts for the first call on new data / a repeated call on the same data. `dB on value-only` = assembled-cache misses inside `evaluate()` (a derivative matrix built and cached although the caller only needs values).

| case | op | raw hit/miss | assembled hit/miss | derivative() calls | dB on value-only | key bytes hashed |
|---|---|---|---|---|---|---|
| boxcox_scaled | distribution | 1/2 → 1/0 | 0/1 → 1/0 | 1 | 1 → 0 | 32,000 → 16,000 |
| boxcox_scaled | quantile | 0/0 → 0/0 | 0/0 → 0/0 | 0 | 0 → 0 | 0 → 0 |
| boxcox_x3 | distribution | 1/2 → 1/0 | 0/1 → 1/0 | 1 | 1 → 0 | 32,000 → 16,000 |
| boxcox_x3 | density | 1/2 → 1/0 | 0/1 → 1/0 | 1 | 1 → 0 | 32,000 → 16,000 |
| boxcox_x3 | quantile | 0/0 → 0/0 | 0/0 → 0/0 | 0 | 0 → 0 | 0 → 0 |
| boxcox_x3 | simulate | 0/0 → 0/0 | 0/0 → 0/0 | 0 | 0 → 0 | 0 → 0 |
| coxph_x3 | distribution | 1/2 → 1/0 | 0/1 → 1/0 | 1 | 1 → 0 | 32,000 → 16,000 |
| coxph_x3 | quantile_varying | 0/0 → 0/0 | 0/0 → 0/0 | 0 | 0 → 0 | 0 → 0 |
| coxph_x3 | quantile_equal | 0/0 → 0/0 | 0/0 → 0/0 | 0 | 0 → 0 | 0 → 0 |
| coxph_x3 | quantile_10prof | 0/0 → 0/0 | 0/0 → 0/0 | 0 | 0 → 0 | 0 → 0 |
| coxph_x3 | quantile_equal_offset | 0/0 → 0/0 | 0/0 → 0/0 | 0 | 0 → 0 | 0 → 0 |
| coxph_x3 | simulate_equal | 0/0 → 0/0 | 0/0 → 0/0 | 0 | 0 → 0 | 0 → 0 |
| ia_bs3x1_m8_n1000 | distribution | 1/4 → 1/0 | 1/2 → 3/0 | 1 | 2 → 0 | 64,000 → 32,000 |
| ia_bs3x1_m8_n1000 | density | 1/4 → 1/0 | 1/2 → 3/0 | 1 | 2 → 0 | 64,000 → 32,000 |
| ia_bs3x1_m8_n1000 | quantile | 0/80 → 0/80 | 0/40 → 0/40 | 0 | 40 → 40 | 912,048 → 912,048 |
| ia_bs3x1_m8_n1000 | simulate | 0/80 → 0/80 | 0/40 → 0/40 | 0 | 40 → 40 | 912,048 → 912,048 |
| ia_bs4x2_m15_n1000 | distribution | 1/4 → 1/0 | 1/2 → 3/0 | 1 | 2 → 0 | 64,000 → 32,000 |
| ia_bs4x2_m15_n1000 | density | 1/4 → 1/0 | 1/2 → 3/0 | 1 | 2 → 0 | 64,000 → 32,000 |
| ia_bs4x2_m15_n1000 | quantile | 0/80 → 0/80 | 0/40 → 0/40 | 0 | 40 → 40 | 912,048 → 912,048 |
| ia_bs4x2_m15_n1000 | simulate | 0/80 → 0/80 | 0/40 → 0/40 | 0 | 40 → 40 | 912,048 → 912,048 |
| ia_bs6x3_m28_n250 | distribution | 1/4 → 1/0 | 1/2 → 3/0 | 1 | 2 → 0 | 64,000 → 32,000 |
| ia_bs6x3_m28_n250 | density | 1/4 → 1/0 | 1/2 → 3/0 | 1 | 2 → 0 | 64,000 → 32,000 |
| ia_bs6x3_m28_n250 | quantile | 0/80 → 0/80 | 0/40 → 0/40 | 0 | 40 → 40 | 912,048 → 912,048 |
| ia_bs6x3_m28_n250 | simulate | 0/80 → 0/80 | 0/40 → 0/40 | 0 | 40 → 40 | 912,048 → 912,048 |
| ia_scaled_bs3x1_m9_n1000 | distribution | 1/4 → 1/0 | 1/2 → 3/0 | 1 | 2 → 0 | 64,000 → 32,000 |
| ia_scaled_bs3x1_m9_n1000 | density | 1/4 → 1/0 | 1/2 → 3/0 | 1 | 2 → 0 | 64,000 → 32,000 |
| ia_scaled_bs3x1_m9_n1000 | quantile | 0/80 → 0/80 | 0/40 → 0/40 | 0 | 40 → 40 | 912,048 → 912,048 |
| ia_scaled_bs3x1_m9_n1000 | simulate | 0/80 → 0/80 | 0/40 → 0/40 | 0 | 40 → 40 | 912,048 → 912,048 |
| right_baseline | distribution | 1/2 → 1/0 | 0/1 → 1/0 | 1 | 1 → 0 | 32,000 → 16,000 |
| right_baseline | quantile | 0/0 → 0/0 | 0/0 → 0/0 | 0 | 0 → 0 | 0 → 0 |
| right_baseline | simulate | 0/0 → 0/0 | 0/0 → 0/0 | 0 | 0 → 0 | 0 → 0 |
| survreg_x3 | distribution | 1/2 → 1/0 | 0/1 → 1/0 | 1 | 1 → 0 | 32,000 → 16,000 |
| survreg_x3 | quantile_varying | 0/2 → 0/0 | 0/1 → 1/0 | 0 | 1 → 0 | 1,200 → 400 |
| survreg_x3 | quantile_equal | 0/2 → 0/0 | 0/1 → 1/0 | 0 | 1 → 0 | 1,200 → 400 |
| survreg_x3 | quantile_10prof | 0/2 → 0/0 | 0/1 → 1/0 | 0 | 1 → 0 | 1,200 → 400 |
| survreg_x3 | quantile_equal_offset | 0/2 → 0/0 | 0/1 → 1/0 | 0 | 1 → 0 | 1,200 → 400 |
| survreg_x3 | simulate_equal | 0/2 → 0/0 | 0/1 → 1/0 | 0 | 1 → 0 | 1,200 → 400 |

Content-key cost (`ascontiguousarray(y).tobytes()` + hash, fresh array): m = 100: 0.3 µs, m = 1,000: 1.5 µs, m = 10,000: 15.0 µs, m = 100,000: 150.9 µs.


## D. Fit-entry eviction by prediction

`score()` on the training data right after `fit()` vs after 40 changing-data CDF calls (m = 1,000) plus 40 quantile calls (m = 100).

| case | fit entries (raw/asm) | surviving | score hot (ms) | score after burst (ms) |
|---|---|---|---|---|
| boxcox_scaled | 2/1 | 0/0 | 0.027 | 0.228 |
| boxcox_x3 | 2/1 | 0/0 | 0.032 | 0.227 |
| coxph_x3 | 4/2 | 0/0 | 0.059 | 0.326 |
| ia_bs3x1_m8_n1000 | 4/2 | 0/0 | 0.064 | 0.228 |
| ia_bs4x2_m15_n1000 | 4/2 | 0/0 | 0.080 | 0.381 |
| ia_bs6x3_m28_n250 | 4/2 | 0/0 | 0.038 | 0.186 |
| ia_scaled_bs3x1_m9_n1000 | 4/2 | 0/0 | 0.066 | 0.260 |
| right_baseline | 4/2 | 0/0 | 0.043 | 0.325 |
| survreg_x3 | 4/2 | 0/0 | 0.073 | 0.370 |

## E. Candidate prototype: interaction value-only midpoint evaluation

`_predict_quantile_interaction` with `y_basis.evaluate` replaced (harness only) by (i) the library call, (ii) values only, uncached, (iii) values + the same dB work, uncached — separating derivative work from cache insertion. Retained/peak traced MiB per call; ✓ = bitwise equal to the library output.

| case | m | library ms | value-only ms | +dB uncached ms | speedup | retained lib → value-only | peak lib → value-only | equal |
|---|---|---|---|---|---|---|---|---|
| ia_bs3x1_m8_n1000 | 100 | 1.62 | 0.59 | 1.16 | 2.75× | 0.38 → 0.01 | 0.42 → 0.04 | ✓ |
| ia_bs3x1_m8_n1000 | 1,000 | 4.53 | 2.07 | 3.89 | 2.18× | 3.46 → 0.07 | 3.77 → 0.35 | ✓ |
| ia_bs3x1_m8_n1000 | 10,000 | 33.38 | 16.13 | 30.87 | 2.07× | 34.29 → 0.69 | 36.80 → 2.97 | ✓ |
| ia_bs3x1_m8_n1000 | 100,000 | 325.05 | 156.54 | 297.69 | 2.08× | 342.59 → 6.87 | 366.48 → 28.46 | ✓ |
| ia_bs4x2_m15_n1000 | 100 | 1.93 | 0.72 | 1.42 | 2.67× | 0.45 → 0.01 | 0.50 → 0.05 | ✓ |
| ia_bs4x2_m15_n1000 | 1,000 | 5.83 | 2.64 | 5.19 | 2.21× | 4.19 → 0.09 | 4.55 → 0.42 | ✓ |
| ia_bs4x2_m15_n1000 | 10,000 | 43.83 | 21.16 | 41.17 | 2.07× | 41.61 → 0.92 | 44.51 → 3.50 | ✓ |
| ia_bs4x2_m15_n1000 | 100,000 | 440.33 | 210.43 | 405.29 | 2.09× | 415.84 → 9.16 | 443.54 → 33.80 | ✓ |
| ia_bs6x3_m28_n250 | 100 | 2.24 | 0.83 | 1.57 | 2.71× | 0.60 → 0.01 | 0.65 → 0.06 | ✓ |
| ia_bs6x3_m28_n250 | 1,000 | 7.57 | 3.46 | 6.70 | 2.19× | 5.66 → 0.12 | 6.12 → 0.54 | ✓ |
| ia_bs6x3_m28_n250 | 10,000 | 59.36 | 28.90 | 57.14 | 2.05× | 56.26 → 1.15 | 59.84 → 4.34 | ✓ |
| ia_bs6x3_m28_n250 | 100,000 | 601.91 | 289.12 | 576.59 | 2.08× | 562.32 → 11.45 | 596.89 → 42.19 | ✓ |
| ia_scaled_bs3x1_m9_n1000 | 100 | 1.80 | 0.71 | 1.42 | 2.54× | 0.37 → 0.01 | 0.42 → 0.04 | ✓ |
| ia_scaled_bs3x1_m9_n1000 | 1,000 | 5.06 | 2.31 | 4.28 | 2.19× | 3.46 → 0.07 | 3.77 → 0.35 | ✓ |
| ia_scaled_bs3x1_m9_n1000 | 10,000 | 35.83 | 17.29 | 33.39 | 2.07× | 34.29 → 0.69 | 36.80 → 2.97 | ✓ |
| ia_scaled_bs3x1_m9_n1000 | 100,000 | 351.94 | 171.90 | 323.59 | 2.05× | 342.59 → 6.87 | 366.48 → 28.46 | ✓ |

## F. Candidate prototype: call-local reuse of the R-style inverse curve

Copy of the R-style loop (`none` must equal the library bitwise — it does), with the per-shift curve (CDF grid → spline → sorted unique inverse) reused for consecutive equal shifts or for any shift already seen in this call. The interp/saturation step still runs per probability.

| case | op | m | distinct shifts | none ms | consecutive ms (curves) | unique ms (curves) | equal | saturation flag |
|---|---|---|---|---|---|---|---|---|
| coxph_x3 | quantile_varying | 100 | 100 | 6.02 | 6.09 (100) | 5.98 (100) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_varying | 1,000 | 1,000 | 60.20 | 60.31 (1,000) | 61.36 (1,000) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_varying | 10,000 | 10,000 | 605.17 | 603.43 (10,000) | 616.47 (10,000) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_equal | 100 | 1 | 6.01 | 0.12 (1) | 0.12 (1) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_equal | 1,000 | 1 | 60.64 | 0.63 (1) | 0.66 (1) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_equal | 10,000 | 1 | 603.12 | 5.72 (1) | 5.85 (1) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_10prof | 100 | 10 | 6.18 | 5.82 (95) | 0.70 (10) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_10prof | 1,000 | 10 | 60.49 | 54.87 (901) | 1.21 (10) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_10prof | 10,000 | 10 | 604.73 | 554.47 (9,032) | 6.64 (10) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_equal_offset | 100 | 100 | 6.16 | 6.07 (100) | 6.11 (100) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_equal_offset | 1,000 | 1,000 | 60.38 | 60.74 (1,000) | 61.53 (1,000) | ✓ | same (lib warned: False) |
| coxph_x3 | quantile_equal_offset | 10,000 | 10,000 | 599.75 | 601.75 (10,000) | 615.25 (10,000) | ✓ | same (lib warned: False) |
| coxph_x3 | simulate_equal | 100 | 1 | 6.02 | 0.12 (1) | 0.12 (1) | ✓ | same (lib warned: False) |
| coxph_x3 | simulate_equal | 1,000 | 1 | 60.53 | 0.62 (1) | 0.65 (1) | ✓ | same (lib warned: False) |
| coxph_x3 | simulate_equal | 10,000 | 1 | 603.25 | 5.62 (1) | 5.83 (1) | ✓ | same (lib warned: True) |
| right_baseline | quantile | 100 | 1 | 5.96 | 0.13 (1) | 0.12 (1) | ✓ | same (lib warned: False) |
| right_baseline | quantile | 1,000 | 1 | 60.00 | 0.64 (1) | 0.66 (1) | ✓ | same (lib warned: False) |
| right_baseline | quantile | 10,000 | 1 | 597.65 | 5.69 (1) | 5.81 (1) | ✓ | same (lib warned: False) |
| right_baseline | simulate | 100 | 1 | 5.93 | 0.12 (1) | 0.12 (1) | ✓ | same (lib warned: False) |
| right_baseline | simulate | 1,000 | 1 | 58.95 | 0.62 (1) | 0.65 (1) | ✓ | same (lib warned: False) |
| right_baseline | simulate | 10,000 | 1 | 592.33 | 5.63 (1) | 5.89 (1) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_varying | 100 | 100 | 6.48 | 6.48 (100) | 6.52 (100) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_varying | 1,000 | 1,000 | 63.97 | 64.66 (1,000) | 65.56 (1,000) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_varying | 10,000 | 10,000 | 659.33 | 662.48 (10,000) | 670.99 (10,000) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_equal | 100 | 1 | 7.42 | 0.13 (1) | 0.13 (1) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_equal | 1,000 | 1 | 65.62 | 0.64 (1) | 0.69 (1) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_equal | 10,000 | 1 | 647.83 | 5.88 (1) | 6.15 (1) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_10prof | 100 | 10 | 6.42 | 6.20 (95) | 0.68 (10) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_10prof | 1,000 | 10 | 63.98 | 59.50 (901) | 1.34 (10) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_10prof | 10,000 | 10 | 651.20 | 597.79 (9,032) | 6.87 (10) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_equal_offset | 100 | 100 | 6.66 | 6.54 (100) | 6.53 (100) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_equal_offset | 1,000 | 1,000 | 65.24 | 64.75 (1,000) | 65.75 (1,000) | ✓ | same (lib warned: False) |
| survreg_x3 | quantile_equal_offset | 10,000 | 10,000 | 650.84 | 654.13 (10,000) | 668.22 (10,000) | ✓ | same (lib warned: False) |
| survreg_x3 | simulate_equal | 100 | 1 | 6.77 | 0.16 (1) | 0.13 (1) | ✓ | same (lib warned: False) |
| survreg_x3 | simulate_equal | 1,000 | 1 | 65.16 | 0.67 (1) | 0.70 (1) | ✓ | same (lib warned: False) |
| survreg_x3 | simulate_equal | 10,000 | 1 | 647.80 | 6.07 (1) | 6.19 (1) | ✓ | same (lib warned: True) |

## G. Shift CDF: unused derivative work

| case | m | predict (new data) ms | derivative() share ms | share |
|---|---|---|---|---|
| boxcox_scaled | 100 | 0.22 | 0.02 | 9% |
| boxcox_scaled | 1,000 | 0.38 | 0.04 | 10% |
| boxcox_scaled | 10,000 | 2.08 | 0.21 | 10% |
| boxcox_scaled | 100,000 | 19.41 | 2.00 | 10% |
| boxcox_x3 | 100 | 0.13 | 0.02 | 13% |
| boxcox_x3 | 1,000 | 0.32 | 0.03 | 10% |
| boxcox_x3 | 10,000 | 1.83 | 0.19 | 10% |
| boxcox_x3 | 100,000 | 17.55 | 1.77 | 10% |
| coxph_x3 | 100 | 0.14 | 0.02 | 13% |
| coxph_x3 | 1,000 | 0.31 | 0.03 | 11% |
| coxph_x3 | 10,000 | 1.86 | 0.19 | 10% |
| coxph_x3 | 100,000 | 17.89 | 1.79 | 10% |
| right_baseline | 100 | 0.13 | 0.02 | 14% |
| right_baseline | 1,000 | 0.27 | 0.03 | 13% |
| right_baseline | 10,000 | 1.88 | 0.19 | 10% |
| right_baseline | 100,000 | 17.77 | 1.80 | 10% |
| survreg_x3 | 100 | 0.22 | 0.03 | 12% |
| survreg_x3 | 1,000 | 0.40 | 0.05 | 12% |
| survreg_x3 | 10,000 | 2.08 | 0.26 | 12% |
| survreg_x3 | 100,000 | 19.40 | 2.51 | 13% |

## H. Quantile profile (m = 1,000, top self-time)

- **boxcox_scaled**: `_h_vec` 83%, `_predict_quantile_scaling` 7%, `ppf` 1%, `<method 'reduce' of 'numpy.ufunc' objects>` 1%, `predict` 1%
- **boxcox_x3**: `_h_vec` 84%, `_predict_quantile` 7%, `<method 'reduce' of 'numpy.ufunc' objects>` 1%, `ppf` 1%, `predict` 1%
- **coxph_x3**: `_predict_quantile` 10%, `prepare_input` 6%, `diff` 6%, `__init__` 6%, `<method 'reduce' of 'numpy.ufunc' objects>` 4%
- **ia_bs3x1_m8_n1000**: `_bernstein_matrix` 52%, `evaluate_with_derivative` 8%, `_predict_quantile_interaction` 4%, `comb` 4%, `<method 'get' of 'dict' objects>` 3%
- **ia_bs4x2_m15_n1000**: `_bernstein_matrix` 57%, `evaluate_with_derivative` 8%, `_predict_quantile_interaction` 4%, `comb` 3%, `_pad_simple` 3%
- **ia_bs6x3_m28_n250**: `_bernstein_matrix` 63%, `evaluate_with_derivative` 7%, `_predict_quantile_interaction` 4%, `comb` 3%, `_pad_simple` 2%
- **ia_scaled_bs3x1_m9_n1000**: `_bernstein_matrix` 52%, `evaluate_with_derivative` 8%, `_predict_quantile_interaction` 4%, `comb` 4%, `<method 'get' of 'dict' objects>` 3%
- **right_baseline**: `_predict_quantile` 10%, `prepare_input` 7%, `diff` 6%, `__init__` 6%, `<method 'reduce' of 'numpy.ufunc' objects>` 4%
- **survreg_x3**: `_predict_quantile` 10%, `prepare_input` 6%, `diff` 6%, `__init__` 6%, `<method 'reduce' of 'numpy.ufunc' objects>` 5%

## Environment

python=3.12.16, numpy=2.4.6, scipy=1.17.1, machine=arm64, platform=macOS-26.6.2-arm64-arm-64bit, blas=accelerate, git_sha=b9069681742c462f448500a7b287668a2b5a0413, omp_threads=None, veclib_threads=None

Per-process RSS (start → end, ru_maxrss): boxcox_scaled 100 → 206 (380); boxcox_x3 100 → 391 (393); coxph_x3 100 → 235 (388); ia_bs3x1_m8_n1000 107 → 433 (582); ia_bs4x2_m15_n1000 101 → 334 (625); ia_bs6x3_m28_n250 100 → 320 (760); ia_scaled_bs3x1_m9_n1000 100 → 435 (544); right_baseline 100 → 203 (379); survreg_x3 100 → 168 (391)
