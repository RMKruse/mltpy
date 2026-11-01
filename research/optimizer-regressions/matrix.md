# Per-case reproduction matrix

Each cell: raw convergence flag (T/F), NLL, actual attempts. `0 / 3` is the configured maximum restarts. Full parameters, row encodings, warning locations/counts and inner solver evaluation counts are in [results.json](results.json). This is descriptive evidence, not an acceptance threshold.

| Case | Restarts | auglag | SLSQP | trust-constr |
|---|---:|---|---|---|
| 161_exponential_literal | 0 | T; -1.938360; 1 | T; -1.938360; 1 | T; -1.938360; 1 |
| 161_exponential_literal | 3 | T; -1.938360; 1 | T; -1.938360; 1 | T; -1.938360; 1 |
| 161_exponential_corrected | 0 | T; 51.140931; 1 | T; 51.140931; 1 | T; 51.140931; 1 |
| 161_exponential_corrected | 3 | T; 51.140931; 1 | T; 51.140931; 1 | T; 51.140931; 1 |
| 162_normal_literal | 0 | F; -246.707729; 1 | T; -411.532274; 1 | T; -183.664792; 1 |
| 162_normal_literal | 3 | F; -246.707729; 4 | T; -411.532274; 1 | T; -183.664792; 1 |
| 162_normal_corrected | 0 | T; 63.415358; 1 | T; 63.415358; 1 | T; 63.415358; 1 |
| 162_normal_corrected | 3 | T; 63.415358; 1 | T; 63.415358; 1 | T; 63.415358; 1 |
| 162_logistic_literal | 0 | F; -254.292010; 1 | F; -2585.257376; 1 | F; -207.903243; 1 |
| 162_logistic_literal | 3 | F; -254.292010; 4 | F; -2585.257376; 4 | F; -212.624393; 4 |
| 162_logistic_corrected | 0 | T; 65.813506; 1 | T; 65.813506; 1 | T; 65.813506; 1 |
| 162_logistic_corrected | 3 | T; 65.813506; 1 | T; 65.813506; 1 | T; 65.813506; 1 |
| 163_NONE | 0 | T; 71.449354; 1 | T; 71.449354; 1 | T; 71.449354; 1 |
| 163_NONE | 3 | T; 71.449354; 1 | T; 71.449354; 1 | T; 71.449354; 1 |
| 163_LEFT | 0 | T; 128.711946; 1 | T; 128.711946; 1 | T; 128.711946; 1 |
| 163_LEFT | 3 | T; 128.711946; 1 | T; 128.711946; 1 | T; 128.711946; 1 |
| 163_INTERVAL | 0 | T; 234.712354; 1 | T; 234.712354; 1 | T; 234.712354; 1 |
| 163_INTERVAL | 3 | T; 234.712354; 1 | T; 234.712354; 1 | T; 234.712354; 1 |
| 160_offset_-0.5 | 0 | T; 5.132565; 1 | F; 5.132565; 1 | T; 5.132566; 1 |
| 160_offset_-0.5 | 3 | T; 5.132565; 1 | F; 5.132565; 4 | T; 5.132566; 1 |
| 160_offset_+0.5 | 0 | T; 5.132565; 1 | F; 5.132563; 1 | T; 5.132566; 1 |
| 160_offset_+0.5 | 3 | T; 5.132565; 1 | F; 5.132557; 4 | T; 5.132566; 1 |
| 165_unscaled | 0 | T; 150.507881; 1 | T; 150.507881; 1 | T; 150.507881; 1 |
| 165_unscaled | 3 | T; 150.507881; 1 | T; 150.507881; 1 | T; 150.507881; 1 |
| 165_scaled | 0 | T; 144.805007; 1 | T; 144.805007; 1 | T; 144.805007; 1 |
| 165_scaled | 3 | T; 144.805007; 1 | T; 144.805007; 1 | T; 144.805007; 1 |
| coverage_interval_shift_False_offset_-0.2 | 0 | F; 252.216073; 1 | T; 241.423059; 1 | T; 241.423067; 1 |
| coverage_interval_shift_False_offset_-0.2 | 3 | F; 237.285199; 4 | T; 241.423059; 1 | T; 241.423067; 1 |
| coverage_interval_shift_False_offset_+0.2 | 0 | F; 241.398366; 1 | T; 241.423059; 1 | F; 299.316927; 1 |
| coverage_interval_shift_False_offset_+0.2 | 3 | F; 241.398295; 4 | T; 241.423059; 1 | T; 241.423066; 2 |
| coverage_interval_shift_True_offset_-0.2 | 0 | F; 252.216073; 1 | T; 241.423059; 1 | F; 339.873182; 1 |
| coverage_interval_shift_True_offset_-0.2 | 3 | F; 240.241988; 4 | T; 241.423059; 1 | T; 241.423096; 2 |
| coverage_interval_shift_True_offset_+0.2 | 0 | F; 241.423250; 1 | T; 241.423059; 1 | T; 241.423066; 1 |
| coverage_interval_shift_True_offset_+0.2 | 3 | F; 239.903820; 4 | T; 241.423059; 1 | T; 241.423066; 1 |
