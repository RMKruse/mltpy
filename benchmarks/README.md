# Benchmarks

Reproducible runtime comparison of `pymlt.MLT.fit()` against the corresponding
R `mlt::mlt()` call across the grid

```
n         ∈ {100, 500, 1000, 5000}
order     ∈ {4, 6, 8}
censoring ∈ {NONE, RIGHT}
```

with 10 repetitions per cell.

The headline numbers and the full grid live in
[`results/benchmark_report.md`](results/benchmark_report.md). The README's
`## Performance` section pulls a representative slice from there.

## Layout

```
benchmarks/
  bench_python.py   driver — generates input data, times pymlt.MLT.fit()
  bench_r.R         driver — reads the same data, times R mlt::mlt()
  report.py         aggregates both CSVs into benchmark_report.md
  data/             generated input CSVs (gitignored — regenerated each run)
  results/          committed outputs
    python_results.csv
    r_results.csv
    benchmark_report.md
```

## Running

From the repo root:

```bash
make benchmark         # bench_python.py → bench_r.R → report.py
```

or each step individually:

```bash
python benchmarks/bench_python.py    # writes data/ + results/python_results.csv
Rscript benchmarks/bench_r.R         # reads data/, writes results/r_results.csv
python benchmarks/report.py          # reads both CSVs, writes benchmark_report.md
```

## Prerequisites

- Python with `pymlt` installed in development mode (`pip install -e .`)
- R ≥ 4.0 with `mlt`, `basefun`, `variables`, `survival` packages

`bench_python.py` is the source of truth for the input datasets — `bench_r.R`
reads them from `data/` and will fail with a clear message if they're missing.

## Output schema

Both result CSVs share the same schema, one row per repetition:

| column      | type | meaning                                      |
| ----------- | ---- | -------------------------------------------- |
| `n`         | int  | sample size                                  |
| `order`     | int  | Bernstein polynomial degree                  |
| `censoring` | str  | `NONE` or `RIGHT`                            |
| `rep`       | int  | repetition index (0-based)                   |
| `time_s`    | float| wall time of one `fit()` call in seconds     |
| `converged` | 0/1  | optimiser convergence flag                   |
| `n_iter`    | int  | optimiser iteration count (`NA` for R mlt)   |

## Reproducibility notes

- Input datasets use a deterministic master seed (`MASTER_SEED` in
  `bench_python.py`) — re-running the benchmark produces byte-identical CSVs
  in `data/`.
- Right-censored cells use a Bernoulli censoring indicator with target rate
  `CENSORING_RATE = 0.30`.
- Absolute timings depend on hardware, R version, and BLAS — the speedup
  ratio (`R median / Python median`) is the meaningful comparison.
- Benchmarks are intentionally not run in CI; runner variance would make the
  numbers misleading. Run locally and commit the refreshed CSVs + report.
