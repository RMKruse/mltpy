# Fit-phase baseline (Wayfinder: "Measure fitting phases and validate optimizer outcomes")

Throwaway research branch. Evidence for #139 on map #138. **No library code is
changed**: the harness wraps `mltpy` internals at runtime only.

Measured commit: `dev` @ `b906968` (src clean). Apple M5 Pro, macOS 26.6.2 arm64,
Python 3.12.16, NumPy 2.4.6 (Accelerate BLAS, no thread env vars set), SciPy 1.17.1.
Every fit uses `OptimizerConfig(random_state=0)`.

## Reproduce

```sh
# from a checkout of the commit to measure, with its .venv
MLTPY_REPO=$(pwd) research/fit-phase-baseline/run_all.sh   # ~3 min, sequential, one process per case
.venv/bin/python research/fit-phase-baseline/summarize.py research/fit-phase-baseline/summary.md
```

`run_all.sh` hard-codes the interpreter as `/Users/entropy/Git/mltpy/.venv/bin/python`;
edit `P=` for another checkout.

Single case: `python harness.py --case ia_bs6x3_m28_n1000 --out results/x.json`;
`python harness.py --list` lists the 25 cases.

## Files

| File | Content |
|---|---|
| `harness.py` | Case matrix, timing, phase instrumentation, independent validation, memory, cProfile |
| `summarize.py` | Builds `summary.md` tables A–G from `results/*.json` |
| `results/*.json` | Raw per-case output, including environment, all timings, statuses, warnings and cProfile text |
| `summary.md` | Evidence tables |
| `diag_*.py` | Diagnosis of the false stall convergence in `ia_bs6x3_m28_n1000` |

## Method notes

- One process per case. *Cold* is the first fit in that process. *Warm* reps use
  fresh model objects; *refit* reps reuse the same instance. GC is disabled
  inside the timed region. Load was not controlled (load average 1.75–2.12).
- Phase attribution comes from a single instrumented fit that wraps
  `auglag_minimize`, `_ncg_polish`, `optimizer._hessian` (polish),
  `model._hessian` (final), `model._score_matrix` and `model.optimize`. It
  counts objective and FD-likelihood calls. Wrapper overhead is <1% (compare
  instrumented and warm times).
- Validation does not rely on solver flags. It recomputes NLL and gradient with
  the public `negative_log_likelihood`, then computes the projected
  stationarity ‖∇NLL − A_actᵀμ‖∞ (μ ≥ 0 by NNLS on rows with slack ≤ 1e-6),
  the minimum slack, and pre- vs post-polish values. It also runs an SLSQP
  cross-check (`tol=1e-12`) and compares against R fixtures where they exist.
  For interaction fits it compares the FD Hessian with a central difference of
  the analytic gradient.
- Memory: tracemalloc peak and retained (with and without the module-level
  Bernstein caches), plus `ru_maxrss` after the cold fit.
