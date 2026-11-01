# trust-constr objective Hessian — issue #166

Decision: pass the existing analytic NLL Hessian on the shift path, except
exponential interval censoring. Keep BFGS on that nonsmooth path and on the
separate interaction path, whose Hessian is finite-difference. No likelihood
formula, public option, SLSQP weight normalization, or auglag behavior changes.

## Objective and reduction

For a free-parameter selection matrix S and the fixed contribution c,
θ = S θ_free + c. The solver therefore receives H_free = Sᵀ H(θ) S, implemented
as the free/free block after reconstructing θ. H uses exactly the objective's
weights, offset and scaling. In particular trust-constr retains the caller's
raw weights; only SLSQP uses mean-1 weights (#137).

The analytic shift Hessian handles exact, left-, right- and interval-censored
observations, all four distributions, scaling, offsets and truncation. The
exception below concerns Newton steps across the exponential support boundary,
not a missing Hessian kernel. The debug `use_gradient=False` path also receives
the analytic Hessian, alongside SciPy's numerical gradient.

## Why passing hess alone is insufficient

With SciPy's default initial barrier parameter 0.1, exact curvature reaches
`gtol` before the barrier has decayed enough. On the unscaled exponential
offset fixture, logLik stopped at −150.509484 rather than −150.507881; the
fixed-beta scaled fit missed θ by up to 4.7e-5. Starting the barrier at
`OptimizerConfig.tol` restores existing tolerances without tightening `gtol`
or relaxing any reference tests. This option is set only when the analytic
objective Hessian is used, leaving the BFGS paths unchanged.

Exponential interval probabilities have a kink at h(lower) = 0. With the
analytic Hessian, the existing inspection-grid case (seed 1, q = 0.5) never
left θ = [0, 0.25, 0.5, 0.75, 1, 0], then reported successful `xtol` termination
with optimality about 44 and logLik −214.931065 instead of −204.008078.
The unconditional change caused 20 failures across the two exponential test
files; the smaller initial barrier alone still left six interval failures.
Retaining BFGS for exponential INTERVAL restores those fits. Replacing the
support treatment or introducing another solver is outside this change.

The objective BFGS `delta_grad` warning can consequently still occur on
exponential interval or interaction fits. The original scale-only nonzero-offset
warning did not reproduce on this SciPy version; that fit now receives analytic
curvature and its existing test promotes that warning to an error.

An attempt starting at an undefined NLL also retains BFGS with its original
barrier settings. Fixed parameters can make h' zero at the initial point even
when the constrained fit is feasible. Evaluating the analytic Hessian there
raised before the finite objective penalty could recover. A public regression
and the added infeasible-fixed-start sweep case preserve baseline recovery.
The original scalar NLL is checked before each attempt, so restarts follow the
same rule; no replacement Hessian is invented at an undefined point.

## Before/after sweep

`before.json` uses optimizer.py from starting commit
`3c3a1dcccfcad24293e5681d5d9b5d458cb96ddf`; `after.json` uses this change.
Both use the same `sweep.py`, existing R fixtures and seeded synthetic inputs.
Each time is the median of three optimize() calls, including setup and restarts,
with one reused basis per case. These small local timings are descriptive,
not a portable performance guarantee. The runs were sequential, after the test
suite and documentation build had finished.

Environment: Python 3.13.13, NumPy 2.5.4, SciPy 1.18.1.

All 32 cases converge before and after; all have zero restarts. Every R logLik comparison passes rtol=1e-6. θ errors against R are retained
in the JSON; weighted Colr has the same existing ~0.0022 absolute discrepancy
before and after, while its logLik and fitted θ are unchanged to round-off.
No new R fixtures were generated because the likelihood is unchanged.

Median speedup across the sweep: 2.07× (individual ratios 0.86–4.13×). The six scaling R fixtures require 20–36 iterations after the change,
compared with 58–139 before. Some small synthetic cases improve little.

| Case | Iterations before → after | Median ms before → after | max abs Δθ | abs ΔlogLik |
|---|---:|---:|---:|---:|
| scaling_boxcox_normal | 58 → 20 | 19.27 → 6.44 | 1.2e-06 | 4.9e-07 |
| scaling_colr | 67 → 27 | 22.98 → 8.37 | 2e-07 | 5.1e-08 |
| scaling_coxph | 64 → 25 | 22.69 → 8.94 | 1.1e-07 | 1.2e-07 |
| scaling_boxcox_interval | 139 → 36 | 50.38 → 12.20 | 2.4e-06 | 2.2e-08 |
| mlt_exponential_scaling | 62 → 25 | 24.85 → 7.59 | 4.9e-08 | 9.6e-08 |
| mlt_exponential_scaling_shift | 58 → 24 | 83.20 → 29.81 | 7e-07 | 4.8e-07 |
| exponential_scale_only_offset | 57 → 22 | 83.34 → 26.06 | 1.9e-07 | 2.6e-07 |
| exponential_unscaled_offset | 56 → 26 | 71.24 → 30.57 | 5.1e-08 | 1.3e-07 |
| exponential_RIGHT | 61 → 25 | 84.18 → 32.42 | 4.1e-06 | 9.6e-08 |
| exponential_LEFT | 52 → 30 | 68.84 → 42.54 | 1.6e-08 | 1.7e-08 |
| exponential_INTERVAL | 59 → 59 | 80.03 → 88.65 | 0 | 0 |
| exponential_fixed_beta | 51 → 24 | 74.83 → 32.04 | 3.4e-07 | 3.9e-07 |
| weights_boxcox | 47 → 20 | 17.71 → 5.42 | 8.7e-08 | 0 |
| weights_colr | 56 → 46 | 18.23 → 13.77 | 1.5e-07 | 0 |
| weights_coxph | 52 → 26 | 17.68 → 8.90 | 3.4e-09 | 1.1e-07 |
| synthetic_NONE_scaled=False_fixed=False | 41 → 18 | 12.57 → 4.96 | 1.6e-07 | 2.6e-07 |
| synthetic_NONE_scaled=False_fixed=True | 41 → 19 | 11.61 → 5.20 | 4.3e-08 | 5.1e-08 |
| synthetic_NONE_scaled=True_fixed=False | 45 → 20 | 14.21 → 5.99 | 1.4e-08 | 5.1e-08 |
| synthetic_NONE_scaled=True_fixed=True | 39 → 18 | 10.87 → 4.98 | 1.5e-08 | 5e-08 |
| synthetic_LEFT_scaled=False_fixed=False | 43 → 35 | 12.70 → 8.85 | 6.2e-08 | 3.1e-08 |
| synthetic_LEFT_scaled=False_fixed=True | 38 → 19 | 10.94 → 5.27 | 1.3e-07 | 5.1e-08 |
| synthetic_LEFT_scaled=True_fixed=False | 48 → 35 | 15.23 → 9.10 | 2.1e-07 | 1.8e-07 |
| synthetic_LEFT_scaled=True_fixed=True | 41 → 38 | 11.70 → 9.99 | 1e-07 | 6.9e-09 |
| synthetic_RIGHT_scaled=False_fixed=False | 41 → 40 | 12.26 → 11.99 | 6.3e-07 | 2.4e-07 |
| synthetic_RIGHT_scaled=False_fixed=True | 39 → 21 | 11.68 → 5.63 | 8.1e-07 | 2.6e-07 |
| synthetic_RIGHT_scaled=True_fixed=False | 49 → 25 | 15.76 → 7.94 | 8.3e-08 | 5.1e-08 |
| synthetic_RIGHT_scaled=True_fixed=True | 36 → 21 | 10.81 → 6.36 | 1.1e-05 | 6.4e-06 |
| synthetic_INTERVAL_scaled=False_fixed=False | 40 → 37 | 12.58 → 12.23 | 1.1e-07 | 2e-07 |
| synthetic_INTERVAL_scaled=False_fixed=True | 40 → 21 | 11.95 → 6.34 | 3.5e-08 | 5.1e-08 |
| synthetic_INTERVAL_scaled=True_fixed=False | 47 → 40 | 15.64 → 14.12 | 1.5e-08 | 9.7e-09 |
| synthetic_INTERVAL_scaled=True_fixed=True | 41 → 35 | 12.57 → 11.75 | 9.8e-08 | 2.1e-09 |
| infeasible_fixed_start | 25 → 25 | 5.64 → 6.55 | 0 | 0 |

## Reproduce

From the repository root:

```bash
git show 3c3a1dc:src/mltpy/optimizer.py > /tmp/mltpy-166-optimizer-before.py
.venv/bin/python - <<'PYBASE' > research/trust-constr-hessian/before.json
from pathlib import Path
import runpy
import mltpy.optimizer as optimizer
exec(compile(Path('/tmp/mltpy-166-optimizer-before.py').read_text(),
             optimizer.__file__, 'exec'), optimizer.__dict__)
runpy.run_path('research/trust-constr-hessian/sweep.py', run_name='__main__')
PYBASE
.venv/bin/python research/trust-constr-hessian/sweep.py \
  > research/trust-constr-hessian/after.json
```

The baseline command replaces the optimizer module in that Python process
only; the checkout stays unchanged. The sweep checks finite results,
monotonicity, exact pins and R logLik parity before writing JSON.

## Validation

- A regression test failed before the change because SciPy received no Hessian.
  The SciPy boundary test now checks curvature against central differences of
  the actual objective gradient across all distributions, censoring types,
  scaled/unscaled designs, unequal weights (including zero), offsets and
  multiple fixed parameters. It also checks the exponential interval fallback.
- Real fits cover analytic and numerical gradients, scaled R θ/logLik,
  weighted R logLik/scores, and the original scale-only offset warning case.
- Full suite: 2,733 passed, three warnings (the retained exponential interval
  BFGS path and two existing prediction warnings).
- Strict mypy, source/test/new-harness Ruff and formatting checks pass.
  Sphinx HTML builds with `-W --keep-going`.
- Repository-wide Ruff has 124 existing findings in 14 unchanged notebooks and
  research scripts; this change leaves those files alone.
