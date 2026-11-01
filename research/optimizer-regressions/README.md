# Five optimizer regressions at the audit baseline (#171)

Evidence only: no library changes, new defaults, numerical acceptance protocol,
or performance campaign. Baseline is `2659daca9057ca7aa4f77589565dcbe8bada1297`,
which was HEAD when these runs were made. Closed issues are historical sources,
not evidence of correctness. See [parent #168](https://github.com/RMKruse/mltpy/issues/168).

## Answer

| Seed report | Verdict at baseline | Evidence and qualification |
|---|---|---|
| [#165](https://github.com/RMKruse/mltpy/issues/165), failed first attempt | **fixed on nominated fixture** | Scaled and unscaled shift + varying-offset fits converge on all three solvers with no restart or warning. NLL 144.805007 and 150.507881 respectively. The original input is retained; the start correction is already in this baseline. |
| [#163](https://github.com/RMKruse/mltpy/issues/163), interval warnings | **fixed on seed 0, q=.8** | All three solvers return NLL 234.712354, with no captured fit-time or final-point warnings, finite gradients and independent NLL agreement. The historical nine-design sweep is already in the test corpus; this probe does not repeat that entire grid. |
| [#162](https://github.com/RMKruse/mltpy/issues/162), alleged clipping optimum | **invalid reproduction of intended censoring** | Literal input encodes 30 exact / 120 censored rows; intended input has 120 / 30. Historical Python theta re-scored with literal flags gives unclipped logLik +257.746684 (normal), +238.563580 (logistic), agreeing with stored R values. Scoring those same parameters with corrected flags gives NLL 2.934e20 / 4.967e10, which explains the original comparison error. Corrected refits agree across solvers at NLL 63.415358 / 65.813506. |
| [#161](https://github.com/RMKruse/mltpy/issues/161), auglag divergence | **fixed on literal and corrected seed-0 input** | The literal report also reverses the flags. Both encodings are retained. All solvers converge, stay monotone and satisfy lower-end support to numerical precision: literal NLL −1.938360, corrected 51.140931. This does not validate the original Type-I interpretation of the literal input. |
| [#160](https://github.com/RMKruse/mltpy/issues/160), omitted offset | **support defect fixed; status unresolved** | Both offset signs return minimum support near zero on all solvers, NLL about 5.132565. SLSQP returns “Positive directional derivative for linesearch”, with and without default restarts. Its default +.5 run has min support −4.23e−8; record this residual rather than calling a raw success flag an acceptance test. |

The [complete matrix](matrix.md) contains all **17 cases × 3 solvers × 2 restart
budgets = 102 runs**, including failures. No run raised or timed out.
[Raw results](results.json) retain every attempt start and returned raw result,
final selected parameters, inner solver statuses and evaluation counts,
iteration/outer-loop/KKT diagnostics where available, warning categories,
locations and counts, versions, configuration, thread settings and data arrays.
An absent solver field is unavailable, not zero. `inner_results` includes
auglag inner solves and any polish; summing its `nfev` does not count external
verification calls or constitute comparable work across algorithms.

## Newly exposed gaps

The four nonzero-offset interval designs deliberately reuse the #163 inspection
grid, with scale only or shift + scale, offsets ±.2. They are **unresolved**:

- With default restarts, auglag reports failure on all four and returns
  infeasible support minima from −.000697 to −.0704. Lower reported NLL here
  cannot be interpreted as a better constrained solution.
- With no restarts, trust-constr stalls at NLL 299.316927 (scale only, +.2)
  and 339.873182 (shift + scale, −.2). Default restarts recover values near
  241.42306. SLSQP reaches about 241.423059 on all four without restarts.
- These are optimizer-level probes. Model post-fit Hessian, covariance and
  prediction behavior remain covered by existing tests rather than by the
  instrumentation here. No production fix is included.

The wider clipping question is tested **separately**. For one exact normal row,
order 1, support [0,1], y=.5, theta=[40,41], the unclipped h is 40.5 and h'=1.
Independent NLL is 821.043939; evaluated NLL is 450.918939 (h clipped to 30).
This **reproduces an objective difference**, not a fitted bad optimum, and does
not revive #162's rejected diagnosis. A literal logistic #162 fit also returns
an extreme unsuccessful point whose independent NLL is about 6.58e256 while
reported NLL is −2585: retain it as an unresolved extreme-point observation,
not a valid converged fit. Literal refits are sensitive to solver trajectories;
the historical same-parameter comparison is the stable evidence for #162.

## Construction and semantics checks

`probe.py` asserts that the censored mask is exactly the NaN exact-row mask;
interval bins contain the latent generated response. NumPy PCG64 draws use seed
0, separate shift seed 50 and restart seed 0. Responses for #161–163 are
Exp(1)/(2 exp(.3 x_s)), **not** exp(response) or log(response). #160 uses its
original uniform response design; #165 loads the existing #115 fixture.
All actual encoded data, covariates and offsets are saved, so reconstruction
need not rely only on seeds. Exact and left controls share the #163 draw.

The independent evaluator writes out
`h=B theta_b exp(.5 X_s gamma)+X beta+offset` and
`h'=B' theta_b exp(.5 X_s gamma)`, then uses scipy distribution logpdf/logcdf/
logsf and survival differences for the finite exponential intervals. It never
calls library likelihood internals. It reuses the public Bernstein evaluator;
it is not an independent basis implementation. Hand-worked exact, right and
interval exponential values and an exact normal value check this oracle.
No truncation or weights enter this small corpus. Normal/logistic intervals
are outside the oracle's exercised scope; use existing R fixtures for those.

Feasibility records minimum coefficient increment and minimum h(a|x) including
the offset. The latter is a constraint **only for exponential**, whose training
support rule is h(a|x_i)≥0 at a=the lower support endpoint, not at min(y_i).
It is meaningful for normal/logistic as a diagnostic value, not a violation.
Small negative exponential residuals are retained verbatim. No new global
feasibility, likelihood, stationarity or parameter tolerance is chosen.

Fit-time warning capture includes trial evaluations and restart-loop activity;
final-point capture re-evaluates NLL and analytical gradient at returned theta.
A warning in a fit is not automatically evidence that the final point warns.
The probe calls `optimize`, so model-level ConvergenceWarning is not generated.
`converged` records the library's flag without upgrading it based on NLL.

## Existing corpus to reuse

Discovery used `search_graph`, `get_code_snippet` and `search_code` in project
`Users-entropy-Git-mltpy`, including optimizer dispatch, shared restart loop,
initialization, result types and existing fixed-time tests. The bounded
[graph inventory](graph-inventory.json) preserves the top-level function query
(237 symbols, no truncation); it does not enumerate class methods. The table
adds existing class-based regression coverage and non-code assets.

| Coverage | Reusable assets | Limits / interpretation |
|---|---|---|
| Three solvers, starts, restarts, raw diagnostics, pins, weights | [test_optimizer.py](../../tests/test_optimizer.py), [test_slsqp_weight_scale.py](../../tests/test_slsqp_weight_scale.py), [test_r_parity_auglag.py](../../tests/test_r_parity_auglag.py), `reference/auglag/` | Inner algorithms, budgets and convergence semantics differ; raw success is not independent acceptance. |
| Exponential scale only, shift + scale, offsets, all four censoring encodings | [test_exponential_scaling_no_shift.py](../../tests/test_exponential_scaling_no_shift.py), [test_exponential_scaling_shift.py](../../tests/test_exponential_scaling_shift.py), [test_exponential_offset_support.py](../../tests/test_exponential_offset_support.py), [test_exponential_interval_support.py](../../tests/test_exponential_interval_support.py) | First-attempt checks, nine-design grids, support-kink and warning cases already exist; extend these after acceptance decisions instead of copying them. |
| R parity at same parameters and valid fitted references | [test_scaling_censoring.py](../../tests/test_scaling_censoring.py), [write_scaling_fixed_time_data.py](../../reference/write_scaling_fixed_time_data.py), [generate_reference.R](../../reference/generate_reference.R) | `scaling_fixed_time_*` keeps literal and corrected flags; `mlt_exponential_scaling_shift_*` uses explicit constraints, not native R mlt support. R-to-Python shift sign must be converted. |
| Other links / derivatives / inference | [test_scaling_distributions.py](../../tests/test_scaling_distributions.py), [test_gradient_verification.py](../../tests/test_gradient_verification.py), [test_scaling_inference.py](../../tests/test_scaling_inference.py), wrapper-specific scaling tests | Dispatch supports normal, logistic, min/max extreme value, Laplace and exponential on shift bases with all solvers. This probe fits only exponential, normal, logistic. |
| Interaction models | [test_interaction_r_parity.py](../../tests/test_interaction_r_parity.py), [test_interaction_scaling.py](../../tests/test_interaction_scaling.py), [test_interaction_exponential_support.py](../../tests/test_interaction_exponential_support.py) | Exact data only. Censored interaction, exponential interaction + scaling and interaction fixed_params are **unsupported** in optimize dispatch; do not count them as solver failures. |
| Broad R prediction/fit validation | [validation/run_validation.py](../../validation/run_validation.py), `validation/references/*/metadata.json`, [test_validation_script.py](../../tests/test_validation_script.py) | Keep provenance and fitted-parameter scoring separate; do not regenerate all references for this task. |
| Benchmark data, environment, failure accounting | [benchmarks/README.md](../../benchmarks/README.md), [bench_python.py](../../benchmarks/bench_python.py), [fit-phase-baseline](../fit-phase-baseline/README.md), [trust-constr-hessian](../trust-constr-hessian/README.md), [interaction-hessian-route](../interaction-hessian-route/README.md) | Prior measurements used earlier versions. Reuse designs, not old timing conclusions. No matched-accuracy campaign here. |
| R route and mathematical decisions | [exp-scaling-r-route](../exp-scaling-r-route/README.md), [ADRs 0001–0005](../../docs/adr/), [#169](https://github.com/RMKruse/mltpy/issues/169), [#170](https://github.com/RMKruse/mltpy/issues/170) | Native R support feasible sets can differ. Mathematical validity precedes parity on inequivalent models. |

## Rerun and provenance

From the repository root, with the baseline library source checked out (or a
separate checkout supplying `PYTHONPATH`):

```sh
.venv/bin/python research/optimizer-regressions/probe.py --self-check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python research/optimizer-regressions/probe.py \
  --out /tmp/optimizer-regressions.json
# Optional: --case 165, --case 163_INTERVAL, etc.
```

The probe uses temporary runtime wrappers and restores them after each fit;
these observe the existing restart and minimize calls without replacing
objectives, starts, tolerances or result selection. Each fit has a 90-second
budget, and errors/timeouts are recorded. Instrumented elapsed seconds are
context, not performance evidence. JSON serializes non-finite values as strings.
The source SHA256 covers sorted `*.py` contents in the imported library
directory (recorded as `library_source`), even with another checkout on
`PYTHONPATH`. `library_checkout` records that library's Git revision;
`checkout` records the evidence script checkout. Checkout commit,
Python, NumPy, SciPy, OS/architecture and BLAS build information are in results.
The current branch requirement from `/implement` was followed; no research
branch was created and nothing was published to GitHub.

Installed R provenance checked locally: R 4.6.1, mlt 1.8.4, tram 1.4.7,
basefun 1.2.6, variables 1.1.2, alabama 2025.1.0. Existing R fixtures were
re-scored, not regenerated. Their generators and historical research reports
remain the source for the reference contract.

## Decisions this evidence leaves open

Choose independent feasibility and stationarity checks, treatment of kinked
interval solutions and unsuccessful-but-close results, and solver-specific
versus common budgets before expanding acceptance claims. Decide whether the
nonzero-offset interval failures and exact-row clipping probe enlarge the
mandatory corpus; add weights, varying offsets under censoring, other links,
interaction cases and multiple starts/seeds at that point. None of these
choices follows merely from matching NLL or an issue being closed.

Validation: probe self-check and 102 recorded runs; offset test file 70 passed;
full suite 2733 passed (3 warnings); strict mypy passes for source and probe.
Ruff passes on source, tests and this probe; formatting checks pass on those
paths. Repository-wide Ruff reports 125 pre-existing errors in notebooks and
older research scripts. Sphinx `-W` completes rendering but fails on three
pre-existing untracked `docs/agents/*.md` files missing from the toctree; those
files were left untouched.
