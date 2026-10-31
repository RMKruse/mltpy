# Prediction runtime and memory baseline (wayfinder #140)

`dev` @ `b906968`. Evidence for [Measure prediction reuse and retained cache
memory](https://github.com/RMKruse/mltpy/issues/140), feeding [Choose the
smallest worthwhile performance interventions](https://github.com/RMKruse/mltpy/issues/141).
No library code was changed. All candidate changes below are harness-local
prototypes, checked for bitwise equality against current library output.

## Reproduce

```sh
PY=/path/to/.venv/bin/python sh research/prediction-baseline/run_all.sh  # ~12 min, sequential
python research/prediction-baseline/summarize.py                          # -> tables.md
python research/prediction-baseline/diag_memo_peak.py                     # memo vs sorted reuse
python research/prediction-baseline/diag_score_scaling.py                 # issue #144
```

- **Files:** `harness.py` (one case per process), `results/*.json` (raw), `tables.md` (tables A–H).
- **Settings:** `random_state=0` everywhere. Caches are cleared before every cell so cells are independent.
- **Environment:** Apple M5 Pro, Python 3.12.16, NumPy 2.4.6 (Accelerate), SciPy 1.17.1. Thread count and system load were not controlled.

## What was measured

There are 9 fitted models, each independently validated against SLSQP (|Δℓ| ≤ 2.2e-7, table A):

| Path | Models |
|---|---|
| Ordinary shift bisection | BoxCox, 3 covariates |
| Shift + scaling bisection | BoxCox with `scaling=` |
| R-style right-censored inversion | MLT baseline, Coxph with 3 covariates |
| Survreg positive-support inversion | Survreg with 3 covariates |
| Interaction inversion | Bs⊗Bs with m = 8, 15, 28, plus a scaled m = 9 model |

The interaction configurations are the ones #139 validated. Large-m, large-n interaction fits are excluded because of #142.

Operations: CDF, density, quantile and simulate, at batch sizes m = 10², 10³, 10⁴, 10⁵. The R-style per-row loop is capped at 10⁴ because its cost is linear (60 µs per row).

Each cell records:

- **Timing**
  - the first call on new data;
  - repeated calls on the same data;
  - repeated calls on changing data.
- **Memory**
  - traced peak and retained memory after one call;
  - global-cache bytes, deduplicated by array and including the content keys;
  - model-instance array bytes;
  - process RSS around a 40-call burst on changing data.
- **Cache instrumentation** at m = 1,000
  - raw and assembled cache hits and misses;
  - `derivative()` calls;
  - derivative matrices built on value-only calls;
  - bytes hashed into keys.

## Findings

### 1. R-style right-censored quantiles

Covers `ConditionalTransformationModel._predict_quantile` (RIGHT branch) and `Survreg._predict_quantile`.

**Cost.**

- 60–65 µs per probability regardless of the data: 0.6 s for 10⁴ quantiles or `simulate(10_000)`.
- Memory is immaterial: ≤ 0.4 MiB peak, nothing retained, no cache use.
- About 99% of the time goes to rebuilding the CDF grid, `CubicSpline` and the sort/unique step for every probability; reusing one curve leaves about 1%.

**What reuse can save.** The curve depends only on the total shift (xβ + offset).

| Shift pattern | Library | Reuse the previous curve | Memo per distinct shift (unbounded) | Stable sort by shift, reuse, scatter back |
|---|---|---|---|---|
| All equal (baseline; one subject; `simulate` on one profile), m = 10⁴ | 600 ms | 5.7 ms (~105×) | 5.8 ms | same 1 curve |
| 10 interleaved profiles, m = 10⁴ | 605 ms | 554 ms (~10%) | 6.6 ms (~90×) | 10 curves, ≤ 0.3 MiB peak |
| All distinct (varying X, any non-constant offset), m = 10⁴ | 605 ms | 603 ms | 616 ms (+2%), **26 MiB transient peak** | ≤ 0.3 MiB peak |

- Every variant is bitwise equal to the library, and the saturation flag and warning are unchanged.
- The same holds for Survreg (its grid starts at positive support, and it does not force the first CDF value to 0) and for the cells where the library emits the saturation warning.
- Reusing only the previous curve, as the reconnaissance proposed, misses interleaved designs.
- The sorted form gets the memo's benefit without the memo's memory.

### 2. Interaction quantiles and simulate

Covers `_predict_quantile_interaction`.

Each of about 40 bisection midpoints calls `y_basis.evaluate(mid)`, which in turn calls `evaluate_with_derivative`. Per midpoint that means:

- 2 raw-cache misses and 1 assembled-cache miss;
- an unused derivative matrix dB built;
- three one-use, m-sized entries plus their content keys inserted.

So a **single call** fills the global caches to their entry caps:

| Model | Retained, m = 10³ | m = 10⁴ | m = 10⁵ |
|---|---|---|---|
| m = 8 (p = 4) | 3.5 MiB | 34 MiB | **342 MiB** |
| m = 15 (p = 5) | 4.2 MiB | 42 MiB | **416 MiB** |
| m = 28 (p = 7) | 5.7 MiB | 56 MiB | **562 MiB** |

The output itself is only 0.76 MiB at m = 10⁵.

The value-only, uncached midpoint evaluation (prototype, table E):

- is bitwise equal on all 16 cells, and **2.05–2.75× faster**;
- retains 7–11 MiB at m = 10⁵, which is the x-basis entries for the call (≈ 2% of before);
- lowers the peak from 366–597 MiB to 28–42 MiB.

The time splits as about 45–50% derivative work and 5–9% cache insertion plus key hashing.

`Survreg(interacting=...)` should behave the same way, because `LogBernsteinBasis` delegates to the same caches. That was not measured directly.

### 3. CDF and density retention (all Bernstein paths)

**Retention.**

- One call on a new `y` retains its design entries: 17.5 MiB at m = 10⁵ for shift paths, 17–28 MiB for interaction paths (including the x-basis).
- The caches are bounded by entry count, not bytes. Twelve changing calls at m = 10⁵ retained 211 MiB on shift paths and 201–339 MiB on interaction paths.
- At the caps (32 assembled / 64 raw entries) the projection is about 0.5 GiB.
- Clearing the caches did not lower RSS on this platform (243 → 387 → 387 MiB).

**But the cache pays for itself.**

- A repeated call on the same `y` is 5–6× faster (3.2 vs 18–19 ms at m = 10⁵).
- The key is `y` content only, so the same time grid scored against different `X` also hits.

**Unused work.** The unused `derivative()` call costs about 10% of a CDF call (table G).

Removing caching from prediction would cost a real speedup. A byte-based bound is the relevant policy option.

### 4. Other observations

- **Model-instance arrays:** prediction never changes model-held arrays (Δ = 0 in every cell). The fitted model itself holds ≤ 0.12 MiB. All retention is in the global caches.
- **Fit-entry eviction:** any prediction burst evicts all fit-time entries. `score()` on the training data goes from 0.03–0.08 ms to 0.19–0.38 ms, which is immaterial at these n.
- **Key hashing:** `tobytes` plus hash costs about 1.5 ns per element (150 µs at m = 10⁵). In an interaction quantile call that is about 120 keys (≈ 4% at m = 10³). It disappears with value-only evaluation.
- **Ordinary and scaled shift quantiles:** these bypass the caches entirely (zero retention), as the ticket expected. They cost about 2.4 µs per row (240 ms at 10⁵), 84% of it in the inlined `_h_vec` bisection. There is no small candidate here; a speedup would mean an algorithm change.

### 5. Correctness defect found (out of scope)

`score()` never forwards `scaling=` to `log_likelihood`:

- On shift + scaling models it returns a wrong ℓ (−1422.09 vs the fitted −1388.59).
- On interaction + scaling models it raises a raw matmul `ValueError`.

Filed as [score() ignores scaling=](https://github.com/RMKruse/mltpy/issues/144). The harness's eviction probe calls `log_likelihood(..., scaling=)` directly for scaled models.

## Candidates, callers and correctness coverage

Each candidate below lists where it applies, why, its current test coverage, and any gaps.

**A. Sort-grouped curve reuse in the R-style inversion**

- **Where:** `model.py:1062-1154` and `tram.py:1354-1460`. The two loops are near-duplicates whose only differences are the grid start and the first-value-zero rule.
- **Why:** ~100× for equal shifts, 50–90× for few profiles, no loss for distinct shifts, O(1) curve memory.
- **Reached from:** `predict(what="quantile")` and `simulate()` on any `CensoringType.RIGHT` shift model without scaling (MLT, Coxph, and every tram wrapper fitted right-censored).
- **Not reached from:** scaled models (they take the bisection path) and interaction models.
- **Covered by:** Coxph R quantile fixture (`tests/test_tram.py:783`), offset quantiles (`tests/test_weights_offset.py:302`), and the quantile tests in `tests/test_model.py`.
- **Gap:** no test exercises the Survreg shift-quantile override at all. Before any change, add a positive-support Survreg test (batch equals per-row calls, plus round trip) and batch-vs-individual tests for baseline, equal, interleaved, changing and offset shifts, including a saturating batch and its warning.

**B. Value-only, uncached midpoint evaluation in interaction bisection**

- **Where:** `model.py:1399-1407`.
- **Why:** ~2× time and about 98% less retained memory at large m.
- **Constraint:** `test_basis.py::test_evaluate_warms_shared_assembled_cache` pins `BernsteinBasis.evaluate()` to warm the shared `(B, dB)` cache. The change should therefore be local to quantile inversion (a private value-only path) and must not change `evaluate()` semantics.
- **Basis types:** any y-basis type the interaction path accepts. `BernsteinBasis` and `LogBernsteinBasis` share the caches; other bases could keep calling `evaluate()`.
- **Covered by:** interaction R quantile parity (`tests/test_interaction_predict_extras.py:123`), interaction-scaling round trips (`tests/test_interaction_scaling.py`), and the `tests/test_tram_*_interacting_scaling.py` quantile R tests (including Survreg, a log-time y-basis).
- **Also covered:** cache bounds and read-only contracts (`tests/test_basis.py:297-337`).

**C. Byte-aware cache policy for prediction-time CDF/density** (broader; lower priority)

- **What:** bound the global caches by bytes, or skip insertion for large one-off batches.
- **Trade-off:** keeps the 5–6× same-`y` win while capping the up to ~0.5 GiB retention.
- **Scope:** touches every consumer of the global caches, including fitting (see #139, where the cache is what makes fitting fast). It needs its own decision on what "large" means.
- **Not recommended:** removing the derivative from CDF calls saves only about 10%.

**D. Leave alone**

- **Ordinary and scaled shift quantiles:** no cache involvement, about 2.4 µs per row.
- **Fit-entry eviction:** immaterial.
- **Model-instance memory:** unchanged by prediction.

## Remaining uncertainty

- One machine, with load and thread count uncontrolled.
- RSS return behaviour is specific to the macOS allocator.
- The R-style path was measured only up to m = 10⁴; its cost is linear.
- Not measured:
  - `Survreg(interacting=)` with a log-time y-basis;
  - Polr `predict_proba`;
  - `confband()` and `residuals()`;
  - weights, and censoring types other than exact and right.
- No R timing comparison was made.
- Large-m, large-n interaction models are not covered until #142 is fixed (m = 28 only at n = 250).
