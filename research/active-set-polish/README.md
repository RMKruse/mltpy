# Prototype: active-set polish for auglag (#177)

Research prototype for [#177](https://github.com/RMKruse/mltpy/issues/177),
built to measure the gain and the cost of the proposed polish for the audit's
prioritisation ([#176](https://github.com/RMKruse/mltpy/issues/176)). **No
library code is changed.** The polish runs outside the library, on the
problem each auglag fit defines.

## Answer

The prototype implements the #177 decisions as revised after #178
(null-space Newton on the active set, a primal active-set loop, the
Lagrangian Hessian for nonlinear rows, KKT-based acceptance).

- **It clears all four start-dependent R-parity failures.** Every case
  matches R from every one of 11 starts, at a worst ratio of 0.04 against
  CLAUDE.md's `rtol=1e-6, atol=1e-10`.
- **It works across the suite.** 99.7% of the 1,925 auglag fits are
  accepted. Median KKT residual drops from 7.8e-7 to 5.5e-14; the largest
  constraint violation drops from 9.2e-8 to 1.4e-13.
- **It is cheap.** 1.1–1.3 s over the whole suite: 0.4 ms median per fit,
  35 ms max, at most 11 iterations.
- **Applied to the whole suite, it breaks no test for a real reason.** The
  three non-xfail failures are artifacts of the probe, or an exact `≥ 0`
  test meeting a boundary at −2.4e-24 (see Apply mode).
- **From perturbed starts it clears 7 of #178's 9 start-dependent tests.** Of
  the other two, one fails only by round-off, at −3e-24. The other starts
  from an infeasible restart (#174) that the polish cannot repair yet (finding
  7).
- **Evidence changes five of #177's rules.** See Findings. Three further
  departures have not been tested (see Deviations). Decision 4's interior
  `gtol` remains open (finding 6).

## Results

### The #178 acceptance cases, 11 starts each (`run_cases.py`)

Ratio is `max |θ − θ_R| / (atol + rtol·|θ_R|)` with `rtol=1e-6,
atol=1e-10`; ≤ 1 passes.

| case | worst ratio, auglag | worst ratio, polished | accepted | median KKT auglag → polished | median ms |
|---|---|---|---|---|---|
| `reference/auglag/exponential_n50_seed43` (50/51 rows active) | 5.02 | **0.00** | 11/11 | 3.6e-11 → 6.0e-11 | 0.4 |
| `mlt_exponential_offset_*` (#160) | 1.71 | **0.00** | 11/11 | 1.4e-5 → 4.2e-12 | 0.5 |
| #115 β pinned | 10.48 | **0.00** | 11/11 | 1.5e-5 → 1.2e-10 | 0.6 |
| one-hot BoxCox interacting scaling (interior) | 1.94 | **0.04** | 11/11 | 2.9e-6 → 3.6e-10 | 4.6 |
| control: `reference/auglag/lm_n50_seed42` | 0.00 | 0.00 | 11/11 | 5.4e-9 → 1.7e-11 | 0.1 |
| control: #115 free | 0.33 | 0.00 | 11/11 | 2.0e-6 → 4.0e-13 | 0.6 |

The interaction case uses production's finite-difference Hessian (#118),
which is why it costs the most.

### Whole suite, polish logged only (measure mode, `polish_probe.py`)

Every auglag `optimize` call in the suite is polished; nothing is returned
changed. 1,924 fits:

- 1,917 accepted (99.7%). One probe error: the test that asserts
  `_get_dist` is not called inside `optimize()`.
- 6 rejected; apply mode rejects the same fits:
  - **4 × "KKT not lower".** These are two profile fits of
    `test_confidence.py::test_confint_profile_colr_matches_R`, each logged
    twice.
    - auglag ends at KKT 9.0e-9 and 1.9e-8, with the profile's equality pin
      off by 9.5e-10 and 1.5e-9.
    - The polish puts the pin back exactly. With finding 6's |NLL|-scaled
      `gtol` it then counts that point as converged at iteration 1, without
      a Newton step.
    - It ends 20–25% higher.
  - **1 × infeasible (4.5e-7)**, `test_extrapolated_x_warns[auglag]`. This is
    finding 7.
  - **1 × "likelihood worse"** by 9.1e-8, against an allowance of 8.7e-8
    (`TestResidualsIntervalCensoring::test_narrow_interval_runs`, KKT 9.0e-6
    → 5.0e-9). The finding 1 allowance is only first order, so a margin this
    small is within its error.
- KKT median 8.8e-7 → 5.7e-14; 90th percentile 4.8e-6 → 3.7e-11.
- Total cost 1.3 s.

### Whole suite, polish applied (apply mode)

When accepted, the polished θ and logLik replace auglag's in the returned
result.

- **Fits:** 1,925; 1,918 accepted (99.7%).
- **KKT:** median 7.8e-7 → 5.5e-14; 90th percentile 4.4e-6 → 3.7e-11.
- **Largest violation:** 9.2e-8 → 1.4e-13.
- **Cost:** 1.1–1.3 s in total over runs; at most 11 iterations.
- **Tests that fail only in apply mode (7):**
  - **The 4 strict xfails of `tests/test_optimizer_start_sensitivity.py`.**
    They now pass, which strict mode reports as a failure: the polish
    clears them.
  - **`test_exponential_scaling_shift::TestSolverDispatch::test_optimize_direct`**
    compares `constraint_A_ineq` with `J(θ)`. The probe replaces only θ, so
    `J` is stale. A production polish recomputes the result fields.
  - **`test_optimizer::TestPolishStep::test_polish_tightens_gradient_on_loose_auglag`**
    compares the existing polish against no polish; in apply mode both runs
    get the prototype.
  - **`test_model::test_integration_r_reference_new_distributions[exponential-…]`**
    asserts `h(a) ≥ 0.0` exactly and gets −2.4e-24. The polished point is
    on the boundary up to round-off. A production polish should restore
    onto `slack = 0⁺` (e.g. one extra correction rounding toward the
    feasible side), or the test should use a tolerance.

Probe artifacts, deselected or excluded in every run:

- `TestInteractionDesignReuse` (3 tests) counts `_InteractionDesign.build`
  calls, and the probe builds the design again.
- One test asserts `_get_dist` is not called inside `optimize()`.
- One test stubs `_optimize_auglag` to return `None`.

### Whole suite, perturbed starts, polish applied

The #178 sweep again (`run_probe.py`, starts k = 0…10), now with the polish
applied, against #178's unpolished sweep from the same starts
(`compare_seeds.py`):

| start k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unpolished: failing tests | 0 | 3 | 4 | 2 | 3 | 2 | 3 | 5 | 2 | 2 | 3 |
| unpolished, rerun: failing tests | 0 | 3 | 4 | 1 | 3 | 2 | 4 | 4 | 2 | 1 | 3 |
| polished: failing tests | 7 | 9 | 9 | 8 | 8 | 8 | 8 | 7 | 9 | 6 | 8 |
| polished, rerun: failing tests | 7 | 9 | 9 | 8 | 7 | 9 | 8 | 7 | 9 | 6 | 9 |
| polished, excluding the expected apply-mode failures (both runs) | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |

The two runs were made at the same commit, and their raw counts still
differ at a few starts. #178 noted that `tests/test_r_parity_auglag.py` fits
with `random_state=None`, so not every fit in a sweep is seeded. The row
that matters is the last one, and it is identical in both runs. Neither run
has a failure that is unexpected under the polish.

- **Every start-dependent failure #178 found is cleared, except one.** Seven
  of #178's nine start-dependent tests pass from every start with the
  polish. These are the four R comparisons, both pin checks, and the
  Bernstein monotonicity check. The rerun's unpolished sweep hit eight
  start-dependent tests. The polish cleared six of them:
  - the four R comparisons;
  - the #115 pinned fit;
  - the Bernstein monotonicity check.

  The other two are listed next.
- **`test_integration_r_reference_new_distributions[exponential-…]`:**
  - Unpolished, `h(a)` was −1.6e-9 to −4.0e-9 at starts 3, 8 and 10. In the
    rerun it failed instead at starts 5 and 10, on the logLik comparison at
    `rtol=1e-6` (relative 1.1e-6).
  - Polished, `h(a)` is −2.6e-25 to −3.1e-24 at 8 of 11 starts (rerun:
    −6.3e-27 to −4.4e-24 at 10 of 11). This is the same exact `≥ 0`
    round-off as in apply mode.
- **`test_extrapolated_x_warns[auglag]` fails from the same 8 starts with and
  without the polish.** The polish is rejected there ("infeasible"):
  - Restart selection (#174) returns a `converged=False` attempt that
    violates two support rows by up to 6.0e-7.
  - The polish ends at a KKT point of its working set (|Zᵀ∇f| = 7.7e-12) that
    still violates them by 5.4e-7. See finding 7.
- **One more probe artifact.**
  `test_optimizer::TestPolishStep::test_polish_skipped_when_disabled` fails at
  starts 1, 2, 3, 5, 8 and 10 in apply mode only. Like
  `test_polish_tightens_gradient_on_loose_auglag`, it compares polish off
  with polish on, and the probe polishes both runs. The polish-off run then
  ends at ‖∇f‖ ≈ 4e-14, and the polish-on run at 2.2e-9. Why the second one
  stops early is finding 6.

"Expected apply-mode failures" are the seven listed under Apply mode, plus
`test_polish_skipped_when_disabled`.

### Scaled Coxph: the R reference is not at the optimum (`coxph_check.py`)

`test_tram_coxph_scaling::test_coxph_theta_beta_gamma_match_R` holds θ to
`rtol=1e-4, atol=1e-5`. auglag reports `converged=True` on this fit:

| point | ratio @ `rtol=1e-6` | max rel Δθ vs R | logLik − R | KKT (prototype measure) |
|---|---|---|---|---|
| auglag | 113 | 1.1e-4 | −1.64e-4 | 4.8 |
| polished | 132 | 1.3e-4 | **+4.3e-7** | 1.2e-14 |

The polished point is feasible, is a verified KKT point, and has a
**higher** likelihood than R's fit. So the R reference (tram) stopped short
of the optimum, and the loose tolerance hides it. This fixture cannot serve
as ground truth at `1e-6` until it is regenerated with tighter optimiser
tolerances. This is for the audit's acceptance protocol (#174).

## Findings for #177

1. **Acceptance: likelihood needs a feasibility allowance.** auglag leaves
   active rows 1e-10 to 1e-7 on either side of the boundary. Putting them
   exactly on it changes the NLL by `λ_i |c_i|` to first order. On #160
   that was +5.6e-9 (≈ 70 × 8e-11); suite-wide it reached +5.8e-6, with all
   200 support rows active. The rule "likelihood not worse" rejected exactly
   the correct answers. The working rule: NLL may rise by at most
   `Σ |λ_i| |c_i(θ₀)|`.
2. **Acceptance: KKT "strictly lower" needs a round-off floor scaled by
   ‖∇f‖∞.** At a constrained optimum, ∇f has the size of the multipliers.
   Residuals of 1e-11 to 1e-10 are round-off there, and "strictly lower"
   rejected the exact boundary points that fix the 50-of-51-active fixture.
   The working rule: KKT not higher, up to `1e-10 · max(1, ‖∇f‖∞)`.
3. **Line search: judge by the reduced gradient below f's resolution.** Near
   the optimum the predicted decrease (~|Zᵀ∇f|²/H ≈ 1e-14) is below f's
   round-off. Armijo then fails at random and the step halves every
   iteration: one plain fit took 30 iterations to halve KKT. Below
   `64·eps·max(1, |f|)` the step is accepted if `‖Zᵀ∇f‖` decreases; that fit
   now converges in 2 iterations to KKT 1.2e-14.
4. **Active-set changes are rare.** Rows were added in 7 fits and dropped in
   6, out of 1,925. The loop matters for correctness on the ambiguous fits,
   not for cost.
5. **Gains beyond R parity.** The polish raises the likelihood where auglag
   stopped short: by 1.6e-4 on the scaled Cox fits (KKT 2.7e-2 reported as
   converged) and 3.2e-6 on a Survreg log-logistic fit.
6. **The stopping tolerance scales with |NLL|, not with the gradient.** The
   prototype stops when `‖Zᵀ∇f‖∞ ≤ gtol · max(1, |NLL₀|)`. With
   `gtol = 1e-10` and |NLL| = 582 (`TestPolishStep`'s 300-row fit), the
   threshold is 5.8e-8. A fit that arrives at ‖Zᵀ∇f‖ = 1.9e-8 therefore
   counts as converged at iteration 1 and is returned unchanged. From
   auglag's raw point the same fit reaches 8.7e-13, because the last Newton
   step overshoots the tolerance. The same effect causes the four "KKT not
   lower" rejections in measure mode. The polish should scale `gtol` by the
   gradient or curvature, not by f. The same applies to `kkt_floor`. Then a
   start near the optimum is not mistaken for one at it.

   **This leaves Decision 4's interior `gtol` open.** #177 asks the
   prototype for the value. The value used here, `1e-10 · max(1, |NLL₀|)`,
   is enough for every R-parity case: the interior BoxCox case reaches a
   ratio of 0.04. But it has the wrong scaling. The prototype has not tested
   a gradient-scaled rule, so it proposes no value.
7. **The working set must be consistent, not only of full null space.**
   Rows within `act_tol` enter the working set as equalities, violated rows
   included.
   - On `test_extrapolated_x_warns[auglag]` the working set holds five
     support rows `h(a|xᵢ) ≥ 0` (rows 84, 100, 121, 122, 165) of rank 3,
     because the x-basis has order 2.
   - As equalities they are inconsistent: the least-squares residual is
     4.1e-7. The offset `−x³` is not representable, so at most three of them
     can be active.
   - `_restore` therefore cannot reach the boundary. Newton converges on the
     wrong manifold, and the acceptance test rejects the result.

   Decision 1's null-space step handles linearly dependent rows that are
   consistent (the 50-of-51 fixture). It does not choose among inconsistent
   ones. Restoring should solve the inequality problem
   `min ‖Δ‖ s.t. A_W(θ + Δ) ≥ b_W`, a small NNLS/QP. Only rows that end tight
   should stay in W. Without that, the polish cannot repair the infeasible
   fits that #174's restart selection returns, which is where it would help
   most.
8. **Production work the prototype skips.**
   - Recompute `mu_ineq`, `kkt_residual`, `constraint_A_ineq` and
     `converged` from the polish.
   - Land on the feasible side of each active row.
   - Use an analytic interaction Hessian once #118 lands (the
     finite-difference one costs ~5 ms per polish).
   - Catch only `InfeasibleParameterError` / `LinAlgError` in the line
     search, as Decision 4 requires. The prototype's line search catches
     every `Exception` as an infeasible trial point.
   - Report when `_restore` stops at its 8-iteration cap with the working
     rows still off. Today only the final feasibility check catches that.

## Deviations from #177's decisions that the findings do not cover

Findings 1, 2 and 6 change Decision 4 on evidence. The prototype also
departs from the decisions in three places that no measurement here
tested:

- **Row drops use min-norm least-squares multipliers, not NNLS.**
  - Decision 2 says to drop the row whose NNLS multiplier is clearly
    negative. An NNLS multiplier is never negative, so the prototype picks
    the row to drop from the unconstrained `lstsq` λ (`lam_free`). The
    sign-constrained BVLS λ (`lam_pos`) is used only for the curvature term
    and for the KKT residual.
  - **Risk:** when LICQ fails, the min-norm λ is not unique and can show
    negative entries that no valid multiplier has. That is why Decision 2
    chose NNLS.
  - **Evidence:** the six fits with drops (finding 4) are all
    `test_exponential_scaling_no_shift.py::TestGuards::test_fits_nonzero_offset`.
    That is three solvers × two offsets, dropping 5 or 9 rows each, and every
    one was accepted. The degenerate 50-of-51 fixture needed no drop.
    Degeneracy combined with a drop is therefore untested.
  - **For production:** a drop test that survives degeneracy. For example,
    drop row j only if the BVLS fit without row j has a lower residual.
- **No separate check that μ ≥ −tol.** BVLS forces λ ≥ 0 on inequality
  rows, so that check could never fail. Instead the KKT residual is computed
  with these sign-constrained λ, so a row that needs a negative multiplier
  shows up as KKT residual. That part of Decision 4 is folded into the KKT
  criterion.
- **Tolerances are normalised, and equalities are held only to 1e-9.**
  - Slack and violation use `s_i / ‖∇s_i‖`, not the raw slack.
  - Equality violation is folded into the same maximum as inequality
    violation and checked against `feas_tol = 1e-9`. Decision 4 asks for
    1e-10.
  - **Evidence:** the largest violation after an accepted polish was
    1.4e-13 suite-wide, which passes the 1e-10 rule too. The looser
    threshold changed no outcome in these runs.

## Files

| file | role |
|---|---|
| `polish.py` | the algorithm and its acceptance test |
| `harness.py` | captures `optimize` calls and rebuilds each fit's constrained problem |
| `run_cases.py` | the six #178 cases from 11 starts |
| `polish_probe.py` | pytest plugin: measure or apply the polish across the suite |
| `summarise.py` | summary of a probe log (`--baseline` diffs failing tests) |
| `compare_seeds.py` | failing tests per start, polished vs #178's unpolished sweep |
| `coxph_check.py` | the scaled Coxph reference check |

## Rerun

From the repository root:

```sh
P=src:research/active-set-polish
PYTHONPATH=$P .venv/bin/python research/active-set-polish/run_cases.py
PYTHONPATH=$P .venv/bin/python research/active-set-polish/coxph_check.py
for mode in measure apply; do
  MLTPY_POLISH_MODE=$mode MLTPY_POLISH_LOG=/tmp/polish_$mode.jsonl PYTHONPATH=$P \
    .venv/bin/python -m pytest tests/ -q -p polish_probe \
    --deselect tests/test_optimizer.py::TestInteractionDesignReuse
done
.venv/bin/python research/active-set-polish/summarise.py /tmp/polish_apply.jsonl \
  --baseline /tmp/polish_measure.jsonl
# perturbed starts with the polish applied
MLTPY_POLISH_MODE=apply PYTHONPATH=$P .venv/bin/python \
  research/auglag-start-sensitivity/run_probe.py --seeds 0-10 \
  --out /tmp/apply_seeds.jsonl tests/ -p polish_probe \
  --deselect tests/test_optimizer.py::TestInteractionDesignReuse
# against #178's unpolished sweep (its README, "Rerun")
.venv/bin/python research/active-set-polish/compare_seeds.py \
  /tmp/apply_seeds.jsonl /tmp/start-probe.jsonl
```

11 × ~35 s per sweep.

## Provenance

Library at `v0.6.0` @ `239f4ad`. Python 3.13.13, NumPy 2.5.4, SciPy 1.18.1,
macOS 26.6.2 arm64.
