# Start sensitivity of auglag R-parity fits (#178)

Evidence for [#177](https://github.com/RMKruse/mltpy/issues/177) (auglag
polish on active sets), within the scope of the optimizer audit
([#174](https://github.com/RMKruse/mltpy/issues/174)). No production code
changes.

## Answer

The whole suite (2733 tests) passes from the default start. From ten
perturbed feasible starts, **9 tests fail from some start (2–5 per start)**.
All nine compare against R (four) or check that a pin or constraint holds
(five); none fails at the default start.

Across the corpus, the polish never runs where most fits end:

- **88% of auglag fits (18,244 of 20,663) end with an active or equality
  row, and the Newton-CG polish never runs on any of them.** It runs on the
  2,151 interior fits and is accepted every time.
- **Interior polishing is not tight either.** `_ncg_polish` calls trust-ncg
  at scipy's default `gtol = 1e-4`, which leaves θ about 1e-6 relative off
  R on the polished one-hot BoxCox / Lm interacting-scaling fits.

Four R-parity cases fail CLAUDE.md's `rtol=1e-6, atol=1e-10` from some start.
`tests/test_optimizer_start_sensitivity.py` pins them as strict xfails:

| case | kind | worst `|Δθ|/(atol+rtol·|θ_R|)` over 11 starts | starts failing |
|---|---|---|---|
| `reference/auglag/exponential_n50_seed43` | active (50 of 51 rows, 3 params) | 5.0 | 1, 3, 4, 6, 7, 10 |
| `mlt_exponential_offset_*` (#160) | active (3 support rows) | 1.7 | 1, 2, 6, 7, 10 |
| `mlt_exponential_scaling_shift_*`, β pinned (#115) | active (2 support rows + pin) | 10.5 | every start except 9 (default included) |
| `stram_onehot_bs_boxcox_*` | interior, polished | 1.9 | 2, 7, 9 |
| controls: `reference/auglag/lm_n50_seed42`; #115 free fit | interior / active | ≤ 1 | none |

Ratio > 1 fails. Values are from `random_state=0` and are deterministic.

## The 9 start-dependent tests

Seeds are the starts `k` at which the existing test fails (it passes at
`k = 0` in every case).

| test | failing k | kind | what fails |
|---|---|---|---|
| `test_exponential_offset_support::TestPerRowOffsetRReference::test_fit_reaches_r_optimum[auglag]` | 1, 2, 6, 7, 10 | R θ, active | θ 1.7e-6 relative off R; KKT 1.4e-4 to 8.6e-4 with `converged=True` |
| `test_r_parity_auglag::test_auglag_exponential_theta_matches_r` | 2, 4, 5, 7, 9 | R θ, degenerate active | entries R has at exactly 0 come out at ±3.6e-10 against `atol=1e-10`; 50 of 51 rows active |
| `test_tram_boxcox_interacting_scaling::TestRParity::test_theta_matches_r[auglag]` | 2, 7, 9 | R θ, interior | 1.3e-6 relative; polish ran at `gtol=1e-4` |
| `test_tram_lm_interacting_scaling::TestRParity::test_quantile_matches_r[auglag]` | 3 | R prediction, interior | quantile 1.3e-6 relative; polish ran at `gtol=1e-4` |
| `test_exponential_scaling_shift::TestComposition::test_fixed_beta_at_optimum_reproduces_fit` | 4, 6 | equality pin | β pin holds only to 1.2e-8 against the test's 1e-8 (θ itself is 7–10× the R tolerance at every start; the test checks θ at 5e-5 since #165) |
| `test_optimizer::TestFixedParamsAuglag::test_fixed_params_pins_theta_index_to_value` | 7 | equality pin | pin off by 4.2e-8 against 1e-8 |
| `test_model::test_integration_r_reference_new_distributions[exponential-…]` | 3, 8, 10 | support feasibility | `h(a) = −4.0e-9`, test requires `≥ 0` |
| `test_interaction_basis_bernstein::TestBernsteinBernsteinFit::test_fit_columns_monotone` | 1 | monotonicity feasibility | column difference −3.8e-8 |
| `test_interaction_exponential_support::TestPredictSupportWarning::test_extrapolated_x_warns[auglag]` | 1, 2, 4–8, 10 | support feasibility | in-sample `h(a|x_i)` below `−1e-7`; see finding 5 |

The four pin/feasibility rows share one cause: auglag holds rows only to
`feas_tol = 1e-7`, and nothing afterwards puts the active rows exactly on the
boundary.

## Findings for #177

1. **The polish gate excludes almost every fit.** 18,244 of 20,663 fits end
   with an active or equality row. Polishing on the active set is the main
   case, not an edge case.
2. **There is no clean slack gap at 1e-6.** The largest active slack is
   9.1e-7 and the smallest inactive slack is 1.05e-6. 64 fits have an
   inactive slack in (1e-6, 1e-4); histogram of each constrained fit's
   smallest inactive slack, by decade: 1e-6: 47, 1e-4: 121, 1e-3: 247,
   1e-2: 1248, 1e-1: 10429, 1e0: 6108. The fits in the band are the
   poorly-solved ones:
   - the scaled Cox fits, including the R-parity
     `test_coxph_theta_beta_gamma_match_R`: `converged=True` at **KKT 2.7e-2**,
     nearest rows at 1.6e-5 to 2.6e-5;
   - the interaction exponential fit: every attempt `converged=False`, KKT
     2e-4 to 7e-4, slacks a continuum from 1e-12 to 4e-6.

   #177 Decision 2's rule "skip the polish when a slack falls in
   (1e-6, 1e-4)" would therefore skip exactly the fits with most to gain.
   **Recommendation:** replace it by a short primal active-set loop. Start
   from slack ≤ 1e-6. When a Newton step would cross an inactive row, stop
   at the row and add it (ratio test). Drop the row with the most negative
   NNLS multiplier. Cap at a few iterations. The acceptance test stays the
   safety net.
3. **Degenerate active sets are real and common.**
   `exponential_n50_seed43` has 50 active rows in 3 parameters. This confirms
   the null-space choice (Decision 1): KKT Newton would face a singular
   saddle-point matrix there.
4. **The interior polish needs a gradient tolerance.** `_ncg_polish` passes
   no options, so trust-ncg stops at `gtol = 1e-4`. #177's acceptance item
   "interior fits unchanged" should become "interior fits polished to a
   stated tolerance". The new polish should set `gtol` explicitly, e.g.
   1e-8 or relative to the gradient scale.
5. **Out of #177's scope, for #174:**
   - **Restart candidate selection** keeps the `converged=False` attempt
     with the best verified NLL, even when it violates a support row by
     2.4e-6 (24 × `feas_tol`). On the interaction exponential fit that
     attempt wins, because breaking the constraint lowers the NLL.
   - **Stall acceptance** reports `converged=True` at KKT 2.7e-2 on the
     scaled Cox fits.
   - **`test_r_parity_auglag.py`** fits with `random_state=None`, so its
     restarts draw a fresh generator on every run and its results are not
     reproducible.

## Method

`start_probe.py` is a pytest plugin. It makes no library changes.

- **`MLTPY_START_SEED=k`:** replaces attempt 0 of every auglag restart loop
  by the path's own perturb-and-project step applied `k` times to the
  default start. Shift fits use `_perturb_and_project` (θ_b + N(0, 0.1),
  sorted and lifted to the support floor); interaction fits use the
  column-wise analogue. The draws come from the fit's own `random_state`
  generator.
- **Unchanged:** restarts, tolerances, the polish and result selection.
- **Logged per auglag call, at auglag's returned point in its own
  coordinates:** KKT, `converged`, equality rows, and every inequality row's
  normalised slack (`(a·x − b)/‖a‖`; box bounds of the increment
  coordinates; `g/‖∇g‖` for the nonlinear support rows). Also whether the
  polish ran and was accepted, and each test's outcome.

`run_probe.py` runs pytest once per seed. `compact.py` reduces the 14 MB raw
log to per-test aggregates (`results.json`) and prints the corpus statistics
above. `analyse.py` prints per-fit diagnostics for chosen tests.

Fits are attributed to the test whose setup or call triggered them; fits
inside module-scoped fixtures go to the first test that requests the fixture.

## Rerun

From the repository root:

```sh
PYTHONPATH=src .venv/bin/python research/auglag-start-sensitivity/run_probe.py \
  --seeds 0-10 --out /tmp/start-probe.jsonl tests/
.venv/bin/python research/auglag-start-sensitivity/compact.py \
  /tmp/start-probe.jsonl research/auglag-start-sensitivity/results.json
.venv/bin/python research/auglag-start-sensitivity/analyse.py /tmp/start-probe.jsonl \
  --nodeid "tests/test_r_parity_auglag.py::test_auglag_exponential_theta_matches_r"
```

11 × ~34 s on the machine below.

## Provenance

Library at `v0.6.0` @ `e3f3f8e`. Python 3.13.13, NumPy 2.5.4, SciPy 1.18.1,
macOS 26.6.2 arm64. R fixtures were scored as committed, not regenerated.
Tests whose own config leaves `random_state=None`
(`test_r_parity_auglag.py`) draw a fresh generator per run, so their failing
seeds in the sweep vary between runs. The new test module pins
`random_state=0`.
