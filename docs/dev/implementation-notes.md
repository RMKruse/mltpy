---
orphan: true
---

# Implementation notes

Developer-facing notes on solver internals and on how individual feature paths
grew, moved out of the agent guide (`CLAUDE.md`) on 2026-11-01. They describe the
code as of `v0.6.0`; when they disagree with the code or with an ADR, the code and
the ADR win. Design decisions live in `docs/adr/`; this file only keeps
the operational detail that has no ADR of its own.

## Solver dispatch (`optimizer.py`)

`OptimizerConfig.solver` defaults to `"auglag"`: the PHR augmented Lagrangian in
`_auglag.py`, which mirrors R `mlt`'s `alabama::auglag` for the closest
reference-implementation parity. `"slsqp"` and `"trust-constr"` remain opt-in
alternatives behind the same `OptimizerConfig`.

`optimize()` is a thin validate-and-dispatch shell. It fails fast on the base
distribution, then routes to one of three sibling helpers:

- `_optimize_auglag`: shift basis, PHR;
- `_optimize_scipy`: shift basis, SLSQP / trust-constr;
- `_optimize_interaction`: dispatches further to `_interaction_auglag` /
  `_interaction_scipy`.

All four solver paths share two helpers:

- `_restart_loop(theta_init, config, perturb_fn, run_fn, success_fn, *, label,
  message_fn)` runs the perturb-and-project attempt loop. Attempt 0 starts from
  `theta_init`, and restarts go through `perturb_fn`. A `LinAlgError` is caught
  and that attempt skipped. The attempt with the best `.fun` is kept.
- `_fallback_result(...)` builds the `converged=False` result when every attempt
  raised `LinAlgError`.

The `random_state` plumbing and the `ConvergenceWarning` semantics are shared by
all paths.

The auglag paths also report `n_outer_iter` and `kkt_residual` on
`OptimizationResult`; both are `None` for the scipy solvers. The shift-basis
auglag path passes `message_fn`, so its per-attempt "did not converge" warning
fires; the interaction paths omit it.

`OptimizerConfig.max_iter` and `.tol` apply only to SLSQP / trust-constr; auglag
has its own outer and inner budgets in `AugLagOptions`.

Auglag can report `converged=False` with strict KKT residuals around 1e-5 on
degenerate active sets, for example stacked monotonicity boundaries, even when
θ matches R to many decimals. Treat the flag as informational and check
`log_likelihood` / Δθ when triaging (ADR 0004).

The shift-basis optimiser starts from `np.linspace(0, 1, p)`, which is already
feasible. On failure it restarts from perturbed points projected back onto the
feasible set.

## Censoring dispatch (`likelihood.py`)

Both `likelihood.py` and `optimizer.py` dispatch on `CensoringType`. The private
`_ll_*` and `_grad_*` functions in `likelihood.py` each handle one censoring
type. `negative_log_likelihood(gradient=True)` returns `(nll, grad)`, which scipy
uses directly as the `jac=True` objective.

## tram convenience layer (`tram.py`)

`BoxCox`, `Coxph`, `Lehmann`, `Colr`, `Lm` and `Survreg` are thin `MLT` subclasses
built on `_TramModel`; the survival models also mix in `_SurvivalMixin`. Each
hard-codes its censoring and base distribution and adds domain methods such as
`fitted_transformation`, `survival` and `hazard`.

`plot()` needs `pip install 'mltpy[plots]'`. It accepts a 2-tuple
`(ax_cdf, ax_pdf)`, a single `Axes` (CDF only), or `None`.

### Polr

`Polr` (proportional-odds ordinal regression) does **not** go through `MLT`. Its
basis depends on the response level count `K`, which is only known at fit time,
so it subclasses `ConditionalTransformationModel` directly and defers `__init__`
until `fit()` has resolved `K`.

It uses `OrdinalBasis(K)`, a degenerate one-hot cutpoint basis, together with
`CensoringType.INTERVAL`. Level-1 rows have `lower=-inf` and level-K rows have
`upper=+inf`; the interval-censored likelihood handles these through an internal
mask split.

Sign convention: mltpy parameterises `h + X·beta`, so `Polr.coef_` is the
**negative** of R `tram::Polr`'s `beta`. Use `predict_proba` / `predict_class`
instead of `predict()`.

### Confidence intervals

`confint(level=0.95)` is the Wald default.

`confint(level=0.95, type="profile")` inverts the χ²₁ likelihood-ratio test. It
refits under `OptimizerConfig.fixed_params={j: v}` and uses brentq to find each
bracket endpoint. A profile interval costs roughly 10× a Wald call per parameter;
pass `parm=[j]` to limit it to selected parameters.

Use profile intervals whenever a Bernstein coefficient sits on the monotonicity
boundary. There the bread `−H⁻¹` is near-singular along the active row, so Wald
widths can be 3–5× the truth, and the HC0 `sandwich_se()` inherits the same bad
bread. The worked comparison, covering the Coxph fixture, all three CI flavours
and a profile log-likelihood sweep, is `docs/examples/06_profile_likelihood.ipynb`.

The inner profile solve fails when:

- the bracket does not bracket;
- an equality constraint conflicts at the pinned value;
- the KKT residual is at least `_PROFILE_INNER_KKT_THRESHOLD = 1e-1`.

How a failure surfaces depends on `parm`:

- `parm is None`: it downgrades to a `ConvergenceWarning` and the interval
  endpoint becomes `±inf` or `nan`.
- `parm` is an explicit sequence: it is re-raised as a `RuntimeError`.

## Scaling terms (ADR 0002, ADR 0003)

`BoxCox`, `Coxph`, `Colr`, `Lm` and `Survreg` accept `scaling=`, a pre-built
`(n, q_s)` design. This mirrors R `tram::*(scale = ~x_s)`. `Polr` does not accept
it.

- **`gamma_`** is the single γ accessor across the hierarchy. It raises
  `NotFittedError` before fit and `ValueError` without `scaling=`, the same
  contract as `coef_`, `sigma_` and `intercept_`. It replaces the older,
  `None`-returning `gamma_coef_`, which has been removed. The scaling column
  names are in `feature_names_scaling_`.
- **`summary()`** adds a *Scaling coefficients* block with Wald SEs.
- **Prediction:** predict-side methods take a parallel `X_scale_new=` of shape
  `(m, q_s)`; `survival` and `hazard` call it `X_scale=`.
- **Worked notebooks:** `docs/examples/05_scaling_terms.ipynb` (shift basis) and
  `docs/examples/07_stratified_heteroskedastic.ipynb` (with `InteractionBasis`:
  LR test on γ, `Theta_` / `gamma_`, prediction with `X_scale_new`, R `stram`
  parity).
- **Positive logLik on censored scaled fits (#162):** check the data first.
  `CensoredData.right_censored(y, censored)` and `left_censored` take a
  *censored* flag, the opposite of R `Surv(y, event)`. Passing the event
  indicator ties every exact row at the censoring time, and the likelihood
  becomes unbounded; R fits that data to a positive logLik too. The `±30` clip
  on `h` was ruled out as the cause (ADR 0005, #162 amendment; fixtures
  `reference/scaling_fixed_time_*`).

### Scaling with `InteractionBasis` (issue #103, ADR 0003)

R `tram` supports this combination (formula `y | s ~ 1 | z`, class `stram`;
γ-parity uses the raw `coef(fit)[scalecoef]`) but flags it "highly experimental".
mltpy therefore treats it as supported but experimental and emits a `UserWarning`
at fit.

- **Scope:** implemented for exact data only. Censored data on this path is out
  of scope, and `base_distribution="exponential"` stays rejected (ADR 0003
  Decision 5).
- **Parameter layout:** `theta_ = [vec_C(Θ) | γ]` with no β block. `Theta_`
  reshapes the leading `p·q` block and `gamma_` returns the trailing γ block.
- **γ standard errors** come from the finite-difference
  `_hessian_interaction_fd`.
- **Lifted guards:** the guards in `MLT.__init__`, `optimize()`, the two
  likelihood dispatchers, `hessian` and `score_matrix` no longer block this path.
- **R parity:** the `stram_bs_bs_normal_*` fixture is wired into
  `reference/generate_reference.R`. Its ratchet test,
  `tests/test_interaction_scaling.py::TestStramRParity`, skips until the `.txt`
  fixtures have been generated under R.

## `base_distribution="exponential"` support rule (ADR 0005)

The rule is `h(lower | x_i) ≥ −offset_i` for every training row. ADR 0005 and its
amendments (#113–#164) are the record; the notes below are a quick map of where
the code enforces the rule on each path.

### Interval rows below the support (#164)

A both-finite interval row uses R's `pexp`, `F(h) = 0` for `h ≤ 0`:
`log P = log F(h_u)` when `h_l < 0 < h_u`, and `−inf` when `h_u ≤ 0`.
`_log_diff_ndtr` applies the value, and `_pair_density_weights` zeroes an
endpoint's Mills weight below `−_EXPON_SUPPORT_TOL`. Both helpers serve every
both-finite block (likelihood, gradient, Hessian, scores, truncation), so no
kernel changes. Within `1e-7` of the boundary the weight keeps its
feasible-side value. Without that, SLSQP iterates sitting on an active support
row to round-off saw the kink at `h_l = 0` and stopped early. Exact rows,
right-censored rows and `DistOps` keep the `−h` extension.

### Shift basis, no scaling (#160)

`build_constraint_matrices` / `build_constraints(offset=...)` add the rows
`θ_b[0] + x_i·β ≥ −offset_i`. Without covariates this collapses to a single row,
`θ_b[0] ≥ max_i(−offset_i)`. Starts and restarts lift `θ_b[0]` to
`max(0, max_i(−offset_i))`.

### Scale-only, zero or no offset (#161)

The rule collapses to the linear row `θ_b[0] ≥ 0`, and the fit works on all three
solvers.

Under the smooth extension `log f(h) = −h`, the auglag subproblem is unbounded
below once `θ_b[0] < 0`. Auglag therefore enforces the row as a hard bound and
runs through `_increment_auglag` (`support_nl=None`, `intercept_lower=b`):

- `θ_b[0] ≥ 0` bounds `δ_0`;
- `D θ_b ≥ 0` bounds each `δ_k ≥ 0`.

These were previously PHR rows only. Censored rows in a y-region that no exact
row covers then drove the NLL down exponentially in γ.

Bound multipliers `ν_lo` fold into `mu_ineq`. Unscaled and non-exponential fits
keep the solve in θ coordinates.

### Scaling with a shift block or a non-zero offset (#115)

The rule is nonlinear in γ: `θ_b[0]·exp(0.5·x_s,i·γ) + x_d,i·β + offset_i ≥ 0`.

- **Building the rows:** `build_constraint_matrices(..., X_scale=)` returns them
  as `ConstraintMatrices.nonlinear_ineq` (`ScaledSupportConstraint`, with an
  analytic Jacobian). At `γ = 0` they equal the linear #160 rows.
- **Auglag:** it carries them through `auglag_minimize(nonlinear_ineq=…)`.
  `mu_ineq` lists the linear rows first, then the support rows, and
  `constraint_A_ineq` reports the support rows as `J(θ̂)`.
- **Increment coordinates:** `_increment_auglag` solves in
  `δ = (θ_b[0], Δθ_b, β, γ)`, so `D θ_b ≥ 0` is a hard box in every inner solve.
  Without this, censored rows (which have no `log h'` barrier) let the NLL fall
  exponentially in γ once monotonicity is violated.
- **Other solvers (#116):** `build_constraints(..., X_scale=)` passes the same
  rows natively: an SLSQP `ineq` dict, or a trust-constr
  `NonlinearConstraint` with `jac=J` and the closed-form
  `hess=ScaledSupportConstraint.hessian`. The BFGS default for the constraint
  Hessian stalled and warned `delta_grad == 0.0`. SLSQP's θ stops about 3e-5
  relative short along the flat `θ_b[1:]` direction, so tests compare it at
  1e-4. Its first attempt from the default start overflows `exp(0.5·x_s·γ)`
  and is recovered by a restart, as for the unscaled exponential shift fit.
- **R parity:** R `mlt` has no support rule, so the parity fixture
  `mlt_exponential_scaling_shift_*` is an explicit-constraint `alabama::auglag`
  reference.

`MLT.__init__` rejects scaling + exponential only for `InteractionBasis`.

### `InteractionBasis`, no scaling, exact data (#158)

`build_constraint_matrices_interaction(nonneg_lower=True, X=..., offset=...)`
appends one linear row per training observation below the Kronecker rows:

```
(a(a) ⊗ b(x_i))ᵀ vec(Θ) ≥ −offset_i
```

All three interaction solvers honour the non-zero `b_ineq`.

Because `b(x)` is nonlinear in `x`, these rows do not cover new `x` values, even
inside the training hull. Interaction prediction therefore runs the #157
`h(a|x) < 0` warning. `_predict_quantile_interaction` uses the shared
`_warn_quantile_saturation` (with support-atom wording), which `simulate()`
drops.
