# ADR 0005 — Exponential Support Rule and the Nonlinear Scaling Feasibility Constraint

**Date:** 2026-11-01  
**Status:** Accepted (supersedes [ADR 0002](0002-scaling-terms.md) Decision 3)  
**Deciders:** RMKruse  
**Issue:** [#113 ADR 0005: exponential support rule and nonlinear scaling feasibility constraint](https://github.com/RMKruse/mltpy/issues/113)  
**Parent:** [#117 Map: scaling= with base_distribution="exponential"](https://github.com/RMKruse/mltpy/issues/117)  
**Inputs:** [#126 Can R mlt fit an exponential-link CTM with a scaling term?](https://github.com/RMKruse/mltpy/issues/126) (findings in `research/exp-scaling-r-route/README.md`), [#155 Exponential link: lower-support-end or observed-point support rule?](https://github.com/RMKruse/mltpy/issues/155)

---

## Context

The exponential base distribution (`base_distribution="exponential"`, rate 1)
has support `[0, ∞)`: `F_Z(z) = 1 − e^{−z}` for `z ≥ 0`, and the
log-density `log f_Z(z) = −z` is only meaningful for `z ≥ 0`. A transformation
model `P(Y ≤ y | x) = F_Z(h(y | x))` built on it is a proper distribution only
if `h(y | x)` never goes negative on the response support. This is the one
base distribution that needs a **support constraint** on top of the Bernstein
monotonicity rows `D θ_b ≥ 0`.

Notation (shift basis; the unscaled `InteractionBasis` path is covered by the
[#158 amendment](#amendment-2026-11-01-158--unscaled-interactionbasis-path)):

- `[a, b]`: the Bernstein support of the response basis. Only the first
  Bernstein polynomial is non-zero at `a`, and it equals 1 there, so
  `h_0(a) = θ_b[0]`.
- `x_d,i` (length `q_d`) and `x_s,i` (length `q_s`): the shift and scaling
  design rows of training observation `i = 1, …, n`.
- `f_i = exp(0.5 · x_s,i · γ) > 0`: the scaling factor. ADR 0002 Decision 4
  fixes the `0.5`.
- `h(y | x_i) = h_0(y) · f_i + x_d,i · β`, with `θ = [θ_b | β | γ]`.

Because `h_0` is non-decreasing, `min_{y ∈ [a,b]} h(y | x_i) = h(a | x_i)`, so

    h(a | x_i) = θ_b[0] · f_i + x_d,i · β.

**Today.** The unscaled path enforces `θ_b[0] + x_d,i · β ≥ 0` for every
training row as linear inequality rows (the `nonneg_lower=True` branch of
`build_constraints` in `src/mltpy/constraints.py`). ADR 0002 Decision 3
rejected `scaling=` with the exponential link outright (`ValueError` at
`MLT.__init__`) because under scaling that row becomes **nonlinear in `γ`**,
and `_auglag.py` only accepts linear rows `A_ineq θ ≥ b_ineq`. The likelihood
side was never the obstacle: the scaled log-likelihood, gradient, Hessian, all
four censoring branches and `predict` already handle the exponential link.

**What R does** ([#126](https://github.com/RMKruse/mltpy/issues/126), mlt
1.8-4 / tram 1.4-7):

- `ctm()` / `tram::tram(..., distribution = "Exponential")` accept a scaling
  term with no restriction.
- R `mlt` has **no** support constraint. Its only `ui`/`ci` rows are the
  Bernstein monotonicity rows. Support is kept implicitly by `dexp` returning
  `−Inf` for `h < 0`, which acts at the observed points (`h(y_i | x_i) ≥ 0`),
  not at the lower support end.
- On shift + exponential (scaled or not), `mlt()` stalls against that `−Inf`
  barrier and still reports `convergence = 0`. On the #126 fixture it stops at
  logLik −185.45 with `h(a | x_i) < 0` on 147 of 150 rows, while the optimum
  under R's own rule (found with an explicit-constraint `alabama::auglag`) is
  −28.12.
- Scale-only (`q_d = 0`) fits cleanly: `mlt(fixed = c("Bs1(y)" = 0))` and
  `tram::Aareg(y ~ 0 | x_s)` agree with mltpy's rule to logLik 2e-11 and θ
  8.6e-6.
- Coefficient order `Bs1(y)…Bs{p}(y), sft_*, scl_*` is exactly `[θ_b | β | γ]`.
  Raw `ctm` shifts by `+x_d · β`, the same sign as mltpy (only `tram::*` flips
  it), and scales by `exp(0.5 · x_s · γ)`.

Two decisions were therefore open: **which support rule** the exponential link
enforces, and **how** that rule is enforced once it is nonlinear in `γ`.

---

## Decision 1 — Support Rule: Lower-Support-End

**Chosen:** the exponential link enforces the **lower-support-end rule**

    h(a | x_i) ≥ 0   for every training row i,

for the **whole** exponential link, scaled and unscaled. On the unscaled path
this ratifies the existing `nonneg_lower` rows with no code change.

**Why.** It is the only rule of the three that gives a proper distribution on
a fixed, parameter-free support `[a, ∞)` for every training `x` while still
allowing shift terms. The parameter space stays regular: the support does not
depend on θ, so Wald standard errors, profile CIs and LR tests keep their
usual asymptotics, with the ordinary boundary caveat when the row is active.
In the scale-only case it gives exact R parity.

**Rejected — R `mlt`'s implicit observed-point rule `h(y_i | x_i) ≥ 0`.**

- It moves the support boundary onto the data. At the MLE, `h = 0` lands on
  some observed `y_i`, just as the shifted-exponential MLE puts the threshold
  at `min(y)`. The support then depends on θ, which makes this a non-regular
  threshold model: estimates converge at rate `n`, not `√n`, and Wald SEs,
  profile CIs and LR tests do not hold in those directions.
- Unclamped, `F(y | x)` is negative on `[a, y₀(x))`, where `y₀(x)` solves
  `h(y₀ | x) = 0`.
- R only follows this rule implicitly, and its optimiser never reaches the
  optimum (it stalls and reports convergence). So **"R is ground truth" does
  not apply on shift + exponential.** This is a recorded, deliberate
  divergence from R `mlt`, not a bug to fix later.

**Rejected — the anchored rule `h(a | x) = 0` (as in `tram::Aareg`).** It
forces `β = 0` whenever `x_d` varies, so it rules out shift terms, which the
map's destination requires.

**Known cost.** The fit pushes `h(a | x_i)` toward 0 on every row, which a
varying `x_d` can only satisfy with `β = 0`. On the #126 shift + scale fixture
a shift covariate that takes both signs ends at `β = 0` (logLik −93.65, against
−28.12 under the observed-point rule). That is the price of a fixed support,
and it is accepted.

**Support atom (documented feature, not a defect).** When `h(a | x) > 0`, the
model puts point mass

    F(a | x) = 1 − e^{−h(a | x)}

on `y = a`. In cumulative-hazard terms a constant shift is a jump in Λ at `a`.

- `predict(what="cdf")` and `predict(what="quantile")` already handle the atom
  consistently: quantiles for `p < F(a | x)` saturate at `a`.
- **Density caveat:** an exact observation with `y_i = a` is scored with the
  density, not the atom mass. This is left as is (it has probability zero
  under a continuous design) and is documented.

---

## Decision 2 — Enforcement: The Exact Nonlinear Constraint

**Chosen (Option 1):** enforce the lower-support-end rule under scaling
**exactly**, as one smooth nonlinear inequality per training row:

    g_i(θ) = θ_b[0] · f_i + x_d,i · β ≥ 0,    f_i = exp(0.5 · x_s,i · γ),
    i = 1, …, n.

The monotonicity rows `D θ_b ≥ 0` stay linear and unchanged. Only the support
block becomes nonlinear. `γ` remains unconstrained as far as monotonicity is
concerned (ADR 0002 Decision 3's monotonicity argument still holds), but on
the exponential link it now enters the support constraint.

**Constraint value and Jacobian.** Row `i` of `J(θ) = ∂g/∂θ`, in the
`[θ_b | β | γ]` layout:

| Block | Entry |
|---|---|
| `∂g_i / ∂θ_b[0]` | `f_i` |
| `∂g_i / ∂θ_b[k]`, `k ≥ 1` | `0` |
| `∂g_i / ∂β` | `x_d,i` |
| `∂g_i / ∂γ` | `0.5 · θ_b[0] · f_i · x_s,i` |

`J` is analytic and costs one `exp` per row (the same `f_i` the likelihood
already computes). For the trust-constr path the second derivatives are also
closed-form: `∂²g_i/∂γ∂γᵀ = 0.25 · θ_b[0] · f_i · x_s,iᵀ x_s,i`,
`∂²g_i/∂θ_b[0]∂γ = 0.5 · f_i · x_s,i`, and zero elsewhere.

**The feasible set is not convex** in `(θ_b[0], β, γ)` once `q_d > 0`. For
example, with scalar covariates the points `(θ_b[0], γ, x_d β) = (1, 5, −e⁵)`
and `(−1, −5, e⁻⁵)` both satisfy `g = 0`, but their midpoint gives
`g = (e⁻⁵ − e⁵)/2 < 0`. Multi-start and the existing restart loop therefore
matter more on this path than on the linear one.

**Structural simplifications.**

1. **`q_d = 0` collapses to linear.** With no shift block,
   `g_i = θ_b[0] · f_i ≥ 0 ⇔ θ_b[0] ≥ 0`, because `f_i > 0`. The whole block is
   the single linear row `θ_b[0] ≥ 0`, identical to the existing no-covariate
   support row. The scale-only case needs **no** nonlinear machinery and goes
   through the existing linear `A_ineq` path.
2. **`γ = 0` is the existing linear row.** At `γ = 0`, `f_i = 1` and
   `g_i = θ_b[0] + x_d,i · β`, which is the current unscaled support row. The
   default start (`θ_b = linspace(0, 1, p)`, so `θ_b[0] = 0`; `β = 0`; `γ = 0`)
   gives `g_i = 0` for every row: feasible, on the boundary.

**Rejected — sufficient-linear (ADR 0002 Decision 3, option 2):**
`θ_b[0] ≥ 0` and `x_d,i · β ≥ 0` for every `i`. It is linear and sufficient,
but each `x_d,i · β ≥ 0` is a half-space through the origin in β-space. For any
design whose rows span `ℝ^{q_d}` in more than one direction (in particular any
covariate taking both signs), the intersection of those half-spaces is
`β ≈ 0`. The shift block collapses to zero, and the fit reports infeasibility
or a wrong optimum on cases the exact constraint fits. It is strictly worse
than the exact rule it approximates.

**Rejected — keep forbidding the combination (ADR 0002 Decision 3, option 3).**
That was a deferral, not a modelling position. The likelihood already supports
the combination, the constraint is smooth with a cheap analytic Jacobian, and
the scale-only case has exact R parity.

---

## Decision 3 — Solver Mechanism

**Auglag (default).** `auglag_minimize` gains an optional **nonlinear
inequality block** `g(θ) ≥ 0` with analytic Jacobian `J(θ)`, alongside the
existing linear `A_ineq θ ≥ b_ineq` and equality rows. The PHR treatment is
the same as for linear rows, with the constant row `A_ineq[i]` replaced by
`J(θ)[i]` at the current iterate:

- penalty term and multiplier update use `g_i(θ)` in place of
  `A_ineq[i] θ − b_ineq[i]`;
- the gradient contribution is `−J(θ)ᵀ φ`, with `φ = max(0, μ − ρ g(θ))`;
- the ADR 0004 projected-stationarity / stall check takes the working-set
  normals for nonlinear rows from `J(θ)` at the current iterate.

This mirrors `alabama::auglag`, the solver R `mlt` uses, whose `hin` is a
general function with Jacobian `hin.jac`. R `mlt` simply only passes linear
rows to it. When `q_d = 0` (simplification 1) the block is empty and auglag
runs exactly as today.

**SLSQP.** The block is passed natively as
`{"type": "ineq", "fun": g, "jac": J}` next to the linear rows.

**trust-constr.** The block is passed natively as
`NonlinearConstraint(g, 0, ∞, jac=J)` next to the linear rows. Whether to pass
the closed-form constraint Hessian above or a quasi-Newton approximation is
decided in the slice that lands trust-constr support.

**Scope.** Shift basis only, all four censoring types. `g` does not depend on
`y`, so censoring changes nothing about the constraint.
`InteractionBasis` + scaling + exponential stays rejected (ADR 0003
Decision 5 is unchanged): per-column support rows there would need their own
decision.

---

## Decision 4 — Validation Route

CLAUDE.md's "R is ground truth" principle is applied where R has a usable
answer and replaced by an explicit-constraint reference where it does not.

- **Scale-only (`q_d = 0`): R parity.** Reference fits from
  `mlt(fixed = c("Bs1(y)" = 0))` and/or `tram::Aareg(y ~ 0 | x_s)`, with tight
  optimiser tolerances on both sides. The support row is active at the optimum
  (`θ_b[0] = 0`), so θ is compared at the active-boundary tolerance observed
  in #126 (≈1e-5) and logLik at the usual tight tolerance. Any relaxation of
  CLAUDE.md's `rtol=1e-6` is stated in the test with this reason.
- **Shift + scale: explicit-constraint reference, not R `mlt()`.**
  `alabama::auglag` driven by mlt's own log-likelihood and score, with mltpy's
  nonlinear `hin` rows `g_i(θ)` written out (`research/exp-scaling-r-route/README.md`
  §9), run multi-start, on a fixture whose active set is **not degenerate**.
  The #126 fixture under this rule lands on `θ_b[0] = β = 0`, `θ_b1 = θ_b2`,
  so it is not suitable. This is cross-checked by analytic-vs-finite-difference
  gradient and Jacobian tests and a KKT residual check. It validates mltpy's
  optimiser on mltpy's model, not R's modelling choice.
- **Unscaled shift + exponential:** the fixture
  `reference/mlt_exponential_*.txt` came from a stalled `mlt()` fit (logLik
  −154.91, true optimum −7.51 under this rule), and its test only checked
  `≥ R − 0.5`. [#156](https://github.com/RMKruse/mltpy/issues/156)
  regenerated it under this rule as a multi-start explicit-constraint
  `alabama::auglag` fit on mlt's likelihood (logLik −7.5084028, cross-checked
  by `mlt(fixed = c("Bs1(y)" = 0))`), and the test now compares logLik and θ.

---

## Amendment (2026-11-01, [#158](https://github.com/RMKruse/mltpy/issues/158)) — Unscaled `InteractionBasis` Path

**Problem.** `InteractionBasis` with `base_distribution="exponential"` and no
`scaling=` was accepted, but the interaction fit enforced only the Kronecker
monotonicity rows `(D ⊗ I_q) vec(Θ) ≥ 0`. The exact-data log-likelihood
`Σ −h(y_i|x_i) + log h'(y_i|x_i)` is then unbounded above, and the fit ran off
to `h → −∞` (logLik ≈ +8·10³, `Θ` ≈ −3·10⁴ on the issue's reproduction).

**Decision.** Decision 1's lower-support-end rule applies unchanged on the
unscaled interaction path. With `h(y|x) = (a(y) ⊗ b(x))ᵀ vec(Θ)` and
column-wise non-decreasing `Θ`, `b(x) ≥ 0` makes `h(·|x)` non-decreasing in
`y`, so `min_y h(y|x_i) = h(a|x_i)` and the rule is one **linear** row per
training observation:

    (a(a) ⊗ b(x_i))ᵀ vec(Θ) ≥ −offset_i,    i = 1, …, n.

For a Bernstein y-basis `a(a) = e_0`, so the row reads `Θ[0, :] · b(x_i) ≥
−offset_i`. `build_constraint_matrices_interaction(nonneg_lower=True, X=…,
offset=…)` appends these rows below the Kronecker block, and all three solvers
(auglag, SLSQP, trust-constr) consume `A_ineq θ ≥ b_ineq` with the non-zero
right-hand side. The start point and every perturb-and-project restart are
lifted column-wise so `a(a)ᵀ Θ[:, j] ≥ max(0, max_i(−offset_i))`, which (the
x-basis being a non-negative partition of unity) makes every support row
feasible.

**Offset.** These rows include the offset in their right-hand side, so they
enforce exactly `h(a|x_i) ≥ 0` with `h` as the likelihood evaluates it. At
the time, the shift path's `nonneg_lower` rows `θ_b[0] + x_d,i · β ≥ 0` did not
include the offset; that was left out of scope for #158 and closed by the
[#160 amendment](#amendment-2026-11-01-160--offset-on-the-shift-path) below.

**Rejected — the per-column variant `Θ[0, j] ≥ 0` (`q` rows).** Because
`h(a|x) = Σ_j Θ[0, j] b_j(x)` is a convex combination of `Θ[0, :]`, these rows
guarantee `h(a|x) ≥ 0` on the whole x-support and would make a predict-side
check unnecessary. They are, however, sufficient and not necessary for a
Bernstein x-basis, so the fit would be stricter than the rule Decision 1
fixes for the rest of the link. The training-row rule was kept so that the
exponential link has one support rule on every path. In practice the two
often coincide: with exact data and no offset the likelihood presses every
`h(a|x_i)` to 0, which pins `Θ[0, :] = 0` once the `b(x_i)` span `ℝ^q`. They
differ under offsets or under-determined designs.

**Prediction.** `b(x)` is non-linear in `x`, so the training rows do **not**
guarantee `h(a|x) ≥ 0` between training points, unlike the unscaled shift
path, where `h(a|x)` is affine in `x`. `predict` on the interaction path
therefore runs the [#157](https://github.com/RMKruse/mltpy/issues/157) check
for every `what`, computing `h(a|x_new)` from the design at `y = a` (offset
included). It emits the same `UserWarning` (rows with `h(a|x) < −1e-7`) and
returns the values unclamped. `_predict_quantile_interaction` reports
`p < F(a|x)` through the shared `_warn_quantile_saturation` helper with the
support-atom wording.

**Validation.** R `mlt` has no support rule (Decision 1), so there is no R
parity fixture. Tests (`tests/test_interaction_exponential_support.py`) check
the constraint rows, training-row feasibility for every solver, agreement
between solvers, the offset, in-hull and extrapolated predict warnings, and
the quantile wording.

**Scope.** Exact data only, as for every interaction fit. `scaling=` +
`InteractionBasis` + exponential stays rejected (ADR 0003 Decision 5): under
scaling the row is non-linear in `γ`.

---

## Amendment (2026-11-01, [#160](https://github.com/RMKruse/mltpy/issues/160)) — Offset on the Shift Path

**Problem.** Decision 1's `h` is the transformation as the likelihood
evaluates it, so on the unscaled shift path
`h(a|x_i) = θ_b[0] + x_d,i · β + offset_i`. The shift-path `nonneg_lower` rows
were `[1, 0, …, 0 | x_d,i] · θ ≥ 0` with a zero right-hand side, i.e. they
enforced `θ_b[0] + x_d,i · β ≥ 0` without the offset. A negative offset left
the support on every training row: logLik was inflated by the smooth
extension `log f = −h`, and the in-sample CDF at `a` was negative while the
fit reported `converged=True`. A positive offset over-constrained the fit,
pinning it at `h(a|x_i) = offset_i` with a spurious atom of mass
`1 − e^{−offset_i}` at `a`.

**Decision.** The shift-path support rows take the offset into their
right-hand side, matching the interaction rows of #158:

    θ_b[0] + x_d,i · β ≥ −offset_i,    i = 1, …, n;

without covariates they collapse to the single row
`θ_b[0] ≥ max_i(−offset_i)`. `_assemble_constraint_arrays` returns the
right-hand side next to the rows, and `build_constraint_matrices` /
`build_constraints` accept `offset=`. auglag consumes `b_ineq`, the SLSQP
dict returns `S θ − b`, and trust-constr uses `lb = b`. The `fixed_params`
reductions carry the non-zero right-hand side through unchanged. The start
point and every perturb-and-project restart lift `θ_b[0]` to
`max(0, max_i(−offset_i))`, which makes every row feasible at `β = 0`. The
auglag Newton-CG polish now checks `A_ineq θ ≥ b_ineq` (support rows
included) instead of monotonicity alone. With `offset=None` the right-hand
side is zero and the lift is the old `θ_b[0] ≥ 0`, so existing fits are
unchanged.

**Validation.** R `mlt` has no support rule (Decision 1), so the reference
(`reference/mlt_exponential_offset_*`, generated in
`reference/generate_reference.R`) is an explicit-constraint
`alabama::auglag` fit on mlt's likelihood (checked against `mlt`'s
`loglik`/`scorei` with `offset=`) with `hin` rows
`θ_b[0] + x_i · β + offset_i ≥ 0` and `D θ_b ≥ 0`. It uses 10 feasible
starts and is then polished by equality-constrained Newton on the common
active set (stationarity ~1e-14). The fixture is cross-checked
independently: with `θ_b = θ_0 + (0, δ)` the problem separates into an LP in
`(θ_0, β)`, whose solution is the upper-hull edge of the points
`(x_i, −offset_i)`, and the no-covariate problem in `δ`, solved by
`mlt(fixed = c("Bs1(y)" = 0))`. `tests/test_exponential_offset_support.py`
compares all three solvers with it. It also covers the issue's
reproduction (a constant offset shifts `θ_b` by exactly `−offset` at
unchanged logLik), the no-covariate row, `offset=None` / zero parity,
`fixed_params`, and silence of the #157 warning in-sample.

**Solver note.** A constant offset, like `offset=None`, makes every support
row active at the optimum (`β = 0`). This is a fully degenerate active set.
auglag resolves it to ~1e-9. trust-constr stops with ~1e-6 interior slack,
and SLSQP can end on its line-search failure (reported `converged=False`).

**Scaled shift path (future).** When scaling + exponential lands (Decision 2),
the nonlinear rows need the same right-hand side:
`g_i(θ) = θ_b[0] · f_i + x_d,i · β ≥ −offset_i`.

---

## Consequences

- **ADR 0002 Decision 3 is superseded** for its exponential corner case. Its
  monotonicity argument (`D θ_b ≥ 0` suffices, `γ` free) still holds.
- **Guards.** On the shift path the only rejection is the `ValueError` in
  `MLT.__init__` (`src/mltpy/model.py`). It is lifted in the implementation
  slices: the scale-only case via the linear collapse first, then shift + scale
  via the nonlinear block. The `InteractionBasis` rejection in `optimize()`
  stays, and `MLT.__init__` keeps rejecting that combination.
- **Unscaled exponential path:** no code change. Decision 1 ratifies the
  existing `nonneg_lower` rows. The unscaled `InteractionBasis` path gained
  the equivalent rows in [#158](https://github.com/RMKruse/mltpy/issues/158)
  (see the amendment above), and
  [#160](https://github.com/RMKruse/mltpy/issues/160) moved the shift-path
  rows' right-hand side to `−offset_i`.
- **Prediction at new `x`.** The training rows guarantee `h(a | x) ≥ 0` only
  at training `x`. Unscaled, that extends to the convex hull of training `X`.
  Scaled, `θ_b[0] · e^{0.5 · x_s · γ} + x_d · β` is convex in `x_s` whenever
  `θ_b[0] > 0`, so it can dip below 0 inside the hull. `predict` warns with a `UserWarning` and returns
  the raw values, with no clamping and no raising
  ([#157](https://github.com/RMKruse/mltpy/issues/157)).
- **Quantile saturation wording.** For `p < F(a | x)` the quantile saturates
  at `a` because of the support atom. The existing saturation warning says the
  value is "not the true asymptotic limit"; for this case that wording is
  wrong and is reworded.
- **Restarts.** `_perturb_and_project` perturbs `θ_b`, projects it onto the
  linear set and shifts `θ_b[0] ≥ 0`, but leaves `β` and `γ` untouched. Once
  `γ ≠ 0` and `β ≠ 0`, a restart point can violate `g`. PHR tolerates
  infeasible starts, so this is not a correctness issue. Whether a
  nonlinear-aware projection is worth adding is decided in the shift-block
  slice.
- **Inference at an active support row.** When `g_i = 0` binds at the optimum
  (always in the scale-only case), Wald widths are unreliable along the active
  direction, as with active monotonicity rows. `confint(type="profile")` is
  the recommended diagnostic.
- **Exponential + scaling remains incompatible with `InteractionBasis`**
  (ADR 0003 Decision 5).
