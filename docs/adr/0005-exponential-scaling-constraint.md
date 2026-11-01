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

## Amendment (2026-11-01, [#114](https://github.com/RMKruse/mltpy/issues/114)) — Scale-Only Path and the Box-Constrained Inner Solve

**Implemented.** Simplification 1 of Decision 2 is in place. With scaling,
no shift block (`q_d = 0`) and a zero (or absent) offset, the support rule is
the single linear row `θ_b[0] ≥ 0`. `build_constraint_matrices` /
`build_constraints` emit it as `[1, 0, …, 0 | 0_{q_s}]` through the new
`q_s=` argument. `MLT.__init__` now rejects exponential + scaling only on an
`InteractionBasis` (ADR 0003 Decision 5). `optimize()` raises
`NotImplementedError` for a shift block or a non-zero offset: the row is
then `θ_b[0] ≥ −offset_i · e^{−0.5 · x_s,i · γ}`, non-linear in `γ`, which is
#115's nonlinear block.

**Problem: the PHR subproblem is unbounded.** mltpy's exponential
log-density is the smooth extension `log f(h) = −h` on the whole line (R
`mlt`'s score uses the same extension; its objective is `−∞` for `h < 0`).
With `h(y | x) = h_0(y) · f_i` and `θ_b[0] = −t < 0`, the negative
log-likelihood falls like `−t · Σ_i f_i`, exponentially in `γ`, while the PHR
term for the violated row grows like `ρ t² / 2`. So `min_{ℝⁿ} L_A` has no
minimiser for any finite `ρ`. On the fixture the default auglag ran off to
`θ_b[0] ≈ −20` and logLik `+2768` from a feasible start. SLSQP and
trust-constr keep the linear rows satisfied and were unaffected. The unscaled
path does not have this problem: there the objective falls only linearly in
`t`, and the quadratic penalty bounds it.

**Decision: box-constrained inner solve.** `auglag_minimize` takes an
optional box `bounds = (lb, ub)`, which every inner L-BFGS-B solve enforces.
This is the lower-level-box form of the PHR method in Birgin & Martínez
(2014), the reference `_auglag.py` already cites, whose ALGENCAN minimises
`L_A` over a box `Ω`. `min_Ω L_A` exists where `min_{ℝⁿ} L_A` does not.

- Bound multipliers are read off the augmented-Lagrangian gradient on active
  coordinates: `ν_lo,j = max(∂_j L_A, 0)` and `ν_up,j = max(−∂_j L_A, 0)`.
- The KKT stationarity uses the box-projected gradient `∇L_A − ν_lo + ν_up`.
- The ADR 0004 stall test adds the active bounds (`±e_j`) to its working set.

On the scaled exponential path the shift-path auglag passes `θ_b[0] ≥ 0` as
`lb[0] = 0` and also keeps the row in `A_ineq`, so `constraint_A_ineq` and
the model's active-constraint diagnostics still see it. That row never goes
infeasible, so its PHR multiplier stays 0. The optimizer adds `ν_lo[0]` to
the row's `mu_ineq` entry.

- **Scope.** Only the scaled exponential shift path passes a box. Every other
  path is unchanged (`bounds=None`), and its fits are byte-identical.
  *(#161: the box alone does not bound the subproblem on censored data; the
  path now solves in #115's increment coordinates, with `θ_b[0] ≥ 0` as the
  bound on `δ_0`, see the #161 amendment.)*
- **Rejected — larger `ρ₀`.** `ρ₀ = 10³` converged on the fixture, but the
  subproblem stays unbounded, so this is tuning, not a fix.
- **Rejected — a barrier on `h(y_i | x_i) < 0`** (R's `dexp = −Inf`). It
  makes the objective discontinuous and risks the line-search stalls R shows
  in #126.
- **#115.** The box covers single-coordinate rows only. The nonlinear shift +
  scale rows `θ_b[0] · f_i + x_d,i · β ≥ −offset_i` face the same
  unboundedness and need their own treatment.

**Validation.** `reference/mlt_exponential_scaling_*` (generated in
`reference/generate_reference.R`) is built in four steps:

1. Fit `mlt(ctm(Bernstein, scaling = ~x_s, todistr = "Exponential"),
   fixed = c("Bs1(y)" = 0))` with tight tolerances.
2. Polish by equality-constrained Newton on its active set, using mlt's own
   score and Hessian closures (stationarity 5.6e-15).
3. KKT-check the result in the full space: the support row (multiplier
   154.8) and `θ_b[2] = θ_b[3]` (multiplier 0.81) are active.
4. Cross-check against `tram::Aareg(y ~ 0 | x_s)` (logLik 2e-8).

`tests/test_exponential_scaling_no_shift.py` compares all three solvers with
it:

| Solver | logLik | θ | Notes |
|---|---|---|---|
| auglag | `rtol = 1e-6` | `rtol = 1e-6`, `atol = 1e-6` (Δθ ≈ 3e-7) | reproduces R's multipliers in `mu_ineq` |
| SLSQP | `rtol = 1e-6` | `1e-4` | stops early along the flat Bs4/Bs5 direction |
| trust-constr | `rtol = 1e-6` | `1e-4` | stops early along the flat Bs4/Bs5 direction |

With the polished reference, auglag's θ tolerance is tighter than the
`≈ 1e-5` anticipated in Decision 4. The tests also cover the guards,
`h(a | x_i) ≥ 0` on every training row, finite predictions, right/left/
interval censoring (auglag against trust-constr), and silence of the #157
warning under extrapolation. That warning cannot fire on this path, because
`h(a | x) = θ_b[0] · f(x) ≥ 0` for every `x`.

---

## Amendment (2026-11-01, [#115](https://github.com/RMKruse/mltpy/issues/115)) — Shift Block and Offset on auglag

**Implemented.** Decision 2's exact rule, with the #160 offset, now fits on
the default auglag solver whenever it does not collapse (a shift block,
`q_d > 0`, or a non-zero offset):

    g_i(θ) = θ_b[0] · f_i + x_d,i · β + offset_i ≥ 0,    f_i = exp(0.5 · x_s,i · γ).

- **Solver (Decision 3).** `auglag_minimize(nonlinear_ineq=g)` takes a
  callable `θ ↦ (g(θ), J(θ))` whose rows are stacked below `A_ineq`. Penalty,
  multiplier update, KKT residual, feasibility and the ADR 0004 stall test
  treat row `i` as a linear row with `A_ineq[i]` replaced by `J(θ)[i]`.
  `mu_ineq` lists the linear rows first. Without the block the solve is
  unchanged.
- **Builder.** `build_scaled_support_constraint(p, X, X_scale, offset)`
  returns a `ScaledSupportConstraint` with the Decision 2 Jacobian.
  `build_constraint_matrices(..., X_scale=)` puts it in
  `ConstraintMatrices.nonlinear_ineq` and keeps only the monotonicity rows in
  `A_ineq`. At `γ = 0` it reproduces the unscaled linear rows exactly
  (simplification 2), so the default start and the `support_floor` lift of
  #160 are feasible unchanged. The scale-only, zero-offset case keeps the
  #114 linear collapse and box.
- **Guards.** `optimize()` rejects the case only for SLSQP / trust-constr
  ([#116](https://github.com/RMKruse/mltpy/issues/116)). `fixed_params`
  equality rows compose with the block unchanged.
- **Reporting.** `constraint_A_ineq` holds the support rows as their Jacobian
  `J(θ̂)`, aligned with `mu_ineq`. The active-set covariance
  (`vcov(regularize='active')`) uses those linearised normals with the NLL
  Hessian; the curvature term `−Σ μ_i ∇²g_i` of the Lagrangian Hessian is not
  included, as for every other first-order constraint diagnostic here.

**Problem: monotonicity must hold inside the inner solve.** The PHR penalty on
`g_i = h(a|x_i)` bounds `h(·|x_i)` from below only while `h_0` is
non-decreasing (then `h(y|x_i) ≥ g_i` for every `y`). When a monotonicity row
`D θ_b ≥ 0` is violated, `h_0(y)` can fall below `θ_b[0]`. Under the smooth
extension `log f(h) = −h`, `log S(h) = −h`, the NLL then falls like
`(h_0(y) − θ_b[0]) · f_i`, i.e. exponentially in `γ`, against a penalty
quadratic in `D θ_b`. This is the #114 mechanism through the monotonicity
rows. Exact rows partly resist it through `−log h'(y_i)`, but censored rows
carry no such barrier. On a right-censored version of the fixture the plain
PHR fit ran off to a non-monotone `θ_b`, logLik `+1471` and `exp` overflow.

**Decision: increment coordinates with a box.** On this path auglag solves in

    δ = T θ,   δ_0 = θ_b[0],   δ_k = θ_b[k] − θ_b[k−1] (k ≥ 1),   identity on [β | γ],

where `D θ_b ≥ 0` is the box `δ_k ≥ 0`, enforced inside every inner L-BFGS-B
solve (the #114 box mechanism). `T` is unit lower-triangular on `θ_b`, so the
model and the constraint functions are unchanged. The monotonicity rows stay
in `A_ineq` for reporting with a zero PHR multiplier, and the bound multiplier
`ν_lo[k]` is folded into row `k − 1`'s `mu_ineq` entry. With `h_0`
non-decreasing at every inner iterate, every censoring branch's NLL falls at
most linearly in a violated `g_i`, against the quadratic penalty. The objective
also runs under `np.errstate(invalid="ignore")`: a left- or interval-censored
trial point with `h(c|x_i) < 0` makes `log F` NaN, which the likelihood
already turns into the infeasibility penalty. *(#163: only for left-censored
and `(−∞, t]` rows. On a both-finite interval row `_log_diff_ndtr` sends the
NaN ratio to its narrow branch, `log f(mid) + log(u − l)` under the extension
`log f = −h`, which is finite and can exceed 0. The value is bounded, not
exponential in `γ`: `h` is clipped to `±30` and `mid` is linear in the
violated endpoint, so the NLL still falls at most linearly in a violated
`g_i`. Since #163 the objective also suppresses numpy's notices for this
case. See the #163 amendment below.)*

- **Rejected — a box on `θ_b[0]` as in #114.** With a shift block `θ_b[0]`
  may legitimately be negative (`x_d · β + offset > 0`), so `θ_b[0] ≥ 0` would
  over-constrain the fit.
- **Rejected — increment coordinates on every path.** The unscaled and
  scale-only paths are bounded without it, and their fits stay
  byte-identical. *(#161: wrong for the scale-only path. Its censored fits
  diverge the same way; it now solves in increment coordinates too, see the
  #161 amendment.)*

**Validation.** R `mlt` has no support rule, so, as Decision 4 prescribes, the
reference `reference/mlt_exponential_scaling_shift_*` (generated in
`reference/generate_reference.R`) is a multi-start explicit-constraint
`alabama::auglag` fit on mlt's likelihood. The hand-written NLL and score are
checked against mlt's `loglik` / `scorei` at a feasible point, and the
Hessian and constraint Jacobian against central differences. The `hin` rows
are `g_i` and `D θ_b ≥ 0`. All 10 feasible starts converge to the same active
set, which is then polished by Newton on its KKT system (with the closed-form
`∇²g_i`) to stationarity 5.7e-14. An offset makes the active set
non-degenerate: two support rows (multipliers ≈ 70), no monotonicity row,
`γ ≈ 0.42`. Without an offset every row binds at `θ_b[0] = β = 0` ("Known
cost"), which is unsuitable as a reference. `tests/test_exponential_scaling_shift.py`
checks:

| Check | Tolerance |
|---|---|
| logLik, θ | `rtol = 1e-6`, `atol = 1e-10` |
| `mu_ineq` (R's KKT multipliers, same active rows) | `rtol = 1e-6`, `atol = 1e-10` |
| full `[θ_b \| β \| γ]` gradient vs central differences at R's θ | `rtol = 1e-6` |
| `J` vs central differences; `γ = 0` rows = linear rows | `1e-7` / `1e-14` |
| `h(y|x_i) ≥ −1e-7` on a support grid, every training row | — |
| right / left / interval censoring and scale-only offset vs an independent trust-constr solve with the rows written inline | logLik `rtol = 1e-6` |

The tests also cover `fixed_params` (β pinned at the optimum reproduces the fit;
`γ` pinned at 0 equals the unscaled #160 fit), the SLSQP / trust-constr
rejection, and the reported `[D ; J(θ̂)]` rows.
`tests/test_exponential_predict_support.py::TestScaledInsideHull` adds the
predict-side case anticipated below: the midpoint of the two binding rows lies
strictly inside the training `(x_d, x_s)` hull, and `h(a|x) < 0` there by the
Jensen gap `θ_b[0] · [f(x̄_s) − (f_i + f_j)/2]`. The #157 warning names it.

---

## Amendment (2026-11-01, [#161](https://github.com/RMKruse/mltpy/issues/161)) — Increment Coordinates on the Scale-Only Path

**Problem.** The #114 box `θ_b[0] ≥ 0` left the monotonicity rows
`D θ_b ≥ 0` as PHR rows only. The #115 mechanism then applies unchanged: a
right-censored row contributes `−log S = h(y_i|x_i)` and no `−log h'` term,
so where a region of `y` holds censored rows but no exact rows, nothing in the
likelihood keeps `h_0` non-decreasing there. A violated monotonicity row
lets `h_0(y_i) < θ_b[0]`, and through `f_i` the NLL falls exponentially in
`γ` against a penalty quadratic in `D θ_b`.

**Which designs trigger it.** The #161 reproducer passes `y ≤ c` as the
`censored` flag of `CensoredData.right_censored`. Every row below the common
time `c` is then censored at its own time, and the exact rows sit tied at
`c`, so `[0, c)` holds censored rows only. On that design (Exp responses,
`n = 150`, seeds 0–2, `c` at the 50/80/95 % quantile) `v0.6.0` auglag
diverged in 9/9 fits (`converged=False`, logLik `+1.8e3 … +2.9e3`,
`min D θ_b ≈ −9 … −18`). trust-constr and SLSQP agreed with each other and
stayed monotone. The left-censored mirror diverged in 1/9 fits. Proper Type-I
censoring (`censored = y > c`, every row above `c` censored at `c`) did
**not** diverge: right, left and interval designs on the same grid already
agreed with trust-constr / SLSQP at `rtol ≤ 1e-6` in 27/27 fits. There the
censored rows are evaluated only where exact rows also constrain `h_0`.

**Decision: option (a), every scaled exponential auglag fit.**
`_optimize_auglag` routes the scale-only path through `_increment_auglag`
as well. `support_nl` is now optional, and the collapsed support row becomes
the bound `δ_0 = θ_b[0] ≥ b` (`intercept_lower`). The model and the
constraint functions are unchanged. The monotonicity and support rows stay in
`A_ineq` for reporting. Their bound multipliers `ν_lo[1:p]` and `ν_lo[0]` are
folded into `mu_ineq`, as on the #115 path.

- **Rejected — (b) every exponential auglag fit.** The unscaled path is
  bounded: without `f_i` the NLL falls at most linearly in a violated row,
  against the quadratic penalty. Its fits stay byte-identical (#158/#160
  fixtures untouched).
- **Rejected — (c) every auglag fit.** Under the normal and logistic links
  `−log S(h)` and `−log f(h)` are bounded below, so no censoring branch can
  drive the NLL to `−∞` through a violated monotonicity row. Changing those
  paths would give up byte-identical R-parity fits for no gain. The large
  positive logLik that shift + scale normal/logistic fits reach on the #161
  design is a separate clipping artefact (`h` clipped at `±30`, `log h'` not),
  tracked in [#162](https://github.com/RMKruse/mltpy/issues/162).

**What changed numerically.** Only scale-only, zero-offset exponential auglag
fits. On the exact-data R fixture `mlt_exponential_scaling_*` the fit moves
*closer* to R: `max |Δθ|` drops from `3.1e-7` to `1.1e-8`, the relative
logLik error from `4.8e-10` to `1.5e-15`, the active row `θ_b[2] = θ_b[3]`
now holds exactly, and `mu_ineq` matches R's KKT multipliers as before. Every
existing test tolerance holds unchanged. The #115 path already solved in
increment coordinates and is byte-identical.

**Validation.** `tests/test_exponential_scaling_no_shift.py`:

- `TestCensoredOnlyRegion` covers the #161 design (nine simulated fits plus
  the R fixture's data). auglag must converge, stay feasible and monotone, and
  match the better converged trust-constr / SLSQP logLik at `rtol = 1e-6`.
  It also checks that the active monotonicity and support rows carry their
  bound multipliers, with `∇nll = Aᵀ μ`, and that `fixed_params` pins
  (`θ_b[4]` or `γ` at the optimum) compose with the increment box and
  reproduce the fit. The left-censored mirror (seeds 0–2, `c` at the median)
  is checked for convergence, feasibility and logLik not above the scipy
  optimum, but not for parity: its optimum sits in the `±30` clip regime of
  #162 (`θ_b[4] ≈ 5·10³`), where the three solvers agree only on the clipped
  objective. 14 of these 16 tests fail on `v0.6.0`; the left mirror passes
  for seeds 0 and 1.
- `TestTypeICensoring` covers administrative censoring at a common time
  (right, left), a fixed inspection grid with an open last interval
  (interval), and right censoring on the fixture data, all under the same
  criteria.

No interval-censored divergence was found. An interval row contributes
`log(F(h(u|x_i)) − F(h(l|x_i)))` with `log F(h) = log(1 − e^{−h})`, which is
undefined, and so turned into the infeasibility penalty, as soon as `h` dips
below 0 at an endpoint or `h_0` decreases across the row. The
exponential-in-`γ` descent of right-censored rows has no interval
counterpart. The inspection-grid designs above, where every row is
interval-censored, already agreed on `v0.6.0`.

**#163 amendment: out-of-support interval rows are not the penalty.** The
paragraph above is wrong on one point. When `h(l|x_i) < 0` at a finite lower
endpoint while the upper endpoint is finite, `log F(h_l)` is NaN, but
`_log_diff_ndtr` does not propagate it. The NaN ratio fails its `wide` test,
and the narrow branch returns a finite `log P`, too large and sometimes
above 0. On the inspection grid (seed 0, `q = 0.8`) a trial point at
`θ_b[0] = −0.5` scores logLik −194.0 against −243.7 at the nearby feasible
`θ_b[0] = 0`. The conclusion stands: the value is bounded (clip at `±30`), so
there is still no interval counterpart of the right-censored descent.

auglag never evaluates such a point on the scale-only path (`θ_b[0] ≥ 0` is
a box). SLSQP did in 120 of 326 interval evaluations of the 9-design sweep,
trust-constr in 18 of 347. Every solver still ends on the support row and
matches auglag's logLik within `2·10⁻⁹` relative.

- **Rejected — propagate the NaN** (so ℓ raises `InfeasibleParameterError`
  and the scipy objective returns the infeasibility penalty). The penalty's
  subgradient points back into the monotone cone only, not across the support
  row. trust-constr then stayed at its start, `θ = [0, ¼, ½, ¾, 1 | γ = 0]`,
  on all three `q = 0.5` designs and reported `converged=True` 8–11 logLik
  units below the optimum. SLSQP moved by at most `1.3·10⁻⁷` in logLik.
- **Deferred — R's `pexp` semantics, `F(h) = 0` for `h ≤ 0`.** Then
  `log P = log F(h_u)` whenever `h_l ≤ 0`, which is continuous at the boundary
  and rewards nothing outside it. It is a new extension, though, and needs a
  matching gradient (zero lower-endpoint weight below the support), so it
  needs its own decision.

`TestTypeICensoring::test_interval_inspection_grid` now requires SLSQP and
trust-constr each to match auglag's logLik (`rtol = 1e-6`). The previous
check, "auglag matches the best converged reference", let the trust-constr
stall through.

---

## Consequences

- **ADR 0002 Decision 3 is superseded** for its exponential corner case. Its
  monotonicity argument (`D θ_b ≥ 0` suffices, `γ` free) still holds.
- **Guards.** On the shift path the only rejection is the `ValueError` in
  `MLT.__init__` (`src/mltpy/model.py`). It is lifted in the implementation
  slices: the scale-only case via the linear collapse first, then shift + scale
  via the nonlinear block. The `InteractionBasis` rejection in `optimize()`
  stays, and `MLT.__init__` keeps rejecting that combination. *(#114: the
  `MLT.__init__` guard now fires only for an `InteractionBasis`; the shift
  path's remaining rejection, for a shift block or a non-zero offset, is a
  `NotImplementedError` in `optimize()`, see the #114 amendment.  #115: that
  rejection now fires only for SLSQP / trust-constr (#116); auglag fits the
  nonlinear block, see the #115 amendment.)*
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
  slice. *(#115: not needed.  `_restart_loop` perturbs the unperturbed
  `theta_init`, whose `β = γ = 0`, so every restart sits at `γ = 0` where the
  rows are the linear `θ_b[0] + x_d,i · β ≥ −offset_i`, and the #160
  `support_floor` lift makes them feasible.)*
- **Inference at an active support row.** When `g_i = 0` binds at the optimum
  (always in the scale-only case), Wald widths are unreliable along the active
  direction, as with active monotonicity rows. `confint(type="profile")` is
  the recommended diagnostic.
- **Exponential + scaling remains incompatible with `InteractionBasis`**
  (ADR 0003 Decision 5).
