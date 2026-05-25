# ADR 0003 — Scaling Terms with the Tensor-Product Interaction Basis

**Date:** 2026-05-25  
**Status:** Accepted  
**Deciders:** RMKruse  
**Issue:** [#102 [HITL] ADR 0003: integrate scaling= with InteractionBasis + verify R support](https://github.com/RMKruse/mltpy/issues/102)  
**Parents:** [#28 Scaling Terms (epic)](https://github.com/RMKruse/mltpy/issues/28), [#33 Architectural extensions](https://github.com/RMKruse/mltpy/issues/33)  
**Supersedes:** [ADR 0002](0002-scaling-terms.md), Decision 2 (the "out of scope" rejection of `scaling=` + `InteractionBasis`)

---

## Context

ADR 0001 introduced the fully-interacting (tensor-product) transformation

    h(y | x) = (a(y) ⊗ b(x))ᵀ vec(Θ),

and ADR 0002 introduced the heteroskedastic *shift-scale* transformation

    h(y | x_d, x_s) = h_0(y) · exp(0.5 · x_s · γ) + x_d · β.

ADR 0002 Decision 2 explicitly deferred the *combination* of the two —
`scaling=` together with an `InteractionBasis` — to "its own integration
ADR", on the grounds that both extensions touch the parameter-vector
layout non-trivially. `MLT.__init__` (`mltpy/model.py:217`) and
`likelihood.hessian` (`mltpy/likelihood.py:3504`) currently reject the
combination with a `ValueError` / `NotImplementedError`.

This ADR ratifies the design for that combination — the *non-proportional,
heteroskedastic* CTM — and records an **empirical** finding on whether R
`tram` exposes the combination, so the downstream (AFK) implementation
slices can be built against a single agreed-on API and validation
strategy. **No production code changes land in this slice; it is design +
docs only.** The runtime rejections at `model.py:217` and
`likelihood.py:3504` remain in place until the downstream slices flip them.

---

## R-support verification (gates the validation strategy)

This was verified empirically against `tram` 1.4.1 / `mlt` 1.7.4. The
reproducible script lives at
`tests/r_scripts/verify_strata_scale_support.R`.

**Finding: SUPPORTED.** R `tram` exposes an interacting/stratified
baseline combined with a scale term through the four-part formula

    y | s ~ x | z

where the parts map onto the underlying `mlt::ctm` arguments as:

| formula part | role | `ctm` argument | mltpy analogue |
|---|---|---|---|
| `y` (LHS 1) | response | `response` | `y_basis` of `InteractionBasis` |
| `s` (LHS 2) | interacting / strata | `interacting` | `x_basis` of `InteractionBasis` |
| `x` (RHS 1) | additive shift | `shifting` | *(no analogue — see Decision 1)* |
| `z` (RHS 2) | scale | `scaling` | `scaling=` ndarray |

Observed behaviour of the combination:

- It **fits** and returns an object of class `stram` (scaled-tram). The
  scale coefficient is recoverable as
  `coef(fit, with_baseline = FALSE)[fit$scalecoef]` (prefix `scl_`).
- It emits `warning("Models with both strata and scale terms are highly
  experimental")`.
- It **errors** (`"scaling variables not allowed as stratifying
  variables"`) if a variable appears in both `s` and `z`.

`mlt::ctm`'s own documentation (Siegfried, Kook & Hothorn 2023,
"shift-scale models") gives the transformation, with `scale_shift = FALSE`
(the default):

    P(Y ≤ y | x) = F_Z( √(exp(s(x)ᵀγ)) · [(a(y) ⊗ b(x))ᵀ ϑ] + d(x)ᵀβ ).

Two consequences fix the rest of this ADR:

1. **`√(exp(·)) = exp(0.5 · ·)`** — the 0.5-in-exponent convention from
   ADR 0002 Decision 4 is confirmed *at the R source* for the interaction
   path, not merely inherited by analogy.
2. The scale factor multiplies **only** the interacting baseline
   `(a(y) ⊗ b(x))ᵀ ϑ`; the additive shift `d(x)ᵀβ` sits outside it. Because
   mltpy's `InteractionBasis` path carries **no** separate additive shift
   block (all covariate effects flow through `vec(Θ)`; ADR 0001 Decision 2),
   the mltpy analogue is the *no-shift* R model `y | s ~ 1 | z`, and the
   `scale_shift = FALSE` / `scale_shift = TRUE` variants **collapse to a
   single form** (there is no `d(x)ᵀβ` to place inside or outside the
   bracket).

**Resulting validation strategy (Decision 6):** because R supports the
combination, downstream slices generate **R parity fixtures** for the `γ`
block as the *primary* gate; because R itself flags the path "highly
experimental", they additionally carry *internal-consistency* tests as a
safety net.

---

## Decision 1 — Combined Parameter-Vector Layout

**Chosen:** `theta_ = [vec_C(Θ) | γ]`, length `p · q + q_s`, where

- `vec_C(Θ)` is the row-major (C-order) vectorisation of the `(p, q)`
  coefficient matrix `Θ` (`p = y_basis.order + 1`,
  `q = x_basis.order + 1`), exactly as in ADR 0001 Decision 2;
- `γ` is the scaling block (length `q_s = scaling.shape[1]`), strictly
  appended.

The interaction path has **no additive `β` shift block** (ADR 0001), so —
unlike the shift-scaling layout `[theta_b | β | γ]` of ADR 0002 Decision 2
— there is no middle block: `γ` is appended directly after `vec_C(Θ)`.

### Accessors

- `coef_` continues to return `Theta_` as the 2-D `(p, q)` matrix (ADR
  0001); the `γ` block is exposed as `gamma_` (mirroring the shift-scaling
  path's `gamma_` / `model.gamma_` accessor from ADR 0002).
- `Theta_` reshapes `theta_[: p · q]` to `(p, q)`; `gamma_` is
  `theta_[p · q :]` (or `None` when `scaling is None`).
- `feature_names_scaling_` names the `γ` columns, identical to the
  shift-scaling path.
- `n_free_params_` = `p · q + q_s`.

### Layout-split guard

The shift-model split `[theta_b | β]` does **not** exist on this path; the
existing `isinstance(basis, InteractionBasis)` guard (CLAUDE.md gotcha) is
extended so that, on the interaction path, the scaling block is sliced as
`theta_[p · q :]` rather than via any `theta_[:p]` / `theta_[p:]`
arithmetic. No `_split_theta` machinery from ADR 0002 is reused on the
interaction path.

---

## Decision 2 — Transformation & Likelihood-Path Generalisation

**Chosen transformation** (mltpy interaction + scaling, no shift block):

    h(y | x, x_s) = ([a(y) ⊗ b(x)]ᵀ vec(Θ)) · exp(0.5 · x_s · γ)   (+ offset),

with derivative in `y`

    ∂h/∂y = ([a'(y) ⊗ b(x)]ᵀ vec(Θ)) · exp(0.5 · x_s · γ).

This is the `scale_shift = FALSE` form of `mlt::ctm` with `d(x)ᵀβ ≡ 0`.

**Likelihood path.** Reuse ADR 0001 Decision 4 and ADR 0002 Decision 4:
the censoring-dispatch `_ll_*` / `_grad_*` functions consume a precomputed
`(h, h_prime, J, J_prime)` quadruple. The only new responsibility is a
builder that, on the interaction + scaling path, assembles:

    f      = exp(0.5 · X_s · γ)                          (n,)   positive factor
    design = InteractionBasis.evaluate(y, X)             (n, p·q)
    ddes   = InteractionBasis.derivative(y, X)           (n, p·q)
    g      = design @ vec(Θ)                             (n,)   interacting baseline
    h      = g · f                                       (n,)
    h_prime = (ddes @ vec(Θ)) · f                        (n,)

with Jacobian rows (columns ordered `[ vec(Θ) | γ ]`):

    J        = [ design · diag(f) | g  · diag(f) · X_s ]   (n, p·q + q_s)
    J_prime  = [ ddes   · diag(f) | g' · diag(f) · X_s ]   (n, p·q + q_s)

where `g' = ddes @ vec(Θ)`. As in ADR 0002, `J` and `J_prime` rebuild every
iteration because the `γ`-columns depend on the current `vec(Θ)` through
`g` / `g'`. The score is `∂ℓ/∂θ = Jᵀ s + (J_prime / h_prime)ᵀ q` — same
formula as both predecessor paths.

**Censoring coverage.** The current `InteractionBasis` release supports
**exact (non-censored) data only** (CLAUDE.md / ADR 0001). The
interaction + scaling path inherits that restriction: it targets
`CensoringType.NONE` for this integration. Extending censoring to the
interaction path (with or without scaling) remains a separate concern and
is out of scope here.

---

## Decision 3 — Monotonicity Strategy

**Chosen:** No new constraint logic. Because

    ∂h/∂y = ([a'(y) ⊗ b(x)]ᵀ vec(Θ)) · exp(0.5 · x_s · γ)

and `exp(0.5 · x_s · γ) > 0` for every finite `γ`, the scale factor cannot
flip the sign of the derivative. Monotonicity in `y` is therefore governed
**entirely** by the interacting baseline `(a(y) ⊗ b(x))ᵀ vec(Θ)`, i.e. by
the existing Kronecker column constraint from ADR 0001 Decision 3:

    (D ⊗ I_q) vec(Θ) ≥ 0      ⟺      D · Θ[:, j] ≥ 0  for every column j,

which requires the x-basis to be non-negative and a partition of unity
(`BernsteinBasis`, `OrdinalBasis`, `InterceptBasis`, `OneHotBasis`). `γ` is
**unconstrained**. The only change to the constraint scaffolding is that
`build_constraint_matrices` appends `q_s` zero columns to the constraint
matrix so its width matches the `[vec_C(Θ) | γ]` parameter vector — exactly
the zero-padding pattern already used to make `D` ignore `β` on the shift
path.

This is the union of ADR 0001 Decision 3 (the Kronecker constraint) and
ADR 0002 Decision 3 (`γ` unconstrained); no genuinely new constraint
mathematics is introduced.

---

## Decision 4 — γ Standard Errors

**Chosen:** `γ` standard errors come from the **finite-difference
interaction Hessian** already used for interaction models
(`_hessian_interaction_fd` in `mltpy/likelihood.py`), with the parameter
vector and Jacobian extended to include the `γ` block. The full observed
information `∂²(-ℓ)/∂θ∂θ'` is computed over `θ = [vec_C(Θ) | γ]`; `vcov()`,
`sandwich_se()`, and Wald CIs subset the `γ` rows/columns exactly as they
subset the `β` block on the shift-scaling path.

**Why finite-difference, not analytic.** The shift-scaling path (ADR 0002
Decision 4) ships an *analytic* `γ`-block Hessian. The interaction path
already abandons analytic second derivatives in favour of
`_hessian_interaction_fd` — the tensor-product structure makes the analytic
Hessian considerably more involved, and the finite-difference observed
information is already the accepted accuracy/effort trade-off for
interaction models. Extending that same finite-difference Hessian to cover
the appended `γ` columns is the path-consistent choice and needs **no**
analytic `γ`-block derivation.

Wald SEs on the interaction + scaling path inherit the same boundary
caveat documented for all monotone-Bernstein models: when a `Θ` column sits
on its monotonicity boundary the bread `−H⁻¹` is near-singular and Wald
widths are unreliable; profile likelihood remains the recommended
diagnostic there. `γ` itself is interior (unconstrained), so its Wald SE is
not subject to that boundary pathology.

---

## Decision 5 — Exponential Base Distribution

**Chosen:** Carry over ADR 0002 Decision 3's rejection. The combination

    scaling != None  AND  isinstance(basis, InteractionBasis)
                     AND  base_distribution == "exponential"

is rejected at `MLT.__init__` with `ValueError`.

**Why.** The exponential link has support `[0, ∞)`, enforced by support-
feasibility rows of the form `h(y_min | x) ≥ 0`. On the interaction +
scaling path that row becomes, per stratum / x-column,

    ([a(y_min) ⊗ b(x)]ᵀ vec(Θ)) · exp(0.5 · x_s · γ) ≥ 0,

which is **non-linear in `γ`** — exactly the obstruction ADR 0002 Decision 3
identified for the shift-scaling path, now reproduced per interacting
column. Lifting it would require the same non-linear-constraint extension
to `_auglag.py` that ADR 0002 deferred. R `tram` does not chase this
combination either (the `stram` machinery applies the same monotonicity
treatment and offers no exponential-link special-case), so there is no
parity to lose. The restriction may be lifted in a future ADR alongside the
non-linear-constraint work.

---

## Decision 6 — Validation Strategy

R supports the combination (see the verification section above), so the
**primary** gate is R parity; because R flags the path "highly
experimental", a **secondary** internal-consistency net is added. Both
must pass before a downstream slice is considered complete.

**Primary — R parity (`γ` block).** Downstream slices add an
`stram` fixture to `reference/generate_reference.R` of the no-shift form
`Coxph(Surv(time, event) | s ~ 1 | z, data = ...)` (and a `BoxCox`
analogue). Following ADR 0002 Decision 5, the fixture writes the **raw**
`coef(fit)[fit$scalecoef]` for `γ` (no sign flip — mltpy's `γ` is sign- and
magnitude-aligned with R's), and the Python test asserts
`np.allclose(model.gamma_, r.gamma)` plus a log-likelihood match. There is
no `β` block to negate on the interaction path.

**Secondary — internal consistency.** Three checks, none of which depend on
R:

1. **Reduction to shift-interaction.** With `scaling` set to all-zero
   columns, `exp(0.5 · 0 · γ) = 1`, so the fit must reproduce the pure
   `InteractionBasis` (ADR 0001) result for `Θ` and log-likelihood.
2. **Analytic gradient vs finite difference.** The analytic `γ`-block score
   from Decision 2 must match `scipy.optimize.approx_fprime` of the
   negative log-likelihood to `rtol ≈ 1e-6`.
3. **Simulate-and-recover.** Draw from a model with a known `γ`; the refit
   must recover `γ` within Monte-Carlo error.

The R "highly experimental" warning is the explicit reason the
internal-consistency net exists: it guards against the (small but nonzero)
chance that the R reference itself is wrong on this path.

---

## Decision 7 — Experimental Status & Runtime Warning

**Chosen:** Mirror R's stance. When the downstream slices enable the path,
`fit()` emits a `UserWarning`:

    "scaling= with InteractionBasis is an experimental path; validate
     against your use case (mirrors R tram's stram warning)."

and the ADR / CLAUDE.md / docstrings label the combination
*supported-but-experimental*. This is an honest first integration of a
combination R itself flags as experimental; the warning is suppressible via
the standard `warnings` filters and does not alter the fit. It is removed
(or downgraded) in a later release once the path has accrued real-world
validation.

---

## Decision 8 — Backward Compatibility

All non-`(scaling + InteractionBasis)` models remain byte-identical:

- Pure shift models (`scaling=None`, scalar basis) — unchanged (ADR 0001/0002).
- Pure interaction models (`scaling=None`, `InteractionBasis`) — unchanged;
  `theta_ = vec_C(Θ)`, `coef_ = Theta_`, no `γ` accessor populated.
- Shift-scaling models (`scaling != None`, scalar basis) — unchanged
  (`theta_ = [theta_b | β | γ]`, ADR 0002).

The new path is reached only when `scaling is not None` **and**
`isinstance(basis, InteractionBasis)`; every existing test passes without
modification. The `gamma_` / `feature_names_scaling_` accessors return the
same values as the shift-scaling path for shift models and `None` for pure
interaction models.

---

## Consequences

- This slice (#102): ADR + the R-support finding + the reproducible
  verification script `tests/r_scripts/verify_strata_scale_support.R`. ADR
  0002 Decision 2 is marked superseded; the CLAUDE.md gotcha is updated.
  **No production code changes.**
- **Downstream (AFK) slices** build against this ADR:
  - the `[vec_C(Θ) | γ]` builder and the exact + normal/min-extreme-value
    scaled-interaction likelihood, with the auglag plumbing for the new
    block and the `UserWarning`;
  - removal of the `ValueError` (`model.py:217`) and `NotImplementedError`
    (`likelihood.py:3504`) once the path is implemented;
  - the `q_s` zero-column extension to `build_constraint_matrices`;
  - the `γ`-column extension to `_hessian_interaction_fd` and `vcov` /
    `sandwich_se` / Wald coverage for `γ`;
  - the `stram` parity fixture in `reference/generate_reference.R` and the
    three internal-consistency tests of Decision 6;
  - `predict()` on the scaled-interaction path — quantile inversion derives
    its per-row bracket from `h(a, x) · f_i` and `h(b, x) · f_i` (the
    existing interaction bracket of `_predict_quantile_interaction`, scaled
    by `f_i`); and the `Coxph` / `BoxCox` convenience wiring.

**Out of scope for this integration (future ADRs):**

- Censored data on the interaction (and hence interaction + scaling) path.
- `base_distribution="exponential"` with interaction + scaling (Decision 5).
- An additive `d(x)ᵀβ` shift block alongside the interacting baseline
  (mltpy's `InteractionBasis` has never carried one; adding it would
  resurrect the `scale_shift` FALSE/TRUE distinction).
- `scaling=` on `Polr` (unchanged from ADR 0002).
