# ADR 0002 — Scaling Terms (Heteroskedastic CTMs)

**Date:** 2026-05-21  
**Status:** Accepted (Decision 2 superseded by [ADR 0003](0003-scaling-with-interaction.md))  
**Deciders:** RMKruse  
**Issue:** [#69 Scaling Terms: ADR for scaled-baseline API and monotonicity strategy](https://github.com/RMKruse/mltpy/issues/69)  
**Parent:** [#28 Scaling Terms (epic)](https://github.com/RMKruse/mltpy/issues/28), [#33 Architectural extensions](https://github.com/RMKruse/mltpy/issues/33)

---

## Context

The shift CTM `h(y|x) = h_0(y) + x_d · β` is homoskedastic — the conditional
transformation differs from the baseline only by a covariate-dependent
translation.  The heteroskedastic generalisation, mirroring R `tram::*(scale=...)`,
multiplies the baseline by a strictly positive covariate-dependent factor:

    h(y | x_d, x_s) = h_0(y) · exp(x_s · γ) + x_d · β.

`exp(x_s · γ) > 0` keeps the transformation monotone in `y` whenever `h_0` is
monotone, but it couples `γ` to `θ_b` non-linearly — the gradient
`∂h / ∂γ = h_0(y) · x_s · exp(x_s · γ)` depends on the current `θ_b` *and* `γ`,
so the Jacobian rebuilds every iteration (unlike the constant `[B | X]` Jacobian
of the shift path).

This ADR fixes the six open questions enumerated in #69 *before* any
likelihood / optimiser code is written, so the downstream tracer-bullet
slice (#70) and the censoring / predict / vcov / tram-class slices
(#71–#77) can be built against a single agreed-on API.

---

## Decision 1 — Public Surface

**Chosen:** A single new kwarg `scaling=` on `ConditionalTransformationModel`,
`MLT`, and every `_TramModel` subclass.  The kwarg accepts an **ndarray of
pre-built scaling-design columns**, supplied at construction.  No new model
class is introduced; existing classes route to the scaled-likelihood path
whenever `self.scaling is not None`.

**Type signature:**

```python
class ConditionalTransformationModel:
    def __init__(
        self,
        basis: BernsteinBasis | InteractionBasis,
        ...,
        scaling: NDArray[np.float64] | None = None,
    ) -> None: ...
```

The same kwarg is threaded through `MLT.__init__` and all `_TramModel`
subclasses (`BoxCox`, `Coxph`, `Colr`, `Lm`, `Survreg`).  `Polr` does **not**
accept `scaling=` in the v0.4 release — ordinal scaling would require a
separate ADR (sign of `γ`, identifiability against the cutpoint block).

**Why ndarray of pre-built columns, not a basis-like spec:**

- `exp(x_s · γ)` is linear-in-`γ` *inside* the exponential — there is no
  basis evaluation step that produces design columns from a raw scaling
  covariate.  Any feature engineering the user wants (polynomial, spline,
  interaction, dummy expansion) can be done externally and passed as raw
  columns, identical to how the shift design `X` is supplied today.
- The `interacting=` kwarg from ADR 0001 accepts a basis because the
  tensor-product structure `a(y) ⊗ b(x)` literally requires a basis
  evaluation on `x` — that argument does not apply here.
- An ndarray fixes `q_s = scaling.shape[1]` at `__init__` time, which lets
  `__init__` validate the parameter-vector layout without having to defer
  to `fit()` (mirrors how `BernsteinBasis(order, support)` fixes `p` up
  front).

**Convention for predict / new data:** at `predict(...)` and any other
post-fit method that takes new `X_new`, a parallel keyword
`X_scale_new: NDArray | None = None` is supplied; its column count must
equal `self.scaling.shape[1]`.  Shape mismatch raises `ValueError` with the
same message family as the existing `X_new` checks.

**Convention at fit:** `y.shape[0]` (or `CensoredData.n`) must equal
`self.scaling.shape[0]`.  Mismatch is a `ValueError`.

**Centring:** users are responsible for column-centring their scaling
design if they want `γ = 0` to correspond to the marginal-mean variance.
No automatic centring is performed — mirroring how mltpy does not
auto-centre `X` today.

**Rejected alternatives:**

- *Basis-like spec* (parallel to `interacting=`): rejected for the reasons
  above; an `evaluate(x)` interface adds nothing because there is no
  expansion step.
- *Boolean flag* (`scaling: bool = False`): rejected because the issue
  requires the `scaling is not None` detection pattern, and a `False`
  default is truthy w.r.t. that idiom.  Also forces the column count to
  be discovered lazily, which complicates `__init__` validation.
- *Integer column count* (`scaling: int | None = None`): pragmatic but
  loses the "ndarray of pre-built columns" framing requested in #69 and
  would still require `X_scale` at fit, adding a second kwarg.  Pre-built
  ndarray subsumes this — `scaling.shape[1]` is the count.

---

## Decision 2 — Parameter-Vector Layout

**Chosen:** `theta_ = [theta_b | β | γ]`, length `p + q_d + q_s`, where

- `theta_b` is the Bernstein coefficient vector (length `p = basis.order + 1`),
- `β` is the shift block (length `q_d = X.shape[1]`, 0 if no `X`),
- `γ` is the scaling block (length `q_s = scaling.shape[1]`, absent if
  `scaling is None`).

The order is fixed so that the existing `theta_[:p]` and `theta_[p:p+q_d]`
slices used throughout `likelihood.py` and `model.py` remain valid *byte-for-byte*
when `scaling is None`; the new `γ` block is strictly appended.

### `coef_`, `feature_names_in_`, `n_free_params_`

- `coef_` returns the same vector as today (the `β` block).  A new
  property `gamma_` returns the `γ` block; it raises `ValueError` when
  scaling is inactive and `NotFittedError` before fit (matching the
  `coef_`/`sigma_`/`intercept_` contract — see the 2026 accessor-vocabulary
  consolidation).  `theta_b` is exposed as `basis_coef_` (already informally
  accessible via `theta_[:p]`).
- `feature_names_in_` continues to name the `β` columns.  A second
  attribute `scaling_feature_names_in_` is populated symmetrically from
  the scaling ndarray's column metadata (DataFrame columns) or
  `["S1", "S2", ...]`.
- `n_free_params_` = `p + q_d + q_s`.  No change to the formula; the new
  block adds `q_s` to the count.

### `_split_theta` generalisation

The current

```python
def _split_theta(theta, p, X) -> (theta_b, beta): ...
```

is generalised to

```python
def _split_theta(
    theta: NDArray[np.float64],
    p: int,
    q_d: int,
    q_s: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None]:
    theta_b = theta[:p]
    beta    = theta[p : p + q_d] if q_d > 0 else None
    gamma   = theta[p + q_d : p + q_d + q_s] if q_s > 0 else None
    return theta_b, beta, gamma
```

Existing call sites that pass `X is None / not None` are mechanically
rewritten to pass `q_d = 0 if X is None else X.shape[1]` and
`q_s = 0` (i.e. unchanged).  The shift-only test surface is byte-identical.

A thin wrapper `_split_theta_shift(theta, p, X)` matching the old
two-return signature is **not** kept — every call site is touched once
in the tracer-bullet slice (#70).  See Decision 6 for backward-compatibility
expectations.

### Interaction × Scaling

> **Superseded by [ADR 0003](0003-scaling-with-interaction.md) (2026-05-25).**
> The "out of scope" deferral below has been resolved: the combination is
> now a ratified (supported-but-experimental) path with layout
> `theta_ = [vec_C(Θ) | γ]`.  R `tram` was confirmed to support it.  The
> paragraph below is retained for historical context only.

Scaling stacks cleanly on top of *shift* models.  Combining `scaling=`
with an `InteractionBasis` is **out of scope for v0.4** and raises
`ValueError("scaling is not supported with InteractionBasis")` at
`__init__`.  Both extensions touch the parameter-vector layout
non-trivially and need their own integration ADR before being mixed.

---

## Decision 3 — Monotonicity Strategy

**Chosen:** The existing `D @ theta_b ≥ 0` constraint suffices for every
base distribution except `"exponential"`; `γ` is unconstrained.  No new
rows are added to `constraints.py` for the scaled path.

**Why `γ` is unconstrained:**

    ∂h(y | x) / ∂y = h_0'(y) · exp(x_s · γ).

The scaling factor `exp(x_s · γ)` is strictly positive for every finite
`γ` and every finite `x_s`.  If `h_0'(y) ≥ 0` (the standard
`D @ theta_b ≥ 0` constraint), then `∂h/∂y ≥ 0`.  Monotonicity in `y`
is therefore inherited from `h_0`; the scaling block contributes a
positive multiplicative factor that cannot flip the sign.

**Boundedness:** the auglag / SLSQP / trust-constr solvers do not require
bounds on `γ` because the negative log-likelihood is coercive in `γ`
for every base distribution we ship (exponential tails of `exp(x_s · γ)`
in the likelihood penalise drift to ±∞).  In practice the solver may
take large interior steps early; the AugLag perturb-and-project restart
loop already handles infeasibility for `θ_b`, and `γ` only affects the
likelihood, not the linear constraint surface.

**The `"exponential"` base-distribution corner case:**

The exponential link has support `[0, ∞)`, enforced today by adding the
row `theta_b[0] + X_i · β ≥ 0` for each training observation
(`nonneg_lower=True` branch of `build_constraints`).  With scaling,
the corresponding constraint becomes

    theta_b[0] · exp(x_s_i · γ) + X_d_i · β ≥ 0

— **non-linear in `γ`**, so the existing linear-constraint scaffolding
does not extend directly.  Three options were considered:

1. Add a non-linear constraint to the auglag PHR loop.  Possible but
   doubles the active-set complexity for a rarely-used link.
2. Restrict `theta_b[0] ≥ 0` *and* `X_d_i · β ≥ 0` for every `i`.  A
   sufficient (not necessary) condition; the latter is hard to enforce
   without bounding `β`.
3. **Forbid `base_distribution="exponential"` with `scaling != None`**
   for v0.4.  Raise `ValueError` at `MLT.__init__`.

**Chosen:** option (3).  The exponential link is the rarest of the seven
base distributions and the only one that requires support-feasibility
rows; combining it with scaling has no R `tram` parity to chase (R does
not expose this combination either).  Lifting the restriction would
require a follow-up ADR and a non-linear-constraint extension to
`_auglag.py`.

`build_constraints` itself does **not** change in this release.  The
docstring is updated to record the new precondition (`scaling=None`
when `nonneg_lower=True`).

---

## Decision 4 — Likelihood-Path Generalisation

**Chosen:** Reuse ADR 0001's Option (a).  The private `_ll_*` and
`_grad_*` censoring-dispatch functions accept a precomputed
`(h, h_prime, J, J_prime)` quadruple rather than re-deriving the design
matrix internally.  A thin builder function assembles the quadruple
from `(theta, basis, y, X, scaling)` once per likelihood evaluation;
the censoring branch then consumes the quadruple uniformly across the
shift, interaction, and scaled paths.

**Notation.** Let

    f_i  := exp(0.5 · x_s_i · γ)      (n,)     positive scaling factor
    b_i  := B_basis(y_i)              (n, p)   y-basis row
    db_i := dB_basis(y_i) / dy        (n, p)   y-basis derivative row.

The factor of **0.5 in the exponent** matches mlt's internal convention
(`mlt:::tmlt` evaluates `sterm <- exp(0.5 * <scaling_predict>)`), so γ is
sign- *and* magnitude-aligned with R `tram`'s scaling coefficient.  This
was discovered while implementing the tracer-bullet slice (#70) — without
the 0.5 factor, mltpy's γ converges to half R's γ even though the fitted
likelihood matches.  See `mltpy/likelihood.py::_ll_none` for the
implementation.

Then

    h_i        = (b_i · θ_b) · f_i + X_d_i · β              (+ offset_i)
    h_prime_i  = (db_i · θ_b) · f_i.

The full Jacobian rows are:

    J        = [ B · diag(f) | X_d | h_0(y) · diag(f) · X_s ]   (n, p + q_d + q_s)
    J_prime  = [ dB · diag(f) |  0  | h_0'(y) · diag(f) · X_s ]  (n, p + q_d + q_s)

where `h_0(y_i) = b_i · θ_b` and `h_0'(y_i) = db_i · θ_b` — both depend
on the current `θ_b`, so `J` and `J_prime` rebuild every iteration in
the scaled path (unlike the shift path, where `[B | X_d]` is fixed).

**Score / gradient.** With `s_i = ∂ℓ_i / ∂h_i` (already computed by the
censoring-specific `_ll_*` for every link) and
`q_i = ∂ℓ_i / ∂(log h'_i) = 1` for exact rows / `0` for censored, the
full gradient is

    ∂ℓ / ∂θ  =  J.T · s  +  (J_prime / h_prime).T · q   (weighted analogously).

Both terms already exist in the shift and interaction paths; the only
new responsibility is the **builder** that assembles `J / J_prime` with
the `diag(f)` factor and the `h_0(y) · diag(f) · X_s` columns.

**Rejected alternative — parallel `scale=` argument.**  Adding a
`scale=...` kwarg to each of the four `_ll_*` and `_ll_and_grad_*`
families would double their surface area for one additional code path.
Option (a) — quadruple-in, scalar-out — already absorbs the interaction
path and is the natural place for scaling to plug in.

**Censoring coverage.** Every censoring branch (`NONE`, `RIGHT`, `LEFT`,
`INTERVAL`) receives the same `(h, h_prime, J, J_prime)` quadruple from
the builder; the only change is that interval-censored rows need `h_a`,
`h_b` *and their J-rows at both endpoints*.  The builder returns
`(h_a, h_b, J_a, J_b)` instead of `(h, J)` for those rows — mirroring
the existing `_ll_interval` structure.

---

## Decision 5 — Sign Convention vs. R `tram`

mltpy parameterises `h + X_d · β`; R `tram::*` parameterises
`h − X_d · β` (documented for `Polr` and inherited by the rest of the
`_TramModel` subclasses).  mltpy's `β` is therefore the **negative** of
R's `β`.

For the scaling block, R `tram::Lm(..., scale = ~ z)` parameterises

    h_R(y | x_d, x_s) = h_0(y) · exp(0.5 · x_s · γ_R) − x_d · β_R,

i.e. the sign on `β` flips but the scaling block enters with the
**same sign *and* magnitude** as ours.  So mltpy's `γ` is sign-aligned
with R `tram`'s `γ` — *no flip required* when checking parity.
(Confirmed against `tram::Lm`'s vignette §2 and against `mlt:::tmlt`
in `mlt/R/methods.R`, which evaluates `sterm <- exp(0.5 * predict(
model, terms = "bscaling"))`.)

**Documentation requirement for parity fixtures (#70, #77):** the
fixture-generating R script (`reference/generate_reference.R`) writes
`-coef(fit)[shift_block]` for the `β` block and the raw `coef(fit)[scale_block]`
for the `γ` block.  The Python parity test compares element-wise without
any further sign manipulation.

A unit test in #70 ratchets this convention: it fits the same toy data
under R and mltpy and asserts `np.allclose(mltpy.coef_, -r.beta)` AND
`np.allclose(mltpy.gamma_, r.gamma)`.  If the test fails, the
convention is wrong, not the implementation.

---

## Decision 6 — Backward Compatibility

**All non-scaling models must be byte-identical to v0.3.x.**  Concretely:

- Every existing test in `tests/` passes without modification.
- The shift-path code path (`self.scaling is None`) does **not** see any
  of the new `_split_theta` machinery: the dispatch branch tests
  `if self.scaling is not None:` and routes to the new path; the
  `else` branch is the existing implementation, verbatim.
- `theta_`, `coef_`, `feature_names_in_`, `n_free_params_`, `hessian_`,
  `vcov()`, `estfun()`, `predict()`, `residuals()`, `simulate()`,
  `score()` all retain their current behaviour for `scaling=None`.
- `build_constraints`, `build_constraint_matrices`, the auglag /
  SLSQP / trust-constr dispatch in `optimizer.py` accept the same
  arguments; no new required parameters anywhere.
- The `_split_theta` rewrite is mechanically equivalent for
  `q_s = 0` — `theta_b, beta, gamma = _split_theta(theta, p, q_d, 0)`
  returns `(theta_b, beta, None)`, and every call site is updated to
  ignore the trailing `None`.

The scaling code path is dead code for every existing test until #70
flips the first row.

---

## Stub files

Alongside this ADR, a `scaling=` kwarg is added to `MLT.__init__` (and to
`ConditionalTransformationModel.__init__`) with a `NotImplementedError`
on any non-`None` value.  This lets downstream slices import the kwarg
name without a follow-up rename, and lets test fixtures fail loudly
("scaled likelihood is not yet implemented") rather than silently
returning a shift-only fit.

No corresponding stub is added to `_TramModel` subclasses yet — those
land in #73–#77 alongside the per-class convergence tuning.

---

## Consequences

- **Slice #70** (tracer bullet) implements `_build_scaled_h_J`, the exact
  + normal scaled likelihood, and the auglag plumbing for the new
  parameter block.  R parity test ratchets sign convention.
- **Slice #71** adds censoring (`RIGHT`, `LEFT`, `INTERVAL`) and the
  remaining base distributions to the scaled path.
- **Slice #72** extends `predict()` (`distribution`, `density`, `hazard`,
  `quantile`) — quantile inversion needs the per-row bracket
  `[theta_b[0] · f_i, theta_b[-1] · f_i] + X_d_i · β` derived from `f_i`.
- **Slices #73–#76** wire `scaling=` through `BoxCox`, `Coxph`, `Colr`,
  `Lm`, `Survreg`.
- **Slice #77** adds `vcov` / sandwich SE / Wald-test coverage for the
  `γ` block.
- **Slice #78** is the vignette covering heteroskedastic regression
  (`Lm + scale`) and non-PH survival (`Coxph + scale`).

**Out of scope for v0.4:**
- `scaling=` combined with `InteractionBasis` (ADR follow-up).
  **→ Resolved by [ADR 0003](0003-scaling-with-interaction.md).**
- `scaling=` combined with `base_distribution="exponential"` (requires a
  non-linear constraint extension).
- `scaling=` on `Polr` (ordinal sign-convention ADR follow-up).
- Basis-like spec for the scaling design (deferred until a user-driven
  need emerges; the ndarray surface is forward-compatible — a basis-like
  argument could be added as a union without breaking existing callers).
