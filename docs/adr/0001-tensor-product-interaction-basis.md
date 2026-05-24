# ADR 0001 — Tensor-Product Interaction Basis

**Date:** 2026-05-17  
**Status:** Accepted  
**Deciders:** RMKruse  
**Issue:** [#62 Interacting Terms: ADR for tensor-product API and monotonicity strategy](https://github.com/RMKruse/pymlt/issues/62)  
**Parent:** [#27 Interacting Terms (Tensor Products)](https://github.com/RMKruse/pymlt/issues/27)

---

## Context

Conditional Transformation Models allow the transformation `h(y|x)` to depend on covariates
either through a linear shift (`h(y|x) = a(y)·θ + x·β`, the current implementation) or
through a fully-interacting tensor-product structure
(`h(y|x) = (a(y) ⊗ b(x))ᵀ vec(Θ)`).  The tensor-product form is the natural
generalisation that covers interaction effects between the response and the covariate.

This ADR resolves five open questions required before any tensor-product code is written.

---

## Decision 1 — Public Surface

**Chosen:** Single new class `InteractionBasis(y_basis, x_basis)` in `pymlt.basis`,
re-exported from `pymlt.__init__`.  The existing `MLT` / `ConditionalTransformationModel`
classes accept this as the `basis=` argument; no new model class is introduced, and no
new model parameter (`interacting=...`) is added.

**Rationale:**
- All existing bases duck-type a common interface (`evaluate`, `derivative`,
  `evaluate_with_derivative`, `integrate`, `order`, `support`).  `InteractionBasis` adds
  `.x_basis` and `.y_basis` attributes and overrides `.evaluate` to return the
  Kronecker-product design row, fitting cleanly into the same slot.
- The model layer detects `isinstance(basis, InteractionBasis)` once at `fit()` time and
  routes to the tensor-product likelihood path.  This is one branch point rather than a
  parallel class hierarchy.
- A separate `Design`/`DesignSpec` object would require a new model parameter and duplicate
  the validation logic already in `ConditionalTransformationModel.__init__`.

**Consequence:** `InteractionBasis` requires *two* arrays at evaluation time (y and X),
so its `evaluate(y, X)` signature differs from the scalar-basis interface.  The model
layer owns this call convention; the basis itself documents it.

---

## Decision 2 — Parameter-Vector Layout

**Chosen:** Row-major (C-order) vectorisation.  `theta_` stores `vec_C(Θ)` of length
`p · q` where `Θ` is the `(p, q)` coefficient matrix (`p = y_basis.order + 1`,
`q = x_basis.order + 1`).  The shaped matrix is exposed as `model.Theta_`.

Specifically, `theta_[i * q + j] = Θ[i, j]`, matching NumPy's default `ravel`/`reshape`
convention and `np.kron` row ordering.

**What breaks for interaction models:**
- `theta_b` (the Bernstein sub-vector) and `beta` (the regression sub-vector) do not exist
  separately; `theta_` is the entire `vec(Θ)`.
- `coef_` returns `Theta_` as a 2-D array `(p, q)` rather than a 1-D vector.
- `n_free_params_` = `p * q` (no separate beta count).
- Existing code that reads `theta_[:p]` or `theta_[p:]` must not be called on an interaction
  model; these paths are guarded by the `isinstance(basis, InteractionBasis)` branch.

**Non-interaction models are unchanged:** the split `[theta_b | beta]` is untouched.

**Gradient computation:**
`∂h(y_i|x_i)/∂theta = a(y_i) ⊗ b(x_i)` — this is the `(1, p·q)` Jacobian row for
observation `i`, which is exactly the Kronecker-product evaluation row from
`InteractionBasis.evaluate(y_i, x_i)`.  The derivative w.r.t. `y` uses the
corresponding row from `InteractionBasis.derivative(y_i, x_i)`.

---

## Decision 3 — Monotonicity Strategy

**Chosen:** Closed-form column-wise constraints for the initial release.  Grid-based
evaluation is deferred.

**Rule:** `h(y|x) = (a(y) ⊗ b(x))ᵀ vec(Θ)` must be non-decreasing in `y` for every
fixed `x`.  When `b(x)` is **non-negative and a partition of unity** (every column of
the evaluated x-basis matrix is ≥ 0 and the row sums to 1), the condition reduces to

    D @ Θ[:, j] ≥ 0   for every column j = 0, …, q-1

where `D` is the standard `(p-1, p)` forward-difference matrix.  In flat form this
is the Kronecker inequality `(D ⊗ I_q) @ vec(Θ) ≥ 0`, giving `(p-1) · q` linear
constraints — cheap, exact, and handled by the existing `ConstraintMatrices` path.

**Supported x-bases (initial release):**
- `BernsteinBasis` — non-negative, partition-of-unity by construction.
- `OrdinalBasis` — non-negative, one-hot rows (partition of unity).
- `InterceptBasis` — single column of ones, trivially non-negative and a PoU.

**Unsupported x-bases (raise `ValueError` from `build_constraints`):**
- `PolynomialBasis` — can take negative values.
- `LegendreBasis` — oscillates; not non-negative.
- `LogBasis` — single column of potentially negative values.
- `LogBernsteinBasis` — not defined on a compact domain in the x direction.

`build_constraints` raises with a descriptive message naming the unsupported x-basis
so the user can substitute a supported one.

Grid-based evaluation (covering general x-bases) can be added in a later slice without
changing the public API; the only change would be lifting the `ValueError` for those types.

---

## Decision 4 — Likelihood-Path Generalisation

**Chosen:** Option (a) — generalise `_ll_*` and `_grad_*` to accept a precomputed `h`
vector and a Jacobian matrix `dh_dtheta`, rather than adding a parallel interaction path.

**Current shift path (unchanged for non-interaction models):**

```
h      = B @ theta_b  +  X @ beta            # shape (n,)
h_prime = dB @ theta_b                        # shape (n,)
J      = [B | X]                              # shape (n, p+q)  — the full Jacobian
```

**Tensor-product path:**

```
design  = InteractionBasis.evaluate(y, X)     # shape (n, p*q)
h       = design @ theta                      # shape (n,)
d_design = InteractionBasis.derivative(y, X)  # shape (n, p*q)
h_prime  = d_design @ theta                   # shape (n,)
J        = design                             # shape (n, p*q)
```

The private functions `_ll_none`, `_ll_right`, `_ll_left`, `_ll_interval` and their
gradient counterparts receive `(h, log_h_prime, J, ...)` — identical calling convention
for both paths.  A thin dispatcher in `negative_log_likelihood` branches on
`isinstance(basis, InteractionBasis)` to build `(h, log_h_prime, J)` before calling
the censoring-specific functions.

**Why option (a) over a parallel path:**
- The censoring dispatch is the stateful part; duplicating it doubles the surface area
  to maintain.
- `h` and `J` are the only quantities that change between the shift and interaction
  paths; everything downstream is identical.
- Analytical gradients remain exact: `∂ℓ/∂theta = Jᵀ s` where `s` is the score
  vector, same formula for both paths.

---

## Decision 5 — Backward Compatibility

**All non-interaction models are unaffected:**
- `theta_`, `coef_`, `predict`, `simulate`, `log_likelihood` semantics do not change.
- The parameter layout `[theta_b | beta]` and `n_free_params_` are unchanged.
- `build_constraints`, `build_constraint_matrices` accept the same arguments.
- No new required parameters are added to any existing class or function.

The `isinstance(basis, InteractionBasis)` branch added to `fit()` and
`negative_log_likelihood` is the only code path that diverges; it is dead code for
all models constructed without `InteractionBasis`.

---

## Stub files

Alongside this ADR, a `class InteractionBasis` skeleton is added to `pymlt/basis.py`
and re-exported from `pymlt/__init__.py`, raising `NotImplementedError` on all calls.
This lets downstream slices import the name without a follow-up rename.

---

## Consequences

- **Slice 2** implements `InteractionBasis.evaluate` and `derivative`.
- **Slice 3** extends `build_constraint_matrices` with the `(D ⊗ I_q)` rows.
- **Slice 4** wires the interaction path into `negative_log_likelihood`.
- **Slice 5** updates `ConditionalTransformationModel.fit` and exposes `Theta_` / `coef_`.
- Grid-based monotonicity (general x-bases) is explicitly out of scope for this release.
