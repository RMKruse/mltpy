# ADR 0004 — Auglag Stall Acceptance and the Infeasible-Objective Contract

**Date:** 2026-10-31  
**Status:** Accepted  
**Deciders:** RMKruse  
**Issue:** [#134 auglag stalls early on InteractionBasis fits and reports converged=True](https://github.com/RMKruse/mltpy/issues/134)

---

## Context

`auglag_minimize` (`src/mltpy/_auglag.py`) has two ways to declare
convergence:

1. **KKT exit:** `max(‖h‖∞, ‖min(g, μ/ρ)‖∞, ‖∇L_A‖∞) < outer_tol` (1e-7).
2. **Stall exit:** the iterate is feasible (`feasibility < feas_tol`) and θ
   stopped moving between outer iterations,
   `‖Δθ‖∞ ≤ theta_tol · (1 + ‖θ‖∞)` with `theta_tol = 1e-8`. This mirrors
   `alabama::auglag`, which R `mlt` uses. It was added because on degenerate
   active sets (stacked monotonicity boundaries) `‖∇L_A‖∞` floors around
   1e-5 even though θ already matches the reference fit.

Issue #134 showed that the stall exit also accepts a **failed inner
solve**. On the `stram_bs_bs_mev` fixture (Bernstein-y order 2 ⊗
Bernstein-x order 1, minimum extreme value, n = 1000) the interaction
objective returned `(inf, 0)` on a trial step that left the column-wise
monotone cone. L-BFGS-B's line search could not backtrack from that, so it
stopped without moving. The stall exit then reported the start region as
converged: NLL ≈ 2248 against ≈ 893 at the MLE, `‖∇‖∞ ≈ 366`, every
multiplier zero and every constraint slack. Only the Newton-CG polish
rescued the fit, and it left Θ about 1e-5 (relative) off the optimum.

The issue left the remedy open for discussion because it affects existing
R-parity results.

---

## Decision 1 — Infeasible-Objective Contract

**Chosen:** Every objective handed to a solver returns a **finite** value
on an infeasible θ (non-finite log-likelihood, typically h′ ≤ 0). It
returns `_INFEASIBLE_NLL = 1e10` (module constant in `optimizer.py`)
together with the subgradient of the monotonicity-violation penalty

    P(θ) = ½ ‖min(0, A θ − b)‖²,   ∇P = Aᵀ min(0, A θ − b).

`A` is the solver's own inequality matrix: the forward-difference `D` on
the shift path, and `(D ⊗ I_q)` padded with `q_s` zero γ columns on the
interaction path. The shift path (`_make_objective`) already did this. The
interaction path (`_optimize_interaction.obj`) now matches it.

**Why:** a finite value lets a line search interpolate and backtrack.
`inf` with a zero gradient gives it nothing to work with. When θ is inside
the cone but h′ = 0 at some data point, the subgradient is zero and the
finite value alone makes the search backtrack.

**Rejected:** bounding the inner line search so it stays inside the cone.
That would mean replacing L-BFGS-B's line search, a larger and riskier
change than matching an existing contract.

---

## Decision 2 — What Counts as "Active" for the Stall Exit

**Chosen:** the stall exit's active-set rationale applies only when some
constraint exerts force on the inner problem:

    constraint_active = (m_eq > 0) or any(μ > 0)

An inequality sitting at its boundary with μ = 0 does **not** count. For
g ≥ 0 its term `max(0, μ − ρ g)` in ∇L_A is zero, so it cannot explain a
floored stationarity. (Before this decision, a probe with a boundary
inequality, μ = 0 and `‖∇f‖∞ = 5` was reported converged "on an active
set".)

---

## Decision 3 — Stationarity Floor for Stalls Without an Active Constraint

**Chosen:** when no constraint is active, the stall exit is accepted only
if the stationarity also meets the companion gradient test of Gill, Murray
& Wright (*Practical Optimization*, 1981, §8.2.3). Read the step test as
their U2 with `theta_tol = √τ_F`, where τ_F is the relative precision of f.
Their U3 is then

    ‖∇L_A‖∞ ≤ τ_F^(1/3) · (1 + |f|) = theta_tol^(2/3) · (1 + |f|)
            ≈ 4.6e-6 · (1 + |f|)   at the default theta_tol = 1e-8.

A stall that meets neither condition means the inner solve failed. Nothing
changes between outer iterations in that regime (μ, λ and ρ stay as they
are at a feasible iterate), so the loop stops at once with
`converged=False` and the message "Inner solver stalled: …". The restart
loop then tries a perturbed start as usual.

**Evidence.** Every auglag solve in the full test suite plus the 24-cell
benchmark grid was logged (1786 solves). Healthy fits that end in a stall
with no active constraint sit at about **4.3e-9 · (1 + |f|)**. Examples:
KKT 2.8e-6 at |f| = 673, 1.3e-6 at |f| = 295, 9.5e-7 at |f| = 219. The
attainable gradient therefore scales linearly with |f|; it is a property of
floating-point f, not a solver defect. The #134 failure sits at
366 / (1 + 2248) ≈ 0.16 relative, more than seven orders of magnitude
above that floor.

**Alternatives considered:**

| Floor | At \|f\| ≈ 900 | Headroom over healthy fits | Verdict |
|---|---|---|---|
| Absolute, near `outer_tol` (e.g. 1e-6) | 1e-6 | < 1 for large n | Rejected: flags healthy large-sample fits as non-converged |
| `theta_tol · (1 + \|f\|)` (√τ_F) | 9e-6 | ≈ 2.3× | Rejected: too thin a margin to adopt without wider data |
| `theta_tol^(2/3) · (1 + \|f\|)` (U3) | 4e-3 | ≈ 1000× | **Chosen** |

**Accepted trade-off:** because the floor is relative, it is loose against
`outer_tol` in absolute terms. At |f| = 1e5 a stall with no active
constraint and stationarity up to about 0.46 would be accepted. The
literal reading of #134's "never converged=True with a KKT residual ≫
outer_tol" is therefore not met. That reading is unattainable with an
f-based line search on large likelihoods, and the issue itself accepts
`‖∇‖∞ < 1e-6` (10× `outer_tol`) as success. The returned `kkt_residual`
still reports the true value, so callers can apply a stricter check.

**Deviation from R:** `alabama::auglag` has no stationarity gate on its
small-step exit. mltpy is stricter here on purpose. R parity is unaffected:
every R-parity suite stays green, and the `stram_bs_bs_mev` Θ / survival /
hazard parity tightened from `rtol=1e-4` to `1e-6`.

---

## Consequences

- `Coxph(interacting=...)`, with or without `scaling=`, reaches the MLE on
  its first auglag attempt without the Newton-CG polish.
- `converged=True` from the stall exit now means either a genuinely active
  constraint set or a gradient at the floating-point floor. A stuck inner
  solve reports `converged=False`.
- The gate also applies to shift-path auglag fits. Across the suite and the
  benchmark grid, no healthy fit changed its exit.
- No new public option: the floor is derived from the existing
  `AugLagOptions.theta_tol`, whose docstring documents it.
- **Revisit** if a real dataset shows a healthy fit rejected by the U3
  floor, or a false acceptance between the √τ_F and U3 floors. The
  tighter √τ_F floor is then the first candidate.
