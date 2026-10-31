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

## Decision 2 — The Stall Exit Checks Projected Stationarity

**Chosen:** a feasible stall is accepted only if the gradient that remains
is explained, **direction by direction**, by the constraints active at θ.
At the stalled iterate the multipliers are re-fitted for the true gradient,

    (λ*, μ*) = argmin_{λ free, μ ≥ 0} ‖∇f − Cᵀλ − A_𝒜ᵀμ‖₂,

over the working set 𝒜 of inequality rows that either exert force in the
current inner problem (`φᵢ = max(0, μᵢ − ρ gᵢ) > 0`) or sit on their
boundary (`gᵢ ≤ feas_tol`). The free λ is eliminated by projecting onto
null(C), leaving a Lawson–Hanson NNLS (`_projected_stationarity` in
`_auglag.py`). Two quantities come back:

- the **projected stationarity** `‖r‖∞`, `r = ∇f − Cᵀλ* − A_𝒜ᵀμ*`. If
  `r ≠ 0` then `d = −r` is a first-order feasible descent direction
  (`C d = 0`, `A_𝒜 d ≥ 0`, `∇fᵀd = −‖r‖₂²`) that the inner solve failed to
  take;
- the **duality gap** `Σᵢ μ*ᵢ · max(gᵢ, 0)` of the working-set rows the fit
  used. PHR leaves a binding row with a small slack `gᵢ ≈ (μᵢ − φᵢ)/ρ`
  while its multiplier estimate is off. Using such a row is legitimate only
  while that slack is negligible in objective units: for a convex problem at
  a stationary point, f exceeds the Lagrangian dual bound by exactly `μ*ᵀg`.

Both must meet the floor of Decision 3.

**Superseded first version.** The first cut of this decision used a
blanket activity flag, `constraint_active = (m_eq > 0) or any(μ > 0)`, that
skipped the gradient check altogether. Review showed that a constraint on
one parameter then excused an arbitrarily large gradient in another,
unconstrained parameter. A smooth quadratic `½‖θ − (0, 1)‖²` with θ₁ = 1
fixed by an equality, started at (1, 2) with `inner_options={"maxls": 1}`,
came back `converged=True` with KKT residual 1.0 and θ₂ one unit from its
optimum. The equality-on-θ₁ / gradient-in-θ₂ probe, the positive-multiplier
probe with the same off-axis gradient, and descent along an equality were
all accepted the same way. All of them are now `converged=False`
(`tests/test_auglag.py::TestStallConvergenceProjectedStationarity`,
`TestStallOnSmoothQuadratic`).

**Rejected:** a narrower working set `gᵢ ≤ feas_tol` without the `φᵢ > 0`
rows. In the full suite it rejected 9 healthy fits (n = 5 and n = 8
parameters, KKT ≈ 1e-5) whose binding rows carried φ ≈ 0.5–6 at slack
6e-7 – 4e-5.

---

## Decision 3 — Stationarity Floor for the Stall Exit

**Chosen:** the projected stationarity and the duality gap must both meet
the companion gradient test of Gill, Murray & Wright (*Practical
Optimization*, 1981, §8.2.3), which they apply to the projected gradient
under linear constraints. Read the step test as their U2 with
`theta_tol = √τ_F`, where τ_F is the relative precision of f. Their U3 is
then

    ‖r‖∞ ≤ τ_F^(1/3) · (1 + |f|) = theta_tol^(2/3) · (1 + |f|)
         ≈ 4.6e-6 · (1 + |f|)   at the default theta_tol = 1e-8.

With no constraint in the working set, `r = ∇f`. That is the original form
of this decision.

A stall that fails it means the inner solve failed. Nothing changes between
outer iterations in that regime (μ, λ and ρ stay as they are at a feasible
iterate), so the loop stops at once with `converged=False` and the message
"Inner solver stalled: …". The restart loop then tries a perturbed start as
usual.

**Evidence.** Every auglag solve in the full test suite plus the 24-cell
benchmark grid was logged (1786 solves). Healthy fits that end in a stall
with no active constraint sit at about **4.3e-9 · (1 + |f|)**. Examples:
KKT 2.8e-6 at |f| = 673, 1.3e-6 at |f| = 295, 9.5e-7 at |f| = 219. The
attainable gradient therefore scales linearly with |f|; it is a property of
floating-point f, not a solver defect. The #134 failure sits at
366 / (1 + 2248) ≈ 0.16 relative, more than seven orders of magnitude
above that floor.

**Evidence for the projected form (Decision 2).** The same census was
repeated with the projected stationarity and gap logged at every stall
exit: 1400 accepted stalls in the suite and all 24 benchmark cells. The
only rejected stalls are the synthetic failure probes. On the benchmark
grid the worst accepted stall sits at 0.0074× the floor. The tightest
healthy case in the suite is `TestExponentialWithCovariates`: 101 stacked
per-row support constraints on 6 parameters at a degenerate MLE, with
`‖∇f‖∞ = 100` almost entirely absorbed by the multipliers. Its projected
stationarity of 7.4e-6 sits at 0.88× the floor of 8.3e-6 (|f| = 0.80).
NNLS and BVLS agree on that residual, and SLSQP from the same data reaches
an f only 5.5e-10 lower. The point is genuinely within the floating-point
floor, and the old activity flag accepted it unconditionally.

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
- `converged=True` from the stall exit now means that, up to the
  floating-point floor, ∇f is a valid KKT combination of the constraints
  active at θ (free-sign equality multipliers, non-negative inequality
  multipliers, negligible slack). A stuck inner solve that leaves any
  feasible descent direction reports `converged=False`, even when some
  other constraint is active.
- The returned `kkt_residual` is unchanged: it still uses the solver's
  running multipliers, so a stall accepted on refitted multipliers can
  report a residual well above `outer_tol` (up to ~1e-2 on stacked active
  sets). The exit message also reports the projected stationarity and the
  gap.
- The gate also applies to shift-path auglag fits. Across the suite and the
  benchmark grid, no healthy fit changed its exit.
- No new public option: the floor is derived from the existing
  `AugLagOptions.theta_tol`, whose docstring documents it.
- **Revisit** if a real dataset shows a healthy fit rejected by the U3
  floor, or a false acceptance between the √τ_F and U3 floors. The
  tighter √τ_F floor is then the first candidate.
