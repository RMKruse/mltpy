# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `scaling=` with `base_distribution="exponential"` **with a shift block or a
  non-zero offset** on the default auglag solver
  (`MLT(..., base_distribution="exponential", scaling=X_s).fit(y, X=X_d,
  offset=o)`). The support rule `h(a|x_i) ≥ 0` is enforced exactly as one
  nonlinear row per training observation,
  `θ_b[0]·exp(0.5·x_s,i·γ) + x_d,i·β + offset_i ≥ 0` (ADR 0005 Decisions 2–3),
  for all four censoring types. R `mlt` has no support rule, so the reference
  (`reference/mlt_exponential_scaling_shift_*`) is a multi-start
  explicit-constraint `alabama::auglag` fit on mlt's likelihood, polished by
  Newton on its non-degenerate KKT system; θ, logLik and the multipliers
  match at `rtol=1e-6`. SLSQP and trust-constr still raise
  `NotImplementedError` for this case (#116). (#115)
- `auglag_minimize(..., nonlinear_ineq=g)`: an optional smooth inequality
  block `g(θ) ≥ 0`, `g: θ ↦ (values, Jacobian)`, stacked below the linear rows
  with the same PHR penalty, multiplier update, KKT residual and ADR 0004
  stall test (working-set normals from `J(θ)`). `mu_ineq` lists the linear
  rows first. Without it the solve is unchanged. (#115)
- `ScaledSupportConstraint` / `build_scaled_support_constraint(p, X, X_scale,
  offset)` in `mltpy.constraints`: the scaled exponential support rows with
  their analytic Jacobian; `build_constraint_matrices(..., X_scale=)` returns
  them as `ConstraintMatrices.nonlinear_ineq`. On this path auglag solves in
  increment coordinates `δ = (θ_b[0], Δθ_b, β, γ)`, so `D θ_b ≥ 0` is a box
  enforced inside every inner solve: censored rows carry no `log h'` barrier,
  and a violated monotonicity row lets the smooth exponential extension fall
  exponentially in `γ` (a right-censored fit ran off to logLik `+1.5·10³`
  without it). The reported `constraint_A_ineq` holds the support rows as
  their Jacobian `J(θ̂)`. (#115)
- `scaling=` with `base_distribution="exponential"` on the shift basis when
  there is no shift block (`MLT(..., base_distribution="exponential",
  scaling=X_s).fit(y)`). The support rule `h(a|x_i) ≥ 0` collapses to the
  linear row `θ_b[0] ≥ 0` (ADR 0005 Decision 2). All three solvers and all
  four censoring types fit it, and `θ_b`, `γ` and the log-likelihood match R
  `mlt(ctm(..., scaling = ~x_s, todistr = "Exponential"), fixed = c("Bs1(y)"
  = 0))` (cross-checked against `tram::Aareg(y ~ 0 | x_s)`; fixture
  `reference/mlt_exponential_scaling_*`). `MLT.__init__` now rejects the
  combination only on an `InteractionBasis`. A shift block (`X` with columns)
  or a non-zero `offset` makes the row non-linear in `γ`; auglag fits that
  case since #115, SLSQP / trust-constr raise `NotImplementedError` (#116).
  (#114)
- `auglag_minimize(..., bounds=(lb, ub))`: optional box bounds enforced inside
  every inner L-BFGS-B solve (box-constrained augmented Lagrangian, Birgin &
  Martínez 2014). Bound multipliers are returned as `nu_lower` / `nu_upper`,
  and the KKT residual and ADR 0004 stall test are projected onto the box.
  The scaled exponential path uses it for `θ_b[0] ≥ 0`. There the smooth
  extension `log f(h) = −h` makes the objective fall like
  `−t·Σ exp(0.5·x_s·γ)` below the row, so the PHR subproblem over `ℝⁿ` has no
  minimiser. (#114)
- `build_constraint_matrices` / `build_constraints` accept `q_s=` (number of
  trailing `γ` columns) for the exponential support row under scaling. (#114)

- `BoxCox(..., interacting=...)` — stratified / fully-interacting Box-Cox
  models `h(y|x) = (a(y) ⊗ b(x))ᵀ vec(Θ)` through the convenience surface,
  mirroring `Coxph(..., interacting=...)`. Accepts a `BernsteinBasis`,
  `OrdinalBasis`, or `OneHotBasis` (categorical strata) x-basis and combines
  with `scaling=` for the scaled-interaction (`stram`) model
  `([a(y) ⊗ b(x)]ᵀ vec(Θ))·exp(0.5·x_s·γ)`, exposing `Theta_` and `gamma_`.
  Exact data only: censored rows raise `NotImplementedError`. `γ`, `Θ`,
  log-likelihood, distribution, density and quantile match R
  `tram::BoxCox(y | s ~ 1 | z)` at `rtol=1e-6` (quantiles with an absolute
  tolerance scaled by the support width) (fixtures
  `reference/stram_onehot_bs_boxcox_*`). (#107)
- `Lm(..., interacting=...)` — stratified / fully-interacting normal linear
  models `h(y|x) = (a(y) ⊗ b(x))ᵀ vec(Θ)` with the order-1 Bernstein `a(y)`,
  so `Y | x` is normal with an `x`-dependent mean and standard deviation.
  Accepts a `BernsteinBasis`, `OrdinalBasis`, or `OneHotBasis` x-basis and
  combines with `scaling=` for the scaled-interaction (`stram`) model
  `([a(y) ⊗ b(x)]ᵀ vec(Θ))·exp(0.5·x_s·γ)`, exposing `Theta_` and `gamma_`.
  `sigma_` / `intercept_` / `coef_` raise `NotImplementedError` on this path
  (no single lm mapping when Θ varies with `x`). Exact data only: censored
  rows raise `NotImplementedError`. `γ`, `Θ`, log-likelihood, distribution,
  density and quantile match R `tram::Lm(y | s ~ 1 | z)` at `rtol=1e-6`
  (tram's polynomial `(1, y)` basis maps onto the order-1 Bernstein basis
  exactly) (fixtures `reference/stram_onehot_lm_*`). (#109)
- `Survreg(..., interacting=...)` — non-proportional / stratified parametric
  survival models `h(t|x) = (a(log t) ⊗ b(x))ᵀ vec(Θ)` through the
  convenience surface, with the log-time `LogBernsteinBasis` as the y-axis
  of the tensor product and the family's link (weibull, lognormal,
  loglogistic). Accepts a `BernsteinBasis`, `OrdinalBasis`, or `OneHotBasis`
  x-basis and combines with `scaling=` for the scaled-interaction (`stram`)
  model `([a(log t) ⊗ b(x)]ᵀ vec(Θ))·exp(0.5·x_s·γ)`, exposing `Theta_` and
  `gamma_`; `survival()` / `hazard()` take the parallel `X_scale=`. Exact
  event times only: censored rows raise `NotImplementedError`. For all three
  families `γ`, `Θ`, log-likelihood, survivor, density, hazard and quantile
  match R `mlt::ctm(log_first = TRUE)` at `rtol=1e-6`, and `γ` matches
  `tram::Survreg(y | x ~ 1 | x_s)` (class `stram`) (fixtures
  `reference/stram_logbs_bs_<dist>_*`). (#110) `interacting=` is inserted
  before `scaling=` (matching `BoxCox` / `Coxph` / `Colr`), so a call that
  passed `scaling` positionally as the fifth argument must now pass it by
  keyword.
- Worked notebook `docs/examples/07_stratified_heteroskedastic.ipynb` — a
  non-proportional Cox model (Bernstein-t ⊗ Bernstein-x) with a scaled
  baseline via `Coxph(interacting=..., scaling=...)`: fit and LR test on `γ`,
  interpretation of `Theta_` (control-point values of the stratified
  baseline) and `gamma_` (the `exp(0.5·x_s·γ)` pivot), prediction and
  quantiles with `X_scale_new`, and an R-parity check against the
  `stram_bs_bs_mev_*` fixture. Cross-referenced from notebooks 04 / 05, whose
  stale "scaling + interacting is out of scope" takeaway is corrected. (#111)
- R-parity fixtures `reference/stram_bs_bs_normal_*` for the base
  `ConditionalTransformationModel(basis=InteractionBasis(...), scaling=...)`
  path (`mlt::ctm`, Bernstein-y ⊗ Bernstein-x, Normal base), materialised with
  R 4.6.1 / mlt 1.8.4. The previously skipping ratchets now run: `γ`, `Θ` and
  log-likelihood match R at `rtol=1e-6` and `γ`'s variance at `rtol=1e-4`.
  This MLE has an active monotonicity row, so the R fit is polished with
  mlt's projected `spg` optimiser instead of the unconstrained BFGS step the
  sibling `stram_*` blocks use (the default alabama fit stopped ~5e-5 short
  on `γ` and ~1e-8 infeasible). (#125)

### Fixed

- `score(..., X_scale=...)` now includes scaling for both shift and interaction
  models. Scaled fits require an explicit scaling design, including when scoring
  training data; unscaled fits reject it. Missing or invalid designs raise
  `ValueError` instead of returning an incorrect likelihood or a matrix error.
  (#144)
- Auglag stalled early on `InteractionBasis` fits and reported
  `converged=True`. On a trial step that left the column-wise monotone cone, the
  interaction objective returned `(inf, 0)`. That stalled the inner L-BFGS-B
  line search: on the `stram_bs_bs_mev` fixture the fit stopped at NLL ≈ 2248
  (optimum ≈ 893, ‖∇‖∞ ≈ 366), and the feasible-and-stalled exit accepted the
  unmoved point. Only the Newton-CG polish rescued the fit, leaving Θ about
  1e-5 off the optimum. The objective now matches the shift path: it returns a
  finite penalty (1e10) and the subgradient of the monotonicity violation. The
  stall exit now refits the multipliers at the stalled point (equality rows
  with any sign, active inequality rows with μ ≥ 0) and requires both the
  unexplained gradient and the duality gap to be at most
  `theta_tol^(2/3)·(1 + |f|)` (Gill–Murray–Wright U3, applied to the projected
  gradient); otherwise auglag stops with `converged=False`. An active
  constraint therefore no longer excuses gradient in parameters it does not
  constrain. `Coxph(interacting=...)` with or without
  `scaling=` now reaches the MLE on its first attempt without the polish, and Θ
  / survival / hazard match R at `rtol=1e-6` (previously `1e-4`). Decision
  record: `docs/adr/0004-auglag-stall-acceptance.md`. (#134)
- The finite infeasibility penalty could be returned as a fitted result when a
  valid weighted likelihood exceeded it. With every weight 1e8, the
  interaction fit settled on the 1e10 penalty as "NLL 1e10" and `fit()` raised
  `InfeasibleParameterError` from the Hessian, and a weighted shift-path
  `Coxph` reported log-likelihood −1e10. The penalty is now
  `max(1e10, 1e6·(1 + |NLL(θ_init)|))`, so it stays above the fit's own
  feasible values. Restart selection, the fallback and the polish now score
  every candidate by the unpenalised likelihood and discard any whose θ is
  infeasible. Unweighted fits are unchanged. (#134)
- Regression coverage for the false auglag stall reported in #142
  (Bernstein-y order 6 ⊗ Bernstein-x order 3, n = 1000: `converged=True`, no
  warning, log-likelihood −2604.7 against −1403.9 at the optimum, KKT residual
  114). The #134 fixes above (merged in #136) resolve it;
  `tests/test_interaction_stall.py` now pins the issue's data and configuration
  at n = 1000 and n = 2000: first-attempt convergence with no
  `ConvergenceWarning`, a KKT residual at the
  floating-point floor, and log-likelihood (`rtol=1e-6`) and Θ (`rtol=1e-3`;
  stacked active monotonicity rows leave Θ weakly determined) parity with an
  R `mlt` fit polished with the projected `spg` optimiser (fixtures
  `reference/interaction_stall_*`, data written by
  `reference/write_interaction_stall_data.py`). The stationarity check in
  `TestAuglagFirstAttempt` is now relative to |f| (`1e-8·(1 + |f|)`) instead
  of an absolute 1e-6, which the scaled fit met with 6% headroom and failed
  on another platform. (#142)
- The exponential-link R reference `reference/mlt_exponential_*.txt` came from
  a plain `mlt()` fit that stalled against the `dexp` −Inf barrier while
  reporting `convergence = 0` (stored logLik −154.91), and its test only
  checked `ll_py ≥ ll_R − 0.5`, so it passed trivially. The fixture is now an
  explicit-constraint `alabama::auglag` fit on mlt's own likelihood under the
  lower-support-end rule `θ_b[0] ≥ 0` (ADR 0005), multi-started, with the KKT
  active set and multipliers checked in R and `mlt(fixed = c("Bs1(y)" = 0))` as
  a cross-check (logLik −7.5084028). The max-extreme-value fixture
  `reference/mlt_maxextrval_*.txt` had a milder version of the same symptom:
  no barrier, but plain `mlt()` reported `convergence = 0` about 1.8e-6 short
  in θ (1.8e-8 in logLik) along a flat face where two monotonicity rows are
  active. It is regenerated by the same multi-start route (shared helper
  `.auglag_reference()` in `reference/generate_reference.R`, which enforces
  start agreement and the KKT active set, multipliers and stationarity with
  `stopifnot`), with plain `mlt()` as the cross-check.
  `test_integration_r_reference_new_distributions` now compares mltpy's fitted
  logLik and θ with both fixtures at `rtol=1e-6`; θ uses `atol=1e-10` for
  max-extreme-value and `atol=1e-7` for the exponential case, whose
  `θ_b[0] ≥ 0` row mltpy's auglag resolves to ~1e-8. The auglag
  `exponential_n50_seed43` fixture already enforced the lower-support-end
  rows explicitly and is unaffected. mltpy's fitting code is unchanged. (#156)
- `InteractionBasis` with `base_distribution="exponential"` enforced no support
  rule: only the Kronecker monotonicity rows were built, so the exact-data
  log-likelihood was unbounded and the fit ran off to `h → −∞` (issue
  reproduction: `converged=False`, logLik +7986, `Θ[0] ≈ −3·10⁴`, CDF at `y = a`
  ≈ −1e13). ADR 0005's lower-support-end rule now applies on the unscaled
  interaction path as one linear row per training observation,
  `(a(a) ⊗ b(x_i))ᵀ vec(Θ) ≥ −offset_i`, via
  `build_constraint_matrices_interaction(nonneg_lower=True, X=..., offset=...)`.
  Auglag, SLSQP and trust-constr all honour the non-zero right-hand side, and
  the reproduction now converges at logLik −3.756 with `h(a|x_i) ≥ 0`. Because
  `b(x)` is non-linear in `x`, the training rows do not cover new `x` even
  inside the hull, so `predict` on this path emits the #157 `h(a|x) < 0`
  `UserWarning` (values unclamped), and quantiles for `p < F(a|x)` use the
  support-atom wording, which `simulate()` drops as on the shift path (#159).
  The interaction quantile clip message now comes from the shared helper.
  `scaling=` + `InteractionBasis` + exponential stays rejected. Recorded as an
  amendment to ADR 0005. (#158)
- Shift path + `base_distribution="exponential"`: the support rows ignored
  `offset`, so ADR 0005's rule `h(a|x_i) = θ_b[0] + x_i·β + offset_i ≥ 0` was
  not enforced. A negative offset left the support on every training row
  (issue reproduction, offset −0.5: logLik +94.87 instead of −5.13, in-sample
  CDF at `y = a` of −0.649, `converged=True`). A positive offset pinned the fit
  at `h(a|x_i) = offset_i`, a spurious atom of mass `1 − e^{−offset}` at `a`.
  The rows are now `θ_b[0] + x_i·β ≥ −offset_i`; without covariates they
  collapse to the single row `θ_b[0] ≥ max_i(−offset_i)`. They are built by
  `build_constraint_matrices(..., offset=...)` / `build_constraints(...,
  offset=...)` and honoured by auglag, SLSQP (`S θ − b`) and trust-constr
  (`lb = b`), including under `fixed_params`. Starts and restarts lift
  `θ_b[0]` to `max(0, max_i(−offset_i))`, and the auglag Newton-CG polish
  checks the support rows too. Both signs of the reproduction now reach
  logLik −5.1326 with `θ_b` shifted by exactly `−offset`. New R reference
  `reference/mlt_exponential_offset_*` (per-row offset + covariate): an
  explicit-constraint `alabama::auglag` fit on mlt's likelihood, Newton-polished
  on its active set and cross-checked by an exact LP + `mlt(fixed=)`
  decomposition. All three solvers match it at logLik `rtol=1e-6`. With
  `offset=None` nothing changes. Recorded as an amendment to ADR 0005. (#160)

## [0.4.0] — 2026-05-25

### Performance

- Assembled Bernstein `(B, dB)` caching — `BernsteinBasis.evaluate_with_derivative`
  (and `evaluate`) now memoise the *assembled* design-and-derivative pair on
  `(order, support, y.tobytes())`, alongside the existing `_bernstein_matrix`
  cache. The inner cache already covered the raw power op, but the optimiser's
  hot path still re-ran the support scan and rebuilt `dB` via `np.pad` + subtract
  on every one of the ~130 likelihood evaluations per fit — even though both
  matrices depend only on `y`, never on `θ`. A cache hit now collapses
  normalise + B + dB into a single dict lookup (≈ **12 µs → ≈ 0.2 µs** per call
  at n=50/order=6). Content keying (not `id`) makes the censored `cd.exact[mask]`
  slice pattern — fresh arrays with stable content — hit the cache. Cached
  matrices are read-only; the cache is a bounded LRU (`maxsize=32`) with the same
  thread-safe `move_to_end` hit-path guard. (#95)

### Added

- Tensor-product interaction basis — `InteractionBasis(y_basis, x_basis)`
  enables fully-interacting CTMs `h(y|x) = (a(y) ⊗ b(x))ᵀ vec(Θ)`. Wire it
  into any model via `ConditionalTransformationModel(basis=InteractionBasis(...))`
  or via `Coxph(..., interacting=...)` for the non-proportional / stratified
  survival flavour (crossing survival curves). The shaped coefficient matrix
  is exposed as `model.Theta_`; `coef_` returns it as 2-D. Monotonicity is
  enforced column-wise, `(D ⊗ I_q) @ vec(Θ) ≥ 0`, requiring a non-negative
  partition-of-unity x-basis. Exact (non-censored) data only in this release.
  Design recorded in `docs/adr/0001-tensor-product-interaction-basis.md`;
  worked notebook at `docs/examples/04_interacting_terms.ipynb`.
- Scaling terms (heteroskedastic / scaled-baseline) — `scaling=` kwarg on
  `BoxCox`, `Coxph`, `Colr`, `Lm`, and `Survreg` mirroring R
  `tram::*(scale = ~x_s)`. The transformation becomes
  `h(y|x) = h_0(y)·exp(0.5·x_s·γ) + x_d·β`; the fitted vector gains a `γ`
  block exposed as `model.gamma_` / `feature_names_scaling_`, and `summary()`
  adds a *Scaling coefficients* block with Wald SEs. Predict-side methods take
  a parallel `X_scale_new=` (or `X_scale=` on `survival`/`hazard`). `γ` is
  sign- and magnitude-aligned with R (no flip). Design recorded in
  `docs/adr/0002-scaling-terms.md`; worked notebook at
  `docs/examples/05_scaling_terms.ipynb`.
- `Survreg` — parametric survival model on the log-time scale (R
  `tram::Survreg`). Fits `h(log t)` under the `"weibull"` (default),
  `"lognormal"`, or `"loglogistic"` family. Re-exported via `mltpy.Survreg`
  and R-validated against `tram::Survreg`.
- `Lehmann` — proportional reverse-time hazards model for right-censored data
  (the dual of `Coxph`), using the new `"max_extreme_value"` base distribution
  (standard Gumbel) to realise `-log F(t|x) = h(t) + x'β`. Re-exported via
  `mltpy.Lehmann`.
- `"max_extreme_value"` base distribution (standard / right Gumbel) — the
  reverse-time-hazards link used by `Lehmann`.
- Additional basis families: `OneHotBasis`, `InterceptBasis` (non-negative
  partition-of-unity x-bases for stratified / interaction terms),
  `PolynomialBasis`, `LegendreBasis`, and `LogBasis`. All re-exported from
  `mltpy`.
- Profile-likelihood confidence intervals — `confint(level, parm, type="profile")`
  inverts the χ²₁ likelihood-ratio test by refitting under
  `OptimizerConfig.fixed_params` and brent-q'ing the bracket. The appropriate
  diagnostic when a Bernstein coefficient sits on the monotonicity boundary,
  where Wald widths can be 3–5× too wide. Worked comparison (all three CI
  flavours) at `docs/examples/06_profile_likelihood.ipynb`.
- `wald_test(R, r, vcov, regularize)` for linear restrictions `Rθ = r`,
  returning a `WaldTestResult` dataclass (`statistic`, `df`, `p_value`,
  `vcov_type`). Uses either the inverse-information or the HC0 sandwich
  variance. Re-exported via `mltpy.WaldTestResult`.
- HC0 sandwich standard errors — `sandwich_se()` and `sandwich_vcov()` on
  `ConditionalTransformationModel`.
- `OptimizerConfig.fixed_params` — pin a subset of parameters to fixed values
  during the fit (the mechanism behind profile-likelihood CIs).

### Changed

- Minimum Python version raised from 3.11 to 3.12. `requires-python`,
  the package classifiers, the CI test matrix, the mypy target, and the
  README/CONTRIBUTING requirement statements are now all consistently on
  3.12 (previously the project mixed 3.10/3.11/3.12 across these files).
- `vcov()` and `standard_errors()` gained a `regularize` parameter
  (`'active'` default, also `'auglag'` or `None`). When the observed-information
  Hessian is singular — which happens whenever a monotonicity constraint is
  active at the MLE (`theta[i+1] == theta[i]`) — the default now recovers a
  usable variance via the **active-set-constrained (bordered-KKT) covariance**
  (the top-left block of `inv([[H, Aᵀ_active], [A_active, 0]])`, with a `pinv`
  fallback) instead of raising `RuntimeError`. This is the exact ρ→∞ limit of a
  penalty-augmented Hessian and is independent of the optimiser's final penalty
  `ρ`. Pass `regularize=None` to restore the old raise-on-singular diagnostic
  (#82).

### Performance

- Bernstein design-matrix caching — `basis._bernstein_matrix` is now memoised on
  the byte content of the (normalised) evaluation points and the basis degree.
  The matrix depends only on `y` and the order, never on the coefficients `θ`,
  yet was previously recomputed on every one of the ~hundreds–thousands of
  likelihood/gradient evaluations per fit (≈ 75 % of fit time in profiling).
  Caching it once per fit — the Python analogue of R `mlt` precomputing the
  model matrix — together with the augmented-Lagrangian changes below makes
  `fit()` roughly **10–45× faster** across the benchmark grid; large-`n` cells
  (n=5000) are now **faster than R `mlt`** (geometric-mean 0.90× R's speed
  overall, up from ~30–50× slower). See `benchmarks/results/benchmark_report.md`.
- Augmented Lagrangian now stops early once converged instead of always running
  its full outer-iteration budget (typically ~8–15 outer iterations instead of
  the 50-iteration cap on degenerate active sets).

### Fixed

- Augmented-Lagrangian penalty inflation — the PHR outer loop grew the penalty
  `ρ` toward `rho_max` (1e8) even after the constraints were already satisfied,
  because the shrink test fires on a tiny-vs-tinier residual. The resulting
  ill-conditioning stalled the inner L-BFGS-B solve and *degraded* an
  already-good iterate (KKT residual climbing back from ~1e-5 to ~1e-2). `ρ` is
  now frozen once the iterate is feasible (`feasibility ≤ feas_tol`).
- Spurious convergence failures on degenerate active sets — on stacked
  monotonicity boundaries the augmented-Lagrangian stationarity floors at ~1e-5,
  above `outer_tol`, so fits reported `converged=False` (and burned all 50 outer
  iterations) even though `θ` had stopped moving and matched the reference fit to
  many decimals. The solver now also accepts the `alabama`-style
  feasible-and-stalled convergence point (feasible **and** `‖Δθ‖∞` below
  tolerance between outer iterations), via new `AugLagOptions.feas_tol` /
  `theta_tol`, and returns the best-KKT iterate seen. Every benchmark cell now
  converges 10/10 (previously several at 2–9/10).

## [0.3.0] — 2026-05-17

### Added

- tram convenience layer — `BoxCox`, `Coxph`, `Colr` as thin subclasses of
  `MLT` with hard-coded censoring and base distribution plus domain-specific
  methods (`fitted_transformation`, `survival`, `hazard`) (`tram.py`)
- `"min_extreme_value"` base distribution (reversed Gumbel / standard minimum
  extreme value) — the link that realises the Cox proportional hazards model
  `log[-log S(t)] = h(t)`; `Coxph` now uses it by default
- Systematic validation harness comparing mltpy output against R `mlt`
  reference runs, with XFAIL/XPASS tracking (`validation/`)
- Gradient verification suite — 79 tests across censoring types, base
  distributions, θ positions, and covariate modes, including narrow-interval
  Taylor-fallback and converged-optimum sanity checks
  (`tests/test_gradient_verification.py`)
- Reproducible runtime benchmark vs. R `mlt` — Python and R driver scripts
  consuming a shared seeded dataset, median + IQR per cell across the grid
  `n × order × censoring`, and an auto-generated markdown report with
  environment metadata (`benchmarks/`, `make benchmark`)
- Sphinx documentation with three executed vignettes: BoxCox regression,
  right-censored survival analysis (vs Kaplan–Meier), and heteroscedastic
  regression with covariates (`docs/`)
- GitHub Actions CI/CD pipeline — lint (ruff), type check (mypy strict),
  tests with coverage gate, and documentation build (`.github/workflows/`)
- Optional dependency extras: `[plots]` (matplotlib), `[examples]` (matplotlib,
  pandas, lifelines, jupyter, ipykernel), `[docs]` (Sphinx + theme + nbsphinx)
- `base_distribution` parameter threaded through `CTM.__init__`, `MLT.__init__`,
  `fit()`, `optimize()`, `score()`, and `log_likelihood()`
- `Lm` class — `_TramModel` subclass fixing `order=1`, normal base, and
  uncensored data; exposes `sigma_`, `intercept_`, `coef_`, and
  `fitted_transformation(y)`. Re-exported via `mltpy.Lm`.
  R-validated against `tram::Lm` and `lm()` for both the intercept-only and
  single-covariate cases (`tram.py`, `reference/lm_*`, `tests/test_tram.py`)
- `Polr` (proportional-odds ordinal regression) — subclass of
  `ConditionalTransformationModel` that defers basis construction until
  `fit()` (the response level count `K` is only known then). Uses
  `OrdinalBasis(K)` (degenerate one-hot cutpoint basis) and interval
  censoring; exposes `predict_proba` / `predict_class`. Sign convention:
  mltpy parameterises `h + X·β`, so `Polr.coef_` is the negative of R
  `tram::Polr`'s `beta`.
- PHR augmented Lagrangian solver (`_auglag.py`: `auglag_minimize`,
  `AugLagOptions`, `AugLagResult`) — mirrors R `mlt`'s `alabama::auglag` and
  is now the default `OptimizerConfig.solver`. `OptimizationResult` gains
  `n_outer_iter` and `kkt_residual` (both `None` for legacy solvers).
  `"slsqp"` and `"trust-constr"` remain available as opt-in alternatives.
- Analytical observed information / variance–covariance machinery:
  new private `_d2_logpdf` and per-censoring `_hess_*` / `_scores_*` in
  `likelihood.py`; public top-level `hessian()` and `score_matrix()`;
  eager Hessian computation in `fit()`; new model methods `vcov()`,
  `estfun()` / `score_contributions()`, and `standard_errors()`.
  `_TramModel.summary()` now emits a Wald coefficient table for β.
  R-validated against `vcov(as.mlt(fit))` and `sandwich::estfun(fit)` for
  BoxCox / Colr / Coxph fits (`likelihood.py`, `model.py`, `tram.py`,
  `reference/vcov_*`, `tests/test_vcov.py`)
- Wald confidence intervals (`confint(level, parm)`) and pointwise
  delta-method confidence bands (`confband(y_grid, X, level, what)`) on
  `ConditionalTransformationModel`. `confband` supports
  `what ∈ {trafo, distribution, survivor, density, hazard}` — the Wald
  interval is computed on the appropriate linear-predictor scale (`h` for
  the first three; `log f(h) + log h'` with an optional `− log S(h)` term
  for density / hazard) and back-transformed so probability bands stay in
  `[0, 1]` and density / hazard bands stay positive.
  R-validated via hand-computed delta-method bands on a baseline MLT fit
  plus Wald CIs for BoxCox / Colr / Coxph (`model.py`,
  `reference/confint_*`, `reference/confband_baseline_*`,
  `tests/test_confidence.py`)
- `residuals(type=...)` method on `ConditionalTransformationModel` —
  per-observation diagnostics mirroring R `mlt::residuals`. Three types:
  `"score"` (default; ∂(-ℓ_i)/∂α at α=0 for an artificial intercept added
  to `h(y|x)` — sign matches R `mlt::residuals`), `"cox-snell"`
  (`-log S(y_i|x_i)`, ~Exp(1) under correct model), and `"deviance"`
  (closed form on Cox-Snell, ~N(0,1) under correct model). All censoring
  types (none, left, right, interval) and base distributions are supported;
  the score residual reuses a new public `intercept_score()` helper in
  `likelihood.py`. Cox-Snell evaluates at the observed point regardless of
  censoring (lower for right-cens, upper for left-cens, midpoint for
  interval-cens). R-validated against `residuals(fit)` and
  `-log(predict(fit, type="survivor"))` for BoxCox / Colr / Coxph fits
  (`mltpy/model.py`, `mltpy/likelihood.py`, `reference/residuals_*`,
  `tests/test_model.py::TestResiduals*`)
- Observation weights and offset support across the full model pipeline
  (`fit`, `predict`, `score`, `confband`, `residuals`, `estfun`):
  - `fit(y, X=None, weights=None, offset=None)` — weighted log-likelihood
    `Σ w_i·ℓ_i`; offset adds a per-observation constant to `h(y|x)` before
    all distribution calls. Weights and offset are stored as `weights_` /
    `offset_` and snapshotted as `_weights_train_` / `_offset_train_` for
    later use by `residuals()`.
  - `predict(..., offset_new=None)` — offset shifts `h` at prediction time.
  - `score(..., weights=None, offset=None)` — evaluates the weighted LL at
    any (y, X) pair.
  - `confband(..., offset=None)` — offset shifts `h` before the delta-method
    band is computed; the Jacobian is unaffected (offset is constant in θ).
  - `Coxph.survival(y, X, offset=None)` and `Coxph.hazard(y, X, offset=None)`.
  - New public helper `_validate_weights_offset(weights, offset, n)` in
    `likelihood.py` — raises `ValueError` for wrong shapes, negative weights,
    or non-finite values.
  - R reference generation extended with weighted BoxCox / Colr / Coxph fits
    (`reference/generate_reference.R`), producing `weights_<model>_*.txt`
    files.
  - Full test coverage in `tests/test_weights_offset.py` (41 tests): input
    validation, identity (`weights=ones ≡ no weights`, `offset=zeros ≡ no
    offset`), uniform-doubling invariance, replication invariance,
    offset-shifts-trafo, quantile-with-offset, R-parity (skipped until R
    files are generated).

### Changed

- `Coxph` default base distribution: `"normal"` → `"min_extreme_value"`
  (fixes validation case 06; the previous default did not realise the
  proportional hazards link)
- Minimum Python version raised from 3.10 to 3.11; CI matrix is 3.11 / 3.12
- Default optimizer changed from SLSQP/trust-constr to the new PHR augmented
  Lagrangian (`OptimizerConfig.solver = "auglag"`) for closer parity with
  R `mlt`'s `alabama::auglag`. `OptimizerConfig.max_iter` and `.tol` now
  apply only to the SLSQP/trust-constr paths; auglag has its own outer/inner
  budgets via `AugLagOptions`.
- Right-censored quantile prediction now follows R `mlt::qmlt` semantics:
  quantiles are obtained by inverting a fixed CDF grid (`K=50`) via cubic
  interpolation, with saturation to grid boundaries when targets fall
  outside the finite inversion range. This removes Coxph quantile
  mismatches caused by strict support-bracket root clipping in the previous
  implementation.
- GitHub Actions workflows are temporarily deactivated while the repository
  is private. Workflow files were moved from `.github/workflows/` to
  `.github/workflows-disabled/`. Move them back to re-enable Actions.

### Fixed

- Drop unused `scipy.stats` imports in `model.py` flagged by Ruff F401
- Interval-censored log-likelihood length-mismatch branch now records an
  explicit failure with actual vs expected lengths instead of relying on
  `np.isnan` as an escape hatch

## [0.1.0] — 2026-04-05

### Added

- `CensoredData`, `CensoringType`, `NumericVariable`, `OrderedVariable`, and
  `SurvivalVariable` data types (`variables.py`)
- `BernsteinBasis` with analytical first and second derivative and integral
  (`basis.py`)
- `MonotonicityConstraint` and `BoundaryConstraint`; `build_constraints()`
  helper supporting both SLSQP and trust-constr solvers (`constraints.py`)
- Log-likelihood with analytical gradients for exact, right-censored,
  left-censored, and interval-censored data (`likelihood.py`)
- `optimize()` with SLSQP/trust-constr solver, restart mechanism, and
  `OptimizerConfig` / `OptimizationResult` dataclasses (`optimizer.py`)
- `ConditionalTransformationModel` and `MLT` with scikit-learn-compatible
  `fit` / `predict` / `score` / `simulate` API (`model.py`)
- 211 tests across 6 modules; Hypothesis property-based tests for
  mathematical invariants
