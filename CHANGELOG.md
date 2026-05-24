# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-05-24

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
  `"lognormal"`, or `"loglogistic"` family. Re-exported via `pymlt.Survreg`
  and R-validated against `tram::Survreg`.
- `Lehmann` — proportional reverse-time hazards model for right-censored data
  (the dual of `Coxph`), using the new `"max_extreme_value"` base distribution
  (standard Gumbel) to realise `-log F(t|x) = h(t) + x'β`. Re-exported via
  `pymlt.Lehmann`.
- `"max_extreme_value"` base distribution (standard / right Gumbel) — the
  reverse-time-hazards link used by `Lehmann`.
- Additional basis families: `OneHotBasis`, `InterceptBasis` (non-negative
  partition-of-unity x-bases for stratified / interaction terms),
  `PolynomialBasis`, `LegendreBasis`, and `LogBasis`. All re-exported from
  `pymlt`.
- Profile-likelihood confidence intervals — `confint(level, parm, type="profile")`
  inverts the χ²₁ likelihood-ratio test by refitting under
  `OptimizerConfig.fixed_params` and brent-q'ing the bracket. The appropriate
  diagnostic when a Bernstein coefficient sits on the monotonicity boundary,
  where Wald widths can be 3–5× too wide. Worked comparison (all three CI
  flavours) at `docs/examples/06_profile_likelihood.ipynb`.
- `wald_test(R, r, vcov, regularize)` for linear restrictions `Rθ = r`,
  returning a `WaldTestResult` dataclass (`statistic`, `df`, `p_value`,
  `vcov_type`). Uses either the inverse-information or the HC0 sandwich
  variance. Re-exported via `pymlt.WaldTestResult`.
- HC0 sandwich standard errors — `sandwich_se()` and `sandwich_vcov()` on
  `ConditionalTransformationModel`.
- `OptimizerConfig.fixed_params` — pin a subset of parameters to fixed values
  during the fit (the mechanism behind profile-likelihood CIs).

### Changed

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
- Systematic validation harness comparing pymlt output against R `mlt`
  reference runs, with XFAIL/XPASS tracking (`validation/`)
- Gradient verification suite — 79 tests across censoring types, base
  distributions, θ positions, and covariate modes, including narrow-interval
  Taylor-fallback and converged-optimum sanity checks (`gradient_validation/`)
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
  `fitted_transformation(y)`. Re-exported via `pymlt.Lm`.
  R-validated against `tram::Lm` and `lm()` for both the intercept-only and
  single-covariate cases (`tram.py`, `reference/lm_*`, `tests/test_tram.py`)
- `Polr` (proportional-odds ordinal regression) — subclass of
  `ConditionalTransformationModel` that defers basis construction until
  `fit()` (the response level count `K` is only known then). Uses
  `OrdinalBasis(K)` (degenerate one-hot cutpoint basis) and interval
  censoring; exposes `predict_proba` / `predict_class`. Sign convention:
  pymlt parameterises `h + X·β`, so `Polr.coef_` is the negative of R
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
  (`pymlt/model.py`, `pymlt/likelihood.py`, `reference/residuals_*`,
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
