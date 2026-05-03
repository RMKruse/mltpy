# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Observation weights and offset support across the full model pipeline
  (`fit`, `predict`, `score`, `confband`, `residuals`, `estfun`):
  - `fit(y, X=None, weights=None, offset=None)` — weighted log-likelihood
    `Σ w_i·ℓ_i`; offset adds a per-observation constant to `h(y|x)` before
    all distribution calls.  Weights and offset are stored as `weights_` /
    `offset_` and snapshotted as `_weights_train_` / `_offset_train_` for
    later use by `residuals()`.
  - `predict(..., offset_new=None)` — offset shifts `h` at prediction time.
  - `score(..., weights=None, offset=None)` — evaluates the weighted LL at any
    (y, X) pair.
  - `confband(..., offset=None)` — offset shifts `h` before the delta-method
    band is computed; the Jacobian is unaffected (offset is constant in θ).
  - `Coxph.survival(y, X, offset=None)` and `Coxph.hazard(y, X, offset=None)`.
  - New public helper `_validate_weights_offset(weights, offset, n)` in
    `likelihood.py` — raises `ValueError` for wrong shapes, negative weights,
    or non-finite values.
  - R reference generation extended with weighted BoxCox / Colr / Coxph fits
    (`reference/generate_reference.R`), producing `weights_<model>_*.txt` files.
  - Full test coverage in `tests/test_weights_offset.py` (41 tests): input
    validation, identity (`weights=ones ≡ no weights`, `offset=zeros ≡ no
    offset`), uniform-doubling invariance, replication invariance, offset-shifts-
    trafo, quantile-with-offset, R-parity (skipped until R files are generated).

- `residuals(type=...)` method on `ConditionalTransformationModel` —
  per-observation diagnostics mirroring R `mlt::residuals`.  Three types:
  `"score"` (default; ∂(-ℓ_i)/∂α at α=0 for an artificial intercept
  added to `h(y|x)` — sign matches R `mlt::residuals`),
  `"cox-snell"` (`-log S(y_i|x_i)`, ~Exp(1) under correct model),
  and `"deviance"` (closed form on Cox-Snell, ~N(0,1) under correct
  model).  All censoring types (none, left, right, interval) and base
  distributions are supported; the score residual reuses a new public
  `intercept_score()` helper in `likelihood.py`.  Cox-Snell evaluates
  at the observed point regardless of censoring (lower for right-cens,
  upper for left-cens, midpoint for interval-cens).
  R-validated against `residuals(fit)` and
  `-log(predict(fit, type="survivor"))` for BoxCox / Colr / Coxph fits
  (`pymlt/model.py`, `pymlt/likelihood.py`,
  `reference/residuals_*`, `tests/test_model.py::TestResiduals*`)
- `Lm` class — `_TramModel` subclass fixing `order=1`, normal base, and
  uncensored data; exposes `sigma_`, `intercept_`, `coef_`, and
  `fitted_transformation(y)`. Re-exported via `pymlt.Lm`.
  R-validated against `tram::Lm` and `lm()` for both the intercept-only and
  single-covariate cases (`tram.py`, `reference/lm_*`, `tests/test_tram.py`)
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
  interval is computed on the appropriate linear-predictor scale (``h`` for
  the first three; ``log f(h) + log h'`` with an optional ``− log S(h)``
  term for density / hazard) and back-transformed so probability bands
  stay in `[0, 1]` and density / hazard bands stay positive.
  R-validated via hand-computed delta-method bands on a baseline MLT fit
  plus Wald CIs for BoxCox / Colr / Coxph (`model.py`,
  `reference/confint_*`, `reference/confband_baseline_*`,
  `tests/test_confidence.py`)

### Changed

- GitHub Actions workflows are temporarily deactivated while the repository is
  private. Workflow files were moved from `.github/workflows/` to
  `.github/workflows-disabled/`. Move them back to re-enable Actions.

## [0.3.0] — 2026-04-17

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

### Changed

- `Coxph` default base distribution: `"normal"` → `"min_extreme_value"`
  (fixes validation case 06; the previous default did not realise the
  proportional hazards link)
- Minimum Python version raised from 3.10 to 3.11; CI matrix is 3.11 / 3.12

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
