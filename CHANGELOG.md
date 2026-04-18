# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
