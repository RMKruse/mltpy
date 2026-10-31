# How R `mmlt` parameterises and estimates multivariate transformation models

Research branch for #129 on map #128. It is not for merge, and no library code changes.
Everything below was read from the R sources listed under *Versions read*, or confirmed
by running `mmlt_probe.R` (output in `mmlt_probe.Rout`).

## Versions read

| Component | Version | Where `mmlt` code lives |
|---|---|---|
| `mlt` | **1.8-4** (CRAN tarball `mlt_1.8-4.tar.gz`) | `R/mmlt.R`: `mmlt()`, all methods (1280 lines) |
| `tram` | **1.4-7** (CRAN `tram_1.4-7.tar.gz`, published 2026-10-13) | `R/mmlt.R`: `Mmlt()` wrapper, `confregion`, `HDR`, `perm_test.mmlt` |
| `mvtnorm` | **1.4-2** | `R/lpmvnorm.R`, `R/interface.R`, `R/ltMatrices.R`: likelihood kernels |
| `alabama` | 2025.1.0 | `auglag2`: optimiser and Hessian |
| R | 4.6.1 | — |

Papers: Klein, Hothorn, Barbanti & Kneib (2022), SJS 49:116–142, doi:10.1111/sjos.12501,
read as arXiv:1906.03151v3 (cited as **KHBK**). Hothorn (2024), *On Nonparanormal
Likelihoods*, arXiv:2408.17346v5 (cited as **NPN**). The `mlt` help page cites NPN as the
reference for the current implementation (`mlt/man/mmlt.Rd`, `\value`).

### Version history (where the code lives)

- `mmlt` was added to **tram 0.5-0** (2020-08-27) for continuous responses only (`tram/inst/NEWS.Rd`).
- tram 0.8-0 added alternative link functions and fixed `coef(type = "Sigma"/"Corr")`, which was wrong for J > 3. tram 1.0-0 rewrote the implementation. tram 1.0-1 added shift-scale margins, and tram 1.0-3 added sequential fits and `coef(fixed =)`.
- **tram 1.1-0 / mlt 1.6-0 (2024-09):** `mmlt` moved into **`mlt`**. That release also added arbitrary mixes of continuous and discrete data, exact and censored observations within the same variable, and missing values coded as `(-Inf, Inf)`. `tram` kept only the user-facing `Mmlt()` with its extra `fit =` strategies (`tram/inst/NEWS.Rd` 1.1-0; `mlt/inst/NEWS.Rd` 1.6-0).
- mlt 1.8-0 added `residuals.mmlt` and `update.mmlt` and removed `coef(type = "Lambdapar")`.
- There was never a separate CRAN package named `mmlt`. It started inside `tram` and now lives in `mlt` (`mlt/NAMESPACE` exports `mmlt`; `tram/NAMESPACE` exports `Mmlt`). The probe confirms this (`environmentName(environment(mmlt)) == "mlt"`).
- **No dedicated mmlt vignette ships with tram 1.4-7.** `tram/vignettes/` contains `tram`, `mtram`, `survtram` and `NAMI`. The closest documentation is:
  - the `NAMI.Rnw` vignette, which uses `Mmlt()` for nonparanormal adjusted marginal inference, including discrete and missing margins (§ "Binary outcome");
  - the example in `tram/man/mmlt.Rd` (`cars`);
  - `demo("undernutrition")` (KHBK §4.2 application);
  - `demo("npn")` / `demo("hcc")` (NPN paper);
  - `tram/inst/simulations/sim2d.R`, `sim5d.R` (KHBK §5).

## 1. Dependence parameterisation

**Model (KHBK §2.2, eq. 2; NPN §2).** Each margin has a transformation `h̃_j(y_j | x) = c_j(y_j, x)ᵀϑ_j`, which is an ordinary `mlt` model. The joint model sets

`h_j(y | x) = Σ_{l<j} λ_{jl}(x) h̃_l(y_l | x) + h̃_j(y_j | x)`,

that is, `h(y) = Λ h̃(y)` with `Λ` **unit lower-triangular** (`λ_jj ≡ 1`). Then `h̃(Y) ~ N_J(0, Σ)` with `Σ = Λ⁻¹Λ⁻ᵀ` (KHBK §2.3).

**What is estimated.** The estimated quantities are always the J(J−1)/2 free entries `λ` of the unit lower-triangular Λ. `mlt/man/mmlt.Rd` `\value` says: "The model is always parameterised in terms of a unit lower-triangular matrix Λ."

- There is no Cholesky-of-correlation or correlation-matrix parameterisation, and no constraint on λ. λ is unconstrained in ℝ: only the marginal `ui`/`ci` monotonicity rows exist (`.models`, mlt R/mmlt.R L57–65).
- Parameter names are `"<resp_j>.<resp_l>.<design column>"`, for example `y2.y1.(Intercept)`. They follow the row-major ("byrow") lower-triangle order `b.a, c.a, c.b, d.a, …` (`.start`, L233–236; checked in R).
- The full vector is `θ = [ϑ_1 | … | ϑ_J | λ]`. Marginal blocks come in the order the models were passed. Λ blocks are ordered by entry first, then design column (`.start`, L228–239).

**Two identifications (`conditional =` argument).** Both use the same likelihood and the same λ; they differ in how Λ is turned into Ω (NPN §2: Ω⁽¹⁾ and Ω⁽²⁾).

- `conditional = FALSE` (**the default**, class `"mmmlt"`): Ω = Λ·diag(D), with D_j = sqrt((Λ⁻¹Λ⁻ᵀ)_jj). This makes `Σ = Ω⁻¹Ω⁻ᵀ` a **correlation matrix**, so each `h̃_j` is *marginally* N(0,1) and marginal parameters keep their univariate interpretation.
  - Code: `standardize = !conditional` (`.mmlt_setup` L341). The standardization is done by `mvtnorm::standardize(invchol =)` → `invcholD` (mvtnorm R/ltMatrices.R L934–959, which scales column j by D_j).
  - The help page writes this as `Ω = Λ diag(Λ⁻¹Λ⁻ᵀ)`. It omits the `^{1/2}`; the code and NPN §2 have the square root.
- `conditional = TRUE` (class `"cmmlt"`): Ω = Λ. This is the original KHBK parameterisation. Margins are then `Φ(h̃_j/σ_j)` (KHBK §2.3, eq. 4).
  - Allowed **only if every margin is probit** (`stop("Conditional models only available for marginal probit-type models.")`, L333–335).
  - The help text says this is "the default as described by Klein et al.", but the code default is `FALSE` (`mmlt` signature L704; `Mmlt` L2).
- Probe §1b confirms both: the same data gives the same logLik (Δ = −6.9e−08, which is optimiser tolerance) and the same λ (−0.78106 vs −0.78106). The y2 Bernstein coefficients differ by the factor σ_2 = 1.2689.

**How correlations are derived (`.coef.mmlt`, L875–923).**
- Λ(x) is built as `t(X %*% prm)`, then `as.invchol(ltMatrices(..., byrow = TRUE, diag = FALSE))` (implicit unit diagonal). For `mmmlt` it is then standardized to Ω.
- `coef(type =)` returns:
  - `"Lambda"` = Λ; `"Omega"` = Ω; `"Lambdainv"`, `"Omegainv"`;
  - `"Precision"` = ΩᵀΩ (`invchol2pre`); `"PartialCorr"` (`invchol2pc`);
  - `"Sigma"` = Ω⁻¹Ω⁻ᵀ (`invchol2cov`); `"Corr"` (`invchol2cor`);
  - `"Spearman"` = `6·asin(r/2)/π`; `"Kendall"` = `2·asin(r)/π`, with r = Corr;
  - `"all"` = θ; `"marginal"` (mmmlt) / `"conditional"` (cmmlt) = list of the ϑ_j.
- With a non-constant `formula`, every type except `"all"`/`"marginal"` **requires `newdata`** and returns one matrix per row (L903–906; probe §2).
- **Sign convention:** corr₂₁ = −λ₂₁/√(1+λ₂₁²). A *negative* λ means a *positive* correlation (probe §1a: λ = −0.7811 → r = 0.6156).
- For `mmmlt`, `"Sigma"` and `"Corr"` coincide (probe §1a).

**Is it a Gaussian copula by construction? Yes.** Every likelihood path goes through `mvtnorm::mvnorm(invchol = Lambda)`: `logLik` / `lLgrad` for continuous data, `lpmvnorm` for discrete data (`.ll`, L244–308). No other copula family exists in the code.
- KHBK §2.3 shows the unconditional model *is* a Gaussian copula with flexible margins. KHBK §2.6 (last paragraph) leaves non-Gaussian dependence "for future research". NPN §1 frames the model as "latent Gaussian copulae" (nonparanormal).

**Non-Gaussian marginal links (logit, cloglog, …) are supported through the Gaussian copula** (KHBK §2.6). Each margin's `h̃_j` is mapped to the latent normal scale by `z_j = Φ⁻¹(F_Z(h̃_j))`, done in log space as `qnorm(todistr$p(tr, log.p = TRUE), log.p = TRUE)` (`.mget` `what = "z"`, L153–157).
- The Jacobian is handled in the same function: `zprime = f_Z(h̃)·h̃′ / φ(z)` (L184–189).
- For interval data the factors are `dzleft` / `dzright` (L163–183).
- The flag `models$normal[j]` (`todistr$name == "normal"`, L16) short-circuits this mapping for probit margins.
- Probe §3 fits a Lm (normal) + Coxph (min-extreme-value, right-censored) + Polr (probit, ordinal) model under `conditional = FALSE`. It converges (`convergence: 0`).

## 2. Estimation

**Joint ML is the default; two-stage is available.** `mmlt(..., domargins = TRUE)` maximises the full likelihood over all of θ. In `mmlt()`, L804–859, the sequence is:

1. Marginal ϑ_j start at the **separately fitted marginal models' coefficients** (`models$mcoef`, L809–810). The input models must already be fitted `mlt`/`tram` objects.
2. λ starts at 0 (L807). When all margins are continuous, the observations share one exact/censored pattern and `formula = ~ 1`, λ instead starts analytically: `L = solve(t(chol(S)))`, rescaled to unit diagonal, where `S` is the ML covariance (`var·(N−1)/N`) of the marginal `h̃_j` (L817–831). Probe §1e reproduces it: start λ₂₁ = −0.78002.
3. A **pseudo-likelihood fit** of λ only, with all marginal parameters fixed (L832–846).
4. A **joint fit** of all parameters starting from (3) (L854–859).
5. `conditional = TRUE` first runs the unconditional pseudo fit, then rescales the marginal starting values by `sqrt(rowMeans(diag(Σ)))` (L788–801).

`tram::Mmlt(fit = )` exposes these and other strategies (tram R/mmlt.R L2–90; help `tram/man/mmlt.Rd`; NPN §4 "three convex approximations"):
- `"jointML"` (default) = `mmlt()`.
- `"pseudo"` = `mmlt(domargins = FALSE)`: **two-step** (margins fixed at the marginal fits, λ estimated). This is NPN §4 approximation 1. Probe §1e: logLik −851.019 vs joint −850.941; λ −0.78003 vs −0.78106; margins verified unchanged.
- `"ACS"` = alternate convex search, starting from pseudo, `ACSiter` times: margins | λ, then λ | margins (NPN §4 approximation 2; KHBK not involved).
- `"sequential"` = add margins one at a time, fixing earlier ones (NPN §4 approximation 3).
- `"none"` = `dofit = FALSE` (an unfitted shell with `logLik` / `score`).
- `fixed =` pins any named parameters. `fixed = c("y2.y1.(Intercept)" = 0)` reproduces the independence model, whose logLik equals the sum of the marginal logLiks (probe §1e: −922.308 both).

NPN Theorem 1: the continuous ("flow") likelihood is **biconvex**, not convex, in (ϑ, λ). Theorem 2: the discrete NPN likelihood is convex in θ given λ but not necessarily in λ. This is why pseudo/ACS/sequential exist "at the very least" as starting-value devices (NPN §4).

**Constraints.**
- Only the marginal monotonicity rows: `ui = bdiag(ui_1, …, ui_J)` padded with zero columns for λ, rows with infinite `ci` dropped (`.models` L57–65; `.mmlt_optimfct` L604–618).
- λ is unconstrained. Fixed parameters are removed and `ci` is shifted accordingly (L606–610).
- If no free parameter carries a constraint, `ui`/`ci` are set to `NULL` (L614–617).

**Optimiser.** The default is `optim = mltoptim(hessian = TRUE)`, which yields the list `auglag`, `constrOptim`, `optim` (mlt R/mltoptim.R L250–251). These are tried in order until one returns `convergence == 0`; otherwise a warning "Optimisation did not converge" is issued (`.mmlt_optimfct` L620–629).
- `auglag` = `alabama::auglag` with `kkt2.check = hessian` (TRUE), inner `method = "BFGS"`, `eps = abstol = 1e-7`, `itmax = 1000`, `maxit = 500` (mltoptim.R L2–8, L56–66).
- **Without constraints, the "auglag" slot actually calls `BB::BBoptim` (spg)** (mltoptim.R L67–75). This applies to pseudo fits and to all-`Lm`/unconstrained margins.
- KHBK "Computational Details" confirms auglag with starting values from the marginal models.
- Optional internal parameter scaling is controlled by `scaleparm = FALSE` (default) and the `scl` vector (L535–543).

**Gradients are analytical. The Hessian is numerical.**
- The score (`.mmlt_setup$sc`, L400–510) chains `mvtnorm::lLgrad` (scores w.r.t. `obs` / `lower` / `upper` / Λ) through each margin's basis (`trafoprime`) and through the `z = Φ⁻¹(F_Z(h̃))` map.
- For discrete or censored dimensions, `slpmvnorm` returns the **exact gradient of the Genz-approximated log-probability**, not an approximation of the true score (NPN §3, "Evaluation of Log-likelihood and Score Functions").
- KHBK eq. 5–6 give closed-form scores for the all-continuous probit case.
- Probe §1c: the analytic score matches `numDeriv::grad` to 9e−08 off-optimum. Probe §3, mixed case: 9e−07 at the optimum.
- **The Hessian** is `alabama`'s `optim_hessian` = `numDeriv::jacobian` of the *augmented-Lagrangian* gradient at the solution (alabama `auglag2`, `if (kkt2.check) a$hessian <- jacobian(x = a$par, func = gradient)`).
  - That gradient includes the penalty term `σ·ui_actᵀui_act` for active constraints.
  - If no `optim_hessian` exists (for example after BBoptim), `Hessian.mmlt` uses `numDeriv::hessian(-logLik)` (L980–993).
  - `vcov.mmlt` = `solve(H)`, retrying with ridge `λ·I`, `λ = 1e−6·(step−1)`, up to 3 times with a warning (L997–1018).
- **Consequence (probe §1d):** with an active Bernstein monotonicity row (slack 0 at `y1.Bs2 = y1.Bs3`), `vcov()` SEs for y1's coefficients are 0.107 vs 0.271/0.433 from the plain numerical Hessian of −logLik. The two Hessians differ by a factor 4.3e3 (max-norm). Parameters away from the active row agree to about 1e−4; λ: 0.07330 vs 0.07354. For the unconstrained pseudo fit (no active rows) the two agree exactly (0.0623697).

## 3. Censoring, discrete margins and missing values

**Supported: any mix, observation by observation (mlt ≥ 1.6-0).**
- A margin is "continuous" for an observation when it is exact (`eY`). Otherwise it is "discrete" (`iY`, with interval bounds `(cleft, cright]`); this covers ordinal/count levels, right/left/interval censoring, and missing values (`.models` L24–35; "censoring and discreteness are treated the same here").
- `mmlt()` builds a per-observation pattern factor `cdpat` from which margins are interval-type (L717–723). Each pattern level gets its own `.mmlt_setup`, the log-likelihood contributions are concatenated back into the original order (L725–770), and the λ / ϑ parameters are shared.
- A missing response is coded by `R()` as `(-Inf, Inf)` and integrated out (mlt NEWS 1.6-0; NAMI vignette, "Binary outcome" section: "this is not a complete case analysis").

**Likelihood per observation (NPN §3, "Mixed Nonparanormal Log-likelihood"; `.mmlt_setup$ll` L366–398; `mvtnorm::logLik.mvnorm` / `ldpmvnorm`).** Write `c` for the exact block and `d` for the interval block.
- `ℓ_i = log φ_{Ω_A}(z_c) + Σ_c log z′_c + log P(lower_d < Z_d ≤ upper_d | Z_c = z_c)`.
- The first term is the marginal normal density of the exact block. The second is the Jacobian. The third uses the conditional normal of the interval block given the exact block, via `marg_mvnorm` / `cond_mvnorm` (mvtnorm R/lpmvnorm.R `ldpmvnorm`; interface.R L245–336, which also permutes when exact and interval margins are interleaved).
- All-continuous: closed form (`ldmvnorm`), **no integration**.
- **dJ = 1 interval dimension:** exact univariate `Φ(b) − Φ(a)`; `lpmvnorm` sets `M = 1` for J = 1 (lpmvnorm.R L106–118).
- **dJ ≥ 2:** Genz (1992) separation-of-variables integration with `M` draws `W ∈ [0,1]^{dJ−1}` (NPN §3 formula for `d_j(W)`, `e_j(W)`). It is computed in C with the score by the chain rule.
  - The weights come from `args = list(seed = 1, M = 1000)`. `.MCw` draws `matrix(runif((dJ−1)·M), ncol = M)` after `set.seed(seed)` (`type = "MC"`), or a generalized Halton sequence with `type = "ghalton"`, which needs `qrng` (L198–220, L343–356).
  - The same `w` is reused for every observation and every optimiser iteration, so the objective is deterministic **given** `w`.
  - The original KHBK (§6) listed censored/discrete data as future work.

## 4. Covariate-dependent dependence

- Argument `formula =` (default `~ 1`) to `mmlt` / `Mmlt`. It accepts a formula or any `basefun` basis object, for example `Bernstein_basis(numeric_var("cage", …), order = 6)` in `demo("undernutrition")`.
- It is converted with `as.basis(formula, data)`, and `lX = model.matrix(bx, data)` (L314–329). For each lower-triangular entry, `λ_{jl}(x) = lX(x)ᵀ γ_{jl}`, with the **same design for all J(J−1)/2 entries** (`Lambda[] <- t(lX %*% .Xparm(parm))`, L358–375). This is the KHBK §4.1 linear form `λ_{jl}(x) = α_{jl} + xᵀγ_{jl}` and NPN eq. 3.
- With `conditional = TRUE` plus non-constant `formula`, a warning says these models are "order-dependent" (L326–328).
- The score multiplies the Λ-score by the design columns (L433–438). λ stays unconstrained.
- Probe §2 (`formula = ~ x`): λ₂₁(x) = −0.7157 − 0.1376·x; Corr at x = 0 / 0.5 / 1 is 0.582 / 0.617 / 0.649.
- Margins carry their own covariates through their own `mlt` models, for example `BoxCox(stunting2 | cage ~ 1)` (a response-varying / interaction basis) in `demo("undernutrition")`.

## 5. Public API and outputs

**Fitting.**
- `mlt::mmlt(..., formula = ~1, data, conditional = FALSE, theta = NULL, fixed = NULL, scaleparm = FALSE, optim = mltoptim(hessian = TRUE), args = list(seed = 1, M = 1000), dofit = TRUE, domargins = TRUE)` (L704).
- `tram::Mmlt(...)` has the same signature plus `fit = c("jointML", "pseudo", "ACS", "sequential", "none")`, `ACSiter = 2`, and `args$type`.
- `...` takes the fitted marginal models. They must share `nobs` and weights (`.models` L18–22). Any `mlt`/`tram` model works: BoxCox, Colr, Coxph, Lm, Polr, Survreg, stram scale models (tram NEWS 1.0-1), and response-varying margins.
- `tram/tests/mmlt-interface.R` checks order invariance of `coef(type = "Omega")` under permutation of the margins, with mixed Lm/right-censored/Polr margins and case weights.

**Methods** (mlt NAMESPACE L96–112; tram `summary.Mmlt`, `confregion`, `HDR`, `perm_test.mmlt`):

| Method | What it does | Deterministic? |
|---|---|---|
| `coef(type = "all"/"marginal"/"conditional"/"Lambda"/"Omega"/…/"Corr"/"Spearman"/"Kendall", newdata, fixed)` | §1 above | **Yes**: closed-form functions of θ |
| `logLik(object, parm, w, newdata)` | Sum of weighted contributions; `df = length(par)` | All-continuous or dJ ≤ 1: **yes**. dJ ≥ 2: deterministic given `w` (seed) |
| `estfun` / `residuals` (`Xmult = FALSE`) | Analytical per-observation scores | Same as logLik |
| `Hessian`, `vcov` | Numerical Jacobian of the AL gradient (or of −logLik); see §2 | Numerical-derivative noise; depends on the active-set penalty |
| `summary.Mmlt` | `multcomp::cftest` Wald z-tests on the free parameters | Inherits vcov |
| `predict(type = "trafo")` | `Ω · z` (joint, L1135–1141) or the marginal `h̃_j` (`margins = j`) | **Yes** |
| `predict(type = "density")` | `ldmvnorm(z; Ω) + Σ log z′` (L1172–1187); all margins must be continuous | **Yes** |
| `predict(type = "distribution"/"survivor")`, J ≥ 2 margins | `lpmvnorm` with `args = object$args` (L1142–1171) | **No**: MC. Probe §1f: seed 1 vs 2 differ by up to 6.6e−3, and |MC − TVPACK exact| = 3.2e−3 at M = 1000 |
| `predict(margins = j, …)` | Univariate `predict.ctm` with the joint coefficients (mmmlt) or rescaled by σ_j (cmmlt, L1084–1127) | **Yes** |
| `simulate(nsim, seed, newdata, K = 50)` | `Z ~ N(0, I)`, `Ω⁻¹Z`, then per-margin inversion on a K-point grid with spline/approx (`.invf`, mlt R/dpq_etc.R L206+) (L1190–1263; KHBK Algorithm 1) | RNG + grid interpolation: distributional checks only |
| `update(weights, theta, fixed)`, `coef<-`, `mkgrid`, `variable.names` | Utilities (mlt 1.8-0) | — |
| `confregion` (J = 2 ellipse on h-scale, J > 2 random directions), `HDR` (simulation) | tram only | Partly random |

No `confint.mmlt` exists. The NAMI vignette builds Wald CIs with `multcomp::glht(m, coef. = …)`.

**Good deterministic parity targets for a Python port:**
1. all-continuous `logLik` at a given θ (closed form);
2. `estfun` / score at a given θ;
3. the fitted θ / λ / `coef(type = "Corr")` for all-continuous fits;
4. `predict(type = "density"/"trafo")`;
5. marginal `predict(margins = j)`;
6. the dJ = 1 case: one censored or ordinal margin per observation, for example (continuous, right-censored), which uses exact univariate normal probabilities.

For dJ ≥ 2 the target is reproducible only if the **same `w` matrix** is used. R's default is literally `set.seed(1); matrix(runif((dJ−1)·M), ncol = M)` (probe §4: `identical(f_def$args$w, W1)`, identical logLik).
- Exporting `w` from R makes those cases deterministic as well.
- Changing the seed moves the logLik by about 8e−3 and the coefficients by about 3.5e−3 (probe §4). Matching at `rtol = 1e−6` is impossible across different draws.
- **Caveat (probe §3):** passing an explicit `args$w` fails when the observation patterns mix dJ = 1 and dJ ≥ 2 subsets. The explicit `w` is forwarded to the dJ = 1 subset, where `lpmvnorm` rejects it ("Length of W incorrect"; `.mmlt_setup` L347–353 vs lpmvnorm.R L106–118). There, only the seed route works.
- `vcov` / SEs are parity targets only up to numerical-derivative accuracy, and only when no monotonicity row is active (§2).

## Implications for the map's open decisions (facts only)

**#130 public surface.**
- R splits the surface into a core fitter `mlt::mmlt(...)` that takes *already fitted* marginal models and a convenience `tram::Mmlt(fit = …)` that adds the estimation strategies.
- Outputs are `coef(type = …)` with 10 matrix-valued types plus `"all"` / `"marginal"`/`"conditional"`, `logLik`, `estfun`, `residuals`, `vcov`, `predict(type = trafo/distribution/survivor/density, margins =)`, `simulate`, `update`, and `summary` (Wald via `multcomp`).
- `coef(type)` and `predict` need `newdata` whenever `formula` is non-constant.
- Parameter layout: `[ϑ_1 | … | ϑ_J | λ]` with byrow lower-triangular ordering and names `resp_j.resp_l.term`.
- The class encodes the identification (`mmmlt` vs `cmmlt`).

**#131 joint vs two-stage.**
- R's default is **joint ML**, but it internally runs a two-stage (pseudo) fit first as the starting value (`mmlt` L804–859).
- Two-stage (`fit = "pseudo"`), ACS and sequential exist as opt-ins.
- The joint likelihood is biconvex at best (NPN Thm 1–2). In the probe, joint and pseudo λ differ by 1e−3, so they are not interchangeable as parity targets.
- Joint ML is also what gives proper SEs for the marginal parameters (NPN §4, NAMI vignette).

**#132 Gaussian-only vs pluggable copula.**
- R implements **only** the Gaussian copula, parameterised by a unit lower-triangular Λ (`mvtnorm::mvnorm(invchol =)`). There is no copula abstraction in the code.
- Non-normal marginal links are handled by `z = Φ⁻¹(F_Z(h̃))` inside the Gaussian copula, which requires `conditional = FALSE`.
- KHBK §2.6 leaves non-Gaussian dependence to future work. A pluggable copula would therefore have **no R reference** to validate against.

**#133 validation against R.**
- Deterministic targets: §5 items 1–6.
- MC-dependent targets: dJ ≥ 2 likelihoods and `predict(type = "distribution"/"survivor")`. These need the `w` matrix exported from R, or, for prediction, a deterministic bivariate/trivariate reference (TVPACK/Miwa) plus a tolerance argument.
- `vcov` in R comes from a numerically differentiated augmented-Lagrangian gradient. It disagrees with the plain observed information when constraints are active (probe §1d), which is the same issue mltpy already documents for profile CIs.
- Optimiser: `alabama::auglag` (BFGS inner, `eps = 1e-7`), or `BB::BBoptim` when unconstrained. The current mltpy auglag mirrors `alabama`.
- The earlier proposal's "ρ tolerance 0.05" is far looser than what is achievable for continuous cases: on the same data R's two identifications agree on λ to about 4e−6.

**Fog: "Not yet specified" on #128.**
- *Censoring combos:* R supports every combination per observation, including mixed exact/censored within one variable and missing-as-`(-Inf, Inf)`. Exactly one interval dimension per observation needs no Monte Carlo; two or more need Genz QMC/MC plus its exact score (`slpmvnorm`). An "exact + right-censored" bivariate first release would fall entirely in the deterministic dJ ≤ 1 regime.
- *Covariate-dependent dependence:* `formula = ~ x` (or a basis), linear in a shared design for all λ entries, unconstrained. It is cheap in R's code path (only a design multiply in ll/score).
- *Margin kinds:* R accepts any `mlt` model: shift, scale (stram, tram ≥ 1.0-1), response-varying/interaction margins (undernutrition demo), Polr ordinal (tested with `method = "probit"`, and non-probit via the z-map), and right/interval-censored margins. `conditional = TRUE` is restricted to probit margins.
- *Inference:* Wald via `vcov` (numerical Hessian). R has no profile CIs for λ. `perm_test.mmlt` (experimental, `coin`) and parametric bootstrap (KHBK §3.4, `demo("undernutrition")` samples from N(θ̂, V̂)) are the alternatives.
- *Worked notebooks:* the R material that could be mirrored is `tram/man/mmlt.Rd` (bivariate `cars`, BoxCox margins, LR test vs independence), `demo("undernutrition")` (trivariate, covariate-dependent Λ) and NAMI (mixed discrete/survival).
- *CI coverage:* each all-continuous fit took well under a second in the probe; the mixed three-margin fit with M = 1000 also ran in seconds (whole probe runs in about 10 s).

## Files

| File | Content |
|---|---|
| `mmlt_probe.R` | R experiment: coef types, identifications, score check, vcov vs numerical Hessian, pseudo/ACS, predict MC vs TVPACK, `formula = ~ x`, mixed continuous/right-censored/ordinal, reproducing the default `w` |
| `mmlt_probe.Rout` | Its output (R 4.6.1, mlt 1.8-4, tram 1.4-7, mvtnorm 1.4-2) |

Reproduce: `Rscript research/mmlt-internals/mmlt_probe.R > research/mmlt-internals/mmlt_probe.Rout 2>&1`.
This needs `tram`, `numDeriv`, `sandwich` and `TH.data` (a `tram` dependency).
