# Can R `mlt` fit an exponential-link CTM with a scaling term?

Research branch for #126. It is not for merge and changes no library code.
Everything below comes from two sources:

- installed R sources, read by deparsing functions, for example `Rscript -e 'mlt:::.mlt_setup'`;
- runs of `probe.R`, whose output is in `probe.Rout`. Section numbers below (**§n**) point into that file.

`crosscheck.py` (output in `crosscheck.out`) evaluates mltpy's own `log_likelihood` at the
R-fitted θ.

## Versions read

| Component | Version |
|---|---|
| `mlt` | 1.8-4 |
| `tram` | 1.4-7 |
| `basefun` | 1.2-6 |
| `variables` | 1.1-2 |
| `alabama` | 2025.1.0 |
| R | 4.6.1 |
| mltpy | this worktree, `src/` at `3e587fc` |

## Short answer

1. **R does fit the model.** `ctm(response, shifting =, scaling =, todistr = "Exponential")` builds it without complaint (§2), and `mlt()` will try to fit it.
2. **R has no support constraint.** It never adds a row for `h ≥ 0`. The only thing that keeps `h` non-negative is that `dexp(h, log = TRUE) = -Inf` for h < 0. That acts as a barrier on the *observed* rows, `h(y_i | x_i) ≥ 0`. It does not act on the lower end of the support, `h(a | x_i) ≥ 0`, which is what mltpy enforces.
3. **The barrier stops `mlt()` before the optimum, and it still reports success.** R's score uses the smooth extension `∂log f/∂h = -1` everywhere. The gradient therefore always points into the `-Inf` region, the line search backtracks to a zero step, and `mlt()` returns `convergence = 0` near its starting value.
   - On a shift+scale fixture this gives logLik **−185.45**. The real constrained optimum is **−28.12**, and even the true data-generating θ scores −31.09 (§4b, §9).
   - The default starting values are infeasible, so a fit without a user-supplied θ ends at `-Inf` with `convergence = 1` (§3a, §4a, §5a).
   - The committed no-covariate fixture `reference/mlt_exponential_*.txt` is stuck the same way: stored logLik −154.91, real optimum −2.35 (§11).
4. **One case gives clean, exact parity: scaling only (no shift term).** There, mltpy's constraint `θ_b[0]·exp(0.5·x_s·γ) ≥ 0` reduces to the linear `θ_b[0] ≥ 0`, and it is always active at the optimum. So `mlt(..., fixed = c("Bs1(y)" = 0))` is exactly mltpy's MLE, and R fits it with no barrier contact (§10, §12). `tram::Aareg(y ~ 0 | x_s)` is the same model (§3c, §10).
5. **For shift+scale, R's own optimiser gives no usable reference.** A reference can still be computed in R from mlt's likelihood and score, using `alabama::auglag` with mltpy's non-linear `hin` rows written out explicitly (§9). That checks mltpy's optimiser, not R's modelling choice.

---

## 1. Does `ctm()` accept a scaling term with `todistr = "Exponential"`? Yes.

`mlt::ctm` has the signature `ctm(response, interacting = NULL, shifting = NULL, scaling = NULL,
scale_shift = FALSE, data = NULL, todistr = c("Normal", …, "Exponential", …), …)`.
- Nothing in its body ties `todistr` to `scaling`. It calls `.distr(todistr)` and concatenates the bases `c(bresponse, [binteracting,] bshifting, bscaling)`.
- §2 builds `ctm(Bernstein_basis(…, ui = "increasing"), shifting = ~ x_d, scaling = ~ x_s, todistr = "Exponential")` without warning (`model$todistr$name == "exponential"`, `scale_shift = FALSE`).

**The "no tram class pairs them" claim is only half true.**
- **`tram::tram(y ~ x_d | x_s, distribution = "Exponential", transformation = "smooth")`** is accepted and returns class `stram`.
  - `tram()` passes `distribution` and the `| z` scale basis directly to `ctm()` (`tram:::tram` body).
  - Its default start is infeasible. Every optimiser fails ("initial value in 'vmmin' is not finite"), and the fit ends with logLik `-Inf`, `convergence = 1` (§3a).
- **`tram::Aareg(y ~ 0 | x_s)`** genuinely pairs the exponential link with a scaling term and fits (§3c, `convergence = 0`). `tram:::Aareg` hard-codes three things:
  - `distribution = "Exponential"` and `negative = FALSE`;
  - support `c(0, ·)` (`su[1] <- 0`);
  - **`fixed` on `Bs1(y) = 0`**.

  It also refuses shift terms: `Aareg(y ~ x_d | x_s)` stops with "no constant linear terms allowed in this model" (§3c). So Aareg covers only the scale-only flavour.
- **`tram::Survreg(dist = "exponential")` is not the exponential link.** `tram:::Survreg` maps `exponential → distribution = "MinExtrVal", transformation = "logarithmic"`, and pins the `log(y)` coefficient at 1 (`scale <- 1` → `fixed`).
  - It accepts `| x_s` (§3b, class `Survreg stram`, todistr "minimum extreme value").
  - The scale term then multiplies the fixed `log(y)` coefficient. The result is a Weibull-type AFT model whose shape depends on x_s, not `F_Z = Exp(1)` with a Bernstein `h`.

## 2. How R enforces the support condition: as a likelihood barrier, not a constraint

**Constraint rows.** `mlt:::.mlt_setup` takes `ui`/`ci` only from `attr(model.matrix(model), "constraint")`, which is the basis constraint. It drops rows where `ci = -Inf` and passes the rest to `alabama::auglag` as `hin = ui %*% par - ci` (`mlt::mltoptim`).
- §2 prints them: the 4 Bernstein "increasing" rows. The two `-Inf` rows on `sft_x_d` and `scl_x_s` are dropped.
- There is no row on `θ_b[0]` and none on `θ_b[0]·exp(0.5 x_s γ) + x_d β`. Nothing in the setup reads `todistr`.

**`todistr` has no support information.** `mlt:::.Exponential()` is
`list(parm, p = pexp, d = dexp, q = qexp, dd = -dexp, ddd = dexp, dd2d = function(x) -1, call, name = "exponential")`.
- It has no `support` element. The `support` fields that exist in mlt belong to the frailty distributions (`.GammaFrailty` etc.).
- A sweep of every function in `mlt`, `tram`, `basefun` and `variables` for `xponential|support` found no use of an Exponential support anywhere in fitting.

**What happens instead.**
- **Log-likelihood:** `mlt:::.mlt_loglik_exact` returns `d$d(h, log = TRUE) + .log(h')`, and `dexp(-0.1, log = TRUE) = -Inf` (§1). Any row with `h(y_i | x_i) < 0` gives `-Inf`. For interval data, `pexp(h < 0) = 0`.
- **Score:** `mlt:::.mlt_score_exact` uses `d$dd2d(h) = -1` for *every* h. That is the gradient of the smooth extension `log f(h) = -h`, which is mltpy's convention, and it stays finite below 0 (§7).
- **Effect on the optimiser:**
  - The objective is `+Inf` on the infeasible side, while the gradient keeps pushing toward it.
  - The exact log-likelihood `Σ(-h_i + log h'_i)` grows without bound under a parallel downward shift of `θ_b`, so every optimum has some row with `h(y_i | x_i) = 0`.
  - BFGS (the auglag inner solver) backtracks on the non-finite values and stops.
  - §4b ends at `min_i h(y_i|x_i) = 1.0e-9`. θ_b there is essentially the start `seq(0.1, 5, 5)` shifted down in parallel, β and γ are ≈ 0, logLik is −185.45, and the Hessian is singular (§8).
  - Warm-starting `mlt()` at the true optimum fails too: one row has `h = -2.9e-12`, so the starting value is `-Inf` (§9).

**Two semantics.** R's barrier enforces `h(y_i | x_i) ≥ 0` at the observations ("R semantics"). mltpy's `nonneg_lower` rows enforce `h(a | x_i) ≥ 0` at the lower support end for every training row ("mltpy semantics"). For scaling that is `θ_b[0]·exp(0.5 x_s,i γ) + x_d,i β ≥ 0`, per ADR 0002 Decision 3.
- They coincide only when every row's active point is `a`. With no covariates, that means `a = min(y)`.
- At R's optimum, `h(a | x_i) < 0` on 147 of 150 rows (§4, §9).
- Fixture A's two optima are far apart:

  | Semantics | logLik | θ |
  |---|---|---|
  | R | −28.12 | θ_b = (−1.547, 0.929, 0.937, 4.248, 6.051), β = 0.541, γ = 0.564 |
  | mltpy | −93.65 | θ_b[0] = 0, **β = 0**, γ = 0.488 |

- Under mltpy semantics with a sign-changing `x_d`, β is pinned to 0, because `θ_b[0] → 0` forces `|β| ≤ θ_b[0] s_i / |x_d,i| → 0`.
- The no-covariate fixture (support [0, 1], min y = 0.022) also differs: −2.35 vs −7.51 (§11).
- **This is a design difference inherited from the existing (unscaled) exponential path, not something scaling introduced.** A follow-up ADR should decide it explicitly.

## 3. Does `mlt()` converge on a small fixture?

Data-generating process: `Z ~ Exp(1)`, `h(y|x) = 2y·exp(0.5·0.6·x_s) + 0.5·x_d`, Bernstein order 4, n = 150.

| Fit | conv | logLik | min_i h(y_i\|x_i) | min_i h(a\|x_i) | Verdict |
|---|---|---|---|---|---|
| A (shift+scale), `mlt()` default start (§4a) | 1 | −Inf | −0.117 | −0.185 | start infeasible |
| A, `mlt()` start `seq(.1,5)`,0,0 (§4b) | **0** | −185.455 | 1.0e−9 | −3.2e−3 (147/150 < 0) | stuck at barrier |
| A, shift only `mlt()` (§4d) | 0 | −185.622 | 5.1e−10 | −3.4e−3 | stuck at barrier |
| A, R semantics, explicit `hin` (§9) | 0 | **−28.116** | −2.9e−12 | −3.56 (147/150) | unique; 5 starts agree to Δθ 1.7e−7 |
| A, mltpy semantics, explicit `hin` (§9) | 0 | **−93.645** | 2.0e−10 | −2.7e−10 (0 < 0) | degenerate active set (θ_b[0] = β = 0, θ_b1 = θ_b2); 5 starts agree to Δθ 9.9e−5 |
| B (scale only, support [0, max y]), `mlt()` default start (§5a) | 1 | −Inf | −3.2e−4 | −0.016 | start infeasible |
| B, `mlt()` start `seq(.1,5)`,0 (§5b) | 0 | −73.663 | 6.7e−10 | −9.6e−3 | stuck at barrier |
| B, R semantics, explicit `hin` (§10) | 0 | −59.549 | 3.3e−10 | −0.034 (150/150) | |
| B, **`mlt(fixed = c("Bs1(y)" = 0))`, default start** (§10) | **0** | **−61.6395588** | 0.012 | 0 | clean, equals mltpy-semantics optimum |
| B, `tram::Aareg(y ~ 0 \| x_s, order = 4)` (§10) | 0 | −61.6395588 | | | same model as the line above |

For fixture B with `mltoptim(abstol = 1e-12, reltol = 1e-12)` (§12):

| Comparison | Agreement |
|---|---|
| `mlt(fixed Bs1 = 0)` vs `alabama` with θ_b[0] pinned and eps 1e-12: logLik | 2e-11 |
| same pair: θ | ≤ 8.6e-6 |
| default `mlt()` tolerance vs tight tolerance: θ | 3e-4 |
| `Aareg` vs raw `ctm`: θ | 2e-4 |

The surface is flat (SEs ≈ 0.18 to 1.7, §10). So θ parity at `rtol = 1e-6` needs tight tolerances on *both* sides. Log-likelihood parity is tight.

The explicit-`hin` objective is checked against mlt's own functions in §9: it equals `logLik(fit)` to all printed digits, and its gradient equals `-colSums(fit$scorei(θ))` to 1.4e-14.

## 4. Coefficient names, ordering and sign versus mltpy's `[θ_b | β | γ]`

- **Names (raw `ctm`):**
  - `Bs1(y) … Bs{p}(y)` for θ_b, in ascending Bernstein order.
  - `sft_<var>` for the shift. `ctm` uses the `sft_` prefix only when `scaling` is non-NULL; §4d shows plain `x_d` without scaling.
  - `scl_<var>` for the scale.
- **Order:** `do.call("c", list(bresponse, bshifting, bscaling))`, which is exactly mltpy's `[θ_b | β | γ]` (§2).
- **Shift sign:** raw `ctm(shifting = ~ x)` calls `as.basis(…, negative = FALSE)` (the default), so `h = … + x_d β`. That is the **same sign as mltpy, with no negation**. Only `tram::*`, which defaults to `negative = TRUE`, needs the flip; `tram::tram` builds `iX` with `negative = negative`.
  - §6: `h` computed by hand as `a(y)'θ_b · exp(0.5 x_s γ) + x_d β` matches `predict(type = "trafo")` to 2.2e-16.
  - `crosscheck.out`: negating β changes mltpy's logLik on fixture A (−28.117 → −30.465).
- **Scale:** `.mlt_setup`'s `.sparm` computes `sterm <- exp(0.5 * Z %*% γ)`. With `scale_shift = FALSE` it multiplies only the non-shift, non-scale columns, giving `h = h_0(y)·exp(0.5 x_s γ) + x_d β`. This is identical to mltpy (ADR 0002 Decision 4) and to `mlt:::tmlt`. **γ needs no flip.**
- **Exponential base distribution:**
  - R `.Exponential` uses `pexp`/`dexp`/`qexp` with rate 1: `F_Z(z) = 1 − e^{−z}`, `log f_Z(z) = −z` for z ≥ 0, `−Inf` for z < 0, and score `dd2d = −1`.
  - mltpy `DistOps("exponential")` (`src/mltpy/likelihood.py`) has `cdf = −expm1(−h)`, `logcdf = log(−expm1(−h))`, `logsf = −h`, and `logpdf = −h` **on the whole line**. The docstring calls the last one an intentional divergence from scipy.
  - They agree for h ≥ 0. For h < 0 mltpy's objective is the smooth extension; R's objective is `-Inf`, although R's score is the same extension.
  - `crosscheck.out`: mltpy's `log_likelihood(…, base_distribution = "exponential", scaling = x_s)` at R's θ matches R's logLik to 2.3e-13, 9.9e-14 and 7.1e-15 on the three fits. `log_likelihood` itself carries no exponential+scaling guard; that guard is only in `MLT.__init__` and `optimizer.optimize`.

## 5. Fixture-generator sketch (for a later port into `reference/generate_reference.R`)

**Exact R parity, scale only.** mltpy's constraint is `θ_b[0] ≥ 0`, which is linear, and it is active at the MLE. The argument: a parallel downward shift of θ_b keeps every monotonicity row unchanged, lowers every `h_i` and leaves every `h'_i` alone, so it strictly raises the likelihood until `θ_b[0] = 0`.

```r
set.seed(<seed>)
n <- 150; x_s <- rnorm(n); y <- rexp(n) / (2 * exp(0.5 * 0.6 * x_s))
d <- data.frame(y = y, x_s = x_s)
sup <- c(0, max(y))                      # any a <= min(y) works; pass the same to mltpy
bs  <- Bernstein_basis(numeric_var("y", support = sup), order = 4, ui = "increasing")
m   <- ctm(bs, scaling = ~ x_s, data = d, todistr = "Exponential")
fit <- mlt(m, data = d, fixed = c("Bs1(y)" = 0),
           optim = mltoptim(abstol = 1e-12, reltol = 1e-12, hessian = TRUE))
stopifnot(fit$convergence == 0)
theta <- coef(fit)              # [θ_b (θ_b[0] = 0) | γ], no sign change for mltpy
ll    <- as.numeric(logLik(fit))
se    <- sqrt(diag(vcov(fit)))   # free parameters only (Bs1 is fixed)
```

Notes for this fixture:
- An equivalent call is `tram::Aareg(y ~ 0 | x_s, data = d, order = 4)`, but Aareg chooses its own support (`[0, max y]` here), so pass that support to mltpy.
- On the mltpy side, either fit with the guard lifted for `X is None` and assert θ_b[0] ≈ 0, or pin it with `OptimizerConfig(fixed_params = {0: 0.0})`.
- Compare logLik at `rtol = 1e-6`. Compare θ at `atol ≈ 1e-5`, or tighten mltpy's auglag as well.
- mltpy's SEs from a free fit sit on an active boundary, so the SE for θ_b[0] is not comparable to R's.

**Shift + scale (no R parity, R-computed optimiser reference).** Port §9 of `probe.R`: the `mk()` objective (validated against `fit$logliki` / `fit$scorei`) plus

```r
alabama::auglag(par = c(seq(0.1, 5, length.out = p), 0, 0), fn = o$nll, gr = o$grad,
                hin = o$hin_low, hin.jac = o$jac_low,   # D θ_b ≥ 0 and θ_b[0]·s_i + x_d,i β ≥ 0
                control.outer = list(trace = FALSE, eps = 1e-9, itmax = 200, kkt2.check = FALSE),
                control.optim = list(maxit = 2000))
```

Caveats for this fixture:
- Check uniqueness with multi-start.
- Choose a design whose active set is not degenerate. Fixture A's sign-changing `x_d` pins β = 0, which makes it a poor parity fixture.

## Verdict

**R parity route: viable with caveats.** The caveats:

- **Scale only:** exact R parity is available through `mlt(ctm(..., scaling = ~x_s, todistr = "Exponential"), fixed = c("Bs1(y)" = 0))` or `tram::Aareg(y ~ 0 | x_s)`.
- **Shift + scale:** not viable. R's `mlt()` stops at the `dexp` `-Inf` barrier while reporting `convergence = 0`, and it enforces different support semantics (`h(y_i|x_i) ≥ 0`, not mltpy's `h(a|x_i) ≥ 0`). Use an analytic objective plus an `alabama::auglag` explicit-`hin` reference built on mlt's likelihood (§9), or analytic + FD.

**Side finding (out of scope, should get its own issue).** `reference/mlt_exponential_*.txt` comes from a stuck `mlt()` fit: stored logLik −154.91, real optimum −2.35 under R semantics and −7.51 under mltpy semantics. `tests/test_model.py::test_integration_r_reference_new_distributions` only asserts `ll_py ≥ ll_r − 0.5`, so it passes trivially and checks nothing about the optimum.

## Files

| File | What it is |
|---|---|
| `probe.R` / `probe.Rout` | R probe and its output (`R CMD BATCH --no-save --no-restore probe.R probe.Rout`, run from this directory) |
| `fixtures.csv`, `fixtures_theta.csv` | Fixture data and R-fitted θ, written by `probe.R` §12 |
| `crosscheck.py` / `crosscheck.out` | mltpy `log_likelihood` at R's θ (`PYTHONPATH=src python research/exp-scaling-r-route/crosscheck.py`) |
