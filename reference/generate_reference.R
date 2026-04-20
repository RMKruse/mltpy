#!/usr/bin/env Rscript
# generate_reference.R — Generates mlt reference values for Python test
#
# Run with: Rscript reference/generate_reference.R
# Requires: mlt, basefun packages
#
# Writes two plain-text files (one number per line):
#   reference/mlt_normal_theta.txt  — Bernstein coefficients (order=4, 5 values)
#   reference/mlt_normal_y.txt      — 200 observations on (0, 1)

suppressPackageStartupMessages({
  library(mlt)
  library(basefun)
  library(survival)
})

# Resolve the directory of this script so output paths are portable across
# checkouts and CI runners (no hard-coded /Users/... paths).
script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", script_args, value = TRUE)[1])
if (is.na(file_arg) || file_arg == "") {
  # Fallback when sourced interactively rather than via Rscript
  file_arg <- "reference/generate_reference.R"
}
out_dir <- normalizePath(dirname(file_arg), mustWork = TRUE)

set.seed(42)
n <- 200
y <- runif(n, 0.02, 0.98)

# Bernstein-Polynom Grad 4 auf (0, 1)
# B() in basefun nutzt aufsteigende Reihenfolge — gleich wie unser Python
m <- numeric_var("y", support = c(0, 1), bounds = c(0, 1))
b <- Bernstein_basis(m, order = 4, ui = "increasing")

ctm <- ctm(b)
fit <- mlt(ctm, data = data.frame(y = y))

theta_hat <- coef(fit)

writeLines(format(theta_hat, digits = 15), con = file.path(out_dir, "mlt_normal_theta.txt"))
writeLines(format(y,         digits = 15), con = file.path(out_dir, "mlt_normal_y.txt"))

cat(sprintf("Wrote %d theta values and %d observations.\n", length(theta_hat), n))
cat("theta =", paste(round(theta_hat, 6), collapse = ", "), "\n")

# ---------------------------------------------------------------------------
# AIC / BIC / anova reference
#
# Fit a small (order=3) and a large (order=6) model on the same y, then write:
#   reference/mlt_aic_bic.txt  — single line: "<aic_small> <bic_small> <aic_large> <bic_large>"
#   reference/mlt_anova.txt    — single line: "<chisq> <df> <p_value>"
# ---------------------------------------------------------------------------

b_small <- Bernstein_basis(m, order = 3, ui = "increasing")
b_large <- Bernstein_basis(m, order = 6, ui = "increasing")

fit_small <- mlt(ctm(b_small), data = data.frame(y = y))
fit_large <- mlt(ctm(b_large), data = data.frame(y = y))

# AIC from mlt works directly; BIC() returns NA because mlt has no nobs()
# method, so compute BIC manually using the same formula pymlt uses:
# BIC = -2*ll + log(n_obs) * n_free_params.
ll_s <- logLik(fit_small)
ll_l <- logLik(fit_large)
aic_small <- AIC(fit_small)
aic_large <- AIC(fit_large)
bic_small <- -2 * as.numeric(ll_s) + log(n) * attr(ll_s, "df")
bic_large <- -2 * as.numeric(ll_l) + log(n) * attr(ll_l, "df")

writeLines(
  paste(
    format(aic_small, digits = 15),
    format(bic_small, digits = 15),
    format(aic_large, digits = 15),
    format(bic_large, digits = 15)
  ),
  con = file.path(out_dir, "mlt_aic_bic.txt")
)

# Likelihood-ratio test: mlt has no anova() method, so compute the LRT
# directly from logLik(). This matches pymlt.anova() (deviance = 2*Δll,
# df = Δn_free_params, p = 1 - Χ²_df(deviance)).
ll_small <- logLik(fit_small)
ll_large <- logLik(fit_large)
chisq  <- 2 * (as.numeric(ll_large) - as.numeric(ll_small))
df_lrt <- attr(ll_large, "df") - attr(ll_small, "df")
p_val  <- pchisq(chisq, df = df_lrt, lower.tail = FALSE)

writeLines(
  paste(
    format(chisq,  digits = 15),
    format(df_lrt, digits = 15),
    format(p_val,  digits = 15)
  ),
  con = file.path(out_dir, "mlt_anova.txt")
)

cat(sprintf("AIC small=%.4f, BIC small=%.4f\n", aic_small, bic_small))
cat(sprintf("AIC large=%.4f, BIC large=%.4f\n", aic_large, bic_large))
cat(sprintf("anova: Chisq=%.4f, df=%d, p=%.4g\n", chisq, df_lrt, p_val))

# ---------------------------------------------------------------------------
# Bernstein design matrix reference
#
# Evaluates the Bernstein basis (order=4, support=(0,1)) on an 11-point grid
# on [0, 1] and writes the resulting 11x5 model matrix to
#   reference/bernstein_reference.txt
# in ascending-column order. Used by tests/test_basis.py::test_reference_npy
# to cross-check pymlt.basis.BernsteinBasis.evaluate() against basefun.
# ---------------------------------------------------------------------------

y_grid <- seq(0, 1, length.out = 11)
B_ref  <- Bernstein_basis(m, order = 4, ui = "increasing")
M_ref  <- model.matrix(B_ref, data = data.frame(y = y_grid))

write.table(
  M_ref,
  file = file.path(out_dir, "bernstein_reference.txt"),
  row.names = FALSE,
  col.names = FALSE
)

cat(sprintf("Wrote Bernstein reference matrix: %dx%d\n", nrow(M_ref), ncol(M_ref)))

# ---------------------------------------------------------------------------
# Right-censored log-likelihood reference
#
# Fit an mlt model on right-censored data using the same basis
# (order=4, support=(0,1)) as the mlt_normal_* fixture, then dump:
#   ll_right_y.txt      — 200 observed/censored thresholds
#   ll_right_event.txt  — 0/1 event indicator (1 = observed, 0 = right-censored)
#   ll_right_theta.txt  — Bernstein coefficients from R's mlt
#   ll_right_ll.txt     — scalar log-likelihood from mlt::logLik
#
# The Python test evaluates pymlt.log_likelihood at θ on (y, event) and
# asserts it matches the scalar LL, cross-validating both the likelihood
# implementation and the censoring dispatch.
# ---------------------------------------------------------------------------

set.seed(7)
n_rc     <- 200
y_rc     <- runif(n_rc, 0.05, 0.95)
event_rc <- rbinom(n_rc, size = 1, prob = 0.7)  # 1 = observed, 0 = censored

b_rc    <- Bernstein_basis(m, order = 4, ui = "increasing")
ctm_rc  <- ctm(b_rc)
fit_rc  <- mlt(ctm_rc, data = data.frame(y = Surv(y_rc, event_rc)))

theta_rc <- coef(fit_rc)
ll_rc    <- as.numeric(logLik(fit_rc))

writeLines(format(y_rc,     digits = 15), con = file.path(out_dir, "ll_right_y.txt"))
writeLines(as.character(event_rc),        con = file.path(out_dir, "ll_right_event.txt"))
writeLines(format(theta_rc, digits = 15), con = file.path(out_dir, "ll_right_theta.txt"))
writeLines(format(ll_rc,    digits = 15), con = file.path(out_dir, "ll_right_ll.txt"))

cat(sprintf(
  "Right-censored: n=%d, observed=%d, ll=%.6f\n",
  n_rc, sum(event_rc == 1), ll_rc
))

# ---------------------------------------------------------------------------
# max_extreme_value (standard Gumbel, right) reference
#
# Fit mlt with todistr = "MaxExtrVal" on uncensored data, then write:
#   mlt_maxextrval_theta.txt — Bernstein coefficients (order=4, 5 values)
#   mlt_maxextrval_y.txt     — 200 observations on (0, 1)
#   mlt_maxextrval_ll.txt    — scalar log-likelihood
# ---------------------------------------------------------------------------

set.seed(11)
y_mev  <- runif(200, 0.02, 0.98)
b_mev  <- Bernstein_basis(m, order = 4, ui = "increasing")
ctm_mev <- ctm(b_mev, todistr = "MaxExtrVal")
fit_mev <- mlt(ctm_mev, data = data.frame(y = y_mev))

theta_mev <- coef(fit_mev)
ll_mev    <- as.numeric(logLik(fit_mev))

writeLines(format(theta_mev, digits = 15),
           con = file.path(out_dir, "mlt_maxextrval_theta.txt"))
writeLines(format(y_mev, digits = 15),
           con = file.path(out_dir, "mlt_maxextrval_y.txt"))
writeLines(format(ll_mev, digits = 15),
           con = file.path(out_dir, "mlt_maxextrval_ll.txt"))

cat(sprintf("MaxExtrVal: n=%d, ll=%.6f\n", length(y_mev), ll_mev))

# ---------------------------------------------------------------------------
# exponential reference
#
# Fit mlt with todistr = "Exponential" on uncensored data, then write:
#   mlt_exponential_theta.txt — Bernstein coefficients (order=4, 5 values)
#   mlt_exponential_y.txt     — 200 observations on (0, 1)
#   mlt_exponential_ll.txt    — scalar log-likelihood
# ---------------------------------------------------------------------------

set.seed(13)
y_exp  <- runif(200, 0.02, 0.98)
b_exp  <- Bernstein_basis(m, order = 4, ui = "increasing")
ctm_exp <- ctm(b_exp, todistr = "Exponential")
# mlt's default starting theta can be infeasible for Exponential (support
# [0, ∞)), producing h < 0 and -Inf log-likelihood. Supply a non-negative
# monotone starting vector so the initial evaluation is finite.
theta_init_exp <- seq(0.1, 5.0, length.out = 5)
fit_exp <- mlt(ctm_exp, data = data.frame(y = y_exp), theta = theta_init_exp)

theta_exp <- coef(fit_exp)
ll_exp    <- as.numeric(logLik(fit_exp))

writeLines(format(theta_exp, digits = 15),
           con = file.path(out_dir, "mlt_exponential_theta.txt"))
writeLines(format(y_exp, digits = 15),
           con = file.path(out_dir, "mlt_exponential_y.txt"))
writeLines(format(ll_exp, digits = 15),
           con = file.path(out_dir, "mlt_exponential_ll.txt"))

cat(sprintf("Exponential: n=%d, ll=%.6f\n", length(y_exp), ll_exp))

# ---------------------------------------------------------------------------
# Lm (normal linear regression as a CTM) reference
#
# tram::Lm with order=1 Bernstein + normal base is equivalent to classical
# normal linear regression.  We write both the CTM Bernstein coefficients
# (from tram::Lm) and the lm() point estimates so pymlt tests can verify
# both the raw theta_ and the derived (intercept, slope, sigma).
#
#   lm_uni_y.txt         — response vector
#   lm_uni_support.txt   — support "a b" on one line
#   lm_uni_theta.txt     — [theta_0, theta_1] from tram::Lm
#   lm_uni_lm_coef.txt   — [intercept, sigma] from lm() + summary()$sigma
#
#   lm_cov_y.txt         — response vector
#   lm_cov_x.txt         — single covariate
#   lm_cov_support.txt   — support "a b" on one line
#   lm_cov_theta.txt     — [theta_0, theta_1, beta_ctm]
#   lm_cov_lm_coef.txt   — [intercept, slope, sigma] from lm()
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tram)
})

# Univariate case --------------------------------------------------------
set.seed(17)
n_lm_uni <- 200
y_lm_uni <- rnorm(n_lm_uni, mean = 2.0, sd = 0.5)
a_uni <- min(y_lm_uni) - 0.1
b_uni <- max(y_lm_uni) + 0.1
df_lm_uni <- data.frame(y = y_lm_uni)

fit_lm_uni    <- tram::Lm(y ~ 1, data = df_lm_uni,
                          support = c(a_uni, b_uni), order = 1)
fit_base_uni  <- lm(y ~ 1, data = df_lm_uni)

theta_lm_uni <- coef(fit_lm_uni, with_baseline = TRUE)
lm_uni_coef  <- c(intercept = unname(coef(fit_base_uni)[1]),
                  sigma     = summary(fit_base_uni)$sigma)

writeLines(format(y_lm_uni, digits = 15),
           con = file.path(out_dir, "lm_uni_y.txt"))
writeLines(paste(format(a_uni, digits = 15),
                 format(b_uni, digits = 15)),
           con = file.path(out_dir, "lm_uni_support.txt"))
writeLines(format(theta_lm_uni, digits = 15),
           con = file.path(out_dir, "lm_uni_theta.txt"))
writeLines(format(lm_uni_coef, digits = 15),
           con = file.path(out_dir, "lm_uni_lm_coef.txt"))

cat(sprintf("Lm univariate: n=%d, intercept=%.6f, sigma=%.6f\n",
            n_lm_uni, lm_uni_coef[1], lm_uni_coef[2]))

# One-covariate case -----------------------------------------------------
set.seed(19)
n_lm_cov <- 200
x_lm_cov <- rnorm(n_lm_cov)
y_lm_cov <- 2.0 + 3.0 * x_lm_cov + rnorm(n_lm_cov, sd = 0.5)
a_cov <- min(y_lm_cov) - 0.1
b_cov <- max(y_lm_cov) + 0.1
df_lm_cov <- data.frame(y = y_lm_cov, x = x_lm_cov)

fit_lm_cov   <- tram::Lm(y ~ x, data = df_lm_cov,
                         support = c(a_cov, b_cov), order = 1)
fit_base_cov <- lm(y ~ x, data = df_lm_cov)

theta_lm_cov <- coef(fit_lm_cov, with_baseline = TRUE)
lm_cov_coef  <- c(intercept = unname(coef(fit_base_cov)[1]),
                  slope     = unname(coef(fit_base_cov)[2]),
                  sigma     = summary(fit_base_cov)$sigma)

writeLines(format(y_lm_cov, digits = 15),
           con = file.path(out_dir, "lm_cov_y.txt"))
writeLines(format(x_lm_cov, digits = 15),
           con = file.path(out_dir, "lm_cov_x.txt"))
writeLines(paste(format(a_cov, digits = 15),
                 format(b_cov, digits = 15)),
           con = file.path(out_dir, "lm_cov_support.txt"))
writeLines(format(theta_lm_cov, digits = 15),
           con = file.path(out_dir, "lm_cov_theta.txt"))
writeLines(format(lm_cov_coef, digits = 15),
           con = file.path(out_dir, "lm_cov_lm_coef.txt"))

cat(sprintf("Lm covariate: n=%d, intercept=%.6f, slope=%.6f, sigma=%.6f\n",
            n_lm_cov, lm_cov_coef[1], lm_cov_coef[2], lm_cov_coef[3]))

# ---------------------------------------------------------------------------
# vcov / estfun / standard errors references
#
# For each of three canonical fits (BoxCox, Colr, Coxph) on data with one
# covariate, we write:
#
#   vcov_<model>_y.txt       — response
#   vcov_<model>_event.txt   — 1=observed, 0=right-censored (Coxph only)
#   vcov_<model>_x.txt       — single covariate
#   vcov_<model>_support.txt — "a b" on one line
#   vcov_<model>_theta.txt   — [theta_basis | beta] evaluated at R's MLE
#   vcov_<model>_vcov.txt    — flattened vcov(fit) matrix (row-major)
#   vcov_<model>_estfun.txt  — flattened sandwich::estfun(fit) matrix (row-major)
#
# pymlt tests load theta and data, call hessian()/score_matrix() at that
# theta, invert the Hessian, and cross-check against R's vcov and estfun.
# This isolates the analytical Hessian formula from optimiser differences.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(sandwich)
})

# Helper: flatten an R matrix by row to match numpy's default row-major layout.
flatten_row_major <- function(mat) {
  as.numeric(t(mat))
}

# --- BoxCox (normal base, exact) -----------------------------------------
set.seed(101)
n_bc <- 150
x_bc <- rnorm(n_bc)
y_bc <- 1.0 + 0.8 * x_bc + rnorm(n_bc, sd = 0.6)
a_bc <- min(y_bc) - 0.1
b_bc <- max(y_bc) + 0.1
fit_bc <- tram::BoxCox(y ~ x, data = data.frame(y = y_bc, x = x_bc),
                       support = c(a_bc, b_bc), order = 4)
theta_bc   <- coef(fit_bc, with_baseline = TRUE)
estfun_bc  <- sandwich::estfun(fit_bc)
# tram's vcov() restricts to beta; mlt's vcov() returns the full (p+q, p+q)
# observed information inverse — that's what we need to match against pymlt.
vcov_full_bc <- vcov(as.mlt(fit_bc))

writeLines(format(y_bc, digits = 15), con = file.path(out_dir, "vcov_boxcox_y.txt"))
writeLines(format(x_bc, digits = 15), con = file.path(out_dir, "vcov_boxcox_x.txt"))
writeLines(paste(format(a_bc, digits = 15),
                 format(b_bc, digits = 15)),
           con = file.path(out_dir, "vcov_boxcox_support.txt"))
writeLines(format(theta_bc, digits = 15),
           con = file.path(out_dir, "vcov_boxcox_theta.txt"))
writeLines(format(flatten_row_major(vcov_full_bc), digits = 15),
           con = file.path(out_dir, "vcov_boxcox_vcov.txt"))
writeLines(format(flatten_row_major(estfun_bc), digits = 15),
           con = file.path(out_dir, "vcov_boxcox_estfun.txt"))

cat(sprintf("BoxCox vcov ref: n=%d, p+q=%d\n", n_bc, length(theta_bc)))

# --- Colr (logistic base, exact) -----------------------------------------
set.seed(103)
n_colr <- 150
x_colr <- rnorm(n_colr)
y_colr <- rlogis(n_colr, location = 0.5 * x_colr, scale = 1.0)
a_colr <- min(y_colr) - 0.1
b_colr <- max(y_colr) + 0.1
fit_colr <- tram::Colr(y ~ x, data = data.frame(y = y_colr, x = x_colr),
                       support = c(a_colr, b_colr), order = 4)
theta_colr  <- coef(fit_colr, with_baseline = TRUE)
estfun_colr <- sandwich::estfun(fit_colr)
vcov_full_colr <- vcov(as.mlt(fit_colr))

writeLines(format(y_colr, digits = 15), con = file.path(out_dir, "vcov_colr_y.txt"))
writeLines(format(x_colr, digits = 15), con = file.path(out_dir, "vcov_colr_x.txt"))
writeLines(paste(format(a_colr, digits = 15),
                 format(b_colr, digits = 15)),
           con = file.path(out_dir, "vcov_colr_support.txt"))
writeLines(format(theta_colr, digits = 15),
           con = file.path(out_dir, "vcov_colr_theta.txt"))
writeLines(format(flatten_row_major(vcov_full_colr), digits = 15),
           con = file.path(out_dir, "vcov_colr_vcov.txt"))
writeLines(format(flatten_row_major(estfun_colr), digits = 15),
           con = file.path(out_dir, "vcov_colr_estfun.txt"))

cat(sprintf("Colr vcov ref: n=%d, p+q=%d\n", n_colr, length(theta_colr)))

# --- Coxph (min extreme value base, right-censored) ----------------------
set.seed(107)
n_cx <- 200
x_cx <- rnorm(n_cx)
t_cx <- rexp(n_cx, rate = exp(0.3 * x_cx))
cens_cx <- rexp(n_cx, rate = 0.4)
y_cx <- pmin(t_cx, cens_cx)
event_cx <- as.integer(t_cx <= cens_cx)
a_cx <- 1e-3
b_cx <- max(y_cx) + 0.1
fit_cx <- tram::Coxph(Surv(y, event) ~ x,
                      data = data.frame(y = y_cx, event = event_cx, x = x_cx),
                      support = c(a_cx, b_cx), order = 4)
theta_cx   <- coef(fit_cx, with_baseline = TRUE)
estfun_cx  <- sandwich::estfun(fit_cx)
vcov_full_cx <- vcov(as.mlt(fit_cx))

writeLines(format(y_cx, digits = 15), con = file.path(out_dir, "vcov_coxph_y.txt"))
writeLines(as.character(event_cx),
           con = file.path(out_dir, "vcov_coxph_event.txt"))
writeLines(format(x_cx, digits = 15), con = file.path(out_dir, "vcov_coxph_x.txt"))
writeLines(paste(format(a_cx, digits = 15),
                 format(b_cx, digits = 15)),
           con = file.path(out_dir, "vcov_coxph_support.txt"))
writeLines(format(theta_cx, digits = 15),
           con = file.path(out_dir, "vcov_coxph_theta.txt"))
writeLines(format(flatten_row_major(vcov_full_cx), digits = 15),
           con = file.path(out_dir, "vcov_coxph_vcov.txt"))
writeLines(format(flatten_row_major(estfun_cx), digits = 15),
           con = file.path(out_dir, "vcov_coxph_estfun.txt"))

cat(sprintf("Coxph vcov ref: n=%d, p+q=%d, observed=%d\n",
            n_cx, length(theta_cx), sum(event_cx == 1)))

# ---------------------------------------------------------------------------
# confint references — Wald 95% CIs for each fitted coefficient.
#
# confint() is ±qnorm(0.975) * sqrt(diag(vcov(fit))) around coef(fit).  R's
# mlt::confint picks a sub-block via `parm` but this is the same formula.
# We emit R-convention CIs for each of the three tram fits above.  tram's
# sign convention differs from pymlt's only for BoxCox (negative=TRUE);
# the Python test flips beta rows accordingly before comparing.
#
#   reference/confint_<model>.txt  — flattened (k, 2) matrix, row-major
#                                     columns = [lower, upper]
# ---------------------------------------------------------------------------

.write_confint_ref <- function(fit, filename, level = 0.95) {
  cf <- coef(fit, with_baseline = TRUE)
  se <- sqrt(diag(vcov(as.mlt(fit))))
  z  <- qnorm(0.5 * (1 + level))
  ci <- cbind(cf - z * se, cf + z * se)
  writeLines(format(flatten_row_major(ci), digits = 15),
             con = file.path(out_dir, filename))
}

.write_confint_ref(fit_bc,   "confint_boxcox.txt")
.write_confint_ref(fit_colr, "confint_colr.txt")
.write_confint_ref(fit_cx,   "confint_coxph.txt")

cat("confint refs: boxcox, colr, coxph written.\n")

# ---------------------------------------------------------------------------
# confband reference — baseline (no-covariate) MLT fit, normal base.
#
# Fits the same order-4 Bernstein basis / uniform(0.02, 0.98) sample as the
# very top of this file, then writes:
#
#   reference/confband_baseline_theta.txt      — [theta_0, ..., theta_p-1]
#   reference/confband_baseline_vcov.txt       — (p, p) flattened row-major
#   reference/confband_baseline_y_grid.txt     — m-point evaluation grid
#   reference/confband_baseline_<what>.txt     — (m, 3) flattened row-major
#                                                 cols = [estimate, lwr, upr]
#
# `what` ∈ {trafo, distribution, survivor, density, hazard}.  Bands are
# pointwise Wald on the transformation / log-density / log-hazard scale,
# back-transformed to the requested output scale.  No covariates → no sign-
# convention issue.
# ---------------------------------------------------------------------------

# Reuse the top-of-file `y` and basis: order=4, support=(0, 1), N(0,1) base.
fit_cb   <- fit            # from line 37
theta_cb <- coef(fit_cb, with_baseline = TRUE)
V_cb     <- vcov(fit_cb)
p_cb     <- length(theta_cb)

# Evaluation grid: inside the support, avoiding the exact endpoints where
# h' -> 0 on the Bernstein basis.
y_grid_cb <- seq(0.05, 0.95, length.out = 25)

# Bernstein model matrix B and its first derivative D on the grid.
B_grid <- model.matrix(b, data = data.frame(y = y_grid_cb))

# First derivative of the Bernstein basis: use variables / basefun's
# `deriv = 1` argument on model.matrix.
D_grid <- model.matrix(b, data = data.frame(y = y_grid_cb), deriv = c(y = 1L))

h_hat  <- as.numeric(B_grid %*% theta_cb)
hp_hat <- as.numeric(D_grid %*% theta_cb)

# Per-grid-point linear-predictor variance via delta method:
#   Var(eta_i) = J_i %*% V %*% t(J_i)
# with J depending on `what`.  Vectorised across the grid.
.var_eta_trafo <- function(B, V) {
  # J = B
  rowSums((B %*% V) * B)
}

.var_eta_density <- function(B, D, hp, psi, V) {
  # J = psi * B + D / hp
  J <- psi * B + D / hp
  rowSums((J %*% V) * J)
}

.var_eta_hazard <- function(B, D, hp, psi, lam, V) {
  # J = (psi + lam) * B + D / hp
  coeff <- psi + lam
  J <- coeff * B + D / hp
  rowSums((J %*% V) * J)
}

.write_band <- function(estimate, lwr, upr, filename) {
  mat <- cbind(estimate, lwr, upr)
  writeLines(format(flatten_row_major(mat), digits = 15),
             con = file.path(out_dir, filename))
}

qn <- qnorm(0.975)

# --- trafo / distribution / survivor (eta = h) ---------------------------
var_h <- .var_eta_trafo(B_grid, V_cb)
se_h  <- sqrt(var_h)
h_lo  <- h_hat - qn * se_h
h_hi  <- h_hat + qn * se_h

.write_band(h_hat,        h_lo,         h_hi,         "confband_baseline_trafo.txt")
.write_band(pnorm(h_hat), pnorm(h_lo),  pnorm(h_hi),  "confband_baseline_distribution.txt")
# 1 - F is monotone decreasing → lower/upper swap.
.write_band(1 - pnorm(h_hat), 1 - pnorm(h_hi), 1 - pnorm(h_lo),
            "confband_baseline_survivor.txt")

# --- density (eta = log f(h) + log h') -----------------------------------
psi_normal  <- -h_hat                 # ψ(h) = d log φ(h)/dh = -h for N(0,1)
eta_dens    <- dnorm(h_hat, log = TRUE) + log(hp_hat)
var_dens    <- .var_eta_density(B_grid, D_grid, hp_hat, psi_normal, V_cb)
se_dens     <- sqrt(var_dens)
dens_lo     <- exp(eta_dens - qn * se_dens)
dens_hi     <- exp(eta_dens + qn * se_dens)
.write_band(exp(eta_dens), dens_lo, dens_hi, "confband_baseline_density.txt")

# --- hazard (eta = log f(h) + log h' - log S(h)) -------------------------
lam_normal  <- dnorm(h_hat) / pnorm(h_hat, lower.tail = FALSE)
eta_haz     <- dnorm(h_hat, log = TRUE) + log(hp_hat) -
               pnorm(h_hat, lower.tail = FALSE, log.p = TRUE)
var_haz     <- .var_eta_hazard(B_grid, D_grid, hp_hat, psi_normal, lam_normal, V_cb)
se_haz      <- sqrt(var_haz)
haz_lo      <- exp(eta_haz - qn * se_haz)
haz_hi      <- exp(eta_haz + qn * se_haz)
.write_band(exp(eta_haz), haz_lo, haz_hi, "confband_baseline_hazard.txt")

# --- shared inputs --------------------------------------------------------
writeLines(format(theta_cb,  digits = 15),
           con = file.path(out_dir, "confband_baseline_theta.txt"))
writeLines(format(flatten_row_major(V_cb), digits = 15),
           con = file.path(out_dir, "confband_baseline_vcov.txt"))
writeLines(format(y_grid_cb, digits = 15),
           con = file.path(out_dir, "confband_baseline_y_grid.txt"))

cat(sprintf("confband baseline refs: p=%d, m=%d\n", p_cb, length(y_grid_cb)))
