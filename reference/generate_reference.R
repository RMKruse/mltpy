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
# Left-truncated (delayed-entry) + right-censored reference
#
# Counting-process Surv(start, stop, event) gives R's delayed-entry encoding:
# observation i is only at risk during [trunc_lo[i], y_lt[i]], so the mlt
# log-likelihood divides each row by P(Y_i > trunc_lo[i] | x_i) — exactly
# the truncation correction we are validating in pymlt.
#
# Files emitted (parallel to the right-censored block above):
#   ll_trunc_y.txt      — 200 observed/censored thresholds
#   ll_trunc_event.txt  — 0/1 event indicator
#   ll_trunc_lower.txt  — left-truncation (delayed-entry) times
#   ll_trunc_theta.txt  — Bernstein coefficients fitted by mlt
#   ll_trunc_ll.txt     — scalar log-likelihood from mlt::logLik
# ---------------------------------------------------------------------------

set.seed(11)
n_lt     <- 200
y_lt     <- runif(n_lt, 0.30, 0.95)
event_lt <- rbinom(n_lt, size = 1, prob = 0.7)
trunc_lo <- y_lt - runif(n_lt, 0.05, 0.25)

b_lt    <- Bernstein_basis(m, order = 4, ui = "increasing")
ctm_lt  <- ctm(b_lt)

# mlt's Surv(start, stop, event) path is buggy in 1.7.4 (see
# https://github.com/cran/mlt — `tmp[[response]] <- object$tleft[il]`
# fails when the truncated rows count differs from the response column).
# The supported encoding for combined right-censoring + left-truncation is
# the explicit ``R()`` constructor wrapped in ``I()`` so ``data.frame``
# preserves its class.
exact_lt  <- ifelse(event_lt == 1, y_lt, NA_real_)
cleft_lt  <- ifelse(event_lt == 1, NA_real_, y_lt)
cright_lt <- ifelse(event_lt == 1, NA_real_, Inf)
yvar_lt   <- R(object = exact_lt,
               cleft  = cleft_lt,
               cright = cright_lt,
               tleft  = trunc_lo)
fit_lt  <- mlt(ctm_lt, data = data.frame(y = I(yvar_lt)))

theta_lt <- coef(fit_lt)
ll_lt    <- as.numeric(logLik(fit_lt))

writeLines(format(y_lt,     digits = 15), con = file.path(out_dir, "ll_trunc_y.txt"))
writeLines(as.character(event_lt),        con = file.path(out_dir, "ll_trunc_event.txt"))
writeLines(format(trunc_lo, digits = 15), con = file.path(out_dir, "ll_trunc_lower.txt"))
writeLines(format(theta_lt, digits = 15), con = file.path(out_dir, "ll_trunc_theta.txt"))
writeLines(format(ll_lt,    digits = 15), con = file.path(out_dir, "ll_trunc_ll.txt"))

cat(sprintf(
  "Left-truncated: n=%d, observed=%d, ll=%.6f\n",
  n_lt, sum(event_lt == 1), ll_lt
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

# ---------------------------------------------------------------------------
# Conditional-quantile reference for a Coxph fit with covariates.
#
# Re-fits the same Coxph model as the vcov_coxph_* fixture, then asks
# mlt::predict for quantiles on a small (X, prob) grid.  The Python test
# refits Coxph on the same (y, event, x) data and checks that
# predict(probs, X_new=X, what="quantile") matches R row-for-row.
#
# One row of the expected matrix per X value, one column per probability.
# Writes the matrix row-major (matching flatten_row_major elsewhere).
# ---------------------------------------------------------------------------

x_grid_pq  <- c(-1.0, -0.5, 0.0, 0.5, 1.0)
prob_grid_pq <- c(0.1, 0.25, 0.5, 0.75, 0.9)

# mlt::predict with type="quantile": returns a (length(prob), length(newdata))
# matrix; columns correspond to rows of newdata.  Transpose so that each row
# of the saved matrix corresponds to one X value.
q_mat_pq <- predict(
  as.mlt(fit_cx),
  newdata = data.frame(x = x_grid_pq),
  type = "quantile",
  prob = prob_grid_pq
)
q_mat_pq <- t(q_mat_pq)  # rows = X, cols = prob

writeLines(format(x_grid_pq, digits = 15),
           con = file.path(out_dir, "predict_quantile_coxph_X.txt"))
writeLines(format(prob_grid_pq, digits = 15),
           con = file.path(out_dir, "predict_quantile_coxph_probs.txt"))
writeLines(format(flatten_row_major(q_mat_pq), digits = 15),
           con = file.path(out_dir, "predict_quantile_coxph_expected.txt"))

cat(sprintf("Coxph predict-quantile ref: %d X x %d probs\n",
            length(x_grid_pq), length(prob_grid_pq)))

# ---------------------------------------------------------------------------
# Residuals reference — reuses the BoxCox / Colr / Coxph fits from the
# vcov_*_* block above.  For each fit we write three vectors:
#
#   residuals_<model>_score.txt    — score residual (R's mlt::residuals)
#                                    = ∂ℓ_i/∂α at α = 0 for h̃ = h + α
#   residuals_<model>_coxsnell.txt — −log S(y_i|x_i) at the observed point
#   residuals_<model>_deviance.txt — sign(r-1) · sqrt(2·|r - log r - 1|)
#
# For Cox-Snell on the right-censored Coxph fit, the observed point for
# censored obs is the censoring time (Surv$time).  This matches pymlt's
# convention of evaluating S at the observed lower bound.
# ---------------------------------------------------------------------------

.cox_snell <- function(surv_vec) -log(surv_vec)

.deviance_from_cs <- function(r) {
  r_safe <- pmax(r, .Machine$double.xmin)
  sign(r - 1) * sqrt(2 * abs(r - log(r_safe) - 1))
}

# --- BoxCox --------------------------------------------------------------
score_bc <- as.numeric(residuals(fit_bc))
surv_bc  <- predict(as.mlt(fit_bc),
                    newdata = data.frame(y = y_bc, x = x_bc),
                    type = "survivor")
cs_bc    <- .cox_snell(surv_bc)
dev_bc   <- .deviance_from_cs(cs_bc)
writeLines(format(score_bc, digits = 15),
           con = file.path(out_dir, "residuals_boxcox_score.txt"))
writeLines(format(cs_bc, digits = 15),
           con = file.path(out_dir, "residuals_boxcox_coxsnell.txt"))
writeLines(format(dev_bc, digits = 15),
           con = file.path(out_dir, "residuals_boxcox_deviance.txt"))

# --- Colr ----------------------------------------------------------------
score_colr <- as.numeric(residuals(fit_colr))
surv_colr  <- predict(as.mlt(fit_colr),
                      newdata = data.frame(y = y_colr, x = x_colr),
                      type = "survivor")
cs_colr    <- .cox_snell(surv_colr)
dev_colr   <- .deviance_from_cs(cs_colr)
writeLines(format(score_colr, digits = 15),
           con = file.path(out_dir, "residuals_colr_score.txt"))
writeLines(format(cs_colr, digits = 15),
           con = file.path(out_dir, "residuals_colr_coxsnell.txt"))
writeLines(format(dev_colr, digits = 15),
           con = file.path(out_dir, "residuals_colr_deviance.txt"))

# --- Coxph (right-censored) ----------------------------------------------
score_cx <- as.numeric(residuals(fit_cx))
# For the survivor, evaluate at the observed y (which is the censoring time
# for censored obs).  predict.mlt requires a Surv() column when the fit was
# made with one — wrap event=1 so it interprets y as exact for evaluation.
surv_cx  <- predict(as.mlt(fit_cx),
                    newdata = data.frame(y = y_cx, x = x_cx),
                    q = y_cx,
                    type = "survivor")
# `predict` with `q = y_cx` returns a matrix (length(q), nrow(newdata)) when
# q is a vector; pull the diagonal so each row uses its own y.
if (is.matrix(surv_cx)) {
  surv_cx <- diag(surv_cx)
}
cs_cx    <- .cox_snell(surv_cx)
dev_cx   <- .deviance_from_cs(cs_cx)
writeLines(format(score_cx, digits = 15),
           con = file.path(out_dir, "residuals_coxph_score.txt"))
writeLines(format(cs_cx, digits = 15),
           con = file.path(out_dir, "residuals_coxph_coxsnell.txt"))
writeLines(format(dev_cx, digits = 15),
           con = file.path(out_dir, "residuals_coxph_deviance.txt"))

cat(sprintf("residuals refs: boxcox (n=%d), colr (n=%d), coxph (n=%d) written.\n",
            length(score_bc), length(score_colr), length(score_cx)))

# ---------------------------------------------------------------------------
# Weights reference
#
# Re-uses the same data (y_bc / x_bc, y_colr / x_colr, y_cx / x_cx) and
# support / order from the vcov block above.  For each model we:
#   1. Draw integer weights w ~ Uniform{1,2,3,4}.
#   2. Fit with those weights (maximise Σ w_i · ℓ_i).
#   3. Record: the weights, theta, log-likelihood, and score matrix (estfun).
#
# R sandwich::estfun convention for weighted mlt models:
#   row i = w_i · ∂ℓ_i/∂θ   (column sums ≈ 0 at MLE)
# ---------------------------------------------------------------------------

# --- BoxCox with weights --------------------------------------------------
set.seed(200)
w_bc_w <- as.numeric(sample(1L:4L, n_bc, replace = TRUE))
fit_bc_w <- tram::BoxCox(y ~ x,
                          data    = data.frame(y = y_bc, x = x_bc),
                          support = c(a_bc, b_bc),
                          order   = 4L,
                          weights = w_bc_w)
theta_bc_w   <- coef(fit_bc_w, with_baseline = TRUE)
loglik_bc_w  <- as.numeric(logLik(as.mlt(fit_bc_w)))
estfun_bc_w  <- sandwich::estfun(fit_bc_w)

writeLines(format(w_bc_w,                          digits = 15),
           con = file.path(out_dir, "weights_boxcox_w.txt"))
writeLines(format(theta_bc_w,                       digits = 15),
           con = file.path(out_dir, "weights_boxcox_theta.txt"))
writeLines(format(loglik_bc_w,                      digits = 15),
           con = file.path(out_dir, "weights_boxcox_ll.txt"))
writeLines(format(flatten_row_major(estfun_bc_w),   digits = 15),
           con = file.path(out_dir, "weights_boxcox_estfun.txt"))

cat(sprintf("BoxCox weights ref: n=%d, sum_w=%d, p+q=%d\n",
            n_bc, sum(w_bc_w), length(theta_bc_w)))

# --- Colr with weights ----------------------------------------------------
set.seed(201)
w_colr_w <- as.numeric(sample(1L:4L, n_colr, replace = TRUE))
fit_colr_w <- tram::Colr(y ~ x,
                          data    = data.frame(y = y_colr, x = x_colr),
                          support = c(a_colr, b_colr),
                          order   = 4L,
                          weights = w_colr_w)
theta_colr_w  <- coef(fit_colr_w, with_baseline = TRUE)
loglik_colr_w <- as.numeric(logLik(as.mlt(fit_colr_w)))
estfun_colr_w <- sandwich::estfun(fit_colr_w)

writeLines(format(w_colr_w,                          digits = 15),
           con = file.path(out_dir, "weights_colr_w.txt"))
writeLines(format(theta_colr_w,                       digits = 15),
           con = file.path(out_dir, "weights_colr_theta.txt"))
writeLines(format(loglik_colr_w,                      digits = 15),
           con = file.path(out_dir, "weights_colr_ll.txt"))
writeLines(format(flatten_row_major(estfun_colr_w),   digits = 15),
           con = file.path(out_dir, "weights_colr_estfun.txt"))

cat(sprintf("Colr weights ref: n=%d, sum_w=%d, p+q=%d\n",
            n_colr, sum(w_colr_w), length(theta_colr_w)))

# --- Coxph with weights ---------------------------------------------------
set.seed(202)
w_cx_w <- as.numeric(sample(1L:4L, n_cx, replace = TRUE))
fit_cx_w <- tram::Coxph(Surv(y, event) ~ x,
                         data    = data.frame(y = y_cx, event = event_cx, x = x_cx),
                         support = c(a_cx, b_cx),
                         order   = 4L,
                         weights = w_cx_w)
theta_cx_w   <- coef(fit_cx_w, with_baseline = TRUE)
loglik_cx_w  <- as.numeric(logLik(as.mlt(fit_cx_w)))
estfun_cx_w  <- sandwich::estfun(fit_cx_w)

writeLines(format(w_cx_w,                          digits = 15),
           con = file.path(out_dir, "weights_coxph_w.txt"))
writeLines(format(theta_cx_w,                       digits = 15),
           con = file.path(out_dir, "weights_coxph_theta.txt"))
writeLines(format(loglik_cx_w,                      digits = 15),
           con = file.path(out_dir, "weights_coxph_ll.txt"))
writeLines(format(flatten_row_major(estfun_cx_w),   digits = 15),
           con = file.path(out_dir, "weights_coxph_estfun.txt"))

cat(sprintf("Coxph weights ref: n=%d, sum_w=%d, p+q=%d\n",
            n_cx, sum(w_cx_w), length(theta_cx_w)))

# ---------------------------------------------------------------------------
# Polr (proportional-odds ordinal regression) reference
#
# Fit tram::Polr with three different links (logistic / probit / cloglog)
# on the same n=300 ordinal sample.  Write per-link:
#   reference/polr_<link>_theta.txt     — coef(fit, with_baseline = TRUE)
#                                         (first K-1 entries are cutpoints,
#                                          remaining are R-side beta which
#                                          satisfy h - X·beta_R = z; pymlt
#                                          uses h + X·beta so beta_pymlt =
#                                          -beta_R)
#   reference/polr_<link>_loglik.txt    — scalar log-likelihood
#   reference/polr_<link>_proba.txt     — predict(fit, type = "density")
#                                         flattened row-major, n*K
#
# Shared inputs (one copy):
#   reference/polr_X.txt   — model.matrix entries (n*q row-major)
#   reference/polr_y.txt   — character labels of the ordered response
# ---------------------------------------------------------------------------

set.seed(42)
n_polr <- 300
levels_polr <- c("low", "mid", "high")
y_polr <- factor(
  sample(levels_polr, n_polr, replace = TRUE, prob = c(0.30, 0.40, 0.30)),
  levels = levels_polr,
  ordered = TRUE
)
x1_polr <- rnorm(n_polr)
x2_polr <- rbinom(n_polr, 1, 0.5)
data_polr <- data.frame(y = y_polr, x1 = x1_polr, x2 = x2_polr)
X_polr <- model.matrix(~ x1 + x2 - 1, data = data_polr)

writeLines(as.character(y_polr), con = file.path(out_dir, "polr_y.txt"))
writeLines(format(flatten_row_major(X_polr), digits = 15),
           con = file.path(out_dir, "polr_X.txt"))

.write_polr_link <- function(method, label) {
  fit <- tram::Polr(y ~ x1 + x2,
                    data   = data_polr,
                    method = method)
  theta <- coef(fit, with_baseline = TRUE)
  ll    <- as.numeric(logLik(as.mlt(fit)))
  # Evaluate per-level probabilities at every observation by passing the full
  # ordered level set as q.  Returns a (K, n) matrix; transpose to (n, K) and
  # write row-major to match pymlt.predict_proba.
  q_levels <- factor(levels_polr, levels = levels_polr, ordered = TRUE)
  proba <- predict(fit, newdata = data_polr, q = q_levels, type = "density")
  proba <- t(as.matrix(proba))

  writeLines(format(theta, digits = 15),
             con = file.path(out_dir, sprintf("polr_%s_theta.txt", label)))
  writeLines(format(ll,    digits = 15),
             con = file.path(out_dir, sprintf("polr_%s_loglik.txt", label)))
  writeLines(format(flatten_row_major(proba), digits = 15),
             con = file.path(out_dir, sprintf("polr_%s_proba.txt", label)))
  cat(sprintf("Polr %-9s ref: n=%d, K=%d, ll=%.6f\n",
              label, n_polr, length(levels_polr), ll))
}

.write_polr_link("logistic", "logistic")
.write_polr_link("probit",   "probit")
.write_polr_link("cloglog",  "cloglog")

# ---------------------------------------------------------------------------
# Bernstein-y × Bernstein-x interacting CTM — issue #65
#
# Fits mlt::ctm(response = Bernstein_basis(y),
#               interacting = Bernstein_basis(x),
#               todistr = <distribution>)
# for two base distributions (Normal, Logistic) on a small but well-mixed
# synthetic dataset.  Writes:
#
#   reference/interaction_bs_bs_y_train.txt   — training y, length n
#   reference/interaction_bs_bs_x_train.txt   — training x, length n
#   reference/interaction_bs_bs_y_support.txt — "a b" on one line
#   reference/interaction_bs_bs_y_grid.txt    — held-out y grid, length m_y
#   reference/interaction_bs_bs_x_grid.txt    — held-out x grid, length m_x
#
# Per distribution (label ∈ {"normal", "logistic"}):
#   reference/interaction_bs_bs_<label>_theta.txt   — coef(fit) in mlt's order
#                                                     (column-major over Θ[i,j],
#                                                     j varies slowest, i within;
#                                                     length p*q)
#   reference/interaction_bs_bs_<label>_loglik.txt  — scalar logLik
#   reference/interaction_bs_bs_<label>_cdf.txt     — fitted CDF on
#                                                     expand.grid(y_grid, x_grid),
#                                                     length m_y*m_x
#   reference/interaction_bs_bs_<label>_pdf.txt     — fitted PDF on same grid
#
# Coefficient-order note: mlt names coefficients "Bs<i>(y):Bs<j>(x)" with the
# y-index ``i`` varying fastest as ``j`` cycles 1..q.  In the Python test we
# reshape ``theta_R`` as ``Θ[i, j] = theta_R.reshape(q, p).T`` and compare to
# ``model.Theta_`` directly.
# ---------------------------------------------------------------------------

set.seed(20260517)
n_int <- 300
p_int <- 3L  # number of Bernstein-y functions
q_int <- 3L  # number of Bernstein-x functions
x_int <- runif(n_int, 0, 1)
# Mild conditional shift, low homoscedastic noise.  Keeps the MLE strictly in
# the interior of the monotone cone — no stacked active constraints — so
# alabama::auglag and pymlt's PHR solver converge to the same θ to within
# their respective KKT tolerances.
y_int <- 0.5 + 1.0 * x_int + rnorm(n_int, sd = 0.6)
a_int <- min(y_int) - 0.2
b_int <- max(y_int) + 0.2

m_y_int <- numeric_var("y", support = c(a_int, b_int))
m_x_int <- numeric_var("x", support = c(0, 1))
b_y_int <- Bernstein_basis(m_y_int, order = p_int - 1L, ui = "increasing")
b_x_int <- Bernstein_basis(m_x_int, order = q_int - 1L)

y_grid_int <- seq(a_int + 0.05 * (b_int - a_int),
                  b_int - 0.05 * (b_int - a_int),
                  length.out = 7L)
x_grid_int <- c(0.10, 0.30, 0.50, 0.70, 0.90)
grid_int   <- expand.grid(y = y_grid_int, x = x_grid_int)  # y fastest

writeLines(format(y_int, digits = 15),
           con = file.path(out_dir, "interaction_bs_bs_y_train.txt"))
writeLines(format(x_int, digits = 15),
           con = file.path(out_dir, "interaction_bs_bs_x_train.txt"))
writeLines(paste(format(a_int, digits = 15), format(b_int, digits = 15)),
           con = file.path(out_dir, "interaction_bs_bs_y_support.txt"))
writeLines(format(y_grid_int, digits = 15),
           con = file.path(out_dir, "interaction_bs_bs_y_grid.txt"))
writeLines(format(x_grid_int, digits = 15),
           con = file.path(out_dir, "interaction_bs_bs_x_grid.txt"))

# Probability grid for quantile predictions on interaction fixtures (issue #67).
probs_int <- c(0.10, 0.25, 0.50, 0.75, 0.90)
writeLines(format(probs_int, digits = 15),
           con = file.path(out_dir, "interaction_bs_bs_probs.txt"))

.write_interaction <- function(todistr, label) {
  ctm_int <- ctm(response = b_y_int,
                 interacting = b_x_int,
                 todistr = todistr)
  fit_int <- mlt(ctm_int, data = data.frame(y = y_int, x = x_int))
  theta_int  <- coef(fit_int)
  ll_int     <- as.numeric(logLik(fit_int))
  cdf_int    <- as.numeric(predict(fit_int, newdata = grid_int,
                                   type = "distribution"))
  pdf_int    <- as.numeric(predict(fit_int, newdata = grid_int,
                                   type = "density"))
  surv_int   <- as.numeric(predict(fit_int, newdata = grid_int,
                                   type = "survivor"))
  haz_int    <- as.numeric(predict(fit_int, newdata = grid_int,
                                   type = "hazard"))
  # Quantile predictions: q[k, j] = inverse h(.|x_grid_int[j]) at probs_int[k]
  q_mat      <- predict(fit_int,
                        newdata = data.frame(x = x_grid_int),
                        type = "quantile",
                        prob = probs_int)
  q_flat     <- as.numeric(q_mat)  # column-major: x varies slowest
  writeLines(format(theta_int, digits = 15),
             con = file.path(out_dir,
                             sprintf("interaction_bs_bs_%s_theta.txt", label)))
  writeLines(format(ll_int, digits = 15),
             con = file.path(out_dir,
                             sprintf("interaction_bs_bs_%s_loglik.txt", label)))
  writeLines(format(cdf_int, digits = 15),
             con = file.path(out_dir,
                             sprintf("interaction_bs_bs_%s_cdf.txt", label)))
  writeLines(format(pdf_int, digits = 15),
             con = file.path(out_dir,
                             sprintf("interaction_bs_bs_%s_pdf.txt", label)))
  writeLines(format(surv_int, digits = 15),
             con = file.path(out_dir,
                             sprintf("interaction_bs_bs_%s_survivor.txt", label)))
  writeLines(format(haz_int, digits = 15),
             con = file.path(out_dir,
                             sprintf("interaction_bs_bs_%s_hazard.txt", label)))
  writeLines(format(q_flat, digits = 15),
             con = file.path(out_dir,
                             sprintf("interaction_bs_bs_%s_quantile.txt", label)))
  cat(sprintf("interaction_bs_bs %-8s ref: n=%d, p=%d, q=%d, ll=%.6f\n",
              label, n_int, p_int, q_int, ll_int))
}

.write_interaction("Normal",   "normal")
.write_interaction("Logistic", "logistic")

# ---------------------------------------------------------------------------
# Scaling-terms tracer (issue #70) — BoxCox + normal base, scale=~x_s.
#
# Heteroskedastic Box-Cox:  h(y | x_d, x_s) = h_0(y) * exp(x_s * gamma) - x_d * beta_R
# (R / tram convention: shift enters with a minus; pymlt parameterises h + x_d * beta,
# so pymlt.coef_ == -coef(fit)["x_d"].  Gamma is sign-aligned across R and pymlt; see
# docs/adr/0002-scaling-terms.md, Decision 5.)
#
# Fixtures emitted:
#   reference/scaling_boxcox_normal_y.txt        — response  (n values)
#   reference/scaling_boxcox_normal_x_d.txt      — shift design column (n values)
#   reference/scaling_boxcox_normal_x_s.txt      — scaling design column (n values)
#   reference/scaling_boxcox_normal_support.txt  — "a b" (basis support)
#   reference/scaling_boxcox_normal_theta.txt    — [theta_b | beta_tram | gamma]
#                                                  flat vector from coef(as.mlt(fit))
#   reference/scaling_boxcox_normal_loglik.txt   — logLik(fit)
# ---------------------------------------------------------------------------
set.seed(70)
n_sc <- 100
x_s_sc <- rnorm(n_sc)
x_d_sc <- rnorm(n_sc)
y_sc <- 1.0 + 0.5 * x_d_sc + rnorm(n_sc, sd = exp(0.3 * x_s_sc))
a_sc <- min(y_sc) - 0.1
b_sc <- max(y_sc) + 0.1
df_sc <- data.frame(y = y_sc, x_d = x_d_sc, x_s = x_s_sc)
fit_sc <- tram::BoxCox(y ~ x_d | x_s, data = df_sc,
                       support = c(a_sc, b_sc), order = 5)
theta_full_sc <- coef(as.mlt(fit_sc))
ll_sc <- as.numeric(logLik(fit_sc))

writeLines(format(y_sc,   digits = 15),
           con = file.path(out_dir, "scaling_boxcox_normal_y.txt"))
writeLines(format(x_d_sc, digits = 15),
           con = file.path(out_dir, "scaling_boxcox_normal_x_d.txt"))
writeLines(format(x_s_sc, digits = 15),
           con = file.path(out_dir, "scaling_boxcox_normal_x_s.txt"))
writeLines(paste(format(a_sc, digits = 15),
                 format(b_sc, digits = 15)),
           con = file.path(out_dir, "scaling_boxcox_normal_support.txt"))
writeLines(format(theta_full_sc, digits = 15),
           con = file.path(out_dir, "scaling_boxcox_normal_theta.txt"))
writeLines(format(ll_sc, digits = 15),
           con = file.path(out_dir, "scaling_boxcox_normal_loglik.txt"))

cat(sprintf("scaling BoxCox normal ref: n=%d, p+q_d+q_s=%d, ll=%.6f\n",
            n_sc, length(theta_full_sc), ll_sc))

# ---------------------------------------------------------------------------
# Scaling-terms censoring + base-distribution coverage (issue #71).
#
# Three new fixtures exercise the scaled-baseline likelihood under each
# remaining censoring branch:
#
#   * Coxph(scale=~x_s)  — RIGHT-censored, min_extreme_value base
#   * Colr(scale=~x_s)   — exact, logistic base (the shift-only Colr ships
#                          a separate vcov fixture; this one adds γ)
#   * BoxCox(scale=~x_s) — INTERVAL-censored, normal base
#
# In each case the parameter file ``scaling_*_theta.txt`` stores the *full*
# parameter vector ``coef(as.mlt(fit))`` flattened as
# ``[theta_b | beta_tram | gamma]``.  pymlt parametrises ``h + X_d·β`` (R
# ``tram`` parametrises ``h − X_d·β``), so the Python parity test flips the
# β block before comparing.  γ is sign-aligned across the two
# parameterisations (ADR 0002, Decision 5).
# ---------------------------------------------------------------------------
library(survival)

# ---- scaling_coxph_right (Coxph + scale=~x_s, RIGHT-censored) ------------
set.seed(710)
n_cxs <- 200
x_s_cxs <- rnorm(n_cxs)
x_d_cxs <- rnorm(n_cxs)
# Heteroskedastic Weibull-flavoured event times: shape varies with x_s,
# scale shifts with x_d.  Rate-1 exponential under exp(.) link.
t_cxs <- rexp(n_cxs, rate = exp(0.4 * x_d_cxs - 0.3 * x_s_cxs))
cens_cxs <- rexp(n_cxs, rate = 0.3)
y_cxs <- pmin(t_cxs, cens_cxs)
event_cxs <- as.integer(t_cxs <= cens_cxs)
a_cxs <- 1e-3
b_cxs <- max(y_cxs) + 0.1
df_cxs <- data.frame(y = y_cxs, event = event_cxs,
                     x_d = x_d_cxs, x_s = x_s_cxs)
fit_cxs <- tram::Coxph(Surv(y, event) ~ x_d | x_s, data = df_cxs,
                       support = c(a_cxs, b_cxs), order = 5)
theta_cxs <- coef(as.mlt(fit_cxs))
ll_cxs <- as.numeric(logLik(fit_cxs))

writeLines(format(y_cxs, digits = 15),
           con = file.path(out_dir, "scaling_coxph_y.txt"))
writeLines(as.character(event_cxs),
           con = file.path(out_dir, "scaling_coxph_event.txt"))
writeLines(format(x_d_cxs, digits = 15),
           con = file.path(out_dir, "scaling_coxph_x_d.txt"))
writeLines(format(x_s_cxs, digits = 15),
           con = file.path(out_dir, "scaling_coxph_x_s.txt"))
writeLines(paste(format(a_cxs, digits = 15),
                 format(b_cxs, digits = 15)),
           con = file.path(out_dir, "scaling_coxph_support.txt"))
writeLines(format(theta_cxs, digits = 15),
           con = file.path(out_dir, "scaling_coxph_theta.txt"))
writeLines(format(ll_cxs, digits = 15),
           con = file.path(out_dir, "scaling_coxph_loglik.txt"))

cat(sprintf("scaling Coxph right-censored ref: n=%d, p+q_d+q_s=%d, ll=%.6f\n",
            n_cxs, length(theta_cxs), ll_cxs))

# ---- scaling_colr (Colr + scale=~x_s, exact, logistic) -------------------
set.seed(711)
n_co <- 150
x_s_co <- rnorm(n_co)
x_d_co <- rnorm(n_co)
y_co <- rlogis(n_co, location = 0.5 * x_d_co,
               scale = exp(0.25 * x_s_co))
a_co <- min(y_co) - 0.1
b_co <- max(y_co) + 0.1
df_co <- data.frame(y = y_co, x_d = x_d_co, x_s = x_s_co)
fit_co <- tram::Colr(y ~ x_d | x_s, data = df_co,
                     support = c(a_co, b_co), order = 5)
theta_co <- coef(as.mlt(fit_co))
ll_co <- as.numeric(logLik(fit_co))

writeLines(format(y_co, digits = 15),
           con = file.path(out_dir, "scaling_colr_y.txt"))
writeLines(format(x_d_co, digits = 15),
           con = file.path(out_dir, "scaling_colr_x_d.txt"))
writeLines(format(x_s_co, digits = 15),
           con = file.path(out_dir, "scaling_colr_x_s.txt"))
writeLines(paste(format(a_co, digits = 15),
                 format(b_co, digits = 15)),
           con = file.path(out_dir, "scaling_colr_support.txt"))
writeLines(format(theta_co, digits = 15),
           con = file.path(out_dir, "scaling_colr_theta.txt"))
writeLines(format(ll_co, digits = 15),
           con = file.path(out_dir, "scaling_colr_loglik.txt"))

# --- predict(..., type="logodds") on a small (k × m) grid -----------------
# Heteroskedastic logistic regression: with `scale=~x_s`, log F/S is no
# longer linear in x_s across the q-axis, so a grid that varies both x_d
# (shift) and x_s (scale) at several response points exercises the
# scaled-predict path (#72) end-to-end.  Fixture layout mirrors
# scaling_predict_* — k response points along rows, m newdata rows along
# columns, flattened row-major.
x_d_new_co <- c(-1.0, 0.0, 0.5, 1.5)
x_s_new_co <- c(-0.5, 0.0, 1.0, 2.0)
stopifnot(length(x_d_new_co) == length(x_s_new_co))
m_co <- length(x_d_new_co)
span_co <- b_co - a_co
q_grid_co <- seq(a_co + 0.1 * span_co, b_co - 0.1 * span_co, length.out = 5)
newdata_co <- data.frame(x_d = x_d_new_co, x_s = x_s_new_co)
logodds_co <- predict(fit_co, newdata = newdata_co,
                      q = q_grid_co, type = "logodds")
stopifnot(is.matrix(logodds_co) && nrow(logodds_co) == length(q_grid_co) &&
          ncol(logodds_co) == m_co)
writeLines(format(x_d_new_co, digits = 15),
           con = file.path(out_dir, "scaling_colr_logodds_x_d_new.txt"))
writeLines(format(x_s_new_co, digits = 15),
           con = file.path(out_dir, "scaling_colr_logodds_x_s_new.txt"))
writeLines(format(q_grid_co, digits = 15),
           con = file.path(out_dir, "scaling_colr_logodds_q_grid.txt"))
writeLines(format(flatten_row_major(logodds_co), digits = 15),
           con = file.path(out_dir, "scaling_colr_logodds.txt"))

cat(sprintf("scaling Colr exact ref: n=%d, p+q_d+q_s=%d, ll=%.6f\n",
            n_co, length(theta_co), ll_co))

# ---- scaling_boxcox_interval (BoxCox + scale=~x_s, interval-censored) ----
set.seed(712)
n_iv <- 150
x_s_iv <- rnorm(n_iv)
x_d_iv <- rnorm(n_iv)
y_true_iv <- 1.0 + 0.5 * x_d_iv + rnorm(n_iv, sd = exp(0.3 * x_s_iv))
# Symmetric ±0.25 window around the latent value → interval-censored
# rows of width 0.5 covering the truth.
lo_iv <- y_true_iv - 0.25
hi_iv <- y_true_iv + 0.25
a_iv <- min(lo_iv) - 1.0
b_iv <- max(hi_iv) + 1.0
df_iv <- data.frame(y_lo = lo_iv, y_hi = hi_iv,
                    x_d = x_d_iv, x_s = x_s_iv)
fit_iv <- tram::BoxCox(Surv(y_lo, y_hi, type = "interval2") ~ x_d | x_s,
                       data = df_iv,
                       support = c(a_iv, b_iv), order = 5,
                       bounds = c(a_iv, b_iv))
theta_iv <- coef(as.mlt(fit_iv))
ll_iv <- as.numeric(logLik(fit_iv))

writeLines(format(lo_iv, digits = 15),
           con = file.path(out_dir, "scaling_boxcox_interval_lo.txt"))
writeLines(format(hi_iv, digits = 15),
           con = file.path(out_dir, "scaling_boxcox_interval_hi.txt"))
writeLines(format(x_d_iv, digits = 15),
           con = file.path(out_dir, "scaling_boxcox_interval_x_d.txt"))
writeLines(format(x_s_iv, digits = 15),
           con = file.path(out_dir, "scaling_boxcox_interval_x_s.txt"))
writeLines(paste(format(a_iv, digits = 15),
                 format(b_iv, digits = 15)),
           con = file.path(out_dir, "scaling_boxcox_interval_support.txt"))
writeLines(format(theta_iv, digits = 15),
           con = file.path(out_dir, "scaling_boxcox_interval_theta.txt"))
writeLines(format(ll_iv, digits = 15),
           con = file.path(out_dir, "scaling_boxcox_interval_loglik.txt"))

cat(sprintf("scaling BoxCox interval ref: n=%d, p+q_d+q_s=%d, ll=%.6f\n",
            n_iv, length(theta_iv), ll_iv))

# ---------------------------------------------------------------------------
# Scaling-terms predict path (issue #72) — distribution / density / hazard /
# survivor / quantile under scaling, evaluated on a small (x_d, x_s) × (q|p)
# grid against the BoxCox + scale=~x_s normal fit from #70.
#
# For each ``what`` value the saved expected matrix is shape ``(k, m)`` with
# ``k`` y-values (or probabilities) along the rows and ``m`` newdata rows
# along the columns, flattened row-major.  The Python test reshapes back to
# ``(k, m)`` and asserts element-wise parity at rtol=1e-6.
#
# Fixtures emitted (re-using fit_sc):
#   scaling_predict_x_d_new.txt   — m newdata shift covariates (length m)
#   scaling_predict_x_s_new.txt   — m newdata scaling covariates (length m)
#   scaling_predict_q_grid.txt    — k response evaluation points (length k)
#   scaling_predict_prob_grid.txt — k_q probability targets (length k_q)
#   scaling_predict_<what>.txt    — flatten_row_major of (k, m) matrix
#                                   for what in {distribution, density,
#                                   hazard, survivor, quantile (size k_q × m)}
# ---------------------------------------------------------------------------
x_d_new_sp <- c(-1.0, 0.0, 0.5, 1.5)
x_s_new_sp <- c(-0.5, 0.0, 1.0, 2.0)
stopifnot(length(x_d_new_sp) == length(x_s_new_sp))
m_sp <- length(x_d_new_sp)

# Five interior evaluation points strictly inside the basis support.
span_sp <- b_sc - a_sc
q_grid_sp <- seq(a_sc + 0.1 * span_sp, b_sc - 0.1 * span_sp, length.out = 5)
prob_grid_sp <- c(0.1, 0.25, 0.5, 0.75, 0.9)

newdata_sp <- data.frame(x_d = x_d_new_sp, x_s = x_s_new_sp)

.write_sp_what <- function(what_name) {
  mat <- predict(fit_sc, newdata = newdata_sp,
                 q = q_grid_sp, type = what_name)
  stopifnot(is.matrix(mat) && nrow(mat) == length(q_grid_sp) &&
            ncol(mat) == m_sp)
  writeLines(format(flatten_row_major(mat), digits = 15),
             con = file.path(out_dir,
                             sprintf("scaling_predict_%s.txt", what_name)))
}

writeLines(format(x_d_new_sp, digits = 15),
           con = file.path(out_dir, "scaling_predict_x_d_new.txt"))
writeLines(format(x_s_new_sp, digits = 15),
           con = file.path(out_dir, "scaling_predict_x_s_new.txt"))
writeLines(format(q_grid_sp, digits = 15),
           con = file.path(out_dir, "scaling_predict_q_grid.txt"))
writeLines(format(prob_grid_sp, digits = 15),
           con = file.path(out_dir, "scaling_predict_prob_grid.txt"))

for (wname in c("distribution", "density", "hazard", "survivor")) {
  .write_sp_what(wname)
}

# Quantile fixture: rows = prob_grid_sp, cols = newdata rows.
q_mat_sp <- predict(fit_sc, newdata = newdata_sp,
                    prob = prob_grid_sp, type = "quantile")
stopifnot(is.matrix(q_mat_sp) && nrow(q_mat_sp) == length(prob_grid_sp) &&
          ncol(q_mat_sp) == m_sp)
writeLines(format(flatten_row_major(q_mat_sp), digits = 15),
           con = file.path(out_dir, "scaling_predict_quantile.txt"))

cat(sprintf("scaling predict ref: m=%d × k=%d (k_q=%d)\n",
            m_sp, length(q_grid_sp), length(prob_grid_sp)))

# ---------------------------------------------------------------------------
# Scaling-terms tracer for tram::Lm and tram::Survreg (issue #76).
#
# Both subclasses share the underlying scaled-baseline + scaled-predict
# machinery wired up in #71/#72 — these fixtures pin per-class R parity for
# θ_b, β, γ and the log-likelihood under the convenience surface.
#
# tram::Lm uses ``order = 1`` Bernstein on a normal base; ``negative = TRUE``
# (so ``h − X_d·β``), so the pymlt β block is sign-flipped at compare time.
# tram::Survreg fits a transformation on log-time (LogBernsteinBasis in
# pymlt); we cover all three distributions (``weibull``, ``lognormal``,
# ``loglogistic``) at right-censored time scales.  γ is sign-aligned with R
# (ADR 0002, Decision 5) for both.
#
# Fixtures written:
#   scaling_lm_*           — y, x_d, x_s, support, theta_full, loglik
#   scaling_survreg_<d>_*  — y, event, x_d, x_s, support, theta_full, loglik
#                            for d ∈ {weibull, lognormal, loglogistic}
# ---------------------------------------------------------------------------

# ---- scaling_lm (Lm + scale=~x_s, exact, normal, order=1) ----------------
set.seed(760)
n_lm_sc <- 200
x_s_lm <- rnorm(n_lm_sc)
x_d_lm <- rnorm(n_lm_sc)
y_lm_sc <- 1.0 + 0.5 * x_d_lm + rnorm(n_lm_sc, sd = exp(0.25 * x_s_lm))
a_lm_sc <- min(y_lm_sc) - 0.1
b_lm_sc <- max(y_lm_sc) + 0.1
df_lm_sc <- data.frame(y = y_lm_sc, x_d = x_d_lm, x_s = x_s_lm)
fit_lm_sc <- tram::Lm(y ~ x_d | x_s, data = df_lm_sc,
                      support = c(a_lm_sc, b_lm_sc))
theta_lm_sc <- coef(as.mlt(fit_lm_sc))
ll_lm_sc <- as.numeric(logLik(fit_lm_sc))

writeLines(format(y_lm_sc, digits = 15),
           con = file.path(out_dir, "scaling_lm_y.txt"))
writeLines(format(x_d_lm, digits = 15),
           con = file.path(out_dir, "scaling_lm_x_d.txt"))
writeLines(format(x_s_lm, digits = 15),
           con = file.path(out_dir, "scaling_lm_x_s.txt"))
writeLines(paste(format(a_lm_sc, digits = 15),
                 format(b_lm_sc, digits = 15)),
           con = file.path(out_dir, "scaling_lm_support.txt"))
writeLines(format(theta_lm_sc, digits = 15),
           con = file.path(out_dir, "scaling_lm_theta.txt"))
writeLines(format(ll_lm_sc, digits = 15),
           con = file.path(out_dir, "scaling_lm_loglik.txt"))

cat(sprintf("scaling Lm ref: n=%d, p+q_d+q_s=%d, ll=%.6f\n",
            n_lm_sc, length(theta_lm_sc), ll_lm_sc))

# ---- scaling_survreg_<dist> (Survreg + scale=~x_s, right-censored) -------
# tram::Survreg uses log-time with order=6 Bernstein on the log scale and
# the chosen parametric link (Weibull / log-normal / log-logistic).  We
# cover all three to match the convenience surface in pymlt.tram.Survreg.
.write_survreg_scaling <- function(dist_name, seed) {
  set.seed(seed)
  n <- 200
  x_s <- rnorm(n)
  x_d <- rnorm(n)
  # Heteroskedastic log-time: mean shifts with x_d, sd shifts with x_s.
  log_t <- 0.5 + 0.4 * x_d + rnorm(n, sd = exp(0.25 * x_s))
  t_obs <- exp(log_t)
  cens  <- rexp(n, rate = 0.25)
  y     <- pmin(t_obs, cens)
  event <- as.integer(t_obs <= cens)
  a <- max(min(y) * 0.9, 1e-3)
  b <- max(y) * 1.1
  df <- data.frame(y = y, event = event, x_d = x_d, x_s = x_s)
  # tram::Survreg uses a 2-parameter parametric baseline on log(t)
  # (h(log t) = (log t − α) / σ) regardless of ``order``.  A
  # ``LogBernsteinBasis(order=1)`` on the pymlt side is affine in
  # log(t) and yields an equivalent two-parameter reparameterisation
  # (matching θ exactly after the basis transformation).  We therefore
  # request ``order = 1`` on the R side as a matching cue, knowing the
  # baseline is two-parameter in either parameterisation.
  fit <- tram::Survreg(Surv(y, event) ~ x_d | x_s, data = df,
                       support = c(a, b), order = 1,
                       dist = dist_name)
  theta <- coef(as.mlt(fit))
  ll <- as.numeric(logLik(fit))
  tag <- sprintf("scaling_survreg_%s", dist_name)
  writeLines(format(y, digits = 15),
             con = file.path(out_dir, sprintf("%s_y.txt", tag)))
  writeLines(as.character(event),
             con = file.path(out_dir, sprintf("%s_event.txt", tag)))
  writeLines(format(x_d, digits = 15),
             con = file.path(out_dir, sprintf("%s_x_d.txt", tag)))
  writeLines(format(x_s, digits = 15),
             con = file.path(out_dir, sprintf("%s_x_s.txt", tag)))
  writeLines(paste(format(a, digits = 15),
                   format(b, digits = 15)),
             con = file.path(out_dir, sprintf("%s_support.txt", tag)))
  writeLines(format(theta, digits = 15),
             con = file.path(out_dir, sprintf("%s_theta.txt", tag)))
  writeLines(format(ll, digits = 15),
             con = file.path(out_dir, sprintf("%s_loglik.txt", tag)))
  cat(sprintf("scaling Survreg %s ref: n=%d, p+q_d+q_s=%d, ll=%.6f\n",
              dist_name, n, length(theta), ll))
}

.write_survreg_scaling("weibull",     761)
.write_survreg_scaling("lognormal",   762)
.write_survreg_scaling("loglogistic", 763)

# ---------------------------------------------------------------------------
# Scaling-terms inference (issue #77) — vcov, sandwich (HC0), and a Wald
# test for the γ block, for BoxCox + Coxph + Colr scaled fits.
#
# Re-uses the fits ``fit_sc`` (BoxCox + normal, exact),
# ``fit_cxs`` (Coxph, right-censored), ``fit_co`` (Colr + logistic, exact)
# generated above.  For each fit the saved fixtures are the
# ``(k × k)`` inverse-information vcov and the ``(k × k)`` HC0 sandwich,
# both flattened row-major (k = p + q_d + q_s).  An additional one-line
# Wald-test fixture pins the statistic and p-value for the contrast that
# picks out γ_1 (H0: γ_1 = 0).
#
# Sign conventions (ADR 0002, Decision 5):
#
# * BoxCox uses ``negative = TRUE`` so β_R = −β_pymlt.  Both rows and
#   columns indexed by β therefore flip sign in the vcov; the Python test
#   applies the diagonal signing matrix ``diag([1]*p + [-1]*q_d + [1]*q_s)``
#   before comparing.
# * Coxph / Colr use ``negative = FALSE`` (default) — β block is sign-
#   aligned, so no signing diagonal is required.
# * γ is sign-aligned across all three classes.
#
# Files written per model in {"boxcox", "coxph", "colr"}:
#   reference/scaling_vcov_<model>.txt          — flatten_row_major(V)
#   reference/scaling_vcov_<model>_HC0.txt      — flatten_row_major(V_HC0)
#   reference/scaling_vcov_<model>_dim.txt      — "p q_d q_s" (one line)
#   reference/scaling_vcov_<model>_wald_gamma.txt
#                                               — "W df pvalue" (one line)
# ---------------------------------------------------------------------------
.write_scaled_vcov <- function(fit, tag, p, q_d, q_s,
                               y, x_d, x_s, support, event = NULL) {
  mlt_fit <- as.mlt(fit)
  V_info <- vcov(mlt_fit)
  # HC0 sandwich = V_info %*% (U'U) %*% V_info, computed directly from the
  # pieces because ``sandwich::vcovHC`` calls ``model.frame`` which fails for
  # multi-formula scaled mlt fits.  The closed form matches HC0 exactly.
  U      <- sandwich::estfun(mlt_fit)
  V_HC0  <- V_info %*% (t(U) %*% U) %*% V_info
  k <- p + q_d + q_s
  stopifnot(nrow(V_info) == k && ncol(V_info) == k)
  stopifnot(nrow(V_HC0)  == k && ncol(V_HC0)  == k)
  # Wald test: pick out γ_1 (first γ column, column p + q_d + 1 in R).
  # H0: γ_1 = 0, info vcov, df = 1.
  theta_full <- coef(mlt_fit)
  Rmat <- matrix(0, nrow = 1, ncol = k)
  Rmat[1, p + q_d + 1L] <- 1.0
  Rtheta <- as.numeric(Rmat %*% theta_full)
  RVRt   <- as.numeric(Rmat %*% V_info %*% t(Rmat))
  W      <- Rtheta * (1 / RVRt) * Rtheta
  pval   <- 1 - pchisq(W, df = 1)
  # Self-contained dataset fixtures so each parity test can rebuild the
  # exact same fit without depending on shared upstream blocks.
  writeLines(format(y, digits = 15),
             con = file.path(out_dir,
                             sprintf("scaling_vcov_%s_y.txt", tag)))
  if (!is.null(event)) {
    writeLines(as.character(event),
               con = file.path(out_dir,
                               sprintf("scaling_vcov_%s_event.txt", tag)))
  }
  writeLines(format(x_d, digits = 15),
             con = file.path(out_dir,
                             sprintf("scaling_vcov_%s_x_d.txt", tag)))
  writeLines(format(x_s, digits = 15),
             con = file.path(out_dir,
                             sprintf("scaling_vcov_%s_x_s.txt", tag)))
  writeLines(paste(format(support[1], digits = 15),
                   format(support[2], digits = 15)),
             con = file.path(out_dir,
                             sprintf("scaling_vcov_%s_support.txt", tag)))
  writeLines(format(theta_full, digits = 15),
             con = file.path(out_dir,
                             sprintf("scaling_vcov_%s_theta.txt", tag)))
  writeLines(format(flatten_row_major(V_info), digits = 15),
             con = file.path(out_dir,
                             sprintf("scaling_vcov_%s.txt", tag)))
  writeLines(format(flatten_row_major(V_HC0),  digits = 15),
             con = file.path(out_dir,
                             sprintf("scaling_vcov_%s_HC0.txt", tag)))
  writeLines(sprintf("%d %d %d", p, q_d, q_s),
             con = file.path(out_dir,
                             sprintf("scaling_vcov_%s_dim.txt", tag)))
  writeLines(sprintf("%.15g %d %.15g", W, 1L, pval),
             con = file.path(out_dir,
                             sprintf("scaling_vcov_%s_wald_gamma.txt", tag)))
  cat(sprintf("scaling %s vcov ref: k=%d, W(gamma)=%.4f, p=%.4g\n",
              tag, k, W, pval))
}

# Dedicated BoxCox vcov fit — n=200, order=4, seed=770 gives an interior
# MLE (no active monotonicity constraints), so the full 8×8 vcov is
# numerically well-conditioned and matches pymlt's ``inv(H)`` element-wise.
set.seed(770)
n_v_bc <- 200
x_s_v_bc <- rnorm(n_v_bc)
x_d_v_bc <- rnorm(n_v_bc)
y_v_bc   <- 1.0 + 0.5 * x_d_v_bc + rnorm(n_v_bc, sd = exp(0.25 * x_s_v_bc))
a_v_bc <- min(y_v_bc) - 0.1
b_v_bc <- max(y_v_bc) + 0.1
df_v_bc <- data.frame(y = y_v_bc, x_d = x_d_v_bc, x_s = x_s_v_bc)
fit_v_bc <- tram::BoxCox(y ~ x_d | x_s, data = df_v_bc,
                         support = c(a_v_bc, b_v_bc), order = 4)
.write_scaled_vcov(fit_v_bc, "boxcox",
                   p = 5, q_d = 1, q_s = 1,
                   y = y_v_bc, x_d = x_d_v_bc, x_s = x_s_v_bc,
                   support = c(a_v_bc, b_v_bc))

# Dedicated Colr vcov fit — n=200, order=4, seed=770 gives an interior MLE.
set.seed(770)
n_v_co <- 200
x_s_v_co <- rnorm(n_v_co)
x_d_v_co <- rnorm(n_v_co)
y_v_co   <- rlogis(n_v_co, location = 0.5 * x_d_v_co,
                   scale = exp(0.25 * x_s_v_co))
a_v_co <- min(y_v_co) - 0.1
b_v_co <- max(y_v_co) + 0.1
df_v_co <- data.frame(y = y_v_co, x_d = x_d_v_co, x_s = x_s_v_co)
fit_v_co <- tram::Colr(y ~ x_d | x_s, data = df_v_co,
                       support = c(a_v_co, b_v_co), order = 4)
.write_scaled_vcov(fit_v_co, "colr",
                   p = 5, q_d = 1, q_s = 1,
                   y = y_v_co, x_d = x_d_v_co, x_s = x_s_v_co,
                   support = c(a_v_co, b_v_co))

# Coxph + scale=~x_s, right-censored.  Coxph fits are structurally
# constraint-binding on the baseline (the tail of the baseline hazard is
# under-determined), so the θ_b block of the vcov reflects R's
# active-constraint penalty and does not match pymlt's bare ``inv(H)``.
# The β / γ sub-block is the practically meaningful block for covariate
# inference and matches at the same tolerance as BoxCox / Colr.
set.seed(770)
n_v_cx <- 300
x_s_v_cx <- rnorm(n_v_cx)
x_d_v_cx <- rnorm(n_v_cx)
t_v_cx <- rexp(n_v_cx, rate = exp(0.5 + 0.4 * x_d_v_cx + 0.25 * x_s_v_cx))
cens_v_cx <- rexp(n_v_cx, rate = 0.10)
y_v_cx   <- pmin(t_v_cx, cens_v_cx)
event_v_cx <- as.integer(t_v_cx <= cens_v_cx)
a_v_cx <- min(y_v_cx) * 0.5
b_v_cx <- max(y_v_cx) * 1.1
df_v_cx <- data.frame(y = y_v_cx, event = event_v_cx,
                      x_d = x_d_v_cx, x_s = x_s_v_cx)
fit_v_cx <- tram::Coxph(Surv(y, event) ~ x_d | x_s, data = df_v_cx,
                        support = c(a_v_cx, b_v_cx), order = 4)
.write_scaled_vcov(fit_v_cx, "coxph",
                   p = 5, q_d = 1, q_s = 1,
                   y = y_v_cx, x_d = x_d_v_cx, x_s = x_s_v_cx,
                   support = c(a_v_cx, b_v_cx), event = event_v_cx)
