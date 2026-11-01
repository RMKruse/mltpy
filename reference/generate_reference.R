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
# method, so compute BIC manually using the same formula mltpy uses:
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
# directly from logLik(). This matches mltpy.anova() (deviance = 2*Δll,
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
# to cross-check mltpy.basis.BernsteinBasis.evaluate() against basefun.
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
# The Python test evaluates mltpy.log_likelihood at θ on (y, event) and
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
# the truncation correction we are validating in mltpy.
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
# .auglag_reference(): multi-start explicit-constraint reference fit (#156)
#
# Minimises fn (a negative log-likelihood built on mlt's own likelihood) over
# the linear rows A %*% theta >= 0 with alabama::auglag from every start in
# `starts`, at tolerances far below mlt()'s defaults.  Enforces, via
# stopifnot, the convergence record a reference fixture needs:
#   * every start converges and all starts agree (theta spread < 1e-6);
#   * KKT at the best fit: the rows with slack < 1e-8 are exactly
#     `expected_active`, their multipliers solve g = A_act' lambda with
#     lambda > 1e-3, and the stationarity residual is < 1e-5.
# Used where plain mlt() reports convergence = 0 short of the optimum on a
# surface that is flat along an active constraint face.
# Last run with R 4.6.1, mlt 1.8-4, alabama 2025.1.0.
# ---------------------------------------------------------------------------

.auglag_reference <- function(fn, gr, A, starts, expected_active) {
  fits <- lapply(starts, function(s) suppressWarnings(alabama::auglag(
    par = s, fn = fn, gr = gr,
    hin = function(th) drop(A %*% th), hin.jac = function(th) A,
    control.outer = list(trace = FALSE, eps = 1e-12, itmax = 500L,
                         method = "BFGS", kkt2.check = FALSE),
    control.optim = list(maxit = 5000L, reltol = 1e-15)
  )))
  conv   <- vapply(fits, `[[`, integer(1), "convergence")
  value  <- vapply(fits, `[[`, numeric(1), "value")
  TH     <- t(vapply(fits, `[[`, numeric(ncol(A)), "par"))
  spread <- max(apply(TH, 2, function(z) diff(range(z))))
  stopifnot(all(conv == 0L), spread < 1e-6, diff(range(value)) < 1e-9)

  theta <- TH[which.min(value), ]
  g     <- gr(theta)
  slack <- drop(A %*% theta)
  act   <- which(slack < 1e-8)
  A_act <- A[act, , drop = FALSE]
  lam   <- qr.solve(t(A_act), g)
  stat  <- max(abs(g - drop(t(A_act) %*% lam)))
  stopifnot(identical(act, unname(expected_active)), all(lam > 1e-3),
            stat < 1e-5, all(slack > -1e-10))
  list(theta = theta, n_conv = sum(conv == 0L), n_starts = length(conv),
       spread = spread, active = act, lambda = lam, stationarity = stat)
}

.report_auglag_reference <- function(label, ref, dll, dth) {
  cat(sprintf(paste0(
    "%s: %d/%d auglag starts converged, theta spread %.1e; ",
    "active rows {%s}, multipliers {%s}, stationarity %.1e; ",
    "cross-check |dll| %.1e, |dtheta| %.1e\n"),
    label, ref$n_conv, ref$n_starts, ref$spread,
    paste(ref$active, collapse = ", "),
    paste(sprintf("%.4g", ref$lambda), collapse = ", "),
    ref$stationarity, dll, dth))
}

# ---------------------------------------------------------------------------
# max_extreme_value (standard Gumbel, right) reference (issue #156)
#
# Model: todistr = "MaxExtrVal", order-4 Bernstein on (0, 1), exact data, no
# covariates.  Writes:
#   mlt_maxextrval_theta.txt — Bernstein coefficients (order=4, 5 values)
#   mlt_maxextrval_y.txt     — 200 observations on (0, 1)
#   mlt_maxextrval_ll.txt    — scalar log-likelihood (mlt's own loglik at theta)
#
# The Gumbel density is positive on the whole line, so there is no -Inf
# barrier and plain mlt() is not stalled.  But two monotonicity rows are
# active at the MLE (theta_b[1] = theta_b[2] = theta_b[3]) and the surface is
# flat along that face: mlt() reports convergence = 0 with theta ~1.8e-6 and
# logLik ~1.8e-8 short of the optimum, and even mltoptim(abstol = reltol =
# 1e-12) does not close the theta gap.  The reference is therefore the
# multi-start explicit-constraint fit on mlt's own loglik / scorei, with
# plain mlt() kept as a cross-check (it was the fixture before #156).
# ---------------------------------------------------------------------------

set.seed(11)
y_mev  <- runif(200, 0.02, 0.98)
b_mev  <- Bernstein_basis(m, order = 4, ui = "increasing")
ctm_mev <- ctm(b_mev, todistr = "MaxExtrVal")
d_mev  <- data.frame(y = y_mev)
w_mev  <- rep(1, nrow(d_mev))
fit_mev <- mlt(ctm_mev, data = d_mev)
mlt_mev <- mlt(ctm_mev, data = d_mev, dofit = FALSE)

# Monotonicity rows D theta_b >= 0; row k is theta_b[k] - theta_b[k-1] >= 0
# (0-indexed theta_b, mltpy's convention).
A_mev <- diff(diag(5L))
set.seed(2024)
starts_mev <- c(list(unname(coef(fit_mev))),
                replicate(9, cumsum(c(rnorm(1, 0, 2), rexp(4))), simplify = FALSE))
ref_mev <- .auglag_reference(
  fn = function(th) -mlt_mev$loglik(th, weights = w_mev),
  gr = function(th) -colSums(mlt_mev$scorei(th, weights = w_mev)),
  A = A_mev, starts = starts_mev,
  expected_active = c(mono_1_2 = 2L, mono_2_3 = 3L)
)

theta_mev <- setNames(ref_mev$theta, names(coef(fit_mev)))
ll_mev    <- mlt_mev$loglik(theta_mev, weights = w_mev)

dll_mev <- abs(as.numeric(logLik(fit_mev)) - ll_mev)
dth_mev <- max(abs(coef(fit_mev) - theta_mev))
stopifnot(fit_mev$convergence == 0, dll_mev < 1e-7, dth_mev < 1e-4)
.report_auglag_reference("MaxExtrVal (cross-check: plain mlt())",
                         ref_mev, dll_mev, dth_mev)

writeLines(format(theta_mev, digits = 15),
           con = file.path(out_dir, "mlt_maxextrval_theta.txt"))
writeLines(format(y_mev, digits = 15),
           con = file.path(out_dir, "mlt_maxextrval_y.txt"))
writeLines(format(ll_mev, digits = 15),
           con = file.path(out_dir, "mlt_maxextrval_ll.txt"))

cat(sprintf("MaxExtrVal: n=%d, ll=%.6f\n", length(y_mev), ll_mev))

# ---------------------------------------------------------------------------
# exponential reference (issue #156)
#
# Model: todistr = "Exponential", order-4 Bernstein on (0, 1), exact data, no
# covariates.  Writes:
#   mlt_exponential_theta.txt — Bernstein coefficients (order=4, 5 values)
#   mlt_exponential_y.txt     — 200 observations on (0, 1)
#   mlt_exponential_ll.txt    — scalar log-likelihood (mlt's own loglik at theta)
#
# Why plain mlt() is NOT used:
#   R's .Exponential() carries no support constraint.  What keeps h >= 0 is
#   only dexp(h, log = TRUE) = -Inf for h < 0, a barrier at the observations
#   h(y_i) >= 0, while mlt's score uses the smooth extension d log f / dh = -1
#   everywhere.  The gradient keeps pointing into the -Inf region, the line
#   search backtracks to a zero step, and mlt() returns convergence = 0 next to
#   its start.  The previous fixture, from
#   mlt(theta = seq(0.1, 5, length.out = 5)), stalled at logLik -154.91 against
#   an optimum of -7.51 (research/exp-scaling-r-route/README.md §11).  mltpy
#   also enforces a different support rule: the lower-support-end rule
#   h(a | x_i) >= 0 (ADR 0005 Decision 1), which with no covariates is the
#   linear row theta_b[0] >= 0, a row R's mlt() never adds.
#
# Route:
#   1. Reference: alabama::auglag on mlt's exact log-likelihood with the rows
#      theta_b[0] >= 0 and D theta_b >= 0 written out explicitly, started from
#      10 feasible points.  The objective is the smooth extension
#      -sum(-h_i + log h'_i), which equals mlt's logLik wherever h_i >= 0
#      (true for every feasible theta, since then h(y) >= h(0) = theta_b[0]);
#      it and its gradient are checked against mlt's loglik / scorei below.
#   2. Cross-check: mlt(fixed = c("Bs1(y)" = 0)).  With no covariates the
#      support row is active at the MLE: shifting theta_b down in parallel
#      leaves every monotonicity row and every h'_i unchanged and lowers every
#      h_i, which strictly raises the likelihood until theta_b[0] = 0.  Pinning
#      Bs1 therefore poses mltpy's problem to R's own optimiser.  mlt() stops
#      slightly early on this flat surface (theta off by ~5e-6 along Bs5, logLik
#      ~3e-10 below the auglag optimum), so it is the cross-check, not the
#      reference.
#   Convergence record, enforced with stopifnot: every start converges and
#   all starts agree; KKT at the reference with active set
#   {theta_b[0] = 0, theta_b[2] = theta_b[3]}, strictly positive multipliers
#   and a small stationarity residual; mlt(fixed) agrees on logLik.
# ---------------------------------------------------------------------------

set.seed(13)
y_exp  <- runif(200, 0.02, 0.98)
b_exp  <- Bernstein_basis(m, order = 4, ui = "increasing")
ctm_exp <- ctm(b_exp, todistr = "Exponential")
d_exp  <- data.frame(y = y_exp)
w_exp  <- rep(1, nrow(d_exp))
p_exp  <- 5L

# mlt's own likelihood machinery (unfitted), used to validate the objective
# and to compute the stored logLik.
mlt_exp <- mlt(ctm_exp, data = d_exp, dofit = FALSE)

B_exp  <- model.matrix(b_exp, data = d_exp)
Bp_exp <- model.matrix(b_exp, data = d_exp, deriv = c(y = 1L))
nll_exp  <- function(th) sum(B_exp %*% th) - sum(log(Bp_exp %*% th))
grad_exp <- function(th) colSums(B_exp) - colSums(Bp_exp / drop(Bp_exp %*% th))

# Constraint rows: support theta_b[0] >= 0, then monotonicity D theta_b >= 0.
A_exp <- rbind(c(1, rep(0, p_exp - 1L)), diff(diag(p_exp)))

th_chk <- c(0.1, 0.5, 1, 2, 4)
stopifnot(
  abs(nll_exp(th_chk) + mlt_exp$loglik(th_chk, weights = w_exp)) < 1e-10,
  max(abs(grad_exp(th_chk) +
          colSums(mlt_exp$scorei(th_chk, weights = w_exp)))) < 1e-10
)

set.seed(2024)
starts_exp <- c(
  list(seq(0.1, 5.0, length.out = p_exp)),
  replicate(9, cumsum(c(runif(1, 0, 2), rexp(p_exp - 1L))), simplify = FALSE)
)
# Row 1 is the support row theta_b[0] >= 0; row 1 + k is the monotonicity
# row theta_b[k] - theta_b[k-1] >= 0 (0-indexed theta_b, mltpy's convention).
ref_exp <- .auglag_reference(
  fn = nll_exp, gr = grad_exp, A = A_exp, starts = starts_exp,
  expected_active = c(support = 1L, mono_2_3 = 4L)
)

# theta_b[0] is stored as the solver returns it (~ -5e-13, within the 1e-10
# slack allowed above).  h(y_i) > 0 at every observation regardless, so mlt's
# loglik is finite there; it is not snapped to 0 to keep the file solver
# output.
theta_exp <- setNames(ref_exp$theta, names(coef(mlt_exp)))
ll_exp    <- mlt_exp$loglik(theta_exp, weights = w_exp)

fit_exp_fixed <- mlt(ctm_exp, data = d_exp, fixed = c("Bs1(y)" = 0),
                     optim = mltoptim(abstol = 1e-12, reltol = 1e-12))
dll_fixed <- abs(as.numeric(logLik(fit_exp_fixed)) - ll_exp)
dth_fixed <- max(abs(coef(fit_exp_fixed) - theta_exp))
stopifnot(fit_exp_fixed$convergence == 0, dll_fixed < 1e-8, dth_fixed < 1e-4)
.report_auglag_reference("Exponential (cross-check: mlt(fixed Bs1 = 0))",
                         ref_exp, dll_fixed, dth_fixed)

writeLines(format(theta_exp, digits = 15),
           con = file.path(out_dir, "mlt_exponential_theta.txt"))
writeLines(format(y_exp, digits = 15),
           con = file.path(out_dir, "mlt_exponential_y.txt"))
writeLines(format(ll_exp, digits = 15),
           con = file.path(out_dir, "mlt_exponential_ll.txt"))

cat(sprintf("Exponential: n=%d, ll=%.6f\n", length(y_exp), ll_exp))

# ---------------------------------------------------------------------------
# exponential + shift covariate + offset reference (issue #160)
#
# Model: todistr = "Exponential", order-3 Bernstein on (0, 1) in y, one linear
# shift term x (h(y|x) = a(y)'theta_b + x * beta + offset, mlt's default
# positive sign), exact data, per-row offset.  Writes:
#   mlt_exponential_offset_y.txt      — 200 responses on (0.05, 0.95)
#   mlt_exponential_offset_x.txt      — 200 covariate values on (0, 1)
#   mlt_exponential_offset_offset.txt — 200 per-row offsets (N(0, 0.3^2))
#   mlt_exponential_offset_theta.txt  — c(theta_b (4), beta (1))
#   mlt_exponential_offset_ll.txt     — mlt's own loglik at theta
#
# The fixture checks that mltpy's support rows include the offset (ADR 0005
# Decision 1): h(a | x_i) = theta_b[0] + x_i * beta + offset_i >= 0 for every
# training row.  R mlt itself has no support rule, so, as for the
# no-covariate exponential fixture above (#156), the reference is an
# explicit-constraint alabama::auglag fit on mlt's likelihood with those rows
# written out, multi-started from feasible points.
#
# Cross-check (independent of auglag): with theta_b = theta_0 + (0, delta)
# the problem separates.  h'(y) depends on delta only, the support rows on
# (theta_0, beta) only, and the objective's (theta_0, beta) part is the
# linear term n * theta_0 + beta * sum(x).  So
#   * (theta_0, beta) solves the LP  min n*theta_0 + beta*sum(x)  subject to
#     theta_0 + x_i*beta >= -offset_i: the upper-hull edge of the points
#     (x_i, -offset_i) above mean(x), found by enumerating point pairs;
#   * delta is the no-covariate exponential problem at theta_0 = 0, solved by
#     plain mlt(fixed = c("Bs1(y)" = 0)) on y alone.
# ---------------------------------------------------------------------------

set.seed(160)
n_eo   <- 200L
x_eo   <- runif(n_eo, 0, 1)
y_eo   <- runif(n_eo, 0.05, 0.95)
off_eo <- rnorm(n_eo, 0, 0.3)
d_eo   <- data.frame(y = y_eo, x = x_eo)
w_eo   <- rep(1, n_eo)

m_eo  <- numeric_var("y", support = c(0, 1), bounds = c(0, 1))
b_eo  <- Bernstein_basis(m_eo, order = 3, ui = "increasing")
sh_eo <- as.basis(~ x, data = d_eo, remove_intercept = TRUE)
ctm_eo <- ctm(b_eo, shifting = sh_eo, todistr = "Exponential")
mlt_eo <- mlt(ctm_eo, data = d_eo, offset = off_eo, dofit = FALSE)
p_eo   <- 4L

B_eo  <- model.matrix(b_eo, data = d_eo)
Bp_eo <- model.matrix(b_eo, data = d_eo, deriv = c(y = 1L))
h_eo  <- function(th) drop(B_eo %*% th[1:p_eo]) + x_eo * th[p_eo + 1L] + off_eo
nll_eo <- function(th) sum(h_eo(th)) - sum(log(Bp_eo %*% th[1:p_eo]))
grad_eo <- function(th) c(
  colSums(B_eo) - colSums(Bp_eo / drop(Bp_eo %*% th[1:p_eo])),
  sum(x_eo)
)

# Constraint rows, hin(theta) = A theta - b >= 0: support rows
# theta_b[0] + x_i beta >= -offset_i, then monotonicity D theta_b >= 0.
A_eo <- rbind(
  cbind(1, matrix(0, n_eo, p_eo - 1L), x_eo),
  cbind(diff(diag(p_eo)), 0)
)
b_eo_rhs <- c(-off_eo, rep(0, p_eo - 1L))

th_chk_eo <- c(1, 1.5, 2, 4, 0.3)
stopifnot(
  abs(nll_eo(th_chk_eo) + mlt_eo$loglik(th_chk_eo, weights = w_eo)) < 1e-10,
  max(abs(grad_eo(th_chk_eo) +
          colSums(mlt_eo$scorei(th_chk_eo, weights = w_eo)))) < 1e-10
)

set.seed(2160)
feasible_start_eo <- function(beta) {
  steps <- cumsum(c(0, rexp(p_eo - 1L)))
  th0 <- max(-off_eo - x_eo * beta) + runif(1, 0.01, 1)
  c(th0 + steps, beta)
}
starts_eo <- c(
  list(c(max(-off_eo) + 0.1 + seq(0, 1, length.out = p_eo), 0)),
  lapply(runif(9, -1, 1), feasible_start_eo)
)
fits_eo <- lapply(starts_eo, function(s) suppressWarnings(alabama::auglag(
  par = s, fn = nll_eo, gr = grad_eo,
  hin = function(th) drop(A_eo %*% th) - b_eo_rhs,
  hin.jac = function(th) A_eo,
  control.outer = list(trace = FALSE, eps = 1e-12, itmax = 500L,
                       method = "BFGS", kkt2.check = FALSE),
  control.optim = list(maxit = 5000L, reltol = 1e-15)
)))
conv_eo  <- vapply(fits_eo, `[[`, integer(1), "convergence")
value_eo <- vapply(fits_eo, `[[`, numeric(1), "value")
TH_eo    <- t(vapply(fits_eo, `[[`, numeric(p_eo + 1L), "par"))
spread_eo <- max(apply(TH_eo, 2, function(z) diff(range(z))))
# Every start converges to the same active set.  (theta_0, beta) sit on an
# LP vertex and agree tightly; the smooth delta block is flat along
# theta_b[3], where BFGS stops ~1e-5 early (cf. #156), hence step 2.
act_by_start <- lapply(seq_len(nrow(TH_eo)), function(k)
  which(drop(A_eo %*% TH_eo[k, ]) - b_eo_rhs < 1e-6))
stopifnot(all(conv_eo == 0L), diff(range(value_eo)) < 1e-6,
          max(apply(TH_eo[, c(1L, p_eo + 1L)], 2, function(z) diff(range(z)))) < 1e-7,
          all(vapply(act_by_start, identical, logical(1), act_by_start[[1L]])))
act_eo   <- act_by_start[[1L]]
A_act_eo <- A_eo[act_eo, , drop = FALSE]
b_act_eo <- b_eo_rhs[act_eo]

# Step 2: equality-constrained Newton on that active set, from the best start.
# theta = theta_p + Z u with A_act Z = 0 and A_act theta_p = b_act.  The
# Hessian of the nll is Bp' diag(1 / (Bp theta_b)^2) Bp on the theta_b block
# (zero on beta); Z' H Z is positive definite because the active support rows
# pin theta_0 and beta.
hess_eo <- function(th) {
  hp <- drop(Bp_eo %*% th[1:p_eo])
  H  <- matrix(0, p_eo + 1L, p_eo + 1L)
  H[1:p_eo, 1:p_eo] <- crossprod(Bp_eo / hp)
  H
}
qr_act <- qr(t(A_act_eo))
Z_eo   <- qr.Q(qr_act, complete = TRUE)[, -seq_len(nrow(A_act_eo)), drop = FALSE]
theta_eo <- TH_eo[which.min(value_eo), ]
theta_eo <- theta_eo - drop(qr.solve(A_act_eo, drop(A_act_eo %*% theta_eo) - b_act_eo))
for (it in 1:50) {
  zg <- drop(crossprod(Z_eo, grad_eo(theta_eo)))
  if (max(abs(zg)) < 1e-12) break
  theta_eo <- theta_eo - drop(Z_eo %*% solve(crossprod(Z_eo, hess_eo(theta_eo) %*% Z_eo), zg))
}
names(theta_eo) <- names(coef(mlt_eo))
ll_eo <- mlt_eo$loglik(theta_eo, weights = w_eo)
dpolish_eo <- max(abs(theta_eo - TH_eo[which.min(value_eo), ]))

# KKT at the polished point: g = A_act' lambda with lambda > 0 on the active
# rows, every row feasible, and no worse than any auglag start.
g_eo     <- grad_eo(theta_eo)
slack_eo <- drop(A_eo %*% theta_eo) - b_eo_rhs
lam_eo   <- qr.solve(t(A_act_eo), g_eo)
stat_eo  <- max(abs(g_eo - drop(t(A_act_eo) %*% lam_eo)))
stopifnot(all(lam_eo > 1e-3), stat_eo < 1e-9, all(slack_eo > -1e-12),
          identical(which(slack_eo < 1e-8), act_eo),
          nll_eo(theta_eo) <= min(value_eo) + 1e-10,
          # two support rows (an upper-hull edge) are active
          sum(act_eo <= n_eo) == 2L)

# Cross-check 1: (theta_0, beta) is the LP vertex.
pairs_eo <- t(combn(n_eo, 2L))
best_lp  <- c(Inf, NA, NA)
for (k in seq_len(nrow(pairs_eo))) {
  i <- pairs_eo[k, 1L]; j <- pairs_eo[k, 2L]
  bt <- (off_eo[i] - off_eo[j]) / (x_eo[j] - x_eo[i])
  t0 <- -off_eo[i] - x_eo[i] * bt
  if (all(t0 + x_eo * bt >= -off_eo - 1e-12)) {
    obj <- n_eo * t0 + bt * sum(x_eo)
    if (obj < best_lp[1L]) best_lp <- c(obj, t0, bt)
  }
}
dlp_eo <- max(abs(theta_eo[c(1L, p_eo + 1L)] - best_lp[2:3]))

# Cross-check 2: delta from plain mlt on y alone with Bs1 pinned at 0.
ctm_eo0 <- ctm(b_eo, todistr = "Exponential")
fit_eo0 <- mlt(ctm_eo0, data = d_eo["y"], fixed = c("Bs1(y)" = 0),
               optim = mltoptim(abstol = 1e-12, reltol = 1e-12))
ddelta_eo <- max(abs((theta_eo[1:p_eo] - theta_eo[1L]) - coef(fit_eo0)))
stopifnot(fit_eo0$convergence == 0, dlp_eo < 1e-10, ddelta_eo < 1e-6)

cat(sprintf(paste0(
  "Exponential + offset: %d/%d auglag starts converged, theta spread %.1e, ",
  "Newton polish moved theta %.1e; active rows {%s}, multipliers {%s}, ",
  "stationarity %.1e; LP |d(theta_0, beta)| %.1e; ",
  "mlt(fixed Bs1 = 0) |d delta| %.1e\n"),
  sum(conv_eo == 0L), length(conv_eo), spread_eo, dpolish_eo,
  paste(act_eo, collapse = ", "), paste(sprintf("%.4g", lam_eo), collapse = ", "),
  stat_eo, dlp_eo, ddelta_eo))

writeLines(format(y_eo, digits = 15),
           con = file.path(out_dir, "mlt_exponential_offset_y.txt"))
writeLines(format(x_eo, digits = 15),
           con = file.path(out_dir, "mlt_exponential_offset_x.txt"))
writeLines(format(off_eo, digits = 15),
           con = file.path(out_dir, "mlt_exponential_offset_offset.txt"))
writeLines(format(theta_eo, digits = 15),
           con = file.path(out_dir, "mlt_exponential_offset_theta.txt"))
writeLines(format(ll_eo, digits = 15),
           con = file.path(out_dir, "mlt_exponential_offset_ll.txt"))

cat(sprintf("Exponential + offset: n=%d, ll=%.10f, theta = %s\n", n_eo, ll_eo,
            paste(sprintf("%.8f", theta_eo), collapse = ", ")))

# ---------------------------------------------------------------------------
# exponential + scaling term, no shift term (issue #114)
#
# Model: todistr = "Exponential", order-4 Bernstein in y on [0, max(y)], one
# scaling term x_s and no shift term, exact data:
#   h(y | x_s) = a(y)'theta_b * exp(0.5 * x_s * gamma).
# Writes:
#   mlt_exponential_scaling_y.txt       — 150 responses
#   mlt_exponential_scaling_xs.txt      — 150 scaling-covariate values
#   mlt_exponential_scaling_support.txt — Bernstein support c(0, max(y))
#   mlt_exponential_scaling_theta.txt   — c(theta_b (5), gamma (1)), in
#                                         mltpy's [theta_b | gamma] order
#   mlt_exponential_scaling_ll.txt      — mlt's own loglik at theta
#   mlt_exponential_scaling_mu.txt      — KKT multipliers of the rows
#                                         c(theta_b[0] >= 0, D theta_b >= 0
#                                         (4 rows)); 0 on inactive rows
#
# Why this case has exact R parity (ADR 0005 Decisions 2 and 4,
# research/exp-scaling-r-route/README.md §5, §10, §12): mltpy's support rule
# h(a | x_i) = theta_b[0] * exp(0.5 * x_s,i * gamma) >= 0 collapses to the
# linear row theta_b[0] >= 0, since the scaling factor is strictly positive.
# That row is active at the MLE: shifting theta_b down in parallel leaves every
# monotonicity row and the sign of every h'_i unchanged and lowers every h_i,
# which strictly raises the likelihood sum(-h_i + log h'_i) until
# theta_b[0] = 0.  So mlt(fixed = c("Bs1(y)" = 0)) poses exactly mltpy's
# problem to R's own optimiser, and with Bs1 pinned h(y_i | x_i) > 0 at every
# observation, so mlt never touches its dexp -Inf barrier.  tram::Aareg
# (y ~ 0 | x_s) is the same model (it pins Bs1 = 0 on support [0, max(y)]).
#
# Coefficient order Bs1..Bs5, scl_x_s is mltpy's [theta_b | gamma]; raw ctm
# scales by exp(0.5 * x_s * gamma), so gamma needs no sign flip.
#
# Route:
#   1. mlt(fixed = c("Bs1(y)" = 0)) with tight optimiser tolerances.  It stops
#      with a stationarity residual of ~2e-5 on this flat surface (#126 §12),
#      so
#   2. it is polished by equality-constrained Newton on its active set (the
#      monotonicity rows with zero slack), using mlt's own score and Hessian
#      closures over the free parameters.
#   3. KKT check in the full parameter space: support row theta_b[0] >= 0 and
#      the active monotonicity rows have strictly positive multipliers, the
#      stationarity residual is ~1e-12, every row is feasible.
#   4. Cross-checks: the unpolished mlt fit and tram::Aareg agree on logLik
#      and theta.
# ---------------------------------------------------------------------------

set.seed(114)
n_es  <- 150L
xs_es <- rnorm(n_es)
y_es  <- rexp(n_es) / (2 * exp(0.5 * 0.6 * xs_es))
d_es  <- data.frame(y = y_es, x_s = xs_es)
w_es  <- rep(1, n_es)
sup_es <- c(0, max(y_es))
p_es  <- 5L

b_es   <- Bernstein_basis(numeric_var("y", support = sup_es, bounds = c(0, Inf)),
                          order = 4, ui = "increasing")
ctm_es <- ctm(b_es, scaling = ~ x_s, data = d_es, todistr = "Exponential")
stopifnot(identical(names(coef(ctm_es)),
                    c(sprintf("Bs%d(y)", 1:5), "scl_x_s")))

# Step 1: R's own fit with Bs1 pinned at 0.
fit_es <- mlt(ctm_es, data = d_es, fixed = c("Bs1(y)" = 0),
              optim = mltoptim(abstol = 1e-12, reltol = 1e-12))
stopifnot(fit_es$convergence == 0)
ll_es_raw    <- as.numeric(logLik(fit_es))
theta_es_raw <- coef(fit_es)

# Step 2: Newton polish over the free parameters (Bs2..Bs5, scl_x_s).
# fit_es$hessian is the Hessian of the negative log-likelihood; the score is
# the gradient of the log-likelihood.  Monotonicity rows over the free
# parameters (theta_b[0] = 0 folded in): Bs2 >= 0, Bs{k+1} - Bs{k} >= 0.
A_free_es <- cbind(rbind(c(1, 0, 0, 0), diff(diag(4L))), 0)
par_es    <- fit_es$par
act_free  <- which(drop(A_free_es %*% par_es) < 1e-6)
stopifnot(length(act_free) >= 1L)
A_act_free <- A_free_es[act_free, , drop = FALSE]
Z_es <- qr.Q(qr(t(A_act_free)), complete = TRUE)[, -seq_along(act_free), drop = FALSE]
par_es <- par_es - drop(qr.solve(A_act_free, drop(A_act_free %*% par_es)))
for (it in 1:50) {
  s_es <- colSums(fit_es$scorei(par_es, weights = w_es))
  zs <- drop(crossprod(Z_es, s_es))
  if (max(abs(zs)) < 1e-12) break
  H_es <- fit_es$hessian(par_es, weights = w_es)
  par_es <- par_es + drop(Z_es %*% solve(crossprod(Z_es, H_es %*% Z_es), zs))
}
theta_es <- c(0, par_es)
names(theta_es) <- names(coef(ctm_es))

# Step 3: KKT in the full space, with mlt's score over all parameters.
mlt_es  <- mlt(ctm_es, data = d_es, dofit = FALSE)
ll_es   <- mlt_es$loglik(theta_es, weights = w_es)
g_es    <- -colSums(mlt_es$scorei(theta_es, weights = w_es))   # grad of nll
A_es    <- cbind(rbind(c(1, rep(0, p_es - 1L)), diff(diag(p_es))), 0)
slack_es <- drop(A_es %*% theta_es)
act_es   <- which(slack_es < 1e-8)
lam_es   <- qr.solve(t(A_es[act_es, , drop = FALSE]), g_es)
stat_es  <- max(abs(g_es - drop(t(A_es[act_es, , drop = FALSE]) %*% lam_es)))
dpolish_es <- max(abs(theta_es - theta_es_raw))
stopifnot(1L %in% act_es, all(lam_es > 1e-3), stat_es < 1e-9,
          all(slack_es > -1e-12), ll_es >= ll_es_raw - 1e-12,
          dpolish_es < 1e-4)

# Step 4: cross-check with tram::Aareg (same model, own support c(0, max y)).
# Aareg evaluates tram_data() unqualified, so tram must be attached.
suppressPackageStartupMessages({
  library(tram)
})
fit_es_aareg <- suppressWarnings(Aareg(y ~ 0 | x_s, data = d_es, order = 4))
dll_aareg_es <- abs(as.numeric(logLik(fit_es_aareg)) - ll_es)
dth_aareg_es <- max(abs(coef(fit_es_aareg, with_baseline = TRUE) - theta_es))
stopifnot(dll_aareg_es < 1e-7, dth_aareg_es < 1e-3)

cat(sprintf(paste0(
  "Exponential + scaling (no shift): mlt(fixed Bs1 = 0) ll %.12f, Newton ",
  "polish moved theta %.1e to ll %.12f; active rows {%s}, multipliers {%s}, ",
  "stationarity %.1e; Aareg |d ll| %.1e, |d theta| %.1e\n"),
  ll_es_raw, dpolish_es, ll_es, paste(act_es, collapse = ", "),
  paste(sprintf("%.4g", lam_es), collapse = ", "), stat_es,
  dll_aareg_es, dth_aareg_es))

writeLines(format(y_es, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_y.txt"))
writeLines(format(xs_es, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_xs.txt"))
writeLines(format(sup_es, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_support.txt"))
writeLines(format(theta_es, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_theta.txt"))
writeLines(format(ll_es, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_ll.txt"))
mu_es <- numeric(nrow(A_es))
mu_es[act_es] <- lam_es
writeLines(format(mu_es, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_mu.txt"))

# ---------------------------------------------------------------------------
# exponential + shift term + scaling term + offset (issue #115)
#
# Model: todistr = "Exponential", order-4 Bernstein in y on [0, max(y)], one
# shift term x_d, one scaling term x_s and a per-row offset, exact data:
#   h(y | x) = a(y)'theta_b * exp(0.5 * x_s * gamma) + x_d * beta + offset.
# Writes:
#   mlt_exponential_scaling_shift_y.txt       — 150 responses
#   mlt_exponential_scaling_shift_xd.txt      — 150 shift-covariate values
#   mlt_exponential_scaling_shift_xs.txt      — 150 scaling-covariate values
#   mlt_exponential_scaling_shift_offset.txt  — 150 per-row offsets
#   mlt_exponential_scaling_shift_support.txt — Bernstein support c(0, max(y))
#   mlt_exponential_scaling_shift_theta.txt   — c(theta_b (5), beta (1),
#                                               gamma (1)), mltpy's
#                                               [theta_b | beta | gamma]
#   mlt_exponential_scaling_shift_ll.txt      — mlt's own loglik at theta
#   mlt_exponential_scaling_shift_mu.txt      — KKT multipliers in mltpy's
#                                               mu_ineq order: D theta_b >= 0
#                                               (4 rows), then the 150
#                                               support rows; 0 on inactive
#
# mltpy's support rule (ADR 0005 Decisions 1-2) is one nonlinear row per
# training observation,
#   g_i(theta) = theta_b[0] * exp(0.5 * x_s,i * gamma) + x_d,i * beta
#                + offset_i >= 0,
# and R mlt has no support rule (it keeps support only through dexp = -Inf at
# the observed points and stalls on shift + exponential, #126).  So, as ADR
# 0005 Decision 4 prescribes for shift + scale, the reference is an
# explicit-constraint alabama::auglag fit on mlt's likelihood (checked against
# mlt's loglik / scorei) with the hin rows g_i and D theta_b >= 0 written out,
# multi-started from feasible points.  It validates mltpy's optimiser on
# mltpy's model, not R's modelling choice.
#
# Without an offset the rule pins every h(a | x_i) to 0 (beta = 0, a fully
# degenerate active set, ADR 0005 "Known cost").  The offset breaks that
# symmetry: two support rows bind (multipliers ~70) and no monotonicity row
# does, with linearly independent normals and strictly positive multipliers —
# a non-degenerate active set — and gamma ~0.42 is clearly non-zero, so the
# gamma column of the Jacobian matters at the optimum.
#
# Route:
#   1. Multi-start alabama::auglag (10 feasible starts); all converge to the
#      same active set and objective.
#   2. Newton on the KKT system of that active set,
#        grad f(theta) - J_A(theta)' lambda = 0,  g_A(theta) = 0,
#      with the analytic Hessian of the nll and the closed-form second
#      derivatives of g_i (ADR 0005 Decision 2), to stationarity ~1e-12.
#   3. KKT check in the full space: every row feasible, lambda > 0 on the
#      active rows, no row with slack < 1e-8 outside the active set, and the
#      polished objective is no worse than any auglag start.
# ---------------------------------------------------------------------------

set.seed(115)
n_ss   <- 150L
xd_ss  <- runif(n_ss, 0, 1)
xs_ss  <- rnorm(n_ss)
off_ss <- rnorm(n_ss, 0, 0.3)
y_ss   <- rexp(n_ss) / (2 * exp(0.5 * 0.6 * xs_ss))
d_ss   <- data.frame(y = y_ss, x_d = xd_ss, x_s = xs_ss)
w_ss   <- rep(1, n_ss)
sup_ss <- c(0, max(y_ss))
p_ss   <- 5L
k_ss   <- p_ss + 2L   # [theta_b (5) | beta | gamma]

b_ss   <- Bernstein_basis(numeric_var("y", support = sup_ss, bounds = c(0, Inf)),
                          order = 4, ui = "increasing")
ctm_ss <- ctm(b_ss, shifting = ~ x_d, scaling = ~ x_s, data = d_ss,
              todistr = "Exponential")
stopifnot(identical(names(coef(ctm_ss)),
                    c(sprintf("Bs%d(y)", 1:5), "sft_x_d", "scl_x_s")))
mlt_ss <- mlt(ctm_ss, data = d_ss, offset = off_ss, dofit = FALSE)

B_ss  <- model.matrix(b_ss, data = d_ss)
Bp_ss <- model.matrix(b_ss, data = d_ss, deriv = c(y = 1L))
ib_ss <- 1:p_ss
f_ss  <- function(th) exp(0.5 * xs_ss * th[k_ss])
# Exact-data exponential nll: sum(h_i) - sum(log h'_i), with
# h_i = h0_i f_i + x_d,i beta + offset_i and h'_i = h0'_i f_i.
nll_ss <- function(th) {
  f <- f_ss(th)
  sum(drop(B_ss %*% th[ib_ss]) * f + xd_ss * th[p_ss + 1L] + off_ss) -
    sum(log(drop(Bp_ss %*% th[ib_ss]))) - sum(0.5 * xs_ss * th[k_ss])
}
grad_ss <- function(th) {
  f  <- f_ss(th)
  h0 <- drop(B_ss %*% th[ib_ss])
  hp <- drop(Bp_ss %*% th[ib_ss])
  c(colSums(B_ss * f) - colSums(Bp_ss / hp),
    sum(xd_ss),
    sum(0.5 * xs_ss * h0 * f) - sum(0.5 * xs_ss))
}
hess_ss <- function(th) {
  f  <- f_ss(th)
  h0 <- drop(B_ss %*% th[ib_ss])
  hp <- drop(Bp_ss %*% th[ib_ss])
  H  <- matrix(0, k_ss, k_ss)
  H[ib_ss, ib_ss] <- crossprod(Bp_ss / hp)
  H[ib_ss, k_ss]  <- H[k_ss, ib_ss] <- colSums(B_ss * (0.5 * xs_ss * f))
  H[k_ss, k_ss]   <- sum(0.25 * xs_ss^2 * h0 * f)
  H
}

# Constraint rows, hin(theta) >= 0: the n support rows g_i, then D theta_b.
D_ss <- cbind(diff(diag(p_ss)), 0, 0)
hin_ss <- function(th)
  c(th[1L] * f_ss(th) + xd_ss * th[p_ss + 1L] + off_ss, drop(D_ss %*% th))
hin_jac_ss <- function(th) {
  f <- f_ss(th)
  J <- matrix(0, n_ss, k_ss)
  J[, 1L] <- f
  J[, p_ss + 1L] <- xd_ss
  J[, k_ss] <- 0.5 * th[1L] * f * xs_ss
  rbind(J, D_ss)
}
# Second derivatives of g_i (ADR 0005 Decision 2): d2/dtheta0 dgamma =
# 0.5 f x_s, d2/dgamma2 = 0.25 theta0 f x_s^2, zero elsewhere.
hin_hess_ss <- function(th, i) {
  f <- f_ss(th)[i]
  G <- matrix(0, k_ss, k_ss)
  G[1L, k_ss] <- G[k_ss, 1L] <- 0.5 * f * xs_ss[i]
  G[k_ss, k_ss] <- 0.25 * th[1L] * f * xs_ss[i]^2
  G
}

# The hand-written nll / score match mlt's own at a feasible check point, and
# the Hessian and constraint Jacobian match central differences.
th_chk_ss <- c(1, 1.5, 2, 4, 5, 0.3, 0.4)
num_jac <- function(fn, th, eps = 1e-6) {
  vapply(seq_along(th), function(j) {
    e <- replace(numeric(length(th)), j, eps)
    (fn(th + e) - fn(th - e)) / (2 * eps)
  }, numeric(length(fn(th))))
}
stopifnot(
  all(hin_ss(th_chk_ss) > 0),
  abs(nll_ss(th_chk_ss) + mlt_ss$loglik(th_chk_ss, weights = w_ss)) < 1e-10,
  max(abs(grad_ss(th_chk_ss) +
          colSums(mlt_ss$scorei(th_chk_ss, weights = w_ss)))) < 1e-10,
  max(abs(hess_ss(th_chk_ss) - num_jac(grad_ss, th_chk_ss))) < 1e-5,
  max(abs(hin_jac_ss(th_chk_ss) - num_jac(hin_ss, th_chk_ss))) < 1e-6,
  max(abs(hin_hess_ss(th_chk_ss, 7L) -
          num_jac(function(t) hin_jac_ss(t)[7L, ], th_chk_ss))) < 1e-6
)

# Step 1: multi-start explicit-constraint auglag from feasible starts.
set.seed(2115)
feasible_start_ss <- function(beta, gamma) {
  f <- exp(0.5 * xs_ss * gamma)
  th0 <- max((-off_ss - xd_ss * beta) / f) + runif(1, 0.01, 1)
  c(th0 + cumsum(c(0, rexp(p_ss - 1L))), beta, gamma)
}
starts_ss <- c(
  list(c(max(-off_ss) + 0.1 + seq(0, 1, length.out = p_ss), 0, 0)),
  Map(feasible_start_ss, runif(9, -1, 1), runif(9, -0.5, 0.5))
)
fits_ss <- lapply(starts_ss, function(s) suppressWarnings(alabama::auglag(
  par = s, fn = nll_ss, gr = grad_ss,
  hin = hin_ss, hin.jac = hin_jac_ss,
  control.outer = list(trace = FALSE, eps = 1e-12, itmax = 500L,
                       method = "BFGS", kkt2.check = FALSE),
  control.optim = list(maxit = 5000L, reltol = 1e-15)
)))
conv_ss  <- vapply(fits_ss, `[[`, integer(1), "convergence")
value_ss <- vapply(fits_ss, `[[`, numeric(1), "value")
TH_ss    <- t(vapply(fits_ss, `[[`, numeric(k_ss), "par"))
spread_ss <- max(apply(TH_ss, 2, function(z) diff(range(z))))
act_by_start_ss <- lapply(seq_len(nrow(TH_ss)), function(k)
  which(hin_ss(TH_ss[k, ]) < 1e-6))
stopifnot(all(conv_ss == 0L), diff(range(value_ss)) < 1e-6,
          all(vapply(act_by_start_ss, identical, logical(1),
                     act_by_start_ss[[1L]])))
act_ss <- act_by_start_ss[[1L]]

# Step 2: Newton on the KKT system of the active set, from the best start.
theta_ss <- TH_ss[which.min(value_ss), ]
lam_ss   <- qr.solve(t(hin_jac_ss(theta_ss)[act_ss, , drop = FALSE]),
                     grad_ss(theta_ss))
m_act <- length(act_ss)
for (it in 1:50) {
  J_A <- hin_jac_ss(theta_ss)[act_ss, , drop = FALSE]
  F_kkt <- c(grad_ss(theta_ss) - drop(crossprod(J_A, lam_ss)),
             hin_ss(theta_ss)[act_ss])
  if (max(abs(F_kkt)) < 1e-13) break
  W <- hess_ss(theta_ss)
  for (a in seq_len(m_act)) if (act_ss[a] <= n_ss)
    W <- W - lam_ss[a] * hin_hess_ss(theta_ss, act_ss[a])
  K <- rbind(cbind(W, -t(J_A)), cbind(J_A, matrix(0, m_act, m_act)))
  step <- solve(K, -F_kkt)
  theta_ss <- theta_ss + step[seq_len(k_ss)]
  lam_ss   <- lam_ss + step[k_ss + seq_len(m_act)]
}
names(theta_ss) <- names(coef(ctm_ss))
ll_ss <- mlt_ss$loglik(theta_ss, weights = w_ss)
dpolish_ss <- max(abs(theta_ss - TH_ss[which.min(value_ss), ]))

# Step 3: KKT in the full space.
J_A_ss    <- hin_jac_ss(theta_ss)[act_ss, , drop = FALSE]
stat_ss   <- max(abs(grad_ss(theta_ss) - drop(crossprod(J_A_ss, lam_ss))))
slack_ss  <- hin_ss(theta_ss)
stopifnot(all(lam_ss > 1e-3), stat_ss < 1e-9, all(slack_ss > -1e-12),
          identical(which(slack_ss < 1e-8), act_ss),
          qr(J_A_ss)$rank == m_act,
          nll_ss(theta_ss) <= min(value_ss) + 1e-10,
          dpolish_ss < 1e-4,
          abs(ll_ss + nll_ss(theta_ss)) < 1e-10,
          abs(theta_ss[k_ss]) > 1e-3)

cat(sprintf(paste0(
  "Exponential + shift + scaling + offset: %d/%d auglag starts converged, ",
  "theta spread %.1e, KKT Newton moved theta %.1e; active rows {%s}, ",
  "multipliers {%s}, stationarity %.1e\n"),
  sum(conv_ss == 0L), length(conv_ss), spread_ss, dpolish_ss,
  paste(act_ss, collapse = ", "), paste(sprintf("%.4g", lam_ss), collapse = ", "),
  stat_ss))

writeLines(format(y_ss, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_shift_y.txt"))
writeLines(format(xd_ss, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_shift_xd.txt"))
writeLines(format(xs_ss, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_shift_xs.txt"))
writeLines(format(off_ss, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_shift_offset.txt"))
writeLines(format(sup_ss, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_shift_support.txt"))
writeLines(format(theta_ss, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_shift_theta.txt"))
writeLines(format(ll_ss, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_shift_ll.txt"))
# mltpy's mu_ineq order: monotonicity rows first, then the support rows.
mu_rows_ss <- numeric(n_ss + p_ss - 1L)
mu_rows_ss[act_ss] <- lam_ss
mu_ss <- c(mu_rows_ss[n_ss + seq_len(p_ss - 1L)], mu_rows_ss[seq_len(n_ss)])
writeLines(format(mu_ss, digits = 15),
           con = file.path(out_dir, "mlt_exponential_scaling_shift_mu.txt"))

cat(sprintf("Exponential + shift + scaling + offset: n=%d, ll=%.10f, theta = %s\n",
            n_ss, ll_ss, paste(sprintf("%.8f", theta_ss), collapse = ", ")))

# ---------------------------------------------------------------------------
# exponential + shift + scaling, interval-censored, outside the support
# (issue #164)
#
# Model: todistr = "Exponential", order-4 Bernstein in y on (0, 1.2), one
# shift term x_d (mlt's positive sign) and one scaling term x_s:
#   h(y | x_i) = a(y)'theta_b * exp(0.5 * x_s,i * gamma) + x_d,i * beta.
# The raw coef order c(theta_b (5), sft_x_d, scl_x_s) is mltpy's
# [theta_b | beta | gamma].  Rows are interval-censored at the common
# inspection times 0.25, 0.5, 0.75, 1 (first interval (0, 0.25]; last one
# open, (1, Inf)).  theta is NOT a fit: theta_b[0] < 0 puts h(y | x_i) below
# the exponential support for some rows, so the both-finite rows fall into
# three classes, each of which must occur:
#   * h_l >= 0:          log P = log(pexp(h_u) - pexp(h_l));
#   * h_l < 0 < h_u:     log P = log pexp(h_u), since pexp(h_l) = 0;
#   * h_u <= 0:          log P = -Inf.
# That is R's pexp semantics, F(h) = 0 for h <= 0, which mltpy adopts for
# both-finite interval rows (ADR 0005, #164 amendment): mlt scores the row as
# .log(pexp(h_u) - pexp(h_l)) and its score uses dexp, which is 0 below the
# support.  The open rows (1, Inf) keep h > 0, where mltpy's log S = -h and
# R's pexp(lower.tail = FALSE) agree.  Writes:
#   mlt_exponential_interval_out_lower.txt   — lower interval ends
#   mlt_exponential_interval_out_upper.txt   — upper interval ends (Inf = open)
#   mlt_exponential_interval_out_xd.txt      — shift covariate
#   mlt_exponential_interval_out_xs.txt      — scaling covariate
#   mlt_exponential_interval_out_support.txt — Bernstein support c(0, 1.2)
#   mlt_exponential_interval_out_theta.txt   — c(theta_b (5), beta, gamma)
#   mlt_exponential_interval_out_logliki.txt — per-row loglik (-Inf allowed)
#   mlt_exponential_interval_out_scorei.txt  — per-row score of the rows with
#       a finite loglik, one row per line (n_finite x 7)
# ---------------------------------------------------------------------------

set.seed(164)
n_io    <- 60L
xd_io   <- runif(n_io, 0, 1)
xs_io   <- rnorm(n_io)
grid_io <- c(0.25, 0.5, 0.75, 1)
k_io    <- sample(0:4, n_io, replace = TRUE)
lo_io   <- c(0, grid_io)[k_io + 1L]
up_io   <- c(grid_io, Inf)[k_io + 1L]
sup_io  <- c(0, 1.2)
d_io    <- data.frame(y = Surv(lo_io, up_io, type = "interval2"),
                      x_d = xd_io, x_s = xs_io)

b_io   <- Bernstein_basis(numeric_var("y", support = sup_io,
                                      bounds = c(0, Inf)),
                          order = 4, ui = "increasing")
ctm_io <- ctm(b_io, shifting = ~ x_d, scaling = ~ x_s, data = d_io,
              todistr = "Exponential")
mlt_io <- mlt(ctm_io, data = d_io, dofit = FALSE)
theta_io <- c(-0.8, -0.4, 0.3, 0.9, 1.6, 0.6, 0.4)
names(theta_io) <- names(coef(ctm_io))
stopifnot(identical(names(theta_io),
                    c(paste0("Bs", 1:5, "(y)"), "sft_x_d", "scl_x_s")))

# Classify the rows with h written out (independent of mlt's internals).
h_io <- function(y) {
  B <- model.matrix(b_io, data = data.frame(y = y))
  drop(B %*% theta_io[1:5]) * exp(0.5 * xs_io * theta_io[7]) +
    xd_io * theta_io[6]
}
fin_io <- is.finite(up_io)
hl_io  <- h_io(lo_io)
hu_io  <- h_io(ifelse(fin_io, up_io, sup_io[2]))
cls_in    <- fin_io & hl_io >= 0
cls_strad <- fin_io & hl_io < 0 & hu_io > 0
cls_below <- fin_io & hu_io <= 0
stopifnot(sum(cls_in) >= 3, sum(cls_strad) >= 3, sum(cls_below) >= 3,
          all(hl_io[!fin_io] > 0))

ll_io <- mlt_io$logliki(theta_io, weights = rep(1, n_io))
# Hand-written pexp semantics agree with mlt's own logliki.
ll_hand_io <- ifelse(fin_io,
                     log(pmax(0, pexp(pmax(hu_io, 0)) - pexp(pmax(hl_io, 0)))),
                     pexp(hl_io, lower.tail = FALSE, log.p = TRUE))
stopifnot(isTRUE(all.equal(ll_io, ll_hand_io, tolerance = 1e-13)),
          all(ll_io[cls_below] == -Inf),
          all(is.finite(ll_io[!cls_below])),
          isTRUE(all.equal(ll_io[cls_strad],
                           pexp(hu_io[cls_strad], log.p = TRUE),
                           tolerance = 1e-13)))
# mlt's score floors P at eps^(1/3); keep every finite row well above it so
# the R score is the exact derivative.
stopifnot(min(exp(ll_io[!cls_below])) > 1e-3)
sc_io <- mlt_io$scorei(theta_io, weights = rep(1, n_io))[!cls_below, ,
                                                         drop = FALSE]
stopifnot(all(is.finite(sc_io)), ncol(sc_io) == 7L)

out_io <- function(x, name)
  writeLines(format(x, digits = 15),
             con = file.path(out_dir, paste0("mlt_exponential_interval_out_",
                                             name, ".txt")))
out_io(lo_io, "lower")
out_io(up_io, "upper")
out_io(xd_io, "xd")
out_io(xs_io, "xs")
out_io(sup_io, "support")
out_io(unname(theta_io), "theta")
out_io(ll_io, "logliki")
writeLines(apply(sc_io, 1L, function(r) paste(format(r, digits = 15),
                                              collapse = " ")),
           con = file.path(out_dir, "mlt_exponential_interval_out_scorei.txt"))

cat(sprintf(paste0(
  "Exponential interval outside the support: n=%d (in %d, straddling %d, ",
  "below %d, open %d), finite-row loglik %.10f\n"),
  n_io, sum(cls_in), sum(cls_strad), sum(cls_below), sum(!fin_io),
  sum(ll_io[!cls_below])))

# ---------------------------------------------------------------------------
# Lm (normal linear regression as a CTM) reference
#
# tram::Lm with order=1 Bernstein + normal base is equivalent to classical
# normal linear regression.  We write both the CTM Bernstein coefficients
# (from tram::Lm) and the lm() point estimates so mltpy tests can verify
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
# mltpy tests load theta and data, call hessian()/score_matrix() at that
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
# observed information inverse — that's what we need to match against mltpy.
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
# sign convention differs from mltpy's only for BoxCox (negative=TRUE);
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
# censored obs is the censoring time (Surv$time).  This matches mltpy's
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
#                                          satisfy h - X·beta_R = z; mltpy
#                                          uses h + X·beta so beta_mltpy =
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
  # write row-major to match mltpy.predict_proba.
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
# alabama::auglag and mltpy's PHR solver converge to the same θ to within
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
# (R / tram convention: shift enters with a minus; mltpy parameterises h + x_d * beta,
# so mltpy.coef_ == -coef(fit)["x_d"].  Gamma is sign-aligned across R and mltpy; see
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
# ``[theta_b | beta_tram | gamma]``.  mltpy parametrises ``h + X_d·β`` (R
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

# ---- scaling_fixed_time (ctm shift + scale, fixed-time RIGHT censoring) ---
#
# Issue #162.  The design is the issue's repro, drawn with NumPy's RNG (which
# R cannot replay) by `python reference/write_scaling_fixed_time_data.py`:
# n = 150, x_s ~ N(0, 1), x_d ~ U(0, 1), y exponential with scale depending
# on x_s, every y above its 80 % quantile c censored at c.
#
# Inputs:  scaling_fixed_time_{y,event,x_d,x_s}.txt  (event: 1 = observed)
# Outputs: scaling_fixed_time_support.txt            — "0 1.01·max(y)"
#          scaling_fixed_time_<dist>_{theta,loglik}.txt
#              — the correctly specified fit, Surv(y, event)
#          scaling_fixed_time_flipped_<dist>_{theta,loglik}.txt
#              — the fit with the event flag flipped, Surv(y, !event).  The
#                issue's repro passed `event` as mltpy's *censored* flag, so
#                this is the model it actually fitted: the 30 exact rows tie
#                at y = c and the likelihood is unbounded (h'(c) → ∞ while the
#                censored rows' h → −∞).  R stops at a large positive logLik
#                with convergence 0; its θ puts censored h far beyond mltpy's
#                ±30 clip, so the logLik there checks that the clip is inert.
#          scaling_fixed_time_flipped_<dist>_loglik_at_mltpy.txt
#              — R's logLik at mltpy's θ on that data (input
#                scaling_fixed_time_flipped_<dist>_mltpy_theta.txt, the
#                issue's reported optimum for normal).
# θ is coef(fit) = [θ_b (5) | β | γ]; ctm's shifting term enters as + x_d·β,
# sign-aligned with mltpy (no flip).  <dist> ∈ {normal, logistic}.
.fixed_time_path <- function(stem) {
  file.path(out_dir, sprintf("scaling_fixed_time_%s.txt", stem))
}
y_ft     <- scan(.fixed_time_path("y"), quiet = TRUE)
event_ft <- scan(.fixed_time_path("event"), quiet = TRUE) == 1
x_d_ft   <- scan(.fixed_time_path("x_d"), quiet = TRUE)
x_s_ft   <- scan(.fixed_time_path("x_s"), quiet = TRUE)
stopifnot(length(y_ft) == 150L, sum(event_ft) == 120L)
support_ft <- c(0, 1.01 * max(y_ft))
writeLines(paste(format(support_ft[1], digits = 17),
                 format(support_ft[2], digits = 17)),
           con = .fixed_time_path("support"))
b_ft <- Bernstein_basis(numeric_var("y", support = support_ft,
                                    bounds = c(0, Inf)),
                        order = 4, ui = "increasing")
for (dist_ft in c("Normal", "Logistic")) {
  ctm_ft <- ctm(b_ft, shifting = ~ x_d, scaling = ~ x_s,
                data = data.frame(y = y_ft, x_d = x_d_ft, x_s = x_s_ft),
                todistr = dist_ft)
  for (flipped_ft in c(FALSE, TRUE)) {
    ev_ft <- if (flipped_ft) !event_ft else event_ft
    fit_ft <- mlt(ctm_ft, data = data.frame(y = Surv(y_ft, ev_ft),
                                            x_d = x_d_ft, x_s = x_s_ft))
    stopifnot(fit_ft$convergence == 0)
    stem_ft <- paste0(if (flipped_ft) "flipped_" else "", tolower(dist_ft))
    ll_ft <- as.numeric(logLik(fit_ft))
    writeLines(format(coef(fit_ft), digits = 17),
               con = .fixed_time_path(paste0(stem_ft, "_theta")))
    writeLines(format(ll_ft, digits = 17),
               con = .fixed_time_path(paste0(stem_ft, "_loglik")))
    cat(sprintf("scaling fixed-time right-censored ref %-17s ll=%.6f\n",
                stem_ft, ll_ft))
    if (flipped_ft) {
      # R's logLik at mltpy's θ on the issue's literal repro (written by the
      # Python script above); there |h| of censored rows reaches ~1e10.
      theta_py_ft <- scan(.fixed_time_path(paste0(stem_ft, "_mltpy_theta")),
                          quiet = TRUE)
      ll_py_ft <- as.numeric(logLik(fit_ft, parm = theta_py_ft))
      writeLines(format(ll_py_ft, digits = 17),
                 con = .fixed_time_path(paste0(stem_ft, "_loglik_at_mltpy")))
      cat(sprintf("  R logLik at mltpy theta (%s): %.6f\n", stem_ft, ll_py_ft))
    }
  }
}

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
# (so ``h − X_d·β``), so the mltpy β block is sign-flipped at compare time.
# tram::Survreg fits a transformation on log-time (LogBernsteinBasis in
# mltpy); we cover all three distributions (``weibull``, ``lognormal``,
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
# cover all three to match the convenience surface in mltpy.tram.Survreg.
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
  # ``LogBernsteinBasis(order=1)`` on the mltpy side is affine in
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
# * BoxCox uses ``negative = TRUE`` so β_R = −β_mltpy.  Both rows and
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
# numerically well-conditioned and matches mltpy's ``inv(H)`` element-wise.
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

# ---------------------------------------------------------------------------
# confband reference — *scaling* BoxCox fit (heteroskedastic baseline).
#
# Reuses the interior seed-770 BoxCox(y ~ x_d | x_s) fit above (so the θ_b
# block of vcov is well-conditioned and matches mltpy's ``inv(H)``; a
# boundary fit would need R's active-set vcov penalty and would not match a
# bare delta-method band).  Builds the pointwise Wald band by the delta
# method on the *full* ``[θ_b | β | γ]`` parameter vector, exercising the γ
# columns of the Jacobian:
#
#   h(y|x_d,x_s) = h_0(y)·s + x_d·β,   s = exp(0.5·x_s·γ),   h' = h_0'·s
#
# Sign convention: ``coef(as.mlt(fit))`` and ``vcov(as.mlt(fit))`` are in
# R's ``h − x_d·β_R`` parameterisation, so the β column of the Jacobian is
# ``∂h/∂β_R = −x_d`` and the back-transformed curve uses ``h = h_0·s −
# x_d·β_R`` — this matches mltpy's ``h + x_d·β`` curve because mltpy stores
# β = −β_R (ADR 0002, Decision 5).  The resulting band is parameterisation-
# invariant, so the mltpy parity test compares directly without any flip.
#
# Writes (m = grid length):
#   reference/confband_scaling_y_grid.txt  — m-point grid
#   reference/confband_scaling_profile.txt — "x_d0 x_s0" (single profile)
#   reference/confband_scaling_<what>.txt  — (m, 3) row-major [est, lwr, upr]
# ---------------------------------------------------------------------------
m_cbs   <- as.mlt(fit_v_bc)
cf_cbs  <- coef(m_cbs)
V_cbs   <- as.matrix(vcov(m_cbs))
p_cbs   <- 5L
nm_cbs  <- names(cf_cbs)
theta_b_cbs <- cf_cbs[1:p_cbs]
beta_r_cbs  <- cf_cbs[grep("x_d", nm_cbs)]
gamma_r_cbs <- cf_cbs[grep("scl", nm_cbs)]

# Single covariate / scaling profile and an interior grid (avoid endpoints
# where h' -> 0 on the Bernstein basis).
xd0_cbs <- 0.6
xs0_cbs <- 0.7
yg_cbs  <- seq(a_v_bc + 0.3, b_v_bc - 0.3, length.out = 15)

bb_cbs <- Bernstein_basis(numeric_var("y", support = c(a_v_bc, b_v_bc)),
                          order = 4, ui = "increasing")
B_cbs  <- model.matrix(bb_cbs, data = data.frame(y = yg_cbs))
D_cbs  <- model.matrix(bb_cbs, data = data.frame(y = yg_cbs), deriv = c(y = 1L))

s_cbs   <- as.numeric(exp(0.5 * xs0_cbs * gamma_r_cbs))
h0_cbs  <- as.numeric(B_cbs %*% theta_b_cbs)
hp0_cbs <- as.numeric(D_cbs %*% theta_b_cbs)
h_cbs   <- h0_cbs * s_cbs - xd0_cbs * beta_r_cbs   # = mltpy's h_0·s + x_d·β
hp_cbs  <- hp0_cbs * s_cbs
qn_cbs  <- qnorm(0.975)

# Jacobian in [θ_b | β_R | γ] order (matching V_cbs).  `with_loghp` adds the
# ∂(log h')/∂θ contribution present in the density / hazard linear predictors.
.cbs_jac <- function(coeff, with_loghp) {
  J <- matrix(0, length(yg_cbs), p_cbs + 2L)
  J[, 1:p_cbs] <- if (with_loghp) coeff * (s_cbs * B_cbs) + D_cbs / hp0_cbs
                  else coeff * (s_cbs * B_cbs)
  J[, p_cbs + 1L] <- coeff * (-xd0_cbs)                       # ∂h/∂β_R = -x_d
  J[, p_cbs + 2L] <- if (with_loghp) 0.5 * xs0_cbs * (coeff * h0_cbs * s_cbs + 1)
                     else coeff * h0_cbs * s_cbs * 0.5 * xs0_cbs
  J
}
.cbs_var <- function(J) rowSums((J %*% V_cbs) * J)

# trafo / distribution / survivor (η = h; coeff = 1, no log h' term).
se_h_cbs <- sqrt(.cbs_var(.cbs_jac(1, FALSE)))
lo_cbs   <- h_cbs - qn_cbs * se_h_cbs
hi_cbs   <- h_cbs + qn_cbs * se_h_cbs
.write_band(h_cbs, lo_cbs, hi_cbs, "confband_scaling_trafo.txt")
.write_band(pnorm(h_cbs), pnorm(lo_cbs), pnorm(hi_cbs),
            "confband_scaling_distribution.txt")
.write_band(1 - pnorm(h_cbs), 1 - pnorm(hi_cbs), 1 - pnorm(lo_cbs),
            "confband_scaling_survivor.txt")

# density (η = log f(h) + log h'; coeff = ψ(h) = -h for N(0,1)).
psi_cbs   <- -h_cbs
se_d_cbs  <- sqrt(.cbs_var(.cbs_jac(psi_cbs, TRUE)))
eta_d_cbs <- dnorm(h_cbs, log = TRUE) + log(hp_cbs)
.write_band(exp(eta_d_cbs), exp(eta_d_cbs - qn_cbs * se_d_cbs),
            exp(eta_d_cbs + qn_cbs * se_d_cbs), "confband_scaling_density.txt")

# hazard (η = log f(h) + log h' - log S(h); coeff = ψ(h) + λ(h)).
lam_cbs   <- dnorm(h_cbs) / pnorm(h_cbs, lower.tail = FALSE)
se_z_cbs  <- sqrt(.cbs_var(.cbs_jac(psi_cbs + lam_cbs, TRUE)))
eta_z_cbs <- dnorm(h_cbs, log = TRUE) + log(hp_cbs) -
             pnorm(h_cbs, lower.tail = FALSE, log.p = TRUE)
.write_band(exp(eta_z_cbs), exp(eta_z_cbs - qn_cbs * se_z_cbs),
            exp(eta_z_cbs + qn_cbs * se_z_cbs), "confband_scaling_hazard.txt")

writeLines(format(yg_cbs, digits = 15),
           con = file.path(out_dir, "confband_scaling_y_grid.txt"))
writeLines(paste(format(xd0_cbs, digits = 15), format(xs0_cbs, digits = 15)),
           con = file.path(out_dir, "confband_scaling_profile.txt"))
cat(sprintf("confband scaling refs: p=%d, m=%d, profile=(%.2f, %.2f)\n",
            p_cbs, length(yg_cbs), xd0_cbs, xs0_cbs))

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
# active-constraint penalty and does not match mltpy's bare ``inv(H)``.
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

# ---------------------------------------------------------------------------
# Scaling-terms vignette (issue #78) — heteroskedastic-regression LR test.
#
# The vignette in ``docs/examples/05_scaling_terms.ipynb`` reuses the
# ``scaling_boxcox_normal_*`` dataset (n=100, seed=70, BoxCox normal) and
# fits the constant-variance baseline ``tram::BoxCox(y ~ x_d)`` against the
# heteroskedastic ``tram::BoxCox(y ~ x_d | x_s)`` on the *same* data and
# *same* support / basis order.  The likelihood-ratio statistic on γ is
# what the notebook reproduces with mltpy; this fixture pins it.
#
# Reference is the LR test on γ from ``tram::BoxCox(..., scale=~x_s)`` vs.
# ``tram::BoxCox(...)`` (the issue spec asks for ``tram::Lm`` parity, but
# the vignette example uses ``BoxCox`` — the conceptual LR check matches
# either model, and BoxCox lets the notebook reuse the existing
# ``scaling_boxcox_normal_*`` dataset without a second fit-data file).
#
# Fixture: "scaling_vignette_boxcox_lr.txt" — single line "chi2 df pvalue".
# ---------------------------------------------------------------------------
fit_bc_null_vig <- tram::BoxCox(y ~ x_d, data = df_sc,
                                support = c(a_sc, b_sc), order = 5)
ll_bc_null_vig <- as.numeric(logLik(fit_bc_null_vig))
df_bc_null_vig <- attr(logLik(fit_bc_null_vig), "df")
df_bc_full_vig <- attr(logLik(fit_sc),            "df")
chisq_bc_vig   <- 2 * (ll_sc - ll_bc_null_vig)
df_lr_vig      <- df_bc_full_vig - df_bc_null_vig
pval_lr_vig    <- pchisq(chisq_bc_vig, df = df_lr_vig, lower.tail = FALSE)
writeLines(sprintf("%.15g %d %.15g",
                   chisq_bc_vig, df_lr_vig, pval_lr_vig),
           con = file.path(out_dir, "scaling_vignette_boxcox_lr.txt"))
cat(sprintf("scaling vignette BoxCox LR: chi^2=%.4f df=%d p=%.4g\n",
            chisq_bc_vig, df_lr_vig, pval_lr_vig))

# ---------------------------------------------------------------------------
# Profile-likelihood confidence intervals (issue #87, parent #60).
#
# Reuses the top-of-file no-covariate fit (`fit`, order-4 Bernstein on
# uniform(0.02, 0.98)) — same object that backs the `confband_baseline_*`
# fixtures.  R `mlt` does not provide a public `confint(type="profile")`
# entry point, so we invert the χ²_1 likelihood-ratio test by hand:
#   for each parameter j, find the two roots v ∈ ℝ of
#       f(v) = 2·(ℓ̂ − ℓ_p(v)) − χ²_{1, 1-α},
#   where ℓ_p(v) is the profile log-likelihood with θ_j pinned to v and
#   the remaining parameters re-optimised under the monotonicity
#   constraints.  Refit is via `mlt(..., fixed = c("Bs(y)5" = v))`-style
#   parameter pinning through `update(fit, fixed = ...)` — `mlt`'s public
#   `fixed=` argument injects equality constraints into the auglag /
#   alabama path used internally.
#
# Reference layout — `profile_ci_baseline.txt`: `(p, 2)` flattened
# row-major with columns `[lower, upper]`, matching the existing
# `confint_<model>.txt` files.
# ---------------------------------------------------------------------------

profile_ci <- function(fit, level = 0.95) {
  theta_hat <- coef(as.mlt(fit), with_baseline = TRUE)
  p         <- length(theta_hat)
  V         <- vcov(as.mlt(fit))
  se        <- sqrt(diag(V))
  ll_hat    <- as.numeric(logLik(fit))
  crit      <- qchisq(level, df = 1)

  # `mlt::mlt` accepts a named `fixed = c(name = value)` argument that
  # pins one coefficient to `value` and re-runs the constrained MLE.
  # The names come from `names(theta_hat)`.
  param_names <- names(theta_hat)

  # Profile log-lik at θ_j = v: refit with that one coordinate pinned.
  profile_ll <- function(j, v) {
    pin           <- setNames(v, param_names[j])
    fit_pin       <- mlt(ctm, data = data.frame(y = y), fixed = pin)
    as.numeric(logLik(fit_pin))
  }

  # f(v) = 2·(ll_hat − ll_p(v)) − crit.  Root-find on each side of θ̂_j.
  f_one <- function(j, v) 2 * (ll_hat - profile_ll(j, v)) - crit

  ci <- matrix(NA_real_, nrow = p, ncol = 2,
               dimnames = list(param_names, c("lower", "upper")))

  for (j in seq_len(p)) {
    # Adaptive bracket: start at θ̂_j ± 3·se_j, double on no sign change.
    th <- theta_hat[j]
    s  <- se[j]
    # Lower root: bracket [lo, th].
    for (k in 1:24) {
      lo <- th - (3 * 2^(k - 1)) * s
      if (f_one(j, lo) > 0) break
    }
    ci[j, 1] <- uniroot(function(v) f_one(j, v),
                        interval = c(lo, th),
                        tol = 1e-6)$root
    # Upper root: bracket [th, hi].
    for (k in 1:24) {
      hi <- th + (3 * 2^(k - 1)) * s
      if (f_one(j, hi) > 0) break
    }
    ci[j, 2] <- uniroot(function(v) f_one(j, v),
                        interval = c(th, hi),
                        tol = 1e-6)$root
  }
  ci
}

profile_ci_baseline <- profile_ci(fit, level = 0.95)

writeLines(format(flatten_row_major(profile_ci_baseline), digits = 15),
           con = file.path(out_dir, "profile_ci_baseline.txt"))

cat(sprintf("profile_ci_baseline: p=%d rows written\n",
            nrow(profile_ci_baseline)))

# ---------------------------------------------------------------------------
# Profile-CI references for the three tram convenience fits (issue #88).
#
# Same χ²_1 LR inversion as the baseline above, but generic over any tram /
# mlt fit object: we refit via `update(fit, fixed = pin)` instead of
# re-creating the `mlt(ctm, data, fixed=pin)` call by hand.  This lets the
# helper handle BoxCox (negative=TRUE — β sign-flipped vs mltpy), Colr, and
# Coxph (right-censored) uniformly.  The Python side flips BoxCox β rows
# before comparing — see `_apply_mltpy_sign` in tests/test_confidence.py.
#
# Output layout — `profile_ci_<model>.txt`: `(k, 2)` flattened row-major
# with columns `[lower, upper]`, identical contract to the Wald fixtures
# `confint_<model>.txt`.
# ---------------------------------------------------------------------------

.profile_ci_update <- function(fit, level = 0.95) {
  m           <- as.mlt(fit)
  theta_hat   <- coef(m, with_baseline = TRUE)
  p           <- length(theta_hat)
  V           <- vcov(m)
  se          <- sqrt(diag(V))
  ll_hat      <- as.numeric(logLik(fit))
  crit        <- qchisq(level, df = 1)
  param_names <- names(theta_hat)

  profile_ll <- function(j, v) {
    pin     <- setNames(v, param_names[j])
    fit_pin <- suppressWarnings(update(fit, fixed = pin))
    as.numeric(logLik(fit_pin))
  }
  f_one <- function(j, v) 2 * (ll_hat - profile_ll(j, v)) - crit

  ci <- matrix(NA_real_, nrow = p, ncol = 2,
               dimnames = list(param_names, c("lower", "upper")))

  for (j in seq_len(p)) {
    th <- theta_hat[j]
    s  <- se[j]
    # Lower root: bracket [lo, th].
    for (k in 1:24) {
      lo <- th - (3 * 2^(k - 1)) * s
      if (f_one(j, lo) > 0) break
    }
    ci[j, 1] <- uniroot(function(v) f_one(j, v),
                        interval = c(lo, th),
                        tol = 1e-6)$root
    # Upper root: bracket [th, hi].
    for (k in 1:24) {
      hi <- th + (3 * 2^(k - 1)) * s
      if (f_one(j, hi) > 0) break
    }
    ci[j, 2] <- uniroot(function(v) f_one(j, v),
                        interval = c(th, hi),
                        tol = 1e-6)$root
  }
  ci
}

.write_profile_ci_ref <- function(fit, filename, level = 0.95) {
  ci <- .profile_ci_update(fit, level = level)
  writeLines(format(flatten_row_major(ci), digits = 15),
             con = file.path(out_dir, filename))
  cat(sprintf("%s: k=%d rows written\n", filename, nrow(ci)))
  invisible(ci)
}

.write_profile_ci_ref(fit_bc,   "profile_ci_boxcox.txt")
.write_profile_ci_ref(fit_colr, "profile_ci_colr.txt")
.write_profile_ci_ref(fit_cx,   "profile_ci_coxph.txt")

# ---------------------------------------------------------------------------
# Scaling + InteractionBasis (stram) — issue #103, ADR 0003.
#
# The non-proportional, heteroskedastic CTM: an interacting Bernstein-y ×
# Bernstein-x baseline scaled by exp(0.5 · x_s · γ), exact data, Normal base.
# This is the mltpy analogue of R's no-shift shift-scale model
# ``y | s ~ 1 | z`` (class ``stram``); R `tram` flags the combination "highly
# experimental" (see docs/adr/0003-scaling-with-interaction.md).
#
# We fit it directly through ``mlt::ctm`` rather than the ``tram`` formula
# interface — exactly as the ``interaction_bs_bs_*`` fixtures above do — so the
# response (Bernstein-y) and interacting (Bernstein-x) bases match mltpy's
# ``InteractionBasis`` term-for-term.  The scale term is a one-column linear
# basis in ``x_s`` (intercept removed), giving s(x) = x_s and the documented
# √(exp(s·γ)) = exp(0.5·x_s·γ) factor (mlt's default ``scale_shift = FALSE``).
#
# mlt's coefficient vector orders [ baseline (response × interacting) |
# scaling ]; with no shift block the trailing q_s entries are γ.  γ is sign-
# AND magnitude-aligned with mltpy (no flip — ADR 0003 Decision 6 / ADR 0002
# Decision 5), so the Python ratchet compares ``model.gamma_`` to
# ``stram_*_gamma.txt`` directly.
#
# Fixtures emitted (prefix ``stram_bs_bs_normal_``):
#   *_y_train.txt    — response, length n
#   *_x_train.txt    — interacting covariate, length n
#   *_x_s_train.txt  — scaling covariate, length n
#   *_y_support.txt  — "a b" on one line
#   *_theta.txt      — full coef(fit): [vec(Θ) (mlt order) | γ], length p*q+q_s
#   *_gamma.txt      — scale coefficients only, length q_s
#   *_gamma_vcov.txt — γ block of vcov(fit), (q_s, q_s) flattened row-major
#                      (issue #105: Wald SEs for γ on the interaction path)
#   *_loglik.txt     — scalar logLik
#
# Active constraint and precision (issue #125): unlike the sibling stram blocks
# this MLE is NOT interior — the middle Θ column (x = 0.5) sits on the
# monotonicity boundary, Θ[1, 2] = Θ[2, 2].  The default alabama fit stops at
# abstol = 1e-7, which leaves γ ~5e-5 (relative) short of the optimum and the
# boundary row infeasible by ~1e-8.  The plain-BFGS polish the sibling blocks
# use is unconstrained and so not valid here; instead the fit is polished with
# mlt's ``spg`` optimiser (BB::BBoptim with an exact linear projection onto
# ``ui θ ≥ ci``), which lands on the boundary exactly.  Starting the same
# polish from a tighter-auglag point reproduces γ to ~2e-9 (relative).
#
# Requires mlt >= 1.7 (``scaling`` / ``scale_shift`` in ``ctm``).  Fixtures
# materialised with R 4.6.1, mlt 1.8.4, basefun 1.2.6, variables 1.1.2,
# BB 2026.1.0, alabama 2025.1.0 (tram 1.4.7 installed; not used by this block).
# ---------------------------------------------------------------------------
set.seed(20260525)
n_st <- 300L
p_st <- 3L  # number of Bernstein-y functions
q_st <- 3L  # number of Bernstein-x functions
x_st   <- runif(n_st, 0, 1)
x_s_st <- rnorm(n_st)
# Heteroskedastic conditional response: location shifts mildly with x, spread
# grows with x_s.  The MLE has one active monotonicity row (see above).
y_st <- 0.5 + 1.0 * x_st + rnorm(n_st, sd = exp(0.5 * 0.3 * x_s_st)) * 0.6
a_st <- min(y_st) - 0.2
b_st <- max(y_st) + 0.2

df_st <- data.frame(y = y_st, x = x_st, x_s = x_s_st)
m_y_st <- numeric_var("y", support = c(a_st, b_st))
m_x_st <- numeric_var("x", support = c(0, 1))
b_y_st <- Bernstein_basis(m_y_st, order = p_st - 1L, ui = "increasing")
b_x_st <- Bernstein_basis(m_x_st, order = q_st - 1L)
# One-column linear scaling basis in x_s (no intercept) → s(x) = x_s.
b_s_st <- as.basis(~ x_s, data = df_st, remove_intercept = TRUE)

ctm_st <- ctm(response = b_y_st,
              interacting = b_x_st,
              scaling = b_s_st,
              scale_shift = FALSE,
              todistr = "Normal")
fit0_st <- mlt(ctm_st, data = df_st)
fit_st  <- mlt(ctm_st, data = df_st, theta = coef(fit0_st),
               optim = mltoptim(spg = list(ftol = 1e-15, gtol = 1e-10,
                                           maxit = 100000L, quiet = TRUE,
                                           checkGrad = FALSE))["spg"])
# Converged, feasible (every Θ column non-decreasing), and on the boundary.
Theta_st <- matrix(coef(fit_st)[1:(p_st * q_st)], p_st, q_st)
stopifnot(fit_st$convergence == 0,
          min(diff(Theta_st)) >= 0,
          min(diff(Theta_st)) < 1e-8,
          as.numeric(logLik(fit_st)) > as.numeric(logLik(fit0_st)))

theta_st <- coef(fit_st)
ll_st    <- as.numeric(logLik(fit_st))
# Scaling coefficients are the trailing q_s entries (no shift block); also
# identifiable by the scaling-basis column name(s).
scl_names_st <- colnames(model.matrix(b_s_st, data = df_st))
gamma_st <- theta_st[scl_names_st]
# γγ block of the inverse observed information.  vcov.mlt is solve(Hessian)
# and ignores the active monotonicity row — the same quantity mltpy's default
# vcov() returns as inv(H) from the finite-difference interaction Hessian.
gamma_vcov_st <- vcov(fit_st)[scl_names_st, scl_names_st, drop = FALSE]

writeLines(format(y_st, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_normal_y_train.txt"))
writeLines(format(x_st, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_normal_x_train.txt"))
writeLines(format(x_s_st, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_normal_x_s_train.txt"))
writeLines(paste(format(a_st, digits = 15), format(b_st, digits = 15)),
           con = file.path(out_dir, "stram_bs_bs_normal_y_support.txt"))
writeLines(format(theta_st, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_normal_theta.txt"))
writeLines(format(as.numeric(gamma_st), digits = 15),
           con = file.path(out_dir, "stram_bs_bs_normal_gamma.txt"))
writeLines(format(flatten_row_major(gamma_vcov_st), digits = 15),
           con = file.path(out_dir, "stram_bs_bs_normal_gamma_vcov.txt"))
writeLines(format(ll_st, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_normal_loglik.txt"))

cat(sprintf("stram_bs_bs normal ref: n=%d, p=%d, q=%d, q_s=%d, ll=%.6f\n",
            n_st, p_st, q_st, length(gamma_st), ll_st))

# ---------------------------------------------------------------------------
# Coxph(interacting=..., scaling=...) — issue #106.
#
# The survival analogue of the stram block above: same Bernstein-y ×
# Bernstein-x interacting baseline scaled by exp(0.5 · x_s · γ), but with the
# minimum extreme value link (``todistr = "MinExtrVal"``) that mltpy's Coxph
# forces, so log[-log S(t | x, x_s)] = h(t | x) · exp(0.5 · x_s · γ).  Exact
# (uncensored) event times only — the interaction likelihood path is
# exact-data only.  Fitted through ``mlt::ctm`` for term-for-term basis
# agreement with mltpy's ``InteractionBasis``; γ is sign- and magnitude-aligned
# with mltpy (no flip).
#
# Data: log[-log S(t)] = exp(0.5·γ·x_s)·(2·(t − 4.5) + 0.8·(x − 0.5)) with
# γ = 0.4 — a non-proportional-hazards model whose baseline is linear in t and
# in x, so it lies exactly in the Bernstein-y (order 2) ⊗ Bernstein-x (order 1)
# span.  With n = 1000 the MLE is strictly interior (every Θ column has
# increments ≈ 3 or more): no active monotonicity constraint leaves a flat
# ridge, so γ is determined well below the 1e-6 parity tolerance.
#
# Precision: mlt's default auglag stops on alabama's inner BFGS tolerance
# (reltol ≈ 1.5e-8 on |ℓ| ≈ 10³), which leaves γ about 1e-6 (relative) short
# of the optimum.  Because the MLE is interior, the fit is polished from that
# solution with plain BFGS at reltol = 1e-15; ``checkconstraints = TRUE``
# confirms the polished point is still feasible.  A second polish moves θ by
# ~1e-15.
#
# Fixtures emitted (prefix ``stram_bs_bs_mev_``):
#   *_y_train.txt    — event times, length n
#   *_x_train.txt    — interacting covariate, length n
#   *_x_s_train.txt  — scaling covariate, length n
#   *_y_support.txt  — "a b" on one line
#   *_theta.txt      — full coef(fit): [vec(Θ) (mlt order) | γ]
#   *_gamma.txt      — scale coefficients only, length q_s
#   *_loglik.txt     — scalar logLik
#   *_grid.txt       — prediction grid, one "t x x_s" row per line
#   *_survivor.txt   — S(t | x, x_s) at each grid row
#   *_hazard.txt     — hazard f/S at each grid row
# ---------------------------------------------------------------------------
set.seed(20261031)
n_mv <- 1000L
p_mv <- 3L  # number of Bernstein-y functions
q_mv <- 2L  # number of Bernstein-x functions
x_mv   <- runif(n_mv, 0, 1)
x_s_mv <- rnorm(n_mv)
# Invert S(t) = exp(-exp(k · h0(t | x))) at S = U for t.
k_mv <- exp(0.5 * 0.4 * x_s_mv)
y_mv <- 4.5 - 0.4 * (x_mv - 0.5) + log(-log(runif(n_mv))) / (2 * k_mv)
a_mv <- min(y_mv) - 0.1
b_mv <- max(y_mv) + 0.1

df_mv <- data.frame(y = y_mv, x = x_mv, x_s = x_s_mv)
m_y_mv <- numeric_var("y", support = c(a_mv, b_mv))
m_x_mv <- numeric_var("x", support = c(0, 1))
b_y_mv <- Bernstein_basis(m_y_mv, order = p_mv - 1L, ui = "increasing")
b_x_mv <- Bernstein_basis(m_x_mv, order = q_mv - 1L)
b_s_mv <- as.basis(~ x_s, data = df_mv, remove_intercept = TRUE)

ctm_mv <- ctm(response = b_y_mv,
              interacting = b_x_mv,
              scaling = b_s_mv,
              scale_shift = FALSE,
              todistr = "MinExtrVal")
fit0_mv <- mlt(ctm_mv, data = df_mv)
fit_mv  <- mlt(ctm_mv, data = df_mv, theta = coef(fit0_mv),
               optim = mltoptim(optim = list(checkconstraints = TRUE,
                                             method = "BFGS",
                                             maxit = 100000L,
                                             reltol = 1e-15))["optim"])
stopifnot(fit_mv$convergence == 0,
          min(diff(matrix(coef(fit_mv)[1:(p_mv * q_mv)], p_mv, q_mv))) > 1)

theta_mv <- coef(fit_mv)
ll_mv    <- as.numeric(logLik(fit_mv))
scl_names_mv <- colnames(model.matrix(b_s_mv, data = df_mv))
gamma_mv <- theta_mv[scl_names_mv]

# Per-row predictions at (t, x, x_s).  ``predict()`` on a ctm with a
# ``scaling`` term errors inside mlt (predict.cbind_bases, observed with
# mlt 1.8.4), so h is assembled from the tensor basis directly:
# h = Z·vec(Θ) · exp(0.5·x_s·γ), h' = Z'·vec(Θ) · exp(0.5·x_s·γ), with the
# first p·q coefficients being vec(Θ) in the box-basis column order.  The
# stopifnot() below checks this reconstruction reproduces logLik(fit).
# Minimum extreme value: S = exp(−exp(h)), hazard = f/S = exp(h)·h'.
bb_mv <- b(bresponse = b_y_mv, binteracting = b_x_mv)
pq_mv <- p_mv * q_mv
.h_mv <- function(d) {
  f <- exp(0.5 * d$x_s * gamma_mv)
  list(h  = drop(model.matrix(bb_mv, data = d) %*% theta_mv[1:pq_mv]) * f,
       hp = drop(model.matrix(bb_mv, data = d, deriv = c(y = 1)) %*%
                   theta_mv[1:pq_mv]) * f)
}
tr_mv <- .h_mv(df_mv)
stopifnot(isTRUE(all.equal(sum(tr_mv$h - exp(tr_mv$h) + log(tr_mv$hp)), ll_mv,
                           tolerance = 1e-10)))

grid_mv <- expand.grid(y = quantile(y_mv, c(0.1, 0.3, 0.5, 0.7, 0.9)),
                       x = c(0.2, 0.8),
                       x_s = c(-1, 0, 1.5))
hg_mv   <- .h_mv(grid_mv)
surv_mv <- exp(-exp(hg_mv$h))
haz_mv  <- exp(hg_mv$h) * hg_mv$hp

writeLines(format(y_mv, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_mev_y_train.txt"))
writeLines(format(x_mv, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_mev_x_train.txt"))
writeLines(format(x_s_mv, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_mev_x_s_train.txt"))
writeLines(paste(format(a_mv, digits = 15), format(b_mv, digits = 15)),
           con = file.path(out_dir, "stram_bs_bs_mev_y_support.txt"))
writeLines(format(theta_mv, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_mev_theta.txt"))
writeLines(format(as.numeric(gamma_mv), digits = 15),
           con = file.path(out_dir, "stram_bs_bs_mev_gamma.txt"))
writeLines(format(ll_mv, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_mev_loglik.txt"))
write.table(format(as.matrix(grid_mv), digits = 15),
            file = file.path(out_dir, "stram_bs_bs_mev_grid.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
writeLines(format(as.numeric(surv_mv), digits = 15),
           con = file.path(out_dir, "stram_bs_bs_mev_survivor.txt"))
writeLines(format(as.numeric(haz_mv), digits = 15),
           con = file.path(out_dir, "stram_bs_bs_mev_hazard.txt"))

cat(sprintf("stram_bs_bs mev ref: n=%d, p=%d, q=%d, q_s=%d, ll=%.6f\n",
            n_mv, p_mv, q_mv, length(gamma_mv), ll_mv))

# ---------------------------------------------------------------------------
# Colr(interacting=..., scaling=...) — issue #108.
#
# The continuous-outcome logistic analogue of the stram_bs_bs_mev block above:
# the same Bernstein-y × Bernstein-x interacting baseline scaled by
# exp(0.5 · x_s · γ), but with the standard logistic link
# (``todistr = "Logistic"``) that mltpy's Colr forces, so
# logit F(y | x, x_s) = h(y | x) · exp(0.5 · x_s · γ) — a non-proportional-odds
# CTM.  Exact (uncensored) responses only — the interaction likelihood path is
# exact-data only.  Fitted through ``mlt::ctm`` for term-for-term basis
# agreement with mltpy's ``InteractionBasis``; γ is sign- and magnitude-aligned
# with mltpy (no flip).
#
# Data: logit F(y) = exp(0.5·γ·x_s)·(2·(y − 4.5) + 0.8·(x − 0.5)) with
# γ = 0.4, whose baseline is linear in y and in x and so lies exactly in the
# Bernstein-y (order 2) ⊗ Bernstein-x (order 1) span.  With n = 1000 the MLE
# is strictly interior (every Θ column increment is well above 1), so γ is
# determined well below the 1e-6 parity tolerance.  The same BFGS polish at
# reltol = 1e-15 as the mev block takes γ past alabama's inner tolerance.
#
# Fixtures emitted (prefix ``stram_bs_bs_logistic_``):
#   *_y_train.txt      — responses, length n
#   *_x_train.txt      — interacting covariate, length n
#   *_x_s_train.txt    — scaling covariate, length n
#   *_y_support.txt    — "a b" on one line
#   *_theta.txt        — full coef(fit): [vec(Θ) (mlt order) | γ]
#   *_gamma.txt        — scale coefficients only, length q_s
#   *_gamma_tram.txt   — γ from the tram::Colr(y | x ~ 1 | x_s) stram fit
#   *_loglik.txt       — scalar logLik
#   *_grid.txt         — prediction grid, one "y x x_s" row per line
#   *_distribution.txt — F(y | x, x_s) at each grid row
#   *_density.txt      — f(y | x, x_s) = dlogis(h)·h' at each grid row
# ---------------------------------------------------------------------------
set.seed(20261108)
n_lg <- 1000L
p_lg <- 3L  # number of Bernstein-y functions
q_lg <- 2L  # number of Bernstein-x functions
x_lg   <- runif(n_lg, 0, 1)
x_s_lg <- rnorm(n_lg)
# Invert F(y) = plogis(k · h0(y | x)) at F = U for y.
k_lg <- exp(0.5 * 0.4 * x_s_lg)
y_lg <- 4.5 - 0.4 * (x_lg - 0.5) + qlogis(runif(n_lg)) / (2 * k_lg)
a_lg <- min(y_lg) - 0.1
b_lg <- max(y_lg) + 0.1

df_lg <- data.frame(y = y_lg, x = x_lg, x_s = x_s_lg)
m_y_lg <- numeric_var("y", support = c(a_lg, b_lg))
m_x_lg <- numeric_var("x", support = c(0, 1))
b_y_lg <- Bernstein_basis(m_y_lg, order = p_lg - 1L, ui = "increasing")
b_x_lg <- Bernstein_basis(m_x_lg, order = q_lg - 1L)
b_s_lg <- as.basis(~ x_s, data = df_lg, remove_intercept = TRUE)

ctm_lg <- ctm(response = b_y_lg,
              interacting = b_x_lg,
              scaling = b_s_lg,
              scale_shift = FALSE,
              todistr = "Logistic")
fit0_lg <- mlt(ctm_lg, data = df_lg)
fit_lg  <- mlt(ctm_lg, data = df_lg, theta = coef(fit0_lg),
               optim = mltoptim(optim = list(checkconstraints = TRUE,
                                             method = "BFGS",
                                             maxit = 100000L,
                                             reltol = 1e-15))["optim"])
stopifnot(fit_lg$convergence == 0,
          min(diff(matrix(coef(fit_lg)[1:(p_lg * q_lg)], p_lg, q_lg))) > 1)

theta_lg <- coef(fit_lg)
ll_lg    <- as.numeric(logLik(fit_lg))
scl_names_lg <- colnames(model.matrix(b_s_lg, data = df_lg))
gamma_lg <- theta_lg[scl_names_lg]

# Cross-check against the literal tram formula route, ``tram::Colr(y | x ~ 1
# | x_s)`` (class ``stram``; tram warns that strata + scale is "highly
# experimental").  Its interacting term is the linear model matrix (1, x),
# which spans the same space as Bernstein-x of order 1 on [0, 1], so the two
# fits share the MLE: same logLik and γ, and Θ maps as
# Θ[, x = 0] = (Intercept) and Θ[, x = 1] = (Intercept) + x.  Polished with
# the same BFGS step as the ctm fit.
suppressPackageStartupMessages(library(tram))
tram_lg0 <- suppressWarnings(
  Colr(y | x ~ 1 | x_s, data = df_lg, order = p_lg - 1L,
       support = c(a_lg, b_lg), bounds = c(a_lg, b_lg)))
stopifnot(inherits(tram_lg0, "stram"))
tram_lg <- mlt(tram_lg0$model, data = df_lg,
               theta = coef(tram_lg0, with_baseline = TRUE),
               optim = mltoptim(optim = list(checkconstraints = TRUE,
                                             method = "BFGS",
                                             maxit = 100000L,
                                             reltol = 1e-15))["optim"])
tram_cf_lg <- coef(tram_lg)
gamma_tram_lg <- tram_cf_lg[paste0("scl_", scl_names_lg)]
tram_theta_lg <- matrix(tram_cf_lg[1:(p_lg * q_lg)], p_lg, q_lg)
stopifnot(
  tram_lg$convergence == 0,
  isTRUE(all.equal(as.numeric(logLik(tram_lg)), ll_lg, tolerance = 1e-10)),
  isTRUE(all.equal(unname(gamma_tram_lg), unname(gamma_lg), tolerance = 1e-6)),
  isTRUE(all.equal(cbind(tram_theta_lg[, 1], rowSums(tram_theta_lg)),
                   matrix(theta_lg[1:(p_lg * q_lg)], p_lg, q_lg),
                   tolerance = 1e-6, check.attributes = FALSE)))

# Per-row predictions at (y, x, x_s), assembled from the tensor basis exactly
# as in the mev block (``predict()`` on a scaled ctm errors in mlt 1.8.4):
# h = Z·vec(Θ) · exp(0.5·x_s·γ), h' = Z'·vec(Θ) · exp(0.5·x_s·γ).  The
# stopifnot() below checks the reconstruction reproduces logLik(fit).
# Logistic: F = plogis(h), f = dlogis(h)·h'.
bb_lg <- b(bresponse = b_y_lg, binteracting = b_x_lg)
pq_lg <- p_lg * q_lg
.h_lg <- function(d) {
  f <- exp(0.5 * d$x_s * gamma_lg)
  list(h  = drop(model.matrix(bb_lg, data = d) %*% theta_lg[1:pq_lg]) * f,
       hp = drop(model.matrix(bb_lg, data = d, deriv = c(y = 1)) %*%
                   theta_lg[1:pq_lg]) * f)
}
tr_lg <- .h_lg(df_lg)
stopifnot(isTRUE(all.equal(sum(dlogis(tr_lg$h, log = TRUE) + log(tr_lg$hp)),
                           ll_lg, tolerance = 1e-10)))

grid_lg <- expand.grid(y = quantile(y_lg, c(0.1, 0.3, 0.5, 0.7, 0.9)),
                       x = c(0.2, 0.8),
                       x_s = c(-1, 0, 1.5))
hg_lg   <- .h_lg(grid_lg)
dist_lg <- plogis(hg_lg$h)
dens_lg <- dlogis(hg_lg$h) * hg_lg$hp

writeLines(format(y_lg, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_logistic_y_train.txt"))
writeLines(format(x_lg, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_logistic_x_train.txt"))
writeLines(format(x_s_lg, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_logistic_x_s_train.txt"))
writeLines(paste(format(a_lg, digits = 15), format(b_lg, digits = 15)),
           con = file.path(out_dir, "stram_bs_bs_logistic_y_support.txt"))
writeLines(format(theta_lg, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_logistic_theta.txt"))
writeLines(format(as.numeric(gamma_lg), digits = 15),
           con = file.path(out_dir, "stram_bs_bs_logistic_gamma.txt"))
writeLines(format(as.numeric(gamma_tram_lg), digits = 15),
           con = file.path(out_dir, "stram_bs_bs_logistic_gamma_tram.txt"))
writeLines(format(ll_lg, digits = 15),
           con = file.path(out_dir, "stram_bs_bs_logistic_loglik.txt"))
write.table(format(as.matrix(grid_lg), digits = 15),
            file = file.path(out_dir, "stram_bs_bs_logistic_grid.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
writeLines(format(as.numeric(dist_lg), digits = 15),
           con = file.path(out_dir, "stram_bs_bs_logistic_distribution.txt"))
writeLines(format(as.numeric(dens_lg), digits = 15),
           con = file.path(out_dir, "stram_bs_bs_logistic_density.txt"))

cat(sprintf("stram_bs_bs logistic ref: n=%d, p=%d, q=%d, q_s=%d, ll=%.6f\n",
            n_lg, p_lg, q_lg, length(gamma_lg), ll_lg))

# ---------------------------------------------------------------------------
# BoxCox(interacting=..., scaling=...) — issue #107.
#
# The stratified, scaled Box-Cox model of ADR 0003 Decision 6, fitted through
# the R ``tram`` formula interface as the no-shift ``stram`` model
#
#     BoxCox(y | s ~ 1 | z)
#
# i.e. h(y | s, z) = (a(y)ᵀ Θ e_s) · exp(0.5 · z · γ) with a Normal link.  The
# stratum s is a 3-level factor; mltpy's analogue is
# ``BoxCox(interacting=OneHotBasis(K=3), scaling=z)``.  tram encodes s with
# treatment contrasts — columns ``Bs_i(y):(Intercept)``, ``Bs_i(y):sb``,
# ``Bs_i(y):sc`` — so its Θ is a linear reparametrisation of mltpy's one-hot
# Θ: column k of the one-hot Θ is the intercept column plus contrast column k
# (zero for the reference level).  The function space, logLik and γ are
# identical at an interior MLE; ``*_theta_onehot.txt`` stores the mapped Θ.
#
# Data: within stratum k, h = exp(0.5·γ·z)·sl_k·(y − mu_k) with γ = 0.4 —
# linear in y, hence exactly in the Bernstein (order 2) span, with different
# slopes and locations per stratum (crossing conditional CDFs).  n = 900 keeps
# the MLE strictly interior (every Θ column increment > 1).
#
# Precision: as in the stram_bs_bs_mev block, tram's default auglag stops
# ~5e-3 short of the optimum on Θ, so the fit is polished with plain BFGS at
# reltol = 1e-15 from that solution (``checkconstraints = TRUE`` confirms
# feasibility); a second polish moves θ by ~1e-16.
#
# ``predict()`` on a stram fit errors inside mlt (observed with mlt 1.8.4), so
# h is assembled from the tensor basis directly and the stopifnot() below
# checks it reproduces logLik(fit).  Normal link: F = Φ(h), f = φ(h)·h'; the
# quantile solves h(y) = Φ⁻¹(p) by uniroot on the support.
#
# Fixtures emitted (prefix ``stram_onehot_bs_boxcox_``):
#   *_y_train.txt       — response, length n
#   *_s_train.txt       — stratum codes 0..K-1 (mltpy OneHotBasis labels)
#   *_x_s_train.txt     — scaling covariate z, length n
#   *_y_support.txt     — "a b" on one line
#   *_theta.txt         — raw coef(fit, with_baseline = TRUE): [Θ (tram
#                         treatment-contrast order) | γ]
#   *_theta_onehot.txt  — one-hot Θ, (p, K) flattened row-major (mltpy vec_C)
#   *_gamma.txt         — scale coefficient coef(fit)[fit$scalecoef], length 1
#   *_loglik.txt        — scalar logLik
#   *_grid.txt          — prediction grid, one "y s z" row per line
#   *_distribution.txt  — F(y | s, z) at each grid row
#   *_density.txt       — f(y | s, z) at each grid row
#   *_qgrid.txt         — quantile grid, one "p s z" row per line
#   *_quantile.txt      — F⁻¹(p | s, z) at each quantile-grid row
# ---------------------------------------------------------------------------
set.seed(20261101)
n_bx <- 900L
p_bx <- 3L  # number of Bernstein-y functions
lev_bx <- c("a", "b", "c")
s_bx   <- factor(sample(lev_bx, n_bx, replace = TRUE), levels = lev_bx)
z_bx   <- rnorm(n_bx)
sl_bx  <- c(a = 1.5, b = 2.5, c = 2.0)[as.character(s_bx)]
mu_bx  <- c(a = 0.0, b = 0.5, c = -0.3)[as.character(s_bx)]
y_bx   <- mu_bx + rnorm(n_bx) / (sl_bx * exp(0.5 * 0.4 * z_bx))
sup_bx <- c(min(y_bx) - 0.2, max(y_bx) + 0.2)
df_bx  <- data.frame(y = y_bx, s = s_bx, z = z_bx)

fit0_bx <- suppressWarnings(  # "highly experimental" stram warning
  BoxCox(y | s ~ 1 | z, data = df_bx, order = p_bx - 1L, support = sup_bx,
         bounds = c(-Inf, Inf)))
fit_bx <- suppressWarnings(
  BoxCox(y | s ~ 1 | z, data = df_bx, order = p_bx - 1L, support = sup_bx,
         bounds = c(-Inf, Inf), theta = coef(fit0_bx, with_baseline = TRUE),
         optim = mltoptim(optim = list(checkconstraints = TRUE,
                                       method = "BFGS",
                                       maxit = 100000L,
                                       reltol = 1e-15))["optim"]))
stopifnot(inherits(fit_bx, "stram"), fit_bx$convergence == 0)

theta_bx <- coef(fit_bx, with_baseline = TRUE)
ll_bx    <- as.numeric(logLik(fit_bx))
gamma_bx <- theta_bx[fit_bx$scalecoef]

# Tensor basis with the same response basis and treatment contrasts as tram;
# its column names match the baseline entries of coef(fit).
bb_bx <- b(bresponse = Bernstein_basis(numeric_var("y", support = sup_bx),
                                       order = p_bx - 1L, ui = "increasing"),
           binteracting = as.basis(~ s, data = df_bx))
th0_bx <- theta_bx[colnames(model.matrix(bb_bx, data = df_bx))]
.h_bx <- function(d) {
  f <- exp(0.5 * d$z * gamma_bx)
  list(h  = drop(model.matrix(bb_bx, data = d) %*% th0_bx) * f,
       hp = drop(model.matrix(bb_bx, data = d, deriv = c(y = 1)) %*% th0_bx) * f)
}
tr_bx <- .h_bx(df_bx)
stopifnot(isTRUE(all.equal(sum(dnorm(tr_bx$h, log = TRUE) + log(tr_bx$hp)),
                           ll_bx, tolerance = 1e-10)))

# One-hot Θ: rows = Bernstein-y functions, columns = strata a, b, c.
C_bx <- matrix(th0_bx, nrow = p_bx)  # columns: (Intercept), sb, sc
Theta_bx <- cbind(C_bx[, 1], C_bx[, 1] + C_bx[, -1, drop = FALSE])
stopifnot(min(diff(Theta_bx)) > 1)

grid_bx <- expand.grid(y = quantile(y_bx, c(0.1, 0.3, 0.5, 0.7, 0.9)),
                       s = factor(lev_bx, levels = lev_bx),
                       z = c(-1, 0, 1.5))
hg_bx <- .h_bx(grid_bx)
cdf_bx <- pnorm(hg_bx$h)
pdf_bx <- dnorm(hg_bx$h) * hg_bx$hp

qgrid_bx <- expand.grid(p = c(0.1, 0.5, 0.9),
                        s = factor(lev_bx, levels = lev_bx),
                        z = c(-1, 0, 1.5))
q_bx <- vapply(seq_len(nrow(qgrid_bx)), function(i) {
  r <- qgrid_bx[i, ]
  uniroot(function(yy) .h_bx(data.frame(y = yy, s = r$s, z = r$z))$h -
            qnorm(r$p), interval = sup_bx, tol = 1e-14)$root
}, numeric(1))

.codes_bx <- function(g) cbind(g[[1]], as.integer(g$s) - 1L, g$z)
writeLines(format(y_bx, digits = 15),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_y_train.txt"))
writeLines(as.character(as.integer(s_bx) - 1L),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_s_train.txt"))
writeLines(format(z_bx, digits = 15),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_x_s_train.txt"))
writeLines(paste(format(sup_bx[1], digits = 15), format(sup_bx[2], digits = 15)),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_y_support.txt"))
writeLines(format(theta_bx, digits = 15),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_theta.txt"))
writeLines(format(flatten_row_major(Theta_bx), digits = 15),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_theta_onehot.txt"))
writeLines(format(as.numeric(gamma_bx), digits = 15),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_gamma.txt"))
writeLines(format(ll_bx, digits = 15),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_loglik.txt"))
write.table(format(.codes_bx(grid_bx), digits = 15),
            file = file.path(out_dir, "stram_onehot_bs_boxcox_grid.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
writeLines(format(cdf_bx, digits = 15),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_distribution.txt"))
writeLines(format(pdf_bx, digits = 15),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_density.txt"))
write.table(format(.codes_bx(qgrid_bx), digits = 15),
            file = file.path(out_dir, "stram_onehot_bs_boxcox_qgrid.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
writeLines(format(q_bx, digits = 15),
           con = file.path(out_dir, "stram_onehot_bs_boxcox_quantile.txt"))

cat(sprintf("stram_onehot_bs boxcox ref: n=%d, p=%d, K=%d, ll=%.6f, gamma=%.6f\n",
            n_bx, p_bx, length(lev_bx), ll_bx, gamma_bx))

# ---------------------------------------------------------------------------
# Lm(interacting=..., scaling=...) — issue #109.
#
# The stratified, scaled normal linear model of ADR 0003 Decision 6, fitted
# through the R ``tram`` formula interface as the no-shift ``stram`` model
#
#     Lm(y | s ~ 1 | z)
#
# i.e. h(y | s, z) = (c0_s + c1_s · y) · exp(0.5 · z · γ) with a Normal link:
# within stratum s, Y is normal with its own mean and standard deviation,
# both modulated by z.  mltpy's analogue is
# ``Lm(interacting=OneHotBasis(K=3), scaling=z)``.
#
# Two reparametrisations separate tram's Θ from mltpy's:
#   * tram::Lm's response basis is the polynomial (1, y) — coefficients
#     ``(Intercept):*`` and ``y:*`` — whereas mltpy's Lm uses the order-1
#     Bernstein basis on the support (a, b).  Both span the affine functions
#     of y: θ0 = c0 + c1·a, θ1 = c0 + c1·b, and c1 ≥ 0 ⇔ θ1 ≥ θ0.
#   * tram encodes s with treatment contrasts; one-hot column k is the
#     intercept column plus contrast column k (zero for the reference level).
# The function space, logLik and γ are therefore identical at an interior
# MLE; ``*_theta_onehot.txt`` stores the mapped Bernstein one-hot Θ.
#
# Data: within stratum k, h = exp(0.5·γ·z)·sl_k·(y − mu_k) with γ = −0.6 —
# exactly the model above.  The support is widened by 1 on each side so every
# grid quantile lies well inside it (the affine model does not depend on it).
#
# Precision: as in the BoxCox block, the default auglag fit is polished with
# plain BFGS at reltol = 1e-15 (``checkconstraints = TRUE`` confirms
# feasibility).  h is assembled from the coefficients directly and the
# stopifnot() below checks it reproduces logLik(fit), which also pins tram's
# response basis to the raw-y polynomial.  F = Φ(h), f = φ(h)·h', and the
# quantile is closed-form: y = (Φ⁻¹(p)·exp(−0.5·z·γ) − c0_s) / c1_s.
#
# Fixtures emitted (prefix ``stram_onehot_lm_``):
#   *_y_train.txt       — response, length n
#   *_s_train.txt       — stratum codes 0..K-1 (mltpy OneHotBasis labels)
#   *_x_s_train.txt     — scaling covariate z, length n
#   *_y_support.txt     — "a b" on one line
#   *_theta.txt         — raw coef(fit, with_baseline = TRUE): [Θ (tram
#                         polynomial-y, treatment-contrast order) | γ]
#   *_theta_onehot.txt  — Bernstein one-hot Θ, (2, K) flattened row-major
#                         (mltpy vec_C)
#   *_gamma.txt         — scale coefficient coef(fit)[fit$scalecoef], length 1
#   *_loglik.txt        — scalar logLik
#   *_grid.txt          — prediction grid, one "y s z" row per line
#   *_distribution.txt  — F(y | s, z) at each grid row
#   *_density.txt       — f(y | s, z) at each grid row
#   *_qgrid.txt         — quantile grid, one "p s z" row per line
#   *_quantile.txt      — F⁻¹(p | s, z) at each quantile-grid row
# ---------------------------------------------------------------------------
set.seed(20261109)
n_lm <- 900L
lev_lm <- c("a", "b", "c")
s_lm   <- factor(sample(lev_lm, n_lm, replace = TRUE), levels = lev_lm)
z_lm   <- rnorm(n_lm)
sl_lm  <- c(a = 1.2, b = 2.0, c = 3.0)[as.character(s_lm)]
mu_lm  <- c(a = 1.0, b = 1.5, c = 0.4)[as.character(s_lm)]
y_lm   <- mu_lm + rnorm(n_lm) / (sl_lm * exp(0.5 * -0.6 * z_lm))
sup_lm <- c(min(y_lm) - 1, max(y_lm) + 1)
df_lm  <- data.frame(y = y_lm, s = s_lm, z = z_lm)

fit0_lm <- suppressWarnings(  # "highly experimental" stram warning
  Lm(y | s ~ 1 | z, data = df_lm))
fit_lm <- suppressWarnings(
  Lm(y | s ~ 1 | z, data = df_lm, theta = coef(fit0_lm, with_baseline = TRUE),
     optim = mltoptim(optim = list(checkconstraints = TRUE,
                                   method = "BFGS",
                                   maxit = 100000L,
                                   reltol = 1e-15))["optim"]))
stopifnot(inherits(fit_lm, "stram"), fit_lm$convergence == 0)

theta_lm <- coef(fit_lm, with_baseline = TRUE)
ll_lm    <- as.numeric(logLik(fit_lm))
gamma_lm <- theta_lm[fit_lm$scalecoef]

# Per-stratum affine coefficients (c0_k, c1_k) from the treatment contrasts.
th0_lm <- theta_lm[setdiff(names(theta_lm), fit_lm$scalecoef)]
C_lm <- matrix(th0_lm, nrow = 2L)  # cols: (Intercept), sb, sc
stopifnot(names(th0_lm) ==
            paste0(c("(Intercept)", "y"), ":",
                   rep(c("(Intercept)", "sb", "sc"), each = 2L)))
Cs_lm <- cbind(C_lm[, 1], C_lm[, 1] + C_lm[, -1, drop = FALSE])  # (2, K)
stopifnot(min(Cs_lm[2, ]) > 0.5)  # interior: every slope well above 0

.h_lm <- function(d) {
  k <- as.integer(d$s)
  f <- exp(0.5 * d$z * gamma_lm)
  list(h = (Cs_lm[1, k] + Cs_lm[2, k] * d$y) * f, hp = Cs_lm[2, k] * f)
}
tr_lm <- .h_lm(df_lm)
stopifnot(isTRUE(all.equal(sum(dnorm(tr_lm$h, log = TRUE) + log(tr_lm$hp)),
                           ll_lm, tolerance = 1e-10)))

# Bernstein (order 1) one-hot Θ: rows = (θ0, θ1), columns = strata a, b, c.
Theta_lm <- rbind(Cs_lm[1, ] + Cs_lm[2, ] * sup_lm[1],
                  Cs_lm[1, ] + Cs_lm[2, ] * sup_lm[2])

grid_lm <- expand.grid(y = quantile(y_lm, c(0.1, 0.3, 0.5, 0.7, 0.9)),
                       s = factor(lev_lm, levels = lev_lm),
                       z = c(-1, 0, 1.5))
hg_lm <- .h_lm(grid_lm)
cdf_lm <- pnorm(hg_lm$h)
pdf_lm <- dnorm(hg_lm$h) * hg_lm$hp

qgrid_lm <- expand.grid(p = c(0.1, 0.5, 0.9),
                        s = factor(lev_lm, levels = lev_lm),
                        z = c(-1, 0, 1.5))
kq_lm <- as.integer(qgrid_lm$s)
q_lm <- (qnorm(qgrid_lm$p) * exp(-0.5 * qgrid_lm$z * gamma_lm) - Cs_lm[1, kq_lm]) /
  Cs_lm[2, kq_lm]
stopifnot(all(q_lm > sup_lm[1] + 0.1), all(q_lm < sup_lm[2] - 0.1))

.codes_lm <- function(g) cbind(g[[1]], as.integer(g$s) - 1L, g$z)
writeLines(format(y_lm, digits = 15),
           con = file.path(out_dir, "stram_onehot_lm_y_train.txt"))
writeLines(as.character(as.integer(s_lm) - 1L),
           con = file.path(out_dir, "stram_onehot_lm_s_train.txt"))
writeLines(format(z_lm, digits = 15),
           con = file.path(out_dir, "stram_onehot_lm_x_s_train.txt"))
writeLines(paste(format(sup_lm[1], digits = 15), format(sup_lm[2], digits = 15)),
           con = file.path(out_dir, "stram_onehot_lm_y_support.txt"))
writeLines(format(theta_lm, digits = 15),
           con = file.path(out_dir, "stram_onehot_lm_theta.txt"))
writeLines(format(flatten_row_major(Theta_lm), digits = 15),
           con = file.path(out_dir, "stram_onehot_lm_theta_onehot.txt"))
writeLines(format(as.numeric(gamma_lm), digits = 15),
           con = file.path(out_dir, "stram_onehot_lm_gamma.txt"))
writeLines(format(ll_lm, digits = 15),
           con = file.path(out_dir, "stram_onehot_lm_loglik.txt"))
write.table(format(.codes_lm(grid_lm), digits = 15),
            file = file.path(out_dir, "stram_onehot_lm_grid.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
writeLines(format(cdf_lm, digits = 15),
           con = file.path(out_dir, "stram_onehot_lm_distribution.txt"))
writeLines(format(pdf_lm, digits = 15),
           con = file.path(out_dir, "stram_onehot_lm_density.txt"))
write.table(format(.codes_lm(qgrid_lm), digits = 15),
            file = file.path(out_dir, "stram_onehot_lm_qgrid.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
writeLines(format(q_lm, digits = 15),
           con = file.path(out_dir, "stram_onehot_lm_quantile.txt"))

cat(sprintf("stram_onehot lm ref: n=%d, K=%d, ll=%.6f, gamma=%.6f\n",
            n_lm, length(lev_lm), ll_lm, gamma_lm))

# ---------------------------------------------------------------------------
# Survreg(interacting=..., scaling=...) — issue #110.
#
# The log-time analogue of the stram_bs_bs_* blocks above.  mltpy's Survreg
# models the transformation on log(t) (``LogBernsteinBasis``), so the response
# basis is Bernstein on log(t) (``log_first = TRUE``) of order 1 — the
# strictly affine baseline ``tram::Survreg`` always fits — tensored with an
# order-1 Bernstein-x interacting basis and scaled by exp(0.5 · x_s · γ):
#
#     F_Z^{-1}(F(t | x, x_s)) = h(t | x) · exp(0.5 · x_s · γ),
#     h(t | x) = (a(log t) ⊗ b(x))ᵀ vec(Θ),
#
# with the link of each Survreg family: weibull ↔ ``MinExtrVal``, lognormal ↔
# ``Normal``, loglogistic ↔ ``Logistic``.  Exact (uncensored) event times
# only — the interaction likelihood path is exact-data only.  Fitted through
# ``mlt::ctm`` for term-for-term basis agreement with mltpy's
# ``InteractionBasis(LogBernsteinBasis, BernsteinBasis)``; γ is sign- and
# magnitude-aligned with mltpy (no flip).
#
# Data: h(t | x) = exp(0.5·γ·x_s)·(2·(log t − 1) + 0.8·(x − 0.5)) with
# γ = 0.4, which is affine in log t and in x and so lies exactly in the
# log-Bernstein-y (order 1) ⊗ Bernstein-x (order 1) span.  With n = 1000 the
# MLE is strictly interior (every Θ column increment is well above 1), and
# the same BFGS polish at reltol = 1e-15 as the stram_bs_bs blocks takes γ
# past alabama's inner tolerance.
#
# Cross-check against the literal tram formula route,
# ``tram::Survreg(y | x ~ 1 | x_s, dist = <dist>)`` (class ``stram``; tram
# warns that strata + scale is "highly experimental").  Its response basis is
# (1, log t) and its interacting term the linear (1, x) design, which span the
# same spaces as the order-1 Bernstein bases, so the two fits share the MLE:
# same logLik and γ, and with C the tram (response × interacting) coefficient
# matrix, Θ = M · [C[, 1], C[, 1] + C[, 2]] where M = [[1, log a], [1, log b]]
# maps (1, log t) coefficients to Bernstein-on-[log a, log b] coefficients.
#
# Fixtures emitted (prefix ``stram_logbs_bs_<dist>_``, dist ∈ weibull,
# lognormal, loglogistic):
#   *_y_train.txt      — event times, length n
#   *_x_train.txt      — interacting covariate, length n
#   *_x_s_train.txt    — scaling covariate, length n
#   *_y_support.txt    — "a b" on one line (original time scale)
#   *_theta.txt        — full coef(fit): [vec(Θ) (mlt order) | γ]
#   *_gamma.txt        — scale coefficients only, length q_s
#   *_gamma_tram.txt   — γ from the tram::Survreg(y | x ~ 1 | x_s) stram fit
#   *_loglik.txt       — scalar logLik
#   *_grid.txt         — prediction grid, one "t x x_s" row per line
#   *_survivor.txt     — S(t | x, x_s) at each grid row
#   *_density.txt      — f(t | x, x_s) = f_Z(h)·h' at each grid row
#   *_hazard.txt       — f(t | x, x_s) / S(t | x, x_s) at each grid row
#   *_qgrid.txt        — quantile grid, one "p x x_s" row per line
#   *_quantile.txt     — F⁻¹(p | x, x_s) at each quantile-grid row
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(tram))

.write_survreg_stram <- function(dist_name, todistr, seed) {
  zfun <- switch(todistr,
    MinExtrVal = list(
      q = function(u) log(-log1p(-u)),
      d = function(z) exp(z - exp(z)),
      s = function(z) exp(-exp(z))),
    Normal = list(q = qnorm, d = dnorm,
                  s = function(z) pnorm(z, lower.tail = FALSE)),
    Logistic = list(q = qlogis, d = dlogis,
                    s = function(z) plogis(z, lower.tail = FALSE)))

  set.seed(seed)
  n_sv <- 1000L
  p_sv <- 2L  # log-Bernstein-y functions (order 1, tram::Survreg's baseline)
  q_sv <- 2L  # Bernstein-x functions
  x_sv   <- runif(n_sv, 0, 1)
  x_s_sv <- rnorm(n_sv)
  # Invert F_Z(k · h0(t | x)) = U for log t.
  k_sv <- exp(0.5 * 0.4 * x_s_sv)
  y_sv <- exp(1 - 0.4 * (x_sv - 0.5) + zfun$q(runif(n_sv)) / (2 * k_sv))
  a_sv <- min(y_sv) * 0.9
  b_sv <- max(y_sv) * 1.1

  df_sv <- data.frame(y = y_sv, x = x_sv, x_s = x_s_sv)
  m_y_sv <- numeric_var("y", support = c(a_sv, b_sv), bounds = c(0, Inf))
  m_x_sv <- numeric_var("x", support = c(0, 1))
  b_y_sv <- Bernstein_basis(m_y_sv, order = p_sv - 1L, ui = "increasing",
                            log_first = TRUE)
  b_x_sv <- Bernstein_basis(m_x_sv, order = q_sv - 1L)
  b_s_sv <- as.basis(~ x_s, data = df_sv, remove_intercept = TRUE)

  polish <- mltoptim(optim = list(checkconstraints = TRUE, method = "BFGS",
                                  maxit = 100000L, reltol = 1e-15))["optim"]
  ctm_sv <- ctm(response = b_y_sv, interacting = b_x_sv, scaling = b_s_sv,
                scale_shift = FALSE, todistr = todistr)
  fit0_sv <- mlt(ctm_sv, data = df_sv)
  fit_sv  <- mlt(ctm_sv, data = df_sv, theta = coef(fit0_sv), optim = polish)
  stopifnot(fit_sv$convergence == 0,
            min(diff(matrix(coef(fit_sv)[1:(p_sv * q_sv)], p_sv, q_sv))) > 1)

  theta_sv <- coef(fit_sv)
  ll_sv    <- as.numeric(logLik(fit_sv))
  scl_names_sv <- colnames(model.matrix(b_s_sv, data = df_sv))
  gamma_sv <- theta_sv[scl_names_sv]

  # Literal tram stram route, polished the same way.
  tram_sv0 <- suppressWarnings(
    Survreg(y | x ~ 1 | x_s, data = df_sv, dist = dist_name,
            support = c(a_sv, b_sv), bounds = c(0, Inf)))
  stopifnot(inherits(tram_sv0, "stram"))
  tram_sv <- mlt(tram_sv0$model, data = df_sv,
                 theta = coef(tram_sv0, with_baseline = TRUE), optim = polish)
  tram_cf_sv <- coef(tram_sv)
  gamma_tram_sv <- tram_cf_sv[paste0("scl_", scl_names_sv)]
  C_sv <- matrix(tram_cf_sv[1:(p_sv * q_sv)], p_sv, q_sv)
  M_sv <- cbind(1, log(c(a_sv, b_sv)))
  stopifnot(
    tram_sv$convergence == 0,
    isTRUE(all.equal(as.numeric(logLik(tram_sv)), ll_sv, tolerance = 1e-10)),
    isTRUE(all.equal(unname(gamma_tram_sv), unname(gamma_sv),
                     tolerance = 1e-6)),
    isTRUE(all.equal(M_sv %*% cbind(C_sv[, 1], rowSums(C_sv)),
                     matrix(theta_sv[1:(p_sv * q_sv)], p_sv, q_sv),
                     tolerance = 1e-6, check.attributes = FALSE)))

  # Per-row predictions at (t, x, x_s), assembled from the tensor basis as in
  # the stram_bs_bs blocks (``predict()`` on a scaled ctm errors in mlt
  # 1.8.4): h = Z·vec(Θ) · exp(0.5·x_s·γ), h' = Z'·vec(Θ) · exp(0.5·x_s·γ),
  # where Z' already carries the 1/t chain-rule factor of log_first.  The
  # stopifnot() below checks the reconstruction reproduces logLik(fit).
  bb_sv <- b(bresponse = b_y_sv, binteracting = b_x_sv)
  pq_sv <- p_sv * q_sv
  .h_sv <- function(d) {
    f <- exp(0.5 * d$x_s * gamma_sv)
    list(h  = drop(model.matrix(bb_sv, data = d) %*% theta_sv[1:pq_sv]) * f,
         hp = drop(model.matrix(bb_sv, data = d, deriv = c(y = 1)) %*%
                     theta_sv[1:pq_sv]) * f)
  }
  tr_sv <- .h_sv(df_sv)
  stopifnot(isTRUE(all.equal(sum(log(zfun$d(tr_sv$h)) + log(tr_sv$hp)),
                             ll_sv, tolerance = 1e-10)))

  grid_sv <- expand.grid(y = quantile(y_sv, c(0.1, 0.3, 0.5, 0.7, 0.9)),
                         x = c(0.2, 0.8),
                         x_s = c(-1, 0, 1.5))
  hg_sv   <- .h_sv(grid_sv)
  surv_sv <- zfun$s(hg_sv$h)
  dens_sv <- zfun$d(hg_sv$h) * hg_sv$hp
  haz_sv  <- dens_sv / surv_sv

  qgrid_sv <- expand.grid(p = c(0.1, 0.5, 0.9),
                          x = c(0.2, 0.8),
                          x_s = c(-1, 0, 1.5))
  quant_sv <- vapply(seq_len(nrow(qgrid_sv)), function(i) {
    r <- qgrid_sv[i, ]
    exp(uniroot(function(lt) .h_sv(data.frame(y = exp(lt), x = r$x,
                                              x_s = r$x_s))$h - zfun$q(r$p),
                interval = log(c(a_sv, b_sv)), tol = 1e-14)$root)
  }, numeric(1))

  out <- function(name) {
    file.path(out_dir, sprintf("stram_logbs_bs_%s_%s.txt", dist_name, name))
  }
  writeLines(format(y_sv, digits = 15), con = out("y_train"))
  writeLines(format(x_sv, digits = 15), con = out("x_train"))
  writeLines(format(x_s_sv, digits = 15), con = out("x_s_train"))
  writeLines(paste(format(a_sv, digits = 15), format(b_sv, digits = 15)),
             con = out("y_support"))
  writeLines(format(theta_sv, digits = 15), con = out("theta"))
  writeLines(format(as.numeric(gamma_sv), digits = 15), con = out("gamma"))
  writeLines(format(as.numeric(gamma_tram_sv), digits = 15),
             con = out("gamma_tram"))
  writeLines(format(ll_sv, digits = 15), con = out("loglik"))
  write.table(format(as.matrix(grid_sv), digits = 15), file = out("grid"),
              row.names = FALSE, col.names = FALSE, quote = FALSE)
  writeLines(format(as.numeric(surv_sv), digits = 15), con = out("survivor"))
  writeLines(format(as.numeric(dens_sv), digits = 15), con = out("density"))
  writeLines(format(as.numeric(haz_sv), digits = 15), con = out("hazard"))
  write.table(format(as.matrix(qgrid_sv), digits = 15), file = out("qgrid"),
              row.names = FALSE, col.names = FALSE, quote = FALSE)
  writeLines(format(quant_sv, digits = 15), con = out("quantile"))

  cat(sprintf(paste0("stram_logbs_bs %s ref: n=%d, p=%d, q=%d, q_s=%d, ",
                     "ll=%.6f, gamma=%.6f\n"),
              dist_name, n_sv, p_sv, q_sv, length(gamma_sv), ll_sv, gamma_sv))
}

.write_survreg_stram("weibull",     "MinExtrVal", 20261110)
.write_survreg_stram("lognormal",   "Normal",     20261111)
.write_survreg_stram("loglogistic", "Logistic",   20261112)

# ---------------------------------------------------------------------------
# Interaction auglag false-stall reproduction — issue #142
#
# Bernstein-y (order 6, support (-4, 4)) ⊗ Bernstein-x (order 3, support
# (-1, 1)), Normal base, exact data.  On this data mltpy's auglag used to
# stall after a failed inner L-BFGS-B solve and report converged=True at a
# non-stationary point (logLik ≈ -2604.7 at n = 1000 instead of ≈ -1403.9).
#
# The training data were drawn with NumPy's PCG64 stream, which R cannot
# replay, so they are read from files written by
# ``python reference/write_interaction_stall_data.py``:
#
#   reference/interaction_stall_n<n>_y_train.txt — training y, length n
#   reference/interaction_stall_n<n>_x_train.txt — training x, length n
#
# Writes, for n ∈ {1000, 2000}:
#
#   reference/interaction_stall_n<n>_theta.txt  — coef(fit) in mlt's order
#                                                 (y-index fastest, i.e.
#                                                 column-major over Θ; length
#                                                 p*q = 28)
#   reference/interaction_stall_n<n>_loglik.txt — scalar logLik
#
# Precision: as in the stram_bs_bs_normal block (#125), the MLE is not
# interior (a Θ column sits on its monotonicity boundary) and the default
# alabama fit stops short of it, so the fit is polished with mlt's projected
# ``spg`` optimiser, which stays exactly inside ``ui θ ≥ ci``.
#
# Fixtures materialised with R 4.6.1, mlt 1.8.4, basefun 1.2.6,
# variables 1.1.2, BB 2026.1.0, alabama 2025.1.0.
# ---------------------------------------------------------------------------

.write_interaction_stall <- function(n) {
  path <- function(name) {
    file.path(out_dir, sprintf("interaction_stall_n%d_%s.txt", n, name))
  }
  y_st <- scan(path("y_train"), quiet = TRUE)
  x_st <- scan(path("x_train"), quiet = TRUE)
  stopifnot(length(y_st) == n, length(x_st) == n)

  b_y_st <- Bernstein_basis(numeric_var("y", support = c(-4, 4)),
                            order = 6, ui = "increasing")
  b_x_st <- Bernstein_basis(numeric_var("x", support = c(-1, 1)), order = 3)
  ctm_st <- ctm(response = b_y_st, interacting = b_x_st, todistr = "Normal")
  df_st  <- data.frame(y = y_st, x = x_st)
  fit0_st <- mlt(ctm_st, data = df_st)
  fit_st  <- mlt(ctm_st, data = df_st, theta = coef(fit0_st),
                 optim = mltoptim(spg = list(ftol = 1e-15, gtol = 1e-10,
                                             maxit = 100000L, quiet = TRUE,
                                             checkGrad = FALSE))["spg"])
  # Converged, feasible (every Θ column non-decreasing), and not worse than
  # the default fit.
  Theta_st <- matrix(coef(fit_st), 7L, 4L)
  stopifnot(fit_st$convergence == 0,
            min(diff(Theta_st)) >= 0,
            as.numeric(logLik(fit_st)) >= as.numeric(logLik(fit0_st)))
  theta_st <- coef(fit_st)
  ll_st    <- as.numeric(logLik(fit_st))

  writeLines(format(theta_st, digits = 15), con = path("theta"))
  writeLines(format(ll_st, digits = 15), con = path("loglik"))
  cat(sprintf("interaction_stall n=%d ref: p*q=%d, ll=%.6f\n",
              n, length(theta_st), ll_st))
}

.write_interaction_stall(1000L)
.write_interaction_stall(2000L)
