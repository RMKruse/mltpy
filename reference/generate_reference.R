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
