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

aic_small <- AIC(fit_small)
bic_small <- BIC(fit_small)
aic_large <- AIC(fit_large)
bic_large <- BIC(fit_large)

writeLines(
  paste(
    format(aic_small, digits = 15),
    format(bic_small, digits = 15),
    format(aic_large, digits = 15),
    format(bic_large, digits = 15)
  ),
  con = file.path(out_dir, "mlt_aic_bic.txt")
)

# anova(reduced, full) → row 2 has the LRT statistics
av <- anova(fit_small, fit_large)
chisq <- av[["Chisq"]][2]
df_lrt <- av[["Chi Df"]][2]
p_val  <- av[["Pr(>Chisq)"]][2]

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
