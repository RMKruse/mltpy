#!/usr/bin/env Rscript
# generate_auglag_reference.R — Generate alabama::auglag reference values
# for the pymlt auglag parity tests.
#
# Fixtures:
#   1. Lm-equivalent (order=1 Bernstein, normal base, no covariates)
#      n=50 standard normal observations, seed=42, no boundary constraints.
#      Tests the monotonicity-only path.
#   2. Bounded Bernstein (order=5 Bernstein, normal base, no covariates)
#      same data, theta[0] pinned to 0.0 and theta[6] pinned to 5.0.
#      Tests the boundary-equality path (slice 2).
#
# Run with:  Rscript tests/r_scripts/generate_auglag_reference.R
# Output:    tests/reference_data/auglag/lm_n50_seed42.json
#            tests/reference_data/auglag/bernstein_bounded_n50_seed42.json
#
# The negative log-likelihood is implemented directly (consistent with pymlt)
# to avoid dependence on mlt internal APIs.

suppressPackageStartupMessages({
  library(alabama)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

set.seed(42)
n     <- 50L
y     <- rnorm(n)
y_min <- min(y)
y_max <- max(y)
span  <- y_max - y_min
order <- 1L
p     <- order + 1L   # 2 parameters: [theta_0, theta_1]

# Normalised t_i = (y_i - y_min) / span  in [0, 1]
t_obs <- (y - y_min) / span

# Order-1 Bernstein basis matrix B, shape (n, p)
#   B[i, 1] = 1 - t_i   (B_{0,1})
#   B[i, 2] = t_i        (B_{1,1})
B <- cbind(1 - t_obs, t_obs)

# ---------------------------------------------------------------------------
# Negative log-likelihood consistent with pymlt likelihood.py
#   LL = sum_i [log phi(h_i) + log h'_i]
#   h_i  = B[i,] %*% theta  (clipped to +/-30 as in pymlt)
#   h'_i = (theta[2] - theta[1]) / span   (constant over i)
# ---------------------------------------------------------------------------

H_CLIP <- 30.0   # matches pymlt._H_CLIP

nll <- function(theta) {
  h      <- as.numeric(B %*% theta)
  h_clip <- pmax(pmin(h, H_CLIP), -H_CLIP)
  hprime <- (theta[2] - theta[1]) / span
  if (hprime <= 0) return(1e10)
  -sum(dnorm(h_clip, log = TRUE)) - n * log(hprime)
}

grad_nll <- function(theta) {
  h      <- as.numeric(B %*% theta)
  h_clip <- pmax(pmin(h, H_CLIP), -H_CLIP)
  hprime <- (theta[2] - theta[1]) / span
  # LL = sum log phi(h_i) + n * log(hprime)
  # d/d(theta_j) sum log phi(h_i) = -sum_i h_i * B[i,j]   (from -0.5*h^2)
  # d/d(theta_j) n*log(hprime) = n * c(-1,1)[j] / (theta[2]-theta[1])
  # gradient of NLL = -(grad_LL) = -(grad_ll + grad_hprime)
  grad_ll      <- -as.numeric(t(B) %*% h_clip)         # gradient of sum log phi(h_i)
  grad_hprime  <- n * c(-1, 1) / (theta[2] - theta[1]) # gradient of n * log(hprime)
  -(grad_ll + grad_hprime)
}

# ---------------------------------------------------------------------------
# Monotonicity constraint: theta[2] - theta[1] >= 0
#   hin(theta) >= 0  (alabama convention)
# ---------------------------------------------------------------------------

hin     <- function(theta) theta[2] - theta[1]
hin_jac <- function(theta) matrix(c(-1, 1), nrow = 1L)

# ---------------------------------------------------------------------------
# Starting point — matches pymlt _initial_theta: linspace(0, 1, p)
# ---------------------------------------------------------------------------

theta_init <- seq(0, 1, length.out = p)

# ---------------------------------------------------------------------------
# Run alabama::auglag
#   mu0=10 (initial penalty), eps=1e-7 (outer tol), itmax=50 (max outer iter)
#   method="L-BFGS-B" is passed via control.outer (alabama convention)
# ---------------------------------------------------------------------------

result <- auglag(
  par           = theta_init,
  fn            = nll,
  gr            = grad_nll,
  hin           = hin,
  hin.jac       = hin_jac,
  control.outer = list(
    mu0    = 10,
    eps    = 1e-7,
    itmax  = 50L,
    method = "L-BFGS-B"
  ),
  control.optim = list(maxit = 500L, pgtol = 1e-8)
)

theta_hat <- result$par
ll_hat    <- -nll(theta_hat)

cat(sprintf("theta:         [%.10f, %.10f]\n", theta_hat[1], theta_hat[2]))
cat(sprintf("log_likelihood: %.10f\n", ll_hat))
cat(sprintf("convergence:    %d  (0 = success)\n", result$convergence))
cat(sprintf("kkt1:           %.3e\n", result$kkt1))

# ---------------------------------------------------------------------------
# Write JSON reference
# ---------------------------------------------------------------------------

out <- list(
  y              = as.numeric(y),
  theta          = as.numeric(theta_hat),
  log_likelihood = as.numeric(ll_hat),
  support        = as.numeric(c(y_min, y_max)),
  order          = order,
  n              = n,
  seed           = 42L,
  starting_point = as.numeric(theta_init),
  r_version      = as.character(getRversion()),
  alabama_version = as.character(packageVersion("alabama"))
)

outfile <- file.path("tests", "reference_data", "auglag", "lm_n50_seed42.json")
write_json(out, outfile, digits = 15, auto_unbox = TRUE)
cat(sprintf("Written: %s\n", outfile))


# ===========================================================================
# Fixture 2: Bounded Bernstein (order = 1, lower = -2.5, no upper)
# ===========================================================================
#
# Same y data and support as fixture 1.  Pins theta[1] = -2.5 (R 1-indexed,
# = pymlt theta[0]).  This exercises the boundary-equality path in
# build_constraint_matrices(): a single C_eq row.
#
# Design notes:
#   * Order = 1 leaves a single free coefficient (theta_2).  The constrained
#     negative log-likelihood is therefore one-dimensional and strictly
#     convex, eliminating multi-modality.  Both alabama::auglag and pymlt's
#     PHR solver converge to the same KKT point regardless of their
#     internal stopping rules, which permits the strict rtol=1e-6 parity
#     assertion.
#   * lower = -2.5 sits a little below the unconstrained MLE of theta_1
#     (about -2.30 on this data), so the equality is genuinely binding and
#     the dual multiplier is non-zero — a no-op pin would not exercise the
#     equality update step in either algorithm.
# ---------------------------------------------------------------------------

cat("\n--- Fixture 2: bounded Bernstein (order=1, lower=-2.5) ---\n")

order2 <- 1L
p2     <- order2 + 1L          # 2 Bernstein coefficients
lower2 <- -2.5
upper2 <- NULL                 # no upper bound this fixture

# Order-1 Bernstein basis: B[i, 1] = 1 - t_i,  B[i, 2] = t_i  (same as fixture 1).
B2 <- cbind(1 - t_obs, t_obs)

# h(y) = theta_1 (1 - t) + theta_2 t.   h'(y) = (theta_2 - theta_1) / span,
# constant across observations.
nll2 <- function(theta) {
  h      <- as.numeric(B2 %*% theta)
  h_clip <- pmax(pmin(h, H_CLIP), -H_CLIP)
  hprime <- (theta[2] - theta[1]) / span
  if (hprime <= 0) return(1e10)
  -sum(dnorm(h_clip, log = TRUE)) - n * log(hprime)
}

grad_nll2 <- function(theta) {
  h      <- as.numeric(B2 %*% theta)
  h_clip <- pmax(pmin(h, H_CLIP), -H_CLIP)
  grad_ll     <- -as.numeric(t(B2) %*% h_clip)
  grad_hprime <- n * c(-1, 1) / (theta[2] - theta[1])
  -(grad_ll + grad_hprime)
}

# Monotonicity:  theta_2 - theta_1 >= 0.
hin2     <- function(theta) theta[2] - theta[1]
hin_jac2 <- function(theta) matrix(c(-1, 1), nrow = 1L)

# Boundary equality:  theta_1 - lower = 0.  (No upper this fixture.)
heq2     <- function(theta) theta[1] - lower2
heq_jac2 <- function(theta) matrix(c(1, 0), nrow = 1L)

# Starting point — same as pymlt _initial_theta with lower provided:
# linspace(0, 1, p2) shifted by lower (theta[0] starts at `lower`).
theta_init2 <- seq(0, 1, length.out = p2) + lower2

result2 <- auglag(
  par           = theta_init2,
  fn            = nll2,
  gr            = grad_nll2,
  hin           = hin2,
  hin.jac       = hin_jac2,
  heq           = heq2,
  heq.jac       = heq_jac2,
  control.outer = list(
    mu0    = 10,
    eps    = 1e-10,
    itmax  = 200L,
    method = "BFGS",
    ilack.max = 20L
  ),
  control.optim = list(maxit = 2000L, reltol = 1e-15)
)

theta_hat2 <- result2$par
ll_hat2    <- -nll2(theta_hat2)

cat(sprintf("theta:          [%s]\n",
            paste(sprintf("%.10f", theta_hat2), collapse = ", ")))
cat(sprintf("log_likelihood: %.10f\n", ll_hat2))
cat(sprintf("convergence:    %d  (0 = success)\n", result2$convergence))
cat(sprintf("kkt1:           %.3e\n", result2$kkt1))
cat(sprintf("kkt2:           %.3e\n", result2$kkt2))

out2 <- list(
  y               = as.numeric(y),
  theta           = as.numeric(theta_hat2),
  log_likelihood  = as.numeric(ll_hat2),
  support         = as.numeric(c(y_min, y_max)),
  order           = order2,
  n               = n,
  seed            = 42L,
  lower           = lower2,
  upper           = if (is.null(upper2)) NA else upper2,
  starting_point  = as.numeric(theta_init2),
  r_version       = as.character(getRversion()),
  alabama_version = as.character(packageVersion("alabama"))
)

outfile2 <- file.path(
  "tests", "reference_data", "auglag", "bernstein_bounded_n50_seed42.json"
)
write_json(out2, outfile2, digits = 15, auto_unbox = TRUE)
cat(sprintf("Written: %s\n", outfile2))
