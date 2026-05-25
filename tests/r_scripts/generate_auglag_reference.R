#!/usr/bin/env Rscript
# generate_auglag_reference.R — Generate alabama::auglag reference values
# for the mltpy auglag parity tests.
#
# Fixtures:
#   1. Lm-equivalent (order=1 Bernstein, normal base, no covariates)
#      n=50 standard normal observations, seed=42, no boundary constraints.
#      Tests the monotonicity-only path.
#   2. Bounded Bernstein (order=1 Bernstein, normal base, no covariates)
#      same data, theta[0] pinned to -2.5.
#      Tests the boundary-equality path (slice 2).
#   3. Exponential with one covariate (order=1 Bernstein, exponential base)
#      n=50 Exp(1) observations + one rnorm covariate.  Adds per-row
#      support inequalities theta_b[0] + X_i * beta >= 0 alongside the
#      monotonicity inequality.  Tests the support-inequality plumbing
#      added in slice 3.
#
# Run with:  Rscript tests/r_scripts/generate_auglag_reference.R
# Output:    tests/reference_data/auglag/lm_n50_seed42.json
#            tests/reference_data/auglag/bernstein_bounded_n50_seed42.json
#            tests/reference_data/auglag/exponential_n50_seed43.json
#
# The negative log-likelihood is implemented directly (consistent with mltpy)
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
# Negative log-likelihood consistent with mltpy likelihood.py
#   LL = sum_i [log phi(h_i) + log h'_i]
#   h_i  = B[i,] %*% theta  (clipped to +/-30 as in mltpy)
#   h'_i = (theta[2] - theta[1]) / span   (constant over i)
# ---------------------------------------------------------------------------

H_CLIP <- 30.0   # matches mltpy._H_CLIP

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
# Starting point — matches mltpy _initial_theta: linspace(0, 1, p)
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
# = mltpy theta[0]).  This exercises the boundary-equality path in
# build_constraint_matrices(): a single C_eq row.
#
# Design notes:
#   * Order = 1 leaves a single free coefficient (theta_2).  The constrained
#     negative log-likelihood is therefore one-dimensional and strictly
#     convex, eliminating multi-modality.  Both alabama::auglag and mltpy's
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

# Starting point — same as mltpy _initial_theta with lower provided:
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


# ===========================================================================
# Fixture 3: exponential with one covariate
# ===========================================================================
#
# Exercises the per-observation support-inequality path added in slice 3.
# Exponential base distribution requires h(y_i|x_i) >= 0 for every training
# observation; under monotone theta_b the minimum of h over y is attained
# at y_min where B_k(y_min) = [1, 0, ..., 0], so the per-row constraints
# reduce to
#   theta_b[1] + X_i * beta >= 0    for i = 1..n
# (R 1-indexed; this is mltpy's theta_b[0] + X_i.beta).
#
# Design notes:
#   * Order = 1 keeps the analytical NLL in a closed form (h'(y) constant
#     across observations: (theta_b[2] - theta_b[1]) / span).
#   * One rnorm covariate.  Both positive and negative values appear in X,
#     so the binding-constraint set is non-trivial — the n=50 support
#     inequalities are *not* all simultaneously slack at the optimum.
#   * Independent seed (43) keeps the fixture deterministic without
#     depending on which earlier fixture last consumed the RNG.
# ---------------------------------------------------------------------------

cat("\n--- Fixture 3: exponential, order=1, one covariate ---\n")

set.seed(43)
n3       <- 50L
x3       <- rnorm(n3)             # single covariate
y3       <- rexp(n3, rate = 1.0)  # Exp(1); guaranteed positive
y3_min   <- min(y3)
y3_max   <- max(y3)
span3    <- y3_max - y3_min
order3   <- 1L
p3       <- order3 + 1L           # 2 Bernstein coefficients
n_beta3  <- 1L
total3   <- p3 + n_beta3

# Order-1 Bernstein basis: B[i, 1] = 1 - t_i, B[i, 2] = t_i.
t3 <- (y3 - y3_min) / span3
B3 <- cbind(1 - t3, t3)

# h_i = B[i, ] %*% theta_b + x_i * beta
# log f_exp(h) = -h  (standard exponential density e^{-h} on [0, inf))
# log h'(y)   = log((theta_b[2] - theta_b[1]) / span)   (constant in i)
# NLL = sum(h_i) - n * log((theta_b[2] - theta_b[1]) / span)
nll3 <- function(theta) {
  tb     <- theta[1:p3]
  beta   <- theta[(p3 + 1):total3]
  h      <- as.numeric(B3 %*% tb + x3 * beta)
  h_clip <- pmax(pmin(h, H_CLIP), -H_CLIP)
  hprime <- (tb[2] - tb[1]) / span3
  if (hprime <= 0) return(1e10)
  sum(h_clip) - n3 * log(hprime)
}

# Gradient of NLL.  Within the clipping range,
#   d/d(theta_b[j]) sum(h) = colSums(B)[j]
#   d/d(beta)        sum(h) = sum(x)
#   d/d(theta_b)    -n log h' = n * c(-1, 1) / (tb[2] - tb[1])
#   d/d(beta)       -n log h' = 0
grad_nll3 <- function(theta) {
  tb         <- theta[1:p3]
  beta       <- theta[(p3 + 1):total3]
  # gradient of sum(h_i) -- ignores clipping (data range stays well within +/-30)
  grad_tb_h  <- colSums(B3)
  grad_b_h   <- sum(x3)
  grad_tb_hp <- n3 * c(-1, 1) / (tb[2] - tb[1])
  c(grad_tb_h - grad_tb_hp, grad_b_h)
}

# Constraints
# - Monotonicity:  theta_b[2] - theta_b[1] >= 0     (1 row)
# - Support:       theta_b[1] + x_i * beta >= 0     (n rows)
hin3 <- function(theta) {
  tb   <- theta[1:p3]
  beta <- theta[(p3 + 1):total3]
  c(tb[2] - tb[1], tb[1] + x3 * beta)
}

hin_jac3 <- function(theta) {
  # Row 1 (monotonicity): d/d(theta_b[1]) = -1, d/d(theta_b[2]) = 1, d/d(beta) = 0
  J_mono <- matrix(c(-1, 1, 0), nrow = 1L)
  # Rows 2..n+1 (support):  d/d(theta_b[1]) = 1, d/d(theta_b[2]) = 0, d/d(beta) = x_i
  J_supp <- cbind(rep(1.0, n3), rep(0.0, n3), x3)
  rbind(J_mono, J_supp)
}

# Starting point: mltpy _initial_theta with no boundary pins:
#   theta_b = linspace(0, 1, p) = c(0, 1);  beta = 0.
# Monotonicity: 1 - 0 = 1 >= 0 (OK).
# Support: 0 + x_i * 0 = 0 >= 0 for every i (feasible at the boundary).
theta_init3 <- c(seq(0, 1, length.out = p3), rep(0.0, n_beta3))

## Tolerances mirror fixture 2: with 50 active per-row support constraints
## bound at zero, the default alabama settings (eps=1e-7, L-BFGS-B) terminate
## with stationarity residual ~1e-2 on the free coordinate theta_b[2].  BFGS
## + reltol=1e-15 + eps=1e-10 reaches the analytical KKT optimum
##   theta_b[2] = n / sum(t)   with theta_b[1] = beta = 0
## so the mltpy parity assertion at rtol=1e-6 holds.
result3 <- auglag(
  par           = theta_init3,
  fn            = nll3,
  gr            = grad_nll3,
  hin           = hin3,
  hin.jac       = hin_jac3,
  control.outer = list(
    mu0       = 10,
    eps       = 1e-10,
    itmax     = 200L,
    method    = "BFGS",
    ilack.max = 20L
  ),
  control.optim = list(maxit = 2000L, reltol = 1e-15)
)

theta_hat3 <- result3$par
ll_hat3    <- -nll3(theta_hat3)

cat(sprintf("theta:          [%s]\n",
            paste(sprintf("%.10f", theta_hat3), collapse = ", ")))
cat(sprintf("log_likelihood: %.10f\n", ll_hat3))
cat(sprintf("convergence:    %d  (0 = success)\n", result3$convergence))
cat(sprintf("kkt1:           %.3e\n", result3$kkt1))

out3 <- list(
  y               = as.numeric(y3),
  X               = matrix(x3, ncol = 1L),
  theta           = as.numeric(theta_hat3),
  log_likelihood  = as.numeric(ll_hat3),
  support         = as.numeric(c(y3_min, y3_max)),
  order           = order3,
  n               = n3,
  seed            = 43L,
  starting_point  = as.numeric(theta_init3),
  base_distribution = "exponential",
  r_version       = as.character(getRversion()),
  alabama_version = as.character(packageVersion("alabama"))
)

outfile3 <- file.path(
  "tests", "reference_data", "auglag", "exponential_n50_seed43.json"
)
write_json(out3, outfile3, digits = 15, auto_unbox = TRUE)
cat(sprintf("Written: %s\n", outfile3))
