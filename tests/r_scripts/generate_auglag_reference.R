#!/usr/bin/env Rscript
# generate_auglag_reference.R — Generate alabama::auglag reference values
# for the pymlt auglag parity test.
#
# Fixture: Lm-equivalent (order=1 Bernstein, normal base, no covariates)
#          n=50 standard normal observations, seed=42.
#
# Run with:  Rscript tests/r_scripts/generate_auglag_reference.R
# Output:    tests/reference_data/auglag/lm_n50_seed42.json
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
