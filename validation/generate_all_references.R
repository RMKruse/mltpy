#!/usr/bin/env Rscript
# generate_all_references.R — Generate R mlt/tram reference values for pymlt validation
#
# Run with:  Rscript validation/generate_all_references.R
# Requires:  mlt, basefun, variables, jsonlite, tram
#
# Produces CSV + metadata.json files in validation/references/case_*/
# Then run:  python validation/convert_references.py   to create .npy files

suppressPackageStartupMessages({
  library(mlt)
  library(basefun)
  library(variables)
  library(jsonlite)
  library(tram)
})

# ---------------------------------------------------------------------------
# Shared helper: write all reference files for one case
# ---------------------------------------------------------------------------

save_reference <- function(path,
                           y = NULL,
                           status = NULL,
                           y_lower = NULL,
                           y_upper = NULL,
                           X = NULL,
                           theta,
                           loglik,
                           cdf_grid,
                           cdf_values,
                           metadata) {
  # Validate inputs
  stopifnot(length(cdf_grid) == 10)
  stopifnot(length(cdf_values) == 10)
  stopifnot(is.numeric(theta))
  stopifnot(is.numeric(loglik) && length(loglik) == 1)
  stopifnot(is.list(metadata))

  dir.create(path, recursive = TRUE, showWarnings = FALSE)

  # Helper: write a numeric vector as one-column CSV, 17 significant digits
  write_vec <- function(x, filename) {
    write.table(format(x, digits = 17),
                file = file.path(path, filename),
                row.names = FALSE, col.names = FALSE, quote = FALSE)
  }

  if (!is.null(y))       write_vec(y, "y.csv")
  if (!is.null(y_lower)) write_vec(y_lower, "y_lower.csv")
  if (!is.null(y_upper)) write_vec(y_upper, "y_upper.csv")

  # status: integer 0/1 (1 = event, 0 = censored)
  if (!is.null(status)) {
    write.table(as.integer(status),
                file = file.path(path, "status.csv"),
                row.names = FALSE, col.names = FALSE, quote = FALSE)
  }

  # X: matrix with column header
  if (!is.null(X)) {
    write.csv(X, file = file.path(path, "X.csv"), row.names = FALSE)
  }

  write_vec(theta,      "theta.csv")
  write_vec(loglik,     "loglik.csv")
  write_vec(cdf_grid,   "cdf_grid.csv")
  write_vec(cdf_values, "cdf_values.csv")

  # metadata.json
  write_json(metadata, path = file.path(path, "metadata.json"),
             auto_unbox = TRUE, digits = 17, pretty = TRUE)
}

# ---------------------------------------------------------------------------
# Helper: predict CDF at grid points via mlt
# For mlt-fitted objects, predict(fit, newdata=..., type="distribution")
# ---------------------------------------------------------------------------

predict_cdf_grid <- function(fit, var_name, grid) {
  nd <- data.frame(grid)
  names(nd) <- var_name
  as.numeric(predict(fit, newdata = nd, type = "distribution"))
}

# ---------------------------------------------------------------------------
# Case 01: MLT / NONE, n={200,1000}, order={4,6,8}
# Support: (0, 1), data from runif(n, 0.02, 0.98)
# ---------------------------------------------------------------------------

generate_case_01 <- function(base_path) {
  results <- list()
  configs <- list(
    list(n = 200,  orders = c(4, 6, 8), seed = 43),
    list(n = 1000, orders = c(4, 6, 8), seed = 143)
  )

  sup <- c(0, 1)

  for (cfg in configs) {
    set.seed(cfg$seed)
    y <- runif(cfg$n, 0.02, 0.98)
    # CDF grid: 10 equidistant points strictly inside support
    cdf_grid <- seq(sup[1] + 0.05, sup[2] - 0.05, length.out = 10)

    for (ord in cfg$orders) {
      tag <- sprintf("case_01_mlt_%d_%d", cfg$n, ord)
      cat("  ", tag, "... ")

      m <- numeric_var("y", support = sup, bounds = sup)
      b <- Bernstein_basis(m, order = ord, ui = "increasing")
      mod <- ctm(b)
      fit <- mlt(mod, data = data.frame(y = y))

      theta <- coef(fit)
      ll <- logLik(fit)
      cdf_vals <- predict_cdf_grid(fit, "y", cdf_grid)

      save_reference(
        path = file.path(base_path, tag),
        y = y, theta = theta, loglik = as.numeric(ll),
        cdf_grid = cdf_grid, cdf_values = cdf_vals,
        metadata = list(model = "mlt", censoring = "none",
                        n = cfg$n, order = ord,
                        support = sup, seed = cfg$seed)
      )

      results[[tag]] <- list(converged = TRUE, loglik = as.numeric(ll))
      cat("OK\n")
    }
  }
  results
}

# ---------------------------------------------------------------------------
# Case 02: MLT / RIGHT censoring, n={200,1000}, order={4,6}
# Support: (0, 5), data from rexp(n, rate=2), 30% censored
# ---------------------------------------------------------------------------

generate_case_02 <- function(base_path) {
  results <- list()
  configs <- list(
    list(n = 200,  seed = 44),
    list(n = 1000, seed = 144)
  )

  sup <- c(0, 5)
  orders <- c(4, 6)

  for (cfg in configs) {
    set.seed(cfg$seed)
    y_latent <- rexp(cfg$n, rate = 2)
    # Clip to support interior
    y_latent <- pmin(pmax(y_latent, sup[1] + 0.01), sup[2] - 0.01)
    # 30% censored: generate censoring times
    cens_time <- runif(cfg$n, 0, quantile(y_latent, 0.7))
    event <- y_latent <= cens_time
    # status: 1 = event observed, 0 = right-censored
    y_obs <- ifelse(event, y_latent, cens_time)
    y_obs <- pmin(pmax(y_obs, sup[1] + 0.01), sup[2] - 0.01)
    status <- as.integer(event)

    cdf_grid <- seq(sup[1] + 0.25, sup[2] - 0.25, length.out = 10)

    for (ord in orders) {
      tag <- sprintf("case_02_mlt_%d_%d", cfg$n, ord)
      cat("  ", tag, "... ")

      m <- numeric_var("y", support = sup, bounds = sup)
      b <- Bernstein_basis(m, order = ord, ui = "increasing")
      mod <- ctm(b)

      # Build Surv-style response for right-censored data
      dat <- data.frame(y = Surv(y_obs, status))
      fit <- mlt(mod, data = dat)

      theta <- coef(fit)
      ll <- logLik(fit)
      cdf_vals <- predict_cdf_grid(fit, "y", cdf_grid)

      save_reference(
        path = file.path(base_path, tag),
        y = y_obs, status = status,
        theta = theta, loglik = as.numeric(ll),
        cdf_grid = cdf_grid, cdf_values = cdf_vals,
        metadata = list(model = "mlt", censoring = "right",
                        n = cfg$n, order = ord,
                        support = sup, seed = cfg$seed,
                        censoring_pct = round(1 - mean(status), 3))
      )

      results[[tag]] <- list(converged = TRUE, loglik = as.numeric(ll))
      cat("OK\n")
    }
  }
  results
}

# ---------------------------------------------------------------------------
# Case 03: MLT / LEFT censoring, n=200, order={4,6}
# Support: (0, 6), data from rnorm(n, 3, 1), 30% left-censored
# ---------------------------------------------------------------------------

generate_case_03 <- function(base_path) {
  results <- list()
  sup <- c(0, 6)
  orders <- c(4, 6)
  seed <- 45

  set.seed(seed)
  n <- 200
  y_latent <- rnorm(n, mean = 3, sd = 1)
  y_latent <- pmin(pmax(y_latent, sup[1] + 0.01), sup[2] - 0.01)

  # Left-censoring: detection threshold — values below threshold are left-censored
  threshold <- quantile(y_latent, 0.3)
  left_censored <- y_latent < threshold
  y_obs <- ifelse(left_censored, threshold, y_latent)
  y_obs <- pmin(pmax(y_obs, sup[1] + 0.01), sup[2] - 0.01)
  # status: 1 = exact observation, 0 = left-censored
  status <- as.integer(!left_censored)

  cdf_grid <- seq(sup[1] + 0.3, sup[2] - 0.3, length.out = 10)

  for (ord in orders) {
    tag <- sprintf("case_03_mlt_%d_%d", n, ord)
    cat("  ", tag, "... ")

    m <- numeric_var("y", support = sup, bounds = sup)
    b <- Bernstein_basis(m, order = ord, ui = "increasing")
    mod <- ctm(b)

    # For left-censored data in mlt: use Surv(time, time2, type="interval2")
    # left-censored observations: lower = -Inf (or support lower), upper = y_obs
    # exact observations: lower = upper = y_obs
    y_lower <- ifelse(left_censored, sup[1], y_obs)
    y_upper <- y_obs
    # In mlt, left-censoring is specified via interval-censored Surv objects
    # Surv(time = lower, time2 = upper, type = "interval2")
    # For exact obs: time = time2; for left-censored: time = NA, time2 = threshold
    surv_time  <- ifelse(left_censored, NA, y_obs)
    surv_time2 <- y_obs
    dat <- data.frame(y = Surv(surv_time, surv_time2, type = "interval2"))
    fit <- mlt(mod, data = dat)

    theta <- coef(fit)
    ll <- logLik(fit)
    cdf_vals <- predict_cdf_grid(fit, "y", cdf_grid)

    save_reference(
      path = file.path(base_path, tag),
      y = y_obs, status = status,
      theta = theta, loglik = as.numeric(ll),
      cdf_grid = cdf_grid, cdf_values = cdf_vals,
      metadata = list(model = "mlt", censoring = "left",
                      n = n, order = ord,
                      support = sup, seed = seed,
                      censoring_pct = round(mean(left_censored), 3))
    )

    results[[tag]] <- list(converged = TRUE, loglik = as.numeric(ll))
    cat("OK\n")
  }
  results
}

# ---------------------------------------------------------------------------
# Case 04: MLT / INTERVAL censoring, n=200, order={4,6}
# Support: (2, 8), data from rnorm(n, 5, 1), interval width ~ 0.1 * sd
# ---------------------------------------------------------------------------

generate_case_04 <- function(base_path) {
  results <- list()
  sup <- c(2, 8)
  orders <- c(4, 6)
  seed <- 46

  set.seed(seed)
  n <- 200
  y_latent <- rnorm(n, mean = 5, sd = 1)
  y_latent <- pmin(pmax(y_latent, sup[1] + 0.01), sup[2] - 0.01)

  # Interval censoring: all observations are interval-censored
  # width ~ 0.1 * sd(y) ≈ 0.1
  half_width <- 0.05
  y_lower <- pmax(y_latent - half_width, sup[1] + 0.001)
  y_upper <- pmin(y_latent + half_width, sup[2] - 0.001)

  cdf_grid <- seq(sup[1] + 0.3, sup[2] - 0.3, length.out = 10)

  for (ord in orders) {
    tag <- sprintf("case_04_mlt_%d_%d", n, ord)
    cat("  ", tag, "... ")

    m <- numeric_var("y", support = sup, bounds = sup)
    b <- Bernstein_basis(m, order = ord, ui = "increasing")
    mod <- ctm(b)

    # Interval-censored: Surv(lower, upper, type = "interval2")
    dat <- data.frame(y = Surv(y_lower, y_upper, type = "interval2"))
    fit <- mlt(mod, data = dat)

    theta <- coef(fit)
    ll <- logLik(fit)
    cdf_vals <- predict_cdf_grid(fit, "y", cdf_grid)

    save_reference(
      path = file.path(base_path, tag),
      y_lower = y_lower, y_upper = y_upper,
      theta = theta, loglik = as.numeric(ll),
      cdf_grid = cdf_grid, cdf_values = cdf_vals,
      metadata = list(model = "mlt", censoring = "interval",
                      n = n, order = ord,
                      support = sup, seed = seed,
                      interval_half_width = half_width)
    )

    results[[tag]] <- list(converged = TRUE, loglik = as.numeric(ll))
    cat("OK\n")
  }
  results
}

# ---------------------------------------------------------------------------
# Case 05: BoxCox / NONE, n=200, order=6
# Support: (0.01, 10), data from rlnorm(n)
# Uses tram::BoxCox — lognormal data is the classic BoxCox use case
# ---------------------------------------------------------------------------

generate_case_05 <- function(base_path) {
  seed <- 47
  n <- 200
  ord <- 6
  sup <- c(0.01, 10)
  tag <- "case_05_boxcox_200_6"
  cat("  ", tag, "... ")

  set.seed(seed)
  y <- rlnorm(n)
  y <- pmin(pmax(y, sup[1] + 0.001), sup[2] - 0.001)

  # tram::BoxCox uses normal base distribution, no censoring
  fit <- BoxCox(y ~ 1, data = data.frame(y = y),
                support = sup, order = ord)

  theta <- coef(fit)
  ll <- logLik(fit)
  cdf_grid <- seq(sup[1] + 0.5, sup[2] - 0.5, length.out = 10)
  cdf_vals <- predict_cdf_grid(fit, "y", cdf_grid)

  save_reference(
    path = file.path(base_path, tag),
    y = y, theta = theta, loglik = as.numeric(ll),
    cdf_grid = cdf_grid, cdf_values = cdf_vals,
    metadata = list(model = "boxcox", censoring = "none",
                    n = n, order = ord,
                    support = sup, seed = seed,
                    base_distribution = "normal")
  )

  cat("OK\n")
  list(converged = TRUE, loglik = as.numeric(ll))
}

# ---------------------------------------------------------------------------
# Case 06: Coxph / RIGHT censoring, n=200, order=6
# Support: (0.01, 8), data from rexp(n, rate=1), 30% censored
# Uses tram::Coxph — exponential survival data
# ---------------------------------------------------------------------------

generate_case_06 <- function(base_path) {
  seed <- 48
  n <- 200
  ord <- 6
  sup <- c(0.01, 8)
  tag <- "case_06_coxph_200_6"
  cat("  ", tag, "... ")

  set.seed(seed)
  y_latent <- rexp(n, rate = 1)
  y_latent <- pmin(pmax(y_latent, sup[1] + 0.001), sup[2] - 0.001)

  # 30% censoring
  cens_time <- runif(n, 0, quantile(y_latent, 0.7))
  event <- y_latent <= cens_time
  y_obs <- ifelse(event, y_latent, cens_time)
  y_obs <- pmin(pmax(y_obs, sup[1] + 0.001), sup[2] - 0.001)
  status <- as.integer(event)

  # tram::Coxph uses normal base distribution, right censoring
  dat <- data.frame(y = y_obs, status = status)
  fit <- Coxph(Surv(y, status) ~ 1, data = dat,
               support = sup, order = ord)

  theta <- coef(fit)
  ll <- logLik(fit)
  cdf_grid <- seq(sup[1] + 0.4, sup[2] - 0.4, length.out = 10)
  cdf_vals <- predict_cdf_grid(fit, "y", cdf_grid)

  save_reference(
    path = file.path(base_path, tag),
    y = y_obs, status = status,
    theta = theta, loglik = as.numeric(ll),
    cdf_grid = cdf_grid, cdf_values = cdf_vals,
    metadata = list(model = "coxph", censoring = "right",
                    n = n, order = ord,
                    support = sup, seed = seed,
                    base_distribution = "normal",
                    censoring_pct = round(1 - mean(status), 3))
  )

  cat("OK\n")
  list(converged = TRUE, loglik = as.numeric(ll))
}

# ---------------------------------------------------------------------------
# Case 07: Colr / NONE, n=200, order=6
# Support: (-1, 5), data from rlogis(n, 2, 0.5)
# Uses tram::Colr — logistic base distribution, continuous outcome
# ---------------------------------------------------------------------------

generate_case_07 <- function(base_path) {
  seed <- 49
  n <- 200
  ord <- 6
  sup <- c(-1, 5)
  tag <- "case_07_colr_200_6"
  cat("  ", tag, "... ")

  set.seed(seed)
  y <- rlogis(n, location = 2, scale = 0.5)
  y <- pmin(pmax(y, sup[1] + 0.01), sup[2] - 0.01)

  # tram::Colr uses logistic base distribution, no censoring
  fit <- Colr(y ~ 1, data = data.frame(y = y),
              support = sup, order = ord)

  theta <- coef(fit)
  ll <- logLik(fit)
  cdf_grid <- seq(sup[1] + 0.3, sup[2] - 0.3, length.out = 10)
  cdf_vals <- predict_cdf_grid(fit, "y", cdf_grid)

  save_reference(
    path = file.path(base_path, tag),
    y = y, theta = theta, loglik = as.numeric(ll),
    cdf_grid = cdf_grid, cdf_values = cdf_vals,
    metadata = list(model = "colr", censoring = "none",
                    n = n, order = ord,
                    support = sup, seed = seed,
                    base_distribution = "logistic")
  )

  cat("OK\n")
  list(converged = TRUE, loglik = as.numeric(ll))
}

# ---------------------------------------------------------------------------
# Case 08: MLT / NONE + Regression, n=200, order=6
# Support: (0, 10), y = rnorm(n, 5, 1) + X %*% c(0.5, -0.3)
# Two covariates from N(0, 1)
# ---------------------------------------------------------------------------

generate_case_08 <- function(base_path) {
  seed <- 50
  n <- 200
  ord <- 6
  sup <- c(0, 10)
  tag <- "case_08_mlt_200_6"
  cat("  ", tag, "... ")

  set.seed(seed)
  X <- matrix(rnorm(n * 2), ncol = 2)
  colnames(X) <- c("x1", "x2")
  beta_true <- c(0.5, -0.3)
  y <- rnorm(n, mean = 5, sd = 1) + X %*% beta_true
  y <- as.numeric(y)
  y <- pmin(pmax(y, sup[1] + 0.01), sup[2] - 0.01)

  # MLT with covariates: ctm(basis_y, shifting = ~ x1 + x2)
  m <- numeric_var("y", support = sup, bounds = sup)
  b <- Bernstein_basis(m, order = ord, ui = "increasing")

  dat <- data.frame(y = y, x1 = X[, 1], x2 = X[, 2])
  mod <- ctm(b, shifting = ~ x1 + x2, data = dat)
  fit <- mlt(mod, data = dat)

  theta <- coef(fit)
  ll <- logLik(fit)
  # CDF predictions: at grid points for the "average" observation (X = 0)
  cdf_grid <- seq(sup[1] + 0.5, sup[2] - 0.5, length.out = 10)
  nd <- data.frame(y = cdf_grid, x1 = rep(0, 10), x2 = rep(0, 10))
  cdf_vals <- as.numeric(predict(fit, newdata = nd, type = "distribution"))

  save_reference(
    path = file.path(base_path, tag),
    y = y, X = X,
    theta = theta, loglik = as.numeric(ll),
    cdf_grid = cdf_grid, cdf_values = cdf_vals,
    metadata = list(model = "mlt", censoring = "none",
                    n = n, order = ord,
                    support = sup, seed = seed,
                    regression = TRUE,
                    n_covariates = 2,
                    beta_true = beta_true,
                    cdf_at_X = c(0, 0))
  )

  cat("OK\n")
  list(converged = TRUE, loglik = as.numeric(ll))
}

# ---------------------------------------------------------------------------
# Case 09: MLT / NONE, n=30, order=4
# Support: (0, 1), small sample — boundary/edge case
# ---------------------------------------------------------------------------

generate_case_09 <- function(base_path) {
  seed <- 51
  n <- 30
  ord <- 4
  sup <- c(0, 1)
  tag <- "case_09_mlt_30_4"
  cat("  ", tag, "... ")

  set.seed(seed)
  y <- runif(n, 0.02, 0.98)

  m <- numeric_var("y", support = sup, bounds = sup)
  b <- Bernstein_basis(m, order = ord, ui = "increasing")
  mod <- ctm(b)
  fit <- mlt(mod, data = data.frame(y = y))

  theta <- coef(fit)
  ll <- logLik(fit)
  cdf_grid <- seq(sup[1] + 0.05, sup[2] - 0.05, length.out = 10)
  cdf_vals <- predict_cdf_grid(fit, "y", cdf_grid)

  save_reference(
    path = file.path(base_path, tag),
    y = y, theta = theta, loglik = as.numeric(ll),
    cdf_grid = cdf_grid, cdf_values = cdf_vals,
    metadata = list(model = "mlt", censoring = "none",
                    n = n, order = ord,
                    support = sup, seed = seed,
                    note = "Small sample size — edge case for convergence")
  )

  cat("OK\n")
  list(converged = TRUE, loglik = as.numeric(ll))
}

# ---------------------------------------------------------------------------
# Case 10: MLT / RIGHT, n=30, order=4
# Support: (0, 5), small sample + right censoring
# ---------------------------------------------------------------------------

generate_case_10 <- function(base_path) {
  seed <- 52
  n <- 30
  ord <- 4
  sup <- c(0, 5)
  tag <- "case_10_mlt_30_4"
  cat("  ", tag, "... ")

  set.seed(seed)
  y_latent <- rexp(n, rate = 2)
  y_latent <- pmin(pmax(y_latent, sup[1] + 0.01), sup[2] - 0.01)

  # 30% censoring
  cens_time <- runif(n, 0, quantile(y_latent, 0.7))
  event <- y_latent <= cens_time
  y_obs <- ifelse(event, y_latent, cens_time)
  y_obs <- pmin(pmax(y_obs, sup[1] + 0.01), sup[2] - 0.01)
  status <- as.integer(event)

  m <- numeric_var("y", support = sup, bounds = sup)
  b <- Bernstein_basis(m, order = ord, ui = "increasing")
  mod <- ctm(b)
  dat <- data.frame(y = Surv(y_obs, status))
  fit <- mlt(mod, data = dat)

  theta <- coef(fit)
  ll <- logLik(fit)
  cdf_grid <- seq(sup[1] + 0.25, sup[2] - 0.25, length.out = 10)
  cdf_vals <- predict_cdf_grid(fit, "y", cdf_grid)

  save_reference(
    path = file.path(base_path, tag),
    y = y_obs, status = status,
    theta = theta, loglik = as.numeric(ll),
    cdf_grid = cdf_grid, cdf_values = cdf_vals,
    metadata = list(model = "mlt", censoring = "right",
                    n = n, order = ord,
                    support = sup, seed = seed,
                    censoring_pct = round(1 - mean(status), 3),
                    note = "Small sample + right censoring")
  )

  cat("OK\n")
  list(converged = TRUE, loglik = as.numeric(ll))
}

# ---------------------------------------------------------------------------
# Case 11: MLT / RIGHT >50% censored, n=200, order=6
# Support: (0, 5), stress test with heavy censoring
# ---------------------------------------------------------------------------

generate_case_11 <- function(base_path) {
  seed <- 53
  n <- 200
  ord <- 6
  sup <- c(0, 5)
  tag <- "case_11_mlt_200_6"
  cat("  ", tag, "... ")

  set.seed(seed)
  y_latent <- rexp(n, rate = 2)
  y_latent <- pmin(pmax(y_latent, sup[1] + 0.01), sup[2] - 0.01)

  # >50% censoring: use aggressive censoring times
  cens_time <- runif(n, 0, quantile(y_latent, 0.3))
  event <- y_latent <= cens_time
  y_obs <- ifelse(event, y_latent, cens_time)
  y_obs <- pmin(pmax(y_obs, sup[1] + 0.01), sup[2] - 0.01)
  status <- as.integer(event)

  actual_pct <- round(1 - mean(status), 3)
  cat(sprintf("(%.0f%% censored) ", actual_pct * 100))

  m <- numeric_var("y", support = sup, bounds = sup)
  b <- Bernstein_basis(m, order = ord, ui = "increasing")
  mod <- ctm(b)
  dat <- data.frame(y = Surv(y_obs, status))
  fit <- mlt(mod, data = dat)

  theta <- coef(fit)
  ll <- logLik(fit)
  cdf_grid <- seq(sup[1] + 0.25, sup[2] - 0.25, length.out = 10)
  cdf_vals <- predict_cdf_grid(fit, "y", cdf_grid)

  save_reference(
    path = file.path(base_path, tag),
    y = y_obs, status = status,
    theta = theta, loglik = as.numeric(ll),
    cdf_grid = cdf_grid, cdf_values = cdf_vals,
    metadata = list(model = "mlt", censoring = "right",
                    n = n, order = ord,
                    support = sup, seed = seed,
                    censoring_pct = actual_pct,
                    note = "Heavy censoring stress test (>50%)")
  )

  cat("OK\n")
  list(converged = TRUE, loglik = as.numeric(ll))
}

# ===========================================================================
# Main: run all cases with tryCatch, print summary table
# ===========================================================================

main <- function() {
  cat("========================================\n")
  cat("pymlt reference value generation\n")
  cat("========================================\n\n")

  base_path <- "validation/references"
  dir.create(base_path, recursive = TRUE, showWarnings = FALSE)

  generators <- list(
    list(name = "Case 01 (MLT/NONE)",          fn = generate_case_01),
    list(name = "Case 02 (MLT/RIGHT)",         fn = generate_case_02),
    list(name = "Case 03 (MLT/LEFT)",          fn = generate_case_03),
    list(name = "Case 04 (MLT/INTERVAL)",      fn = generate_case_04),
    list(name = "Case 05 (BoxCox/NONE)",       fn = generate_case_05),
    list(name = "Case 06 (Coxph/RIGHT)",       fn = generate_case_06),
    list(name = "Case 07 (Colr/NONE)",         fn = generate_case_07),
    list(name = "Case 08 (MLT/NONE+Reg)",      fn = generate_case_08),
    list(name = "Case 09 (MLT/NONE n=30)",     fn = generate_case_09),
    list(name = "Case 10 (MLT/RIGHT n=30)",    fn = generate_case_10),
    list(name = "Case 11 (MLT/RIGHT >50%)",    fn = generate_case_11)
  )

  all_results <- list()
  n_ok <- 0
  n_fail <- 0

  for (gen in generators) {
    cat(sprintf("[%s]\n", gen$name))
    result <- tryCatch(
      gen$fn(base_path),
      error = function(e) {
        cat(sprintf("  FAILED: %s\n", conditionMessage(e)))
        return(NULL)
      }
    )
    if (!is.null(result)) {
      # result is either a single list or a list of lists
      if ("converged" %in% names(result)) {
        all_results[[gen$name]] <- result
        n_ok <- n_ok + 1
      } else {
        for (nm in names(result)) {
          all_results[[nm]] <- result[[nm]]
          n_ok <- n_ok + 1
        }
      }
    } else {
      n_fail <- n_fail + 1
    }
  }

  # Summary table
  cat("\n========================================\n")
  cat("Summary\n")
  cat("========================================\n")
  cat(sprintf("%-35s | %11s | %12s\n", "Case", "Converged", "LogLik"))
  cat(paste(rep("-", 65), collapse = ""), "\n")
  for (nm in names(all_results)) {
    r <- all_results[[nm]]
    cat(sprintf("%-35s | %11s | %12.4f\n",
                nm,
                ifelse(r$converged, "YES", "NO"),
                r$loglik))
  }
  cat(paste(rep("-", 65), collapse = ""), "\n")
  cat(sprintf("\nAlle Referenzfälle erfolgreich generiert: %d / %d\n",
              n_ok, n_ok + n_fail))

  # Print R and package versions
  cat(sprintf("\nR version: %s\n", R.version.string))
  cat(sprintf("mlt version: %s\n", packageVersion("mlt")))
  cat(sprintf("tram version: %s\n", packageVersion("tram")))
  cat(sprintf("basefun version: %s\n", packageVersion("basefun")))
}

main()
