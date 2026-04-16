#!/usr/bin/env Rscript
# bench_r.R — Runtime benchmark for R `mlt::mlt()`, mirroring bench_python.py.
#
# Reads the byte-identical input CSVs written by `bench_python.py` from
# benchmarks/data/ and writes benchmarks/results/r_results.csv with columns:
#
#   n, order, censoring, rep, time_s, converged, n_iter
#
# Run with:
#   Rscript benchmarks/bench_r.R
# or via the `benchmark` Makefile target.

suppressPackageStartupMessages({
  for (pkg in c("survival", "variables", "basefun", "mlt")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf(
        "Required R package '%s' is not installed.\n  install.packages('%s')",
        pkg, pkg
      ))
    }
    library(pkg, character.only = TRUE)
  }
})

# ---------------------------------------------------------------------------
# Configuration — must mirror benchmarks/bench_python.py exactly
# ---------------------------------------------------------------------------

N_LIST         <- c(100L, 500L, 1000L, 5000L)
ORDER_LIST     <- c(4L, 6L, 8L)
CENSORING_LIST <- c("NONE", "RIGHT")
N_REPS         <- 10L
SUPPORT        <- c(0.0, 10.0)

# ---------------------------------------------------------------------------
# Path resolution (works under `Rscript`)
# ---------------------------------------------------------------------------

resolve_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", m[1L]))))
  }
  # Fallback when sourced interactively from repo root
  file.path(getwd(), "benchmarks")
}

SCRIPT_DIR  <- resolve_script_dir()
DATA_DIR    <- file.path(SCRIPT_DIR, "data")
RESULTS_DIR <- file.path(SCRIPT_DIR, "results")
RESULTS_CSV <- file.path(RESULTS_DIR, "r_results.csv")

if (!dir.exists(RESULTS_DIR)) {
  dir.create(RESULTS_DIR, recursive = TRUE)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

load_dataset <- function(n, censoring) {
  path <- file.path(DATA_DIR, sprintf("n%d_%s.csv", n, censoring))
  if (!file.exists(path)) {
    stop(sprintf(
      paste0("Input dataset not found: %s\n",
             "Run `python benchmarks/bench_python.py` first to generate it."),
      path
    ))
  }
  read.csv(path)
}

build_data <- function(df, censoring) {
  if (censoring == "RIGHT") {
    data.frame(y = Surv(df$y, df$status))
  } else {
    data.frame(y = df$y)
  }
}

build_model <- function(order) {
  yvar <- numeric_var("y", support = SUPPORT, bounds = SUPPORT)
  B    <- Bernstein_basis(yvar, order = order, ui = "increasing")
  ctm(B)
}

# Best-effort extraction of optimiser iteration count. mlt's internal
# optimiser structure differs across versions, so fall back to NA when
# nothing usable is exposed on the fit object.
extract_n_iter <- function(fit) {
  for (field in c("counts", "iter", "iterations")) {
    val <- tryCatch(fit[[field]], error = function(e) NULL)
    if (!is.null(val)) {
      val <- suppressWarnings(as.integer(val[[1L]]))
      if (length(val) == 1L && !is.na(val)) return(val)
    }
  }
  NA_integer_
}

is_converged <- function(fit) {
  conv <- tryCatch(fit$convergence, error = function(e) NULL)
  if (is.null(conv)) {
    return(TRUE) # mlt() returned without error → treat as converged
  }
  isTRUE(as.integer(conv) == 0L)
}

time_one_fit <- function(model, data) {
  invisible(gc(verbose = FALSE)) # encourage clean state before timing
  t0 <- Sys.time()
  fit <- tryCatch(
    suppressWarnings(mlt(model, data = data)),
    error = function(e) e
  )
  elapsed <- as.numeric(Sys.time() - t0, units = "secs")
  if (inherits(fit, "error")) {
    return(list(elapsed = elapsed, converged = FALSE, n_iter = NA_integer_))
  }
  list(
    elapsed   = elapsed,
    converged = is_converged(fit),
    n_iter    = extract_n_iter(fit)
  )
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

cat(sprintf("R mlt benchmark — mlt %s, basefun %s, R %s.%s\n",
            as.character(packageVersion("mlt")),
            as.character(packageVersion("basefun")),
            R.version$major, R.version$minor))
cat(sprintf("  data dir:    %s\n", DATA_DIR))
cat(sprintf("  results csv: %s\n", RESULTS_CSV))
cat(sprintf("  grid: n=%s × order=%s × censoring=%s × reps=%d\n",
            paste(N_LIST, collapse = ","),
            paste(ORDER_LIST, collapse = ","),
            paste(CENSORING_LIST, collapse = ","),
            N_REPS))
cat("\nRunning fit() benchmarks …\n")

total_cells <- length(N_LIST) * length(ORDER_LIST) * length(CENSORING_LIST)

# Pre-allocate result columns to avoid quadratic rbind cost
total_rows <- as.integer(total_cells * N_REPS)
out <- data.frame(
  n         = integer(total_rows),
  order     = integer(total_rows),
  censoring = character(total_rows),
  rep       = integer(total_rows),
  time_s    = character(total_rows),
  converged = integer(total_rows),
  n_iter    = integer(total_rows),
  stringsAsFactors = FALSE
)

row_idx  <- 0L
cell_idx <- 0L
for (n in N_LIST) {
  for (order in ORDER_LIST) {
    for (censoring in CENSORING_LIST) {
      cell_idx <- cell_idx + 1L
      df_in    <- load_dataset(n, censoring)
      data     <- build_data(df_in, censoring)
      model    <- build_model(order)

      cell_times <- numeric(0)
      for (rep in seq_len(N_REPS) - 1L) {
        res <- time_one_fit(model, data)
        row_idx <- row_idx + 1L
        out[row_idx, ] <- list(
          n         = n,
          order     = order,
          censoring = censoring,
          rep       = rep,
          time_s    = sprintf("%.9f", res$elapsed),
          converged = as.integer(res$converged),
          n_iter    = if (is.na(res$n_iter)) NA_integer_ else res$n_iter
        )
        if (isTRUE(res$converged)) {
          cell_times <- c(cell_times, res$elapsed)
        }
      }

      med <- if (length(cell_times) > 0L) median(cell_times) else NA_real_
      cat(sprintf(
        "[%2d/%2d] n=%4d order=%d cens=%-5s median=%8.2f ms  (%d/%d converged)\n",
        cell_idx, total_cells, n, order, censoring,
        med * 1000, length(cell_times), N_REPS
      ))
      if (!is.na(med) && med < 0.010) {
        cat(sprintf(
          "    note: median below 10 ms — timing noise may dominate.\n"
        ))
      }
    }
  }
}

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

write.csv(out, RESULTS_CSV, row.names = FALSE, quote = FALSE)
cat(sprintf("\nWrote %d timing rows to %s\n", nrow(out), RESULTS_CSV))
