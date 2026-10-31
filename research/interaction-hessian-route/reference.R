# Same-theta R references for the research-only interaction Hessian candidates.
# Run from the repository root: Rscript research/interaction-hessian-route/reference.R
suppressPackageStartupMessages(library(mlt))
if (!requireNamespace("jsonlite", quietly = TRUE))
    stop("This research script requires the optional jsonlite R package")

script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
here <- dirname(normalizePath(script))
root <- normalizePath(file.path(here, "../.."))
ref <- file.path(root, "reference")
out <- file.path(here, "results")
options(digits = 17)

check_case <- function(name, distribution, scaled = FALSE) {
    prefix <- if (scaled) "stram_bs_bs_normal" else "interaction_bs_bs"
    load_fixture <- function(suffix)
        scan(file.path(ref, paste0(prefix, "_", suffix, ".txt")), quiet = TRUE)
    y <- load_fixture("y_train")
    x <- load_fixture("x_train")
    support <- load_fixture("y_support")
    data <- data.frame(y = y, x = x)
    y_basis <- Bernstein_basis(numeric_var("y", support = support),
                               order = 2L, ui = "increasing")
    x_basis <- Bernstein_basis(numeric_var("x", support = c(0, 1)), order = 2L)
    if (scaled) {
        data$x_s <- load_fixture("x_s_train")
        scale_basis <- as.basis(~ x_s, data = data, remove_intercept = TRUE)
        model <- ctm(response = y_basis, interacting = x_basis,
                     scaling = scale_basis, scale_shift = FALSE,
                     todistr = distribution)
    } else {
        model <- ctm(response = y_basis, interacting = x_basis,
                     todistr = distribution)
    }
    theta <- scan(file.path(out, paste0(name, "_theta.txt")), quiet = TRUE)
    H_candidate <- as.matrix(read.table(file.path(out, paste0(name, "_H.txt"))))
    # R varies the y index fastest; Python stores vec_C(Theta).
    perm <- as.vector(t(matrix(seq_len(9L), nrow = 3L, ncol = 3L)))
    if (scaled)
        perm <- c(perm, 10L)
    theta_R <- theta[perm]
    # Reconstruct at theta rather than refitting. The scaled R Hessian closure
    # uses its stored par in optimHess, even when given an explicit parm.
    object <- mlt(model, data = data, theta = theta_R,
                  dofit = FALSE, scaleparm = FALSE)
    H_R <- Hessian(object, parm = theta_R)[order(perm), order(perm)]
    write.table(H_R, file.path(out, paste0(name, "_R_H.txt")),
                row.names = FALSE, col.names = FALSE, quote = FALSE)
    info <- list(
        reference = if (scaled) "R optimHess of analytic gradient" else "R analytic",
        same_theta = TRUE,
        log_likelihood_R = as.numeric(logLik(object)),
        max_absolute_H_error = max(abs(H_candidate - H_R)),
        relative_max_H_error = max(abs(H_candidate - H_R)) / max(abs(H_R)),
        min_eigenvalue_R = min(eigen(H_R, symmetric = TRUE, only.values = TRUE)$values),
        analytic_allclose_rtol_1e_6_atol_1e_10 =
            all(abs(H_candidate - H_R) <= 1e-10 + 1e-6 * abs(H_R))
    )
    # Compare raw inverses only for positive-definite information; vcov.mlt's
    # singular-matrix regularization must not mask an information mismatch.
    if (info$min_eigenvalue_R > 0 &&
        min(eigen(H_candidate, symmetric = TRUE, only.values = TRUE)$values) > 0) {
        V_R <- solve(H_R)
        V_candidate <- solve(H_candidate)
        info$relative_max_vcov_error <- max(abs(V_candidate - V_R)) / max(abs(V_R))
        info$max_relative_SE_error <- max(abs(sqrt(diag(V_candidate) / diag(V_R)) - 1))
    } else {
        info$SE_status <- "undefined: information is not positive definite"
    }
    info$required_analytic_check_passed <- scaled ||
        info$analytic_allclose_rtol_1e_6_atol_1e_10
    info
}

cases <- list(
    ia_R_normal_3x3 = list(distribution = "Normal"),
    ia_R_logistic_3x3 = list(distribution = "Logistic"),
    "ia_scaled_R_stram_3x3+1" = list(distribution = "Normal", scaled = TRUE)
)
results <- lapply(names(cases), function(name) {
    tryCatch(do.call(check_case, c(list(name = name), cases[[name]])),
             error = function(e) list(error = conditionMessage(e),
                                      required_analytic_check_passed = FALSE))
})
names(results) <- names(cases)
jsonlite::write_json(list(R = R.version.string,
                         mlt = as.character(packageVersion("mlt")),
                         basefun = as.character(packageVersion("basefun")),
                         cases = results),
                    file.path(out, "r_reference.json"),
                    pretty = TRUE, auto_unbox = TRUE, digits = 17)
stopifnot(all(vapply(results, function(x) isTRUE(x$required_analytic_check_passed),
                    logical(1))))
print(results)
