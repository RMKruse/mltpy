# verify_strata_scale_support.R
#
# R-support verification for ADR 0003 (issue #102): does R `tram` expose the
# combination of an interacting/stratified baseline with a scale term?
#
# Finding (tram 1.4.1 / mlt 1.7.4): SUPPORTED via the formula  y | s ~ x | z
#   - s (2nd LHS part) -> interacting/strata basis   (ctm `interacting=`)
#   - x (1st RHS part) -> shift terms                (ctm `shifting=`)
#   - z (2nd RHS part) -> scale terms                (ctm `scaling=`, prefix scl_)
#   - emits warning("Models with both strata and scale terms are highly experimental")
#   - errors if a scale variable is also a strata variable
#   - returns an object of class `stram`; scale coef extractable via
#     coef(fit, with_baseline = FALSE)[fit$scalecoef]
#
# mlt::ctm doc (scale_shift = FALSE, the default) gives the transformation:
#   P(Y<=y|x) = F_Z( sqrt(exp(s(x)'gamma)) * [(a(y) (x) b(x))' theta] + d(x)'beta )
# and sqrt(exp(.)) = exp(0.5 * .) confirms the ADR 0002 0.5-in-exponent convention.
#
# mltpy's InteractionBasis path carries no additive d(x)'beta shift block, so the
# relevant analogue is the no-shift form  Surv | s ~ 1 | z  (or  y | s ~ 1 | z ).

suppressMessages({library(tram); library(survival)})
cat("tram", as.character(packageVersion("tram")),
    "/ mlt", as.character(packageVersion("mlt")), "\n\n")

set.seed(1)
n  <- 300
z  <- rnorm(n)                               # scale covariate
s  <- factor(sample(c("a", "b"), n, TRUE))   # stratifying factor
x  <- rnorm(n)                               # shift covariate
ti <- pmax(rexp(n, rate = exp(0.5 * x)), 1e-3)
ev <- rbinom(n, 1, 0.85)
d  <- data.frame(time = ti, event = ev, y = ti + 0.5, z = z, s = s, x = x)

run <- function(label, expr) {
  cat("=====", label, "=====\n")
  w <- NULL
  r <- withCallingHandlers(
    tryCatch(expr, error = function(e) {
      cat("ERROR:", conditionMessage(e), "\n"); NULL
    }),
    warning = function(wn) {
      w <<- c(w, conditionMessage(wn)); invokeRestart("muffleWarning")
    })
  for (m in unique(w)) cat("WARN:", m, "\n")
  if (!is.null(r)) {
    cat("class:", paste(class(r), collapse = ","), "\n")
    print(coef(r, with_baseline = FALSE))
    cat("logLik:", as.numeric(logLik(r)), "\n")
    cat("scalecoef:", r$scalecoef, "| stratacoef:", r$stratacoef,
        "| shiftcoef:", r$shiftcoef, "\n")
  }
  cat("\n"); invisible(r)
}

run("BoxCox strata+scale: y | s ~ x | z",            BoxCox(y | s ~ x | z, data = d))
run("Coxph strata+scale: Surv | s ~ x | z",          Coxph(Surv(time, event) | s ~ x | z, data = d))
run("Coxph strata+scale, no shift: Surv | s ~ 1 | z", Coxph(Surv(time, event) | s ~ 1 | z, data = d))
run("ERROR CASE: scale var == strata var",           BoxCox(y | z ~ x | z, data = d))
