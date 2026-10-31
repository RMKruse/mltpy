## Probe for GitHub issue #126:
## "Can R mlt fit an exponential-link CTM with a scaling term?"
##
## Run:  R CMD BATCH --no-save --no-restore probe.R probe.Rout
##
## Sections
##   0. versions
##   1. source evidence: .Exponential, .log, exact loglik/score, ctm() signature
##   2. constraint rows: what ui/ci R actually passes to auglag
##   3. tram-level routes: tram::tram(distribution = "Exponential", y ~ x | z),
##      tram::Survreg(dist = "exponential", y ~ x | z), tram::Aareg(... | z)
##   4. raw ctm fixture A: Bernstein(y) + shift x_d + scale x_s, y on R
##   5. raw ctm fixture B: Bernstein(y) + scale x_s only, y > 0, support [0, b]
##   6. formula / sign check: h computed by hand vs predict(type = "trafo")
##   7. what happens when h(y_i | x_i) < 0 (barrier, not a constraint)
##   8. values a fixture would store for fixture A's plain mlt() fit (stuck)
##   9. explicit-constraint fits: mlt likelihood + alabama::auglag(hin = ...),
##      under R semantics (h(y_i|x_i) >= 0) and mltpy semantics (h(a|x_i) >= 0)
##  10. fixture B: mlt(fixed = c("Bs1(y)" = 0)) and tram::Aareg routes
##  11. the existing no-covariate fixture reference/mlt_exponential_*.txt
##  12. tight-tolerance mlt() on the fixed-Bs1 route; data dump for crosscheck.py
##
## Writes fixtures.csv and fixtures_theta.csv (read by crosscheck.py).

suppressPackageStartupMessages({
  library(mlt)
  library(tram)
  library(survival)
})
options(digits = 10, width = 110)

hr <- function(txt) cat("\n", strrep("=", 78), "\n## ", txt, "\n",
                         strrep("=", 78), "\n", sep = "")

## ---------------------------------------------------------------------------
hr("0. versions")
cat(R.version.string, "\n")
for (p in c("mlt", "tram", "basefun", "variables", "alabama"))
  cat(sprintf("%-10s %s\n", p, as.character(packageVersion(p))))

## ---------------------------------------------------------------------------
hr("1. source evidence")
cat("-- mlt:::.Exponential (todistr object; note: no 'support' element)\n")
print(mlt:::.Exponential)
cat("names(.Exponential()):", names(mlt:::.Exponential()), "\n")
cat("-- dexp(-0.1, log = TRUE) =", dexp(-0.1, log = TRUE),
    "; pexp(-0.1) =", pexp(-0.1), "\n")
cat("-- mlt:::.mlt_loglik_exact\n");  print(mlt:::.mlt_loglik_exact)
cat("-- mlt:::.mlt_score_exact (dd2d = -1 for Exponential, everywhere)\n")
print(mlt:::.mlt_score_exact)
cat("-- mlt:::.log (log h'; h' < 0 -> -Inf)\n"); print(mlt:::.log)
cat("-- args(mlt::ctm)\n"); print(args(mlt::ctm))

## ---------------------------------------------------------------------------
## Data-generating process (both fixtures).
## Z ~ Exp(1); true h(y | x) = 2 * y * exp(0.5 * x_s * g0) + x_d * b0,
## i.e. h_0(y) = 2y is linear (exactly representable by any Bernstein order
## >= 1), y = (Z - x_d * b0) / (2 * exp(0.5 * x_s * g0)).
g0 <- 0.6; b0 <- 0.5

set.seed(126)
nA   <- 150
xdA  <- rnorm(nA); xsA <- rnorm(nA); zA <- rexp(nA)
yA   <- (zA - xdA * b0) / (2 * exp(0.5 * xsA * g0))
dA   <- data.frame(y = yA, x_d = xdA, x_s = xsA)
supA <- range(yA)

set.seed(1126)
nB   <- 150
xsB  <- rnorm(nB); zB <- rexp(nB)
yB   <- zB / (2 * exp(0.5 * xsB * g0))
dB   <- data.frame(y = yB, x_s = xsB)
supB <- c(0, max(yB))

order_y <- 4L

## ---------------------------------------------------------------------------
hr("2. constraint rows that reach the optimiser")
yvA  <- numeric_var("y", support = supA)
bsA  <- Bernstein_basis(yvA, order = order_y, ui = "increasing")
mA   <- ctm(bsA, shifting = ~ x_d, scaling = ~ x_s, data = dA,
            todistr = "Exponential")
cat("ctm() accepted scaling + todistr = 'Exponential'; class:", class(mA), "\n")
cat("coef names (ctm order):", names(coef(mA)), "\n")
cat("model$todistr$name:", mA$todistr$name, "; scale_shift:", mA$scale_shift, "\n")
s0  <- mlt(mA, data = dA, dofit = FALSE)
YA  <- model.matrix(mA, data = dA)
uiA <- attr(YA, "constraint")$ui; ciA <- attr(YA, "constraint")$ci
cat("ui (rows passed to alabama::auglag as hin = ui %*% par - ci):\n")
print(as(uiA, "matrix")); cat("ci:", ciA, "\n")
cat("=> only the Bernstein 'increasing' rows; no row on theta_b[0], none on\n",
    "   theta_b[0]*exp(0.5 x_s g) + x_d b, nothing involving todistr.\n")

## ---------------------------------------------------------------------------
hr("3. tram-level routes")
cat("-- 3a tram::tram(y ~ x_d | x_s, distribution = 'Exponential')\n")
fT <- try(tram::tram(y ~ x_d | x_s, data = dA, distribution = "Exponential",
                     transformation = "smooth", order = order_y,
                     support = supA, add = c(0, 0)))
if (!inherits(fT, "try-error")) {
  cat("class:", class(fT), "\n")
  print(coef(as.mlt(fT)))
  cat("logLik:", as.numeric(logLik(fT)), " convergence:",
      as.mlt(fT)$convergence, "\n")
}
cat("\n-- 3b tram::Survreg(Surv(y) ~ x_d | x_s, dist = 'exponential'), y > 0\n")
dS <- data.frame(y = yB, x_d = rnorm(nB), x_s = xsB)
fS <- try(tram::Survreg(Surv(y) ~ x_d | x_s, data = dS, dist = "exponential"))
if (!inherits(fS, "try-error")) {
  cat("class:", class(fS), "\n")
  cat("todistr used by Survreg(dist='exponential'):",
      as.mlt(fS)$model$todistr$name, "\n")
  print(coef(as.mlt(fS)))
  cat("fixed:", names(as.mlt(fS)$fixed), "=", as.mlt(fS)$fixed, "\n")
}
cat("\n-- 3c tram::Aareg(y ~ 0 | x_s)  (Aareg is the only tram class with todistr = Exponential)\n")
fAa <- try(tram::Aareg(y ~ 0 | x_s, data = dS))
if (!inherits(fAa, "try-error")) {
  cat("class:", class(fAa), "\n"); print(coef(as.mlt(fAa)))
  cat("todistr:", as.mlt(fAa)$model$todistr$name, "; convergence:",
      as.mlt(fAa)$convergence, "; logLik:", as.numeric(logLik(fAa)), "\n")
  cat("fixed:", names(as.mlt(fAa)$fixed), "=", as.mlt(fAa)$fixed, "\n")
  cat("response support:", support(as.vars(as.mlt(fAa)$model$bases$response))$y, "\n")
  cat("Aareg(y ~ x_d | x_s) (shift term):\n")
  print(try(tram::Aareg(y ~ x_d | x_s, data = dS)))
}

## ---------------------------------------------------------------------------
## helpers
hfun <- function(fit, data, at = NULL) {
  ## h(y|x) by hand from coefficients: theta_b' a(y) * exp(0.5 x_s g) + x_d b
  cf <- coef(fit)
  ynm <- if (is.null(at)) data$y else rep(at, nrow(data))
  B  <- model.matrix(fit$model$bases$response, data = data.frame(y = ynm))
  tb <- cf[grepl("^Bs", names(cf))]
  g  <- cf[grepl("^scl_", names(cf))]
  b  <- cf[!grepl("^Bs|^scl_", names(cf))]
  s  <- exp(0.5 * drop(as.matrix(data[, sub("^scl_", "", names(g)), drop = FALSE]) %*% g))
  sh <- if (length(b)) drop(as.matrix(data[, sub("^sft_", "", names(b)), drop = FALSE]) %*% b) else 0
  drop(B %*% tb) * s + sh
}
`%||%` <- function(a, b) if (is.null(a)) b else a

report <- function(fit, data, a, label) {
  cat("\n--", label, "\n")
  cat("convergence:", fit$convergence, "\n")
  print(coef(fit))
  cat("logLik:", format(as.numeric(logLik(fit)), digits = 15), "\n")
  h_obs <- hfun(fit, data)
  h_low <- hfun(fit, data, at = a)
  cat(sprintf("min_i h(y_i | x_i)          = % .6e\n", min(h_obs)))
  cat(sprintf("min_i h(a | x_i), a=%8.5f = % .6e   (#rows < 0: %d of %d)\n",
              a, min(h_low), sum(h_low < 0), length(h_low)))
  invisible(list(h_obs = h_obs, h_low = h_low))
}

## ---------------------------------------------------------------------------
hr("4. raw ctm, fixture A: Bernstein(y) + shift x_d + scale x_s (y on R)")
cat("n =", nA, "; support =", supA, "; order =", order_y, "\n")
cat("-- 4a default start (mlt's .mlt_start, no theta supplied)\n")
th_def <- try(mlt:::.mlt_start(mA, data = dA, y = R(dA$y),
                               pstart = mlt:::pstart(dA$y), offset = rep(0, nA)))
print(th_def)
fA_def <- try(mlt(mA, data = dA))
if (!inherits(fA_def, "try-error")) rA_def <- report(fA_def, dA, supA[1], "4a fit")

cat("\n-- 4b supplied feasible start: theta_b = seq(0.1, 5, length = 5), b = g = 0\n")
thA <- c(seq(0.1, 5, length.out = order_y + 1), 0, 0)
names(thA) <- names(coef(mA))
cat("start loglik:", sum(mlt(mA, data = dA, theta = thA, dofit = FALSE)$logliki(thA, rep(1, nA))),
    "\n")
fA <- mlt(mA, data = dA, theta = thA)
rA <- report(fA, dA, supA[1], "4b fit")
cat("True (DGP) theta for comparison: theta_b = 2*a + 2*(b-a)*k/4, b0 =", b0,
    ", g0 =", g0, "\n")
print(2 * supA[1] + 2 * diff(supA) * (0:order_y) / order_y)
cat("logLik at DGP theta:",
    sum(fA$logliki(c(2 * supA[1] + 2 * diff(supA) * (0:order_y) / order_y,
                     b0, g0), rep(1, nA))), "\n")

cat("\n-- 4c tram::tram on the same data (negative = TRUE): logLik equal, b flips\n")
if (!inherits(fT, "try-error")) {
  cat("raw ctm logLik:", format(as.numeric(logLik(fA)), digits = 15),
      "  tram logLik:", format(as.numeric(logLik(fT)), digits = 15), "\n")
  cat("raw coef:  ", format(coef(fA), digits = 8), "\n")
  cat("tram coef: ", format(coef(as.mlt(fT)), digits = 8), "\n")
}

cat("\n-- 4d raw ctm with shift only (no scaling), same data, for contrast\n")
mA0 <- ctm(bsA, shifting = ~ x_d, data = dA, todistr = "Exponential")
fA0 <- mlt(mA0, data = dA, theta = c(seq(0.1, 5, length.out = order_y + 1), 0))
cat("coef names:", names(coef(fA0)), "\n")
rA0 <- report(fA0, dA, supA[1], "4d fit (shift only)")

## ---------------------------------------------------------------------------
hr("5. raw ctm, fixture B: Bernstein(y) + scale x_s only (y > 0, support [0, b])")
yvB <- numeric_var("y", support = supB)
bsB <- Bernstein_basis(yvB, order = order_y, ui = "increasing")
mB  <- ctm(bsB, scaling = ~ x_s, data = dB, todistr = "Exponential")
cat("coef names:", names(coef(mB)), "\n")
cat("-- 5a default start\n")
fB_def <- try(mlt(mB, data = dB))
if (!inherits(fB_def, "try-error")) rB_def <- report(fB_def, dB, supB[1], "5a fit")
cat("\n-- 5b supplied start\n")
fB <- mlt(mB, data = dB, theta = c(seq(0.1, 5, length.out = order_y + 1), 0))
rB <- report(fB, dB, supB[1], "5b fit")
cat("theta_b[0] =", coef(fB)[1], " (h(0|x) = theta_b[0] * exp(0.5 x_s g))\n")
cat("min(y) =", min(yB), "\n")

## ---------------------------------------------------------------------------
hr("6. formula and sign check against predict(type = 'trafo')")
nd <- dA[1:5, ]
hp <- predict(fA, newdata = nd, type = "trafo")
cat("predict(type='trafo') is a y-grid x row matrix; take diag at y = nd$y:\n")
hp_obs <- sapply(1:5, function(i) predict(fA, newdata = nd[i, ], q = nd$y[i], type = "trafo"))
print(cbind(by_hand = hfun(fA, nd), predict = hp_obs))
cat("max |diff| =", max(abs(hfun(fA, nd) - hp_obs)), "\n")
cat("=> raw ctm: h = theta_b'a(y) * exp(0.5 * x_s * g) + x_d * b  ('+' on shift)\n")
cat("by-hand logLik sum(log dexp(h) + log h'):\n")
Bp  <- model.matrix(bsA, data = data.frame(y = dA$y), deriv = c(y = 1L))
hpr <- drop(Bp %*% coef(fA)[1:(order_y + 1)]) * exp(0.5 * dA$x_s * coef(fA)["scl_x_s"])
cat(format(sum(dexp(rA$h_obs, log = TRUE) + log(hpr)), digits = 15),
    " vs logLik:", format(as.numeric(logLik(fA)), digits = 15), "\n")

## ---------------------------------------------------------------------------
hr("7. the support condition is a likelihood barrier, not a constraint")
th <- coef(fA)
shift_down <- th; shift_down["Bs1(y)"] <- th["Bs1(y)"] - 0.05
shift_down[2:(order_y + 1)] <- th[2:(order_y + 1)] - 0.05  # parallel shift of h_0
cat("logLik at fitted theta     :", sum(fA$logliki(th, rep(1, nA))), "\n")
cat("logLik with h_0 shifted -0.05:", sum(fA$logliki(shift_down, rep(1, nA))), "\n")
cat("score at shifted theta (finite? smooth extension dd2d = -1):\n")
print(colSums(fA$scorei(shift_down, rep(1, nA))))
cat("rows whose h(y_i|x_i) is within 1e-4 of 0 at the optimum:",
    sum(rA$h_obs < 1e-4), "\n")
cat("argmin row: y =", dA$y[which.min(rA$h_obs)], ", x_d =",
    dA$x_d[which.min(rA$h_obs)], ", x_s =", dA$x_s[which.min(rA$h_obs)], "\n")

## ---------------------------------------------------------------------------
hr("8. fixture-generator sketch: values a parity test would read")
cat("theta (R raw ctm order, = mltpy [theta_b | beta | gamma], no sign flip):\n")
cat(format(coef(fA), digits = 15), sep = "\n")
cat("logLik:", format(as.numeric(logLik(fA)), digits = 15), "\n")
cat("Hessian-based vcov available at the (stuck) mlt() optimum? ")
v <- try(vcov(fA)); cat(!inherits(v, "try-error"), "\n")
if (!inherits(v, "try-error")) print(sqrt(diag(v)))

## ---------------------------------------------------------------------------
hr("9. explicit-constraint fits: mlt's likelihood + alabama::auglag with hin")
## Objective: -sum(log f_Z(h_i) + log h'_i), with log f_Z(h) = -h extended to
## the whole line (finite everywhere; equals mlt's logLik wherever h_i >= 0,
## checked in section 6).  Gradient by hand, checked against mlt's own score.
library(alabama)
p   <- order_y + 1L
BA  <- model.matrix(bsA, data = data.frame(y = dA$y))
BpA <- model.matrix(bsA, data = data.frame(y = dA$y), deriv = c(y = 1L))
aA  <- drop(model.matrix(bsA, data = data.frame(y = supA[1])))  # a(y = a) = e_1
cat("a(lower support end) =", aA, "\n")
Dm  <- diff(diag(p))
mk <- function(B, Bp, xd, xs) {
  parts <- function(th) {
    tb <- th[1:p]; b <- if (is.null(xd)) 0 else th[p + 1]; g <- th[length(th)]
    s <- exp(0.5 * xs * g)
    h0 <- drop(B %*% tb); h0p <- drop(Bp %*% tb)
    list(tb = tb, b = b, g = g, s = s, h0 = h0, h0p = h0p,
         h = h0 * s + (if (is.null(xd)) 0 else xd * b), hp = h0p * s,
         hlow = tb[1] * s + (if (is.null(xd)) 0 else xd * b))
  }
  nll <- function(th) { q <- parts(th); sum(q$h) - sum(log(q$hp)) }
  grad <- function(th) {
    q <- parts(th)
    gb <- colSums(B * q$s) - colSums(Bp / q$h0p)
    gg <- sum(0.5 * xs * q$s * q$h0) - sum(0.5 * xs)
    c(gb, if (is.null(xd)) NULL else sum(xd), gg)
  }
  ## R semantics: h(y_i | x_i) >= 0 at every observation (what dexp's -Inf enforces)
  hin_obs <- function(th) { q <- parts(th); c(Dm %*% q$tb, q$h) }
  jac_obs <- function(th) {
    q <- parts(th)
    J <- cbind(B * q$s, if (is.null(xd)) NULL else xd, 0.5 * xs * q$s * q$h0)
    rbind(cbind(Dm, matrix(0, p - 1, length(th) - p)), J)
  }
  ## mltpy semantics: h(a | x_i) = theta_b[0] * exp(0.5 x_s g) + x_d b >= 0
  hin_low <- function(th) { q <- parts(th); c(Dm %*% q$tb, q$hlow) }
  jac_low <- function(th) {
    q <- parts(th)
    J <- cbind(q$s, matrix(0, length(q$s), p - 1),
               if (is.null(xd)) NULL else xd, 0.5 * xs * q$s * q$tb[1])
    rbind(cbind(Dm, matrix(0, p - 1, length(th) - p)), J)
  }
  list(parts = parts, nll = nll, grad = grad, hin_obs = hin_obs,
       jac_obs = jac_obs, hin_low = hin_low, jac_low = jac_low)
}
oA <- mk(BA, BpA, dA$x_d, dA$x_s)
th0 <- c(seq(0.1, 5, length.out = p), 0, 0)
cat("objective check vs mlt logLik at the section-4b theta:",
    format(-oA$nll(coef(fA)), digits = 15), "vs",
    format(as.numeric(logLik(fA)), digits = 15), "\n")
cat("gradient check vs -colSums(mlt score) at th0: max |diff| =",
    max(abs(oA$grad(th0) + colSums(fA$scorei(th0)))), "\n")

fit_al <- function(o, th, which) {
  hin <- if (which == "obs") o$hin_obs else o$hin_low
  jac <- if (which == "obs") o$jac_obs else o$jac_low
  auglag(par = th, fn = o$nll, gr = o$grad, hin = hin, hin.jac = jac,
         control.outer = list(trace = FALSE, eps = 1e-9, itmax = 200,
                              kkt2.check = FALSE),
         control.optim = list(maxit = 2000))
}
show_al <- function(r, o, label, nms) {
  q <- o$parts(r$par)
  cat("\n--", label, "\n")
  cat("convergence:", r$convergence, "; outer iters:", r$outer.iterations,
      "; kkt1:", r$kkt1, "\n")
  names(r$par) <- nms; print(r$par)
  cat("logLik:", format(-r$value, digits = 15), "\n")
  cat(sprintf("min_i h(y_i|x_i) = % .3e ; min_i h(a|x_i) = % .3e (#<0: %d)\n",
              min(q$h), min(q$hlow), sum(q$hlow < -1e-8)))
  r$par <- setNames(r$par, nms)
  invisible(r)
}
nmA <- names(coef(mA))
thA_dgp <- c(2 * supA[1] + 2 * diff(supA) * (0:order_y) / order_y, b0, g0)
cat("\n### fixture A (shift + scale)\n")
alA_obs <- show_al(fit_al(oA, th0, "obs"), oA, "A / R semantics  h(y_i|x_i) >= 0", nmA)
alA_low <- show_al(fit_al(oA, th0, "low"), oA, "A / mltpy semantics h(a|x_i) >= 0", nmA)
cat("\nmlt() (section 4b) logLik:", as.numeric(logLik(fA)),
    "  DGP-theta logLik:", -oA$nll(thA_dgp), "\n")

cat("\n-- multi-start stability (5 random feasible starts each)\n")
set.seed(9)
starts <- lapply(1:5, function(k) c(sort(runif(p, 0.5, 6)), rnorm(1, 0, 0.1),
                                     rnorm(1, 0, 0.3)))
ms <- function(which) {
  hin <- if (which == "obs") oA$hin_obs else oA$hin_low
  t(sapply(starts, function(st) {
    if (any(hin(st) < 0)) st <- th0
    r <- fit_al(oA, st, which); c(r$par, ll = -r$value, conv = r$convergence)
  }))
}
for (w in c("low", "obs")) {
  res <- ms(w)
  cat(if (w == "low") "mltpy semantics:\n" else "R semantics:\n"); print(res)
  cat("max |dtheta| across starts:",
      max(apply(res[, 1:(p + 2)], 2, function(z) diff(range(z)))), "\n")
}

cat("\n-- mlt() warm-started at the explicit-constraint optimum (R semantics)\n")
fA_warm <- mlt(mA, data = dA, theta = alA_obs$par)
report(fA_warm, dA, supA[1], "mlt() warm start")
cat("max |coef(mlt warm) - alabama R-semantics| =",
    max(abs(coef(fA_warm) - alA_obs$par)), "\n")
v <- try(vcov(fA_warm)); if (!inherits(v, "try-error")) { cat("SEs:\n"); print(sqrt(diag(v))) }

## ---------------------------------------------------------------------------
hr("10. fixture B (scale only): theta_b[0] = 0 route, no barrier involved")
BB  <- model.matrix(bsB, data = data.frame(y = dB$y))
BpB <- model.matrix(bsB, data = data.frame(y = dB$y), deriv = c(y = 1L))
oB  <- mk(BB, BpB, NULL, dB$x_s)
thB0 <- c(seq(0.1, 5, length.out = p), 0)
nmB <- names(coef(mB))
alB_low <- show_al(fit_al(oB, thB0, "low"), oB, "B / mltpy semantics theta_b[0] * s_i >= 0", nmB)
alB_obs <- show_al(fit_al(oB, thB0, "obs"), oB, "B / R semantics h(y_i|x_i) >= 0", nmB)
cat("\n-- raw mlt() with fixed = c('Bs1(y)' = 0), DEFAULT start\n")
fBf <- mlt(mB, data = dB, fixed = c("Bs1(y)" = 0))
report(fBf, dB, supB[1], "mlt(fixed Bs1 = 0), default start")
cat("max |coef(mlt fixed) - alabama mltpy-semantics| =",
    max(abs(coef(fBf) - alB_low$par)), "\n")
cat("logLik diff (mlt fixed - alabama mltpy-semantics) =",
    as.numeric(logLik(fBf)) + alB_low$value, "\n")
v <- try(vcov(fBf)); if (!inherits(v, "try-error")) { cat("SEs:\n"); print(sqrt(diag(v))) }

cat("\n-- Aareg on fixture B with the same order: compare with raw ctm on Aareg's support\n")
fAaB <- tram::Aareg(y ~ 0 | x_s, data = dB, order = order_y)
supAa <- support(as.vars(as.mlt(fAaB)$model$bases$response))$y
cat("Aareg support:", format(supAa, digits = 15), "; convergence:",
    as.mlt(fAaB)$convergence, "; logLik:",
    format(as.numeric(logLik(fAaB)), digits = 15), "\n")
print(coef(as.mlt(fAaB)))
bsAa <- Bernstein_basis(numeric_var("y", support = supAa), order = order_y,
                        ui = "increasing")
fRaw <- mlt(ctm(bsAa, scaling = ~ x_s, data = dB, todistr = "Exponential"),
            data = dB, fixed = c("Bs1(y)" = 0))
cat("raw ctm on Aareg support, fixed Bs1 = 0: logLik",
    format(as.numeric(logLik(fRaw)), digits = 15), "\n")
cat("max |coef diff| Aareg vs raw:", max(abs(coef(fRaw) - coef(as.mlt(fAaB)))), "\n")

cat("\n-- fixture B values a fixture would store\n")
cat("y support:", format(supB, digits = 15), "\n")
cat("theta [theta_b | gamma]:\n"); cat(format(coef(fBf), digits = 15), sep = "\n")
cat("logLik:", format(as.numeric(logLik(fBf)), digits = 15), "\n")

## ---------------------------------------------------------------------------
hr("11. existing fixture reference/mlt_exponential_*.txt (no covariates)")
yE <- scan("../../reference/mlt_exponential_y.txt", quiet = TRUE)
thE_R <- scan("../../reference/mlt_exponential_theta.txt", quiet = TRUE)
llE_R <- scan("../../reference/mlt_exponential_ll.txt", quiet = TRUE)
bsE <- Bernstein_basis(numeric_var("y", support = c(0, 1), bounds = c(0, 1)),
                       order = 4, ui = "increasing")
BE  <- model.matrix(bsE, data = data.frame(y = yE))
BpE <- model.matrix(bsE, data = data.frame(y = yE), deriv = c(y = 1L))
## no covariates: reuse mk() with a zero scaling column and gamma pinned at 0
oE <- mk(BE, BpE, NULL, rep(0, length(yE)))
cat("stored theta:", thE_R, "\nstored logLik:", llE_R, "\n")
cat("min_i h(y_i) at stored theta:", min(drop(BE %*% thE_R)), "\n")
for (w in c("obs", "low")) {
  r <- fit_al(oE, c(seq(0.1, 5, length.out = 5), 0), w)
  cat(sprintf("\nexplicit-constraint fit (%s semantics): conv %d, logLik %.10f\n",
              if (w == "obs") "R" else "mltpy", r$convergence, -r$value))
  print(r$par[1:5])
}

## ---------------------------------------------------------------------------
hr("12. tight-tolerance mlt() on fixture B fixed-Bs1 route; data dump")
fBt <- mlt(mB, data = dB, fixed = c("Bs1(y)" = 0),
           optim = mltoptim(abstol = 1e-12, reltol = 1e-12, hessian = TRUE))
cat("convergence:", fBt$convergence, "; logLik:",
    format(as.numeric(logLik(fBt)), digits = 15), "\n")
print(coef(fBt))
alBt <- auglag(par = thB0[-1], fn = function(t) oB$nll(c(0, t)),
               gr = function(t) oB$grad(c(0, t))[-1],
               hin = function(t) diff(c(0, t[1:(p - 1)])),
               hin.jac = function(t) cbind(diff(diag(p))[, -1], 0),
               control.outer = list(trace = FALSE, eps = 1e-12, itmax = 500,
                                    kkt2.check = FALSE),
               control.optim = list(maxit = 5000, reltol = 1e-14))
cat("alabama, theta_b[0] pinned at 0, eps 1e-12: logLik",
    format(-alBt$value, digits = 15), "\n")
print(c(0, alBt$par))
cat("max |coef(mlt tight) - alabama tight| =",
    max(abs(coef(fBt) - c(0, alBt$par))), "\n")
cat("max |coef(mlt default) - mlt tight|  =", max(abs(coef(fBf) - coef(fBt))), "\n")

write.csv(rbind(data.frame(fixture = "A", y = dA$y, x_d = dA$x_d, x_s = dA$x_s),
                data.frame(fixture = "B", y = dB$y, x_d = 0, x_s = dB$x_s)),
          "fixtures.csv", row.names = FALSE)
write.csv(data.frame(
  fit = c(rep("A_R_semantics", p + 2), rep("A_mltpy_semantics", p + 2),
          rep("B_fixed_Bs1", p + 1)),
  name = c(nmA, nmA, nmB),
  value = c(alA_obs$par, alA_low$par, coef(fBt)),
  loglik = c(rep(-alA_obs$value, p + 2), rep(-alA_low$value, p + 2),
             rep(as.numeric(logLik(fBt)), p + 1)),
  a = c(rep(supA[1], 2 * (p + 2)), rep(supB[1], p + 1)),
  b = c(rep(supA[2], 2 * (p + 2)), rep(supB[2], p + 1))),
  "fixtures_theta.csv", row.names = FALSE)
cat("wrote fixtures.csv and fixtures_theta.csv\n")
