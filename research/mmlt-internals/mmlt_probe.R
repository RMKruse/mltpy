## mmlt_probe.R -- small experiments confirming how mlt::mmlt / tram::Mmlt behave.
## Run: Rscript research/mmlt-internals/mmlt_probe.R > research/mmlt-internals/mmlt_probe.Rout 2>&1
## Read alongside README.md in this directory.

suppressPackageStartupMessages({
  library("tram")      # attaches mlt, basefun, variables, mvtnorm
  library("survival")
  library("numDeriv")
  library("sandwich")  # estfun() generic
})
cat("R:", R.version.string, "\n")
for (p in c("tram", "mlt", "mvtnorm", "basefun", "variables", "alabama"))
  cat(sprintf("%-10s %s\n", p, as.character(packageVersion(p))))
cat("mmlt lives in namespace:", environmentName(environment(mmlt)), "\n")
cat("Mmlt lives in namespace:", environmentName(environment(Mmlt)), "\n")

## ---------------------------------------------------------------------------
## 1. Bivariate continuous data from a Gaussian copula with non-normal margins
## ---------------------------------------------------------------------------
set.seed(29)
N <- 300
rho <- 0.6
Z <- cbind(rnorm(N), rnorm(N))
Z[, 2] <- rho * Z[, 1] + sqrt(1 - rho^2) * Z[, 2]
d <- data.frame(y1 = exp(Z[, 1]),                 # log-normal margin
                y2 = qgamma(pnorm(Z[, 2]), 2))    # gamma margin
d$x <- runif(N)

m1 <- BoxCox(y1 ~ 1, data = d, order = 4)
m2 <- BoxCox(y2 ~ 1, data = d, order = 4)

cat("\n== 1a. default fit (conditional = FALSE) ==\n")
mm <- mmlt(m1, m2, data = d)
print(class(mm))
print(coef(mm))                                  # all parameters, names
cat("convergence:", mm$convergence, "\n")
cat("logLik:", format(logLik(mm), digits = 12), "\n")

cat("\n-- coef types --\n")
for (ty in c("Lambda", "Omega", "Lambdainv", "Omegainv", "Precision",
             "PartialCorr", "Sigma", "Corr", "Spearman", "Kendall")) {
  cat(ty, ":\n"); print(coef(mm, type = ty))
}
str(coef(mm, type = "marginal"))

cat("\n-- verify Corr from Lambda by hand (Omega = Lambda diag(sqrt(diag(Lambda^-1 Lambda^-T)))) --\n")
lam <- coef(mm)["y2.y1.(Intercept)"]
L <- matrix(c(1, lam, 0, 1), 2)                  # unit lower triangular Lambda
S <- solve(L) %*% t(solve(L))
cat("Sigma from Lambda:", S, "\n")
cat("cov2cor:", cov2cor(S)[2, 1], " coef(type='Corr'):",
    as.array(coef(mm, type = "Corr"))[2, 1, 1], "\n")
cat("closed form -lam/sqrt(1+lam^2):", -lam / sqrt(1 + lam^2), "\n")
r <- as.array(coef(mm, type = "Corr"))[2, 1, 1]
cat("Spearman 6/pi asin(r/2):", 6 / pi * asin(r / 2),
    " Kendall 2/pi asin(r):", 2 / pi * asin(r), "\n")

cat("\n== 1b. conditional = TRUE: same likelihood, different parameters ==\n")
mc <- mmlt(m1, m2, data = d, conditional = TRUE)
print(class(mc))
print(coef(mc))
cat("logLik(cond) - logLik(marg):", logLik(mc) - logLik(mm), "\n")
cat("Corr (cond):", as.array(coef(mc, type = "Corr"))[2, 1, 1], "\n")

cat("\n== 1c. analytic score vs numerical gradient ==\n")
s_an <- -colSums(estfun(mm))
s_nd <- grad(function(p) logLik(mm, parm = p), coef(mm, fixed = FALSE))
print(rbind(analytic = s_an, numDeriv = s_nd))
s_an0 <- -colSums(estfun(mm, parm = coef(mm, fixed = FALSE) + 0.05))
s_nd0 <- grad(function(p) logLik(mm, parm = p), coef(mm, fixed = FALSE) + 0.05)
cat("off-optimum max |analytic - numDeriv|:", max(abs(s_an0 - s_nd0)), "\n")

cat("\n== 1d. vcov (inverse of optimiser Hessian) ==\n")
cat("optim_hessian present:", !is.null(mm$optim_hessian), "\n")
V <- vcov(mm)
print(sqrt(diag(V)))
H_nd <- hessian(function(p) -logLik(mm, parm = p), coef(mm, fixed = FALSE))
cat("max |optim_hessian - numDeriv::hessian(-logLik)| / max|H|:",
    max(abs(mm$optim_hessian - H_nd)) / max(abs(H_nd)), "\n")
cat("monotonicity slacks ui %*% theta - ci (0 = active constraint):\n")
print(round(c(mm$ui %*% coef(mm)[1:10] - mm$ci), 6))
cat("SE from vcov() vs SE from solve(numDeriv Hessian of -logLik):\n")
print(rbind(vcov = sqrt(diag(V)),
            numDeriv = sqrt(abs(diag(solve(H_nd))))))
## same comparison for a fit without active constraints (lambda only, margins fixed)
mp0 <- Mmlt(m1, m2, data = d, fit = "pseudo")
cat("pseudo fit: free params:", names(coef(mp0, fixed = FALSE)),
    " vcov SE:", sqrt(diag(vcov(mp0))),
    " numDeriv SE:", sqrt(1 / hessian(function(p) -logLik(mp0, parm = p),
                                       coef(mp0, fixed = FALSE))), "\n")

cat("\n== 1e. starting values / two-stage ('pseudo') vs joint ML ==\n")
cat("marginal-fit coefs:\n"); print(c(coef(as.mlt(m1)), coef(as.mlt(m2))))
cat("joint-fit marginal coefs:\n"); print(coef(mm)[1:10])
mp <- Mmlt(m1, m2, data = d, fit = "pseudo")
cat("pseudo logLik:", logLik(mp), " jointML logLik:", logLik(mm), "\n")
cat("pseudo lambda:", coef(mp)["y2.y1.(Intercept)"],
    " jointML lambda:", coef(mm)["y2.y1.(Intercept)"], "\n")
cat("pseudo keeps margins fixed:",
    isTRUE(all.equal(unname(coef(mp)[1:10]),
                     unname(c(coef(as.mlt(m1)), coef(as.mlt(m2)))))), "\n")
cat("analytic start for lambda (continuous, formula = ~1): solve(t(chol(S)))",
    "of the marginal z's, row-rescaled to unit diagonal:\n")
Zh <- cbind(predict(as.mlt(m1), newdata = d, type = "trafo"),
            predict(as.mlt(m2), newdata = d, type = "trafo"))
S <- var(Zh) * (N - 1) / N
Ls <- solve(t(chol(S))); Ls <- Ls %*% diag(1 / diag(Ls))
cat("start lambda21 =", Ls[2, 1], "\n")
cat("ACS (ACSiter = 1) logLik:", logLik(Mmlt(m1, m2, data = d, fit = "ACS", ACSiter = 1)), "\n")
cat("independence LR: logLik(m1)+logLik(m2) =", logLik(m1) + logLik(m2), "\n")
mI <- mmlt(m1, m2, data = d, fixed = c("y2.y1.(Intercept)" = 0))
cat("fixed lambda = 0 logLik:", logLik(mI), "\n")

cat("\n== 1f. predict (deterministic vs MC) ==\n")
nd <- d[1:3, ]
cat("density (closed form):\n"); print(predict(mm, newdata = nd, type = "density"))
cat("trafo (Omega %*% z):\n"); print(predict(mm, newdata = nd, type = "trafo"))
cat("joint distribution (lpmvnorm, args = object$args):\n")
print(str(mm$args))
p_a <- predict(mm, newdata = nd, type = "distribution")
p_b <- predict(mm, newdata = nd, type = "distribution", args = list(seed = 2, M = 1000))
p_c <- predict(mm, newdata = nd, type = "distribution", args = list(seed = 2, M = 10))
print(rbind(seed1_M1000 = p_a, seed2_M1000 = p_b, seed2_M10 = p_c))
## deterministic reference: bivariate normal CDF via TVPACK on the latent scale
## (margins = j returns the marginal transformation h_j(y_j) at the joint coefs)
z1 <- c(predict(mm, newdata = nd, margins = 1, type = "trafo"))
z2 <- c(predict(mm, newdata = nd, margins = 2, type = "trafo"))
R <- as.array(coef(mm, type = "Corr"))[, , 1]
p_tv <- sapply(1:3, function(i)
  pmvnorm(upper = c(z1[i], z2[i]), corr = R, algorithm = TVPACK()))
cat("TVPACK (deterministic):", p_tv, "\n")
cat("max |MC(seed1,M1000) - TVPACK|:", max(abs(p_a - p_tv)), "\n")
cat("marginal CDF margin 1 (= pnorm(trafo), no MC):\n")
print(cbind(predict = c(predict(mm, newdata = nd, margins = 1, type = "distribution")),
            pnorm_z1 = pnorm(z1)))

cat("\n== 1g. simulate (grid inversion, K = 50) ==\n")
s1 <- simulate(mm, newdata = d[1:3, ], seed = 1)
print(s1)

## ---------------------------------------------------------------------------
## 2. Covariate-dependent dependence: formula = ~ x
## ---------------------------------------------------------------------------
cat("\n== 2. formula = ~ x (lambda(x) = alpha + x gamma) ==\n")
mx <- mmlt(m1, m2, formula = ~ x, data = d)
print(coef(mx)[-(1:10)])
cat("Corr at x = 0, 0.5, 1:\n")
print(as.array(coef(mx, newdata = data.frame(x = c(0, .5, 1)), type = "Corr"))[2, 1, ])
cat("coef(type='Corr') without newdata errors:",
    inherits(try(coef(mx, type = "Corr"), silent = TRUE), "try-error"), "\n")

## ---------------------------------------------------------------------------
## 3. Mixed: continuous + right-censored + ordinal (Polr probit)
## ---------------------------------------------------------------------------
cat("\n== 3. mixed continuous / right-censored / ordinal ==\n")
set.seed(3)
N3 <- 200
Z3 <- matrix(rnorm(3 * N3), ncol = 3) %*% chol(matrix(c(1, .5, .3, .5, 1, .4, .3, .4, 1), 3))
cens <- rexp(N3, 0.3)
tt <- exp(Z3[, 2])
d3 <- data.frame(y = Z3[, 1],
                 t = Surv(pmin(tt, cens), as.numeric(tt <= cens)),
                 o = cut(Z3[, 3], c(-Inf, -.5, .5, Inf), ordered_result = TRUE))
c1 <- Lm(y ~ 1, data = d3)
c2 <- Coxph(t ~ 1, data = d3, order = 4)            # cloglog margin
c3 <- Polr(o ~ 1, data = d3, method = "probit")
cat("event rate:", mean(d3$t[, 2]), "\n")
mix <- mmlt(c1, c2, c3, data = d3, optim = mltoptim(hessian = TRUE))
cat("convergence:", mix$convergence,
    " logLik (default args seed=1, M=1000):", format(logLik(mix), digits = 12), "\n")
print(coef(mix, type = "Corr"))
## Monte Carlo dependence: the integrated dimensions use Genz with M draws
mix2 <- mmlt(c1, c2, c3, data = d3, args = list(seed = 7, M = 1000))
cat("logLik (seed=7):", format(logLik(mix2), digits = 12), "\n")
## explicit weights w with a mixed exact/censored margin: some observations
## have dJ = 1 (t exact, o discrete), some dJ = 2 (t censored, o discrete);
## an explicit w is passed through to the dJ = 1 subsets as well and fails there
set.seed(11); W <- matrix(runif(1 * 500), nrow = 1)
mixw <- suppressWarnings(mmlt(c1, c2, c3, data = d3, args = list(w = W)))
cat("explicit w with per-observation dJ in {1, 2}: convergence =", mixw$convergence,
    "; logLik() errors:",
    inherits(try(logLik(mixw), silent = TRUE), "try-error"), "\n")
s_an <- -colSums(estfun(mix))
s_nd <- grad(function(p) logLik(mix, parm = p), coef(mix, fixed = FALSE))
cat("mixed: max |analytic - numDeriv| score at optimum:", max(abs(s_an - s_nd)), "\n")
cat("conditional = TRUE with non-probit margin errors:",
    inherits(try(mmlt(c1, c2, c3, data = d3, conditional = TRUE), silent = TRUE),
             "try-error"), "\n")
cat("per-observation censoring patterns (event indicator of t):\n")
print(table(d3$t[, 2]))

## ---------------------------------------------------------------------------
## 4. Reproducing R's default integration weights (all observations dJ = 2)
## ---------------------------------------------------------------------------
cat("\n== 4. continuous + two ordinal margins: default w is set.seed(1); runif ==\n")
d4 <- data.frame(y = Z3[, 1],
                 o1 = cut(Z3[, 2], c(-Inf, -.5, .5, Inf), ordered_result = TRUE),
                 o2 = cut(Z3[, 3], c(-Inf, -.3, .7, Inf), ordered_result = TRUE))
e1 <- Lm(y ~ 1, data = d4)
e2 <- Polr(o1 ~ 1, data = d4, method = "probit")
e3 <- Polr(o2 ~ 1, data = d4, method = "probit")
f_def <- mmlt(e1, e2, e3, data = d4)                        # args = list(seed = 1, M = 1000)
set.seed(1); W1 <- matrix(runif((2 - 1) * 1000), ncol = 1000) # mlt:::.MCw(J = dJ, M, seed)
f_w <- mmlt(e1, e2, e3, data = d4, args = list(w = W1))
cat("logLik default:", format(logLik(f_def), digits = 15),
    " logLik explicit W1:", format(logLik(f_w), digits = 15), "\n")
cat("identical stored w:", identical(f_def$args$w, W1), "\n")
f_s7 <- mmlt(e1, e2, e3, data = d4, args = list(seed = 7, M = 1000))
cat("logLik seed 7:", format(logLik(f_s7), digits = 15), "\n")
cat("max |coef(seed1) - coef(seed7)|:", max(abs(coef(f_def) - coef(f_s7))), "\n")
## logLik evaluated at a FIXED parameter under two different w: MC error alone
p0 <- coef(f_def, fixed = FALSE)
cat("logLik at fixed parm, w(seed1) vs w(seed7):",
    format(c(logLik(f_def, parm = p0)), digits = 12),
    format(c(logLik(f_s7, parm = p0)), digits = 12), "\n")
print(coef(f_def, type = "Corr"))
