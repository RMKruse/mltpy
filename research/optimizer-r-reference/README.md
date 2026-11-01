# R reference contract for the optimizer audit

Research resolution for [Establish the versioned R optimizer reference contract](https://github.com/RMKruse/mltpy/issues/169). Inspected 2026-11-01 against mltpy commit `2659daca9057ca7aa4f77589565dcbe8bada1297`. This branch is evidence only: no library changes and no claim that the empirical audit is complete.

## Conclusion

Use R as a versioned implementation reference, with independent validity checks. A matching package name, algorithm family, convergence flag or scalar tolerance does not establish an equivalent problem or accepted solution. Use same-parameter likelihood/score comparisons before comparing fitted parameters; distinguish end-to-end `mlt()` from a controlled R solver call with Python-equivalent constraints.

The audit can proceed with the comparison routes below. Scaled truncation, interaction truncation, boundary cases and invalid reference fits require explicit verification before entering the accepted timing population. These are verification requirements, not blanket declarations that those paths are broken.

## Pinned environment and reproduction

Directly inspected installed versions:

| Component | Version |
|---|---|
| R | 4.6.1, 2026-06-24 |
| mlt | 1.8-4 |
| tram | 1.4-7 |
| basefun | 1.2-6 |
| variables | 1.1-2 |
| alabama | 2025.1.0 |
| Python | 3.13.13 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |

R prints dotted package versions (`1.8.4`), whereas release tags use `1.8-4`. Do not upgrade packages while reproducing this baseline. The web SciPy reference inspected identifies itself as 1.18.0; it supplies documented semantics, not proof of every installed 1.18.1 implementation detail. Record the installed source in the empirical runs. A current [CRAN basefun page](https://cran.r-project.org/web/packages/basefun/index.html) can be cached or move; the installed version above is authoritative for this baseline.

Reproduce the version/source inspection without fitting:

```r
cat(R.version.string, "\n")
for (p in c("mlt", "tram", "basefun", "variables", "alabama"))
    cat(p, as.character(packageVersion(p)), "\n")
print(mlt::mltoptim)
print(alabama::auglag)
print(alabama:::auglag2)
print(mlt:::.mlt_setup)
print(mlt:::.mlt_loglik_exact)
print(mlt:::.mlt_loglik_interval)
print(mlt:::.Exponential)
```

Sources were inspected directly through these installed functions and the release-tagged CRAN mirrors linked below. No complete byte-for-byte upstream package comparison was performed in the final research pass.

## Actual optimizer route

`ctm()` composes response/interaction, shift and scale bases. Its `sumconstr` default depends on how the interacting basis was supplied, so identifiability constraints must be compared explicitly. `scale_shift=FALSE` is the default. [ctm source](https://raw.githubusercontent.com/cran/mlt/1.8-4/R/ctm.R)

In `.mlt_setup`, basis constraint attributes provide `ui` and `ci`; fixed coefficients modify the right-hand side and remove columns. The objective minimizes the weighted negative log-likelihood; the score has the corresponding negative sign. Optional numerical parameter scaling transforms parameters and constraint columns. Optimization tries configured methods in order and stops at the first zero convergence code. Post-fit Hessian construction is separate work, including numerical differentiation for scale models. [setup source](https://raw.githubusercontent.com/cran/mlt/1.8-4/R/mlt.R)

`mltoptim()` defaults to the ordered routes auglag, SPG, nloptr, constrOptim, nlminb, optim, unless configured otherwise. Its auglag wrapper uses BFGS and passes the settings as `control.outer`; the listed `maxit=500` is therefore not forwarded as `control.optim$maxit`. The wrapper keeps parameters, convergence, value and optionally a Hessian, rather than all AL diagnostics. The `optim` fallback returns negative infinity for violated inequalities; the `nlminb` fallback returns positive infinity. This sign difference is a source observation, not a reproduced failure. Exclude those fallback results from authoritative constrained references until directly checked. [wrapper source](https://raw.githubusercontent.com/cran/mlt/1.8-4/R/mltoptim.R)

`alabama::auglag` separates outer settings from inner optimizer settings. Defaults include BFGS, outer `itmax=50`, inner `maxit=100`, `lam0=10`, and `sig0=100`. The inequality implementation adjusts its initial penalty using the initial residual, then uses a quarter-reduction test and tenfold penalty growth. It can stop after lack of parameter progress. Its separate `kkt1` and optional `kkt2` fields are not equivalent to the returned convergence code. [AL source](https://raw.githubusercontent.com/cran/alabama/2025.1.0/R/auglag.R)

Python's [AugLagOptions](https://github.com/RMKruse/mltpy/blob/2659daca9057ca7aa4f77589565dcbe8bada1297/src/mltpy/_auglag.py#L54) uses L-BFGS-B, initial penalty 10, a penalty cap, and its own stall gate. Its claim that L-BFGS-B matches the R inner solver is inaccurate for the inspected defaults. This is a concrete documentation mismatch; it does not by itself prove incorrect numerical results. The shared PHR ancestry is not iteration-by-iteration parity.

## Equivalent-problem contract

The following are audit requirements derived from the source inspection and existing project research, not assertions that every fixture already meets them.

| Item | Required alignment |
|---|---|
| Transformation | Match response support, Bernstein order and coefficient direction, intercept/centering constraints, and every design column. |
| Scaling | Match `h=h0*exp(0.5*Xs*gamma)+Xd*beta+offset`; do not confuse model scaling with numerical coefficient scaling or `scale_shift=TRUE`. |
| Shift signs | Compare design matrices. Raw `ctm` and tram wrappers need not use the same sign convention; blanket negation is unsafe. |
| Interaction layout | Derive an explicit coefficient permutation from basis columns; do not assume R flattening equals Python's row-major layout. Match sum constraints. |
| Observation type | Serialize censoring endpoints explicitly, including infinities. Confirm whether a Boolean denotes an event or censoring. |
| Truncation | Match the conditional probability denominator and its derivative at the same parameters. Censoring and truncation are not interchangeable. |
| Weights/offsets | Match row alignment, absolute weight scale and offsets in both likelihood and support constraints; do not silently normalize weights. |
| Constraints | Match feasible sets, fixed coefficients, bounds, support and nonlinear Jacobians. Record stronger sufficient constraints as a different problem. |
| Start/budget | Compare both a controlled shared feasible start and each package's default start. Measure actual attempts/evaluations; iteration labels are not interchangeable. |
| Result | Re-evaluate original NLL and constraints independently after restarts or polishing. Preserve raw status and diagnostics. |

R's public API specifies case weights and offsets, with offsets excluded from the scaling term. [mlt API](https://search.r-project.org/CRAN/refmans/mlt/html/mlt.html) Direct inspection of `.mlt_loglik_exact` and `.mlt_loglik_interval` shows subtraction of the truncation interval log-probability when truncation is supplied. `.Exponential` uses `pexp`/`dexp`, while its log-density derivative ratio is the smooth constant −1. The source-inspection commands above identify these exact installed primary functions.

## Reference routes by case family

| Python case family | Recommended R route | Conditions / limits |
|---|---|---|
| Unscaled shift; non-exponential base; exact/censored | Matching `ctm`/tram plus `mlt`; same-parameter values and scores first | Match base distribution, sign, bounds, weights and censoring endpoints. Validate solution independently. |
| Scaled shift; non-exponential | `ctm(..., scaling=..., scale_shift=FALSE)` or matching tram wrapper | Match 0.5 factor and shift convention. Treat R's numerical Hessian as a numerical reference, not an analytic oracle. |
| Exponential, scale-only | Existing fixed-lower-coefficient `mlt`/Aareg route when the model deliberately fixes that coefficient; otherwise explicit equivalent constraints | Fixing a coefficient is a different feasible set unless justified for the particular comparison. Do not generalize one fixture's active boundary to censored models. |
| Exponential with shift/offset, scaled or unscaled | R likelihood/score with explicit lower-support-end constraints matching ADR 0005, solved directly with `alabama` or another checked constrained solver | Raw `mlt` observed-point behavior is a separate modeling comparator. Include offset in the constraint RHS; scaling can make the support rows nonlinear. |
| Supported exact interaction, non-exponential | Matching tensor design in R `ctm` | Verify column order, identifiability and constraints; use explicit same-parameter design checks. |
| Unscaled exact exponential interaction | Same likelihood plus explicit per-training-row lower-end support constraints | Do not substitute nonnegative coefficient constraints unless both sides deliberately use that stronger model. |
| Scaled exponential interaction; censored interaction | No Python parity target where current API rejects the combination | Verify the documented rejection; do not include these as solver failures. |
| Truncated cases | Matched conditional R likelihood and independent denominator calculation | Unscaled shift has existing machinery. Scaled/interaction truncation stays unverified until the mathematical audit checks actual dispatch and gradients. |

The existing [exponential R investigation](https://github.com/RMKruse/mltpy/blob/2659daca9057ca7aa4f77589565dcbe8bada1297/research/exp-scaling-r-route/README.md) documents materially different observed-point versus lower-support-end feasible sets, stalled R fits and an explicit-constraint reference. Those are historical measurements, not reruns in this ticket. [ADR 0005](https://github.com/RMKruse/mltpy/blob/2659daca9057ca7aa4f77589565dcbe8bada1297/docs/adr/0005-exponential-scaling-constraint.md) governs the Python model choice.

## Audit concerns and acceptance implications

1. **Keep value parity separate from optimizer parity.** At identical parameters evaluate the unpenalized objective, score, constraints and selected predictions. Only then compare independently accepted solutions. Flat parameter directions can permit comparable likelihoods/predictions with different coefficients.
2. **Do not infer a common tolerance from a common number.** R `optim` has method-specific stopping and scaling controls. [R optim manual](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html) SLSQP's `ftol` participates in several checks; trust-constr distinguishes stationarity, trust radius and barrier thresholds. [SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html), [trust-constr](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html)
3. **Treat the five supplied reports as regression seeds.** All were closed at intake. The [clipping report correction](https://github.com/RMKruse/mltpy/issues/162#issuecomment-6560288766) identifies reversed censoring flags and matching unclipped/R likelihoods. It does not settle clipping behavior elsewhere. Reproduce first-attempt failures separately from restart recovery.
4. **Inspect scaled truncation explicitly.** `_build_truncation_context` accepts no scaling argument and builds a shift-only transformed endpoint. This is a directly inspected source concern, not a reproduced end-to-end defect. Trace callers, rejection guards and gradient dimensions before assigning severity. [pinned helper](https://github.com/RMKruse/mltpy/blob/2659daca9057ca7aa4f77589565dcbe8bada1297/src/mltpy/likelihood.py#L1189)
5. **Require an independent check beyond AL versus AL.** R alabama and Python auglag share mathematical structure. Include analytic small problems and independent constrained methods so shared errors are detectable.

This resolves which references to use and which comparisons remain conditional. It does not certify the current five regressions, every model combination, R fallback behavior or runtime superiority. Those belong to the linked baseline, correctness and performance tickets. No package installation, production change, full test run or benchmark was performed.
