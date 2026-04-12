# What's missing from pymlt? A Plan.

 pymlt is a Python port of Hothorn's R mlt package for Conditional
 Transformation Models. The core framework is solid: 28/29 R validation
 cases pass, all 4 censoring types work with analytical gradients, 3 base distributions are supported, and the tram convenience layer provides
 BoxCox, Coxph, and Colr. However, the R ecosystem (mlt + tram + basefun + extensions) offers substantially more: inference tools (vcov,
 confint), additional model classes (Polr, Survreg, Lm, Lehmann, Aareg),
 more prediction types, truncation, weights, residuals, model comparison,
  and advanced features like interacting/scaling terms and multivariate
 models.

 This plan catalogs every gap, prioritizes them by practical value, and
 sequences them for implementation.

 ---
 ## Feature Comparison Matrix

 ### Model Classes (tram)

```
 ┌─────────┬───────────────┬──────────────┬───────────────────────────┐
 │  Model  │       R       │    pymlt     │           Notes           │
 ├─────────┼───────────────┼──────────────┼───────────────────────────┤
 │ BoxCox  │ tram::BoxCox  │ pymlt.BoxCox │ Complete                  │
 ├─────────┼───────────────┼──────────────┼───────────────────────────┤
 │ Coxph   │ tram::Coxph   │ pymlt.Coxph  │ Complete                  │
 │         │               │              │ (min_extreme_value fixed) │
 ├─────────┼───────────────┼──────────────┼───────────────────────────┤
 │ Colr    │ tram::Colr    │ pymlt.Colr   │ Complete                  │
 ├─────────┼───────────────┼──────────────┼───────────────────────────┤
 │ Lm      │ tram::Lm      │ --           │ Normal linear model as    │
 │         │               │              │ CTM (order=1)             │
 ├─────────┼───────────────┼──────────────┼───────────────────────────┤
 │ Polr    │ tram::Polr    │ --           │ Ordinal regression        │
 │         │               │              │ (proportional odds)       │
 ├─────────┼───────────────┼──────────────┼───────────────────────────┤
 │         │               │              │ Parametric survival       │
 │ Survreg │ tram::Survreg │ --           │ (Weibull, log-normal,     │
 │         │               │              │ log-logistic)             │
 ├─────────┼───────────────┼──────────────┼───────────────────────────┤
 │ Lehmann │ tram::Lehmann │ --           │ Proportional reverse-time │
 │         │               │              │  hazards                  │
 ├─────────┼───────────────┼──────────────┼───────────────────────────┤
 │         │               │              │ Aalen additive hazards    │
 │ Aareg   │ tram::Aareg   │ --           │ (time-varying             │
 │         │               │              │ coefficients)             │
 └─────────┴───────────────┴──────────────┴───────────────────────────┘
```

### Prediction Types (predict.mlt)

```
 ┌────────────────┬─────┬────────────────────────────┬───────────────┐
 │      Type      │  R  │           pymlt            │     Notes     │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ distribution   │ Yes │ predict(what="distribution │ Complete      │
 │ (CDF)          │     │ ")                         │               │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ density (PDF)  │ Yes │ predict(what="density")    │ Complete      │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ quantile       │ Yes │ predict(what="quantile")   │ Complete      │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ hazard         │ Yes │ predict(what="hazard")     │ Complete      │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ survivor       │ Yes │ Coxph.survival() only      │ Not a general │
 │                │     │                            │  predict type │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ trafo          │ Yes │ --                         │ Raw transform │
 │                │     │                            │ ation h(y)    │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ cumhazard      │ Yes │ --                         │ -log(S(y))    │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ odds           │ Yes │ --                         │ F(h)/(1-F(h)) │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ logdistributio │ Yes │ --                         │ log CDF       │
 │ n              │     │                            │               │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ logsurvivor    │ Yes │ --                         │ log S(y)      │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ logdensity     │ Yes │ --                         │ log PDF       │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ loghazard      │ Yes │ --                         │ log hazard    │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │                │     │                            │ log           │
 │ logcumhazard   │ Yes │ --                         │ cumulative    │
 │                │     │                            │ hazard        │
 ├────────────────┼─────┼────────────────────────────┼───────────────┤
 │ logodds        │ Yes │ --                         │ log odds      │
 └────────────────┴─────┴────────────────────────────┴───────────────┘
```

### Base Distributions
```
 ┌────────────────────────┬─────┬───────┬────────────────────────────┐
 │      Distribution      │  R  │ pymlt │           Notes            │
 ├────────────────────────┼─────┼───────┼────────────────────────────┤
 │ Normal                 │ Yes │ Yes   │                            │
 ├────────────────────────┼─────┼───────┼────────────────────────────┤
 │ Logistic               │ Yes │ Yes   │                            │
 ├────────────────────────┼─────┼───────┼────────────────────────────┤
 │ MinExtrVal (rev.       │ Yes │ Yes   │ Cox PH link                │
 │ Gumbel)                │     │       │                            │
 ├────────────────────────┼─────┼───────┼────────────────────────────┤
 │ MaxExtrVal (Gumbel)    │ Yes │ --    │ Lehmann link               │
 ├────────────────────────┼─────┼───────┼────────────────────────────┤
 │ Exponential            │ Yes │ --    │ Piecewise exponential      │
 │                        │     │       │ models                     │
 ├────────────────────────┼─────┼───────┼────────────────────────────┤
 │ Laplace                │ Yes │ --    │ Median regression link     │
 ├────────────────────────┼─────┼───────┼────────────────────────────┤
 │ Cauchy                 │ Yes │ --    │ Heavy-tailed link          │
 └────────────────────────┴─────┴───────┴────────────────────────────┘
```

### Basis Functions (basefun)
```
 ┌─────────────────────┬─────┬────────────────┬──────────────────────┐
 │        Basis        │  R  │     pymlt      │        Notes         │
 ├─────────────────────┼─────┼────────────────┼──────────────────────┤
 │ Bernstein_basis     │ Yes │ BernsteinBasis │ Complete             │
 ├─────────────────────┼─────┼────────────────┼──────────────────────┤
 │ polynomial_basis    │ Yes │ --             │                      │
 ├─────────────────────┼─────┼────────────────┼──────────────────────┤
 │ Legendre_basis      │ Yes │ --             │ Orthogonal, better   │
 │                     │     │                │ conditioning         │
 ├─────────────────────┼─────┼────────────────┼──────────────────────┤
 │ log_basis           │ Yes │ --             │ Needed for Survreg   │
 ├─────────────────────┼─────┼────────────────┼──────────────────────┤
 │ cyclic_basis        │ Yes │ --             │ Periodic data        │
 ├─────────────────────┼─────┼────────────────┼──────────────────────┤
 │ intercept_basis     │ Yes │ --             │ Ordinal cutpoints    │
 ├─────────────────────┼─────┼────────────────┼──────────────────────┤
 │ Basis composition   │ Yes │ --             │ Tensor products for  │
 │ (b(), c())          │     │                │ interacting terms    │
 └─────────────────────┴─────┴────────────────┴──────────────────────┘
```

### Inference
```
 ┌──────────────┬──────────────────────┬─────────────┬───────────────┐
 │   Feature    │          R           │    pymlt    │     Notes     │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │              │                      │ model.score │ Exists but no │
 │ logLik       │ logLik()             │ ()          │  AIC/BIC      │
 │              │                      │             │ extraction    │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │              │                      │             │ Requires      │
 │ vcov         │ vcov()               │ --          │ Hessian       │
 │              │                      │             │ computation   │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │ confint      │ confint()            │ --          │ Wald CIs from │
 │ (parameters) │                      │             │  vcov         │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │ confband     │                      │             │ Delta-method  │
 │ (CDF bands)  │ confband()           │ --          │ confidence    │
 │              │                      │             │ bands         │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │ estfun       │                      │             │ Per-observati │
 │ (score contr │ estfun()             │ --          │ on gradients  │
 │ ibutions)    │                      │             │               │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │              │                      │             │ Second        │
 │ Hessian      │ Hessian()            │ --          │ derivatives   │
 │              │                      │             │ of neg-loglik │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │ sandwich SE  │ via sandwich pkg     │ --          │ Robust        │
 │              │                      │             │ covariance    │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │ Wald test    │ via lmtest pkg       │ --          │ Hypothesis    │
 │              │                      │             │ testing       │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │ profile      │ confint(type="profil │             │ More accurate │
 │ likelihood   │ e")                  │ --          │  than Wald    │
 │ CI           │                      │             │               │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │              │                      │             │ Cox-Snell,    │
 │ residuals    │ residuals()          │ --          │ deviance,     │
 │              │                      │             │ score         │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │ AIC / BIC    │ AIC() / BIC()        │ --          │ Model         │
 │              │                      │             │ selection     │
 ├──────────────┼──────────────────────┼─────────────┼───────────────┤
 │ anova (LRT)  │ anova()              │ --          │ Nested model  │
 │              │                      │             │ comparison    │
 └──────────────┴──────────────────────┴─────────────┴───────────────┘
```

### Model Specification (ctm)
```
 ┌───────────────┬─────┬───────┬─────────────────────────────────────┐
 │   Component   │  R  │ pymlt │                Notes                │
 ├───────────────┼─────┼───────┼─────────────────────────────────────┤
 │ Response a(y) │ Yes │ Yes   │ Bernstein basis only                │
 ├───────────────┼─────┼───────┼─────────────────────────────────────┤
 │ Shifting d(x) │ Yes │ Yes   │ Linear X@beta                       │
 ├───────────────┼─────┼───────┼─────────────────────────────────────┤
 │ Interacting   │ Yes │ --    │ Tensor product with response        │
 │ b(x)          │     │       │ (non-proportional effects)          │
 ├───────────────┼─────┼───────┼─────────────────────────────────────┤
 │ Scaling s(x)  │ Yes │ --    │ Heteroskedastic models              │
 └───────────────┴─────┴───────┴─────────────────────────────────────┘
```

### Data Handling
```
 ┌────────────────┬─────┬────────────┬───────────────────────────────┐
 │    Feature     │  R  │   pymlt    │             Notes             │
 ├────────────────┼─────┼────────────┼───────────────────────────────┤
 │ Exact          │ Yes │ Yes        │                               │
 │ observations   │     │            │                               │
 ├────────────────┼─────┼────────────┼───────────────────────────────┤
 │ Right          │ Yes │ Yes        │                               │
 │ censoring      │     │            │                               │
 ├────────────────┼─────┼────────────┼───────────────────────────────┤
 │ Left censoring │ Yes │ Yes        │                               │
 ├────────────────┼─────┼────────────┼───────────────────────────────┤
 │ Interval       │ Yes │ Yes        │                               │
 │ censoring      │     │            │                               │
 ├────────────────┼─────┼────────────┼───────────────────────────────┤
 │                │     │ Stored but │ CensoredData has trunc        │
 │ Truncation     │ Yes │  unused    │ fields, likelihood ignores    │
 │                │     │            │ them                          │
 ├────────────────┼─────┼────────────┼───────────────────────────────┤
 │ Weights        │ Yes │ --         │ Weighted likelihood           │
 ├────────────────┼─────┼────────────┼───────────────────────────────┤
 │ Offset         │ Yes │ --         │ Fixed linear predictor        │
 │                │     │            │ component                     │
 ├────────────────┼─────┼────────────┼───────────────────────────────┤
 │ Subset         │ Yes │ --         │ Trivial (user slices data)    │
 ├────────────────┼─────┼────────────┼───────────────────────────────┤
 │ Clusters       │ Yes │ --         │ Cluster-robust SE             │
 └────────────────┴─────┴────────────┴───────────────────────────────┘
```

 ---
 
 ## Tiered Implementation Roadmap

### Tier 1 -- High Priority (core statistical completeness)

 These make pymlt usable for real-world analysis that requires reporting
 uncertainty.

#### Unit 1A: Additional Prediction Types

 - What: Add survivor, cumhazard, trafo, odds to predict(what=...), plus
 all log-scale variants (logdistribution, logsurvivor, logdensity,
 loghazard, logcumhazard, logodds)
 - Why: R supports 14 prediction types, pymlt supports 4. Log-scale
 versions are essential for numerical stability. trafo needed for
 diagnostic plots. cumhazard is standard survival output.
 - Math: All trivial compositions of existing h, h', dist.cdf, dist.sf,
 dist.logcdf, dist.logsf, dist.logpdf
 - Files: model.py (extend predict() with new branches)
 - Dependencies: None
 - Complexity: Small
 - R validation: predict(mlt_fit, type="survivor") etc.

#### Unit 1B: AIC / BIC / Model Comparison

 - What: aic(), bic() methods and anova(model1, model2) function for
 likelihood ratio tests
 - Why: Cannot do model selection (polynomial order, covariate inclusion)
  without these
 - Math: AIC = -2loglik + 2k; BIC = -2*loglik + log(n)k; LRT: D =
 2(ll_full - ll_reduced) ~ chi2(df)
 - Files: model.py (add methods, track n_obs_ and n_free_params_)
 - Dependencies: None
 - Complexity: Small
 - R validation: AIC(mlt_fit), anova(fit1, fit2)

#### Unit 1C: Additional Base Distributions

 - What: max_extreme_value (Gumbel, scipy.stats.gumbel_r) and exponential
 - Why: MaxExtrVal enables Lehmann model. These two + existing three
 cover all core tram distributions.
 - Math: MaxExtrVal score: 1 - exp(-h). Exponential score: 1 (constant),
 but requires h >= 0 constraint.
 - Files: likelihood.py (_get_dist, _neg_score,
 _VALID_BASE_DISTRIBUTIONS)
 - Dependencies: None
 - Complexity: Small (MaxExtrVal), Medium (Exponential needs additional
 constraint)
 - R validation: mlt with todistr="maxextrval"

#### Unit 1D: Hessian, vcov, and Score Contributions

 - What: Analytical Hessian of neg-loglik, vcov() method (= H^{-1}),
 per-observation score contributions (estfun())
 - Why: The single most critical gap. Without standard errors, users
 cannot report confidence intervals for covariate effects, cannot do
 hypothesis tests, cannot assess parameter uncertainty. Every published
 transformation model analysis reports CIs for beta.
 - Math: H_{jk} = sum_i [(d^2 log f / dh^2) * B_j * B_k + ...] with
 censoring-type-specific terms. Second derivative of log-density: normal
 = -1, logistic = 2f(h)(1-2F(h)), min_extreme_value = exp(h). vcov =
 H^{-1}. estfun = (n, p+q) matrix of per-observation gradient vectors
 (before summation).
 - Files: likelihood.py (add _hess_none, _hess_right, _hess_left,
 _hess_interval, public hessian()), model.py (add vcov(), estfun(), store
  hessian_ after fit), tram.py (update summary() to show SE and z-values)
 - Dependencies: None (builds on existing gradient infrastructure)
 - Complexity: Large
 - R validation: vcov(mlt_fit) returns reference matrix

#### Unit 1E: Confidence Intervals and Confidence Bands

 - What: confint(level=0.95) for beta parameters, confband(y_grid, X,
 level=0.95) for CDF/survival curves
 - Why: Direct user-facing output of the Hessian. confint for beta goes
 in tables; confband for CDFs goes in survival plots.
 - Math: Wald CI: beta_hat +/- z * sqrt(vcov[j,j]). Confband via delta
 method: Var(F(y|x)) = (dF/dtheta)^T * vcov * (dF/dtheta)
 - Files: model.py (add confint(), confband())
 - Dependencies: Unit 1D (vcov)
 - Complexity: Medium
 - R validation: confint(mlt_fit), confband(mlt_fit, ...)

### Tier 2 -- Medium Priority (extended functionality)

 These round out the package for broader use cases.

#### Unit 2A: Observation Weights and Offset

 - What: weights param in fit(), offset param for fixed linear predictor
 component
 - Why: Weights needed for survey data, case-control, bootstrap. Offset
 standard in epidemiology.
 - Math: Weighted loglik: sum_i w_i * ell_i. Offset: h_total = B@theta_b
 + X@beta + offset (not optimized).
 - Files: likelihood.py (add weights to all _ll_* and _grad_*),
 optimizer.py (thread through), model.py (add to fit())
 - Dependencies: None, but should coordinate with Unit 1D (Hessian needs
 weights too)
 - Complexity: Medium (simple math but touches 8+ functions across 3
 files)
 - R validation: mlt(model, data, weights=...)

#### Unit 2B: Polr (Ordinal Regression)

 - What: Polr tram model for proportional-odds ordinal outcomes
 - Why: Ordinal outcomes (pain scales, severity grades, Likert items) are
  extremely common. OrderedVariable already exists in variables.py but is
  unused.
 - Math: For K levels, h(y_k) = theta_k for k=1..K-1. Likelihood is
 interval-censored: log(F(theta_k + x'beta) - F(theta_{k-1} + x'beta)).
 Uses existing interval-censored likelihood with logistic base.
 - Files: tram.py (add Polr class), variables.py (connect OrderedVariable
  to fitting), possibly basis.py (ordinal basis / intercept basis)
 - Dependencies: Existing interval-censoring machinery
 - Complexity: Medium (data pipeline from ordinal levels to interval
 bounds is the main work)
 - R validation: tram::Polr(ordered_response ~ covariates, data=...)

#### Unit 2C: Truncation in Likelihood

 - What: Left/right truncation (delayed entry) in the log-likelihood
 computation
 - Why: CensoredData stores trunc_lower / trunc_upper but likelihood
 ignores them. Essential for survival analysis with delayed entry.
 - Math: Under truncation [l_i, u_i], add denominator: ell_i = [standard
 term] - log(F(h(u_i)) - F(h(l_i))). Uses existing _log_diff_ndtr.
 - Files: likelihood.py (modify all 8 _ll_* and _grad_* functions)
 - Dependencies: None
 - Complexity: Medium (8 functions to modify, truncation gradient must be
  derived and verified)
 - R validation: Surv(time, event, type="counting")

#### Unit 2D: Survreg (Parametric Survival)

 - What: Survreg tram model for Weibull, log-normal, log-logistic
 parametric survival
 - Why: Workhorses of parametric survival analysis in reliability and
 clinical trials
 - Math: Survreg = Coxph on log(T). Weibull = min_extreme_value on
 log-scale; log-normal = normal on log-scale; log-logistic = logistic on
 log-scale. Derivative: h'(y) = (1/y) * h'(log y).
 - Files: tram.py (add Survreg), basis.py (log-transform flag or wrapper)
 - Dependencies: Log-transform handling in basis
 - Complexity: Medium (composition of existing pieces but log-transform
 affects derivatives)
 - R validation: tram::Survreg(Surv(time, status) ~ covariates,
 dist="weibull")

#### Unit 2E: Residuals

 - What: residuals(type="cox-snell"|"deviance"|"score") method
 - Why: Primary model diagnostic tool. Cox-Snell residuals ~ Exp(1) if
 model correct.
 - Math: Cox-Snell: r_i = -log(S_hat(y_i|x_i)). Deviance: d_i =
 sign(r_i-1) * sqrt(2|r_i - log(r_i) - 1|). Score: per-observation
 gradient at theta_hat.
 - Files: model.py (add residuals())
 - Dependencies: Score residuals need Unit 1D (estfun).
 Cox-Snell/deviance are independent.
 - Complexity: Small (Cox-Snell), Medium (deviance with censored data)
 - R validation: residuals(mlt_fit)

#### Unit 2F: Lm (Linear Model as CTM)

 - What: Lm tram model with order=1, normal base distribution
 - Why: Demonstrates classical linear regression as CTM special case.
 Good for teaching, testing, baseline comparison.
 - Math: order=1 Bernstein = linear function. With normal base + linear
 shift = exact normal linear model.
 - Files: tram.py (one-line subclass fixing order=1,
 base_distribution="normal")
 - Dependencies: None
 - Complexity: Small (trivially implemented)
 - R validation: Compare against tram::Lm() and base R lm()

### Tier 3 -- Low Priority (advanced/niche)

#### Unit 3A: Sandwich SE and Wald Tests

 - What: sandwich_vcov(), wald_test(), profile likelihood CIs
 - Math: V_sandwich = H^{-1} M H^{-1} where M = sum_i grad_i grad_i^T
 - Files: model.py
 - Dependencies: Unit 1D (Hessian, estfun)
 - Complexity: Medium (sandwich), Large (profile likelihood requires
 refitting with fixed params)

#### Unit 3B: Additional Basis Functions

 - What: polynomial_basis, Legendre_basis, log_basis, intercept_basis
 - Files: basis.py (new classes implementing same interface as
 BernsteinBasis)
 - Dependencies: None, but driven by specific models (Survreg, Polr)
 - Complexity: Small per basis (~50 lines each), Medium overall

#### Unit 3C: Lehmann Model

 - What: Proportional reverse-time hazards (dual of Coxph)
 - Math: Uses max_extreme_value base distribution, right censoring
 - Files: tram.py (thin subclass like Coxph)
 - Dependencies: Unit 1C (MaxExtrVal distribution)
 - Complexity: Small

#### Unit 3D: Interacting Terms (Tensor Products)

 - What: Non-proportional effects via h(y|x) = sum_j a_j(y) * b_j(x)
 instead of h_0(y) + x'beta
 - Why: Needed for crossing survival curves, non-proportional hazards,
 stratum-specific effects
 - Math: Design matrix becomes a(y) kron b(x), parameter vector length =
 p*q, monotonicity must hold for all x
 - Files: basis.py (TensorProductBasis), constraints.py (per-x
 monotonicity), likelihood.py (redesign h computation), model.py (ctm
 specification)
 - Dependencies: Unit 3B (covariate basis)
 - Complexity: Large (architectural change)

#### Unit 3E: Scaling Terms

 - What: Heteroskedastic models via h(y|x) = h_0(y) * exp(x_s @ gamma) +
 x_d @ beta
 - Files: likelihood.py, model.py, optimizer.py (parameter vector becomes
  [theta_b | beta | gamma])
 - Dependencies: None strictly, pairs with Unit 3D
 - Complexity: Large (non-linear coupling between theta and gamma)

#### Unit 3F: Laplace and Cauchy Base Distributions

 - What: Two additional base distributions. Laplace = median regression
 link, Cauchy = heavy tails.
 - Math: Laplace score: sign(h). Cauchy score: 2h/(1+h^2). Both in
 scipy.stats.
 - Files: likelihood.py (same pattern as existing distributions)
 - Complexity: Small

### Tier 4 -- Out of Scope / Deferred

#### Feature: mtram (clustered/mixed models)
 Why Deferred: Requires random effects, marginal likelihood with
 numerical
   integration (Gauss-Hermite or Laplace approx). Entirely different
   optimization problem. R uses lme4-style infrastructure.
 ────────────────────────────────────────
#### Feature: mmlt (multivariate CTMs)
 Why Deferred: Requires copula/vine structures, multivariate normal CDFs,

   quadratic growth in parameters (correlation matrix). Separate modeling

   paradigm.
 ────────────────────────────────────────
#### Feature: Aareg (Aalen additive hazards)
 Why Deferred: Requires time-varying coefficients (interacting terms with

   specific structure). Build on Unit 3D.
 ────────────────────────────────────────
#### Feature: update()
 Why Deferred: Low impact. Users can call fit() again. Warm-start could
 be
   an optional warm_start param.
 ────────────────────────────────────────
#### Feature: coef(fixed=TRUE)
 Why Deferred: Minor API convenience. model.theta_ already has the full
   vector.

 ---
 
## Recommended Implementation Sequence

### Phase 1 — Foundation for inference (highest value-to-effort ratio):
   1. Unit 1A  Additional prediction types          [Small]
   2. Unit 1B  AIC / BIC / anova                    [Small]
   3. Unit 1C  MaxExtrVal + Exponential distros     [Small]
   4. Unit 2F  Lm model                             [Small]
   5. Unit 1D  Hessian + vcov + estfun              [Large — but
 critical]
   6. Unit 1E  confint + confband                   [Medium — builds on
 1D]

### Phase 2 — Extended models and diagnostics:
   7. Unit 2E  Residuals                            [Small/Medium]
   8. Unit 2A  Weights + offset                     [Medium]
   9. Unit 2B  Polr (ordinal regression)            [Medium]
   10. Unit 2C  Truncation in likelihood            [Medium]

### Phase 3 — Specialized models:
   11. Unit 2D  Survreg                             [Medium]
   12. Unit 3C  Lehmann                             [Small — needs 1C]
   13. Unit 3A  Sandwich SE + Wald tests            [Medium — needs 1D]
   14. Unit 3F  Laplace + Cauchy distributions      [Small]

### Phase 4 — Architectural extensions:
   15. Unit 3B  Additional basis functions           [Medium]
   16. Unit 3D  Tensor products / interacting terms  [Large]
   17. Unit 3E  Scaling terms                        [Large]

## Critical Files by Impact

  1. Unit 1C  MaxExtrVal + Exponential distros     [Small]
  2. Unit 2F  Lm model                             [Small]
  3. Unit 1D  Hessian + vcov + estfun              [Large — but
critical]
  1. Unit 1E  confint + confband                   [Medium — builds
on 1D]

Phase 2 — Extended models and diagnostics:
  7. Unit 2E  Residuals                            [Small/Medium]
  8. Unit 2A  Weights + offset                     [Medium]
  9. Unit 2B  Polr (ordinal regression)            [Medium]
  10. Unit 2C  Truncation in likelihood            [Medium]

Phase 3 — Specialized models:
  11. Unit 2D  Survreg                             [Medium]
  12. Unit 3C  Lehmann                             [Small — needs 1C]
  13. Unit 3A  Sandwich SE + Wald tests            [Medium — needs
1D]
  14. Unit 3F  Laplace + Cauchy distributions      [Small]

Phase 4 — Architectural extensions:
  15. Unit 3B  Additional basis functions           [Medium]
  16. Unit 3D  Tensor products / interacting terms  [Large]
  17. Unit 3E  Scaling terms                        [Large]

```
     ┌──────────────────────┬────────────────────────────────────────────┐
     │         File         │            Units That Touch It             │
     ├──────────────────────┼────────────────────────────────────────────┤
     │ pymlt/likelihood.py  │ 1C, 1D, 2A, 2C, 3E, 3F                     │
     ├──────────────────────┼────────────────────────────────────────────┤
     │ pymlt/model.py       │ 1A, 1B, 1D, 1E, 2A, 2E                     │
     ├──────────────────────┼────────────────────────────────────────────┤
     │ pymlt/tram.py        │ 2B, 2D, 2F, 3C (+ summary updates from 1D) │
     ├──────────────────────┼────────────────────────────────────────────┤
     │ pymlt/basis.py       │ 2D, 3B, 3D                                 │
     ├──────────────────────┼────────────────────────────────────────────┤
     │ pymlt/constraints.py │ 3D                                         │
     ├──────────────────────┼────────────────────────────────────────────┤
     │ pymlt/optimizer.py   │ 2A, 3E                                     │
     ├──────────────────────┼────────────────────────────────────────────┤
     │ pymlt/variables.py   │ 2B                                         │
     └──────────────────────┴────────────────────────────────────────────┘
```

## Verification Strategy

Each unit should:
1. Generate R reference values (extend
validation/generate_all_references.R)
2. Add parametrized pytest tests comparing pymlt output to R
reference
3. For inference features (vcov, confint): compare element-wise
against vcov(mlt_fit), confint(mlt_fit)
4. For new models (Polr, Survreg, Lm, Lehmann): add validation cases
following existing pattern
5. Run full validation suite (python validation/run_validation.py) to
confirm no regressions
6. Run gradient tests for any new distribution or likelihood
modification

Current coverage: pymlt implements the core CTM framework well — 3/8
tram models, 4/14 prediction types, 3/7 base distributions, 0/12
inference features.

Biggest gaps by impact:

1. Inference layer (vcov, confint, confband) — blocks any real-world
analysis that needs uncertainty quantification
2. Model selection (AIC/BIC/anova) — blocks systematic model comparison
3. Missing prediction types — 10 types including survivor, cumhazard,
and all log-scale variants
4. Missing model classes — Lm, Polr, Survreg, Lehmann are all
straightforward given existing infrastructure

Recommended starting point: The four small units (1A, 1B, 1C, 2F) can be knocked out quickly for immediate value, then tackle the large but
critical Hessian/vcov unit (1D) which unlocks the entire inference
chain.
