# Mathematical and literature criteria for the optimizer audit

Resolution for [Establish mathematical and literature criteria for optimizer validity](https://github.com/RMKruse/mltpy/issues/170), researched 2026-11-01 against mltpy `2659daca9057ca7aa4f77589565dcbe8bada1297`. This is a source-backed audit checklist and candidate assessment, not a solver replacement or a completed correctness audit.

## Main conclusion

Audit mathematical model, numerical evaluation, constrained stationarity and empirical reliability separately. Retain auglag, SLSQP and trust-constr as the initial comparison set. Add a modern safeguarded AL implementation or Ipopt only if the existing methods reveal a concrete gap worth an additional dependency or experiment. No inspected literature establishes that one method is fastest or most reliable for this repository.

The highest-value targeted probes are objective translation/scaling, boundedness of inner AL problems, nonlinear/degenerate active constraints, original-objective rescoring, and gradient/Hessian consistency at tails and support boundaries. Current documentation overstates some connections to R and some interpretations of stationarity; the probes below distinguish those concerns from demonstrated runtime defects.

## Optimality contract

Use the repository's convention `min f(x)` subject to `c(x)=0`, `g(x)>=0`, and simple bounds. With `L=f-lambda^T c-mu^T g`, inequality multipliers satisfy `mu>=0`. Independently report primal violation, dual-sign violation, stationarity, and complementarity. Include bound normals or a correctly scaled projected-gradient mapping; an ordinary gradient need not vanish at a bound optimum.

KKT conditions are sufficient under appropriate convexity conditions; general nonconvex stationarity is not a certificate of a global optimum. Constraint qualifications matter for necessity and multiplier existence. A dual lower bound uses an infimum of the Lagrangian, not merely its value at a stationary point. These distinctions follow the authors' [duality/KKT slides, chapter 5, PDF pages 173–175](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf). Do not assume the scaled CTM objective or feasible set is convex without proving it for that family.

Proposed evidence fields (thresholds remain a human acceptance-protocol decision):

| Check | Evidence to preserve |
|---|---|
| Objective fidelity | Original, unpenalized NLL at final x; independent same-parameter reference; finite/domain status |
| Primal feasibility | Each constraint block's maximum violation in original units, plus documented scaled values |
| Stationarity | Lagrangian residual with multiplier sign convention and bound treatment; projected/refitted residual separately |
| Complementarity | Products or an explicitly defined equivalent residual, not only small primal violation |
| Local quality | Feasible perturbations, selected critical-direction curvature, multistart comparisons; qualifications stated |
| Identifiability | Rank/conditioning, objective and prediction agreement as well as coefficient distance |
| Reliability | Original solver exit, attempt counts, inner failure reason, final validated status, warnings and budgets |

## PHR and modern AL practice

For inequalities in this sign convention, the PHR term is `([max(0,mu-rho*g)]^2-mu^2)/(2*rho)` summed over rows. Its gradient contributes `-Jg^T max(0,mu-rho*g)`. The equality term is `-lambda^T c + rho*||c||^2/2`. Differentiating these expressions gives the signs implemented in `_make_auglag_obj`; this algebra checks the local formula, not the correctness of the complete update/termination loop. [pinned implementation](https://github.com/RMKruse/mltpy/blob/2659daca9057ca7aa4f77589565dcbe8bada1297/src/mltpy/_auglag.py#L186)

Birgin, Haeser and Martínez's 2023 [scaled-subproblem stopping paper](https://optimization-online.org/wp-content/uploads/2023/10/bhm.pdf), especially Algorithm 2.1 and its modified criterion, provides a useful explicit baseline: box-constrained inner first-order accuracy, a residual-based penalty update, and safeguarded multipliers. Its proposed multiplier-based scaling changes the theory; it is not a justification for arbitrary objective-value scaling. The inner accuracy sequence and assumptions must be compared with the actual implementation before invoking a convergence theorem.

The authors' 2024 [general-class AL study](https://optimization-online.org/2024/05/on-the-global-convergence-of-a-general-class-of-augmented-lagrangian-methods/) extends analysis to some nonsafeguarded updates under regularity assumptions. Its abstract was inspected; no theorem from the full paper is claimed here. Likewise the 2026 [elastic-safeguarding preprint](https://arxiv.org/abs/2607.02697v1) is a current research lead, not evidence for adopting an untested method. The abstract reports comparisons with ALGENCAN; this ticket has not reproduced them or verified implementation availability.

Audit the following implementation obligations: inner problems must be meaningful and solvable to the needed accuracy; finite penalty caps and fixed inner tolerances need explicit failure behavior rather than an unqualified asymptotic guarantee; multiplier updates must use consistent residuals; a failed inner solve must not become success solely because its parameters stopped moving; domain extensions must preserve the true model on the feasible region.

## ADR 0004: justified checks and limits

The [projected-stationarity helper](https://github.com/RMKruse/mltpy/blob/2659daca9057ca7aa4f77589565dcbe8bada1297/src/mltpy/_auglag.py#L288) eliminates equality normals and fits nonnegative inequality-normal coefficients. In exact arithmetic for the stated linear working-set problem, the residual argument is sound: orthogonality and NNLS conditions yield a direction with negative objective directional derivative and feasible active linearized constraints. Numerical rank decisions and the selected working set still need testing. A row with slack should not absorb a gradient without checking its multiplier-slack contribution.

The claimed gap `mu^T max(g,0)` is a complementarity measure. It equals a primal-minus-dual-bound quantity only with the required feasibility, stationarity and global Lagrangian-minimization conditions (e.g. a suitable convex setting). With nonconvex scaling or nonlinear constraints, it should not be presented as a certified objective-error bound. The helper's docstring itself qualifies the convex interpretation; callers and external reporting must preserve that qualification.

ADR 0004 attributes `theta_tol^(2/3)*(1+abs(f))` to Gill, Murray and Wright, *Practical Optimization* §8.2.3. The original section was not available for direct inspection in this research. Its attribution and transfer from function precision to this specific finite-tolerance constrained test therefore remain **unverified**, not disproved. The audit can test its actual behavior independently; it must not treat the citation as a verified universal bound.

### Four analytical probes derived for this audit

These are mathematical counterexamples to overbroad interpretations, not claims that a production fit has reproduced each failure.

1. **Objective translation.** At `x=1`, `f(x)=x^2/2` has gradient 1. With `theta_tol=1e-8`, the current threshold is about `6.96e-6`. Replacing f by `f+1e6` preserves its minimizer and gradient but raises the threshold to about `4.64`. Thus the stationarity sub-test changes from rejection to acceptance at the same x. Full solver acceptance also requires a stall and feasibility; that must be tested rather than inferred. Add constant-shift tests independently of tests for multiplying observation weights.
2. **Complementarity is not a global gap.** For unconstrained `f(x)=x^4-x^2`, x=0 is stationary and has zero complementarity, yet its objective is 1/4 above the global minimum. A small first-order residual is not even sufficient for a local minimum in general.
3. **Failed constraint qualification.** For `min x` subject to `-x^2>=0`, x=0 is the only feasible point. The constraint Jacobian is zero there, so no finite multiplier cancels objective gradient 1. A normal-fit residual can reject the true constrained optimum. This is an intentionally degenerate example: it tests classification and documentation, not a proposed relaxation of validity checks. With regular curved constraints, linearized feasibility also need not make a straight finite step feasible.
4. **An unbounded AL subproblem.** Minimize `-t*exp(s)` subject to `-t>=0`. At a fixed infeasible t=1, any fixed finite quadratic penalty in t and finite multiplier leave the AL objective tending to negative infinity as s grows. Raising rho finitely does not cure this inner problem. Hard domain constraints or an equivalent formulation may be necessary; do not infer safety from feasibility of the original optimum alone. This captures the mechanism to examine in censored exponential scaling.

Numerical checks for probes 1, 2 and 4 are shown below; probe 3 follows directly from the singleton feasible set. These checks were run successfully using the standard library and do not invoke production solver behavior.

```python
import math
k = (1e-8)**(2/3)
assert k*1.5 < 1 < k*(1+0.5+1e6)
x = 1/math.sqrt(2)
assert math.isclose(x**4-x*x, -0.25)
rho, mu = 100.0, 10.0
def al_at_t_one(s):
    return -math.exp(s) + mu + rho/2
assert al_at_t_one(20) < al_at_t_one(10) < al_at_t_one(0)
```

## Maintained implementations and candidate comparison

| Method | Primary-source facts | Relevance and cost |
|---|---|---|
| Existing PHR + SciPy L-BFGS-B | L-BFGS-B distinguishes relative objective decrease, projected-gradient tolerance, line-search and iteration/evaluation limits. [official API](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) | No new dependency. Audit inner stopping, box coordinates and failed-line-search propagation before replacing it. |
| SciPy SLSQP | `ftol` affects several optimality/progress checks; returned QP multipliers omit bound multipliers. [official API](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html) | Existing independent comparison route. Reconstruct full constraints and bound checks; monitor initial step and restart recovery. |
| SciPy trust-constr | Separate Lagrangian-gradient/feasibility, trust-radius and barrier criteria; method switches between equality SQP and an interior-point path. [API](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html), [method description](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) | Already installed. Verify exact installed exit semantics, analytic curvature and nonsmooth boundary exceptions. |
| ALGENCAN-style safeguarded AL | The inspected 2023 algorithm supplies a concrete multiplier/inner-accuracy comparison. | Candidate conceptual baseline; package availability, integration, licensing and benchmark benefit still need a bounded follow-up before adoption. |
| Ipopt | Interior-point filter line-search method targeting local solutions; expects twice continuously differentiable objective/constraints and supports nonconvex NLPs. [official documentation](https://www.coin-or.org/Ipopt/) | Useful external comparator for smooth nonlinear constraints if existing solvers leave a demonstrated gap. Native-library/dependency cost and support kinks matter. No installation or speed claim here. |

The original BFGS/SLSQP/PHR publications are foundational context, but no full derivation from an inaccessible original was assumed. This report uses inspectable algorithm statements, maintained documentation and direct algebra. It does not claim an exhaustive state-of-the-art survey.

## Likelihood and numerical audit protocol

Recommendations derived for this codebase:

- Write the exact, censored and conditional-on-truncation likelihood before differentiating. Include scaling, offsets, weights and support conditions in that expression. Keep data-generation and censoring flags independently checked.
- Compare the same-parameter value, gradient and Hessian against analytic small cases and version-pinned R where the model is equivalent. Use several finite-difference step sizes and directional derivatives in smooth interior regions; use one-sided checks or explicit branch formulas at support kinks. Complex-step is not a blanket oracle for clipping, `abs`, maxima or nonanalytic distribution branches.
- Cross tail boundaries and scale extremes deliberately. Check that the value and supplied derivative describe the same function; avoid mistaking a line-search penalty for a likelihood. The rejected clipping reproduction is not evidence that every clipping path is correct.
- Test fixed parameters, redundant constraints, variable/constraint rescaling, zero or very large weights, failed inner solves, infeasible initial candidates, budget exhaustion and post-polish diagnostic recomputation.
- Keep proof obligations modest: mathematical equivalence where derivable, numerical evidence where measured, and unverified assumptions where necessary. A small Hessian eigenvalue can reflect identifiability; a positive full Hessian is not the only relevant criterion at an active constrained optimum.

## Performance at comparable validity

Use time to an independently accepted solution, with timeout/failure outcomes retained. Report solver-only and end-to-end fit cost, first-attempt success, restart work, evaluations, memory and variability. Compare matched feasible sets and likelihoods; include absolute timing as well as relative ratios. Dolan and Moré's [performance profiles](https://optimization-online.org/wp-content/uploads/2001/03/300.pdf) provide a useful distributional comparison across problems. They do not choose the validity gate for this project; that must be fixed before measurement. Tiny times, compilation/cache warmup, BLAS threads and censoring workload mix require explicit experimental controls.

## Resolution and remaining uncertainty

The next protocol decision should select block-specific feasibility/stationarity/complementarity tolerances, independent reference routes, representative cases, and finite budgets. No automatic loosening, new solver or default change follows from this research. Direct verification of the Gill–Murray–Wright passage, the exact installed SciPy paths, empirical false-acceptance probes and full mathematical/performance audits remain open work in the existing tickets. Research branches are evidence assets and must not be merged.
