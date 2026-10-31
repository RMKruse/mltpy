# Interaction Hessian route: evidence for #148

**Recommendation: choose an analytic Hessian for the smooth interaction path.**
Keep the library unchanged until the numerical changes below are explicitly
accepted in [#148](https://github.com/RMKruse/mltpy/issues/148). Gradient
differencing remains excluded. This report supplies the route recommendation;
it does not resolve the human decision or authorize a production replacement.

Measured on 2026-10-31 against `dev` at
`d2e252d538963d5714e02dbbc71f81e80c5a268b`, after the false-stall fix and
Kronecker design reuse. Both prerequisites are present in that commit.
All additions are research files; no library, defaults, exports, fixtures, or
existing tests changed. The standalone public-boundary checks were agreed in
the accompanying chat.

## Why this route

- **Analytic:** materially faster, directly expresses the observed information,
  reuses the existing distribution curvature and scaling-block mathematics,
  and matches the independent unscaled R analytic implementation to machine
  precision. At 28 parameters and 10,000 rows, warm fit time falls from
  **181.27 to 61.97 ms (2.92×)** in the research prototype.
- **Gradient differencing:** faster than scalar differencing and close to the
  analytic reference on these smooth cases, but still step-dependent. Its
  kernel is 9.5–20.8× slower than the analytic candidate here. It buys no simpler
  numerical contract around clipping or Laplace kinks and remains excluded by
  the ticket. Measuring it is not acceptance of its SE changes.
- **Keep scalar differencing:** avoids an inference change today, but leaves a
  material runtime cost and cancellation error. It is a useful compatibility
  fallback, not a numerical reference for accepting a replacement.

The recommendation includes **explicitly accepting changed covariance/SE
results**. Even an exact analytic replacement does not preserve the current
finite-difference SEs. On positive-definite scalar-FD cases here, the largest
SE discrepancy from the analytic reference is **1.50%**. On four valid
28-parameter cases, scalar-FD information is indefinite and no valid Wald SE
comparison exists, while the analytic information is positive definite. These
are numerical changes to disclose, not a bitwise-preserving optimization.
This recommendation is not itself that acceptance.

## Reproduce and inspect

From the repository root, with the existing Python environment and R packages:

```sh
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/interaction-hessian-route/measure.py
Rscript research/interaction-hessian-route/reference.R
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/interaction-hessian-route/check.py
```

- [measure.py](measure.py) imports the existing
  [#139 case generators and validation](../fit-phase-baseline/harness.py), uses
  its complete interaction case matrix, and adds m=28 at n=2000 and n=10000.
  It replaces the private Hessian callables **only inside its own process**,
  restoring them afterwards. The prototype refuses clipping and Laplace at
  zero, so it cannot silently stand in for a complete production solution.
- [measurements.json](results/measurements.json) contains every recorded fit
  (216 timed fits: 12 cases × 3 routes × 6), all timings, warnings, projected
  stationarity, feasibility, SLSQP checks, matrix errors and peak allocations.
  No timed fit failed or returned `converged=False`; none was filtered out.
- [reference.R](reference.R) reconstructs R models at the **same supplied theta**
  with `dofit=FALSE`, maps R's y-fast coefficient order to Python's C order,
  and writes the reordered raw information matrices and
  [R comparison metadata](results/r_reference.json). It does not compare
  Hessians at different solver endpoints or use regularized covariance as a
  substitute for raw information.
- [check.py](check.py) checks the public `likelihood.hessian` boundary with the
  candidate temporarily installed: two independent worked Normal examples
  (including scaling, weights and offset), explicit rejection of excluded
  nonsmooth cases, and all three same-theta R matrices. The worked unscaled
  example is exact to 1e-12 with the candidate; the existing scalar FD has
  2.78e-7 absolute error on that example.

Python 3.13.13, NumPy 2.5.4, SciPy 1.18.1, macOS 26.6.2 arm64, Apple M5 Pro;
Accelerate BLAS, `VECLIB_MAXIMUM_THREADS=OMP_NUM_THREADS=1`, no explicit
`OPENBLAS_NUM_THREADS`. R 4.6.1, mlt 1.8.4, basefun 1.2.6. Restart seed is 0.
The raw JSON captures interpreter, platform and thread settings.

Each route has six fresh-model fits per case; the first is excluded from its
five-repetition warm median. These are warm-process timings, not fresh-process
cold starts. Routes are interleaved for kernel timings; whole-fit repetitions
run consecutively per route. GC is enabled, load is uncontrolled, and some
cells have substantial spread: m28/n2000 analytic 23.27–35.35 ms, scaled
m9/n1000 scalar FD 9.82–16.74 ms. Do not interpret their precise speedups as
stable estimates. At m28/n10000, scalar FD is 175.29–204.36 ms and analytic
61.20–62.93 ms, comfortably beyond the 5% noise floor. Historical #139/#147
absolute times are not substituted for this rebaseline.

## Measured fitting and inference results

Times are warm median ms, ordered scalar FD / gradient FD / analytic.
The SE column compares scalar FD with the analytic reference **at identical
theta**, only when both information matrices are positive definite.

| Case | m | n | Fit ms: scalar / gradient / analytic | Scalar → analytic | Max scalar SE discrepancy |
|---|---:|---:|---|---:|---|
| ia_R_normal_3x3 | 9 | 300 | 4.67 / 2.10 / 1.64 | 2.85× | 0.040% |
| ia_R_logistic_3x3 | 9 | 300 | 7.13 / 3.55 / 2.96 | 2.41× | 0.178% |
| ia_scaled_R_stram_3x3+1 | 10 | 300 | 8.27 / 6.32 / 5.93 | 1.39× | 0.037% |
| ia_bs3x1_m8_n250 | 8 | 250 | 4.50 / 3.52 / 3.27 | 1.38× | 0.066% |
| ia_bs3x1_m8_n1000 | 8 | 1000 | 6.19 / 3.09 / 2.43 | 2.54× | 0.099% |
| ia_bs3x1_m8_n4000 | 8 | 4000 | 15.47 / 7.22 / 5.73 | 2.70× | 0.040% |
| ia_bs4x2_m15_n1000 | 15 | 1000 | 17.88 / 13.07 / 12.01 | 1.49× | 1.496% |
| ia_bs6x3_m28_n250 | 28 | 250 | 33.02 / 21.55 / 20.66 | 1.60× | undefined; scalar H indefinite |
| ia_bs6x3_m28_n1000 | 28 | 1000 | 45.47 / 25.13 / 24.07 | 1.89× | undefined; scalar H indefinite |
| ia_scaled_bs3x1_m9_n1000 | 9 | 1000 | 14.41 / 4.91 / 3.66 | 3.94× | 0.201% |
| ia_bs6x3_m28_n2000 | 28 | 2000 | 57.10 / 25.89 / 30.34 | 1.88× | undefined; scalar H indefinite |
| ia_bs6x3_m28_n10000 | 28 | 10000 | 181.27 / 67.96 / 61.97 | 2.92× | undefined; scalar H indefinite |

Across all recorded fits, theta and log-likelihood are numerically unchanged
from the scalar-FD baseline, projected stationarity is at most 1.13e-5, and
minimum constraint slack is at least -3.4e-8. Baseline likelihoods agree with
independent SLSQP to at most 7.54e-9; unchanged route likelihoods inherit that
cross-check. Maximum fitted |h| is 4.72: none of these timing cases clips.

At m28/n10000, Hessian-only median time is **111.76 / 5.42 / 0.35 ms**.
All three fit peaks are **8.93 MiB** in tracemalloc. Component basis matrices
are already warm in the shared caches before this tracing; dense Kronecker
designs are rebuilt per fit. This is an incremental warm-fit peak, not total
process memory. No retained-memory or RSS improvement
is claimed. The candidate creates dense n×m temporaries, as the current code
does; this does not settle the separate contraction/memory question in #138.

The scalar-FD / analytic smallest eigenvalues at m28 are:

| n | Scalar FD | Analytic | Analytic condition number |
|---:|---:|---:|---:|
| 250 | -5.06e-4 | 2.75e-6 | 4.54e6 |
| 1000 | -8.69e-4 | 8.05e-5 | 6.21e5 |
| 2000 | -3.05e-3 | 2.38e-4 | 4.21e5 |
| 10000 | -7.57e-3 | 2.17e-3 | 2.29e5 |

Gradient-FD SEs differ from the analytic reference by at most 2.35e-6 relative
on this matrix. That is evidence about these cases, not an acceptance bound
for other steps, parameter scales, tails, or distributions.

## Formula and independent reference

For observation i, define design rows A=a(y)⊗b(x), D=a'(y)⊗b(x), t=vec_C(Θ),
g=A·t, v=D·t>0, scaling row s, f=exp(s·γ/2), z=f·g, and h=z+offset.
Without scaling f=1 and the γ blocks are absent. Let
r(h)=-∂log f_Z(h)/∂h and c(h)=-∂²log f_Z(h)/∂h².
The per-observation NLL is -log f_Z(h)-log(v)-s·γ/2. Away from clipping/kinks:

```text
H_tt = w [c f² A Aᵀ + D Dᵀ/v²]
H_tγ = (w f/2) (c z + r) A sᵀ
H_γγ = (w z/4) (c z + r) s sᵀ
```

Sum over rows, with H_γt=H_tγᵀ. Offset enters r,c through h; the nonlinear
chain factors use z, **not h including offset**. The scaling term in log(vf)
is linear in γ and has zero Hessian. These blocks match the existing
`_hess_none` and `_add_scaled_gamma_blocks_h` in
[likelihood.py](../../src/mltpy/likelihood.py), with the interaction design
replacing the y design and no shift block. `_d2_logpdf` already supports all
seven distributions. Production implementation can reuse those helpers rather
than adding distribution formulas.

Independent primary reference: installed R `mlt:::.mlt_hessian_exact` computes
unscaled information analytically using the design and derivative-design
crossproducts. R `.mlt_setup` and the fitted Hessian closure use
`stats::optimHess` on the analytic gradient for scaling, with a numerical
fallback. Inspect with `print(mlt:::.mlt_hessian_exact)` and
`print(mlt:::.mlt_setup)`; the executable same-theta reference is
[reference.R](reference.R). The
[existing fixture generator](../../reference/generate_reference.R) documents
the parameter permutation and scaling convention.

| Same-theta reference | Relative max H error | Max relative SE error | Nature of reference |
|---|---:|---:|---|
| R Normal | 3.49e-16 | 2.22e-15 | Independent analytic implementation |
| R Logistic | 2.43e-16 | 8.66e-15 | Independent analytic implementation |
| R scaled Normal | 6.23e-7 | 3.10e-7 | R numerical observed information |

All three pass componentwise `rtol=1e-6, atol=1e-10` against the R information.
R scaled information satisfies the ticket's permitted R reference route, but
is **not an independent analytic proof**. The weighted/offset worked example
checks the nonlinear scaling blocks independently. This is sufficient to
recommend a route; it is not full production acceptance coverage.

## Proposed acceptance for the implementation ticket

1. Explicitly accept that smooth-region covariance/SEs change to the analytic
   observed information, including the 1.50% discrepancy and indefinite-FD
   cases above. Do not retain noisy scalar FD as the truth or accept gradient
   differencing by implication.
2. Use analytic worked examples plus independent same-theta R raw information
   and unregularized covariance, with `rtol=1e-6, atol=1e-10` on well-conditioned
   fixtures. For ill-conditioned matrices, check H first, report eigenvalues
   and conditioning, and require separately justified covariance tolerances;
   never silently relax matrix tolerances to retain old SEs.
3. Cover all supported distributions, weights (including zero weights), offsets,
   multiple scaling columns, infeasible derivatives, and log response bases.
   Preserve float64, finite-result errors, and public validation. Keep existing
   R parity, vcov, interaction scaling, optimizer, and mutation-isolation tests.
4. Specify clipping/Laplace behavior before replacement. The scalar likelihood
   clips h to ±30; the current gradient omits that clip's derivative. The smooth
   formula above is not its Hessian outside the clip, and a classical Hessian
   may not exist at a boundary. Laplace has no classical Hessian at h=0; its
   scalar-FD kink curvature depends on step size. Prefer retaining the current
   FD behavior as a narrowly guarded fallback for these cases, with explicit
   tests at and across the boundaries, over changing their semantics in this
   performance task. This guard has not been designed or measured here.
5. Apply the chosen kernel to both Newton-CG polish and the eager final Hessian;
   revalidate solver endpoints against projected stationarity, feasibility,
   SLSQP and R. The unchanged theta/likelihood result here is a target to verify,
   not a guarantee for every Newton-CG trajectory.
6. Re-run the full #139 matrix after implementation (including unaffected
   controls), ≥5 warm reps and the valid large case, reporting failures, spreads,
   and peak memory. Require a >5% fit benefit on the large case. No solver-default,
   tolerance, clipping, constraint or covariance-regularization change is bundled.

Not covered by this decision study: production dispatch/fallback, other base
distributions beyond Normal/Logistic, large scaled fits, log-response bases,
censoring, truncation, or memory-contraction alternatives. Inference still uses
unconstrained observed information, as current `vcov` does; positive definiteness
does not establish Wald coverage when monotonicity constraints bind.

Validation of this research-only change: 2045 existing tests passed (six
warnings), the standalone worked examples and three R checks passed, and Ruff
passed on source, tests and the new research scripts. Strict mypy still reports
the same three pre-existing redundant casts in `basis.py:540`, `basis.py:565`
and `optimizer.py:288`; no source file was changed to silence them.
