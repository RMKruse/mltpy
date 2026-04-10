1. Shift primary comparison from theta to functional outputs

  Theta is an internal parameterization detail. What matters is what the
  model predicts. Compare:
  - CDF at a dense grid (not just 10 points — use 50-100)
  - PDF / density (tests the gradient of the transformation)
  - Quantiles at standard probability levels (0.01, 0.05, 0.1, 0.25, 0.5,
  0.75, 0.9, 0.95, 0.99)
  - Hazard rate and cumulative hazard (critical for survival analysis
  users)
  - Log-likelihood (already covered, keep it)

  Keep theta comparison as informational (reported but non-blocking),
  exactly as we just implemented.

  2. Expand case coverage systematically

  Current gaps in the 20-case suite:
```
  ┌────────────────────┬──────────────────────────────────────────────┐
  │        Gap         │                Why it matters                │
  ├────────────────────┼──────────────────────────────────────────────┤
  │ Coxph model        │ No case_06 — only BoxCox and Colr are tested │
  │                    │  for tram                                    │
  ├────────────────────┼──────────────────────────────────────────────┤
  │ Covariates +       │ case_08 tests covariates with exact obs; no  │
  │ censoring          │ covariate + right/left/interval cases        │
  ├────────────────────┼──────────────────────────────────────────────┤
  │ Left censoring     │ cases 03/04 exist but only order 4/6 — no    │
  │                    │ stress tests (heavy censoring, small n)      │
  ├────────────────────┼──────────────────────────────────────────────┤
  │ Interval censoring │ No heavy-interval-censoring case analogous   │
  │  stress            │ to case_11                                   │
  ├────────────────────┼──────────────────────────────────────────────┤
  │ Higher order (10+) │ Only up to order 8; higher orders stress     │
  │                    │ numerical stability                          │
  ├────────────────────┼──────────────────────────────────────────────┤
  │ Logistic base      │ Only Colr uses it; no standalone MLT +       │
  │ distribution       │ logistic test                                │
  ├────────────────────┼──────────────────────────────────────────────┤
  │ Near-degenerate    │ All observations nearly identical, or all    │
  │ data               │ censored except 1-2                          │
  └────────────────────┴──────────────────────────────────────────────┘
```
  3. Ask the R authors for canonical test cases

  This is where your relationship is most valuable. Specifically ask for:

  - Cases where mlt is known to struggle — edge cases they've fixed bugs
  for, or cases where auglag needs many restarts. These reveal shared
  numerical challenges.
  - Their tolerance expectations — what theta precision do they consider
  meaningful? They likely know which coefficients are identifiable.
  - Whether auglag guarantees a global optimum — if not, comparing to R's
  theta is comparing to one local optimum among many. They may confirm
  that functional outputs (CDF, density) are the right comparison targets.
  - Real-world benchmark datasets — e.g., the GBSG2 breast cancer dataset
  used in their papers, bodyfat, or wine datasets from the TH.data
  package.

  4. Gradient verification against finite differences

  Not a comparison to R, but critical for correctness:

  # For every censoring type × base distribution:
  # Compare analytical gradient vs finite-difference gradient
  # at multiple theta values (initial, mid-optimization, converged)

  This catches bugs that R comparison might miss (if both implementations
  have the same bug, theta and LL would match but both would be wrong).

  5. Practical recommendation for structure
```
  validation/
    references/          # R-generated reference data (existing)
    r_scripts/
      generate_functional_references.R   # CDF, PDF, quantile, hazard at
  dense grids
      generate_stress_cases.R            # edge cases from section 2
    run_validation.py    # updated to compare functional outputs
```
  The R script would export per-case:
  - density_grid.npy, density_values.npy
  - quantile_probs.npy, quantile_values.npy
  - hazard_grid.npy, hazard_values.npy

  If I had to pick the single highest-value action: ask the authors
  whether they consider theta comparison meaningful under
  non-identifiability, and get 3-4 real-data benchmark cases from them.
  That would settle the validation philosophy and give you cases that
  matter in practice.
