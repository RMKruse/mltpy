## 1. Data Flow
```
 validation/references/case_**/
     metadata.json ─────────────────────────┐
     y.npy / y_lower.npy / y_upper.npy ─────┤
     status.npy ────────────────────────────┤
     X.npy ─────────────────────────────────┤
     theta.npy ─────────────────────────────┼──→ load_reference()
     loglik.npy ────────────────────────────┤       │
     cdf_grid.npy ──────────────────────────┤       ▼
     cdf_values.npy ────────────────────────┘   ReferenceCase
                                                    │
                                                    ├──→ fit_python_model(case)
                                                    │       │
                                                    │       ▼
                                                    │    FittedResult
                                                    │       │
                                                    └───┬───┘
                                                        │
                                                        ▼
                                               compare_results(ref, fit)
                                                        │
                                                        ▼
                                               ValidationResult
                                                        │
                                           ┌────────────┼────────────┐
                                           ▼            ▼            ▼
                                    print_report   save .md/.json   exit code
                                     (stdout)     (validation/       0 = all pass
                                                   results/)         1 = tolerance fail
                                                                     2 = no ref data

```


## 2. Model Dispatch: fit_python_model()

 Maps metadata.model × metadata.censoring to mltpy class + data preparation:

```
 metadata.model  │ metadata.censoring │ mltpy class │ base_dist  │ Data preparation
 ────────────────┼────────────────────┼─────────────┼────────────┼──────────────────────────────
 "mlt"           │ "none"             │ MLT         │ "normal"   │ fit(y) or fit(y, X=X)
 "mlt"           │ "right"            │ MLT         │ "normal"   │ CensoredData.right_censored(y, ~status.astype(bool))
 "mlt"           │ "left"             │ MLT         │ "normal"   │ CensoredData.left_censored(y, ~status.astype(bool))
 "mlt"           │ "interval"         │ MLT         │ "normal"   │ CensoredData.interval_censored(y_lower, y_upper)
 "boxcox"        │ "none"             │ BoxCox      │ (forced)   │ fit(y)
 "coxph"         │ "right"            │ Coxph       │ (forced)   │ CensoredData.right_censored(y, ~status.astype(bool))
 "colr"          │ "none"             │ Colr        │ (forced)   │ fit(y)
```

 Key detail: status.npy stores 1=event, 0=censored. mltpy's CensoredData.right_censored(y, censored=...) expects True=censored. So: censored = ~status.astype(bool)
 (invert).

 Key detail: For "mlt" with "left" or "right" censoring, the MLT constructor needs censoring=CensoringType.RIGHT / CensoringType.LEFT explicitly.

 Regression (case 08): Detected via metadata.get("regression", False). If true, load X.npy and pass to fit(y, X=X). CDF prediction uses predict(cdf_grid,
 X_new=np.zeros((len(cdf_grid), n_cov))).

 Implementation: Simple if/elif chain, not a registry or dict — there are only 4 model types and 4 censoring types, and some combinations need special data prep
 that doesn't factor cleanly.

 ---
 
 ## 3. --case Filter
```
 parser = argparse.ArgumentParser()
 parser.add_argument("--case", type=str, default=None,
                     help="Run only cases matching this prefix, e.g. 'case_01' or 'case_05_boxcox'")
 parser.add_argument("--verbose", action="store_true",
                     help="Show per-component delta details")
```
 ### In run_all_validations():
 ```
 case_dirs = sorted(ref_dir.glob("case_*"))
 if args.case:
     case_dirs = [d for d in case_dirs if d.name.startswith(args.case)]

 Prefix match — --case case_01 runs all 6 variants (200/1000 × 4/6/8), --case case_05 runs only BoxCox.
```
 ---

## 4. Tolerances
```
 TOL_THETA  = 0.05   # max absolute component-wise difference
 TOL_LOGLIK = 0.1    # absolute difference in log-likelihood
 TOL_CDF    = 0.02   # max absolute CDF difference at grid points
```