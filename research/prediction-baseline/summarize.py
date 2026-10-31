"""Build summary tables (markdown) from results/*.json."""

import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
cases = {p.stem: json.loads(p.read_text()) for p in sorted(R.glob("*.json"))}
out: list[str] = []
w = out.append


def f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


w("## A. Fitted models (independently validated)\n")
w("| case | path | θ length | converged | ℓ | ℓ − ℓ_SLSQP | max abs Δθ vs SLSQP | valid |")
w("|---|---|---|---|---|---|---|---|")
for n, d in cases.items():
    x = d["fit"]
    w(f"| {n} | {d['path']} | {x['n_params']} | {x['converged']} | {x['ll']:.4f} | "
      f"{x['ll_minus_slsqp']:.1e} | {x['max_abs_dtheta_vs_slsqp']:.1e} | {x['valid']} |")

w("\n## B. Runtime and memory per operation\n")
w("Times in ms (median). *first* = first call on new data with cleared caches; "
  "*same* = repeated call on identical inputs; *changing* = new inputs every call. "
  "Memory in MiB: traced peak and retained after one call (output excluded), "
  "global-cache bytes after one call (raw/assembled entries), then cache bytes "
  "and process RSS (before → after burst → after clearing caches) after a "
  "burst of changing-data calls (40 calls; 12 at m = 100,000). Model-instance "
  "array bytes never changed during prediction (Δ = 0 in every cell).\n")
w("| case | op | m | first | same | changing | peak | retained | cache after 1 call | burst cache | RSS burst |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
mdl_delta = 0.0
for n, d in cases.items():
    for op, rows in d["cells"].items():
        for r in rows:
            t = r["traced"]
            b = r["burst"]
            c1 = t["cache_after_one_call"]
            mdl_delta = max(mdl_delta, abs(t["model_mib_delta"]))
            w(f"| {n} | {op} | {r['m']:,} | {f(r['first_call'])} | "
              f"{f(r['same_data']['median_ms'])} | {f(r['changing_data']['median_ms'])} | "
              f"{f(t['peak_mib'])} | {f(t['retained_mib'])} | "
              f"{f(c1['total_mib'])} ({c1['raw_entries']}/{c1['assembled_entries']}) | "
              f"{f(b['cache']['total_mib'], 1)} ({b['cache']['raw_entries']}/"
              f"{b['cache']['assembled_entries']}, {b['calls']} calls) | "
              f"{b['rss_before_mib']:.0f} → {b['rss_after_burst_mib']:.0f} → "
              f"{b['rss_after_clear_mib']:.0f} |")
w(f"\nMax |model-instance Δ| over all cells: {mdl_delta:.3g} MiB.\n")

w("\n## C. Cache traffic per call (m = 1,000; instrumented run)\n")
w("Counts for the first call on new data / a repeated call on the same data. "
  "`dB on value-only` = assembled-cache misses inside `evaluate()` (a derivative "
  "matrix built and cached although the caller only needs values).\n")
w("| case | op | raw hit/miss | assembled hit/miss | derivative() calls | dB on value-only | key bytes hashed |")
w("|---|---|---|---|---|---|---|")
for n, d in cases.items():
    for op, r in d["instrumented"].items():
        a, s = r["first_call"], r["repeat_same_data"]
        w(f"| {n} | {op} | {a['raw_hit']}/{a['raw_miss']} → {s['raw_hit']}/{s['raw_miss']} | "
          f"{a['asm_hit']}/{a['asm_miss']} → {s['asm_hit']}/{s['asm_miss']} | "
          f"{a['derivative_calls']} | {a['dB_built_on_value_only']} → "
          f"{s['dB_built_on_value_only']} | {a['key_bytes_hashed']:,} → "
          f"{s['key_bytes_hashed']:,} |")
k = next(iter(cases.values()))["key_cost"]
w("\nContent-key cost (`ascontiguousarray(y).tobytes()` + hash, fresh array): "
  + ", ".join(f"m = {int(m):,}: {v['tobytes_plus_hash_us']:.1f} µs" for m, v in k.items())
  + ".\n")

w("\n## D. Fit-entry eviction by prediction\n")
w("`score()` on the training data right after `fit()` vs after 40 changing-data "
  "CDF calls (m = 1,000) plus 40 quantile calls (m = 100).\n")
w("| case | fit entries (raw/asm) | surviving | score hot (ms) | score after burst (ms) |")
w("|---|---|---|---|---|")
for n, d in cases.items():
    e = d["fit_entry_eviction"]
    w(f"| {n} | {e['fit_raw_entries']}/{e['fit_asm_entries']} | "
      f"{e['surviving_raw']}/{e['surviving_asm']} | {e['score_hot_ms']:.3f} | "
      f"{e['score_after_burst_ms']:.3f} |")

w("\n## E. Candidate prototype: interaction value-only midpoint evaluation\n")
w("`_predict_quantile_interaction` with `y_basis.evaluate` replaced (harness only) "
  "by (i) the library call, (ii) values only, uncached, (iii) values + the same "
  "dB work, uncached — separating derivative work from cache insertion. "
  "Retained/peak traced MiB per call; ✓ = bitwise equal to the library output.\n")
w("| case | m | library ms | value-only ms | +dB uncached ms | speedup | retained lib → value-only | peak lib → value-only | equal |")
w("|---|---|---|---|---|---|---|---|---|")
for n, d in cases.items():
    for m, r in d.get("proto_interaction", {}).items():
        L, V, D = r["library"], r["value_only_uncached"], r["with_dB_uncached"]
        eq = "✓" if V["bitwise_equal_library"] and D["bitwise_equal_library"] else "✗"
        w(f"| {n} | {int(m):,} | {f(L['median_ms'])} | {f(V['median_ms'])} | "
          f"{f(D['median_ms'])} | {L['median_ms'] / V['median_ms']:.2f}× | "
          f"{f(L['retained_mib'])} → {f(V['retained_mib'])} | "
          f"{f(L['peak_mib'])} → {f(V['peak_mib'])} | {eq} |")

w("\n## F. Candidate prototype: call-local reuse of the R-style inverse curve\n")
w("Copy of the R-style loop (`none` must equal the library bitwise — it does), "
  "with the per-shift curve (CDF grid → spline → sorted unique inverse) reused "
  "for consecutive equal shifts or for any shift already seen in this call. "
  "The interp/saturation step still runs per probability.\n")
w("| case | op | m | distinct shifts | none ms | consecutive ms (curves) | unique ms (curves) | equal | saturation flag |")
w("|---|---|---|---|---|---|---|---|---|")
for n, d in cases.items():
    for key, r in d.get("proto_rstyle", {}).items():
        op = key.rsplit("_m", 1)[0]
        N, C, U = r["none"], r["consecutive"], r["unique"]
        eq = all(x["bitwise_equal_library"] for x in (N, C, U))
        sat = {N["saturated_flag"], C["saturated_flag"], U["saturated_flag"]}
        w(f"| {n} | {op} | {r['m']:,} | {r['distinct_shifts']:,} | {f(N['median_ms'])} | "
          f"{f(C['median_ms'])} ({C['curves_built']:,}) | {f(U['median_ms'])} "
          f"({U['curves_built']:,}) | {'✓' if eq else '✗'} | "
          f"{'same' if len(sat) == 1 else 'DIFFERS'} (lib warned: {r['library_warned']}) |")

w("\n## G. Shift CDF: unused derivative work\n")
w("| case | m | predict (new data) ms | derivative() share ms | share |")
w("|---|---|---|---|---|")
for n, d in cases.items():
    for m, r in d.get("proto_shift_derivative_share", {}).items():
        w(f"| {n} | {int(m):,} | {f(r['predict_new_data_ms'])} | "
          f"{f(r['derivative_call_ms'])} | "
          f"{100 * r['derivative_call_ms'] / r['predict_new_data_ms']:.0f}% |")

w("\n## H. Quantile profile (m = 1,000, top self-time)\n")
for n, d in cases.items():
    top = ", ".join(f"`{p['func'].split(':', 2)[2]}` {100 * p['tottime_share']:.0f}%"
                    for p in d["profile_quantile_m1000"][:5])
    w(f"- **{n}**: {top}")

w("\n## Environment\n")
e = next(iter(cases.values()))["env"]
w(", ".join(f"{k}={v}" for k, v in e.items()))
w("\nPer-process RSS (start → end, ru_maxrss): " + "; ".join(
    f"{n} {d['rss_start_mib']:.0f} → {d['rss_end_mib']:.0f} ({d['maxrss_mib']:.0f})"
    for n, d in cases.items()))
Path(sys.argv[1] if len(sys.argv) > 1 else R.parent / "tables.md").write_text(
    "\n".join(out) + "\n")
