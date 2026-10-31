"""Quick console dump of one harness result file (development aid)."""

import json
import sys

d = json.load(open(sys.argv[1]))
print(d["case"], d["path"], d["fit"])
print("model_mib", round(d["model_mib"], 3), "cache_after_fit", d["cache_after_fit"])
print("eviction", d["fit_entry_eviction"])
for op, rows in d["cells"].items():
    for r in rows:
        t = r["traced"]
        b = r["burst"]
        c1 = t["cache_after_one_call"]
        print(
            f'{op:22s} m={r["m"]:>6} first {r["first_call"]:8.2f} '
            f'same {r["same_data"]["median_ms"]:8.2f} chg {r["changing_data"]["median_ms"]:8.2f}'
            f' | peak {t["peak_mib"]:7.2f} ret {t["retained_mib"]:7.2f} '
            f'cache1 {c1["total_mib"]:7.2f} ({c1["raw_entries"]}/{c1["assembled_entries"]})'
            f' mdl+ {t["model_mib_delta"]:.3f}'
            f' | burst cache {b["cache"]["total_mib"]:7.1f} '
            f'({b["cache"]["raw_entries"]}/{b["cache"]["assembled_entries"]}) rss '
            f'{b["rss_before_mib"]:.0f}->{b["rss_after_burst_mib"]:.0f}->{b["rss_after_clear_mib"]:.0f}'
        )
for op, r in d["instrumented"].items():
    print("instr", op, r["first_call"], r["repeat_same_data"])
for p in d["profile_quantile_m1000"][:8]:
    print("  prof", p["func"], p["ncalls"], round(p["tottime_share"], 3))
for k in ("proto_interaction", "proto_rstyle", "proto_shift_derivative_share"):
    if k in d:
        for m, r in d[k].items():
            if isinstance(r, dict) and "m" in r:
                print(k, m, {kk: ((round(v["median_ms"], 2), v.get("bitwise_equal_library"),
                                   v.get("curves_built"), round(v.get("retained_mib", 0), 2),
                                   round(v.get("peak_mib", 0), 2))
                                  if isinstance(v, dict) else v)
                             for kk, v in r.items() if kk != "m"})
            else:
                print(k, m, r)
print("key_cost", d["key_cost"])
print("rss", d["rss_start_mib"], d["rss_end_mib"], d["maxrss_mib"])
