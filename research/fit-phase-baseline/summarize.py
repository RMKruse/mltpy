"""Turn results/*.json into Markdown evidence tables (wayfinder ticket #139)."""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

R = Path(__file__).parent / "results"
rows = [json.loads(p.read_text()) for p in sorted(R.glob("*.json"))]
order = ["shift", "shift+X", "shift+X+right", "shift+scaling", "interaction",
         "interaction+scaling", "profile-ci"]
rows.sort(key=lambda d: (order.index(d["family"]), d.get("n_params", 0), d["n"], d["case"]))


def ms(x: float) -> str:
    return f"{x * 1e3:,.1f}" if x < 10 else f"{x:,.1f} s"


def e(x: float | None) -> str:
    return "—" if x is None else f"{x:.1e}"


out: list[str] = []
env = rows[0]["env"]
out.append("### Environment\n")
out.append(
    f"{env['cpu']} ({env['ncpu']} logical CPUs), {env['platform']}; Python {env['python']}, "
    f"NumPy {env['numpy']}, SciPy {env['scipy']}, mltpy {env['mltpy']}; BLAS "
    f"`{env['blas'].get('name')}` ({env['blas'].get('detection method')}); thread env "
    f"{env['thread_env']}; HEAD `{env['git_head'][:10]}`, src dirty={env['git_dirty']}; "
    f"restart `random_state={env['restart_random_state']}`. Load average at case start "
    f"ranged {min(d['env']['loadavg'][0] for d in rows):.2f}–"
    f"{max(d['env']['loadavg'][0] for d in rows):.2f} (not controlled).\n")

out.append("### A. Timing distributions (all runs, nothing filtered)\n")
out.append("Times in ms unless marked. *Cold* = first fit in a fresh process. *Warm* = "
           "subsequent fresh-model fits. *Refit* = same instance refitted.\n")
out.append("| Case | n | m | Cold | Warm median [min–max] (reps) | Refit median (reps) "
           "| Raw `converged` | Restarts | Stored KKT (max) | Warnings |")
out.append("|---|---:|---:|---:|---|---|---|---:|---:|---|")
for d in rows:
    f = d["fresh_seconds"]
    w = f[1:]
    rf = d["refit_seconds"]
    sts = d["fresh_status"]
    conv = sum(s["raw_converged"] for s in sts)
    kk = [s["kkt_residual_stored"] for s in sts if s["kkt_residual_stored"] is not None]
    warns = sorted({x.split(":")[0] for s in sts for x in s["warnings"]})
    out.append(
        f"| `{d['case']}` | {d['n']} | {d.get('n_params', '')} | {ms(f[0])} | "
        f"{ms(st.median(w))} [{ms(min(w))}–{ms(max(w))}] ({len(w)}) | "
        + (f"{ms(st.median(rf))} ({len(rf)})" if rf else "—")
        + f" | {conv}/{len(sts)} | {max(s['n_restarts'] for s in sts)} | "
        f"{e(max(kk)) if kk else '—'} | {', '.join(warns) or '—'} |")

out.append("\n### B. Phase attribution (one instrumented fit per case)\n")
out.append("Exclusive phases: *solver* = all auglag attempts; *polish* = trust-ncg "
           "excluding its Hessians; *polish H* = Hessians requested inside polish; "
           "*final H* = eager post-fit Hessian; *score* = eager score matrix; *rest* = "
           "fit() minus those (validation, copies, constraint build). Percent of the "
           "instrumented fit. Obj = likelihood+gradient evaluations by phase; FD = "
           "scalar interaction log-likelihood calls made by the FD Hessian.\n")
out.append("| Case | Instr. fit | Solver | Polish | Polish H (calls) | Final H | Score "
           "| Rest | Obj solver/polish | FD calls |")
out.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
for d in rows:
    if d["family"] == "profile-ci":
        continue
    p = d["phases"]
    s, c = p["seconds"], p["calls"]
    T = d["instrumented_seconds"]
    sol = s.get("solver_attempts", 0)
    ph = s.get("polish_hessian", 0)
    po = s.get("polish", 0) - ph
    fh = s.get("final_hessian", 0)
    sc = s.get("score_matrix", 0)
    rest = T - sol - s.get("polish", 0) - fh - sc
    pc = lambda x: f"{100 * x / T:.0f}%"  # noqa: E731
    oc = p["objective_calls_by_phase"]
    fd = sum(p["ll_interaction_calls_by_phase"].values())
    out.append(
        f"| `{d['case']}` | {ms(T)} | {pc(sol)} | {pc(po)} | {pc(ph)} "
        f"({c.get('polish_hessian', 0)}) | {pc(fh)} | {pc(sc)} | {pc(rest)} | "
        f"{oc.get('solver_attempts', 0)}/{oc.get('polish', 0)} | {fd:,} |")

out.append("\n### C. Outcome validation (instrumented fit; independent of solver flags)\n")
out.append("*Stored KKT* is auglag's pre-polish value on `result_`. *Pre/Final stat.* = "
           "‖∇NLL − A_actᵀμ‖∞ with μ ≥ 0 by NNLS over rows with slack ≤ 1e-6 "
           "(recomputed from the public `negative_log_likelihood`). ΔNLL polish = "
           "after − before. SLSQP = `optimize(solver='slsqp', tol=1e-12)` from the "
           "default start; Δℓ = auglag − SLSQP (positive = auglag better).\n")
out.append("| Case | Raw conv. | Stored KKT | Polish | ΔNLL polish | Pre stat. | "
           "Final stat. | Active rows | Min slack | SLSQP Δℓ | SLSQP max|Δθ| | R reference |")
out.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
for d in rows:
    v = d["validation"]
    s = d["instrumented_status"]
    if d["family"] == "profile-ci":
        rr = v.get("r_reference", {})
        out.append(f"| `{d['case']}` | {s['raw_converged']} | {e(s['kkt_residual_stored'])} "
                   f"| — | — | — | — | — | — | — | — | max|ΔCI| {rr.get('max_abs_dCI_vs_R', 0):.1e}"
                   f", non-finite {rr.get('nonfinite_ci_entries')} |")
        continue
    pol = d["phases"]["polish"]
    if pol:
        pa = "accepted" if pol[-1]["accepted"] else "rejected"
        dn = (f"{pol[-1]['nll_after'] - pol[-1]['nll_before']:.1e}"
              if pol[-1]["accepted"] else "—")
    else:
        pa, dn = "skipped", "—"
    fin, pre = v["final"], v.get("pre_polish")
    sl = v.get("slsqp", {})
    rr = v.get("r_reference")
    rtxt = "—"
    if rr:
        bits = []
        if "max_abs_dTheta_vs_R" in rr:
            bits.append(f"max|ΔΘ| {rr['max_abs_dTheta_vs_R']:.1e}")
        if "abs_dll_vs_R" in rr:
            bits.append(f"|Δℓ| {rr['abs_dll_vs_R']:.1e}")
        if "abs_dgamma_vs_R" in rr:
            bits.append(f"|Δγ| {rr['abs_dgamma_vs_R']:.1e}")
        rtxt = ", ".join(bits)
    out.append(
        f"| `{d['case']}` | {s['raw_converged']} | {e(s['kkt_residual_stored'])} | {pa} | {dn} | "
        f"{e(pre['stationarity_inf']) if pre else '—'} | {e(fin['stationarity_inf'])} | "
        f"{fin['n_active']} | {fin['min_slack'] if fin['min_slack'] is None else f'{fin['min_slack']:.1e}'} | "
        + (f"{sl['ll_auglag_minus_slsqp']:.1e} | {sl['max_abs_dtheta']:.1e}"
           if "ll" in sl else f"{sl.get('error', '?')} | —")
        + f" | {rtxt} |")

out.append("\n### D. Memory (one traced fit; tracemalloc + process high-water)\n")
out.append("| Case | Traced peak MiB | Retained w/ basis caches MiB | Retained model only MiB "
           "| Score matrix MiB | maxRSS after cold fit MiB |")
out.append("|---|---:|---:|---:|---:|---:|")
for d in rows:
    mm = d["memory"]
    sc = mm["score_matrix_mib"]
    out.append(f"| `{d['case']}` | {mm['traced_peak_mib']:.2f} | "
               f"{mm['retained_with_caches_mib']:.2f} | {mm['retained_model_only_mib']:.2f} "
               f"| {'—' if sc is None else f'{sc:.3f}'} | {d['maxrss_after_cold_mib']:.0f} |")

out.append("\n### E. Interaction Hessian: scalar FD (current) vs central difference of the "
           "analytic gradient\n")
out.append("| Case | m | max|ΔH| | rel. | max rel. SE diff | min eig H | cond H | "
           "grad-FD time | current final-H time |")
out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for d in rows:
    h = d["validation"].get("hessian_fd_vs_gradfd")
    if not h:
        continue
    out.append(f"| `{d['case']}` | {d['n_params']} | {h['max_abs_diff']:.1e} | "
               f"{h['rel_diff']:.1e} | {h.get('max_rel_se_diff', float('nan')):.1e} | "
               f"{h['min_eig_fd']:.2e} | {h['cond_fd']:.1e} | {ms(h['seconds_gradfd'])} | "
               f"{ms(d['phases']['seconds'].get('final_hessian', 0))} |")

out.append("\n### F. Solver attempts (instrumented fit)\n")
out.append("| Case | Attempt | Seconds | Converged | KKT | Outer | Inner | Message |")
out.append("|---|---:|---:|---|---:|---:|---:|---|")
for d in rows:
    if d["family"] == "profile-ci":
        continue
    for i, a in enumerate(d["phases"]["solver_attempts"]):
        out.append(f"| `{d['case']}` | {i} | {ms(a['seconds'])} | {a['converged']} | "
                   f"{e(a['kkt'])} | {a['outer']} | {a['inner']} | {a['message'][:70]} |")

out.append("\n### G. Profile-CI workload (whole fit + `confint(type='profile')`)\n")
out.append("| Case | n | m | Cold | Warm | optimize() calls | solver attempts | "
           "time in optimize() |")
out.append("|---|---:|---:|---:|---|---:|---:|---:|")
for d in rows:
    if d["family"] != "profile-ci":
        continue
    p = d["phases"]
    out.append(f"| `{d['case']}` | {d['n']} | {d['n_params']} | {ms(d['fresh_seconds'][0])} | "
               f"{', '.join(ms(x) for x in d['fresh_seconds'][1:])} | "
               f"{p['calls'].get('optimize_total', 0)} | {p['calls'].get('solver_attempts', 0)} | "
               f"{100 * p['seconds'].get('optimize_total', 0) / d['instrumented_seconds']:.0f}% |")

Path(sys.argv[1] if len(sys.argv) > 1 else "summary.md").write_text("\n".join(out) + "\n")
