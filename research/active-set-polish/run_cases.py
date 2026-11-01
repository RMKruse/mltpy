"""Prototype polish on the #178 acceptance cases, from 11 starts each.

Usage, from the repository root::

    PYTHONPATH=src:research/active-set-polish .venv/bin/python \\
        research/active-set-polish/run_cases.py [--json OUT]

The cases, the start perturbation and the R comparison are imported from
``tests/test_optimizer_start_sensitivity.py``, so "ratio" is exactly that
test's ``max |θ − θ_R| / (atol + rtol·|θ_R|)`` (≤ 1 passes).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from harness import build_problem, capture
from polish import polish

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "start_sensitivity", ROOT / "tests" / "test_optimizer_start_sensitivity.py"
)
assert _spec is not None and _spec.loader is not None
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def ratio(theta: np.ndarray, theta_r: np.ndarray) -> float:
    return float(
        np.max(np.abs(theta - theta_r) / (T._ATOL + T._RTOL * np.abs(theta_r)))
    )


def run_case(name: str) -> list[dict[str, Any]]:
    build, _ = T._CASES[name]
    fit, theta_r = build()
    out = []
    for k in T._SEEDS:
        mp = pytest.MonkeyPatch()
        try:
            T._perturbed_starts(mp, k)
            with capture(T) as calls, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                theta_fit = fit()
        finally:
            mp.undo()
        call = calls[-1]
        theta_hat = np.asarray(call.result.theta, dtype=np.float64)
        prob = build_problem(call.kwargs)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = polish(prob, theta_hat)
        dt = time.perf_counter() - t0
        out.append(
            {
                "case": name,
                "k": k,
                "same_theta": bool(np.array_equal(theta_hat, theta_fit)),
                "ratio_before": ratio(theta_hat, theta_r),
                "ratio_after": ratio(res.theta, theta_r),
                "ratio_candidate": ratio(res.candidate, theta_r),
                "accepted": res.accepted,
                "reason": res.reason,
                "kkt_before": res.kkt_before,
                "kkt_after": res.kkt_after,
                "viol_before": res.viol_before,
                "viol_after": res.viol_after,
                "dnll": res.nll_after - res.nll_before,
                "n_iter": res.n_iter,
                "n_active": res.n_active,
                "added": res.n_added,
                "dropped": res.n_dropped,
                "seconds": dt,
                "history": res.history,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--case", action="append")
    args = ap.parse_args()
    names = args.case or sorted(T._CASES)
    rows: list[dict[str, Any]] = []
    hdr = (
        f"{'case':28s} {'worst before':>12s} {'worst after':>11s} {'accepted':>8s} "
        f"{'KKT before (med)':>16s} {'KKT after (med)':>15s} {'ms (med)':>8s}"
    )
    print(hdr)
    for name in names:
        rs = run_case(name)
        rows += rs
        acc = sum(r["accepted"] for r in rs)
        print(
            f"{name:28s} {max(r['ratio_before'] for r in rs):12.2f} "
            f"{max(r['ratio_after'] for r in rs):11.2f} {acc:5d}/{len(rs):<2d} "
            f"{np.median([r['kkt_before'] for r in rs]):16.1e} "
            f"{np.median([r['kkt_after'] for r in rs]):15.1e} "
            f"{1e3 * np.median([r['seconds'] for r in rs]):8.1f}"
        )
        for r in rs:
            if not r["accepted"] or not r["same_theta"] or r["ratio_after"] > 1.0:
                print(
                    f"    k={r['k']}: {r['reason']}; ratio {r['ratio_after']:.2f} "
                    f"(candidate {r['ratio_candidate']:.2f}); "
                    f"same_theta={r['same_theta']}; {r['history'][-2:]}"
                )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
