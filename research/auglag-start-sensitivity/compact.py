"""Compact a start-probe JSONL log and print corpus-wide statistics (#178).

Usage::

    .venv/bin/python research/auglag-start-sensitivity/compact.py RAW.jsonl OUT.json

The raw log keeps every row's slack (14 MB for the full suite x 11 seeds).
``OUT.json`` keeps, per test (aggregated over its auglag fits and the 11
seeds): number of fits, KKT max and median, converged count, number of
equality rows, inequality rows, the largest number of active rows (normalised
slack <= 1e-6), the largest active slack, the smallest inactive slack, the
most negative slack and polish counts; plus every test outcome that is not a
pass.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ACTIVE_TOL = 1e-6


def compact_fit(r: dict[str, Any]) -> dict[str, Any]:
    sl = r["slacks"]
    act = [s for s in sl if s <= ACTIVE_TOL]
    ina = [s for s in sl if s > ACTIVE_TOL]
    return {
        "nodeid": r["nodeid"],
        "seed": r["seed"],
        "kkt": r["kkt"],
        "converged": r["converged"],
        "n_eq": r["n_eq"],
        "n_ineq": len(sl),
        "n_active": len(act),
        "max_active_slack": max(act) if act else None,
        "min_inactive_slack": min(ina) if ina else None,
        "min_slack": min(sl) if sl else None,
        "polish": r["polish"],
    }


def _opt(values: list[float | None], fn: Any) -> float | None:
    vals = [v for v in values if v is not None]
    return fn(vals) if vals else None


def aggregate(fs: list[dict[str, Any]]) -> dict[str, Any]:
    kkts = sorted(f["kkt"] for f in fs if not math.isnan(f["kkt"]))
    return {
        "n_fits": len(fs),
        "seeds": sorted({f["seed"] for f in fs}),
        "kkt_max": kkts[-1] if kkts else None,
        "kkt_median": kkts[len(kkts) // 2] if kkts else None,
        "n_converged": sum(f["converged"] for f in fs),
        "n_eq": max(f["n_eq"] for f in fs),
        "n_ineq": max(f["n_ineq"] for f in fs),
        "n_active_max": max(f["n_active"] for f in fs),
        "max_active_slack": _opt([f["max_active_slack"] for f in fs], max),
        "min_inactive_slack": _opt([f["min_inactive_slack"] for f in fs], min),
        "min_slack": _opt([f["min_slack"] for f in fs], min),
        "polish": dict(Counter(str(f["polish"]) for f in fs)),
    }


def main() -> None:
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    recs = [json.loads(line) for line in raw.read_text().splitlines() if line]
    fits = [compact_fit(r) for r in recs if r["type"] == "fit"]
    fails = [r for r in recs if r["type"] == "outcome" and r["outcome"] != "passed"]
    per_test: dict[str, list[dict[str, Any]]] = {}
    for f in fits:
        per_test.setdefault(str(f["nodeid"]), []).append(f)
    tests = {nid: aggregate(fs) for nid, fs in sorted(per_test.items())}
    out.write_text(
        json.dumps(
            {"active_tol": ACTIVE_TOL, "tests": tests, "non_passing": fails}, indent=1
        )
        + "\n"
    )

    n = len(fits)
    constrained = [f for f in fits if f["n_active"] > 0 or f["n_eq"] > 0]
    print(f"auglag fits: {n} ({len({f['nodeid'] for f in fits})} tests, 11 seeds)")
    print(f"with active rows or equality rows: {len(constrained)}")
    print("polish status, all fits:", Counter(str(f["polish"]) for f in fits))
    print(
        "polish status, constrained fits:",
        Counter(str(f["polish"]) for f in constrained),
    )
    acts = [f["max_active_slack"] for f in fits if f["max_active_slack"] is not None]
    inas = [
        f["min_inactive_slack"]
        for f in constrained
        if f["min_inactive_slack"] is not None
    ]
    mins = [f["min_slack"] for f in fits if f["min_slack"] is not None]
    print(f"largest active slack over all fits: {max(acts):.2e}")
    print(f"smallest inactive slack over constrained fits: {min(inas):.2e}")
    band = [
        f
        for f in fits
        if f["min_inactive_slack"] is not None and f["min_inactive_slack"] < 1e-4
    ]
    print(f"fits with an inactive slack in (1e-6, 1e-4): {len(band)}")
    for f in band[:10]:
        print("   ", f["nodeid"], f["seed"], f"{f['min_inactive_slack']:.2e}")
    print(f"most negative slack (max violation): {min(mins):.2e}")
    kkts = sorted(f["kkt"] for f in constrained if not math.isnan(f["kkt"]))
    q = lambda p: kkts[int(p * (len(kkts) - 1))]  # noqa: E731
    print(
        f"KKT, constrained fits: median {q(0.5):.1e}  p90 {q(0.9):.1e}  "
        f"max {kkts[-1]:.1e}"
    )
    hist = Counter(int(math.floor(math.log10(s))) if s > 0 else -99 for s in inas)
    print("smallest inactive slack, decades:", dict(sorted(hist.items())))


if __name__ == "__main__":
    main()
