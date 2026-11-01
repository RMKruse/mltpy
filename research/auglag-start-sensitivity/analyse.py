"""Per-test fit diagnostics from a start-probe JSONL log (#178).

Usage::

    .venv/bin/python research/auglag-start-sensitivity/analyse.py results.jsonl \\
        [--module tests/test_x.py ...]

For every test that passes at the default start and fails from a perturbed
one, and for every fit logged in the given modules, prints per seed: KKT,
converged, number of equality rows, number of active inequality rows
(normalised slack <= 1e-6), the slack gap (largest active slack / smallest
inactive slack) and the polish status.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ACTIVE_TOL = 1e-6


def _fmt(x: float) -> str:
    return "   -   " if x is None or math.isnan(x) else f"{x:.1e}"


def describe(fit: dict[str, Any]) -> str:
    sl = fit["slacks"]
    act = [s for s in sl if s <= ACTIVE_TOL]
    ina = [s for s in sl if s > ACTIVE_TOL]
    return (
        f"kkt {_fmt(fit['kkt'])} conv {str(fit['converged'])[0]} eq {fit['n_eq']:2d} "
        f"active {len(act):3d}/{len(sl):3d} "
        f"act≤{_fmt(max(act) if act else float('nan'))} "
        f"inact≥{_fmt(min(ina) if ina else float('nan'))} "
        f"min {_fmt(min(sl) if sl else float('nan'))} polish {fit['polish']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--module", action="append", default=[])
    ap.add_argument("--nodeid", action="append", default=[])
    args = ap.parse_args()
    recs = [json.loads(line) for line in args.log.read_text().splitlines() if line]
    fits: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in recs:
        if r["type"] == "fit":
            fits[str(r["nodeid"])][r["seed"]].append(r)
    wanted = [
        nid
        for nid in fits
        if any(nid.startswith(m) for m in args.module) or nid in args.nodeid
    ]
    for nid in wanted:
        print(nid)
        for seed in sorted(fits[nid]):
            for i, fit in enumerate(fits[nid][seed]):
                print(f"   k={seed:2d} fit {i}: {describe(fit)}")


if __name__ == "__main__":
    main()
