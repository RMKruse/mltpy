"""Summarise a polish-probe JSONL log (#177).

Usage::

    .venv/bin/python research/active-set-polish/summarise.py LOG.jsonl \\
        [--baseline B.jsonl]

Prints acceptance, rejection reasons, KKT and violation before/after, the
size of the change, the cost, and the failing tests.  With ``--baseline``
(a measure-mode log of the same run), lists tests that fail only in this log.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line]
    fits = [r for r in recs if r["type"] == "fit"]
    fails = [r for r in recs if r["type"] == "outcome" and r["outcome"] == "failed"]
    return fits, fails


def q(vals: list[float], p: float) -> float:
    v = sorted(x for x in vals if x is not None and not math.isnan(x))
    return v[int(p * (len(v) - 1))] if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--baseline", type=Path)
    args = ap.parse_args()
    fits, fails = load(args.log)
    ok = [f for f in fits if "error" not in f]
    err = [f for f in fits if "error" in f]
    acc = [f for f in ok if f["accepted"]]
    print(
        f"auglag fits: {len(fits)}  polished without error: {len(ok)}  "
        f"errors: {len(err)}"
    )
    for e, c in Counter(f["error"][:90] for f in err).most_common(5):
        print(f"   error x{c}: {e}")
    print(f"accepted: {len(acc)} ({100 * len(acc) / max(1, len(ok)):.1f}%)")
    reasons = Counter(
        f["reason"].split(" (")[0].split(" by ")[0] for f in ok if not f["accepted"]
    )
    for r, c in reasons.most_common():
        print(f"   rejected x{c}: {r}")
    by_path = Counter((f["path"], f["accepted"]) for f in ok)
    print("by path (path, accepted):", dict(by_path))
    for label, sel in (("accepted", acc), ("all", ok)):
        print(
            f"{label}: KKT before median {q([f['kkt_before'] for f in sel], 0.5):.1e} "
            f"p90 {q([f['kkt_before'] for f in sel], 0.9):.1e} | after median "
            f"{q([f['kkt_after'] for f in sel], 0.5):.1e} "
            f"p90 {q([f['kkt_after'] for f in sel], 0.9):.1e}"
        )
    print(
        "accepted: violation before max "
        f"{max((f['viol_before'] for f in acc), default=0):.1e}"
        f", after max {max((f['viol_after'] for f in acc), default=0):.1e}"
    )
    print(
        f"accepted: Δθ rel median {q([f['dtheta_rel'] for f in acc], 0.5):.1e} "
        f"p99 {q([f['dtheta_rel'] for f in acc], 0.99):.1e} max "
        f"{max((f['dtheta_rel'] for f in acc), default=0):.1e}"
    )
    print(
        f"accepted: ΔNLL min {min((f['dnll'] for f in acc), default=0):.1e} "
        f"max {max((f['dnll'] for f in acc), default=0):.1e}"
    )
    big = sorted(acc, key=lambda f: -f["dtheta_rel"])[:8]
    for f in big:
        print(
            f"   largest Δθ {f['dtheta_rel']:.1e} ΔNLL {f['dnll']:+.1e} {f['nodeid']}"
        )
    secs = [f["seconds"] for f in fits]
    print(
        f"cost: total {sum(secs):.1f} s over {len(secs)} fits; median "
        f"{1e3 * q(secs, 0.5):.2f} ms, p99 {1e3 * q(secs, 0.99):.1f} ms, "
        f"max {1e3 * max(secs):.0f} ms"
    )
    print(
        "iterations median/max:",
        q([f["n_iter"] for f in ok], 0.5),
        max((f["n_iter"] for f in ok), default=0),
        "| rows added/dropped (fits with any):",
        sum(f["added"] > 0 for f in ok),
        sum(f["dropped"] > 0 for f in ok),
    )
    print(f"failing tests: {len({f['nodeid'] for f in fails})}")
    if args.baseline:
        _, base_fails = load(args.baseline)
        base = {f["nodeid"] for f in base_fails}
        new = sorted({f["nodeid"] for f in fails} - base)
        fixed = sorted(base - {f["nodeid"] for f in fails})
        print(f"fail only here: {len(new)}")
        for nid in new:
            msg = next(f["message"] for f in fails if f["nodeid"] == nid)
            print(f"   {nid}\n      {msg[:240]}")
        print(f"fail only in baseline: {len(fixed)}")
        for nid in fixed:
            print(f"   {nid}")


if __name__ == "__main__":
    main()
