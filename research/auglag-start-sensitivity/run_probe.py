"""Run R-parity test files under perturbed auglag starts and summarise (#178).

Usage, from the repository root::

    PYTHONPATH=src .venv/bin/python research/auglag-start-sensitivity/run_probe.py \\
        --seeds 0-10 --out research/auglag-start-sensitivity/results.jsonl \\
        tests/test_exponential_scaling_shift.py ...

Each seed runs pytest once over the given files with the ``start_probe``
plugin (see ``start_probe.py``) and appends its JSONL records to ``--out``.
``--summary`` prints the per-test table from an existing ``--out`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ACTIVE_TOL = 1e-6  # #177 decision 2: active iff normalised slack <= 1e-6


def _seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def run(files: list[str], seeds: list[int], out: Path, extra: list[str]) -> None:
    out.write_text("")
    for k in seeds:
        log = out.with_suffix(f".seed{k}.jsonl")
        env = dict(os.environ)
        env["MLTPY_START_SEED"] = str(k)
        env["MLTPY_START_LOG"] = str(log)
        env["PYTHONPATH"] = os.pathsep.join([str(HERE), env.get("PYTHONPATH", "src")])
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *files,
            "-q",
            "-p",
            "start_probe",
            "-p",
            "no:cacheprovider",
            "--tb=short",
            *extra,
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        tail = proc.stdout.strip().splitlines()[-1:] if proc.stdout else []
        print(f"seed {k}: {' '.join(tail)}", flush=True)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(log.read_text(encoding="utf-8"))
        log.unlink()


def _gap(slacks: list[float]) -> tuple[int, float, float]:
    act = [s for s in slacks if s <= ACTIVE_TOL]
    ina = [s for s in slacks if s > ACTIVE_TOL]
    return (
        len(act),
        max(act) if act else float("nan"),
        min(ina) if ina else float("nan"),
    )


def summary(out: Path) -> dict[str, Any]:
    recs = [json.loads(line) for line in out.read_text().splitlines() if line]
    outcomes: dict[str, dict[int, str]] = defaultdict(dict)
    messages: dict[str, dict[int, str]] = defaultdict(dict)
    for r in recs:
        if r["type"] != "outcome":
            continue
        prev = outcomes[r["nodeid"]].get(r["seed"])
        if prev != "failed":
            outcomes[r["nodeid"]][r["seed"]] = r["outcome"]
        if r["outcome"] == "failed":
            messages[r["nodeid"]][r["seed"]] = r["message"]
    fits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if r["type"] == "fit":
            fits[str(r["nodeid"])].append(r)
    flaky = {
        nid: by_seed
        for nid, by_seed in outcomes.items()
        if by_seed.get(0) == "passed"
        and any(v == "failed" for s, v in by_seed.items() if s > 0)
    }
    broken = {nid for nid, by_seed in outcomes.items() if by_seed.get(0) == "failed"}
    return {
        "outcomes": outcomes,
        "messages": messages,
        "fits": fits,
        "start_dependent": flaky,
        "fail_at_default": broken,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--seeds", default="0-10")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", action="store_true")
    args, extra = ap.parse_known_args()
    if not args.summary:
        run(args.files, _seeds(args.seeds), args.out, extra)
    s = summary(args.out)
    print(f"\nfail at the default start: {len(s['fail_at_default'])}")
    for nid in sorted(s["fail_at_default"]):
        print("  ", nid)
    print(
        f"start-dependent (pass at k=0, fail at some k>0): {len(s['start_dependent'])}"
    )
    for nid, by_seed in sorted(s["start_dependent"].items()):
        failed = sorted(k for k, v in by_seed.items() if v == "failed")
        fits = s["fits"].get(nid, [])
        kkt = max((f["kkt"] for f in fits), default=float("nan"))
        print(f"   {nid}\n      failing seeds {failed}  max KKT {kkt:.1e}")
        first = failed[0]
        print(f"      seed {first}: {s['messages'][nid].get(first, '')[:300]}")


if __name__ == "__main__":
    main()
