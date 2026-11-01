"""Per-start failing tests with the polish applied vs without it (#177).

Usage::

    .venv/bin/python research/active-set-polish/compare_seeds.py \\
        POLISHED.jsonl UNPOLISHED.jsonl

Both logs come from ``research/auglag-start-sensitivity/run_probe.py`` over
the same seeds: ``POLISHED`` with ``-p polish_probe`` and
``MLTPY_POLISH_MODE=apply``, ``UNPOLISHED`` without.  Each failure under the
polish is tagged as one of the expected apply-mode failures (see the README)
or ``other``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

# Failures that apply mode produces by construction (README, "Apply mode").
EXPECTED = {
    "test_optimizer_start_sensitivity.py::test_theta_matches_r_from_every_start": (
        "strict xfail cleared"
    ),
    "TestSolverDispatch::test_optimize_direct": "probe: stale J",
    "TestPolishStep::test_polish_tightens_gradient_on_loose_auglag": (
        "probe: both runs polished"
    ),
    "TestPolishStep::test_polish_skipped_when_disabled": "probe: both runs polished",
    "test_integration_r_reference_new_distributions[exponential": "h(a) >= 0 exact",
}


def tag(nodeid: str) -> str:
    return next((v for k, v in EXPECTED.items() if k in nodeid), "other")


def failures(path: Path) -> tuple[dict[int, dict[str, str]], list[int]]:
    out: dict[int, dict[str, str]] = defaultdict(dict)
    seeds: set[int] = set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("type") != "outcome":
            continue
        seeds.add(r["seed"])
        if r["outcome"] == "failed":
            out[r["seed"]][r["nodeid"]] = r["message"]
    return out, sorted(seeds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("polished", type=Path)
    ap.add_argument("unpolished", type=Path)
    args = ap.parse_args()
    pol, seeds = failures(args.polished)
    unp, _ = failures(args.unpolished)

    print("seed | unpolished fail | polished fail | polished, by tag")
    for k in seeds:
        tags: dict[str, int] = defaultdict(int)
        for nid in pol.get(k, {}):
            tags[tag(nid)] += 1
        print(
            f"{k:>4} | {len(unp.get(k, {})):>15} | {len(pol.get(k, {})):>13} | "
            f"{dict(tags)}"
        )

    def by_test(pred: Callable[[str, int], bool]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        for k in seeds:
            for nid in set(pol.get(k, {})) | set(unp.get(k, {})):
                if pred(nid, k):
                    out[nid].append(k)
        return out

    sections = (
        (
            "unpolished failures the polish clears",
            lambda n, k: n in unp.get(k, {}) and n not in pol.get(k, {}),
        ),
        (
            "failing with and without the polish",
            lambda n, k: n in unp.get(k, {}) and n in pol.get(k, {}),
        ),
        (
            "failing only under the polish, not expected",
            lambda n, k: (
                n in pol.get(k, {}) and n not in unp.get(k, {}) and tag(n) == "other"
            ),
        ),
    )
    for title, pred in sections:
        print(f"\n{title}:")
        for nid, ks in by_test(pred).items():
            print(f"  {nid}\n    starts {ks}  [{tag(nid)}]")


if __name__ == "__main__":
    main()
