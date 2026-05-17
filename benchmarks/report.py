"""Generate ``benchmarks/results/benchmark_report.md`` from the raw CSVs.

Reads ``python_results.csv`` and ``r_results.csv`` (both produced by
``bench_python.py`` / ``bench_r.R``), aggregates per-cell median + IQR,
computes the R/Python speedup ratio, and writes a markdown report with
environment metadata and a short interpretation.

Run with::

    python benchmarks/report.py

or via the ``benchmark`` Makefile target.
"""

from __future__ import annotations

import csv
import datetime as _dt
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

import pymlt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PYTHON_CSV = RESULTS_DIR / "python_results.csv"
R_CSV = RESULTS_DIR / "r_results.csv"
REPORT_MD = RESULTS_DIR / "benchmark_report.md"

CensoringStr = Literal["NONE", "RIGHT"]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellKey:
    n: int
    order: int
    censoring: str


@dataclass
class CellStats:
    median: float
    iqr: float
    n_reps: int
    n_converged: int


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a benchmark CSV. Raises FileNotFoundError with a helpful message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required benchmark CSV not found: {path}\n"
            f"  Run `python benchmarks/bench_python.py` and "
            f"`Rscript benchmarks/bench_r.R` first (or `make benchmark`)."
        )
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def aggregate(rows: list[dict[str, str]]) -> dict[CellKey, CellStats]:
    """Group rows by ``(n, order, censoring)``, compute median + IQR.

    Only converged reps contribute to the timing aggregation; the count of
    converged reps is preserved on the result for diagnostic display.
    """
    times: defaultdict[CellKey, list[float]] = defaultdict(list)
    totals: defaultdict[CellKey, int] = defaultdict(int)
    for r in rows:
        key = CellKey(n=int(r["n"]), order=int(r["order"]), censoring=r["censoring"])
        totals[key] += 1
        if int(r["converged"]) == 1:
            times[key].append(float(r["time_s"]))

    out: dict[CellKey, CellStats] = {}
    for key, total_reps in totals.items():
        cell_times = np.asarray(times[key], dtype=float)
        if cell_times.size == 0:
            out[key] = CellStats(
                median=float("nan"),
                iqr=float("nan"),
                n_reps=total_reps,
                n_converged=0,
            )
            continue
        q1, med, q3 = np.quantile(cell_times, [0.25, 0.5, 0.75])
        out[key] = CellStats(
            median=float(med),
            iqr=float(q3 - q1),
            n_reps=total_reps,
            n_converged=int(cell_times.size),
        )
    return out


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------


def _safe_run(cmd: list[str]) -> str | None:
    """Run a command, return stdout stripped or ``None`` on failure."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def _cpu_brand() -> str:
    """Best-effort CPU brand string. Falls back to ``platform.processor()``."""
    if sys.platform == "darwin":
        brand = _safe_run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _r_versions() -> tuple[str, str]:
    """Return (R version, mlt version), each ``"unknown"`` on failure."""
    r_ver = _safe_run(
        [
            "Rscript",
            "-e",
            'cat(paste(R.version$major, R.version$minor, sep="."))',
        ]
    )
    mlt_ver = _safe_run(["Rscript", "-e", 'cat(as.character(packageVersion("mlt")))'])
    return r_ver or "unknown", mlt_ver or "unknown"


def _git_sha() -> str:
    sha = _safe_run(["git", "rev-parse", "--short", "HEAD"])
    return sha or "unknown"


def collect_environment() -> dict[str, str]:
    """Snapshot of versions/platform for reproducibility."""
    import scipy

    r_ver, mlt_ver = _r_versions()
    return {
        "Timestamp (UTC)": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "Git SHA": _git_sha(),
        "OS": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "CPU": _cpu_brand(),
        "Python": platform.python_version(),
        "NumPy": np.__version__,
        "SciPy": scipy.__version__,
        "pymlt": pymlt.__version__,
        "R": r_ver,
        "R mlt": mlt_ver,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def _fmt_seconds(x: float) -> str:
    """Format a duration in seconds with 4 significant digits."""
    if not np.isfinite(x):
        return "—"
    if x >= 1.0:
        return f"{x:.3f} s"
    if x >= 0.001:
        return f"{x * 1000:.2f} ms"
    return f"{x * 1e6:.1f} µs"


def _fmt_speedup(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "—"
    return f"{ratio:.2f}×"


def _format_env_block(env: dict[str, str]) -> str:
    lines = ["| Field | Value |", "|---|---|"]
    for k, v in env.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _format_table(
    censoring: CensoringStr,
    py: dict[CellKey, CellStats],
    r: dict[CellKey, CellStats],
    n_list: list[int],
    order_list: list[int],
) -> str:
    header = (
        "| n | order | Python (median) | Python IQR | "
        "R (median) | R IQR | Speedup (R/Py) |"
    )
    sep = "|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for n in n_list:
        for order in order_list:
            key = CellKey(n=n, order=order, censoring=censoring)
            ps = py.get(key)
            rs = r.get(key)
            if ps is None or rs is None:
                lines.append(
                    f"| {n} | {order} | — | — | — | — | "
                    f"missing in {'Python' if ps is None else 'R'} |"
                )
                continue
            ratio = (
                rs.median / ps.median
                if (np.isfinite(rs.median) and np.isfinite(ps.median) and ps.median > 0)
                else float("nan")
            )
            lines.append(
                f"| {n} | {order} | "
                f"{_fmt_seconds(ps.median)} | {_fmt_seconds(ps.iqr)} | "
                f"{_fmt_seconds(rs.median)} | {_fmt_seconds(rs.iqr)} | "
                f"{_fmt_speedup(ratio)} |"
            )
    return "\n".join(lines)


def _format_convergence_notes(
    py: dict[CellKey, CellStats],
    r: dict[CellKey, CellStats],
) -> list[str]:
    """One bullet per cell where any rep failed to converge."""
    notes: list[str] = []
    all_keys = sorted(set(py) | set(r), key=lambda k: (k.n, k.order, k.censoring))
    for key in all_keys:
        ps = py.get(key)
        rs = r.get(key)
        py_fail = ps is not None and ps.n_converged < ps.n_reps
        r_fail = rs is not None and rs.n_converged < rs.n_reps
        if py_fail or r_fail:
            parts = []
            if ps is not None and py_fail:
                parts.append(f"Python {ps.n_converged}/{ps.n_reps}")
            if rs is not None and r_fail:
                parts.append(f"R {rs.n_converged}/{rs.n_reps}")
            notes.append(
                f"- n={key.n}, order={key.order}, "
                f"censoring={key.censoring}: " + ", ".join(parts)
            )
    return notes


def _interpret(
    py: dict[CellKey, CellStats],
    r: dict[CellKey, CellStats],
) -> str:
    """Generate a short paragraph summarising the speedup distribution."""
    ratios: dict[CellKey, float] = {}
    for key in sorted(set(py) & set(r), key=lambda k: (k.n, k.order, k.censoring)):
        ps, rs = py[key], r[key]
        if (
            np.isfinite(ps.median)
            and np.isfinite(rs.median)
            and ps.median > 0
            and rs.median > 0
        ):
            ratios[key] = rs.median / ps.median

    if not ratios:
        return (
            "No directly comparable cells available — both backends must "
            "have at least one converged rep per cell to compute a speedup."
        )

    values = np.asarray(list(ratios.values()), dtype=float)
    geomean = float(np.exp(np.mean(np.log(values))))
    fastest_key = max(ratios, key=lambda k: ratios[k])
    slowest_key = min(ratios, key=lambda k: ratios[k])
    n_pymlt_faster = int(np.sum(values > 1.0))

    # Per-censoring geomeans, computed only over cells with both backends.
    by_censoring: defaultdict[str, list[float]] = defaultdict(list)
    for key, ratio in ratios.items():
        by_censoring[key.censoring].append(ratio)
    censoring_summaries = ", ".join(
        f"{cens} {float(np.exp(np.mean(np.log(np.asarray(rs))))):.2f}×"
        for cens, rs in sorted(by_censoring.items())
    )

    parts = [
        f"Across {len(ratios)} comparable cells, pymlt is on geometric "
        f"mean **{geomean:.2f}× the speed of R mlt**, and is the faster "
        f"backend in {n_pymlt_faster} of {len(ratios)} cells. ",
        f"By censoring scheme: {censoring_summaries}. ",
        f"The largest speedup is at "
        f"n={fastest_key.n}, order={fastest_key.order}, "
        f"censoring={fastest_key.censoring} ({_fmt_speedup(ratios[fastest_key])}); ",
        f"the smallest is at "
        f"n={slowest_key.n}, order={slowest_key.order}, "
        f"censoring={slowest_key.censoring} ({_fmt_speedup(ratios[slowest_key])}). ",
        "Both backends now use the same PHR augmented Lagrangian solver — "
        "pymlt's pure-Python ``_auglag.py`` and R `mlt`'s `alabama::auglag` "
        "share the outer multiplier-update loop and the inner BFGS sub-problem, "
        "so iteration counts at the same precision land in the same ballpark "
        "and the divergent-basin caveat that applied under SLSQP no longer "
        "holds.  The remaining per-fit gap reflects per-iteration cost: every "
        "outer step in pymlt re-enters ``scipy.optimize.minimize``, whereas "
        "`alabama` keeps the inner solver state in Fortran across updates.  "
        "Absolute per-fit times of a few milliseconds at small `n` mean "
        "timing noise can shift individual cells by 10–30%, so treat "
        "small-`n` ratios as indicative rather than precise.",
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    py: dict[CellKey, CellStats],
    r: dict[CellKey, CellStats],
    env: dict[str, str],
) -> str:
    keys = set(py) | set(r)
    n_list = sorted({k.n for k in keys})
    order_list = sorted({k.order for k in keys})
    censorings: list[CensoringStr] = sorted(
        {k.censoring for k in keys}  # type: ignore[misc]
    )

    parts = [
        "# pymlt vs. R mlt — Runtime Benchmark",
        "",
        "Generated by `benchmarks/report.py`. Re-run with `make benchmark` to refresh.",
        "",
        "## Environment",
        "",
        _format_env_block(env),
        "",
        "## Methodology",
        "",
        "- For each `(n, order, censoring)` cell the same Python-generated "
        "input dataset is fitted by both backends.",
        "- Each cell is timed across multiple repetitions; medians and IQRs "
        "are computed over the converged reps only.",
        "- pymlt: `pymlt.MLT(order, support).fit(...)`; "
        "R: `mlt::mlt(ctm(Bernstein_basis(...)), data=...)`.",
        "- Right-censored cells use `CensoredData.right_censored` (Python) "
        "and `Surv(y, status)` (R), with ~30% censoring drawn from a "
        "deterministic seed.",
        "- `Speedup` is `R median / Python median`; values > 1 mean pymlt is faster.",
        "",
    ]

    for cens in censorings:
        parts.extend(
            [
                f"## Censoring: {cens}",
                "",
                _format_table(cens, py, r, n_list, order_list),
                "",
            ]
        )

    parts.extend(["## Interpretation", "", _interpret(py, r), ""])

    notes = _format_convergence_notes(py, r)
    if notes:
        parts.extend(["## Convergence issues", "", *notes, ""])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Reading benchmark CSVs …")
    py_rows = _read_csv_rows(PYTHON_CSV)
    r_rows = _read_csv_rows(R_CSV)
    print(f"  python: {len(py_rows)} rows from {PYTHON_CSV.name}")
    print(f"  r:      {len(r_rows)} rows from {R_CSV.name}")

    py_stats = aggregate(py_rows)
    r_stats = aggregate(r_rows)

    py_keys = set(py_stats)
    r_keys = set(r_stats)
    only_py = py_keys - r_keys
    only_r = r_keys - py_keys
    if only_py:
        print(f"  warning: {len(only_py)} cells present only in Python results")
    if only_r:
        print(f"  warning: {len(only_r)} cells present only in R results")

    print("Collecting environment metadata …")
    env = collect_environment()

    print("Building report …")
    report = build_report(py_stats, r_stats, env)
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
