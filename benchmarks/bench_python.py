"""Runtime benchmark for ``pymlt.MLT.fit`` across a grid of (n, order, censoring).

Generates the input datasets in ``benchmarks/data/`` so that ``bench_r.R`` can
consume the byte-identical CSVs (Alternative B in the benchmark plan).

Outputs ``benchmarks/results/python_results.csv`` with columns::

    n, order, censoring, rep, time_s, converged, n_iter

Run with::

    python benchmarks/bench_python.py

or via the ``benchmark`` Makefile target.
"""

from __future__ import annotations

import csv
import gc
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

import pymlt
from pymlt import CensoredData, CensoringType, ConvergenceWarning

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_LIST: tuple[int, ...] = (100, 500, 1000, 5000)
ORDER_LIST: tuple[int, ...] = (4, 6, 8)
CENSORING_LIST: tuple[Literal["NONE", "RIGHT"], ...] = ("NONE", "RIGHT")
N_REPS: int = 10

# Master seed; per-dataset seeds are derived deterministically below so that
# repeated runs produce byte-identical CSVs.
MASTER_SEED: int = 20260416

# Target fraction of right-censored observations.
CENSORING_RATE: float = 0.30

# Data support — chosen wide enough to contain all generated observations
# with comfortable margin on both ends.
SUPPORT: tuple[float, float] = (0.0, 10.0)

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "python_results.csv"


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _dataset_seed(n: int, censoring: str) -> int:
    """Deterministic per-dataset seed derived from MASTER_SEED."""
    cens_offset = {"NONE": 0, "RIGHT": 1}[censoring]
    return MASTER_SEED + n * 10 + cens_offset


def generate_dataset(
    n: int, censoring: str
) -> tuple[NDArray[np.float64], NDArray[np.int_] | None]:
    """Generate a single (y, status) dataset.

    For ``censoring="NONE"`` the second element is ``None``.
    For ``censoring="RIGHT"`` ``status`` is 1 (event observed) or 0 (censored).
    """
    rng = np.random.default_rng(_dataset_seed(n, censoring))
    # Lognormal-ish positive values, well inside SUPPORT
    raw = rng.normal(loc=0.0, scale=1.0, size=n)
    y = np.clip(np.exp(0.5 * raw + 0.7), 0.05, 9.95)

    if censoring == "NONE":
        return y, None

    # RIGHT: independent Bernoulli censoring indicator
    censored = rng.uniform(size=n) < CENSORING_RATE
    status = (~censored).astype(np.int_)  # 1 = event observed, 0 = censored
    return y, status


def write_data_csvs() -> None:
    """Materialise every (n, censoring) dataset as a CSV in ``data/``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for n in N_LIST:
        for censoring in CENSORING_LIST:
            y, status = generate_dataset(n, censoring)
            path = DATA_DIR / f"n{n}_{censoring}.csv"
            with path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                if status is None:
                    writer.writerow(["y"])
                    for v in y:
                        writer.writerow([f"{v:.17g}"])
                else:
                    writer.writerow(["y", "status"])
                    for v, s in zip(y, status, strict=True):
                        writer.writerow([f"{v:.17g}", int(s)])


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


def _build_model(order: int, censoring: str) -> pymlt.MLT:
    """Construct a fresh MLT instance for the given cell."""
    cens = CensoringType.NONE if censoring == "NONE" else CensoringType.RIGHT
    return pymlt.MLT(order=order, support=SUPPORT, censoring=cens)


def _make_fit_input(
    y: NDArray[np.float64], status: NDArray[np.int_] | None
) -> NDArray[np.float64] | CensoredData:
    """Wrap the response for ``fit()`` according to the censoring scheme."""
    if status is None:
        return y
    censored_mask = status == 0
    return CensoredData.right_censored(y, censored_mask)


def _time_one_fit(model: pymlt.MLT, fit_input: object) -> tuple[float, bool, int]:
    """Time a single ``fit()`` call. Returns (seconds, converged, n_iter)."""
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            # ConvergenceWarning: recorded separately via result_.converged.
            # RuntimeWarning: harmless boundary probes from the optimiser
            #   (e.g. log of a non-monotone proposal during line search).
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(fit_input)  # type: ignore[arg-type]
        elapsed = time.perf_counter() - t0
    finally:
        gc.enable()
    if model.result_ is None:
        raise RuntimeError("fit() returned without populating result_")
    return elapsed, bool(model.result_.converged), int(model.result_.n_iter)


def _iter_cells() -> Iterable[tuple[int, int, str]]:
    for n in N_LIST:
        for order in ORDER_LIST:
            for censoring in CENSORING_LIST:
                yield n, order, censoring


def run_benchmark() -> list[dict[str, object]]:
    """Execute every (n, order, censoring) cell ``N_REPS`` times.

    Returns a list of result rows ready to be written as CSV.
    """
    rows: list[dict[str, object]] = []
    total_cells = len(N_LIST) * len(ORDER_LIST) * len(CENSORING_LIST)
    for cell_idx, (n, order, censoring) in enumerate(_iter_cells(), start=1):
        y, status = generate_dataset(n, censoring)
        fit_input = _make_fit_input(y, status)
        model = _build_model(order, censoring)
        for rep in range(N_REPS):
            elapsed, converged, n_iter = _time_one_fit(model, fit_input)
            rows.append(
                {
                    "n": n,
                    "order": order,
                    "censoring": censoring,
                    "rep": rep,
                    "time_s": f"{elapsed:.9f}",
                    "converged": int(converged),
                    "n_iter": n_iter,
                }
            )
        cell_times = [
            float(cast(str, r["time_s"])) for r in rows[-N_REPS:] if r["converged"] == 1
        ]
        median = float(np.median(cell_times)) if cell_times else float("nan")
        print(
            f"[{cell_idx:>2}/{total_cells}] "
            f"n={n:>4} order={order} cens={censoring:<5} "
            f"median={median * 1000:8.2f} ms  "
            f"({len(cell_times)}/{N_REPS} converged)"
        )
        if cell_times and median < 0.010:
            print(
                f"    note: median below 10 ms — timing noise may dominate "
                f"for n={n}, order={order}, cens={censoring}."
            )
    return rows


def write_results_csv(rows: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["n", "order", "censoring", "rep", "time_s", "converged", "n_iter"]
    with RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"pymlt benchmark — version {pymlt.__version__}")
    print(f"  data dir:    {DATA_DIR}")
    print(f"  results csv: {RESULTS_CSV}")
    print(
        f"  grid: n={list(N_LIST)} × order={list(ORDER_LIST)} "
        f"× censoring={list(CENSORING_LIST)} × reps={N_REPS}"
    )
    print()
    print("Writing input datasets …")
    write_data_csvs()
    print(f"  wrote {len(N_LIST) * len(CENSORING_LIST)} CSV files to {DATA_DIR}")
    print()
    print("Running fit() benchmarks …")
    rows = run_benchmark()
    write_results_csv(rows)
    print()
    print(f"Wrote {len(rows)} timing rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
