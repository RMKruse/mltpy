#!/usr/bin/env python3
"""Convert R-generated CSV reference files to .npy format.

Run after:  Rscript validation/generate_all_references.R
Usage:      python validation/convert_references.py

Reads all .csv files from validation/references/case_*/ and writes
corresponding .npy files alongside them.  metadata.json is left untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# CSV files that contain column headers (need skiprows=1)
_HAS_HEADER = {"X.csv"}

# CSV files that should be loaded as integer then cast to bool
_BOOL_FILES = {"status.csv"}

# CSV files that contain a scalar value (stored as 0-d array)
_SCALAR_FILES = {"loglik.csv"}


def convert_case(case_dir: Path) -> list[Path]:
    """Convert all .csv files in *case_dir* to .npy.

    Parameters
    ----------
    case_dir:
        Directory containing .csv files and metadata.json for one case.

    Returns
    -------
    list[Path]
        Paths of all generated .npy files.
    """
    created: list[Path] = []

    for csv_path in sorted(case_dir.glob("*.csv")):
        npy_path = csv_path.with_suffix(".npy")
        name = csv_path.name

        if name in _BOOL_FILES:
            data = np.loadtxt(csv_path, dtype=int).astype(bool)
        elif name in _HAS_HEADER:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        elif name in _SCALAR_FILES:
            data = np.float64(np.loadtxt(csv_path))
        else:
            data = np.loadtxt(csv_path)

        np.save(npy_path, data)
        created.append(npy_path)

    return created


def main() -> None:
    """Find all case_* directories and convert their CSVs to .npy."""
    ref_dir = Path(__file__).parent / "references"

    if not ref_dir.is_dir():
        print(f"ERROR: {ref_dir} does not exist. Run the R script first.")
        sys.exit(1)

    case_dirs = sorted(ref_dir.glob("case_*"))
    if not case_dirs:
        print(f"ERROR: No case_* directories found in {ref_dir}.")
        sys.exit(1)

    total_files: list[Path] = []

    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue
        created = convert_case(case_dir)
        n = len(created)
        print(f"  {case_dir.name}: {n} .npy files")
        total_files.extend(created)

    print(f"\nTotal: {len(total_files)} .npy files from {len(case_dirs)} cases")
    print("\nGenerated files:")
    for p in total_files:
        print(f"  {p.relative_to(ref_dir.parent)}")


if __name__ == "__main__":
    main()
