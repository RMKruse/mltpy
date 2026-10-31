"""Write the issue-#142 interaction-stall training data for the R reference.

Run with: python reference/write_interaction_stall_data.py

Issue #142 reproduces a false ``converged=True`` auglag stall on data drawn
with NumPy's PCG64 generator.  R cannot replay that stream, so the data are
drawn here and read back by the ``interaction_stall`` block of
``reference/generate_reference.R``, which writes the R ``mlt`` fit.

Writes, for n ∈ {1000, 2000} (one number per line, 17 significant digits so
R parses the exact doubles):

    reference/interaction_stall_n<n>_y_train.txt
    reference/interaction_stall_n<n>_x_train.txt

``tests/test_interaction_stall.py`` redraws the data with :func:`draw` and
checks it against these files before comparing to R.
"""

from __future__ import annotations

import pathlib

import numpy as np

SEED = 20261031
SIZES = (1000, 2000)
OUT_DIR = pathlib.Path(__file__).parent


def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw the issue-#142 sample of size ``n``.

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    y : np.ndarray
        Response of shape ``(n,)``, ``y ~ N(0.4 x, 1)`` clipped to
        ``[-3.9, 3.9]`` (inside the Bernstein support ``(-4, 4)``).
    X : np.ndarray
        Covariate of shape ``(n, 1)``, ``x ~ U(-1, 1)``.
    """
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-1, 1, (n, 1))
    y = np.clip(rng.normal(0.4 * X[:, 0], 1), -3.9, 3.9)
    return y, X


def main() -> None:
    """Write ``interaction_stall_n<n>_{y,x}_train.txt`` for every n in SIZES."""
    for n in SIZES:
        y, X = draw(n)
        np.savetxt(OUT_DIR / f"interaction_stall_n{n}_y_train.txt", y, fmt="%.17g")
        np.savetxt(
            OUT_DIR / f"interaction_stall_n{n}_x_train.txt", X[:, 0], fmt="%.17g"
        )


if __name__ == "__main__":
    main()
