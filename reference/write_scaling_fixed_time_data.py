"""Write the issue-#162 fixed-time censoring design for the R reference.

Run with: python reference/write_scaling_fixed_time_data.py

Issue #162's repro draws its design with NumPy's PCG64 generator.  R cannot
replay that stream, so the data are drawn here and read back by the
``scaling_fixed_time`` block of ``reference/generate_reference.R``, which
writes R ``mlt``'s fits.

Writes (one number per line, 17 significant digits so R parses the exact
doubles):

    reference/scaling_fixed_time_y.txt      — min(y, c), c the 80 % quantile
    reference/scaling_fixed_time_event.txt  — 1 if y <= c (observed), else 0
    reference/scaling_fixed_time_x_d.txt    — shift covariate
    reference/scaling_fixed_time_x_s.txt    — scaling covariate
    reference/scaling_fixed_time_flipped_<dist>_mltpy_theta.txt
        — mltpy's θ = [θ_b | β | γ] on the issue's literal repro, which passed
          ``event`` as the *censored* flag (trust-constr, ``random_state=0``),
          for <dist> in {normal, logistic}.  R evaluates its logLik there.

``tests/test_scaling_censoring.py`` redraws the data with :func:`draw` and
checks it against these files before comparing to R.  The θ files are an
input to R, not an expected value: the test compares mltpy's ℓ at that θ
with R's logLik at the same θ.
"""

from __future__ import annotations

import pathlib

import numpy as np

N = 150
OUT_DIR = pathlib.Path(__file__).parent


def draw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw the issue-#162 design.

    Returns
    -------
    y : np.ndarray
        Response of shape ``(N,)``, censored at the 80 % quantile ``c`` of the
        uncensored draw: ``min(y, c)``.
    event : np.ndarray
        Boolean of shape ``(N,)``; ``True`` where ``y <= c`` (observed).  This
        is R's ``Surv`` event flag, the negation of mltpy's ``censored``.
    x_d : np.ndarray
        Shift covariate of shape ``(N, 1)``, ``U(0, 1)``.
    x_s : np.ndarray
        Scaling covariate of shape ``(N, 1)``, ``N(0, 1)``.
    """
    rng = np.random.default_rng(0)
    x_s = rng.normal(size=(N, 1))
    y = rng.exponential(size=N) / (2 * np.exp(0.5 * 0.6 * x_s[:, 0]))
    x_d = np.random.default_rng(50).uniform(size=(N, 1))
    c = float(np.quantile(y, 0.8))
    event = y <= c
    return np.minimum(y, c), event, x_d, x_s


def fit_flipped_repro(dist: str) -> np.ndarray:
    """mltpy's θ on the issue's literal repro (``event`` passed as censored).

    Parameters
    ----------
    dist : str
        ``"normal"`` or ``"logistic"``.

    Returns
    -------
    np.ndarray
        ``theta_`` of the fit, ``[θ_b (5) | β | γ]``.
    """
    # Imported here so that draw() needs only NumPy.
    import warnings

    from mltpy import MLT
    from mltpy.optimizer import OptimizerConfig
    from mltpy.variables import CensoredData, CensoringType

    y, event, x_d, x_s = draw()
    model = MLT(
        order=4,
        support=(0.0, 1.01 * float(y.max())),
        censoring=CensoringType.RIGHT,
        base_distribution=dist,  # type: ignore[arg-type]
        scaling=x_s,
        optimizer_config=OptimizerConfig(solver="trust-constr", random_state=0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(CensoredData.right_censored(y, event), X=x_d)
    return np.asarray(model.theta_)


def main() -> None:
    """Write the design files and mltpy's flipped-repro θ files."""
    y, event, x_d, x_s = draw()
    np.savetxt(OUT_DIR / "scaling_fixed_time_y.txt", y, fmt="%.17g")
    np.savetxt(OUT_DIR / "scaling_fixed_time_event.txt", event.astype(int), fmt="%d")
    np.savetxt(OUT_DIR / "scaling_fixed_time_x_d.txt", x_d[:, 0], fmt="%.17g")
    np.savetxt(OUT_DIR / "scaling_fixed_time_x_s.txt", x_s[:, 0], fmt="%.17g")
    for dist in ("normal", "logistic"):
        np.savetxt(
            OUT_DIR / f"scaling_fixed_time_flipped_{dist}_mltpy_theta.txt",
            fit_flipped_repro(dist),
            fmt="%.17g",
        )


if __name__ == "__main__":
    main()
