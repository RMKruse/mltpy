"""Scaled Coxph R fixture: auglag vs prototype polish against R (#177).

``tests/test_tram_coxph_scaling.py::test_coxph_theta_beta_gamma_match_R``
holds θ to ``rtol=1e-4, atol=1e-5``: auglag reports ``converged=True`` at
KKT 2.7e-2 on this fit.  This script measures both points at CLAUDE.md's
``rtol=1e-6, atol=1e-10``.

Usage::

    PYTHONPATH=src:research/active-set-polish .venv/bin/python \\
        research/active-set-polish/coxph_check.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from harness import build_problem, capture
from polish import polish

from mltpy.tram import Coxph
from mltpy.variables import CensoredData

REF = Path(__file__).resolve().parents[2] / "reference"


def load(name: str) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(REF / name))


def main() -> None:
    y = load("scaling_coxph_y.txt")
    event = load("scaling_coxph_event.txt").astype(bool)
    x_d = load("scaling_coxph_x_d.txt").reshape(-1, 1)
    x_s = load("scaling_coxph_x_s.txt").reshape(-1, 1)
    sup = load("scaling_coxph_support.txt")
    theta_r = load("scaling_coxph_theta.txt")
    ll_r = float(load("scaling_coxph_loglik.txt")[0])
    p = theta_r.size - 2
    model = Coxph(support=(float(sup[0]), float(sup[1])), order=p - 1, scaling=x_s)
    with capture() as calls, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(CensoredData.right_censored(y, ~event), X=x_d)
    call = calls[-1]
    theta0 = np.asarray(call.result.theta, dtype=np.float64)
    assert np.array_equal(theta0, np.asarray(model.theta_)), "layout differs"
    res = polish(build_problem(call.kwargs), theta0)
    bound = 1e-10 + 1e-6 * np.abs(theta_r)
    for label, th, ll in (
        ("auglag", theta0, call.result.log_likelihood),
        # The candidate, not res.theta: nll_after belongs to the candidate,
        # and res.theta falls back to θ₀ when the polish is rejected.
        ("polished", res.candidate, -res.nll_after),
    ):
        ratio = float(np.max(np.abs(th - theta_r) / bound))
        rel = float(np.max(np.abs(th - theta_r) / np.maximum(np.abs(theta_r), 1e-12)))
        print(
            f"{label:9s} ratio@1e-6 {ratio:9.2f}  max rel Δθ {rel:.1e}  "
            f"logLik − R {ll - ll_r:+.2e}"
        )
    print(
        f"polish: {res.reason}; KKT {res.kkt_before:.1e} -> {res.kkt_after:.1e}; "
        f"iterations {res.n_iter}; active rows {res.n_active}"
    )


if __name__ == "__main__":
    main()
