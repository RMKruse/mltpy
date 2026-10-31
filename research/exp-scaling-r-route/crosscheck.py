"""Evaluate mltpy's log-likelihood at the R-fitted thetas from probe.R.

Run from the repo root:
    PYTHONPATH=src python research/exp-scaling-r-route/crosscheck.py \
        > research/exp-scaling-r-route/crosscheck.out

Only the public ``log_likelihood`` is called; it carries no exponential+scaling
guard (that guard lives in ``MLT.__init__`` and ``optimizer.optimize``), so it
evaluates the formula h = (a(y)' theta_b) * exp(0.5 * x_s * gamma) + x_d * beta
with log f_Z(h) = -h.  theta is passed in R's raw-ctm order with NO sign change.
"""

import csv
from pathlib import Path

import numpy as np

import mltpy
from mltpy.basis import BernsteinBasis
from mltpy.likelihood import log_likelihood

HERE = Path(__file__).parent
print(
    "mltpy imported from:",
    Path(mltpy.__file__).resolve().relative_to(HERE.resolve().parents[1]),
)

with (HERE / "fixtures.csv").open() as fh:
    rows = list(csv.DictReader(fh))
with (HERE / "fixtures_theta.csv").open() as fh:
    thetas = list(csv.DictReader(fh))

for fit in ("A_R_semantics", "A_mltpy_semantics", "B_fixed_Bs1"):
    fx = fit[0]
    data = [r for r in rows if r["fixture"] == fx]
    y = np.array([float(r["y"]) for r in data])
    x_d = np.array([[float(r["x_d"])] for r in data])
    x_s = np.array([[float(r["x_s"])] for r in data])
    th_rows = [t for t in thetas if t["fit"] == fit]
    theta = np.array([float(t["value"]) for t in th_rows])
    names = [t["name"] for t in th_rows]
    ll_r = float(th_rows[0]["loglik"])
    a, b = float(th_rows[0]["a"]), float(th_rows[0]["b"])
    basis = BernsteinBasis(order=4, support=(a, b))
    X = x_d if fx == "A" else None
    ll_py = log_likelihood(
        theta, basis, y, X=X, base_distribution="exponential", scaling=x_s
    )
    print(f"\n{fit}: names={names}")
    print(f"  R logLik     = {ll_r:.12f}")
    print(f"  mltpy logLik = {ll_py:.12f}")
    print(f"  |diff|       = {abs(ll_py - ll_r):.3e}")
    # Flip the beta sign: must NOT match (confirms raw ctm uses '+x_d*beta').
    if fx == "A":
        th_flip = theta.copy()
        th_flip[5] = -th_flip[5]
        ll_flip = log_likelihood(
            th_flip, basis, y, X=X, base_distribution="exponential", scaling=x_s
        )
        print(
            f"  with beta negated: {ll_flip:.6f} "
            "(differs unless beta ~ 0, as in A_mltpy_semantics)"
        )
