"""Diagnose the false stall-convergence on the m=28 interaction fit."""
import sys, time, warnings
import numpy as np
sys.argv = ["x", "--case", "x", "--out", "x"]
import harness as H
import mltpy._auglag as A
import mltpy.optimizer as mo
from mltpy import OptimizerConfig

orig = A.minimize
log = []
def spy(fun, x0, **kw):
    r = orig(fun, x0, **kw)
    log.append((r.nit, r.nfev, str(r.message)[:60], float(r.fun), bool(np.isfinite(r.fun))))
    return r
A.minimize = spy

def fit(n, seed_data=20261031, oy=6, ox=3, polish=True):
    rng = np.random.default_rng(seed_data)
    X = rng.uniform(-1, 1, (n, 1)); y = np.clip(rng.normal(0.4 * X[:, 0], 1), -3.9, 3.9)
    log.clear()
    m = H._ia_model(oy, ox)()
    m.optimizer_config = OptimizerConfig(random_state=0, polish=polish)
    t = time.perf_counter()
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); m.fit(y, X)
    dt = time.perf_counter() - t
    r = m.result_
    # NLL at the default start
    import mltpy.likelihood as ml
    p, q = oy + 1, ox + 1
    th0 = np.outer(np.linspace(0, 1, p), np.ones(q)).ravel()
    nll0 = ml.negative_log_likelihood(th0, m.basis, y, X, gradient=False)
    return dict(n=n, sec=round(dt, 2), conv=r.converged, kkt=f"{r.kkt_residual:.1e}", ll=round(r.log_likelihood, 2),
                outer=r.n_outer_iter, inner=r.n_iter, nll_start=nll0, inner_log=list(log))

for n in (250, 500, 750, 1000, 2000):
    d = fit(n, polish=False)
    print({k: v for k, v in d.items() if k != "inner_log"}); print("   inner:", d["inner_log"][:4])
for s in (1, 2, 3):
    d = fit(1000, seed_data=s, polish=False)
    print("data seed", s, {k: v for k, v in d.items() if k != "inner_log"})
