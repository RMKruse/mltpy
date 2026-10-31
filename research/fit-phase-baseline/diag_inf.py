import sys, warnings
import numpy as np
sys.argv = ["x"]
import harness as H
import mltpy.optimizer as mo
from mltpy.likelihood import InfeasibleParameterError
from mltpy import OptimizerConfig
orig = mo._negative_log_likelihood_from_dist
ev = []
def spy(theta, *a, **k):
    try:
        r = orig(theta, *a, **k); ev.append(("ok", float(r[0]) if isinstance(r, tuple) else float(r))); return r
    except InfeasibleParameterError:
        ev.append(("INFEASIBLE", None)); raise
mo._negative_log_likelihood_from_dist = spy
for n in (750, 1000):
    ev.clear()
    y, X = H._ia_data(n)
    m = H._ia_model(6, 3)(); m.optimizer_config = OptimizerConfig(random_state=0, polish=False)
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); m.fit(y, X)
    print(n, "evals", len(ev), "infeasible", sum(e[0] == "INFEASIBLE" for e in ev), "first 12:", [e[0][:3] + ("" if e[1] is None else f"{e[1]:.1f}") for e in ev[:12]])
