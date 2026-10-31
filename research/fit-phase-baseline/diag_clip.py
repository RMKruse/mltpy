import sys, warnings
import numpy as np
sys.argv = ["x"]
import harness as H
import mltpy.likelihood as ml
from mltpy import OptimizerConfig
y, X = H._ia_data(1000)
m = H._ia_model(6, 3)(); m.optimizer_config = OptimizerConfig(random_state=0, polish=False)
with warnings.catch_warnings(): warnings.simplefilter("ignore"); m.fit(y, X)
th = m.theta_
B, dB = m.basis.evaluate_with_derivative(y, X)
h, hp = B @ th, dB @ th
print("h range", h.min(), h.max(), "frac |h|>=30:", np.mean(np.abs(h) >= 30), "min h'", hp.min())
nll, g = ml.negative_log_likelihood(th, m.basis, y, X, gradient=True)
print("nll", nll, "grad_inf", np.abs(g).max())
print("Theta col ranges", np.round(m.Theta_[[0, -1]], 2))
print("y range", y.min(), y.max(), "basis support (-4,4)")
