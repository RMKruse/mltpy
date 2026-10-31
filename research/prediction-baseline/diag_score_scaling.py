"""score() on scaled models: compare with the fitted log-likelihood (finding, #140)."""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import harness  # noqa: E402

warnings.simplefilter("ignore")
C = harness.build_cases()
for name in ("boxcox_scaled", "ia_scaled_bs3x1_m9_n1000"):
    c = C[name]
    m = c.make().fit(*c.fit_args, **c.fit_kwargs)
    X = c.fit_kwargs.get("X", c.fit_args[1] if len(c.fit_args) > 1 else None)
    try:
        s = m.score(c.fit_args[0], X=X)
        print(name, "fit ll", m.result_.log_likelihood, "score()", s,
              "diff", s - m.result_.log_likelihood, "theta len", m.theta_.size)
    except Exception as e:  # noqa: BLE001 - recording the failure is the point
        print(name, "score() raised", type(e).__name__, str(e)[:120])
    # sanity: same model, score without scaling at all (gamma = 0) for contrast
    print("  gamma_", np.asarray(m.gamma_))
