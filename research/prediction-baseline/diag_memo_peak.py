"""Transient peak of the per-distinct-shift memo vs no reuse (#140, table F)."""

import sys
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import harness as h  # noqa: E402
from mltpy.likelihood import _get_dist  # noqa: E402

warnings.simplefilter("ignore")
c = h.build_cases()["coxph_x3"]
model = c.make().fit(*c.fit_args, **c.fit_kwargs)
dist = _get_dist(model.base_distribution)
for op in ("quantile_varying", "quantile_10prof"):
    inp = c.ops[op](np.random.default_rng(1), 10_000)
    hb, qg, shift, zf = h.rstyle_setup(model, inp)
    probs = np.asarray(inp["y_new"], dtype=float)
    for reuse in ("none", "consecutive", "unique"):
        tracemalloc.start()
        b0 = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = h.rstyle(probs, shift, hb, qg, dist, zf, reuse)
        peak = tracemalloc.get_traced_memory()[1] - b0
        tracemalloc.stop()
        print(f"{op:18s} {reuse:12s} peak {peak / 2**20:6.2f} MiB curves {out[2]}")
    # sorted variant: stable argsort by shift, consecutive reuse, scatter back
    lib = h.rstyle(probs, shift, hb, qg, dist, zf, "none")
    tracemalloc.start()
    b0 = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    import time

    t0 = time.perf_counter()
    order = np.argsort(shift, kind="stable")
    o_s, sat, built = h.rstyle(probs[order], shift[order], hb, qg, dist, zf,
                               "consecutive")
    res = np.empty_like(o_s)
    res[order] = o_s
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] - b0
    tracemalloc.stop()
    print(f"{op:18s} {'sorted':12s} peak {peak / 2**20:6.2f} MiB curves {built} "
          f"{1e3 * dt:.1f} ms bitwise_equal={np.array_equal(res, lib[0])} "
          f"sat_same={sat == lib[1]}")
