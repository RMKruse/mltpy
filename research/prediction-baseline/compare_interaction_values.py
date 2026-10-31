"""Compare #146 with a git ref using #140's fitted cases and input seeds.

Run from this directory: python compare_interaction_values.py --base REF --out FILE
Uses the original interaction quantile method on the same fitted model; checks
actual public predict/simulate outputs and warnings before alternating warm
timings (at least five reps). Each memory measurement starts with empty caches.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
import subprocess
import time
import tracemalloc
import warnings
from collections.abc import Callable
from types import MethodType
from typing import cast

import harness as h
import numpy as np
from numpy.typing import NDArray

import mltpy.model as mm


def original_method(base: str) -> Callable[..., NDArray[np.float64]]:
    path = "src/mltpy/model.py"
    source = subprocess.check_output(["git", "show", f"{base}:{path}"], text=True)
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ConditionalTransformationModel"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_predict_quantile_interaction"
    )
    namespace = dict(vars(mm))
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec"), namespace)
    return cast(Callable[..., NDArray[np.float64]], namespace[method.name])


def call(
    model: h.ConditionalTransformationModel, inp: h.Inputs, op: str, seed: int
) -> NDArray[np.float64]:
    if op == "simulate":
        return model.simulate(
            len(inp["y_new"]),
            X=inp["X_new"],
            X_scale=inp.get("X_scale_new"),
            random_state=seed,
        )
    return model.predict(**inp)


def recorded(
    model: h.ConditionalTransformationModel, inp: h.Inputs, op: str, seed: int
) -> tuple[NDArray[np.float64], list[tuple[str, str]]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = call(model, inp, op, seed)
    return result, [(w.category.__name__, str(w.message)) for w in caught]


def memory(
    model: h.ConditionalTransformationModel, inp: h.Inputs, op: str, seed: int
) -> dict[str, object]:
    h.clear_caches()
    tracemalloc.start()
    try:
        output = call(model, inp, op, seed)
        retained, peak = tracemalloc.get_traced_memory()
        return {
            "retained_mib": retained / 2**20,
            "peak_mib": peak / 2**20,
            "cache": h.cache_stats(),
            "output_mib": output.nbytes / 2**20,
        }
    finally:
        tracemalloc.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()
    assert args.reps >= 5
    original = original_method(args.base)
    cases = {k: c for k, c in h.build_cases().items() if c.path == "interaction"}
    # Add log-time Survreg through its public constructor, with and without scale.
    for scaled in (False, True):
        rng = np.random.default_rng(146)
        X = rng.uniform(-1, 1, (1000, 1))
        Xs = rng.uniform(-1, 1, (1000, 1)) if scaled else None
        spread = 1.0 if Xs is None else np.exp(0.2 * Xs[:, 0])
        y = np.exp(np.clip(0.4 * X[:, 0] + spread * rng.normal(size=1000), -3.9, 3.9))
        name = "survreg_interacting" + ("_scaled" if scaled else "")
        template = cases["ia_scaled_bs3x1_m9_n1000" if scaled else "ia_bs3x1_m8_n1000"]
        cases[name] = h.Case(
            name,
            "interaction",
            lambda Xs=Xs: h.Survreg(
                order=1,
                support=(float(np.exp(-4)), float(np.exp(4))),
                distribution="weibull",
                interacting=h.BernsteinBasis(1, (-1.0, 1.0)),
                scaling=Xs,
                optimizer_config=h.cfg(),
            ),
            (y, X),
            {},
            template.ops,
            y,
        )
    report = {
        "base": args.base,
        "reps": args.reps,
        "environment": h.env(),
        "fits": {},
        "cells": [],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, case in cases.items():
            model, validation = h.fit_and_validate(case)
            assert validation["valid"], validation
            report["fits"][name] = validation
            old = MethodType(original, model)
            new = model._predict_quantile_interaction
            for m in h.M_SIZES:
                for op in ("quantile", "simulate"):
                    seed = 1000 + m
                    inp = case.ops[op](np.random.default_rng(seed), m)
                    h.clear_caches()
                    model._predict_quantile_interaction = old
                    before, old_warnings = recorded(model, inp, op, seed)
                    model._predict_quantile_interaction = new
                    after, new_warnings = recorded(model, inp, op, seed)
                    np.testing.assert_array_equal(
                        before.view(np.uint64), after.view(np.uint64)
                    )
                    assert old_warnings == new_warnings
                    times = {"before_ms": [], "after_ms": []}
                    for rep in range(args.reps):
                        order = [("before_ms", old), ("after_ms", new)]
                        if rep % 2:
                            order.reverse()
                        for key, method in order:
                            model._predict_quantile_interaction = method
                            start = time.perf_counter()
                            call(model, inp, op, seed)
                            times[key].append(1000 * (time.perf_counter() - start))
                    cell = {
                        "case": name,
                        "op": op,
                        "m": m,
                        **times,
                        "before_median_ms": statistics.median(times["before_ms"]),
                        "after_median_ms": statistics.median(times["after_ms"]),
                        "bitwise_equal": True,
                        "warnings": new_warnings,
                    }
                    for key, method in (("before", old), ("after", new)):
                        model._predict_quantile_interaction = method
                        cell[key] = memory(model, inp, op, seed)
                    report["cells"].append(cell)
                    print(json.dumps(cell), flush=True)
            del model._predict_quantile_interaction
    with open(args.out, "w") as out:
        json.dump(report, out, indent=2)


if __name__ == "__main__":
    main()
