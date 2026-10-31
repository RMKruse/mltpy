"""Compare #145 against a git ref using the #140 fitted cases and input seeds.

Run: python research/prediction-baseline/compare_shift_reuse.py --base REF --out FILE
Only the original quantile methods are loaded from REF; both versions use the
same fitted model. Alternating warm timings exclude fitting and tracing.
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
from types import MethodType

import harness as h
import numpy as np

import mltpy.model as mm
import mltpy.tram as tram


def original_method(base, module, cls):
    path = f"src/mltpy/{module.__name__.split('.')[-1]}.py"
    source = subprocess.check_output(["git", "show", f"{base}:{path}"], text=True)
    tree = ast.parse(source)
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(
        n
        for n in model.body
        if isinstance(n, ast.FunctionDef) and n.name == "_predict_quantile"
    )
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec"), namespace)
    return namespace["_predict_quantile"]


def recorded(model, inp):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = model.predict(**inp)
    return result, [(w.category.__name__, str(w.message)) for w in caught]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reps", type=int, default=9)
    args = parser.parse_args()
    assert args.reps >= 5
    originals = {
        "right_baseline": original_method(
            args.base, mm, "ConditionalTransformationModel"
        ),
        "coxph_x3": original_method(args.base, mm, "ConditionalTransformationModel"),
        "survreg_x3": original_method(args.base, tram, "Survreg"),
    }
    report = {"base": args.base, "reps": args.reps, "environment": h.env(), "cells": []}
    cases = h.build_cases()
    for name, original in originals.items():
        case = cases[name]
        model, validation = h.fit_and_validate(case)
        assert validation["valid"], validation
        old = MethodType(original, model)
        new = model._predict_quantile
        for op, make in case.ops.items():
            if op == "distribution":
                continue
            for m in (100, 1000, 10_000):
                inp = make(np.random.default_rng(1000 + m), m)
                h.clear_caches()
                model._predict_quantile = old
                before, old_warnings = recorded(model, inp)
                model._predict_quantile = new
                after, new_warnings = recorded(model, inp)
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
                        model._predict_quantile = method
                        start = time.perf_counter()
                        h.call(model, inp)
                        times[key].append(1000 * (time.perf_counter() - start))
                model._predict_quantile = new
                h.clear_caches()
                tracemalloc.start()
                h.call(model, inp)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                cell = {
                    "case": name,
                    "op": op,
                    "m": m,
                    **times,
                    "before_median_ms": statistics.median(times["before_ms"]),
                    "after_median_ms": statistics.median(times["after_ms"]),
                    "peak_mib": peak / 2**20,
                    "bitwise_equal": True,
                    "warnings": new_warnings,
                }
                report["cells"].append(cell)
                print(json.dumps(cell), flush=True)
        del model._predict_quantile
    with open(args.out, "w") as out:
        json.dump(report, out, indent=2)


if __name__ == "__main__":
    main()
