"""Issue #166: run unchanged cases before/after wiring the NLL Hessian.

Run from the repository root: .venv/bin/python research/trust-constr-hessian/sweep.py
Results go to stdout as JSON; redirect to before.json / after.json.
"""

import json
import platform
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray

from mltpy.basis import BernsteinBasis
from mltpy.optimizer import OptimizerConfig, optimize
from mltpy.variables import CensoredData, CensoringType

REF = Path(__file__).resolve().parents[2] / "reference"

# Dynamic optimize() kwargs mix arrays, censoring enums and distribution names.
type Case = tuple[
    str,
    BernsteinBasis,
    NDArray[np.float64] | CensoredData,
    dict[str, Any],
    dict[int, float],
    NDArray[np.float64] | None,
    float | None,
]


def load(prefix: str, name: str) -> NDArray[np.float64]:
    return np.loadtxt(REF / f"{prefix}_{name}.txt")


def cases() -> Iterator[Case]:
    # R fixtures: exact, right and interval censoring, all four distributions.
    for prefix, dist, sign, cens in [
        ("scaling_boxcox_normal", "normal", -1, CensoringType.NONE),
        ("scaling_colr", "logistic", 1, CensoringType.NONE),
        ("scaling_coxph", "min_extreme_value", 1, CensoringType.RIGHT),
        ("scaling_boxcox_interval", "normal", -1, CensoringType.INTERVAL),
        ("mlt_exponential_scaling", "exponential", 1, CensoringType.NONE),
        ("mlt_exponential_scaling_shift", "exponential", 1, CensoringType.NONE),
    ]:
        theta = load(prefix, "theta")
        exp = prefix.startswith("mlt_exponential")
        xs = load(prefix, "xs" if exp else "x_s").reshape(-1, 1)
        x_name = "xd" if exp else "x_d"
        x_path = REF / f"{prefix}_{x_name}.txt"
        x = np.loadtxt(x_path).reshape(-1, 1) if x_path.exists() else None
        p = len(theta) - (0 if x is None else x.shape[1]) - xs.shape[1]
        theta[p : p + (0 if x is None else x.shape[1])] *= sign
        if cens is CensoringType.INTERVAL:
            y = CensoredData.interval_censored(load(prefix, "lo"), load(prefix, "hi"))
        elif cens is CensoringType.RIGHT:
            y = CensoredData.right_censored(
                load(prefix, "y"), load(prefix, "event") == 0
            )
        else:
            y = load(prefix, "y")
        kwargs = dict(X=x, scaling=xs, censoring=cens, base_distribution=dist)
        offset_path = REF / f"{prefix}_offset.txt"
        if offset_path.exists():
            kwargs["offset"] = np.loadtxt(offset_path)
        yield (
            prefix,
            BernsteinBasis(p - 1, tuple(load(prefix, "support"))),
            y,
            kwargs,
            {},
            theta,
            float(load(prefix, "ll" if exp else "loglik")),
        )
        if prefix == "mlt_exponential_scaling_shift":
            basis = BernsteinBasis(p - 1, tuple(load(prefix, "support")))
            yield (
                "exponential_scale_only_offset",
                basis,
                y,
                dict(kwargs, X=None),
                {},
                None,
                None,
            )
            yield (
                "exponential_unscaled_offset",
                basis,
                y,
                dict(kwargs, scaling=None),
                {},
                None,
                None,
            )
            for censored in (
                CensoringType.RIGHT,
                CensoringType.LEFT,
                CensoringType.INTERVAL,
            ):
                if censored is CensoringType.RIGHT:
                    c = np.quantile(y, 0.8)
                    data = CensoredData.right_censored(np.minimum(y, c), y >= c)
                elif censored is CensoringType.LEFT:
                    c = np.quantile(y, 0.2)
                    data = CensoredData.left_censored(np.maximum(y, c), y <= c)
                else:
                    data = CensoredData.interval_censored(
                        y * 0.8, np.minimum(y * 1.2, basis.support[1])
                    )
                yield (
                    f"exponential_{censored.name}",
                    basis,
                    data,
                    dict(kwargs, censoring=censored),
                    {},
                    None,
                    None,
                )
            yield (
                "exponential_fixed_beta",
                basis,
                y,
                kwargs,
                {p: float(theta[p])},
                theta,
                float(load(prefix, "ll")),
            )

    for name, dist, sign, cens in [
        ("boxcox", "normal", -1, CensoringType.NONE),
        ("colr", "logistic", 1, CensoringType.NONE),
        ("coxph", "min_extreme_value", 1, CensoringType.RIGHT),
    ]:
        prefix = f"vcov_{name}"
        theta = load(f"weights_{name}", "theta")
        theta[-1] *= sign
        y = load(prefix, "y")
        if cens is CensoringType.RIGHT:
            y = CensoredData.right_censored(y, load(prefix, "event") == 0)
        kwargs = dict(
            X=load(prefix, "x").reshape(-1, 1),
            weights=load(f"weights_{name}", "w"),
            base_distribution=dist,
            censoring=cens,
        )
        yield (
            f"weights_{name}",
            BernsteinBasis(len(theta) - 2, tuple(load(prefix, "support"))),
            y,
            kwargs,
            {},
            theta,
            float(load(f"weights_{name}", "ll")),
        )

    # Controlled data exercises left censoring, offsets, scaling and free blocks.
    rng = np.random.default_rng(166)
    x = rng.uniform(-1, 1, (100, 1))
    xs = rng.uniform(-1, 1, (100, 1))
    y = np.clip(rng.normal(0.3 * x[:, 0], np.exp(-0.2 * xs[:, 0])), -2.9, 2.9)
    for cens in CensoringType:
        if cens is CensoringType.RIGHT:
            data = CensoredData.right_censored(np.minimum(y, 0.7), y >= 0.7)
        elif cens is CensoringType.LEFT:
            data = CensoredData.left_censored(np.maximum(y, -0.7), y <= -0.7)
        elif cens is CensoringType.INTERVAL:
            data = CensoredData.interval_censored(y - 0.05, y + 0.05)
        else:
            data = y
        for scaled in (False, True):
            kwargs = dict(
                X=x,
                censoring=cens,
                offset=0.1 * x[:, 0],
                weights=rng.uniform(0.5, 3, len(y)),
            )
            if scaled:
                kwargs["scaling"] = xs
            for fixed in (False, True):
                pins = (
                    {4: 0.0, 5: 0.2} if fixed and scaled else {4: 0.0} if fixed else {}
                )
                yield (
                    f"synthetic_{cens.name}_{scaled=}_{fixed=}",
                    BernsteinBasis(3, (-3.0, 3.0)),
                    data,
                    kwargs,
                    pins,
                    None,
                    None,
                )

    yield (
        "infeasible_fixed_start",
        BernsteinBasis(3, (0.0, 1.0)),
        np.array([0.1, 0.2, 0.4, 0.5, 0.7, 0.9]),
        {},
        {1: 5 / 3},
        None,
        None,
    )


def main() -> None:
    results = []
    for name, basis, y, kwargs, pins, theta_r, ll_r in cases():
        times = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                start = time.perf_counter()
                result = optimize(
                    basis,
                    y,
                    config=OptimizerConfig(
                        solver="trust-constr", random_state=0, fixed_params=pins
                    ),
                    **kwargs,
                )
                times.append(time.perf_counter() - start)
        assert np.isfinite(result.log_likelihood)
        assert np.all(np.isfinite(result.theta))
        assert np.all(np.diff(result.theta[: basis.order + 1]) >= -1e-7)
        assert all(result.theta[j] == value for j, value in pins.items())
        row = dict(
            name=name,
            seconds=float(np.median(times)),
            converged=result.converged,
            iterations=result.n_iter,
            restarts=result.n_restarts,
            theta=result.theta.tolist(),
            loglik=result.log_likelihood,
            warnings=sorted({str(w.message) for w in caught}),
        )
        if theta_r is not None:
            assert ll_r is not None
            np.testing.assert_allclose(result.log_likelihood, ll_r, rtol=1e-6)
            row.update(
                max_abs_theta_error=float(np.max(np.abs(result.theta - theta_r))),
                loglik_error=result.log_likelihood - ll_r,
            )
        results.append(row)
    print(
        json.dumps(
            dict(
                python=platform.python_version(),
                numpy=np.__version__,
                scipy=scipy.__version__,
                results=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
