"""Capture auglag fits and rebuild their constrained problem for the prototype.

:func:`capture` wraps ``optimize`` (in ``mltpy.optimizer`` and where
``mltpy.model`` imports it) and records every auglag call's inputs and
result.  :func:`build_problem` rebuilds the problem those inputs define, the
way ``_optimize_auglag`` / ``_optimize_interaction`` do:

- **shift:** ``build_constraint_matrices`` plus one equality row per
  ``fixed_params`` entry; the analytic NLL Hessian.
- **interaction:** ``build_constraint_matrices_interaction``; the
  finite-difference Hessian production's interaction polish uses (#118).

The objective is the library's ``negative_log_likelihood`` with the caller's
weights (the true likelihood, not SLSQP's mean-1 rescaling).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from polish import Problem

import mltpy.model as model_module
import mltpy.optimizer as optimizer_module
from mltpy.basis import InteractionBasis
from mltpy.constraints import (
    build_constraint_matrices,
    build_constraint_matrices_interaction,
)
from mltpy.likelihood import (
    _get_dist,
    _hessian_interaction_fd,
    _InteractionDesign,
    hessian,
    negative_log_likelihood,
)
from mltpy.optimizer import OptimizerConfig

_SIG = inspect.signature(optimizer_module.optimize)


@dataclass
class Call:
    kwargs: dict[str, Any]
    result: Any


@contextmanager
def capture(*extra_modules: Any) -> Iterator[list[Call]]:
    """Record every auglag ``optimize`` call made inside the block.

    ``extra_modules`` are modules that bound ``optimize`` by name at import
    time (``from mltpy.optimizer import optimize``); their binding is patched
    too.
    """
    calls: list[Call] = []
    original = optimizer_module.optimize

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = _SIG.bind(*args, **kwargs)
        bound.apply_defaults()
        result = original(*args, **kwargs)
        cfg = bound.arguments["config"] or OptimizerConfig()
        if cfg.solver == "auglag":
            calls.append(Call(dict(bound.arguments), result))
        return result

    modules = (optimizer_module, model_module, *extra_modules)
    saved = [m.optimize for m in modules]
    for m in modules:
        m.optimize = wrapper
    try:
        yield calls
    finally:
        for m, orig in zip(modules, saved, strict=True):
            m.optimize = orig


def build_problem(kw: dict[str, Any]) -> Problem:
    basis, y, X = kw["basis"], kw["y"], kw["X"]
    censoring, dist_name = kw["censoring"], kw["base_distribution"]
    weights, offset, scaling = kw["weights"], kw["offset"], kw["scaling"]
    config = kw["config"] or OptimizerConfig()
    nonneg = dist_name == "exponential"
    q_s = scaling.shape[1] if scaling is not None else 0

    def f(t: np.ndarray) -> tuple[float, np.ndarray]:
        v, g = negative_log_likelihood(
            t,
            basis,
            y,
            X,
            censoring=censoring,
            gradient=True,
            base_distribution=dist_name,
            weights=weights,
            offset=offset,
            scaling=scaling,
        )
        return float(v), np.asarray(g, dtype=np.float64)

    if isinstance(basis, InteractionBasis):
        x_arr = basis._coerce_x(X)
        cm = build_constraint_matrices_interaction(
            basis,
            q_s=q_s,
            nonneg_lower=nonneg,
            X=x_arr if nonneg else None,
            offset=offset if nonneg else None,
        )
        kron = _InteractionDesign.build(basis, y, x_arr)
        dist = _get_dist(dist_name)

        def hess(t: np.ndarray) -> np.ndarray:
            return _hessian_interaction_fd(
                kron, t, dist, weights=weights, offset=offset, scaling=scaling
            )
    else:
        n_params = basis.order + 1
        total = n_params + (X.shape[1] if X is not None else 0) + q_s
        cm = build_constraint_matrices(
            n_params,
            lower=config.lower,
            upper=config.upper,
            total_params=total,
            nonneg_lower=nonneg,
            X=X if nonneg else None,
            offset=offset if nonneg else None,
            q_s=q_s,
            X_scale=scaling if nonneg else None,
        )
        if config.fixed_params:
            rows = np.zeros((len(config.fixed_params), total))
            rhs = np.empty(len(config.fixed_params))
            for r, (idx, val) in enumerate(config.fixed_params.items()):
                rows[r, idx], rhs[r] = 1.0, float(val)
            cm.C_eq = np.vstack([cm.C_eq, rows])  # type: ignore[misc]
            cm.d_eq = np.concatenate([cm.d_eq, rhs])  # type: ignore[misc]

        def hess(t: np.ndarray) -> np.ndarray:
            return hessian(
                t,
                basis,
                y,
                X,
                censoring,
                dist_name,
                weights=weights,
                offset=offset,
                scaling=scaling,
            )

    g = cm.nonlinear_ineq
    return Problem(
        f=f,
        hess=hess,
        A=np.atleast_2d(np.asarray(cm.A_ineq, dtype=np.float64)),
        b=np.asarray(cm.b_ineq, dtype=np.float64),
        C=np.atleast_2d(np.asarray(cm.C_eq, dtype=np.float64)).reshape(
            -1, cm.A_ineq.shape[1]
        ),
        d=np.asarray(cm.d_eq, dtype=np.float64),
        g=g,
        g_hess=getattr(g, "hessian", None),
    )
