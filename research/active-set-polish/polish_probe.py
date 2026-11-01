"""pytest plugin: run the prototype polish after every auglag fit (#177).

Load with ``-p polish_probe`` (this directory and ``src`` on ``PYTHONPATH``).
It composes with ``research/auglag-start-sensitivity/start_probe.py``
(``-p start_probe``) for perturbed starts.

Environment variables:

``MLTPY_POLISH_MODE``
    ``measure`` (default): polish and log, but return auglag's result
    unchanged.  ``apply``: when the polish is accepted, return the result
    with ``theta`` and ``log_likelihood`` replaced by the polished values
    (every other field is auglag's), so the suite runs as if the polish
    shipped.
``MLTPY_POLISH_LOG``
    JSONL path: one line per auglag fit and one per non-passing test phase.

Every binding of ``optimize`` is wrapped: ``mltpy.optimizer``,
``mltpy.model`` and each collected test module that imported the name.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from harness import _SIG, build_problem
from polish import polish

import mltpy.model as model_module
import mltpy.optimizer as optimizer_module
from mltpy.basis import InteractionBasis
from mltpy.optimizer import OptimizerConfig

_MODE = os.environ.get("MLTPY_POLISH_MODE", "measure")
_LOG = os.environ.get("MLTPY_POLISH_LOG")
_SEED = int(os.environ.get("MLTPY_START_SEED", "0"))
_current: dict[str, Any] = {"nodeid": None}
_records: list[dict[str, Any]] = []
_ORIGINAL = optimizer_module.optimize


def _wrapper(*args: Any, **kwargs: Any) -> Any:
    result = _ORIGINAL(*args, **kwargs)
    bound = _SIG.bind(*args, **kwargs)
    bound.apply_defaults()
    kw = dict(bound.arguments)
    cfg = kw["config"] or OptimizerConfig()
    # Some tests replace the solver path with a stub returning None.
    if (
        result is None
        or cfg.solver != "auglag"
        or not np.all(np.isfinite(result.theta))
    ):
        return result
    rec: dict[str, Any] = {
        "type": "fit",
        "nodeid": _current["nodeid"],
        "seed": _SEED,
        "path": "interaction" if isinstance(kw["basis"], InteractionBasis) else "shift",
        "n_params": int(np.asarray(result.theta).size),
        "auglag_converged": bool(result.converged),
    }
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = build_problem(kw)
            res = polish(prob, np.asarray(result.theta, dtype=np.float64))
    except Exception as exc:  # research probe: record, never break the suite
        rec.update(
            error=f"{type(exc).__name__}: {exc}"[:200], seconds=time.perf_counter() - t0
        )
        _records.append(rec)
        return result
    theta0 = np.asarray(result.theta, dtype=np.float64)
    scale = np.maximum(np.abs(theta0), 1.0)
    rec.update(
        seconds=time.perf_counter() - t0,
        accepted=res.accepted,
        reason=res.reason,
        kkt_before=res.kkt_before,
        kkt_after=res.kkt_after,
        viol_before=res.viol_before,
        viol_after=res.viol_after,
        dnll=res.nll_after - res.nll_before,
        dtheta_rel=float(np.max(np.abs(res.candidate - theta0) / scale)),
        n_iter=res.n_iter,
        n_active=res.n_active,
        added=res.n_added,
        dropped=res.n_dropped,
    )
    _records.append(rec)
    if _MODE == "apply" and res.accepted:
        return replace(result, theta=res.theta, log_likelihood=-res.nll_after)
    return result


def pytest_configure(config: pytest.Config) -> None:
    optimizer_module.optimize = _wrapper  # type: ignore[assignment]
    model_module.optimize = _wrapper  # type: ignore[assignment]


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    seen: set[int] = set()
    for item in items:
        mod = getattr(item, "module", None)
        if mod is None or id(mod) in seen:
            continue
        seen.add(id(mod))
        if getattr(mod, "optimize", None) is _ORIGINAL:
            mod.optimize = _wrapper


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: Any) -> Iterator[None]:
    _current["nodeid"] = item.nodeid
    yield
    _current["nodeid"] = None


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.outcome == "passed" and report.when != "call":
        return
    msg = ""
    if report.failed:
        lines = [ln for ln in str(report.longrepr).splitlines() if ln.startswith("E ")]
        msg = " | ".join(ln[1:].strip() for ln in lines[:6])
    _records.append(
        {
            "type": "outcome",
            "nodeid": report.nodeid,
            "seed": _SEED,
            "when": report.when,
            "outcome": report.outcome,
            "message": msg,
        }
    )


def pytest_unconfigure(config: pytest.Config) -> None:
    if _LOG:
        with open(_LOG, "w", encoding="utf-8") as fh:
            for rec in _records:
                fh.write(json.dumps(rec) + "\n")
