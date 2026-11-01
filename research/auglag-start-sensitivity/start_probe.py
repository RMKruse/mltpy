"""pytest plugin: rerun auglag fits from perturbed starts and log diagnostics (#178).

Load with ``-p start_probe`` (this directory on ``PYTHONPATH``).  Controlled by
environment variables:

``MLTPY_START_SEED``
    ``k ≥ 0``.  ``0`` leaves every start unchanged.  ``k > 0`` replaces attempt
    0 of every auglag restart loop by ``perturb_fn`` applied ``k`` times to
    the path's default start.  ``perturb_fn`` is the path's own
    perturb-and-project step (shift: :func:`_perturb_and_project`, θ_b ~
    N(0, 0.1) then projected to the monotone cone and the exponential support
    floor; interaction: its column-wise analogue), drawn from the fit's own
    ``random_state`` generator, so each ``k`` is a distinct feasible start and
    the run is reproducible.  Restarts, tolerances, polish and result
    selection are untouched.
``MLTPY_START_LOG``
    Path of a JSONL file; one line per auglag fit with the test node id.

Recorded per fit: the seed, the returned auglag point's KKT residual and
``converged`` flag, whether the Newton-CG polish ran and whether it was
accepted, the number of equality rows, and the normalised slacks of every
inequality row (linear rows, box bounds of the increment coordinates, and the
nonlinear support block), evaluated at auglag's returned point in its own
coordinates.  Slacks are ``(a_i·x − b_i)/‖a_i‖`` (nonlinear rows:
``g_i(x)/‖∇g_i(x)‖``), so a row's slack is its distance to the boundary.

The wrappers only observe and reorder starts; they never change objectives,
constraints or the result a fit returns.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import mltpy.optimizer as opt

_SEED = int(os.environ.get("MLTPY_START_SEED", "0"))
_LOG = os.environ.get("MLTPY_START_LOG")
_current: dict[str, Any] = {"nodeid": None, "fit": None}
_records: list[dict[str, Any]] = []


def _norm_slack(values: NDArray[np.float64], rows: NDArray[np.float64]) -> list[float]:
    norms = np.linalg.norm(rows, axis=1)
    norms = np.where(norms > 0.0, norms, 1.0)
    return [float(v) for v in values / norms]


def _slacks(kwargs: dict[str, Any], x: NDArray[np.float64]) -> list[float]:
    out: list[float] = []
    A, b = kwargs.get("A_ineq"), kwargs.get("b_ineq")
    if A is not None and np.asarray(A).shape[0] > 0:
        A = np.atleast_2d(np.asarray(A, dtype=np.float64))
        rhs = np.zeros(A.shape[0]) if b is None else np.asarray(b, dtype=np.float64)
        out += _norm_slack(A @ x - rhs, A)
    bounds = kwargs.get("bounds")
    if bounds is not None:
        lo, hi = (np.asarray(v, dtype=np.float64) for v in bounds)
        out += [float(v) for v in (x - lo)[np.isfinite(lo)]]
        out += [float(v) for v in (hi - x)[np.isfinite(hi)]]
    nl = kwargs.get("nonlinear_ineq")
    if nl is not None:
        g, J = nl(x)
        out += _norm_slack(np.asarray(g, dtype=np.float64), np.atleast_2d(J))
    return out


def _wrap_restart_loop(orig: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(
        theta_init: NDArray[np.float64],
        config: Any,
        perturb_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if _SEED > 0 and config.solver == "auglag":
            start = np.array(theta_init, dtype=np.float64)
            for _ in range(_SEED):
                start = perturb_fn(np.array(theta_init, dtype=np.float64))
            theta_init = start
        return orig(theta_init, config, perturb_fn, *args, **kwargs)

    return wrapper


def _wrap_auglag(orig: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(fun: Any, x0: NDArray[np.float64], **kwargs: Any) -> Any:
        res = orig(fun, x0, **kwargs)
        x = np.asarray(res.theta, dtype=np.float64)
        C = kwargs.get("C_eq")
        fit = {
            "type": "fit",
            "nodeid": _current["nodeid"],
            "seed": _SEED,
            "kkt": float(res.kkt_residual),
            "converged": bool(res.converged),
            "n_eq": 0 if C is None else int(np.asarray(C).shape[0]),
            "slacks": _slacks(kwargs, x),
            "polish": None,
        }
        _records.append(fit)
        _current["fit"] = fit
        return res

    return wrapper


def _wrap_polish(orig: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        out = orig(*args, **kwargs)
        if _current["fit"] is not None:
            _current["fit"]["polish"] = "accepted" if out is not None else "rejected"
        return out

    return wrapper


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: Any) -> Iterator[None]:
    # Covers setup, call and teardown, so fits made inside module-scoped
    # fixtures are attributed to the first test that requests them.
    _current["nodeid"] = item.nodeid
    _current["fit"] = None
    yield
    _current["nodeid"] = None


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call" or report.outcome != "passed":
        msg = ""
        if report.failed:
            lines = [
                ln for ln in str(report.longrepr).splitlines() if ln.startswith("E ")
            ]
            msg = " | ".join(ln[1:].strip() for ln in lines[:8])
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


def pytest_configure(config: pytest.Config) -> None:
    opt._restart_loop = _wrap_restart_loop(opt._restart_loop)  # type: ignore[assignment]
    opt.auglag_minimize = _wrap_auglag(opt.auglag_minimize)  # type: ignore[assignment]
    opt._ncg_polish = _wrap_polish(opt._ncg_polish)  # type: ignore[assignment]


def pytest_unconfigure(config: pytest.Config) -> None:
    if _LOG:
        with open(_LOG, "w", encoding="utf-8") as fh:
            for rec in _records:
                fh.write(json.dumps(rec) + "\n")
