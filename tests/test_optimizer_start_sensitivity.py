"""Start robustness of auglag R parity (issue #178, acceptance test for #177).

Each case refits an auglag R-parity fixture from the default start (k = 0)
and from ten perturbed feasible starts (k = 1…10), and compares θ with R at
CLAUDE.md's ``rtol=1e-6, atol=1e-10`` for **every** start.

A perturbed start replaces attempt 0 of the fit's restart loop with the
path's own perturb-and-project step applied ``k`` times to the default start
(:func:`_perturbed_starts`).  The perturbation is drawn from the fit's own
``random_state`` generator, so every case is deterministic.  Restarts,
tolerances, the Newton-CG polish and result selection are unchanged.

Cases marked ``xfail(strict=True)`` match R only from some starts today
(measured in ``research/auglag-start-sensitivity/``):

- **Active-set fits** stop at auglag's KKT floor, and the polish skips every
  fit with an active or equality row.
- **Interior fits** are polished by trust-ncg at scipy's default
  ``gtol=1e-4``, which leaves θ ~1e-6 relative off the optimum.

#177 removes the marks; a strict xfail that starts passing fails the run.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import mltpy.optimizer as optimizer_module
from mltpy import MLT, ConditionalTransformationModel, OneHotBasis
from mltpy.basis import BernsteinBasis
from mltpy.optimizer import OptimizerConfig, optimize
from mltpy.tram import BoxCox

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_SEEDS = range(11)
_RTOL, _ATOL = 1e-6, 1e-10

Fit = Callable[[], NDArray[np.float64]]


def _perturbed_starts(monkeypatch: pytest.MonkeyPatch, k: int) -> None:
    """Start attempt 0 of every auglag restart loop from ``perturb_fn^k``."""
    original = optimizer_module._restart_loop

    def wrapper(
        theta_init: NDArray[np.float64],
        config: OptimizerConfig,
        perturb_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if k > 0 and config.solver == "auglag":
            start = np.array(theta_init, dtype=np.float64)
            for _ in range(k):
                start = perturb_fn(np.array(theta_init, dtype=np.float64))
            theta_init = start
        return original(theta_init, config, perturb_fn, *args, **kwargs)

    monkeypatch.setattr(optimizer_module, "_restart_loop", wrapper)


def _worst_ratio(
    fit: Fit, theta_r: NDArray[np.float64], monkeypatch: pytest.MonkeyPatch
) -> tuple[float, dict[int, float]]:
    """Largest ``|θ̂ − θ_R| / (atol + rtol·|θ_R|)`` over all starts (≤ 1 passes)."""
    ratios: dict[int, float] = {}
    for k in _SEEDS:
        with monkeypatch.context() as mp:
            _perturbed_starts(mp, k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                theta = fit()
        bound = _ATOL + _RTOL * np.abs(theta_r)
        ratios[k] = float(np.max(np.abs(theta - theta_r) / bound))
    return max(ratios.values()), ratios


def _load_txt(prefix: str, name: str) -> NDArray[np.float64]:
    return np.atleast_1d(np.loadtxt(REF_DIR / f"{prefix}{name}.txt"))


# ---------------------------------------------------------------------------
# Cases: each returns (fit, θ_R)
# ---------------------------------------------------------------------------


def _auglag_json(name: str) -> tuple[Fit, NDArray[np.float64]]:
    """``reference/auglag/<name>.json`` (alabama::auglag, test_r_parity_auglag)."""
    ref = json.loads((REF_DIR / "auglag" / f"{name}.json").read_text())
    y = np.asarray(ref["y"], dtype=np.float64)
    X = np.asarray(ref["X"], dtype=np.float64) if "X" in ref else None
    basis = BernsteinBasis(
        order=ref["order"], support=tuple(float(v) for v in ref["support"])
    )
    dist = "exponential" if "exponential" in name else "normal"

    def fit() -> NDArray[np.float64]:
        # random_state=0 (test_r_parity_auglag leaves it None, so its
        # restarts draw a fresh generator on every run).
        return optimize(
            basis,
            y,
            X=X,
            config=OptimizerConfig(random_state=0),
            base_distribution=dist,
        ).theta

    return fit, np.asarray(ref["theta"], dtype=np.float64)


def _exponential_offset() -> tuple[Fit, NDArray[np.float64]]:
    """#160 per-row offset + covariate fixture (test_exponential_offset_support)."""
    p = "mlt_exponential_offset_"
    y, x, off = _load_txt(p, "y"), _load_txt(p, "x"), _load_txt(p, "offset")

    def fit() -> NDArray[np.float64]:
        model = ConditionalTransformationModel(
            BernsteinBasis(order=3, support=(0.0, 1.0)),
            base_distribution="exponential",
            optimizer_config=OptimizerConfig(random_state=0),
        )
        model.fit(y, X=x[:, None], offset=off)
        assert model.theta_ is not None
        return model.theta_

    return fit, _load_txt(p, "theta")


def _scaling_shift(*, pin_beta: bool) -> tuple[Fit, NDArray[np.float64]]:
    """#115 exponential + scaling + shift fixture (test_exponential_scaling_shift)."""
    p = "mlt_exponential_scaling_shift_"
    y, x_d, x_s = _load_txt(p, "y"), _load_txt(p, "xd"), _load_txt(p, "xs")
    off, sup, theta_r = (
        _load_txt(p, "offset"),
        _load_txt(p, "support"),
        _load_txt(p, "theta"),
    )
    fixed = {5: float(theta_r[5])} if pin_beta else None

    def fit() -> NDArray[np.float64]:
        model = MLT(
            order=4,
            support=(float(sup[0]), float(sup[1])),
            base_distribution="exponential",
            scaling=x_s[:, None],
            optimizer_config=OptimizerConfig(random_state=0, fixed_params=fixed),
        )
        model.fit(y, X=x_d[:, None], offset=off)
        return np.asarray(model.theta_, dtype=np.float64)

    return fit, theta_r


def _boxcox_interacting_scaling() -> tuple[Fit, NDArray[np.float64]]:
    """One-hot interacting BoxCox + scaling (test_tram_boxcox_interacting_scaling)."""
    p = "stram_onehot_bs_boxcox_"
    names = ("y_train", "s_train", "x_s_train", "y_support", "theta_onehot", "gamma")
    ref = {n: _load_txt(p, n) for n in names}
    a, b = ref["y_support"]

    def fit() -> NDArray[np.float64]:
        model = BoxCox(
            support=(float(a), float(b)),
            order=2,
            interacting=OneHotBasis(K=3),
            scaling=ref["x_s_train"].reshape(-1, 1),
            optimizer_config=OptimizerConfig(random_state=0),
        )
        model.fit(ref["y_train"], X=ref["s_train"])
        return np.asarray(model.theta_, dtype=np.float64)

    return fit, np.concatenate([ref["theta_onehot"], ref["gamma"]])


_ACTIVE = "active-set fit: the polish skips fits with active or equality rows"
_INTERIOR = "interior fit: the trust-ncg polish stops at scipy's gtol=1e-4"

_CASES: dict[str, tuple[Callable[[], tuple[Fit, NDArray[np.float64]]], str | None]] = {
    "auglag_json_lm": (lambda: _auglag_json("lm_n50_seed42"), None),
    "scaling_shift_free": (lambda: _scaling_shift(pin_beta=False), None),
    "auglag_json_exponential": (
        lambda: _auglag_json("exponential_n50_seed43"),
        _ACTIVE + " (50 of 51 rows active in 3 parameters)",
    ),
    "exponential_offset": (_exponential_offset, _ACTIVE),
    "scaling_shift_beta_pinned": (lambda: _scaling_shift(pin_beta=True), _ACTIVE),
    "boxcox_interacting_scaling": (_boxcox_interacting_scaling, _INTERIOR),
}


@pytest.fixture(params=sorted(_CASES))
def case(request: pytest.FixtureRequest) -> Iterator[str]:
    _, reason = _CASES[request.param]
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
    yield request.param


def test_theta_matches_r_from_every_start(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """θ matches R at rtol=1e-6, atol=1e-10 from the default and ten perturbed starts."""
    build, _ = _CASES[case]
    fit, theta_r = build()
    worst, ratios = _worst_ratio(fit, theta_r, monkeypatch)
    failing = {k: round(r, 2) for k, r in ratios.items() if r > 1.0}
    assert worst <= 1.0, f"{case}: |Δθ|/(atol + rtol|θ_R|) > 1 at starts {failing}"


def test_perturbed_start_changes_the_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness really moves attempt 0 (k = 0 leaves it alone)."""
    seen: list[NDArray[np.float64]] = []
    original = optimizer_module._restart_loop

    def spy(theta_init: NDArray[np.float64], *args: Any, **kwargs: Any) -> Any:
        seen.append(np.array(theta_init))
        return original(theta_init, *args, **kwargs)

    monkeypatch.setattr(optimizer_module, "_restart_loop", spy)
    fit, _ = _auglag_json("lm_n50_seed42")
    for k in (0, 1, 2):
        with monkeypatch.context() as mp:
            _perturbed_starts(mp, k)
            fit()
    default, first, second = seen
    np.testing.assert_array_equal(default, np.linspace(0.0, 1.0, default.size))
    assert np.all(np.diff(first) >= 0.0) and np.all(np.diff(second) >= 0.0)
    assert not np.array_equal(first, default)
    assert not np.array_equal(first, second)
