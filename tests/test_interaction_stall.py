"""Interaction auglag must not report a false stall as converged — #135, #142.

The #142 case uses Bernstein-y (order 6, support (-4, 4)) ⊗ Bernstein-x
(order 3, support (-1, 1)), Normal base, exact data, n ∈ {1000, 2000}, fitted
with the issue's exact configuration ``OptimizerConfig(random_state=0)``.
Before the ADR 0004
fixes (#134, merged in #136) the first L-BFGS-B inner solve hit an
infeasible trial point, the interaction objective returned ``(inf, 0)``, the
line search stopped without moving, and the auglag stall exit accepted that
point as ``converged=True`` with no ``ConvergenceWarning``:

* n = 1000: logLik ≈ -2604.70 (optimum ≈ -1403.88), KKT residual ≈ 114;
* n = 2000: KKT residual ≈ 230.

Restarts never ran because attempt 0 had "converged".  At the true MLE some
monotonicity multipliers are non-zero, so the Newton-CG polish does not run
and the fit is auglag's own.

The data are NumPy draws (seed 20261031) written by
``reference/write_interaction_stall_data.py``; the R ``mlt`` fit on the same
data comes from the ``interaction_stall`` block of
``reference/generate_reference.R``.

Tolerances:

* ``log_likelihood`` parity at the CLAUDE.md ``rtol=1e-6, atol=1e-10``.
* ``Θ`` parity at ``rtol=1e-3, atol=1e-4``.  The MLE has many stacked active
  monotonicity rows (at n = 2000 one Θ column is completely flat), so Θ is
  weakly determined along them: mltpy's three solvers agree with each other
  but sit up to ~1e-4 (relative) from R's spg-polished Θ at a logLik
  difference of ~2e-9.  The false stall was off by up to 7.3 in Θ.
* KKT residual at most ``1e-6 · (1 + |logLik|)``; healthy fits here sit
  near 1e-9 · (1 + |logLik|), the false stall at ≈ 0.04 · (1 + |logLik|).
"""

from __future__ import annotations

import importlib.util
import pathlib
import warnings
from types import ModuleType
from typing import NamedTuple

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mltpy import (
    ConditionalTransformationModel,
    ConvergenceWarning,
    Coxph,
    InteractionBasis,
    OptimizerConfig,
)
from mltpy.basis import BernsteinBasis

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_KKT_RTOL = 1e-6


def _load_data_writer() -> ModuleType:
    """Import the reference data writer (single source of the issue data)."""
    path = REF_DIR / "write_interaction_stall_data.py"
    spec = importlib.util.spec_from_file_location("_interaction_stall_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DATA_WRITER = _load_data_writer()
_SIZES: tuple[int, ...] = _DATA_WRITER.SIZES


class _Fit(NamedTuple):
    model: ConditionalTransformationModel
    convergence_warnings: list[str]


def _load_reference(n: int) -> dict[str, np.ndarray]:
    names = ["y_train", "x_train", "theta", "loglik"]
    paths = {k: REF_DIR / f"interaction_stall_n{n}_{k}.txt" for k in names}
    if not all(p.exists() for p in paths.values()):
        pytest.skip(
            f"interaction_stall_n{n}_* fixtures not generated — run "
            "python reference/write_interaction_stall_data.py, then "
            "Rscript reference/generate_reference.R"
        )
    return {k: np.loadtxt(p) for k, p in paths.items()}


def _basis() -> InteractionBasis:
    return InteractionBasis(
        y_basis=BernsteinBasis(order=6, support=(-4.0, 4.0)),
        x_basis=BernsteinBasis(order=3, support=(-1.0, 1.0)),
    )


@pytest.fixture(scope="module")
def stall_refs() -> dict[int, dict[str, np.ndarray]]:
    return {n: _load_reference(n) for n in _SIZES}


@pytest.fixture(scope="module")
def stall_fits(stall_refs: dict[int, dict[str, np.ndarray]]) -> dict[int, _Fit]:
    """The issue-#142 fit per ``n``, with every ``ConvergenceWarning`` it
    raised (recorded although ``pyproject.toml`` filters them)."""
    fits: dict[int, _Fit] = {}
    for n, ref in stall_refs.items():
        model = ConditionalTransformationModel(
            basis=_basis(), optimizer_config=OptimizerConfig(random_state=0)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(ref["y_train"], ref["x_train"].reshape(-1, 1))
        fits[n] = _Fit(
            model,
            [
                str(w.message)
                for w in caught
                if issubclass(w.category, ConvergenceWarning)
            ],
        )
    return fits


@pytest.mark.parametrize("n", _SIZES)
def test_fixture_is_issue_data(
    stall_refs: dict[int, dict[str, np.ndarray]], n: int
) -> None:
    """The committed training data are exactly the issue-#142 NumPy draw."""
    y, X = _DATA_WRITER.draw(n)
    assert_array_equal(stall_refs[n]["y_train"], y)
    assert_array_equal(stall_refs[n]["x_train"], X[:, 0])


@pytest.mark.parametrize("n", _SIZES)
def test_converges_on_first_attempt_without_warning(
    stall_fits: dict[int, _Fit], n: int
) -> None:
    result = stall_fits[n].model.result_
    assert result is not None
    assert result.converged
    assert result.n_restarts == 0
    assert stall_fits[n].convergence_warnings == []


@pytest.mark.parametrize("n", _SIZES)
def test_kkt_residual_at_floating_point_floor(
    stall_fits: dict[int, _Fit], n: int
) -> None:
    """The false stall reported KKT ≈ 114 (n = 1000) and ≈ 230 (n = 2000)."""
    result = stall_fits[n].model.result_
    assert result is not None
    assert result.kkt_residual is not None
    assert result.kkt_residual <= _KKT_RTOL * (1.0 + abs(result.log_likelihood))


@pytest.mark.parametrize("n", _SIZES)
def test_loglik_matches_r(
    stall_fits: dict[int, _Fit],
    stall_refs: dict[int, dict[str, np.ndarray]],
    n: int,
) -> None:
    result = stall_fits[n].model.result_
    assert result is not None
    assert_allclose(
        result.log_likelihood,
        float(stall_refs[n]["loglik"]),
        rtol=1e-6,
        atol=1e-10,
    )


@pytest.mark.parametrize("n", _SIZES)
def test_theta_matches_r(
    stall_fits: dict[int, _Fit],
    stall_refs: dict[int, dict[str, np.ndarray]],
    n: int,
) -> None:
    """R orders coef(fit) y-index fastest (column-major over Θ)."""
    Theta = stall_fits[n].model.Theta_
    assert Theta is not None
    p, q = Theta.shape
    theta_r = stall_refs[n]["theta"].reshape(q, p).T
    assert_allclose(Theta, theta_r, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("polish", [True, False])
def test_active_monotonicity_fit_never_falsely_converges(polish: bool) -> None:
    """Issue #135: accept the boundary optimum or warn about non-convergence.

    The original auglag fit claimed convergence at logLik ≈ -1937.55 with
    KKT ≈ 236 and no active row; trust-constr reached logLik ≈ -480.61.
    """
    rng = np.random.default_rng(111)
    n = 800
    x = rng.uniform(0.0, 1.0, size=n)
    x_s = rng.uniform(-1.5, 1.5, size=n)
    z = np.log(-np.log(rng.uniform(size=n)))
    t = 10.0 + (z / np.exp(0.4 * x_s) - x) / (2.0 + 2.0 * x)
    support = (float(t.min()) - 0.25, float(t.max()) + 0.25)
    x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))

    reference = Coxph(
        support=support,
        order=2,
        interacting=x_basis,
        optimizer_config=OptimizerConfig(solver="trust-constr", random_state=0),
    ).fit(t, X=x)
    assert reference.result_ is not None
    assert reference.result_.converged
    assert reference.Theta_ is not None
    assert abs(np.diff(reference.Theta_, axis=0).min()) < 1e-6

    model = Coxph(
        support=support,
        order=2,
        interacting=x_basis,
        optimizer_config=OptimizerConfig(random_state=0, polish=polish),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(t, X=x)

    assert model.result_ is not None
    if model.result_.converged:
        assert_allclose(
            model.result_.log_likelihood,
            reference.result_.log_likelihood,
            rtol=1e-6,
            atol=1e-10,
        )
    else:
        assert any(issubclass(w.category, ConvergenceWarning) for w in caught)
