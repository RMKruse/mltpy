"""Diagnose the case_06 Coxph failure.

Context
-------
``case_06_coxph_200_6`` is the last remaining genuine failure in the
R-comparison suite. pymlt converges to a *better* log-likelihood than R
(-98.9 vs -109.1) but to a structurally different ``theta``. The
gradient-verification suite (``test_gradient_verification.py``) has ruled
out an analytical-gradient bug, so the divergence must be caused by one of:

1. A different objective function (h-clipping or hazard clamping deforming
   pymlt's likelihood surface relative to R's)
2. A different local basin (SLSQP vs alabama::auglag starting from
   different initialisations landing in different minima)
3. Constraint handling differences (SLSQP vs augmented Lagrangian on the
   active monotonicity face)

This script runs the five diagnostic experiments described in the plan and
prints a labelled block for each. See ``GRADIENT_VALIDATION.md`` and
``VALIDATION.md`` for the broader context.

No production code is modified. Experiment 4 temporarily monkey-patches
``pymlt.likelihood._grad_right`` for a single ``optimize()`` call and then
restores the original.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

from pymlt import likelihood as lik
from pymlt.basis import BernsteinBasis
from pymlt.constraints import build_constraints
from pymlt.likelihood import (
    _H_CLIP,
    _neg_score,
    _shift,
    _split_theta,
    negative_log_likelihood,
)
from pymlt.optimizer import (
    OptimizerConfig,
    _make_objective,
    _scipy_options,
    optimize,
)
from pymlt.tram import Coxph
from pymlt.variables import CensoredData, CensoringType

REFERENCE_DIR = Path(__file__).resolve().parent.parent / (
    "validation/references/case_06_coxph_200_6"
)


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------


def load_reference() -> dict[str, Any]:
    """Load the case_06 R-reference arrays and metadata."""
    y = np.load(REFERENCE_DIR / "y.npy")
    status = np.load(REFERENCE_DIR / "status.npy").astype(bool)
    theta_R = np.load(REFERENCE_DIR / "theta.npy")
    loglik_R = float(np.load(REFERENCE_DIR / "loglik.npy"))
    with open(REFERENCE_DIR / "metadata.json") as f:
        meta = json.load(f)
    return {
        "y": y,
        "status": status,
        "theta_R": theta_R,
        "loglik_R": loglik_R,
        "meta": meta,
    }


def print_header(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def build_case() -> tuple[
    BernsteinBasis,
    CensoredData,
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.float64],
    float,
    dict[str, Any],
]:
    ref = load_reference()
    y = ref["y"]
    status = ref["status"]
    theta_R = ref["theta_R"]
    loglik_R = ref["loglik_R"]
    meta = ref["meta"]

    support = tuple(meta["support"])
    order = int(meta["order"])

    basis = BernsteinBasis(order=order, support=(float(support[0]), float(support[1])))
    # status=True -> event observed (exact); status=False -> censored
    censored_mask = ~status
    cd = CensoredData.right_censored(y, censored=censored_mask)

    print(
        f"loaded case_06: n={len(y)}, order={order}, "
        f"support=({support[0]}, {support[1]}), "
        f"exact={int(status.sum())}, censored={int((~status).sum())}"
    )
    print(f"  R loglik = {loglik_R:.6f}")
    print(f"  R theta  = {np.array2string(theta_R, precision=5)}")
    return basis, cd, y, status, theta_R, loglik_R, meta


def fit_pymlt(basis: BernsteinBasis, cd: CensoredData) -> NDArray[np.float64]:
    """Fit pymlt's Coxph with default settings and return theta_py."""
    model = Coxph(support=basis.support, order=basis.order)
    model.fit(cd)
    assert model.theta_ is not None
    return cast(NDArray[np.float64], model.theta_)


def nll_and_grad(
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    cd: CensoredData,
) -> tuple[float, NDArray[np.float64]]:
    result = negative_log_likelihood(
        theta,
        basis,
        cd,
        None,
        CensoringType.RIGHT,
        gradient=True,
        base_distribution="normal",
    )
    assert isinstance(result, tuple)
    nll, grad = result
    return float(nll), grad


# ---------------------------------------------------------------------------
# Experiment 1 — Smoking gun: evaluate pymlt at R's theta
# ---------------------------------------------------------------------------


def experiment_1(
    basis: BernsteinBasis,
    cd: CensoredData,
    theta_R: NDArray[np.float64],
    theta_py: NDArray[np.float64],
    loglik_R: float,
) -> None:
    print_header("Experiment 1 — Smoking gun (nll and grad at R's theta)")

    nll_R_in_py, grad_R_in_py = nll_and_grad(theta_R, basis, cd)
    nll_py, grad_py = nll_and_grad(theta_py, basis, cd)

    ll_R_in_py = -nll_R_in_py
    ll_py = -nll_py
    grad_R_norm = float(np.linalg.norm(grad_R_in_py))
    grad_py_norm = float(np.linalg.norm(grad_py))

    print(f"  R's loglik (from file) .............. {loglik_R:15.6f}")
    print(f"  pymlt's ll evaluated at R's theta ... {ll_R_in_py:15.6f}")
    print(f"  pymlt's ll evaluated at pymlt theta . {ll_py:15.6f}")
    print(f"  | delta_ll (R file vs pymlt@theta_R)| {abs(loglik_R - ll_R_in_py):15.2e}")
    print(f"  ||grad pymlt @ theta_R|| ............ {grad_R_norm:15.2e}")
    print(f"  ||grad pymlt @ theta_py|| ........... {grad_py_norm:15.2e}")

    # Diagnosis table lookup
    ll_agrees = abs(loglik_R - ll_R_in_py) <= 1e-6
    grad_small = grad_R_norm < 1e-3  # SLSQP ftol ~1e-6, allow slack

    if ll_agrees and grad_small:
        verdict = (
            "Both implementations agree theta_R is stationary — "
            "different BASIN (cause #2, optimizer initialisation)."
        )
    elif ll_agrees and not grad_small:
        verdict = (
            "Same ll function, but theta_R is not stationary for pymlt — "
            "cause #2 or #3 (initialisation or constraint handling)."
        )
    elif (not ll_agrees) and (not grad_small):
        verdict = (
            "pymlt's objective differs from R's at theta_R — cause #1 "
            "(OBJECTIVE mismatch, clipping/clamping active)."
        )
    else:
        verdict = (
            "pymlt finds theta_R stationary on a DIFFERENT objective — "
            "strong evidence for cause #1 (objective mismatch)."
        )
    print(f"\n  Diagnosis: {verdict}")


# ---------------------------------------------------------------------------
# Experiment 2 — Check h-clipping activation
# ---------------------------------------------------------------------------


def _compute_h(
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    cd: CensoredData,
) -> dict[str, NDArray[np.float64]]:
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, None)

    result: dict[str, NDArray[np.float64]] = {}
    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        B_e = basis.evaluate(y_e)
        h_e_raw = _shift(B_e @ theta_b, None, beta)
        result["h_exact"] = h_e_raw

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        B_c = basis.evaluate(y_c)
        h_c_raw = _shift(B_c @ theta_b, None, beta)
        result["h_censored"] = h_c_raw
    return result


def _print_h_stats(
    label: str, h_map: dict[str, NDArray[np.float64]], threshold: float = 29.0
) -> None:
    print(f"  {label}")
    for key, h in h_map.items():
        n_clipped = int((np.abs(h) >= threshold).sum())
        print(
            f"    {key:12s}: min={h.min():10.4f}  max={h.max():10.4f}  "
            f"|h|>={threshold:<4.0f}: {n_clipped}/{len(h)}"
        )


def experiment_2(
    basis: BernsteinBasis,
    cd: CensoredData,
    theta_R: NDArray[np.float64],
    theta_py: NDArray[np.float64],
) -> None:
    print_header(f"Experiment 2 — h-clipping activation (_H_CLIP = {_H_CLIP})")

    h_R = _compute_h(theta_R, basis, cd)
    h_py = _compute_h(theta_py, basis, cd)
    _print_h_stats("At theta_R:", h_R)
    _print_h_stats("At theta_py:", h_py)

    clipped_R = sum(int((np.abs(h) >= _H_CLIP).sum()) for h in h_R.values())
    clipped_py = sum(int((np.abs(h) >= _H_CLIP).sum()) for h in h_py.values())
    if clipped_R == 0 and clipped_py == 0:
        verdict = "h-clipping inactive at both optima — rules out cause #1a."
    elif clipped_py > 0:
        verdict = (
            f"h-clipping ACTIVE at theta_py ({clipped_py} obs); pymlt is "
            "optimising a different function where clipping binds."
        )
    else:
        verdict = (
            f"h-clipping active at theta_R ({clipped_R} obs) but not at "
            "theta_py; would bias pymlt's ll upward at R's solution."
        )
    print(f"\n  Diagnosis: {verdict}")


# ---------------------------------------------------------------------------
# Experiment 3 — Hazard clamping activation
# ---------------------------------------------------------------------------


def _hazard_at(
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    cd: CensoredData,
) -> NDArray[np.float64]:
    """Compute raw hazard f(h)/S(h) for censored observations (no clamp)."""
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, None)

    mask_c = cd.is_right_censored_mask
    y_c = cd.lower[mask_c]
    B_c = basis.evaluate(y_c)
    h_c = np.clip(_shift(B_c @ theta_b, None, beta), -_H_CLIP, _H_CLIP)
    return np.exp(norm.logpdf(h_c) - norm.logsf(h_c))


def experiment_3(
    basis: BernsteinBasis,
    cd: CensoredData,
    theta_R: NDArray[np.float64],
    theta_py: NDArray[np.float64],
) -> None:
    print_header(f"Experiment 3 — hazard clamping (threshold = {_H_CLIP})")

    for label, theta in (("theta_R", theta_R), ("theta_py", theta_py)):
        haz = _hazard_at(theta, basis, cd)
        n_clamped = int((haz > _H_CLIP).sum())
        print(
            f"  at {label:8s}: hazard min={haz.min():10.4f}  "
            f"max={haz.max():12.4f}  mean={haz.mean():10.4f}  "
            f"> {_H_CLIP:.0f}: {n_clamped}/{len(haz)}"
        )

    haz_R = _hazard_at(theta_R, basis, cd)
    haz_py = _hazard_at(theta_py, basis, cd)
    active_R = int((haz_R > _H_CLIP).sum())
    active_py = int((haz_py > _H_CLIP).sum())
    if active_R == 0 and active_py == 0:
        verdict = "Hazard clamp inactive at both optima — rules out cause #1b."
    elif active_py > 0:
        verdict = (
            f"Hazard clamp ACTIVE at theta_py ({active_py} obs) — pymlt's "
            "gradient is deformed there; grad/ll mismatch within pymlt."
        )
    else:
        verdict = (
            f"Hazard clamp active at theta_R ({active_R} obs) but not at "
            "theta_py — pymlt's gradient would push AWAY from R's solution."
        )
    print(f"\n  Diagnosis: {verdict}")


# ---------------------------------------------------------------------------
# Experiment 3b — Decompose the log-likelihood term by term at theta_R
# ---------------------------------------------------------------------------


def _decompose_ll(
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    cd: CensoredData,
) -> dict[str, float]:
    """Return the three additive pieces of the right-censored log-likelihood.

    Mirrors ``_ll_right`` exactly, but keeps the pieces separate so we can
    see which term is responsible for a discrepancy.
    """
    p = basis.order + 1
    theta_b, beta = _split_theta(theta, p, None)

    mask_e = cd.is_exact_mask
    y_e = cd.exact[mask_e]
    B_e = basis.evaluate(y_e)
    D_e = basis.derivative(y_e, order=1)
    h_e = np.clip(_shift(B_e @ theta_b, None, beta), -_H_CLIP, _H_CLIP)
    hp_e = D_e @ theta_b

    mask_c = cd.is_right_censored_mask
    y_c = cd.lower[mask_c]
    B_c = basis.evaluate(y_c)
    h_c = np.clip(_shift(B_c @ theta_b, None, beta), -_H_CLIP, _H_CLIP)

    with np.errstate(invalid="ignore", divide="ignore"):
        term_logpdf = float(np.sum(norm.logpdf(h_e)))
        log_hp = np.log(hp_e)
        term_log_hp = float(np.sum(log_hp))
        term_logsf = float(np.sum(norm.logsf(h_c)))

    return {
        "logpdf_exact": term_logpdf,
        "log_hprime_exact": term_log_hp,
        "logsf_censored": term_logsf,
        "hp_min": float(hp_e.min()),
        "hp_max": float(hp_e.max()),
        "n_hp_nonpos": int((hp_e <= 0).sum()),
        "n_hp_lt_1e6": int((hp_e < 1e-6).sum()),
        "n_log_hp_nan": int(np.isnan(log_hp).sum()),
    }


def experiment_3b(
    basis: BernsteinBasis,
    cd: CensoredData,
    theta_R: NDArray[np.float64],
    theta_py: NDArray[np.float64],
    loglik_R: float,
) -> None:
    print_header("Experiment 3b — log-likelihood decomposition at theta_R/theta_py")

    dec_R = _decompose_ll(theta_R, basis, cd)
    dec_py = _decompose_ll(theta_py, basis, cd)

    print(
        f"  {'term':<22}  {'at theta_R':>15}  {'at theta_py':>15}"
    )
    for k in ("logpdf_exact", "log_hprime_exact", "logsf_censored"):
        print(f"  {k:<22}  {dec_R[k]:15.6f}  {dec_py[k]:15.6f}")
    print(
        f"  {'TOTAL ll':<22}  "
        f"{sum(dec_R[k] for k in ('logpdf_exact', 'log_hprime_exact', 'logsf_censored')):15.6f}  "
        f"{sum(dec_py[k] for k in ('logpdf_exact', 'log_hprime_exact', 'logsf_censored')):15.6f}"
    )
    print(f"  R's reported ll ............. {loglik_R:15.6f}")
    print()
    print("  h' = D @ theta_b at exact observations:")
    for label, dec in (("theta_R", dec_R), ("theta_py", dec_py)):
        print(
            f"    {label}: min={dec['hp_min']:.3e}  max={dec['hp_max']:.3e}  "
            f"n(h'<=0)={dec['n_hp_nonpos']}  n(h'<1e-6)={dec['n_hp_lt_1e6']}  "
            f"n(log nan)={dec['n_log_hp_nan']}"
        )

    # Also print the raw forward differences of theta_R's Bernstein coefs
    p = basis.order + 1
    theta_b_R, _ = _split_theta(theta_R, p, None)
    diffs = np.diff(theta_b_R)
    print("\n  Forward differences of theta_R's Bernstein coefficients:")
    print(f"    {np.array2string(diffs, precision=3, formatter={'float_kind': lambda x: f'{x:.3e}'})}")

    diff_ll = loglik_R - (
        dec_R["logpdf_exact"] + dec_R["log_hprime_exact"] + dec_R["logsf_censored"]
    )
    if abs(diff_ll - (dec_R["log_hprime_exact"] - dec_R["log_hprime_exact"])) < 1e-6:
        pass  # placeholder
    # Determine which term dominates the discrepancy
    delta_pdf = dec_R["logpdf_exact"] - dec_py["logpdf_exact"]
    delta_hp = dec_R["log_hprime_exact"] - dec_py["log_hprime_exact"]
    delta_sf = dec_R["logsf_censored"] - dec_py["logsf_censored"]
    print(f"\n  Per-term delta (theta_R − theta_py):")
    print(f"    delta logpdf(h)    = {delta_pdf:+.4f}")
    print(f"    delta log(h')      = {delta_hp:+.4f}")
    print(f"    delta logsf(h)     = {delta_sf:+.4f}")

    if abs(delta_hp) > max(abs(delta_pdf), abs(delta_sf)) * 5:
        verdict = (
            "log(h') term DOMINATES the discrepancy — pymlt is penalising "
            "theta_R because near-collapsed middle coefficients drive h'(y) "
            "toward zero at exact observations. This is the true cause."
        )
    else:
        verdict = (
            "No single term dominates — discrepancy is spread across the "
            "likelihood decomposition."
        )
    print(f"\n  Diagnosis: {verdict}")


# ---------------------------------------------------------------------------
# Experiment 4 — Refit with hazard clamp removed
# ---------------------------------------------------------------------------


def _grad_right_no_clamp(
    cd: CensoredData,
    theta: NDArray[np.float64],
    basis: BernsteinBasis,
    X: NDArray[np.float64] | None,
    dist: Any = norm,
) -> NDArray[np.float64]:
    """Drop-in replacement for ``_grad_right`` with the hazard clamp removed.

    Identical to the production implementation except the
    ``hazard = np.minimum(hazard, _H_CLIP)`` line is removed.
    """
    p = basis.order + 1
    q = X.shape[1] if X is not None else 0
    theta_b, beta = _split_theta(theta, p, X)
    grad = np.zeros(p + q)

    mask_e = cd.is_exact_mask
    if mask_e.any():
        y_e = cd.exact[mask_e]
        X_e = X[mask_e] if X is not None else None
        B_e = basis.evaluate(y_e)
        D_e = basis.derivative(y_e, order=1)
        h_e = np.clip(_shift(B_e @ theta_b, X_e, beta), -_H_CLIP, _H_CLIP)
        hp_e = D_e @ theta_b
        ns = _neg_score(h_e, dist)
        grad[:p] += B_e.T @ ns - D_e.T @ (1.0 / hp_e)
        if X_e is not None:
            grad[p:] += X_e.T @ ns

    mask_c = cd.is_right_censored_mask
    if mask_c.any():
        y_c = cd.lower[mask_c]
        X_c = X[mask_c] if X is not None else None
        B_c = basis.evaluate(y_c)
        h_c = np.clip(_shift(B_c @ theta_b, X_c, beta), -_H_CLIP, _H_CLIP)
        hazard = np.exp(dist.logpdf(h_c) - dist.logsf(h_c))
        # NOTE: no np.minimum(hazard, _H_CLIP) here — the purpose of this
        # experiment is to see whether the clamp is responsible for the
        # divergence from R.
        grad[:p] += B_c.T @ hazard
        if X_c is not None:
            grad[p:] += X_c.T @ hazard

    return cast(NDArray[np.float64], grad)


def experiment_4(
    basis: BernsteinBasis,
    cd: CensoredData,
    theta_R: NDArray[np.float64],
    theta_py: NDArray[np.float64],
    loglik_R: float,
) -> None:
    print_header("Experiment 4 — refit with hazard clamp removed")

    original = lik._grad_right
    lik._grad_right = _grad_right_no_clamp  # type: ignore[assignment]
    try:
        result = optimize(
            basis=basis,
            y=cd,
            X=None,
            censoring=CensoringType.RIGHT,
            config=OptimizerConfig(),
            base_distribution="normal",
        )
    finally:
        lik._grad_right = original  # type: ignore[assignment]

    theta_nc = result.theta
    ll_nc = result.log_likelihood

    # Recompute pymlt's ll at theta_py for a same-code-path comparison
    nll_py, _ = nll_and_grad(theta_py, basis, cd)
    ll_py = -nll_py

    print(f"  pymlt (clamp on)  theta_py loglik ... {ll_py:15.6f}")
    print(f"  pymlt (clamp off) theta_nc loglik ... {ll_nc:15.6f}")
    print(f"  R                 theta_R  loglik ... {loglik_R:15.6f}")
    print(f"  converged: {result.converged}  msg: {result.solver_message}")
    print()
    print(f"  theta_py = {np.array2string(theta_py, precision=5)}")
    print(f"  theta_nc = {np.array2string(theta_nc, precision=5)}")
    print(f"  theta_R  = {np.array2string(theta_R,  precision=5)}")

    dtheta_R = float(np.max(np.abs(theta_nc - theta_R)))
    dtheta_py = float(np.max(np.abs(theta_nc - theta_py)))
    dll_R = abs(ll_nc - loglik_R)
    print(f"\n  max|theta_nc - theta_R|  = {dtheta_R:.4e}")
    print(f"  max|theta_nc - theta_py| = {dtheta_py:.4e}")
    print(f"  |ll_nc - ll_R|           = {dll_R:.4e}")

    if dll_R < 1e-2 and dtheta_R < 1e-1:
        verdict = "Clamp removed -> matches R. Cause = hazard clamp (#1b)."
    elif dtheta_py < 1e-3:
        verdict = "Clamp removed -> unchanged. Clamp was inactive; not the cause."
    else:
        verdict = (
            "Clamp removed -> new optimum, neither R nor previous pymlt. "
            "Clamp contributes but does not fully explain divergence."
        )
    print(f"\n  Diagnosis: {verdict}")


# ---------------------------------------------------------------------------
# Experiment 5 — Initialise pymlt at R's theta and re-optimise
# ---------------------------------------------------------------------------


def experiment_5(
    basis: BernsteinBasis,
    cd: CensoredData,
    theta_R: NDArray[np.float64],
    theta_py: NDArray[np.float64],
    loglik_R: float,
) -> None:
    print_header("Experiment 5 — pymlt initialised at R's theta")

    config = OptimizerConfig()
    n_params = basis.order + 1
    constraints = build_constraints(
        n_params, solver=config.solver, total_params=n_params
    )
    obj = _make_objective(
        basis, cd, None, CensoringType.RIGHT, config.use_gradient,
        base_distribution="normal",
    )
    options = _scipy_options(config)

    res = minimize(
        obj,
        theta_R.copy(),
        method=config.solver,
        jac=True,
        constraints=constraints,
        options=options,
    )
    theta_from_R_init = res.x
    ll_from_R_init = float(-res.fun)

    dtheta_from_R = float(np.max(np.abs(theta_from_R_init - theta_R)))
    dtheta_to_py = float(np.max(np.abs(theta_from_R_init - theta_py)))

    print(f"  converged: {bool(res.success)}  msg: {res.message}")
    print(f"  starting  ll (at theta_R) ......... {-float(obj(theta_R)[0]):15.6f}")
    print(f"  R         ll (from file) .......... {loglik_R:15.6f}")
    print(f"  optimiser ll (from theta_R init) .. {ll_from_R_init:15.6f}")
    print(f"  max|theta_final - theta_R|  = {dtheta_from_R:.4e}")
    print(f"  max|theta_final - theta_py| = {dtheta_to_py:.4e}")

    if dtheta_from_R < 1e-3:
        verdict = (
            "Optimiser stayed at R's theta — both implementations agree "
            "R's solution is locally optimal. Cause = basin (#2)."
        )
    elif dtheta_to_py < 1e-3:
        verdict = (
            "Optimiser moved from R's theta back to pymlt's solution — "
            "R's theta is NOT a local optimum of pymlt's objective. "
            "Cause = objective (#1) or constraints (#3)."
        )
    else:
        verdict = (
            "Optimiser moved from R's theta to a third point — "
            "points at a more rugged likelihood surface than expected."
        )
    print(f"\n  Diagnosis: {verdict}")


# ---------------------------------------------------------------------------
# Experiment 6 — Test alternative base distributions at theta_R
# ---------------------------------------------------------------------------


def experiment_6(
    basis: BernsteinBasis,
    cd: CensoredData,
    theta_R: NDArray[np.float64],
    loglik_R: float,
) -> None:
    """Confirm that tram::Coxph uses the MinExtrVal distribution, not Normal.

    Cox proportional hazards has the structural relationship
    ``log[-log S(t)] = h(t) + x'beta``. This is equivalent to assuming
    ``h(T) ~ MinExtrVal`` (the standard minimum extreme value / reversed
    Gumbel distribution), not ``~ Normal``. If pymlt's ``Coxph`` class
    (which hardcodes ``base_distribution="normal"``) is wrong about this,
    then evaluating the ll at theta_R with MinExtrVal should reproduce R's
    reported value.
    """
    print_header("Experiment 6 — alternative base distributions at theta_R")

    p = basis.order + 1
    theta_b, _ = _split_theta(theta_R, p, None)

    mask_e = cd.is_exact_mask
    y_e = cd.exact[mask_e]
    B_e = basis.evaluate(y_e)
    D_e = basis.derivative(y_e, order=1)
    h_e = np.clip(B_e @ theta_b, -_H_CLIP, _H_CLIP)
    hp_e = D_e @ theta_b

    mask_c = cd.is_right_censored_mask
    y_c = cd.lower[mask_c]
    B_c = basis.evaluate(y_c)
    h_c = np.clip(B_c @ theta_b, -_H_CLIP, _H_CLIP)

    # Normal (pymlt's current assumption)
    ll_normal = float(
        np.sum(norm.logpdf(h_e))
        + np.sum(np.log(hp_e))
        + np.sum(norm.logsf(h_c))
    )

    # Minimum extreme value (Gumbel-min) — what tram::Coxph actually uses
    # logpdf(x) = x - exp(x);  logsf(x) = -exp(x)
    log_pdf_mev = h_e - np.exp(h_e)
    log_sf_mev = -np.exp(h_c)
    ll_mev = float(np.sum(log_pdf_mev) + np.sum(np.log(hp_e)) + np.sum(log_sf_mev))

    print(f"  ll(theta_R | Normal base) ........ {ll_normal:15.6f}")
    print(f"  ll(theta_R | MinExtrVal base) .... {ll_mev:15.6f}")
    print(f"  R's reported ll .................. {loglik_R:15.6f}")
    print()
    print(f"  delta vs Normal     = {loglik_R - ll_normal:+.6f}")
    print(f"  delta vs MinExtrVal = {loglik_R - ll_mev:+.6e}")

    if abs(loglik_R - ll_mev) < 1e-6:
        verdict = (
            "CONFIRMED: tram::Coxph uses the MinExtrVal (Gumbel-min) "
            "distribution, not Normal. pymlt's Coxph class (hardcoded to "
            "base_distribution='normal') is solving a different model. The "
            "-98.9 vs -109.1 ll gap is a model mismatch, not a bug in the "
            "optimizer or gradient."
        )
    else:
        verdict = "MinExtrVal does not reproduce R's ll — investigate further."
    print(f"\n  Diagnosis: {verdict}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    basis, cd, _y, _status, theta_R, loglik_R, _meta = build_case()
    theta_py = fit_pymlt(basis, cd)
    nll_py, _ = nll_and_grad(theta_py, basis, cd)
    print(f"  pymlt theta = {np.array2string(theta_py, precision=5)}")
    print(f"  pymlt loglik = {-nll_py:.6f}")

    experiment_1(basis, cd, theta_R, theta_py, loglik_R)
    experiment_2(basis, cd, theta_R, theta_py)
    experiment_3(basis, cd, theta_R, theta_py)
    experiment_3b(basis, cd, theta_R, theta_py, loglik_R)
    experiment_4(basis, cd, theta_R, theta_py, loglik_R)
    experiment_5(basis, cd, theta_R, theta_py, loglik_R)
    experiment_6(basis, cd, theta_R, loglik_R)


if __name__ == "__main__":
    main()
