"""Small checks at the agreed public likelihood.hessian boundary."""

import json
import warnings

import measure
import numpy as np

from mltpy import BernsteinBasis, InteractionBasis

likelihood = measure.likelihood


def main() -> None:
    basis = InteractionBasis(
        y_basis=BernsteinBasis(order=1, support=(-1.0, 1.0)),
        x_basis=BernsteinBasis(order=0, support=(-1.0, 1.0)),
    )
    common = dict(basis=basis, y=np.array([0.0]), X=np.array([[0.0]]))
    original = likelihood._hessian_interaction_fd
    # At y=0, A=(1/2,1/2), D=(-1/2,1/2), h'=1.
    # Normal curvature plus the log-Jacobian curvature gives H=I/2.
    expected = np.array([[0.5, 0.0], [0.0, 0.5]])
    old = likelihood.hessian(np.array([-1.0, 1.0]), **common)
    scalar_error = float(np.max(np.abs(old - expected)))
    try:
        likelihood._hessian_interaction_fd = measure.analytic
        np.testing.assert_allclose(
            likelihood.hessian(np.array([-1.0, 1.0]), **common),
            expected,
            rtol=0,
            atol=1e-12,
        )
        # g=1, f=1, offset=1/2, w=2, s=1: H_tt=I,
        # H_tg=(5/4,5/4)', H_gg=5/4. Offset is absent from dg/dgamma.
        np.testing.assert_allclose(
            likelihood.hessian(
                np.array([0.0, 2.0, 0.0]),
                **common,
                weights=np.array([2.0]),
                offset=np.array([0.5]),
                scaling=np.array([[1.0]]),
            ),
            np.array([[1.0, 0.0, 1.25], [0.0, 1.0, 1.25], [1.25, 1.25, 1.25]]),
            rtol=0,
            atol=1e-12,
        )
        # These are excluded from the smooth prototype, never silently accepted.
        for theta, dist in (
            (np.array([30.0, 32.0]), "normal"),
            (np.array([-1.0, 1.0]), "laplace"),
        ):
            try:
                likelihood.hessian(theta, **common, base_distribution=dist)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Prototype unexpectedly accepted {dist}: {theta}")
        r_results = json.loads((measure.HERE / "results/r_reference.json").read_text())[
            "cases"
        ]
        cases = measure.baseline.build_cases()
        for name in r_results:
            case = cases[name]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = case.make()
            inp = measure.baseline._fit_inputs(case, model)
            theta = np.loadtxt(measure.HERE / "results" / f"{name}_theta.txt")
            H_R = np.loadtxt(measure.HERE / "results" / f"{name}_R_H.txt")
            np.testing.assert_allclose(
                likelihood.hessian(theta, **inp), H_R, rtol=1e-6, atol=1e-10
            )
    finally:
        likelihood._hessian_interaction_fd = original
    print(
        "Worked examples and three same-theta R checks passed; "
        f"scalar-FD worked-example error: {scalar_error:.3g}"
    )


if __name__ == "__main__":
    main()
