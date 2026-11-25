import numpy as np
import pytest
from numpy.linalg import norm

from ffthompy.general.solver import linear_solver
from ffthompy.projections import scalar
from ffthompy.tensors import DFT, Operator, Tensor
from ffthompy.tensors.projection import scalar as scalar_tensor


def test_projections():
    dim = 2
    n = 5
    N = n * np.ones(dim, dtype=int)

    hG0N, hG1N, hG2N = scalar(N, Y=np.ones(dim))
    hG0Nt, hG1Nt, hG2Nt = scalar_tensor(N, Y=np.ones(dim))

    assert norm(hG0N.val - hG0Nt.val) == pytest.approx(0, rel=0, abs=1e-13)
    assert norm(hG1N.val - hG1Nt.val) == pytest.approx(0, rel=0, abs=1e-13)
    assert norm(hG2N.val - hG2Nt.val) == pytest.approx(0, rel=0, abs=1e-13)


@pytest.mark.parametrize(
    "solver", ["CG", "scipy_cg", "scipy_bicg", "richardson", "chebyshev"]
)
def test_solvers(solver):
    dim = 2
    n = 5
    N = n * np.ones(dim, dtype=int)

    _, hG1Nt, _ = scalar_tensor(N, Y=np.ones(dim))

    FN = DFT(name="FN", inverse=False, N=N)
    FiN = DFT(name="FiN", inverse=True, N=N)

    G1N = Operator(name="G1", mat=[[FiN, hG1Nt, FN]])

    A = Tensor(
        name="A",
        val=np.einsum(
            "ij,...->ij...",
            np.eye(dim),
            1.0 + 10.0 * np.random.default_rng(0).random(N),
        ),
        order=2,
        N=N,
        multype=21,
    )

    E = np.zeros((dim,) + dim * (n,))
    E[0] = 1.0  # set macroscopic loading
    E = Tensor(name="E", val=E, order=1, N=N)

    GAfun = Operator(name="GA", mat=[[G1N, A]])
    GAfun.define_operand(E)

    B = GAfun(-E)
    x0 = E.copy(name="x0")
    x0.val[:] = 0

    par = {
        "tol": 1e-10,
        "maxiter": int(1e3),
        "alpha": 0.5 * (1.0 + 10.0),
        "eigrange": [1.0, 10.0],
    }

    # reference solution
    X, _ = linear_solver(Afun=GAfun, B=B, x0=x0, par=par, solver="CG")

    x, _ = linear_solver(Afun=GAfun, ATfun=GAfun, B=B, x0=x0, par=par, solver=solver)
    assert norm(X.val - x.val) == pytest.approx(0, rel=0, abs=1e-8), solver
