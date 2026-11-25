import numpy as np
import pytest

from ffthompy.matvecs import DFT, VecTri


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n", [4, 5])
def test_matrix_versions(dim, n):
    N = n * np.ones(dim, dtype=int)
    ur = VecTri(name="rand", dim=2, N=N, valtype="rand")
    FN = DFT(name="FN", inverse=False, N=N, d=dim)
    FiN = DFT(name="FiN", inverse=True, N=N, d=dim)
    msg = "Operations .matrix() .vec() do not work properly!"
    Fur = FN(ur)
    val = np.linalg.norm(Fur.vec() - FN.matrix().dot(ur.vec()))
    assert val == pytest.approx(0, rel=0, abs=1e-13), msg
    val = np.linalg.norm(ur.vec() - FiN.matrix().dot(Fur.vec()))
    assert val == pytest.approx(0, rel=0, abs=1e-13), msg


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n", [5])
@pytest.mark.parametrize("i", [0, 1])
def test_projection(dim, n, i):
    N = n * np.ones(dim, dtype=int)
    uN = VecTri(name="rand", dim=dim, N=N, valtype="rand")
    msg = "dim={0}, n={1}, i={2}".format(dim, n, i)
    assert (uN == uN.project(2 * N - i).project(N)) == pytest.approx(
        0, rel=0, abs=1e-13
    ), msg
