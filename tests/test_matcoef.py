import numpy as np
import pytest
from numpy.linalg import norm

from ffthompy.mechanics.matcoef import ElasticTensor as ET


def get_rand_sym(dim, ndim):
    A = np.random.random(ndim * (dim,))
    if ndim == 2:
        A = A + A.T
    else:
        A = 0.5 * (A + np.einsum("ijkl->klij", A))
        A = 0.5 * (A + np.einsum("ijkl->ijlk", A))
        A = 0.5 * (A + np.einsum("ijkl->jikl", A))
    return A


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("ndim", [2, 4])
def test_mandel(dim, ndim):
    A = get_rand_sym(dim=dim, ndim=ndim)

    Am = ET.create_mandel(A, ndim=None)
    assert norm(Am - Am.T) == pytest.approx(0, rel=0, abs=5e-8)
    A2 = ET.dispose_mandel(Am, ndim=None)
    Am2 = ET.create_mandel(A2, ndim=None)
    assert norm(Am - Am2) == pytest.approx(0, rel=0, abs=1e-14), (
        "mandel in dim={0} and ndim={1}".format(dim, ndim)
    )
    assert norm(A - A2) == pytest.approx(0, rel=0, abs=1e-14), (
        "mandel in dim={0} and ndim={1}".format(dim, ndim)
    )


def test_plane():
    A = get_rand_sym(dim=3, ndim=4)
    Am = ET.create_mandel(A)
    Amplane = ET.get_plane_in_engineering(Am).squeeze()

    Aplane = ET.get_plane_in_tensor(A)
    Aplanem = ET.create_mandel(Aplane).squeeze()
    assert norm(Aplanem - Amplane) == pytest.approx(0, rel=0, abs=5e-8)
