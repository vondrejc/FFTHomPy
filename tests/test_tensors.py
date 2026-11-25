import numpy as np
import pytest

from ffthompy.tensors import Tensor

fft_forms = ["r", 0, "c"]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n", [4, 5])
@pytest.mark.parametrize("fft_form", fft_forms)
def test_grid_projection(dim, n, fft_form):

    msg = "Tensors with: dim={}, n={}, fft_form={}".format(dim, n, fft_form)

    N = dim * (n,)
    M = tuple(2 * np.array(N))

    u = Tensor(name="test", shape=(), N=N, Fourier=False, fft_form=fft_form)
    u.randomize()
    Fu = u.fourier(copy=True)
    FuM = Fu.project(M)
    uM = FuM.fourier()

    if n % 2 == 0:
        assert u.norm() >= FuM.norm(), msg
        assert u.norm(componentwise=True) >= FuM.norm(componentwise=True), msg
        assert u.norm() >= uM.norm(), msg
        assert u.norm(componentwise=True) >= uM.norm(componentwise=True), msg
    else:
        assert FuM.norm() == pytest.approx(u.norm(), rel=0, abs=5e-8), msg
        assert FuM.norm(componentwise=True) == pytest.approx(
            u.norm(componentwise=True), rel=0, abs=5e-8
        ), msg
        assert uM.norm() == pytest.approx(u.norm(), rel=0, abs=5e-8), msg
        assert uM.norm(componentwise=True) == pytest.approx(
            u.norm(componentwise=True), rel=0, abs=5e-8
        ), msg

    assert u.mean() - FuM.mean() == pytest.approx(0, rel=0, abs=5e-8), msg
    assert uM.mean() == pytest.approx(u.mean(), rel=0, abs=5e-8), msg

    # testing that that interpolation on double grid have exactly the same values
    slc = tuple(
        u.order
        * [
            slice(None),
        ]
        + [slice(0, M[i], 2) for i in range(dim)]
    )
    assert np.linalg.norm(u.val - uM.val[slc]) == pytest.approx(0, rel=0, abs=5e-8), msg
