import itertools

import numpy as np
import pytest
from numpy.linalg import norm

import ffthompy.projections as proj
from ffthompy.tensors import (
    DFT,
    Operator,
    Tensor,
    div,
    grad,
    grad_div_tensor,
    potential,
    symgrad,
)
from ffthompy.tensors.projection import (
    elasticity_large_deformation,
    elasticity_small_strain,
    scalar,
)

fft_forms = [0, "r", "c"]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("fft_form", fft_forms)
def test_operators(dim, fft_form):
    N = 5 * np.ones(dim, dtype=int)
    F = DFT(N=N, inverse=False, fft_form=fft_form)
    iF = DFT(N=N, inverse=True, fft_form=fft_form)

    # Fourier transform

    print(F)  # checking representation

    u = Tensor(name="u", shape=(), N=N, Fourier=False, fft_form=fft_form).randomize()
    Fu = F(u)
    u2 = iF(Fu)
    assert (u == u2)[1] == pytest.approx(0, rel=0, abs=1e-13), "Fourier transform"

    fft_formsC = fft_forms.copy()
    fft_formsC.remove(fft_form)
    for fft_formc in fft_formsC:
        FuC = Fu.set_fft_form(fft_formc, copy=True)
        Fu2 = FuC.set_fft_form(fft_form, copy=True)
        msg = "Tensor.set_fft_form()"
        assert Fu.norm() - FuC.norm() == pytest.approx(0, rel=0, abs=1e-13), msg
        assert norm(Fu.mean() - FuC.mean()) == pytest.approx(0, rel=0, abs=1e-13), msg
        assert (Fu == Fu2)[1] == pytest.approx(0, rel=0, abs=1e-13), msg

    # scalar problem
    u = Tensor(name="u", shape=(1,), N=N, Fourier=False, fft_form=fft_form).randomize()
    u.val -= np.mean(u.val)
    Fu = F(u)
    Fu2 = potential(grad(Fu))
    assert (Fu == Fu2)[1] == pytest.approx(0, rel=0, abs=1e-13), (
        "scalar problem, Fourier=True"
    )

    u2 = potential(grad(u))
    assert (u == u2)[1] == pytest.approx(0, rel=0, abs=1e-13), (
        "scalar problem, Fourier=False"
    )

    hG, hD = grad_div_tensor(N, fft_form=fft_form)
    assert (hD(hG(Fu)) == div(grad(Fu)))[1] == pytest.approx(0, rel=0, abs=1e-13), (
        "scalar problem, Fourier=True"
    )

    # vectorial problem
    u = Tensor(name="u", shape=(dim,), N=N, Fourier=False, fft_form=fft_form)
    u.randomize()
    u.add_mean(-u.mean())

    Fu = F(u)
    Fu2 = potential(grad(Fu))
    assert (Fu == Fu2)[1] == pytest.approx(0, rel=0, abs=1e-13), (
        "vectorial problem, Fourier=True"
    )

    u2 = potential(grad(u))
    assert (u == u2)[1] == pytest.approx(0, rel=0, abs=1e-13), (
        "vectorial problem, Fourier=False"
    )

    # 'vectorial problem - symetric gradient
    Fu2 = potential(symgrad(Fu), small_strain=True)
    assert (Fu == Fu2)[1] == pytest.approx(0, rel=0, abs=1e-13), (
        "vectorial - sym, Fourier=True"
    )

    u2 = potential(symgrad(u), small_strain=True)
    assert (u == u2)[1] == pytest.approx(0, rel=0, abs=1e-13), (
        "vectorial - sym, Fourier=False"
    )

    # matrix version of DFT
    u = Tensor(name="u", shape=(1,), N=N, Fourier=False, fft_form="c").randomize()
    F = DFT(N=N, inverse=False, fft_form="c")
    Fu = F(u)
    dft = F.matrix(shape=u.shape)
    Fu2 = dft.dot(u.val.ravel())
    assert norm(Fu.val.ravel() - Fu2) == pytest.approx(0, rel=0, abs=1e-13)


@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("fft_form", fft_forms)
def test_compatibility(dim, fft_form):
    N = 5 * np.ones(dim, dtype=int)
    F = DFT(inverse=False, N=N, fft_form=fft_form)
    iF = DFT(inverse=True, N=N, fft_form=fft_form)

    # scalar problem
    _, G1l, G2l = scalar(N, Y=np.ones(dim), fft_form=fft_form)
    P1 = Operator(name="P1", mat=[[iF, G1l, F]])
    P2 = Operator(name="P2", mat=[[iF, G2l, F]])
    u = Tensor(name="u", shape=(1,), N=N, Fourier=False, fft_form=fft_form)
    u.randomize()

    grad_u = grad(u)
    assert (P1(grad_u) - grad_u).norm() == pytest.approx(0, rel=0, abs=1e-13)
    assert P2(grad_u).norm() == pytest.approx(0, rel=0, abs=1e-13)

    e = P1(
        Tensor(
            name="u", shape=(dim,), N=N, Fourier=False, fft_form=fft_form
        ).randomize()
    )
    e2 = grad(potential(e))
    assert (e - e2).norm() == pytest.approx(0, rel=0, abs=1e-13)

    # vectorial problem
    hG = elasticity_large_deformation(N=N, Y=np.ones(dim), fft_form=fft_form)
    P1 = Operator(name="P", mat=[[iF, hG, F]])
    u = Tensor(name="u", shape=(dim,), N=N, Fourier=False, fft_form=fft_form)
    u.randomize()
    grad_u = grad(u)
    val = (P1(grad_u) - grad_u).norm()
    assert val == pytest.approx(0, rel=0, abs=1e-13)

    e = Tensor(name="F", shape=(dim, dim), N=N, Fourier=False, fft_form=fft_form)
    e = P1(e.randomize())
    e2 = grad(potential(e))
    assert (e - e2).norm() == pytest.approx(0, rel=0, abs=1e-13)

    # transpose
    P1TT = P1.transpose().transpose()
    assert (P1(grad_u) == P1TT(grad_u))[0]

    assert (hG == (hG.transpose_left().transpose_left()))[0]
    assert (hG == (hG.transpose_right().transpose_right()))[0]

    # vectorial problem - symetric gradient
    hG = elasticity_small_strain(N=N, Y=np.ones(dim), fft_form=fft_form)
    P1 = Operator(name="P", mat=[[iF, hG, F]])
    u = Tensor(name="u", shape=(dim,), N=N, Fourier=False, fft_form=fft_form)
    u.randomize()
    grad_u = symgrad(u)
    val = (P1(grad_u) - grad_u).norm()
    assert val == pytest.approx(0, rel=0, abs=1e-13)

    e = Tensor(name="strain", shape=(dim, dim), N=N, Fourier=False, fft_form=fft_form)
    e = P1(e.randomize())
    e2 = symgrad(potential(e, small_strain=True))
    assert (e - e2).norm() == pytest.approx(0, rel=0, abs=1e-13)

    # means
    Fu = F(u)
    E = np.random.random(u.shape)
    u.set_mean(E)
    assert norm(u.mean() - E) == pytest.approx(0, rel=0, abs=1e-13)
    Fu.set_mean(E)
    assert norm(Fu.mean() - E) == pytest.approx(0, rel=0, abs=1e-13)

    # __repr__

    print(P1)
    print(u)

    assert (P1 == P1.transpose()) == pytest.approx(0, rel=0, abs=1e-13)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("fft_form", fft_forms)
def test_projections(dim, fft_form):

    N = dim * (5,)
    Y = np.ones(dim)

    # scalar projections
    projections = proj.scalar(N, Y, tensor=True, fft_form=fft_form)

    for P, Q in itertools.product(projections, repeat=2):
        if (P == Q)[0]:
            assert (P * P - P).norm() == pytest.approx(
                0, rel=0, abs=1e-13
            )  # idempotent
        else:
            assert (P * Q).norm() == pytest.approx(0, rel=0, abs=1e-13)  # orthgonality

    # linear elasticity projections
    projections = proj.elasticity(N, Y, tensor=True, fft_form=fft_form)

    # checking idempotent property
    for P, Q in itertools.product(projections, repeat=2):
        if (P == Q)[0]:
            assert (P * P - P).norm() == pytest.approx(
                0, rel=0, abs=1e-13
            )  # idempotent
        else:
            assert (P * Q).norm() == pytest.approx(0, rel=0, abs=1e-13)  # orthgonality
