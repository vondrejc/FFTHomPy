import logging
import timeit

import numpy as np
import pytest
from numpy.linalg import norm

from examples.lowRankTensorApproximations.setting import (
    get_default_parameters,
    get_material_coef,
)
from ffthompy import Timer
from ffthompy.tensors import Tensor
from ffthompy.tensorsLowRank.homogenisation import (
    homog_Ga_full_potential,
    homog_Ga_sparse,
    homog_GaNi_full_potential,
    homog_GaNi_sparse,
)
from ffthompy.tensorsLowRank.objects import CanoTensor, SparseTensor

logger = logging.getLogger(__name__)


@pytest.fixture
def tensors_2d():
    tensor = np.random.rand(5, 10)
    return tensor, np.random.rand(*tensor.shape)


@pytest.fixture
def tensors_3d():
    tensor = np.random.rand(10, 5, 20)
    return tensor, np.random.rand(*tensor.shape)


@pytest.mark.parametrize(
    ("dim", "material", "kind"), [(2, 0, 0), (2, 3, 1), (3, 1, 1), (3, 0, 2)]
)
def test_sparse_solver(dim, material, kind):

    tic = Timer("homogenisation")
    N = 5
    logger.debug('dim=%s, material=%s, kind=%s', dim, material, kind)
    pars, pars_sparse = get_default_parameters(dim, N, material, kind)
    pars_sparse.debug = True
    pars_sparse.solver.update(dict(rank=5, maxiter=10))

    Aga, Agani, Agas, Aganis = get_material_coef(material, pars, pars_sparse)

    if material in [0, 3]:
        resP_Ga = homog_Ga_full_potential(Aga, pars)
        resS_Ga = homog_Ga_sparse(Agas, pars_sparse)

    if material in [1, 2, 4]:
        resP_GaNi = homog_GaNi_full_potential(Agani, Aga, pars)
        resS_GaNi = homog_GaNi_sparse(Aganis, Agas, pars_sparse)

    if material in [0, 3]:
        assert 0 == pytest.approx(resP_Ga.Fu.mean(), rel=0, abs=5e-8)
        assert 0 == pytest.approx(resS_Ga.Fu.mean(), rel=0, abs=5e-8)
        assert 0 == pytest.approx(np.abs(resP_Ga.AH - resS_Ga.AH), rel=0, abs=1e-4)

    if material in [1, 2, 4]:
        assert 0 == pytest.approx(resP_GaNi.Fu.mean(), rel=0, abs=5e-8)
        assert 0 == pytest.approx(resS_GaNi.Fu.mean(), rel=0, abs=5e-8)
        assert 0 == pytest.approx(np.abs(resP_GaNi.AH - resS_GaNi.AH), rel=0, abs=1e-4)

    tic.measure()


def test_cano_tensor(tensors_2d):
    T2d, T2dOther = tensors_2d

    u1, s1, vt1 = np.linalg.svd(T2d, full_matrices=0)
    a = CanoTensor(name="a", core=s1, basis=[u1.T, vt1])
    assert 0 == pytest.approx(norm(a.full() - T2d), rel=0, abs=5e-8)

    a = SparseTensor(kind="cano", val=T2d)
    assert 0 == pytest.approx(norm(a.full() - T2d), rel=0, abs=5e-8)

    b = SparseTensor(kind="cano", val=T2dOther)

    assert 0 == pytest.approx(norm((a + b).full() - T2d - T2dOther), rel=0, abs=5e-8)
    assert 0 == pytest.approx(norm((a * b).full() - T2d * T2dOther), rel=0, abs=5e-8)


def test_tucker(tensors_3d):
    T3d, T3dOther = tensors_3d

    a = SparseTensor(kind="tucker", val=T3d)
    assert 0 == pytest.approx(norm(a.full() - T3d), rel=0, abs=5e-8)

    b = SparseTensor(kind="tucker", val=T3dOther)

    assert 0 == pytest.approx(norm((a + b).full() - T3d - T3dOther), rel=0, abs=5e-8)
    assert 0 == pytest.approx(norm((a * b).full() - T3d * T3dOther), rel=0, abs=5e-8)


def test_tensor_train(tensors_3d):
    T3d, T3dOther = tensors_3d

    a = SparseTensor(kind="tt", val=T3d)
    assert 0 == pytest.approx(norm(a.full() - T3d), rel=0, abs=5e-8)

    b = SparseTensor(kind="tt", val=T3dOther)

    assert 0 == pytest.approx(norm((a + b).full() - T3d - T3dOther), rel=0, abs=5e-8)
    assert 0 == pytest.approx(norm((a * b).full() - T3d * T3dOther), rel=0, abs=5e-8)

    # A rank cap at the exact rank must preserve the sum.
    rounded = (a + b).truncate(rank=max(T3d.shape[0], T3d.shape[-1]))
    assert norm(rounded.full().val - T3d - T3dOther) < 5e-8


def test_tensor_train_vector_and_scalar():
    values = np.arange(5.0)
    train = SparseTensor(kind="tt", val=values)
    assert norm((train + train).full().val - 2 * values) < 1e-12
    assert norm((train * np.float64(2)).full().val - 2 * values) < 1e-12
    assert norm(train.fourier().full().val - np.fft.rfft(values)) < 1e-12


@pytest.mark.parametrize("bits", [(2, 2), (2, 3), (2, 2, 2)])
def test_qtt_fft_small(bits):
    shape = tuple(2**n for n in bits)
    values = np.arange(1, np.prod(shape) + 1, dtype=float).reshape(shape, order="F")
    quantized = values.reshape([2] * sum(bits), order="F")
    transformed = SparseTensor(kind="tt", val=quantized).qtt_fft(bits, tol=1e-10)
    reversed_shape = shape[::-1]
    actual = transformed.full().val.reshape(reversed_shape, order="F").transpose(
        tuple(reversed(range(len(bits))))
    )
    assert norm(actual - np.fft.fftn(values) / values.size) < 1e-8


def test_fourier(tensors_2d, tensors_3d):
    T2d, _ = tensors_2d
    T3d, _ = tensors_3d

    for opt in [0, "c"]:
        a = SparseTensor(kind="cano", val=T2d, fft_form=opt)
        T = Tensor(val=T2d, order=0, N=T2d.shape, Fourier=False, fft_form=opt)
        assert 0 == pytest.approx(
            norm(a.fourier().full(fft_form=opt).val - T.fourier(copy=True).val),
            rel=0,
            abs=5e-8,
        )
        assert norm(a.fourier().fourier(real_output=True).full().val.imag) == 0

        a = SparseTensor(kind="tucker", val=T3d, fft_form=opt)
        T = Tensor(val=T3d, order=0, N=T3d.shape, Fourier=False, fft_form=opt)
        assert 0 == pytest.approx(
            norm(a.fourier().full(fft_form=opt) - T.fourier(copy=True)), rel=0, abs=5e-8
        )
        assert norm(a.fourier().fourier(real_output=True).full().val.imag) == 0

        a = SparseTensor(kind="tt", val=T3d, fft_form=opt)
        T = Tensor(val=T3d, order=0, N=T3d.shape, Fourier=False, fft_form=opt)
        assert 0 == pytest.approx(
            norm(a.fourier().full(fft_form=opt) - T.fourier(copy=True).val),
            rel=0,
            abs=5e-8,
        )
        assert norm(a.fourier().fourier(real_output=True).full().val.imag) == 0
    # checking shifting fft_forms
    sparse_opt = "sr"
    for full_opt in [0, "c"]:
        a = SparseTensor(kind="cano", val=T2d, fft_form=sparse_opt)
        T = Tensor(val=T2d, order=0, N=T2d.shape, Fourier=False, fft_form=full_opt)
        assert 0 == pytest.approx(
            norm(a.fourier().full(fft_form=full_opt) - T.fourier(copy=True)),
            rel=0,
            abs=5e-8,
        )
        assert 0 == pytest.approx(
            (
                a.fourier().set_fft_form(full_opt) - a.set_fft_form(full_opt).fourier()
            ).norm(),
            rel=0,
            abs=5e-8,
        )

        a = SparseTensor(kind="tucker", val=T3d, fft_form=sparse_opt)
        T = Tensor(val=T3d, order=0, N=T3d.shape, Fourier=False, fft_form=full_opt)
        assert 0 == pytest.approx(
            norm(a.fourier().full(fft_form=full_opt) - T.fourier(copy=True)),
            rel=0,
            abs=5e-8,
        )
        assert 0 == pytest.approx(
            (
                a.fourier().set_fft_form(full_opt) - a.set_fft_form(full_opt).fourier()
            ).norm(),
            rel=0,
            abs=5e-8,
        )

        a = SparseTensor(kind="tt", val=T3d, fft_form=sparse_opt)
        T = Tensor(val=T3d, order=0, N=T3d.shape, Fourier=False, fft_form=full_opt)
        assert 0 == pytest.approx(
            norm(a.fourier().full(fft_form=full_opt) - T.fourier(copy=True)),
            rel=0,
            abs=5e-8,
        )
        assert 0 == pytest.approx(
            (
                a.fourier().set_fft_form(full_opt) - a.set_fft_form(full_opt).fourier()
            ).norm(),
            rel=0,
            abs=5e-8,
        )


def test_fourier_truncation():
    N = np.random.randint(20, 50, size=3)
    a = np.arange(1, np.prod(N) + 1).reshape(N)
    cases = [[None] * 2, [None] * 2]
    # first a random test case
    cases[0] = [np.random.random(N), np.random.random(N)]
    # this produces a "smooth", more realistic, tensor with modest rank
    cases[1] = [np.sin(a) / a, np.exp(np.sin(a) / a)]

    for i in range(len(cases)):
        for fft_form in [0, "c", "sr"]:
            a = cases[i][0]
            b = cases[i][1]
            ta = SparseTensor(
                kind="tt", val=a, fft_form=fft_form
            )  # Fourier truncation works the best with option 'sr'
            tb = SparseTensor(kind="tt", val=b, fft_form=fft_form)
            tc = ta + tb
            k = tc.r[1:-1].max() / 2 - 5

            tct = tc.truncate(rank=k)

            err_normal_truncate = (tct - tc).norm()
            #                print("loss in normal  domain truncation:",norm(tct.full().val-(a+b) ))

            taf = ta.fourier()
            tbf = tb.fourier()
            tcf = taf + tbf
            tcft = tcf.truncate(rank=k)
            tcfti = tcft.fourier()
            #                print("norm of imag part of F inverse tensor",norm(tcfti.full().val.imag))

            err_Fourier_truncate = (tcfti - tc).norm()
            #                print("loss in Fourier domain truncation:",norm(tcfti.full().val-(a+b) ))

            # assert the two truncation errors are in the same order
            assert err_Fourier_truncate == pytest.approx(
                err_normal_truncate, rel=0, abs=err_normal_truncate * 3
            )


@pytest.mark.skip(reason="The testing is too slow.")
def test_qtt_fft():
    L1 = 3
    L2 = 4
    L3 = 5
    tol = 1e-6
    # v=np.random.rand(2**L1,2**L2)
    v = np.array(list(range(1, 2 ** (L1 + L2 + L3) + 1)))
    v = np.sin(v) / v  # to increase the rank

    v1 = np.reshape(v, (2**L1, 2**L2, 2**L3), order="F")
    # vFFT= DFT.fftnc(v, [2**L1, 2**L2])
    # start = time.clock()
    v1fft = np.fft.fftn(v1) / 2 ** (L1 + L2 + L3)
    # print("FFT time:     ", (time.clock() - start))

    vq = np.reshape(v, [2] * (L1 + L2 + L3), order="F")  # a quantic tensor
    vqtt = SparseTensor(kind="tt", val=vq)  # a qtt

    # start = time.clock()
    vqf = vqtt.qtt_fft([L1, L2, L3], tol=tol)
    # print("QTT_FFT time: ", (time.clock() - start))

    vqf_full = vqf.full().reshape((2**L3, 2**L2, 2**L1), order="F")

    logger.debug('discrepancy:   %s', norm(vqf_full.T - v1fft) / norm(v1fft))
    logger.debug('maximum rank of the qtt is: %s', np.max(vqtt.r))

    assert norm(vqf_full.T - v1fft) / norm(v1fft) < 3 * tol

    #        qtt_fft_time= timeit.timeit('vqf= vqtt.fourier() ', number=50,
    #              setup="from ffthompy.tensorsLowRank.objects import SparseTensor; import numpy as np; L1=9; L2=8; L3=7; tol=1e-6; v1=np.array(range(1,2**(L1+L2+L3)+1));  v1=np.sin(v1)/v1; vq= np.reshape(v1,[2]*(L1+L2+L3),order='F'); vqtt= SparseTensor(kind='tt', val=vq )")
    #        print("QTT FFT time:",qtt_fft_time)

    tt_fft_time = timeit.timeit(
        "v1f= v1tt.fourier()",
        number=10,
        setup="from ffthompy.tensorsLowRank.objects import SparseTensor; import numpy as np; L1=9; L2=8; L3=7; v1=np.array(range(1,2**(L1+L2+L3)+1)); v1=np.sin(v1)/v1; v1tt= SparseTensor(kind='tt', val=v1,eps=1e-6 )",
    )
    logger.debug('  TT FFT time: %s', tt_fft_time)

    qtt_fft_time = timeit.timeit(
        "vqf= vqtt.qtt_fft( [L1,L2,L3],tol= tol) ",
        number=10,
        setup="from ffthompy.tensorsLowRank.objects import SparseTensor; import numpy as np; L1=9; L2=8; L3=7; tol=1e-6; v1=np.array(range(1,2**(L1+L2+L3)+1)); v1=np.sin(v1)/v1; vq= np.reshape(v1,[2]*(L1+L2+L3),order='F'); vqtt= SparseTensor(kind='tt', val=vq )",
    )
    logger.debug('QTT FFT time: %s', qtt_fft_time)

    assert qtt_fft_time < 0.1 * tt_fft_time


def test_orthogonalise(tensors_2d, tensors_3d):
    T2d, T2dOther = tensors_2d
    T3d, T3dOther = tensors_3d

    a = SparseTensor(kind="cano", val=T2d)
    b = SparseTensor(kind="cano", val=T2dOther)
    c = a + b
    co = c.orthogonalise()
    for i in range(co.order):
        identity = np.eye(co.N[i])
        assert np.dot(co.basis[i], co.basis[i].T).any() == identity.any()

    a = SparseTensor(kind="tucker", val=T3d)
    b = SparseTensor(kind="tucker", val=T3dOther)
    c = a + b
    co = c.orthogonalise()
    for i in range(co.order):
        identity = np.eye(co.N[i])
        assert np.dot(co.basis[i], co.basis[i].T).any() == identity.any()

    a = SparseTensor(kind="tt", val=T3d)
    b = SparseTensor(kind="tt", val=T3dOther)
    c = a + b
    co = c.orthogonalise(direction="lr")
    cr = co.to_list(co)
    for i in range(co.d):
        cr[i] = np.reshape(cr[i], (-1, co.r[i + 1]))
        identity = np.eye(co.N[i])
        assert np.dot(cr[i].T, cr[i]).any() == identity.any()

    co = c.orthogonalise(direction="rl")
    cr = co.to_list(co)
    for i in range(co.d):
        cr[i] = np.reshape(cr[i], (co.r[i], -1))
        identity = np.eye(co.N[i])
        assert np.dot(cr[i], cr[i].T).any() == identity.any()

    aSubTrain = c.tt_chunk(0, 1)
    co, ru = aSubTrain.orthogonalise(direction="rl", r_output=True)
    cr = co.to_list(co)
    for i in range(co.d):
        cr[i] = np.reshape(cr[i], (co.r[i], -1))
        identity = np.eye(co.N[i])
        assert np.dot(cr[i], cr[i].T).any() == identity.any()


def test_mean(tensors_2d, tensors_3d):
    T2d, _ = tensors_2d
    T3d, _ = tensors_3d
    a = SparseTensor(kind="cano", val=T2d)
    assert a.mean() == pytest.approx(np.mean(T2d), rel=0, abs=5e-8)
    assert a.fourier().mean() == pytest.approx(np.mean(T2d), rel=0, abs=5e-8)

    a = SparseTensor(kind="tucker", val=T3d)
    assert a.mean() == pytest.approx(np.mean(T3d), rel=0, abs=5e-8)
    assert a.fourier().mean() == pytest.approx(np.mean(T3d), rel=0, abs=5e-8)

    a = SparseTensor(kind="tt", val=T3d)
    assert a.mean() == pytest.approx(np.mean(T3d), rel=0, abs=5e-8)
    assert a.fourier().mean() == pytest.approx(np.mean(T3d), rel=0, abs=5e-8)
