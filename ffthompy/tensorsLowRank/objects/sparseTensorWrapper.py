import logging

import numpy as np

from ffthompy.tensorsLowRank.objects.canoTensor import CanoTensor
from ffthompy.tensorsLowRank.objects.tensors import fft_form_default
from ffthompy.tensorsLowRank.objects.tucker import Tucker

logger = logging.getLogger(__name__)


def SparseTensor(kind='tt', val=None, core=None, basis=None, eps=None, rank=None,
                 Fourier=False, name='unnamed', vectorObj=None, fft_form=fft_form_default):
    """
    A uniform wrapper of different tensorsLowRank tensor format

    :param kind: type of tensorsLowRank tensor, can be 'cano','tucker' or 'tt', or more variants (see the code).
    :type kind: string

    :param val: a full tensor to be approximated
    :type val: n-D array

    :param core: core for canonical, tucker or TT tensorsLowRank tensor
    :type core: 1-D array for canonical tensor, n-D arary for tucker, list of arrays for TT.

    :param basis: basis for canonical or tucker tensorsLowRank tensor.
    :type basis: list of arrays.

    :param eps: approximation accuracy.
    :type eps: float.

    :param rank: rank of the cano and tucker tensorsLowRank tensor, maximum rank of TT tensorsLowRank tensor.
    :type rank: int for cano and TT, list of int for tucker.

    :param vectorObj: a SeeMPS MPS object, to be cast into a TensorTrain object.
    :type vectorObj: SeeMPS MPS

    :returns: object of tensorsLowRank tensor.
    """
    if type(rank) is list or type(rank) is np.ndarray:
        rmax=max(rank)
        r=min(rank)
    else:
        rmax=r=rank

    if kind.lower() in ['cano', 'canotensor']:
        return CanoTensor(name=name, val=val, core=core,basis=basis,Fourier=Fourier,fft_form=fft_form).truncate(rank=r, tol=eps)
    elif kind.lower() in ['tucker']:
        return Tucker(name=name, val=val, core=core, basis=basis,Fourier=Fourier,fft_form=fft_form).truncate(rank=rank, tol=eps)
    elif kind.lower() in ['tt', 'tensortrain']:
        from ffthompy.tensorsLowRank.objects.tensorTrain import TensorTrain
        return TensorTrain(val=val, core=core, eps=eps, rmax=rmax, name=name,
                           Fourier=Fourier, vectorObj=vectorObj,fft_form=fft_form)
    else:
        raise ValueError("Unexpected argument value: '" + kind +"'")


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from seemps.state import MPS, NO_TRUNCATION

    logger.info("")
    logger.info('----testing "Repeat" function ----')
    logger.info("")

    v1 = np.random.rand(3, 3,3)

    tt = SparseTensor(kind='tt', val=v1)
    # tt.fourier()
    logger.info('%s', tt.full())
    tt.repeat(6)

    logger.info('%s', tt.full())

    logger.info('\n----testing wrapper function ----\n')

    v1=np.random.rand(20, 30)

    cano=SparseTensor(kind='cano', val=v1)
    logger.info('%s', cano)

    cano2=SparseTensor(kind='cano', val=v1, rank=10)
    logger.info('%s', cano2)

    cano3=SparseTensor(kind='cano', core=np.array([1.]), basis=[np.atleast_2d(np.ones(5)) for ii in range(2)], Fourier=False)
    logger.info('%s', cano3)

    v1=np.random.rand(20, 30, 40)

    tucker1=SparseTensor(kind='tucker', val=v1)
    logger.info('%s', tucker1)

    tucker2=SparseTensor(kind='tucker', val=v1, rank=[10, 20, 35])
    logger.info('%s', tucker2)

    tucker3=SparseTensor(kind='tucker', core=np.array([1.]), basis=[np.atleast_2d(np.ones(5)) for ii in range(3)])
    logger.info('%s', tucker3)

    tt1=SparseTensor(kind='tt', val=v1)
    logger.info('%s', tt1)

    tt2=SparseTensor(kind='tt', val=v1, eps=2e-1)
    logger.info('%s', tt2)

    tt_vec=MPS.from_tensor(v1, strategy=NO_TRUNCATION, normalize=False)
    tt3=SparseTensor(kind='TT', vectorObj=tt_vec)
    logger.info('%s', tt3)

    tt4=SparseTensor()
    logger.info('%s', tt4)

    v1=np.random.rand(20, 30)
    cano=SparseTensor(kind='CAno', val=v1)
    logger.info('%s', cano)

    logger.info('END')
