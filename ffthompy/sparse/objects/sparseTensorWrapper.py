import numpy as np
from tt.core.vector import vector
from ffthompy.sparse.objects.tucker import Tucker
from ffthompy.sparse.objects.canoTensor import CanoTensor
from ffthompy.sparse.objects.tensorTrain import TensorTrain
from ffthompy.sparse.objects.tensors import fft_form_default


def SparseTensor(kind='tt', val=None, core=None, basis=None, eps=None, rank=None,
                 Fourier=False, name='unnamed', vectorObj=None, fft_form=fft_form_default):
    """
    A uniform wrapper of different sparse tensor format

    :param kind: type of sparse tensor, can be 'cano','tucker' or 'tt', or more variants (see the code).
    :type kind: string

    :param val: a full tensor to be approximated
    :type val: n-D array

    :param core: core for canonical, tucker or TT sparse tensor
    :type core: 1-D array for canonical tensor, n-D arary for tucker, list of arrays for TT.

    :param basis: basis for canonical or tucker sparse tensor.
    :type basis: list of arrays.

    :param eps: approximation accuracy.
    :type eps: float.

    :param rank: rank of the cano and tucker sparse tensor, maximum rank of TT sparse tensor.
    :type rank: int for cano and TT, list of int for tucker.

    :param vectorObj: a TTPY vector class object, to be cast into tensorTrain object.
    :type vectorObj:  TTPY vector

    :returns: object of sparse tensor.
    """
    if type(rank) is list or type(rank) is np.ndarray :
        rmax=max(rank)
        r=min(rank)
    else:
        rmax=r=rank

    if kind.lower() in ['cano','canotensor'] :
        return CanoTensor(name=name, val=val, core=core,basis=basis,Fourier=Fourier,fft_form=fft_form).truncate(rank=r, tol=eps)
    elif kind.lower() in ['tucker']:
        return Tucker(name=name, val=val, core=core, basis=basis,Fourier=Fourier,fft_form=fft_form).truncate(rank=rank, tol=eps)
    elif kind.lower() in ['tt', 'tensortrain']:
        return TensorTrain(val=val, core=core, eps=eps, rmax=rmax, name=name,
                           Fourier=Fourier, vectorObj=vectorObj,fft_form=fft_form)
    else:
        raise ValueError("Unexpected argument value: '" + kind +"'")


if __name__=='__main__':

    print
    print('----testing wrapper function ----')
    print

    v1=np.random.rand(20, 30)

    cano=SparseTensor(kind='cano', val=v1)
    print(cano)

    cano2=SparseTensor(kind='cano', val=v1, rank=10)
    print(cano2)

    cano3=SparseTensor(kind='cano', core=np.array([1.]), basis=[np.atleast_2d(np.ones(5)) for ii in range(2)], Fourier=False)
    print(cano3)

    v1=np.random.rand(20, 30, 40)

    tucker1=SparseTensor(kind='tucker', val=v1)
    print(tucker1)

    tucker2=SparseTensor(kind='tucker', val=v1, rank=[10, 20, 35])
    print(tucker2)

    tucker3=SparseTensor(kind='tucker', core=np.array([1.]), basis=[np.atleast_2d(np.ones(5)) for ii in range(3)])
    print(tucker3)

    tt1=SparseTensor(kind='tt', val=v1)
    print(tt1)

    tt2=SparseTensor(kind='tt', val=v1, eps=2e-1)
    print(tt2)

    tt_vec=vector(v1)
    tt3=SparseTensor(kind='TT', vectorObj=tt_vec)
    print(tt3)

    tt4=SparseTensor()
    print (tt4)

    v1=np.random.rand(20, 30)
    cano=SparseTensor(kind='CAno', val=v1)
    print(cano)


    print('END')
