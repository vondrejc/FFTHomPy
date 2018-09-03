import numpy as np
from ffthompy.trigpol import Grid
from ffthompy.sparse.canoTensor import CanoTensor
from ffthompy.sparse.tucker import Tucker
from ffthompy.sparse.tensorTrain import TensorTrain


def grad_tensor(N, Y, tensor):
    assert(tensor in [CanoTensor, Tucker, TensorTrain])
    dim=Y.size
    freq=Grid.get_xil(N, Y)
    hGrad_s=[]

    for ii in range(dim):
        basis=[]
        for jj in range(dim):
            if ii==jj:
                basis.append(np.atleast_2d(freq[jj]*2*np.pi*1j))
            else:
                basis.append(np.atleast_2d(np.ones(N[jj])))

        if tensor in [CanoTensor, Tucker]:
            hGrad_s.append(tensor(name='hGrad({})'.format(ii), core=np.array([1]), basis=basis,
                                  Fourier=True))

        elif tensor is TensorTrain:
            cl = [bas.reshape((1,-1,1)) for bas in basis]
            hGrad_s.append(TensorTrain.from_list(cl, name='hGrad({})'.format(ii), Fourier=True))

    return hGrad_s
