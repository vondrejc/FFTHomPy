import numpy as np
from ffthompy.trigpol import Grid
from ffthompy.sparse.canoTensor import CanoTensor


def grad_tensor(N, Y):
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
        hGrad_s.append(CanoTensor(name='hGrad({})'.format(ii), core=np.array([1]), basis=basis,
                                  Fourier=True))
    return hGrad_s
