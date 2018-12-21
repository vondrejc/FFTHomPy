import numpy as np
from ffthompy.trigpol import Grid
from ffthompy.sparse.objects import SparseTensor

def grad_tensor(N, Y, kind='TensorTrain'):
    assert(kind.lower() in ['cano','canotensor','tucker','tt','tensortrain'])

    dim=Y.size
    freq=Grid.get_xil(N, Y, fft_form='c')
    hGrad_s=[]

    for ii in range(dim):
        basis=[]
        for jj in range(dim):
            if ii==jj:
                basis.append(np.atleast_2d(freq[jj]*2*np.pi*1j))
            else:
                basis.append(np.atleast_2d(np.ones(N[jj])))

        if kind.lower() in ['cano', 'canotensor','tucker']:
            hGrad_s.append(SparseTensor(kind=kind, name='hGrad({})'.format(ii), core=np.array([1.]),
                                        basis=basis, Fourier=True, fft_form='c').set_fft_form())
        elif kind.lower() in ['tt','tensortrain']:
            cl = [bas.reshape((1,-1,1)) for bas in basis]
            hGrad_s.append(SparseTensor(kind=kind, core=cl, name='hGrad({})'.format(ii),
                                        Fourier=True, fft_form='c').set_fft_form())

    return hGrad_s
