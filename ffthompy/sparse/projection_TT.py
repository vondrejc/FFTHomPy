import numpy as np
from ffthompy.trigpol import Grid
#from ffthompy.sparse.canoTensor import CanoTensor
#from ffthompy.sparse.tucker import Tucker
from ffthompy.sparse.tensorTrain import TensorTrain

def grad_tensor(N, Y):
    dim=Y.size
    freq=Grid.get_xil(N, Y)
    hGrad_s=[]

    for ii in range(dim):        
        #basis=[]
        cl=[]
        for jj in range(dim):
            if ii==jj:
                #basis.append(np.atleast_2d(freq[jj]*2*np.pi*1j))
                cl.append(np.reshape(freq[jj]*2*np.pi*1j,(1,-1,1)))  # since it is rank 1
            else:
                #basis.append(np.atleast_2d(np.ones(N[jj])))
                cl.append(np.ones((1,N[jj],1)))  # since it is rank 1            
            
        #hGrad_s.append(Tucker(name='hGrad({})'.format(ii), core=np.array([1]), basis=basis, Fourier=True))
        hGrad_s.append(TensorTrain.from_list(cl, name='hGrad({})'.format(ii), Fourier=True))
        
    return hGrad_s
