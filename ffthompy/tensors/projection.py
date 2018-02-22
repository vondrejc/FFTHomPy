import numpy as np
from ffthompy.trigpol import Grid, get_Nodd, mean_index
from .objects import Tensor
import itertools

def scalar_tensor(N, Y, centered=True, NyqNul=True):
    dim = np.size(N)
    N = np.array(N, dtype=np.int)
    if NyqNul:
        Nred = get_Nodd(N)
    else:
        Nred = N
    assert(np.allclose(N, Nred))

    xi = Grid.get_xil(N, Y)
    hGrad = np.zeros((dim,)+ tuple(N)) # zero initialize
    for ind in itertools.product(*[range(n) for n in N]):
        for i in range(dim):
            hGrad[i][ind] = xi[i][ind[i]]

    kok= np.einsum('i...,j...->ij...', hGrad, hGrad).real
    k2 = np.einsum('i...,i...', hGrad, hGrad).real
    ind_center=mean_index(N)
    k2[ind_center]=1.

    G0lval=np.zeros_like(kok)
    Ival=np.zeros_like(kok)
    for ii in range(dim): # diagonal components
        G0lval[ii, ii][ind_center] = 1
        Ival[ii, ii] = 1
    G1l=Tensor(name='G1', val=kok/k2, order=2, Y=Y, Fourier=True, multype=21)
    G0l=Tensor(name='G1', val=G0lval, order=2, Y=Y, Fourier=True, multype=21)
    I = Tensor(name='I', val=Ival, order=2, Y=Y, Fourier=True, multype=21)
    G2l=I-G1l-G0l
    return G0l, G1l, G2l

def elasticity_small_strain(N, Y, centered=True, NyqNul=True):
    N = np.array(N, dtype=np.int)
    dim = N.size
    assert(dim==3)
    Ghat = np.zeros(np.hstack([dim*np.ones(4, dtype=np.int), N]))
    freq = Grid.get_xil(N, Y)
    delta  = lambda i,j: np.float(i==j)

    for i, j, k, l in itertools.product(range(dim), repeat=4):
        for x, y, z in np.ndindex(*N):
            q = np.array([freq[0][x], freq[1][y], freq[2][z]])
            if not q.dot(q) == 0:
                Ghat[i,j,k,l,x,y,z] = -q[i]*q[j]*q[k]*q[l]/(q.dot(q))**2 + \
                    .5*(delta(i,k)*q[j]*q[l]+delta(i,l)*q[j]*q[k] +\
                        delta(j,k)*q[i]*q[l]+delta(j,l)*q[i]*q[k] ) / (q.dot(q))

    Ghat_tensor = Tensor(name='Ghat', val=Ghat, order=4, Fourier=True,
                         multype=42)
    return Ghat_tensor


def elasticity_large_deformation(N, Y, centered=True, NyqNul=True):
    N = np.array(N, dtype=np.int)
    dim = N.size
    assert(dim==3)
    Ghat = np.zeros(np.hstack([dim*np.ones(4, dtype=np.int), N]))
    freq = Grid.get_xil(N, Y)
    delta  = lambda i,j: np.float(i==j)

    for i, j, k, l in itertools.product(range(dim), repeat=4):
        for x, y, z in np.ndindex(*N):
            q = np.array([freq[0][x], freq[1][y], freq[2][z]])
            if not q.dot(q) == 0:
                Ghat[i,j,k,l,x,y,z] = delta(i,k)*q[j]*q[l] / (q.dot(q))

    Ghat_tensor = Tensor(name='Ghat', val=Ghat, order=4, Fourier=True,
                         multype=42)
    return Ghat_tensor
