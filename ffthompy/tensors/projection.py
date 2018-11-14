import numpy as np
from ffthompy.trigpol import Grid, get_Nodd, mean_index, fft_form_default
from .objects import Tensor
import itertools

def scalar(N, Y, NyqNul=True, fft_form=fft_form_default):
    dim = np.size(N)
    N = np.array(N, dtype=np.int)
    if NyqNul:
        Nred = get_Nodd(N)
    else:
        Nred = N
    assert(np.allclose(N, Nred))

    xi = Grid.get_freq(N, Y, fft_form=fft_form)
    N_fft=tuple(xi[i].size for i in range(dim))
    hGrad = np.zeros((dim,)+N_fft) # zero initialize
    for ind in itertools.product(*[range(n) for n in N_fft]):
        for i in range(dim):
            hGrad[i][ind] = xi[i][ind[i]]

    kok= np.einsum('i...,j...->ij...', hGrad, hGrad).real
    k2 = np.einsum('i...,i...', hGrad, hGrad).real
    ind_center=mean_index(N, fft_form=fft_form)
    k2[ind_center]=1.

    G0lval=np.zeros_like(kok)
    Ival=np.zeros_like(kok)
    for ii in range(dim): # diagonal components
        G0lval[ii, ii][ind_center] = 1
        Ival[ii, ii] = 1
    G1l=Tensor(name='G1', val=kok/k2, order=2, Y=Y, multype=21, Fourier=True, fft_form=fft_form)
    G0l=Tensor(name='G1', val=G0lval, order=2, Y=Y, multype=21, Fourier=True, fft_form=fft_form)
    I = Tensor(name='I', val=Ival, order=2, Y=Y, multype=21, Fourier=True, fft_form=fft_form)
    G2l=I-G1l-G0l
    return G0l, G1l, G2l

def elasticity_small_strain(N, Y, fft_form=fft_form_default):
    N = np.array(N, dtype=np.int)
    dim = N.size
    assert(dim==3)
    freq = Grid.get_freq(N, Y, fft_form=fft_form)
    N_fft=tuple(freq[i].size for i in range(dim))
    Ghat = np.zeros(np.hstack([dim*np.ones(4, dtype=np.int), N_fft]))
    delta  = lambda i,j: np.float(i==j)

    for i, j, k, l in itertools.product(range(dim), repeat=4):
        for x, y, z in np.ndindex(*N_fft):
            q = np.array([freq[0][x], freq[1][y], freq[2][z]])
            if not q.dot(q) == 0:
                Ghat[i,j,k,l,x,y,z] = -q[i]*q[j]*q[k]*q[l]/(q.dot(q))**2 + \
                    .5*(delta(i, k)*q[j]*q[l]+delta(i, l)*q[j]*q[k]+
                        delta(j, k)*q[i]*q[l]+delta(j, l)*q[i]*q[k])/(q.dot(q))

    Ghat_tensor = Tensor(name='Ghat', val=Ghat, order=4, multype=42, Fourier=True, fft_form=fft_form)
    return Ghat_tensor

def elasticity_large_deformation(N, Y, fft_form=fft_form_default):
    N = np.array(N, dtype=np.int)
    dim = N.size
    assert(dim==3)
    freq = Grid.get_freq(N, Y, fft_form=fft_form)
    N_fft=tuple(freq[i].size for i in range(dim))
    Ghat = np.zeros(np.hstack([dim*np.ones(4, dtype=np.int), N_fft]))
    delta  = lambda i,j: np.float(i==j)

    for i, j, k, l in itertools.product(range(dim), repeat=4):
        for x, y, z in np.ndindex(*N_fft):
            q = np.array([freq[0][x], freq[1][y], freq[2][z]])
            if not q.dot(q) == 0:
                Ghat[i,j,k,l,x,y,z] = delta(i,k)*q[j]*q[l] / (q.dot(q))

    Ghat_tensor = Tensor(name='Ghat', val=Ghat, order=4, multype=42,
                         Fourier=True, fft_form=fft_form)
    return Ghat_tensor
