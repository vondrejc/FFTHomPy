import numpy as np
import scipy as sp
from ffthompy.trigpol import Grid, get_Nodd, mean_index, fft_form_default
from ffthompy.matvecs import Matrix
from ffthompy.tensors import Tensor
import itertools


def scalar(N, Y, NyqNul=True, tensor=True, fft_form=fft_form_default):
    """
    Assembly of discrete kernels in Fourier space for scalar elliptic problems.

    Parameters
    ----------
    N : numpy.ndarray
        no. of discretization points
    Y : numpy.ndarray
        size of periodic unit cell

    Returns
    -------
    G1l : numpy.ndarray
        discrete kernel in Fourier space; provides projection
        on curl-free fields with zero mean
    G2l : numpy.ndarray
        discrete kernel in Fourier space; provides projection
        on divergence-free fields with zero mean
    """
    if fft_form in ['r']:
        fft_form_r=True
        fft_form=0
    else:
        fft_form_r=False

    d = np.size(N)
    N = np.array(N, dtype=np.int)
    if NyqNul:
        Nred = get_Nodd(N)
    else:
        Nred = N

    xi = Grid.get_xil(Nred, Y, fft_form=fft_form)
    xi2 = []
    for m in np.arange(d):
        xi2.append(xi[m]**2)

    G0l = np.zeros(np.hstack([d, d, Nred]))
    G1l = np.zeros(np.hstack([d, d, Nred]))
    G2l = np.zeros(np.hstack([d, d, Nred]))
    num = np.zeros(np.hstack([d, d, Nred]))
    denom = np.zeros(Nred)

    ind_center = mean_index(Nred, fft_form=fft_form)
    for m in np.arange(d): # diagonal components
        Nshape = np.ones(d, dtype=np.int)
        Nshape[m] = Nred[m]
        Nrep = np.copy(Nred)
        Nrep[m] = 1
        a = np.reshape(xi2[m], Nshape)
        num[m][m] = np.tile(a, Nrep) # numerator
        denom = denom + num[m][m]
        G0l[m, m][ind_center] = 1

    for m in np.arange(d): # upper diagonal components
        for n in np.arange(m+1, d):
            NshapeM = np.ones(d, dtype=np.int)
            NshapeM[m] = Nred[m]
            NrepM = np.copy(Nred)
            NrepM[m] = 1
            NshapeN = np.ones(d, dtype=np.int)
            NshapeN[n] = Nred[n]
            NrepN = np.copy(Nred)
            NrepN[n] = 1
            num[m][n] = np.tile(np.reshape(xi[m], NshapeM), NrepM) \
                * np.tile(np.reshape(xi[n], NshapeN), NrepN)

    # avoiding a division by zero
    denom[ind_center] = 1

    # calculation of projections
    for m in np.arange(d):
        for n in np.arange(m, d):
            G1l[m][n] = num[m][n]/denom
            G2l[m][n] = (m == n)*np.ones(Nred) - G1l[m][n]
            G2l[m][n][ind_center] = 0

    # symmetrization
    for m in np.arange(1, d):
        for n in np.arange(m):
            G1l[m][n] = G1l[n][m]
            G2l[m][n] = G2l[n][m]

    if tensor:
        G0l = Tensor(name='hG0', val=G0l, order=2, multype=21, Fourier=True, fft_form=fft_form)
        G1l = Tensor(name='hG1', val=G1l, order=2, multype=21, Fourier=True, fft_form=fft_form)
        G2l = Tensor(name='hG2', val=G2l, order=2, multype=21, Fourier=True, fft_form=fft_form)
    else:
        G0l = Matrix(name='hG0', val=G0l, Fourier=True)
        G1l = Matrix(name='hG1', val=G1l, Fourier=True)
        G2l = Matrix(name='hG2', val=G2l, Fourier=True)

    if NyqNul:
        G0l = G0l.enlarge(N)
        G1l = G1l.enlarge(N)
        G2l = G2l.enlarge(N)

    if fft_form_r:
        for tensor in [G0l, G1l, G2l]:
            tensor.set_fft_form(fft_form='r')
            tensor.val/=np.prod(tensor.N)

    return G0l, G1l, G2l

def elasticity(N, Y, NyqNul=True, tensor=True, fft_form=fft_form_default):
    """
    Projection matrix on a space of admissible strain fields
    INPUT =
        N : ndarray of e.g. stiffness coefficients
        d : dimension; d = 2
        D : dimension in engineering notation; D = 3
        Y : the size of periodic unit cell
    OUTPUT =
        G1h,G1s,G2h,G2s : projection matrices of size DxDxN
    """
    if fft_form in ['r']:
        fft_form_r=True
        fft_form=0
    else:
        fft_form_r=False

    xi = Grid.get_xil(N, Y, fft_form=fft_form)
    N = np.array(N, dtype=np.int)
    d = N.size
    D = int(d*(d+1)/2)

    if NyqNul:
        Nred = get_Nodd(N)
    else:
        Nred = N

    xi2 = []
    for ii in range(d):
        xi2.append(xi[ii]**2)

    num = np.zeros(np.hstack([d, d, Nred]))
    norm2_xi = np.zeros(Nred)
    for mm in np.arange(d): # diagonal components
        Nshape = np.ones(d, dtype=np.int)
        Nshape[mm] = Nred[mm]
        Nrep = np.copy(Nred)
        Nrep[mm] = 1
        num[mm][mm] = np.tile(np.reshape(xi2[mm], Nshape), Nrep) # numerator
        norm2_xi += num[mm][mm]

    norm4_xi = norm2_xi**2
    ind_center = mean_index(Nred, fft_form=fft_form)
    # avoid division by zero
    norm2_xi[ind_center] = 1
    norm4_xi[ind_center] = 1

    for m in np.arange(d): # upper diagonal components
        for n in np.arange(m+1, d):
            NshapeM = np.ones(d, dtype=np.int)
            NshapeM[m] = Nred[m]
            NrepM = np.copy(Nred)
            NrepM[m] = 1
            NshapeN = np.ones(d, dtype=np.int)
            NshapeN[n] = Nred[n]
            NrepN = np.copy(Nred)
            NrepN[n] = 1
            num[m][n] = np.tile(np.reshape(xi[m], NshapeM), NrepM) \
                * np.tile(np.reshape(xi[n], NshapeN), NrepN)

    # G1h = np.zeros([D,D]).tolist()
    G1h = np.zeros(np.hstack([D, D, Nred]))
    G1s = np.zeros(np.hstack([D, D, Nred]))
    IS0 = np.zeros(np.hstack([D, D, Nred]))
    mean = np.zeros(np.hstack([D, D, Nred]))
    Lamh = np.zeros(np.hstack([D, D, Nred]))
    S = np.zeros(np.hstack([D, D, Nred]))
    W = np.zeros(np.hstack([D, D, Nred]))
    WT = np.zeros(np.hstack([D, D, Nred]))

    for m in np.arange(d):
        S[m][m] = 2*num[m][m]/norm2_xi
        for n in np.arange(d):
            G1h[m][n] = num[m][m]*num[n][n]/norm4_xi
            Lamh[m][n] = np.ones(Nred)/d
            Lamh[m][n][ind_center] = 0

    for m in np.arange(D):
        IS0[m][m] = np.ones(Nred)
        IS0[m][m][ind_center] = 0
        mean[m][m][ind_center] = 1

    if d == 2:
        S[0][2] = 2**0.5*num[0][1]/norm2_xi
        S[1][2] = 2**0.5*num[0][1]/norm2_xi
        S[2][2] = np.ones(Nred)
        S[2][2][ind_center] = 0
        G1h[0][2] = 2**0.5*num[0][0]*num[0][1]/norm4_xi
        G1h[1][2] = 2**0.5*num[0][1]*num[1][1]/norm4_xi
        G1h[2][2] = 2*num[0][0]*num[1][1]/norm4_xi
        for m in np.arange(d):
            for n in np.arange(d):
                W[m][n] = num[m][m]/norm2_xi
            W[2][m] = 2**.5*num[0][1]/norm2_xi

    elif d == 3:
        for m in np.arange(d):
            S[m+3][m+3] = 1 - num[m][m]/norm2_xi
            S[m+3][m+3][ind_center] = 0
        for m in np.arange(d):
            for n in np.arange(m+1, d):
                S[m+3][n+3] = num[m][n]/norm2_xi
                G1h[m+3][n+3] = num[m][m]*num[n][n]/norm4_xi
        for m in np.arange(d):
            for n in np.arange(d):
                ind = sp.setdiff1d(np.arange(d), [n])
                S[m][n+3] = (0 == (m == n))*2**.5*num[ind[0]][ind[1]]/norm2_xi
                G1h[m][n+3] = 2**.5*num[m][m]*num[ind[0]][ind[1]]/norm4_xi
                W[m][n] = num[m][m]/norm2_xi
                W[n+3][m] = 2**.5*num[ind[0]][ind[1]]/norm2_xi
        for m in np.arange(d):
            for n in np.arange(d):
                ind_m = sp.setdiff1d(np.arange(d), [m])
                ind_n = sp.setdiff1d(np.arange(d), [n])
                G1h[m+3][n+3] = 2*num[ind_m[0]][ind_m[1]] \
                    * num[ind_n[0]][ind_n[1]] / norm4_xi
    # symmetrization
    for n in np.arange(D):
        for m in np.arange(n+1, D):
            S[m][n] = S[n][m]
            G1h[m][n] = G1h[n][m]
    for m in np.arange(D):
        for n in np.arange(D):
            G1s[m][n] = S[m][n] - 2*G1h[m][n]
            WT[m][n] = W[n][m]
    G2h = 1./(d-1)*(d*Lamh + G1h - W - WT)
    G2s = IS0 - G1h - G1s - G2h

    if tensor:
        G0 = Tensor(name='hG0', val=mean, order=2, Fourier=True, multype=21, fft_form=fft_form)
        G1h = Tensor(name='hG1h', val=G1h, order=2, Fourier=True, multype=21, fft_form=fft_form)
        G1s = Tensor(name='hG1s', val=G1s, order=2, Fourier=True, multype=21, fft_form=fft_form)
        G2h = Tensor(name='hG2h', val=G2h, order=2, Fourier=True, multype=21, fft_form=fft_form)
        G2s = Tensor(name='hG2s', val=G2s, order=2, Fourier=True, multype=21, fft_form=fft_form)
    else:
        G0 = Matrix(name='hG0', val=mean, Fourier=True)
        G1h = Matrix(name='hG1h', val=G1h, Fourier=True)
        G1s = Matrix(name='hG1s', val=G1s, Fourier=True)
        G2h = Matrix(name='hG2h', val=G2h, Fourier=True)
        G2s = Matrix(name='hG2s', val=G2s, Fourier=True)

    if NyqNul:
        G0 = G0.enlarge(N)
        G1h = G1h.enlarge(N)
        G1s = G1s.enlarge(N)
        G2h = G2h.enlarge(N)
        G2s = G2s.enlarge(N)

    if fft_form_r:
        for tensor in [G0, G1h, G1s, G2h, G2s]:
            tensor.set_fft_form(fft_form='r')
            tensor.val=1./np.prod(tensor.N)*tensor.val

    return G0, G1h, G1s, G2h, G2s
