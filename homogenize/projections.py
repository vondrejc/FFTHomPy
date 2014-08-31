import numpy as np
from homogenize.trigonometric import TrigPolynomial

def get_Fourier_projections(N, Y, centered=True, NyqNul=True):
    """
    Assembly of discrete kernels in Fourier space for scalar elliptic problems.

    Parameters
    ----------
    N - no. of discretization points
    Y - size of periodic unit cell

    Returns
    -------
    G1l - discrete kernel in Fourier space; provides projection
        on curl-free fields with zero mean
    G2l - discrete kernel in Fourier space; provides projection
        on divergence-free fields with zero mean
    """
    d = np.size(N)
    xi2 = []
    N = np.array(N)

    xi = TrigPolynomial.get_xil(N, Y)
    for ii in np.arange(d):
        if NyqNul and (N[ii] % 2) == 0:
            xi[ii][0] = 0.

    for m in np.arange(d):
        xi2.append(xi[m]**2)

    G0l = np.zeros(np.hstack([d, d, N]))
    G1l = np.zeros([d, d]).tolist()
    G2l = np.zeros([d, d]).tolist()
    num = np.zeros([d, d]).tolist()
    denom = np.zeros(N)

    ind_center = tuple(np.fix(np.array(N)/2))
    for m in np.arange(d): # diagonal components
        Nshape = np.ones(d)
        Nshape[m] = N[m]
        Nrep = np.copy(N)
        Nrep[m] = 1
        a = np.reshape(xi2[m], Nshape)
        num[m][m] = np.tile(a, Nrep) # numerator
        denom = denom + num[m][m]
        G0l[m, m][ind_center] = 1

    for m in np.arange(d): # upper diagonal components
        for n in np.arange(m+1, d):
            NshapeM = np.ones(d)
            NshapeM[m] = N[m]
            NrepM = np.copy(N)
            NrepM[m] = 1
            NshapeN = np.ones(d)
            NshapeN[n] = N[n]
            NrepN = np.copy(N)
            NrepN[n] = 1
            num[m][n] = np.tile(np.reshape(xi[m], NshapeM), NrepM) \
                        * np.tile(np.reshape(xi[n], NshapeN), NrepN)

    denom[ind_center] = 1 #avoiding a division by zero

    # calculation of projections
    for m in np.arange(d):
        for n in np.arange(m, d):
            G1l[m][n] = num[m][n]/denom
            G2l[m][n] = (m==n)*np.ones(N) - G1l[m][n]
            G2l[m][n][ind_center] = 0

    # symmetrization
    for m in np.arange(1, d):
        for n in np.arange(m):
            G1l[m][n] = G1l[n][m]
            G2l[m][n] = G2l[n][m]

    if not centered:
        for m in np.arange(d):
            for n in np.arange(d):
                G1l[m][n] = np.fft.ifftshift(G1l[m][n])
                G2l[m][n] = np.fft.ifftshift(G2l[m][n])
    return G0l, G1l, G2l
