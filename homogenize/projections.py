import numpy as np
from homogenize.trigonometric import TrigPolynomial
from homogenize.matvec_fun import enlarge_M


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
    N = np.array(N)
    if NyqNul:
        Nred = N - ((N+1) % 2)
    else:
        Nred = N

    xi = TrigPolynomial.get_xil(Nred, Y)
    xi2 = []
    for m in np.arange(d):
        xi2.append(xi[m]**2)

    G0l = np.zeros(np.hstack([d, d, Nred]))
    G1l = np.zeros(np.hstack([d, d, Nred]))
    G2l = np.zeros(np.hstack([d, d, Nred]))
    num = np.zeros(np.hstack([d, d, Nred]))
#     G1l = np.zeros([d, d]).tolist()
#     G2l = np.zeros([d, d]).tolist()
#     num = np.zeros([d, d]).tolist()
    denom = np.zeros(Nred)

    ind_center = tuple(np.fix(np.array(Nred)/2))
    for m in np.arange(d): # diagonal components
        Nshape = np.ones(d)
        Nshape[m] = Nred[m]
        Nrep = np.copy(Nred)
        Nrep[m] = 1
        a = np.reshape(xi2[m], Nshape)
        num[m][m] = np.tile(a, Nrep) # numerator
        denom = denom + num[m][m]
        G0l[m, m][ind_center] = 1

    for m in np.arange(d): # upper diagonal components
        for n in np.arange(m+1, d):
            NshapeM = np.ones(d)
            NshapeM[m] = Nred[m]
            NrepM = np.copy(Nred)
            NrepM[m] = 1
            NshapeN = np.ones(d)
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

    if NyqNul:
        G1l = enlarge_M(G1l, N)
        G2l = enlarge_M(G2l, N)

    if not centered:
        for m in np.arange(d):
            for n in np.arange(d):
                G1l[m][n] = np.fft.ifftshift(G1l[m][n])
                G2l[m][n] = np.fft.ifftshift(G2l[m][n])
    return G0l, G1l, G2l


if __name__ == '__main__':
    execfile('../main_test.py')
