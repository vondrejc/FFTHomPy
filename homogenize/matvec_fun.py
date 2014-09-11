import numpy as np
from homogenize.trigonometric import TrigPolynomial


def get_inverse(A):
    """
    It calculates the inverse of conductivity coefficients at grid points,
    i.e. of matrix A_GaNi
    """
    B = np.copy(A)
    N = np.array(B[0][0].shape)
    d = N.size
    invA = np.eye(d).tolist()
    for m in np.arange(d):
        Bdiag = np.copy(B[m][m])
        B[m][m] = np.ones(N)
        for n in np.arange(m+1, d):
            B[m][n] = B[m][n]/Bdiag
        for n in np.arange(d):
            invA[m][n] = invA[m][n]/Bdiag
        for k in np.arange(m+1, d):
            Bnull = np.copy(B[k][m])
            for l in np.arange(d):
                B[k][l] = B[k][l] - B[m][l]*Bnull
                invA[k][l] = invA[k][l] - invA[m][l]*Bnull
    for m in np.arange(d-1, -1, -1):
        for k in np.arange(m-1, -1, -1):
            Bnull = np.copy(B[k][m])
            for l in np.arange(d):
                B[k][l] = B[k][l] - B[m][l]*Bnull
                invA[k][l] = invA[k][l] - invA[m][l]*Bnull
    invA = np.array(invA)
    return invA


def enlarge(xN, M):
    xM = np.zeros(M, dtype=xN.dtype)
    M = np.array(M)
    N = np.array(np.shape(xN))
    if np.allclose(M, N):
        return xN
    dim = np.size(N)
    ibeg = (M-N+(N % 2))/2
    iend = (M+N+(N % 2))/2
    if dim == 2:
        xM[ibeg[0]:iend[0], ibeg[1]:iend[1]] = xN
    elif dim == 3:
        xM[ibeg[0]:iend[0], ibeg[1]:iend[1], ibeg[2]:iend[2]] = xN
    return xM


def enlarge_M(xN, M):
    M = np.array(M, dtype=np.int32)
    N = np.array(xN.shape[2:])
    if np.allclose(M, N):
        return xN
    dim = np.size(M)
    xM = np.zeros(np.hstack([dim, dim, M]))
    for m in np.arange(dim):
        for n in np.arange(dim):
            xM[m][n] = enlarge(xN[m][n], M)
    return xM


def decrease(xM, N):
    N = np.array(N, dtype=np.int32)
    M = np.array(xM.shape, dtype=np.int32)
    dim = M.size
    ibeg = (M-N+(N % 2))/2
    iend = (M+N+(N % 2))/2
    if dim == 2:
        xN = xM[ibeg[0]:iend[0], ibeg[1]:iend[1]]
    elif dim == 3:
        xN = xM[ibeg[0]:iend[0], ibeg[1]:iend[1], ibeg[2]:iend[2]]
    return xN


def get_Nodd(N):
    Nodd = N - ((N + 1) % 2)
    return Nodd
