import numpy as np
from numpy.linalg import svd, norm
from numpy import tensordot

def unfold(T, dim):
    """
    Unfolds a tensor T into a matrix, taking the dimension "dim" of T as the first dimension of the matrix,
    and flattening all the other dimensions into the other one dimension of the matrix.

    dim starts from 0.

    :param T: a tensor .
    :type T: numpy.ndarray
    :param dim: the dimension based on which the unfolding is made
    :type dim: int
    :returns: 2D numpy.array -- a matricisation of T.
    """
    Tm=np.moveaxis(T, dim, 0)
    return Tm.reshape(T.shape[dim],-1)

def nModeProduct(T, M, n):
    """
    n-Mode product of a tensor T  and a matrix M,  the summation is made along the nth dim.
    definition in paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE

    For example, n with value 0, 1, or 2, would specify the 1st, 2nd or 3rd dim of the tensor T.
    For the matrix M, this function always take the second dimension, as if to multiply T by M on the left side.

    :param T: a tensor .
    :type T: numpy.ndarray
    :param M: a matrix
    :type M: numpy.array
    :param n: serial number of a dimension of T along which the summation is made.
    :type n: int
    :returns: numpy.ndarray -- a result tensor .
    """

    P=tensordot(T, M, axes=([n], [1]))
    return np.rollaxis(P, len(T.shape)-1, n)

def subTensor(T, k=None, index=None):
    """
    extract a sub-tensor t = T[:k,:k,:k,...] or t = T[index].

    :param T: a tensor .
    :type T: numpy.ndarray
    :param k: a list of integer k, indices larger than k is to be truncated. if "index" presents, this argument is overridden.
    :type k: list
    :param index: a list of list of integer indicating extracted indices on every dimension.
    :type index: list
    :returns: numpy.ndarray -- a sub-tensor .
    """
    if k.any()==None and index.any()==None:
        return T
    elif index is not None:
        return T[np.ix_(*index)]
    else:
        if isinstance(k, int): # if only one integer is assigned to k
            k=k*np.ones((len(T.shape),), dtype=int)

        index=[None]*len(k)
        for i in range(len(k)):
            index[i]=range(k[i])

        return T[np.ix_(*index)]


def HOSVD(A, k=None, tol=None):
    """
    High order svd of d-dim tensor A. so that A = S (*1) u1 (*2) u2 (*3) u3 ... (*d) ud,
    "(*n)" means n-mode product. S is the core, u1,u2,u3, ... are orthogonal basis.
    definition in paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION"
    by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE

    :param A: a full tensor .
    :type A: numpy.ndarray

    :param k: rank for the truncation.
    :type k: numpy.list of integer or a single integer

    :param tol:  error torlerance of the truncation
    :type tol: numpy.float

    :returns: numpy.ndarray -- the core tensor S,
              numpy.list    -- a list of array containing basis
    """

    d=len(A.shape)

    if d==2:
        u, s, vt=svd(A, full_matrices=False)
        U=[u, vt.T]
        S=np.diag(s)
    else:
        U=[None]*d
        for j in range(0, d):
            U[j], s, vt=svd(unfold(A, j), full_matrices=False)

        S=A.copy()
        for i in range(0, d):
            S=nModeProduct(S, U[i].T, i)

    if k is not None:
        if isinstance(k, int): # if only one integer is assigned to k
            k=k*np.ones((len(A.shape),), dtype=int)

        S=subTensor(S, k=k)
        for j in range(0, d):
            U[j]=U[j][:, :k[j]]

    return S, U


if __name__=='__main__':

    # A = np.random.rand(7,7)
    A=np.arange(49).reshape((7, 7))
    print A

    S, U=HOSVD(A)
    print S
    print U
    print norm(A-np.dot(U[0], np.dot(S, U[1].T)))

    S2, U2=HOSVD(A, 5)
    print S2
    print U2
    print norm(A-np.dot(U2[0], np.dot(S2, U2[1].T)))
#    S,U=HOSVD2(A)
#    print S
#    print U
#    print norm(A-np.dot(U[0], np.dot(S, U[1].T)))

    print('END')
