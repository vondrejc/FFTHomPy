import numpy as np
from numpy.linalg import svd, norm
from numpy import dot, kron,newaxis, argsort, tensordot, rollaxis

def dCA_matrix_input(M, k, tol=1e-16):
    """
    diagonally pivoted Cross Approximation, a low rank approximation algorithm
    only work with symmetric positive semi-definite (SPSD) matrices, e.g. covariance matrices or correlation matrices

    it approximates a N-by-N SPSD matrix M with a N-by-k matrix A , so that M and A*A' is roughly equal, it also gives the maximum error in the output.

    input:
            M-- a SPSD matrix

            k -- integer, size of approximation, k <= N
            tol -- float, error tolerance in maximum norm

    output:

            A -- N-by-k matrix

            max_err -- maximum element-wise error of this approximation

            k_actual -- actual k based on tol

    """
    N=M.shape[0]

    diagonal=np.copy(np.diag(M))

    A=np.empty((N, k), dtype=np.float)

    for i in range(0, k):

        p=diagonal.argmax()

        dia_max=diagonal[p]

        A[:, i]=(M[:, p]-np.dot(A[:, :i], A[p, :i].T))/np.sqrt(dia_max)

        diagonal=diagonal-A[:, i]**2

        max_err=np.max(diagonal)
        # print max_err
        if max_err<=tol:
            break

    k_actual=i+1
    if k!=k_actual:
        print "The actual rank k is %d in stead of %d"%(k_actual, k)

    A=A[:, :k_actual]
    return A, k_actual, max_err

def my_epsilon(x):

    eps_1=np.finfo(np.float64).eps

    epsilon=2**(np.rint(np.floor(np.log(x)/np.log(2))))*eps_1
    return epsilon

def ICD_matrix_input(M, tol=0):
    r"""
    A modified Diagonally pivoted Cross Approximation algorithm that produce incomplete Cholesky decomposition.  only works with symmetric positive semi-definite (SPSD) matrices, e.g. covariance matrices or correlation matrices. It approximates an N-by-N SPSD matrix :math:`C` with an lower triangular matrix :math:`A` so that :math:`C` permuted according to a new index and :math:`A\;A^T` is roughly equal. It also gives the maximum error in the output.

    The method returns the lower triangular matrix *A* , rank of *A*, the maximum elementwise error, and a row index based on which *A* is permuted.

    :M: the source matrix
    :type M: np.ndarray
    :param tol: error tolerance
    :type tol: float
    :rtype: tuple (np.ndarray (2D) integer float np.ndarray (1D))
    """

    N=M.shape[0]

    diagonal=np.copy(np.diag(M))
    # ZEROS = np.zeros( diagonal.shape)

    A=np.empty((N, N), dtype=np.float)

    max_ind=np.empty((N,))
    max_ind=max_ind.astype(int)

    # max_ind = array.array('i',(0 for i in range(0,N)))
    normal_exit=True

    ind=range(N)

    e1=my_epsilon(M.mean())

    for i in range(N):
        p=diagonal.argmax()

        dia_max=diagonal[p]

        my_tol=max((i+1)*e1, tol)
        # my_tol = max(N*e1, tol)
        if dia_max<=my_tol : # or (max_ind[i] in max_ind[0:i]):
            normal_exit=False
            break

        A[:, i]=(M[:, p]-np.dot(A[:, :i], A[p, :i].T))/np.sqrt(dia_max)
        A[max_ind[0:i], i]=0

        diagonal=diagonal-A[:, i]**2

        diagonal[p]=0
        diagonal[diagonal<0]=0

        max_ind[i]=p

    if normal_exit:
        k=i+1
    else:
        k=i

    # A = A[:,0:k]
    diff=np.setdiff1d(ind, max_ind[0:k])
    new_ind=np.hstack((max_ind[0:k], diff))
    A=A[new_ind, :]
    max_err=np.max(abs(diagonal))

    for i in range(k):
        if A[i, i]==0:
            A[i, i]=my_epsilon(M.mean())

    return A[:, 0:k], k, max_err, new_ind

def PCA_matrix_input(C, N, M, k, tol=2.22e-16):
    r"""
    Partially pivoted Cross Approximation. A low rank approximation algorithm that approximates an N-by-N SPSD matrix :math:`C` with an N-by-k matrix :math:`A` and a k-by-M matrix :math:`B`  so that :math:`C` and :math:`A\;B` is rou
    ghly equal. It also gives an error estimate in the output.

    The method returns matrices *A* and *B*, together with an Frobenius error estimate *err*

    :param C: A matrix
    :type C: float
    :param N: Number of rows of C
    :type N: integer
    :param M: Number of columns of C
    :type M: integer
    :param k: Rank of approximation, k <= N
    :type k: integer
    :param tol: Frobenius error tolerance
    :type tol: float

    :rtype: tuple (np.ndarray (2D), np.ndarray (2D) , int, float)
    """

    A=np.zeros((N, k))
    B=np.zeros((k, M))
    Pi=np.array([])
    Pj=np.array([])

    istar=0
    i_list=np.arange(0, N, 1)
    j_list=np.arange(0, M, 1)

    for i in range(0, k):
        j_list_small=np.setdiff1d(j_list, Pj)
        Row=C[istar, :]
        jstar=np.argmax(abs(Row[j_list_small]))
        jstar=j_list_small[jstar]

        Pj=np.hstack((Pj, jstar))

        if i>0:
            max_value=Row[jstar]-np.dot(A[istar, 0:i], B[0:i, jstar])
        else:
            max_value=Row[jstar]
        if max_value==0:
            break
        Column=C[:, jstar]
        A[:, i]=Column-np.dot(A[:, 0:i], B[0:i, jstar])
        B[i, :]=(Row-np.dot(A[istar, 0:i], B[0:i, :]))/max_value

        Pi=np.hstack((Pi, istar))
        i_list_small=np.setdiff1d(i_list, Pi)

        if i_list_small.shape[0]<1:
            break
        istar=np.argmax(abs(Column[i_list_small]))
        istar=i_list_small[istar]

        err=np.linalg.norm(A[:, i])*np.linalg.norm(B[i, :])
        # print err
        if err<=tol:
            break

    k_actual=i+1

    if np.sum(np.abs(A[:, i]))<=2.22e-16 or np.sum(np.abs(B[i, :]))<=2.22e-16 :
        k_actual=k_actual-1

    if k!=k_actual:
        print "Warning: the output's actual rank k is %d in stead of %d"%(k_actual, k)
    A=A[:, :k_actual]
    B=B[:k_actual, :]

    return A, B, k_actual, err

def PCA(N, M, k, tol, get_column, get_row, *args):
    r"""
    Partially pivoted Cross Approximation. A low rank approximation algorithm that approximates an N-by-N SPSD matrix :math:`C` with an N-by-k matrix :math:`A` and a k-by-M matrix :math:`B`  so that :math:`C` and :math:`A\;B` is rou
    ghly equal. It also gives an error estimate in the output.

    The method returns matrices *A* and *B*, together with an error estimate *err*

    :param N: Number of rows of C
    :type N: integer
    :param M: Number of columns of C
    :type M: integer
    :param k: Rank of approximation, k <= N
    :type k: integer
    :param tol: error tolerance in Frobenius norm
    :type tol: float
    :param get_column: A function that returns the :math:`i^{\text{th}}` column of *C*. It receives as parameters the integer :math:`i` indicating the index of the targeted column
     and the arguments *\*args*.
    :type get_column: function
    :param get_row: A function that returns the :math:`i^{\text{th}}` row of *C*. It receives as parameters the integer :math:`i` indicating the index of the targeted row
     and the arguments *\*args*.
    :type get_row: function
    :param args: Arguments that are passed to the function *get_column* and *get_row* after the index :math:`i`.
    :rtype: tuple (np.ndarray (2D), np.ndarray (2D) , float)
    """

    A=np.zeros((N, k))
    B=np.zeros((k, M))
    Pi=np.array([])
    Pj=np.array([])

    istar=0
    i_list=np.arange(0, N, 1)
    j_list=np.arange(0, M, 1)

    for i in range(0, k):
        j_list_small=np.setdiff1d(j_list, Pj)
        Row=get_row(istar, *args)
        jstar=np.argmax(abs(Row[j_list_small]))
        jstar=j_list_small[jstar]

        Pj=np.hstack((Pj, jstar))
        Pi=np.hstack((Pi, istar))

        pivot=Row[jstar]-np.dot(A[istar, 0:i], B[0:i, jstar])

        if pivot==0:
            pivot=1e-16
        Column=get_column(jstar, *args)
        A[:, i]=Column-np.dot(A[:, 0:i], B[0:i, jstar])
        B[i, :]=(Row-np.dot(A[istar, 0:i], B[0:i, :]))/pivot

        i_list_small=np.setdiff1d(i_list, Pi)

        if i_list_small.shape[0]<1:
            break
        istar=np.argmax(abs(Column[i_list_small]))
        istar=i_list_small[istar]
        err=np.linalg.norm(A[:, i])*np.linalg.norm(B[i, :])
        if err<tol:
            break
    k_actual=i+1
    return A[:, :k_actual], B[:k_actual, :], k_actual, err

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
