import numpy as np


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
#     if k!=k_actual:
#         print "The actual rank k is %d in stead of %d"&(k_actual, k)

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

    # my_tol = N*my_epsilon(max(diagonal))
    # if tol<my_tol:
    #    tol=my_tol
    # e1 = my_epsilon(max(diagonal))
    e1=my_epsilon(M.mean())

    for i in range(N):
        p=diagonal.argmax()

        # max_ind[i] = i  # this produce the same result as outer product Cholesky algorithm in Golub and Loan's book (algorithm 4.2.2, page 145 and 148)
        dia_max=diagonal[p]

        # print diagonal
        # print max_ind[0:i+1]

        my_tol=max((i+1)*e1, tol)
        # my_tol = max(N*e1, tol)
        if dia_max<=my_tol : # or (max_ind[i] in max_ind[0:i]):
            normal_exit=False
            break

        # start= time.clock()
        # A[:, i] = np.reshape( (M[:,max_ind[i]] - np.dot(A[:, :i], A[max_ind[i], :i].T))/math.sqrt(dia_max),(N,))
        # A[:, i] =   (M[:,max_ind[i]] - np.dot(A[:, :i], A[max_ind[i], :i].T))/ sqrt(dia_max)
        A[:, i]=(M[:, p]-np.dot(A[:, :i], A[p, :i].T))/np.sqrt(dia_max)
        A[max_ind[0:i], i]=0
        # t1= t1+(time.clock() - start)

        # start= time.clock()
        # diagonal = diagonal - np.power(A[:, i], 2)

        diagonal=diagonal-A[:, i]**2

        diagonal[p]=0
        diagonal[diagonal<0]=0
        # t2= t2+(time.clock() - start)

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
    # print "t1 my", t1
    # print "t2 my", t2

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


if __name__=='__main__':
    print('END')
