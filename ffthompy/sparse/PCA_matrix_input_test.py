
"""
This is to test the PCA module
"""

import numpy as np

import PCA_matrix_input

import time
from time import clock

#######################################################################################################
if __name__=="__main__":

    N=70
    M=90
    x=np.linspace(-np.pi, np.pi, M)
    y=np.linspace(-np.pi, np.pi, N)

    S=np.sin(x[np.newaxis, :]+y[:, np.newaxis])*(x[np.newaxis, :]+y[:, np.newaxis])

    # S =S + 0.001*np.random.rand(N, M)
    # S = np.random.rand(N, M)

#    print S
#
#    print get_column(0,x,y)
#    print get_column(2,x,y)
#
#    print get_row(0,x,y)
#    print get_row(2,x,y)

    k=60

    # A, B, err = PCA.PCA(N, M, k, get_column, get_row, x, y)
    A2, B2, k_actual, err2=PCA_matrix_input.PCA_matrix_input(S, N, M, k, tol=0.1)
    # print A
    # print B

#    #print "diff is \n", S - np.dot(A, B)
#    print "max_err is \n", np.amax(abs(S - np.dot(A, B)))
#
#    print "Frobenius norm of err is  ", np.linalg.norm(S - np.dot(A, B))
#    print "error estimate is ", err

    print
    print
    # print "diff is \n", S - np.dot(A2, B2)
    print "max_err is \n", np.amax(abs(S-np.dot(A2, B2)))

    print "Frobenius norm of err is  ", np.linalg.norm(S-np.dot(A2, B2))
    print "error estimate is ", err2
    print "actual k is ", k_actual

    """
    ###################################################################################

    start = time.clock()

    t1= time.clock() - start

    """
