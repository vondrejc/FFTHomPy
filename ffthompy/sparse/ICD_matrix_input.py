import numpy as np  
import math
import time
from time import clock 
import scipy
import scipy.linalg   # SciPy Linear Algebra Library

from math import sqrt

import array 
 

def my_epsilon(x):
 
    eps_1= np.finfo(np.float64).eps
    
    epsilon= 2**(np.rint(np.floor(np.log(x)/np.log(2)))) * eps_1
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
    
 
    N = M.shape[0] 
    
    diagonal = np.copy(np.diag(M))
    #ZEROS = np.zeros( diagonal.shape)  

    A = np.empty((N, N),dtype=np.float)
 
    max_ind = np.empty((N,))
    max_ind = max_ind.astype(int) 
    
    #max_ind = array.array('i',(0 for i in range(0,N)))
    normal_exit=True
    
    ind = range(N)
    
    #my_tol = N*my_epsilon(max(diagonal))
    #if tol<my_tol: 
    #    tol=my_tol       
    #e1 = my_epsilon(max(diagonal)) 
    e1 = my_epsilon(M.mean()) 
 
    for i in range(N):
        p=diagonal.argmax()
        
        #max_ind[i] = i  # this produce the same result as outer product Cholesky algorithm in Golub and Loan's book (algorithm 4.2.2, page 145 and 148)
        dia_max = diagonal[p]
        
        #print diagonal
        #print max_ind[0:i+1]
        
        
        my_tol = max((i+1)*e1, tol)
        #my_tol = max(N*e1, tol)
        if dia_max<= my_tol :#or (max_ind[i] in max_ind[0:i]):
            normal_exit = False
            break 
         
        #start= time.clock()                
        #A[:, i] = np.reshape( (M[:,max_ind[i]] - np.dot(A[:, :i], A[max_ind[i], :i].T))/math.sqrt(dia_max),(N,))   
        #A[:, i] =   (M[:,max_ind[i]] - np.dot(A[:, :i], A[max_ind[i], :i].T))/ sqrt(dia_max) 
        A[:, i] =   (M[:,p] - np.dot(A[:, :i], A[p, :i].T))/ sqrt(dia_max)   
        A[max_ind[0:i],i]=0
        #t1= t1+(time.clock() - start)  
 
        #start= time.clock() 
        #diagonal = diagonal - np.power(A[:, i], 2) 
 
        
        diagonal = diagonal - A[:, i]**2 
        
        
        diagonal[p]=0
        diagonal[diagonal<0]=0
        #t2= t2+(time.clock() - start)  
       
        max_ind[i]=p   
         
    
    if normal_exit:
        k=i +1
    else:
        k=i
   
    #A = A[:,0:k]        
    diff = np.setdiff1d(ind, max_ind[0:k]) 
    new_ind = np.hstack((max_ind[0:k], diff))        
    A = A[new_ind, :]   
    max_err = np.max(abs(diagonal))    
    #print "t1 my", t1
    #print "t2 my", t2    

    for i in range(k):
        if A[i,i]==0:
            A[i,i] = my_epsilon(M.mean()) 
            
    return A[:,0:k], k, max_err, new_ind  



if __name__ == "__main__": 
    
    """
    A=np.array([0,2,1])
    print (2 in A)
    
    B = np.array([1,2,3, -1, -2])
    
    print np.maximum(B, np.zeros((5,)))
    print range(6)
    
    B1 = np.array([1,2,3 ])
    B2 = np.array([4,5,6 ])
    
    C = np.vstack((B1,B2))
    
    print C
    print C[:,A]
    """
    N=500
    
    B = np.random.rand(N,N) 
    C = np.dot(B, B.T)
    
    w1, v1 = np.linalg.eigh(C)
    
    print w1
    
    
    start=  clock()  
    L,m,max_err,new_ind = ICD_matrix_input(C, 1e-1)   
    print "PCD time", clock() - start
    
    
    w2, v2 = np.linalg.eigh(np.dot(L,L.T))
    
    print w2
    
    print w1-w2
    
    print v1-v2    
    
    
    start=  clock()  
    L2 = scipy.linalg.cholesky(C, lower=True )  
    print "chol time", clock() - start
    
    print "C[new_ind, new_ind]-LL'\n", np.amax(abs(C[np.ix_(new_ind,new_ind)] -np.dot(L,L.T) ))
    print "m=",m
    
    print "error given by ICD\n", max_err
    print "C -L2L2'\n", np.amax(abs(C -np.dot(L2,L2.T) ))
	