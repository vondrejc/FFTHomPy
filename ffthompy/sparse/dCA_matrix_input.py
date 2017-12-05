import numpy as np  
from math import sqrt

def dCA_matrix_input(M,   k): 
    """
    diagonally pivoted Cross Approximation, a low rank approximation algorithm
    only work with symmetric positive semi-definite (SPSD) matrices, e.g. covariance matrices or correlation matrices  
    
    it approximates a N-by-N SPSD matrix M with a N-by-k matrix A , so that M and A*A' is roughly equal, it also gives the maximum error in the output.
    
    input:
            M-- a SPSD matrix

            k -- integer, size of approximation, k <= N
    
    output:

            A -- N-by-k matrix
 
            max_err -- maximum error of this approximation
            
    by Dishi Liu    
    Jan 2014
            
    """ 
    N = M.shape[0] 
    index=np.arange(N)
    
    diagonal = np.copy(np.diag(M))
    max_ind = np.empty((k,))
    A=np.empty((N, k),dtype=np.float)
 
    
    for i in range(0,k):

        p=diagonal.argmax()
 
        dia_max = diagonal[p]
 
        A[:, i] =   (M[:,p] - np.dot(A[:, :i], A[p, :i].T))/ sqrt(dia_max)   
        
        diagonal = diagonal - A[:, i]**2 
    
    max_err= np.max( diagonal ) 
    
    return A, max_err 

#######################################################################################################
#if __name__=="__main__":