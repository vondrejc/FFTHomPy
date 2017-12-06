
"""
This is to test the dCA module
"""


import numpy as np 
 
import dCA_matrix_input

#import time
#from time import clock 

 
def cor_fun(x,y,theta): 
    """
    A correlation function of Gaussian type, 
    if x, y are d-by-N array, it returns a N-length vector as a column or row of the correlation matrix
    if x, y are d-by-N-by-1 and d-by-1-by-N array, it returns an N-by-N correlation matrix
    d is number of dimensions
    """
    return  np.exp(-np.sum( ((x-y)/theta)**2,axis=0)) 
#######################################################################################################
if __name__=="__main__":
      
  
    
    d=2  # number of dimension, could be any integer
    N=1000   # number if nodes, also the size of covariance or correlation matrix
    k=20    # the rank of the approximation
    
    X=np.random.rand(d,N) # generate N random points in a d-dimensional [0,1]^d domain
    theta=np.array([1, 2 ] ) # set correlation length
 
    # To check the accuracy, generate the true correlation matrix 
    # WARNING, this can take long time if the matrix is large
    C = cor_fun(X[:,:,np.newaxis],X[:,np.newaxis,:],theta[:,np.newaxis,np.newaxis])   
  
    
    A, k_actual, max_err=dCA_matrix_input.dCA_matrix_input(C,  k, tol=1e-5)  
    
    # and the approximate  matrix computed from A  
    C_approx = np.dot( A, A.T )
    
    # the maximum error
    true_max_err = np.amax(abs(C- C_approx))
    
    print "dCA giving maximum error %r:" % max_err
    print "true       maximum error %r:" % true_max_err
    print "size of   original matrix: %s" % str(C.shape)
    print "size of low rank matrices: %s " % str(A.shape)
    
 
    