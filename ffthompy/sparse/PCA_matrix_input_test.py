
"""
This is to test the PCA module
""" 

import numpy as np 
import PCA
import PCA_matrix_input
 

import time
from time import clock 
 

def func(x, y):     
    return np.sin(x[np.newaxis, :] + y[:, np.newaxis]) * (x[np.newaxis, :] + y[:, np.newaxis])

def get_row(i, x, y):    
    return np.sin(x + y[i]) * (x + y[i])    

def get_column(i, x, y):    
    return np.sin(x[i] + y) * (x[i] + y)  

    
#######################################################################################################
if __name__ == "__main__":
    
    N = 6
    M = 4  
    x = np.linspace(-np.pi, np.pi, M)
    y = np.linspace(-np.pi, np.pi, N)
    
    #S = np.sin(x[np.newaxis, :] + y[:, np.newaxis]) * (x[np.newaxis, :] + y[:, np.newaxis])
    
    S = np.random.rand(N, M)
    
#    print S
#   
#    print get_column(0,x,y)
#    print get_column(2,x,y)
#    
#    print get_row(0,x,y)
#    print get_row(2,x,y)
   
    
    k = 3
    
    #A, B, err = PCA.PCA(N, M, k, get_column, get_row, x, y)
    A2, B2, err2 = PCA_matrix_input.PCA_matrix_input(S, N, M, k)   
    # print A
    # print B
    
#    #print "diff is \n", S - np.dot(A, B)
#    print "max_err is \n", np.amax(abs(S - np.dot(A, B)))
#    
#    print "Frobenius norm of err is  ", np.linalg.norm(S - np.dot(A, B))
#    print "error estimate is ", err
    
    print
    print
    #print "diff is \n", S - np.dot(A2, B2)
    print "max_err is \n", np.amax(abs(S - np.dot(A2, B2)))
    
    print "Frobenius norm of err is  ", np.linalg.norm(S - np.dot(A2, B2))
    print "error estimate is ", err2
    
    """ 
    ###################################################################################
    
    start = time.clock() 
 
    t1= time.clock() - start    
 
    """
 
    
 
    
