import unittest
import numpy as np 

from decompositions import PCA

def func(x, y):     
    return np.sin(x[np.newaxis, :] + y[:, np.newaxis]) * (x[np.newaxis, :] + y[:, np.newaxis])

def get_row(i, x, y):    
    return np.sin(x + y[i]) * (x + y[i])    

def get_column(i, x, y):    
    return np.sin(x[i] + y) * (x[i] + y)  

#####################################################

class Test_decompositions(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_PCA(self):
        N = 160
        M = 170   
        x = np.linspace(-np.pi, np.pi, M)
        y = np.linspace(-np.pi, np.pi, N)
        
        S = np.sin(x[np.newaxis, :] + y[:, np.newaxis]) * (x[np.newaxis, :] + y[:, np.newaxis])
      
        k = 20
        tol=1e-6
        
        ## S = A*B with error tol
        A, B, k_actual, err = PCA(N, M, k, tol, get_column, get_row, x, y)
        
        print "maximum element-wise error is ", np.amax(abs(S - np.dot(A, B)))
        
        print "Frobenius norm of err is      ", np.linalg.norm(S - np.dot(A, B))
        print "PCA gives error estimate      ", err
        print "PCA returns actual k          ", k_actual

 
if __name__ == "__main__":
    unittest.main()
