import numpy as np 
def PCA_matrix_input(C, N, M, k):
		r"""
		Partially pivoted Cross Approximation. A low rank approximation algorithm that approximates an N-by-N SPSD matrix :math:`C` with an N-by-k matrix :math:`A` and a k-by-M matrix :math:`B`  so that :math:`C` and :math:`A\;B` is rou
ghly equal. It also gives an error estimate in the output.

		The method returns matrices *A* and *B*, together with an error estimate *err* 
  
           :param C: A matrix 
        	:type C: float   
		:param N: Number of rows of C
		:type N: integer
		:param M: Number of columns of C
		:type M: integer
		:param k: Rank of approximation, k <= N
		:type k: integer
  
		:rtype: tuple (np.ndarray (2D), np.ndarray (2D) , float)
		"""
        
		A = np.zeros((N, k))
		B = np.zeros((k, M))
		Pi = np.array([])
		Pj = np.array([])
		
		istar = 0
		i_list = np.arange(0, N, 1)
		j_list = np.arange(0, M, 1)  
		
		for i in range(0, k):
		    
			j_list_small = np.setdiff1d(j_list, Pj)
			Row = C[istar,:]
			jstar = np.argmax(abs(Row[j_list_small]))
			jstar = j_list_small[jstar]
			
			Pj = np.hstack((Pj, jstar)) 
			
			if i > 0:
				max_value = Row[jstar] - np.dot(A[istar, 0:i], B[0:i, jstar])
			
			else:
				max_value = Row[jstar]
			
			if max_value == 0:
				break
			Column = C[:,jstar] 
			A[:, i] = Column - np.dot(A[:, 0:i], B[0:i, jstar])
			B[i, :] = (Row - np.dot(A[istar, 0:i], B[0:i, :])) / max_value
			
			Pi = np.hstack((Pi, istar))             
			i_list_small = np.setdiff1d(i_list, Pi)
			
			if i_list_small.shape[0] < 1:
				break
			istar = np.argmax(abs(Column[i_list_small]))
			istar = i_list_small[istar]
		
		err = np.linalg.norm(A[:, i]) * np.linalg.norm(B[i, :]) 
		return A, B, err
