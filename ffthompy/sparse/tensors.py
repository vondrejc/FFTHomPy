import sys
import numpy as np
import numpy.fft as fft
from scipy.linalg import block_diag
from ffthompy.tensors import TensorFuns
import itertools

sys.path.append("/home/disliu/ffthompy-sparse")


def multiply(A, B, *args, **kwargs):
    """element-wise (Hadamard) product of A and B"""
    dim=A.__len__()
    assert(dim==B.__len__())
    C=[]
    for ii in range(dim):
        shape=(A[ii].shape[0]*B[ii].shape[0], A[ii].shape[1])
        val=np.empty(shape)
        for iimn, (mm, nn) in enumerate(itertools.product(range(A[ii].shape[0]), range(B[ii].shape[0]))):
            val[iimn] = A[ii][mm]*B[ii][nn]
        C.append(val)
    return C


class SparseTensorFuns(TensorFuns):
    pass

class CanonicalTensor():
    pass

class TuckerTensor():
    pass

if __name__=='__main__':
    # check multiplication
    r=2
    n=50
    m=55
    As = [np.random.random([r,n]), np.random.random([r, m])]
    A = np.einsum('ij,ik->jk', As[0], As[1])
    Bs = [np.random.random([r,n]), np.random.random([r, m])]
    B = np.einsum('ij,ik->jk', Bs[0], Bs[1])
    C0 = A*B
    C1s = multiply(As, Bs)
    C1 = np.einsum('ij,ik->jk', C1s[0], C1s[1])
    print(np.linalg.norm(C0-C1))
    print('END')
