#import sys
import numpy as np
#sys.path.append("/home/disliu/fft_new/ffthompy-sparse")

from ffthompy.sparse.tensors import SparseTensorFuns
from scipy.linalg import block_diag
import numpy.fft as fft

from numpy.linalg import svd, norm
from numpy import dot, kron,newaxis


def unfold(T, dim):
    """
    Unfolds a tensor T along dimension dim
    """
    return np.rollaxis(T, dim, 0).reshape(T.shape[dim], -1)
    
def nModeProduct(T, M, n):
    """
    n-Mode product of a tensor(3d for now) and a matrix,  sumation along the nth dim
    definition in paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE

    """
    # multiply along the nth dimension
    if n==1:
        return  np.einsum('ijk,li->ljk', T, M)
    elif n==2:
        return  np.einsum('jik,li->jlk', T, M)
    elif n==3:
        return  np.einsum('jki,li->jkl', T, M)
    else:
        pass 
  
def HOSVD(A):
    r"""
    High order svd of 3-dim tensor A. so that A = S (*1) u1 (*2) u2 (*3) u3, "(*n)" means n-mode product. 
    definition in paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE

    """ 
    A1=unfold(A, 0)
    
    u1,s1,vt1= svd( A1)
    u2,s2,vt2= svd( unfold(A, 1))
    u3,s3,vt3= svd( unfold(A, 2))
    
    S1 =dot(u1.T,A1)
    S1= dot(S1,kron(u2,u3))
    
    S = np.reshape(S1, A.shape)
    
    return S,u1,u2,u3

class Tucker(SparseTensorFuns):

    def __init__(self, name='', core=None, basis=None, Fourier=False,
                 r=[3,3], N=[5,5], randomise=False):
        self.name=name
        self.Fourier=Fourier
        if core is not None and basis is not None:
            self.order=basis.__len__()
            self.basis=basis
            self.core=core
            self.r=np.empty(self.order)
            self.N=np.empty(self.order)
            for ii in range(self.order):
                self.r[ii],self.N[ii]=basis[ii].shape
        else:
            self.order=r.__len__()
            self.r=np.array(r)
            self.N=np.array(N)
            if randomise:
                self.randomise()
            else:
                self.core=np.zeros(r)
                self.basis=[np.zeros([self.r[ii],self.N[ii]]) for ii in range(self.order)]

    def randomise(self):
        self.core=np.random.random(self.r)
        self.basis=[np.random.random([self.r[ii],self.N[ii]]) for ii in range(self.order)]

    def __add__(self, Y, tol=None, rank=None):
        X=self
        core= block_diag(X.core,Y.core)
        basis=[np.vstack([X.basis[ii],Y.basis[ii]]) for ii in range(X.order)]
        return Tucker(name=X.name+'+'+Y.name, core=core, basis=basis)

    def __neg__(self):
        return Tucker(core=-self.core, basis=self.basis)

    def __mul__(self, Y, tol=None, rank=None):
        "element-wise multiplication of two Tucker tensors"
        X = self
        new_r=X.r*Y.r
        A=X.basis[0]
        B=X.basis[1]
        A2=Y.basis[0]
        B2=Y.basis[1]

        newA=np.zeros((new_r[0], X.N[0]))
        newB=np.zeros((new_r[1], X.N[1]))
        for i in range(0, X.r[0]):
            for j in range(0, Y.r[0]):
                newA[i*Y.r[0]+j, :]=A[i, :]*A2[j, :]

        for i in range(0, X.r[1]):
            for j in range(0, Y.r[1]):
                newB[i*Y.r[1]+j, :]=B[i, :]*B2[j, :]

        newC=np.kron(X.core, Y.core)

        newBasis=[newA, newB]

        return (Tucker(name='a*b', core=newC, basis=newBasis))

    def full(self):
        "return a full tensor"
        if self.order==2:
            return np.einsum('ij,ik,jl->kl', self.core, self.basis[0],self.basis[1])
        elif self.order==3:
            # A = S (*1) u1 (*2) u2 (*3) u3, "(*n)" means n-mode product. 
            # from paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE
            temp= nModeProduct(self.core,self.basis[0].T,1)
            temp= nModeProduct(temp,self.basis[1].T,2)
            return nModeProduct(temp,self.basis[2].T,3)
        else:
            raise NotImplementedError()

    def truncate(self, tol=None, rank=None):
        "return truncated tensor"
        raise NotImplementedError()

    def __repr__(self, full=False, detailed=False):
        keys = ['name', 'Fourier', 'N', 'r']
        ss = "Class : {0}({1}) \n".format(self.__class__.__name__, self.order)
        skip = 4*' '
        nstr = np.array([key.__len__() for key in keys]).max()

        for key in keys:
            attr = getattr(self, key)
            if callable(attr):
                ss += '{0}{1}{3} = {2}\n'.format(skip, key, str(attr()), (nstr-key.__len__())*' ')
            else:
                ss += '{0}{1}{3} = {2}\n'.format(skip, key, str(attr), (nstr-key.__len__())*' ')

        return ss


if __name__=='__main__': 
    
    
    N=np.array([5,6])
    a = Tucker(name='a', r=np.array([2,3]), N=N, randomise=True)
    b = Tucker(name='b', r=np.array([4,5]), N=N, randomise=True)
    print(a)
    print(b)

    # addition
    c = a+b
    print(c)
    c2 = a.full()+b.full()
    print('testing addition...')
    print(np.linalg.norm(c.full()-c2))

    # multiplication
    c = a*b
    c2 = a.full()*b.full()
    print('testing multiplication...')
    print(np.linalg.norm(c.full()-c2))

    #DFT
    print('testing DFT...')
    from ffthompy.operators import DFT
    Fa = a.fourier()
    print(Fa)
    Fa2 = DFT.fftnc(a.full(), a.N)
    print(np.linalg.norm(Fa.full()-Fa2))
    
    
######### 3-d tenssor test #########
    print
    print('testing 3d tucker  ...')
    print
    
    N1=3
    N2=4
    N3=5
    x=np.linspace(-np.pi, np.pi, N1)
    y=np.linspace(-np.pi, np.pi, N2)
    z=np.linspace(-np.pi, np.pi, N3)
    
    # creat 3d tensor for test
    T=np.sin(x[:,newaxis,newaxis]+y[newaxis,:, newaxis] + z[ newaxis, newaxis, :] )
    
    #decompose the tensor into core and orthogonal basis by HOSVD
    S,u1,u2,u3 = HOSVD(T)
    
    #creat tucker format tensor
    a = Tucker(name='a', core=S, basis=[u1.T, u2.T, u3.T] )  
    print(a)    
    
    print('testing 3d tucker representation error...')
    print "a.full - T = ",  norm(a.full()-T)
    
    
    print('END')