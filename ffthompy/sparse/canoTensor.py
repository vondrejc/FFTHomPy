import sys
import numpy as np
sys.path.append("/home/disliu/ffthompy-sparse")

from ffthompy.sparse.tensors import SparseTensorFuns
from scipy.linalg import block_diag
import decompositions as dc


class CanoTensor(SparseTensorFuns):

    def __init__(self, name='', core=None, basis=None, Fourier=False,
                 r=3, N=[5,5], randomise=False):
        self.name=name
        self.Fourier=Fourier
        if core is not None and basis is not None:
            self.order=basis.__len__()
            self.basis=basis
            self.core=core
            self.r = basis[0].shape[0] # since all basis share the same r
            self.N=np.empty(self.order, dtype=np.int)
            for ii in range(self.order):
                self.N[ii]=basis[ii].shape[1]
        else:
            self.order=N.__len__()
            self.r=r
            self.N=N
            if randomise:
                self.randomise()
            else:
                self.core=np.zeros(r)
                self.basis=[np.zeros([self.r[ii],self.N[ii]]) for ii in range(self.order)]

    def randomise(self):
        self.core=np.diag(np.random.random((self.r,)))
        self.basis=[np.random.random([self.r,self.N[ii]]) for ii in range(self.order)]

    def __add__(self, Y):
        X=self
        core= np.hstack([X.core,Y.core])
        basis=[np.vstack([X.basis[ii],Y.basis[ii]]) for ii in range(self.order)]
        return CanoTensor(name=X.name+'+'+Y.name, core=core, basis=basis)

    def __neg__(self):
        return CanoTensor(core=-self.core, basis=self.basis)

    def __mul__(self, Y):
        "element-wise multiplication of two Tucker tensors"
        new_r=self.r*Y.r
        A=self.basis[0]
        B=self.basis[1]
        A2=Y.basis[0]
        B2=Y.basis[1]

        newA=np.zeros((new_r, self.N[0]))
        newB=np.zeros((new_r, self.N[1]))
        coeff=np.zeros((new_r,))

        for i in range(0, self.r):
            for j in range(0, Y.r):
                newA[i*Y.r+j, :]=A[i, :]*A2[j, :]
                newB[i*Y.r+j, :]=B[i, :]*B2[j, :]
                coeff[i*Y.r+j]=self.core[i]*Y.core[j]

        # # normalize the basis
        norm_A=np.linalg.norm(newA, axis=1)
        norm_B=np.linalg.norm(newB, axis=1)

        newA=newA/np.reshape(norm_A, (newA.shape[0], 1))
        newB=newB/np.reshape(norm_B, (newB.shape[0], 1))

        # put the normalizing coefficient into the coefficient vector
        coeff=coeff*(norm_A*norm_B)

        newBasis=[newA, newB]

        return (CanoTensor(name='a*b', core=coeff, basis=newBasis))

    def add(self, Y, tol=None, rank=None):
        return (self+Y).truncate(tol=tol, rank=rank)

    def multiply(self, Y, tol=None, rank=None):
        # element-wise multiplication
        return (self*Y).truncate(tol=tol, rank=rank)

    def full(self):
        "return a full tensor"
        if self.order==2:
            return np.einsum('i,ik,il->kl', self.core, self.basis[0],self.basis[1])
#             return(np.dot(np.dot(self.basis[0].T, self.core), self.basis[1]))
        else:
            raise NotImplementedError()

    def truncate(self, tol=None, rank=None):
        "return truncated tensor"

        basis=self.basis
        coeff=self.core

        ind=(-coeff).argsort() # the index of the core diagonal that sorts it into a descending order
        coeff=coeff[ind]

        if tol is None and rank is None:
            rank=self.r
        elif tol is not None:
            # determine the truncation rank so that (1.0-tol)*100% of the trace of the core is perserved.
            rank=np.searchsorted(np.cumsum(np.abs(coeff))/np.sum(np.abs(coeff)), 1.0-tol)+1

        # truncation
        core=coeff[:rank]

        for ii in range(self.order):
            basis[ii]=basis[ii][ind[:rank], :]

        return CanoTensor(name=self.name+'_truncated', core=core, basis=basis)

    def __repr__(self, full=False, detailed=False):
        keys=['name', 'N', 'Fourier', 'r']
        ss="Class : {0}({1}) \n".format(self.__class__.__name__, self.order)
        skip=4*' '
        nstr=np.array([key.__len__() for key in keys]).max()

        for key in keys:
            attr=getattr(self, key)
            if callable(attr):
                ss+='{0}{1}{3} = {2}\n'.format(skip, key, str(attr()), (nstr-key.__len__())*' ')
            else:
                ss+='{0}{1}{3} = {2}\n'.format(skip, key, str(attr), (nstr-key.__len__())*' ')

        return ss


if __name__=='__main__':
#    N=[10,20]
#    a = CanoTensor(name='a', r=3, N=N, randomise=True)
#    b = CanoTensor(name='b', r=3, N=N, randomise=True)
#    print(a)
#    print(b)
#    # addition
#    c = a+b
#    print(c)
#    c2 = a.full()+b.full()
#    print(np.linalg.norm(c.full()-c2))
#    # multiplication
#
#    c = a*b
#    c2 = a.full()*b.full()
#    print(np.linalg.norm(c.full()-c2))

    # DFT
    ########################################## test with "smoother" matices
    N=30
    M=20
    x=np.linspace(-np.pi, np.pi, M)
    y=np.linspace(-np.pi, np.pi, N)
    # creat matrix for test
    S1=np.sin(x[np.newaxis, :]+y[:, np.newaxis])*(x[np.newaxis, :]+y[:, np.newaxis])
    S2=np.cos(x[np.newaxis, :]-y[:, np.newaxis])*(x[np.newaxis, :]-y[:, np.newaxis])

    k=10
    # factorize the matrix
    A1, B1, k_actual, err=dc.PCA_matrix_input(S1, N, M, k)
    k1=k_actual

    A2, B2, k_actual, err=dc.PCA_matrix_input(S2, N, M, k)
    k2=k_actual


    #  nomalize the A and B
    dA1=np.linalg.norm(A1, axis=0)
    dB1=np.linalg.norm(B1, axis=1)

    A1u=A1/np.reshape(dA1, (1, A1.shape[1]))
    B1u=B1/np.reshape(dB1, (B1.shape[0], 1))

    # put the normalizing coefficient in C
    C1=dA1*dB1
#    temp=np.dot(A1u,C1)
#    ACB=np.dot(temp,B1u)
#    print(np.linalg.norm( ACB - np.dot(A1,B1) ))

    #  nomalize the A and B
    dA2=np.linalg.norm(A2, axis=0)
    dB2=np.linalg.norm(B2, axis=1)

    A2u=A2/np.reshape(dA2, (1, A2.shape[1]))
    B2u=B2/np.reshape(dB2, (B2.shape[0], 1))

    # put the normalizing coefficient in C
    C2=dA2*dB2

    # construct  canoTensors with the normalized basis and the corresponding coefficients core
    a=CanoTensor(name='a', core=C1, basis=[A1u.T, B1u])
    b=CanoTensor(name='b', core=C2, basis=[A2u.T, B2u])

    # addition
    c=a+b
    c2=a.add(b, tol=0.05)

    c_add=a.full()+b.full()
    print
    print "(a+b).full - (a.full+b.full)    = ", (np.linalg.norm(c.full()-c_add))
    print "add(a,b).full - (a.full+b.full) = ", (np.linalg.norm(c2.full()-c_add))

    # multiplication
    c=a*b
    c3=a.multiply(b, tol=0.01)

    c_mul=a.full()*b.full()
    print
    print  "(a*b).full - (a.full*b.full)         = ", (np.linalg.norm(c.full()-c_mul))
    print  "multiply(a,b).full - (a.full*b.full) = ", (np.linalg.norm(c3.full()-c_mul))
    # DFT
    print('testing DFT...')
    from ffthompy.operators import DFT
    Fa = a.fourier()
    print(Fa)
    Fa2 = DFT.fftnc(a.full(), a.N)
    print(np.linalg.norm(Fa.full()-Fa2))
    print('END')
