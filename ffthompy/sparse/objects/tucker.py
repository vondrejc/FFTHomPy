import numpy as np

from ffthompy.sparse.objects.canoTensor import CanoTensor
from ffthompy.tensors.operators import DFT
from ffthompy.sparse.decompositions import HOSVD, nModeProduct

from numpy.linalg import norm
from numpy import newaxis

#np.set_printoptions(precision=3)
#np.set_printoptions(linewidth=999999)

class Tucker(CanoTensor):

    def __init__(self, name='unnamed', val=None, core=None, basis=None, Fourier=False, orthogonal=False,
                 r=None, N=None, randomise=False):

        self.name=name
        self.Fourier=Fourier
        self.orthogonal=orthogonal

        if val is not None:
            if len(val.shape)==2: # if 2D, use CanoTensor instead
                self.__class__=CanoTensor
                CanoTensor.__init__(self, name=name, val=val, Fourier=Fourier, orthogonal=orthogonal)
            else:
                self.core, self.basis=HOSVD(val, k=r)
                for i in range(0, len(self.basis)):
                    self.basis[i]=self.basis[i].T

                self.order=self.basis.__len__()

                self.N=np.empty(self.order).astype(int)
                for ii in range(self.order):
                    self.N[ii]=self.basis[ii].shape[1]

                self.r=np.empty(self.order).astype(int)
                for ii in range(self.order):
                    self.r[ii]=self.basis[ii].shape[0]

        elif core is not None and basis is not None:
            self.order=basis.__len__()
            self.basis=basis

            self.N=np.empty(self.order).astype(int)
            for ii in range(self.order):
                self.N[ii]=self.basis[ii].shape[1]

            if self.order==2: # if 2D, use CanoTensor instead
                self.__class__=CanoTensor
                CanoTensor.__init__(self, name=name, core=core, basis=basis, Fourier=Fourier, orthogonal=orthogonal)
            else:
                self.r=np.empty(self.order).astype(int)
                for ii in range(self.order):
                    self.r[ii]=basis[ii].shape[0]

                if len(core.shape)==1:
                    self.core=np.diag(core)
                else:
                    self.core=core
        else:
            if N.any()==None:
                self.N=np.array([5,5,5])
            else:
                self.N=N

            if r.any()==None:
                self.r =np.array([3,3,3])
            else:
                self.r=r

            self.order=self.r.__len__()

            if self.order==2: # if 2D, use CanoTensor instead
                self.__class__=CanoTensor
                CanoTensor.__init__(self, name=name, Fourier=Fourier, orthogonal=orthogonal, r=min(r), randomise=randomise)
            else:
                self.core=np.zeros(self.r)
                self.basis=[np.zeros([self.r[ii], self.N[ii]]) for ii in range(self.order)]

            if randomise:
                self.randomise()

    def randomise(self):
        if self.order==2:
            self.core=np.random.random((self.r,))
            self.basis=[np.random.random([self.r, self.N[ii]]) for ii in range(self.order)]
        else:
            self.core=np.random.random(self.r)
            self.basis=[np.random.random([self.r[ii], self.N[ii]]) for ii in range(self.order)]

    def __add__(self, Y):
        """element-wise addition of two Tucker tensors"""
        assert((self.N==Y.N).any())
        X=self

        r_new=X.r+Y.r

        if X.order==3:
            core=np.zeros((r_new[0], r_new[1], r_new[2]))
            core[:X.r[0], :X.r[1], :X.r[2]]=X.core
            core[X.r[0]:, X.r[1]:, X.r[2]:]=Y.core
        else:
            raise NotImplementedError("currently only support two and three dimensional tensor")

        newBasis=[np.vstack([X.basis[ii], Y.basis[ii]]) for ii in range(X.order)]

        result=Tucker(name=self.name+'+'+Y.name, core=core, basis=newBasis, orthogonal=False, Fourier=self.Fourier)

        return result.truncate(rank=self.N)

    def __mul__(self, Y):
        """element-wise multiplication of two Tucker tensors"""

        if isinstance(Y, float) or isinstance(Y, int) :
            R=self.copy()
            R.core=self.core*Y
            return R
        elif isinstance(Y, np.ndarray) and Y.shape==(1,):
            R=self.copy()
            R.core=self.core*Y[0]
            return R
        else:
            assert((self.N==Y.N).any())

            newCore=np.kron(self.core, Y.core)
            newBasis=[None]*self.order
            for d in range(0, self.order):
                # newBasis[d]= np.zeros((new_r[d], X.N[d]))
                newBasis[d]=np.multiply(self.basis[d][:, newaxis, :], Y.basis[d][newaxis, :, :])
                newBasis[d]=np.reshape(newBasis[d], (-1, self.N[d]))

            return Tucker(name=self.name+'*'+Y.name, core=newCore, basis=newBasis,Fourier=self.Fourier).truncate(rank=self.N)

    def orthogonalise(self):
        """re-orthogonalise the basis"""
        if self.orthogonal:
            return self
        else:
            newBasis=[]
            core=self.core
            # orthogonalise the basis
            for i in range(0, self.order):
                Q, R=np.linalg.qr(self.basis[i].T, 'reduced')
                newBasis.append(Q.T)
                core=nModeProduct(core, R.real, i)

            S, U=HOSVD(core)
            for i in range(0, self.order):
                newBasis[i]=np.dot(U[i].T, newBasis[i])

            return Tucker(name=self.name, core=S, basis=newBasis, orthogonal=True, Fourier=self.Fourier)

    def full(self):
        """convert a tucker representation to a full tensor
        A = CORE (*1) Basis1 (*2) Basis2 (*3) Basis3 ..., with (*n)  means n-mode product.
        from paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE
        """
        # if core is a single scalar value, make it in a n-D array shape
        if np.prod(self.core.shape)==1:
            self.core=np.reshape(self.core, tuple(self.r))

        d=self.N.shape[0]

        res=self.core.copy()
        for i in range(d):
            res=nModeProduct(res, self.basis[i].T, i)
        return res

    def truncate(self, tol=None, rank=None):
        """return truncated tensor. tol, if presented, would override rank as truncation criteria.
        """
        if np.any(tol) is None and np.any(rank) is None:
            # print ("Warning: No truncation criteria input, truncation aborted!")
            return self
        elif np.any(tol) is None and np.all(rank>=self.r)==True :
            # print ("Warning: Truncation rank not smaller than the original ranks, truncation aborted!")
            return self

        if isinstance(rank, int) :
            rank=rank*np.ones((self.order,), dtype=int)
            sg=np.sign(rank)
            rank=np.minimum(abs(rank), self.r)
            rank*=sg
        if isinstance(tol, float) :
            tol=tol*np.ones((self.order,))

        self=self.orthogonalise()
        #self=self.sort()

        basis=list(self.basis)
        core=self.core

        # to determine the rank of truncation
        if np.any(tol) is not None:
            rank=np.zeros((self.order), dtype=int)
            # determine the truncation rank so that (1.0-tol)*100% of the norm of the core in that direction  is perserved.
            if self.order==3:
                sorted_norm_d0=np.sum(abs(core), axis=(1, 2))
                sorted_norm_d1=np.sum(abs(core), axis=(0, 2))
                sorted_norm_d2=np.sum(abs(core), axis=(0, 1))

                rank[0]=np.searchsorted(np.cumsum(sorted_norm_d0)/np.sum(sorted_norm_d0), 1.0-tol[0])+1
                rank[1]=np.searchsorted(np.cumsum(sorted_norm_d1)/np.sum(sorted_norm_d1), 1.0-tol[1])+1
                rank[2]=np.searchsorted(np.cumsum(sorted_norm_d2)/np.sum(sorted_norm_d2), 1.0-tol[2])+1

        if self.order==3:
            if rank[0]<0 :
                target_storage=np.prod(-rank)

                R=np.array([1, 1, 1])
                storage=1

                while storage<=target_storage:

                    newCore=core[:R[0], :R[1], :R[2]]
                    newCore_enlarged=core[:min(self.r[0], R[0]+1), :min(self.r[1], R[1]+1), :min(self.r[2], R[2]+1)]

                    sorted_norm_d0=np.sum(abs(newCore_enlarged), axis=(1, 2))
                    sorted_norm_d1=np.sum(abs(newCore_enlarged), axis=(0, 2))
                    sorted_norm_d2=np.sum(abs(newCore_enlarged), axis=(0, 1))

                    candi=np.zeros((3,))

                    if R[0]<self.r[0]:
                        candi[0]=sorted_norm_d0[R[0]]
                    else:
                        candi[0]=-999

                    if R[1]<self.r[1]:
                        candi[1]=sorted_norm_d1[R[1]]
                    else:
                        candi[1]=-999

                    if R[2]<self.r[2]:
                        candi[2]=sorted_norm_d2[R[2]]
                    else:
                        candi[2]=-999

                    if candi[0]>=candi[1] and candi[0]>=candi[2]:
                        R[0]+=1
                    elif candi[1]>=candi[0] and candi[1]>=candi[2]:
                        R[1]+=1
                    elif candi[2]>=candi[0] and candi[2]>=candi[1]:
                        R[2]+=1

                    storage=np.prod(R)

                rank=np.array(newCore.shape)
                core=newCore
            else:
                core=core[:rank[0], :rank[1], :rank[2]]
        else:
            raise NotImplementedError("currently only support two and three dimensional tensor")

        for ii in range(self.order):
            basis[ii]=basis[ii][:rank[ii], :]

        return Tucker(name=self.name+'_truncated', core=core, basis=basis, orthogonal=True, Fourier=self.Fourier)

    def norm(self, ord='core'):

        if ord=='fro':
            R=self*self.conj()
            val=0.
            L=[None]*R.order
            for d in range(R.order):
                L[d]=range(R.r[d])

            ind=np.meshgrid(*L)
            ind=np.array(ind)
            ind=np.reshape(ind, (R.order,-1))

            for i in range(ind.shape[1]):
                val_piece=R.core[tuple(ind[:, i])]
                for k in range(R.order):
                    val_piece*=np.sum(R.basis[k][ind[k, i]]).real

                val+=val_piece

            val=val**0.5
        elif ord==1:
            pass
        elif ord=='inf':
            pass
        elif ord=='core':
            if not self.orthogonal:
                self=self.orthogonalise()
            return np.linalg.norm(self.core)
        else:
            raise NotImplementedError()

        return val

    def mean(self): 
        val=0.

        L=[None]*self.order
        for d in range(self.order):
            L[d]=range(self.r[d])

        ind=np.meshgrid(*L)
        ind=np.array(ind)
        ind=np.reshape(ind, (self.order,-1))

        for i in range(ind.shape[1]):
            val_piece=self.core[tuple(ind[:, i])]
            for k in range(self.order):
                if self.Fourier:
                    val_piece*=self.basis[k][ind[k, i],self.mean_index()[k]].real
                else:
                    val_piece*=np.mean(self.basis[k][ind[k, i]])

            val+=val_piece
        return val 

    @property
    def memory(self):
        "return the number of floating point numbers that consist of the tucker tensor"
        return np.prod(self.r)+sum(self.r*self.N)

if __name__=='__main__':

    print
    print('----testing "repeat" function ----')
    print
#    n=6
#    T1= np.zeros((n,n))
#    T1[n/3:2*n/3, n/3:2*n/3]=1

    # #this is a rank-2 tensor
    N1=4
    N2=5
    T1=np.arange(N1*N2)
    T1=np.reshape(T1, (N1, N2))

    # T1 = np.random.random((5,7 ))
    S, U=HOSVD(T1) #
    # creat tucker format tensor
    for i in range(0, len(U)):
        U[i]=U[i].T

    a=Tucker(name='a', core=S, basis=U, orthogonal=True)

    print(a)

    T2=np.random.random((N1, N2))
    S, U=HOSVD(T2) #
    # creat tucker format tensor
    for i in range(0, len(U)):
        U[i]=U[i].T

    b=Tucker(name='b', core=S, basis=U, orthogonal=True)

    print(b)

    c=a*b

    c2=T1*T2

    print('testing multiplication...')
    print(np.linalg.norm(c.full()-c2))
    print(np.linalg.norm(c.full()-c2)/np.linalg.norm(c2))

    b=a.repeat(np.array([8, 10]))
    print (b)

    af=a.fourier()
    af2=af.enlarge([5, 7])
    a2=af2.fourier()

    a3=a.project([5, 7])

    print
    print('----testing tucker basic operations ----')
    print
    N=np.array([3, 4])
    a=Tucker(name='a', r=np.array([3, 4]), N=N, randomise=True)
    b=Tucker(name='b', r=np.array([5, 6]), N=N, randomise=True)
    print(a)
    print(b)

    # addition
    c=a+b
    print(c)

    c2=a.full()+b.full()
    print('testing addition...')
    print(np.linalg.norm(c.full()-c2)/np.linalg.norm(c2))

    c_ortho=c.orthogonalise()
    print('testing addition and then orthogonalise ...')
    print(np.linalg.norm(c.full()-c_ortho.full())/np.linalg.norm(c_ortho.full()))
    print

    # multiplication
    c=a*b

    c2=a.full()*b.full()
    print c
    print('testing multiplication...')
    print(np.linalg.norm(c.full()-c2)/np.linalg.norm(c2))

    # DFT
    print('testing DFT...')

    Fa=a.fourier()
    print(Fa)
    Fa2=DFT.fftnc(a.full(), a.N)
    print(np.linalg.norm(Fa.full()-Fa2))

######### 3-d tenssor test #########

    print
    print('----testing 3d tucker ----')
    print

    N1=10 # warning: in 3d multiplication too large N number could kill the machine.
    N2=20
    N3=30

    # creat 3d tensor for test

#    x=np.linspace(-np.pi, np.pi, N1)
#    y=np.linspace(-np.pi, np.pi, N2)
#    z=np.linspace(-np.pi, np.pi, N3)
#    #this is a rank-1 tensor
#    T=np.sin(x[:,newaxis,newaxis]+y[newaxis,:, newaxis] + z[ newaxis, newaxis, :] )

    # this is a rank-2 tensor
    T=np.arange(N1*N2*N3)
    T=np.reshape(T, (N1, N2, N3))

    # a full rank tensor
    # T=np.random.random((N1, N2, N3))

#    #decompose the tensor into core and orthogonal basis by HOSVD
    # S2,U2 = HOSVD2(T)
#
#    #creat tucker format tensor
#    a = Tucker(name='a', core=S, basis=[u1.T, u2.T, u3.T], orthogonal=True )
#    print(a)
#
#    print('testing 3d tucker representation error...')
#    print "a.full - T = ",  norm(a.full()-T)

#    #decompose the tensor into core and orthogonal basis by HOSVD
#
#    t1=timeit.timeit("S,U = HOSVD(T)", setup='from __main__ import HOSVD, T', number=20)
#
#    t2=timeit.timeit("S2,U2 = HOSVD2(T)", setup='from __main__ import HOSVD2, T', number=20)
#
#    #t3=timeit.timeit("S3,U3 = HOSVD3(T)", setup='from __main__ import HOSVD3, T', number=10)
#
#    print
#    print "1st HOSVD costs: %f"%t1
#    print "2nd HOSVD costs: %f"%t2
#    #print "3rd HOSVD costs: %f"%t3



    S, U=HOSVD(T)
    # creat tucker format tensor

    for i in range(0, len(U)):
        U[i]=U[i].T

    a=Tucker(name='a', core=S, basis=U, orthogonal=True)
    print(a)


    print('testing nd tucker representation error...')
    print "a.full - T = ", norm(a.full()-T)
#

#    S3,U3 = HOSVD3(T)
#    basis = U3
#    for i in range(0,len(basis)):
#        basis[i] = basis[i].T
#
#    a3 = Tucker(name='a3', core=S3, basis=basis, orthogonal=True )
#    print(a3)
#
#
#    print('testing nd tucker representation error...')
#    print "a3.full - T = ",  norm(a3.full()-T)
    # print sss
#    #decompose the tensor into core and orthogonal basis by HOSVD
#    T2=np.sin(T)

    # this is a rank-2 tensor
    T2=np.sqrt(T)
    S2, U2=HOSVD(T2)

    basis=U2
    for i in range(0, len(basis)):
        basis[i]=basis[i].T
    # creat tucker format tensor
    b=Tucker(name='b', core=S2, basis=basis, orthogonal=True)
    print(b)


#    b_trunc=b.truncate(rank=[5, 18, 15])
#    print(b_trunc)
#
#    print('testing 3d tucker core sorting and  rank-based truncation ...')
#    print "b_truncated.full - b.full = ", norm(b_trunc.full()-b.full())
#    print "norm(b_truncated.full - b.full)/norm(b.full) = ", norm(b_trunc.full()-b.full())/norm(b.full())
#
#    b_trunc=b.truncate(rank=9)
#    print(b_trunc)
#
#    print('testing 3d tucker core sorting and  rank-based truncation ...')
#    print "b_truncated.full - b.full = ", norm(b_trunc.full()-b.full())
#    print "norm(b_truncated.full - b.full)/norm(b.full) = ", norm(b_trunc.full()-b.full())/norm(b.full())
#
#    b_trunc=b.truncate(rank=-9)
#    print(b_trunc)
#
#    print('testing 3d tucker core sorting and  rank-based truncation ...')
#    print "b_truncated.full - b.full = ", norm(b_trunc.full()-b.full())
#    print "norm(b_truncated.full - b.full)/norm(b.full) = ", norm(b_trunc.full()-b.full())/norm(b.full())
#
#    print


#    print ss
#
#    b_trunc=b.truncate(tol=[1e-6, 1e-6, 1e-6])
#    print(b_trunc)
#
#    print('testing 3d tucker  tol-based truncation ...')
#    print "b_truncated.full - b.full = ", norm(b_trunc.full()-b.full())
#    print "norm(b_truncated.full - b.full)/norm(b.full) = ", norm(b_trunc.full()-b.full())/norm(b.full())
#    print
#
#    c=a+b
#    print(c)
#    print
#
#    print('testing 3d re-orthogonalization after addition ...')
#    print "(a+b).full - (a.full+b.full) = ", norm(c.full()-a.full()-b.full())
#    print
#
#
#
    # multiplication
    T1=np.random.rand(N1,N2)
    T2=np.random.rand(N1,N2)
    
    #T1=np.sin(T )
    
    a = Tucker(val=T1,name='a' )
    b = Tucker(val=T2,name='b' )
    c=a*b
    print(c)

    print('testing 3d multiplication and re-orthogonalization...')
    print "(a*b).full - (a.full*b.full) = ", norm(c.full()-a.full()*b.full())
    print "((a*b).full - (a.full*b.full))/|(a.full*b.full)| = ", norm(c.full()-a.full()*b.full())/norm(a.full()*b.full())
    print "max((a*b).full - (a.full*b.full))/mean(a.full*b.full) = ", np.max(c.full()-a.full()*b.full())/np.mean(a.full()*b.full())
#
    c_trunc=c.truncate(rank=13)
    print(c_trunc)

    print('testing 3d tucker core sorting and  rank-based truncation ...')
    print "c_truncated.full - c.full = ", norm(c_trunc.full()-T2*T1)
#    print "norm(c_truncated.full - c.full)/norm(c.full) = ", norm(c_trunc.full()-T2*T1)/norm(T2*T1)

    c_trunc=c.truncate(rank=29)
    print(c_trunc)

    print('testing 3d tucker core sorting and  rank-based truncation ...')
    print "c_truncated.full - c.full = ", norm(c_trunc.full()-T2*T1)
    print "norm(c_truncated.full - c.full)/norm(c.full) = ", norm(c_trunc.full()-T2*T1)/norm(T2*T1)
    print
    
#    k=N2
#    while k>2:
#        c_trunc=c.truncate(rank=k)
#        print "norm(c_truncated.full - c.full)/norm(c.full) = ", norm(c_trunc.full()-T2*T1)/norm(T2*T1)
#        k-=1
    
    N1=10  
    N2=20
    N3=30
    T1=np.random.rand(N1,N2,N3)
    T2=np.random.rand(N1,N2,N3) 
    
    a = Tucker(val=T1,name='a' )
    b = Tucker(val=T2,name='b' )
    
    c=a+b
    
    print (np.mean(c.full()) - c.mean())
    print (np.mean(c.full()) - c.fourier().mean())
#    c_trunc2=c.truncate2(rank=13)
#    print(c_trunc2)
#
#    print('testing 3d tucker core sorting and  rank-based truncation ...')
#    print "c_truncated.full - c.full = ", norm(c_trunc2.full()-T2*T1)
#    print "norm(c_truncated.full - c.full)/norm(c.full) = ", norm(c_trunc2.full()-T2*T1)/norm(T2*T1)

#    T12=T1*T2
#    c2 = Tucker(val=T12,name='c2',r=8  )
#    print c2
#
#    print('testing 3d tucker core sorting and  rank-based truncation ...')
#    print "c2_truncated.full - c2.full = ", norm(c2.full()-T12)
#    print "norm(c2_truncated.full - c2.full)/norm(c2.full) = ", norm(c2.full()-T12)/norm(T12)

#

 #    t1=timeit.timeit("c=a*b", setup='from __main__ import a, b', number=10)
 #
 #    t2=timeit.timeit("c=a.mul(b)", setup='from __main__ import a, b', number=10)
 #
 #
 #    print
 #    print "t1: %f"%t1
 #    print "t2: %f"%t2
 #
 #    c1=a*b
 #    c2=a.mul(b)
 #
 #    print "error: %f"% norm(c1.full()-c2.full())

#    print('testing DFT...')
#
#    from ffthompy.tensors.operators import DFT
#
#
#    Fa=a.fourier()
#    Fa2=DFT.fftnc(a.full(), a.N)
#
#    print(np.linalg.norm(Fa.full()-Fa2))
# #
# #    print('Comparing time cost of tensor of 1-D FFT and n-D FFT ...')
# #    t1=timeit.timeit("a.fourier()", setup='from __main__ import a', number=10)
# #    afull=a.full()
# #    t2=timeit.timeit("DFT.fftnc(afull, a.N)", setup='from ffthompy.tensors.operators import DFT;from __main__ import a, afull', number=10)
# #    # t1=timeit.timeit("aa=a.truncate(tol=0.05); aa.fourier()", setup='from __main__ import a', number=10000)
# #    print
# #    print "Tensor of 1D FFT costs: %f"%t1
# #    print "n-D FFT costs         : %f"%t2
# #
# #    print('END')
#     
#    a2= Tucker(N=np.array([7,7,7]),r=np.array([5,5,5]), randomise=True)
#    a2t=a2.truncate(rank=3)
#    
#    print "a2_truncated.full - a2.full = ", norm(a2t.full()-a2.full())
#    print "norm(a2_truncated.full - a2.full)/norm(a2.full) = ", norm(a2t.full()-a2.full())/norm(a2.full())
#
#    a2t2=a2.truncate2(rank=3)
#    
#    print
#    print "a2_truncated.full - a2.full = ", norm(a2t2.full()-a2.full())
#    print "norm(a2_truncated.full - a2.full)/norm(a2.full) = ", norm(a2t2.full()-a2.full())/norm(a2.full())    
#    print('END')
