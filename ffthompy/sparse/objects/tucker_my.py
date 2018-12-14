import numpy as np

from ffthompy.sparse.objects.canoTensor import CanoTensor
from ffthompy.sparse.decompositions import HOSVD, nModeProduct
from ffthompy.tensors import Tensor
from ffthompy.sparse.objects.tensors import fft_form_default
from ffthompy.tensors.objects import full_fft_form_default

from numpy.linalg import norm
from numpy import newaxis

#np.set_printoptions(precision=3)
#np.set_printoptions(linewidth=999999)

class Tucker(CanoTensor):

    def __init__(self, name='unnamed', val=None, core=None, basis=None, Fourier=False, orthogonal=False,
                 r=None, N=None, randomise=False, fft_form=fft_form_default):

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

        self._set_fft(fft_form)

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
        core=np.zeros(r_new)

        L1=[None]*self.order
        L2=[None]*self.order
        for d in range(self.order):
            L1[d]=range(X.r[d])
            L2[d]=range(X.r[d],r_new[d])

        X_locations=np.ix_(*L1)
        Y_locations=np.ix_(*L2)

        core[X_locations] = X.core
        core[Y_locations] = Y.core

        newBasis=[np.vstack([X.basis[ii], Y.basis[ii]]) for ii in range(X.order)]

        return Tucker(name=self.name+'+'+Y.name, core=core, basis=newBasis, orthogonal=False,
                      Fourier=self.Fourier)

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
            if self.Fourier and self.fft_form=='sr': #product of scipy rfft tensors need a special multiplication
                for d in range(0, self.order):
                    B=np.empty((self.r[d]*Y.r[d], self.N[d]))
                    B[:,0]=np.kron(self.basis[d][:,0],Y.basis[d][:,0])
                    if self.N[d]%2 != 0:
                        ar=self.basis[d][:,1::2]
                        ai=self.basis[d][:,2::2]
                        br=Y.basis[d][:,1::2]
                        bi=Y.basis[d][:,2::2]
                        B[:,1::2]=(ar[:, newaxis, :]*br[newaxis, :, :]-ai[:, newaxis, :]*bi[newaxis, :, :]).reshape(self.r[d]*Y.r[d],-1)
                        B[:,2::2]=(ar[:, newaxis, :]*bi[newaxis, :, :]+ai[:, newaxis, :]*br[newaxis, :, :]).reshape(self.r[d]*Y.r[d],-1)
                    else:
                        B[:,-1]=np.kron(self.basis[d][:,-1],Y.basis[d][:,-1])
                        ar=self.basis[d][:,1:-1:2]
                        ai=self.basis[d][:,2:-1:2]
                        br=Y.basis[d][:,1:-1:2]
                        bi=Y.basis[d][:,2:-1:2]
                        B[:,1:-1:2]=(ar[:, newaxis, :]*br[newaxis, :, :]-ai[:, newaxis, :]*bi[newaxis, :, :]).reshape(self.r[d]*Y.r[d],-1)
                        B[:,2:-1:2]=(ar[:, newaxis, :]*bi[newaxis, :, :]+ai[:, newaxis, :]*br[newaxis, :, :]).reshape(self.r[d]*Y.r[d],-1)
                    newBasis[d]=B
            else:
                for d in range(0, self.order):
                    newBasis[d]=np.multiply(self.basis[d][:, newaxis, :], Y.basis[d][newaxis, :, :])
                    newBasis[d]=np.reshape(newBasis[d], (-1, self.N[d]))

            return Tucker(name=self.name+'*'+Y.name, core=newCore, basis=newBasis,
                          Fourier=self.Fourier, fft_form=self.fft_form)

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

            return Tucker(name=self.name, core=S, basis=newBasis, orthogonal=True, Fourier=self.Fourier, fft_form=self.fft_form)

    def full(self, fft_form=full_fft_form_default):
        """convert a tucker representation to a full tensor object
        A = CORE (*1) Basis1 (*2) Basis2 (*3) Basis3 ..., with (*n)  means n-mode product.
        from paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE
        """
        # if core is a single scalar value, make it in a n-D array shape
        if np.prod(self.core.shape)==1:
            self.core=np.reshape(self.core, tuple(self.r))

        if self.Fourier:
            res=self.fourier()
        else:
            res=self

        val=res.core.copy()
        for i in range(self.order):
            val=nModeProduct(val, res.basis[i].T, i)

        T=Tensor(name=res.name, val=val, order=0, N=val.shape, Fourier=False, fft_form=fft_form)

        if self.Fourier:
            T.fourier()

        return T

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

        if isinstance(tol, float) :
            tol=tol*np.ones((self.order,))

        self=self.orthogonalise()

        basis=list(self.basis)
        core=self.core
        rank=np.minimum(rank, self.r)

        # to determine the rank of truncation
        if np.any(tol) is not None:
            rank=np.zeros((self.order), dtype=int)
            # determine the truncation rank so that (1.0-tol)*100% of the norm of the core in that direction  is perserved.
            sorted_norm=[]
            ind=range(self.order)
            for i in ind:
                sorted_norm.append(np.sum(abs(core), axis=tuple(np.setdiff1d(ind, i))))
                rank[i]=np.searchsorted(np.cumsum(sorted_norm[i])/np.sum(sorted_norm[i]), 1.0-tol[i])+1

        L=[None]*self.order
        for d in range(self.order):
            L[d]=range(rank[d])

        part_taken_ind=np.ix_(*L)
        core=core[part_taken_ind] # truncate the core

        for ii in range(self.order):
            basis[ii]=basis[ii][:rank[ii], :]

        return Tucker(name=self.name+'_truncated', core=core, basis=basis, orthogonal=True, Fourier=self.Fourier,fft_form=self.fft_form)

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

        basis_mean=[None]*self.order
        mean_kronecker=1

        for k in range(self.order):
            if self.Fourier:
                basis_mean[k]=self.basis[k][:,self.mean_index()[k]].real
            else:
                basis_mean[k]=np.mean(self.basis[k], axis=1)
        for k in range(self.order):
            mean_kronecker = np.kron(mean_kronecker,basis_mean[k] )

        return np.sum(mean_kronecker*self.core.ravel())

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

#    # DFT
#    print('testing DFT...')
#
#    Fa=a.fourier()
#    print(Fa)
#    Fa2=DFT.fftnc(a.full(), a.N)
#    print(np.linalg.norm(Fa.full()-Fa2))

######### 3-d tenssor test #########

    print
    print('----testing 3d tucker ----')
    print

    N1=10 # warning: in 3d multiplication too large N number could kill the machine.
    N2=12
    N3=15

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
    N4=5
    # multiplication
    T1=np.random.rand(N1,N2,N3,N4)
    T2=np.random.rand(N1,N2,N3,N4)

    #T1=np.sin(T )

    a = Tucker(val=T1,name='a' )
    b = Tucker(val=T2,name='b' )

    #a=a.truncate(rank=a.r/2)

    print(np.mean(T1) -a.mean() )
    print(np.mean(T1) -a.fourier().mean() )

    c=a+b
    print(c)

    print('testing 3d addtion ...')
    print "(a+b).full - (a.full+b.full) = ", norm(c.full()-a.full()-b.full())
    print "((a+b).full - (a.full+b.full))/|(a.full+b.full)| = ", norm(c.full()-a.full()-b.full())/norm(a.full()+b.full())
    print "max((a+b).full - (a.full+b.full))/mean(a.full+b.full) = ", np.max(c.full().val-a.full().val-b.full().val)/np.mean(a.full().val+b.full().val)
#
    print(np.mean(T1+T2) -c.mean() )


    c=a*b
    print(c)

    print('testing 3d multiplication and re-orthogonalization...')
    print "(a*b).full - (a.full*b.full) = ", norm(c.full()-a.full()*b.full())
    print "((a*b).full - (a.full*b.full))/|(a.full*b.full)| = ", norm(c.full()-a.full()*b.full())/norm(a.full()*b.full())
    print "max((a*b).full - (a.full*b.full))/mean(a.full*b.full) = ", np.max(c.full().val-a.full().val*b.full().val)/np.mean(a.full()*b.full())
#
    c_trunc=c.truncate(rank=8)
    print(c_trunc)

    print('testing 3d tucker rank-based truncation ...')
    print "c_truncated.full - c.full = ", norm(c_trunc.full()-T2*T1)
    print "norm(c_truncated.full - c.full)/norm(c.full) = ", norm(c_trunc.full()-T2*T1)/norm(T2*T1)


#    c_trunc=c.truncate(rank=-5)
#    print(c_trunc)
#
#    print('testing 3d tucker core sorting and  rank-based truncation ...')
#    print "c_truncated.full - c.full = ", norm(c_trunc.full()-T2*T1)
#    print "norm(c_truncated.full - c.full)/norm(c.full) = ", norm(c_trunc.full()-T2*T1)/norm(T2*T1)

    c_trunc=c.truncate(tol=0.2)
    print(c_trunc)

    print('testing 3d tucker tol-based truncation ...')
    print "c_truncated.full - c.full = ", norm(c_trunc.full()-T2*T1)
    print "norm(c_truncated.full - c.full)/norm(c.full) = ", norm(c_trunc.full()-T2*T1)/norm(T2*T1)
    print

#    k=N2
#    while k>2:
#        c_trunc=c.truncate(rank=k)
#        print "norm(c_truncated.full - c.full)/norm(c.full) = ", norm(c_trunc.full()-T2*T1)/norm(T2*T1)
#        k-=1

    N1=13
    N2=20
    N3=12
    T1=np.random.rand(N1,N2,N3)
    T2=np.random.rand(N1,N2,N3)

    a = Tucker(val=T1,name='a' )
    b = Tucker(val=T2,name='b' )

    c=a+b

    print (np.mean(c.full().val) - c.mean())
    print (np.mean(c.full().val) - c.fourier().mean())

    ### test Fourier Hadamard product #####
    af=a.set_fft_form('c').fourier()
    bf=b.set_fft_form('c').fourier()

    afbf=af*bf

    af2=a.set_fft_form('sr').fourier()
    bf2=b.set_fft_form('sr').fourier()

    afbf2=af2*bf2

    print( (afbf.fourier()-afbf2.fourier()).norm())
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
