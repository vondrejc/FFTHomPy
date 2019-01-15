import numpy as np
from numpy import newaxis
from ffthompy.sparse.objects.tensors import SparseTensorFuns
from ffthompy.tensors import Tensor
from ffthompy.sparse.objects.tensors import fft_form_default


class CanoTensor(SparseTensorFuns):
    kind='cano'

    def __init__(self, name='unnamed', val=None, core=None, basis=None, orthogonal=False,
                 Fourier=False, r=None, N=None, randomise=False, fft_form=fft_form_default):

        self.name=name
        self.Fourier=Fourier # TODO: dtype instead of Fourier
        self.orthogonal=orthogonal

        if val is not None:
            if len(val.shape)==2:
                u, s, vt=np.linalg.svd(val, full_matrices=0)

                self.order=2
                self.basis=[u.T, vt]
                self.core=s
                self.r=self.core.shape[0]

                self.N=np.empty(self.order, dtype=np.int)
                for ii in range(self.order):
                    self.N[ii]=self.basis[ii].shape[1]
            else:
                raise ValueError("Canonical format not yet applicable to tensors higher than 2 dimensional.")

        elif core is not None and basis is not None:
            self.order=basis.__len__()
            self.basis=basis

            if len(core.shape)==2:
                self.core=np.diag(core)
            else:
                self.core=core

            self.r=basis[0].shape[0] # since all basis share the same r

            self.N=np.empty(self.order, dtype=np.int)
            for ii in range(self.order):
                self.N[ii]=basis[ii].shape[1]
        else:
            self.r=3
            self.N=[5, 5]
            self.order=self.N.__len__()
            if randomise:
                self.randomise()
            else:
                self.core=np.zeros(r)
                self.basis=[np.zeros([self.r[ii], self.N[ii]]) for ii in range(self.order)]

        self._set_fft(fft_form)

    def set_fft_form(self, fft_form=fft_form_default, copy=False):
        if copy:
            R=self.copy()
        else:
            R=self

        if self.fft_form==fft_form:
            return R

        if R.Fourier:
            for i in range(R.order):
                R.basis[i]= R.ifft(R.basis[i], R.N[i])

            R._set_fft(fft_form)
            for i in range(R.order):
                R.basis[i]= R.fft(R.basis[i], R.N[i])
            R.orthogonal=False # FFT modes makes the basis no longer orthonormal

        else:
            R._set_fft(fft_form)

        return R

    def randomise(self):
        self.core=np.random.random((self.r,))
        self.basis=[np.random.random([self.r, self.N[ii]]) for ii in range(self.order)]

    def orthogonalise(self):
        """re-orthogonalise the basis"""
        if self.orthogonal:
            return self
        else:
            # re-orthogonalise the basis by QR and SVD
            qa, ra=np.linalg.qr(self.basis[0].T, 'reduced')
            qb, rb=np.linalg.qr(self.basis[1].T, 'reduced')

            core=ra.real*self.core[np.newaxis, :]
            core=np.dot(core, rb.T.real)

            u, s, vt=np.linalg.svd(core,full_matrices=False)

            newA=np.dot(qa, u)
            newB=np.dot(vt, qb.T)

            newBasis=[newA.T, newB]
            return CanoTensor(name=self.name, core=s, basis=newBasis, orthogonal=True,
                              Fourier=self.Fourier, fft_form=self.fft_form)

    def __add__(self, Y):
        X=self
        assert(X.Fourier==Y.Fourier)
        core=np.hstack([X.core, Y.core])
        basis=[np.vstack([X.basis[ii], Y.basis[ii]]) for ii in range(self.order)]

        return self.copy(name=X.name+'+'+Y.name, core=core, basis=basis)

    def __mul__(self, Y):
        "element-wise multiplication of two canonical tensors"

        if isinstance(Y, float) or isinstance(Y, int) :
            R=self.copy()
            R.core=self.core*Y
            return R
        elif isinstance(Y, np.ndarray) and Y.shape==(1,):
            R=self.copy()
            R.core=self.core*Y[0]
            return R
        else:
            coeff=np.kron(self.core, Y.core)
            newBasis=[None]*self.order
            if self.Fourier and self.fft_form=='sr': #product of scipy rfft tensors need a special multiplication
                for d in range(0, self.order):
                    B=np.empty((self.r*Y.r, self.N[d]))
                    B[:,0]=np.kron(self.basis[d][:,0],Y.basis[d][:,0])
                    if self.N[d]%2 != 0:
                        ar=self.basis[d][:,1::2]
                        ai=self.basis[d][:,2::2]
                        br=Y.basis[d][:,1::2]
                        bi=Y.basis[d][:,2::2]
                        B[:,1::2]=(ar[:, newaxis, :]*br[newaxis, :, :]-ai[:, newaxis, :]*bi[newaxis, :, :]).reshape(self.r*Y.r,-1)
                        B[:,2::2]=(ar[:, newaxis, :]*bi[newaxis, :, :]+ai[:, newaxis, :]*br[newaxis, :, :]).reshape(self.r*Y.r,-1)
                    else:
                        B[:,-1]=np.kron(self.basis[d][:,-1],Y.basis[d][:,-1])
                        ar=self.basis[d][:,1:-1:2]
                        ai=self.basis[d][:,2:-1:2]
                        br=Y.basis[d][:,1:-1:2]
                        bi=Y.basis[d][:,2:-1:2]
                        B[:,1:-1:2]=(ar[:, newaxis, :]*br[newaxis, :, :]-ai[:, newaxis, :]*bi[newaxis, :, :]).reshape(self.r*Y.r,-1)
                        B[:,2:-1:2]=(ar[:, newaxis, :]*bi[newaxis, :, :]+ai[:, newaxis, :]*br[newaxis, :, :]).reshape(self.r*Y.r,-1)
                    newBasis[d]=B
            else:
                for d in range(0, self.order):
                    newBasis[d]=np.multiply(self.basis[d][:, newaxis, :], Y.basis[d][newaxis, :, :])
                    newBasis[d]=np.reshape(newBasis[d], (-1, self.N[d]))

            return self.copy(name=self.name+'*'+Y.name, core=coeff, basis=newBasis)

    def full(self, **kwargs):
        "return a full tensor object"

        if self.order==2:
            if self.Fourier:
                res=self.fourier(copy=True)
            else:
                res=self

            # generate Tensor in real domain
            val=np.einsum('i,ik,il->kl', res.core, res.basis[0], res.basis[1])
            kwargsT=dict(name=res.name, val=val, order=0, N=val.shape, Fourier=False, fft_form='c')
            kwargsT.update(kwargs)
            T=Tensor(**kwargsT)

            if self.Fourier:
                T.fourier()
            return T
        else:
            raise NotImplementedError()

    def truncate(self, rank=None, tol=None):
        "return truncated tensor"
        # tol is the maximum "portion" of the core trace to be lost, e.g. tol=0.01 means at most 1 percent could be lost in the truncation.
        # if tol is not none, it will override rank as the truncation criteria.
        if tol is None and rank is None:
            return self

        if rank>=self.r:
            #print ("Warning: Rank of the truncation not smaller than the original rank, truncation aborted!")
            return self

        self=self.orthogonalise()

        basis=list(self.basis)
        core=self.core

        if tol is not None:
            # determine the truncation rank so that (1.0-tol)*100% of the trace of the core is perserved.
            rank=np.searchsorted(np.cumsum(np.abs(core))/np.sum(np.abs(core)), 1.0-tol)+1

        # truncation
        core=core[:rank]
        for ii in range(self.order):
            basis[ii]=basis[ii][:rank, :]

        return self.copy(name=self.name+'_truncated', core=core, basis=basis, orthogonal=True)

    def norm(self, ord='core'):
        if ord=='fro':   # this is the same as using the 'core' option if the tensor is orthogonalised.
            R=self*self.conj()
            val=0.
            for ii in range(R.r):
                valii=R.core[ii]
                for jj in range(R.order):
                    valii*=np.sum(R.basis[jj][ii]).real
                val+=valii
            val=val**0.5
        elif ord==1:
            pass
        elif ord=='inf':
            pass
        elif ord=='core':
            if self.Fourier:
                newObj=self.set_fft_form('c',copy=True)
            else:
                newObj=self

            if not newObj.orthogonal:
                newObj=newObj.orthogonalise()

            if newObj.Fourier:
                return np.linalg.norm(newObj.core)
            else:
                return np.linalg.norm(newObj.core)/np.sqrt(np.prod(newObj.N)) # this normalized 2-norm is to keep consistance to the full tensor norm()
        else:
            raise NotImplementedError()
        return val

    def mean(self):

        basis_mean=[None]*self.order
        val=self.core.copy()

        for k in range(self.order):
            if self.Fourier:
                basis_mean[k]=self.basis[k][:,self.mean_index()[k]].real
            else:
                basis_mean[k]=np.mean(self.basis[k], axis=1)

        for k in range(self.order):
            val *= basis_mean[k]

        return np.sum(val)

    def enlarge(self, M):
        dtype=self.basis[0].dtype
        assert(self.Fourier)

        M=np.array(M, dtype=np.int)
        N=np.array(self.N)

        if np.allclose(M, N):
            return self

        r=self.r
        if isinstance(r, int) :
            r=r*np.ones((self.order,), dtype=int)
        if self.fft_form in ['c']:
            ibeg=np.ceil(np.array(M-N, dtype=np.float)/2).astype(dtype=np.int)
            iend=np.ceil(np.array(M+N, dtype=np.float)/2).astype(dtype=np.int)
        elif self.fft_form in ['sr']:
            ibeg=np.zeros(N.shape).astype(dtype=np.int)
            iend=N

        basis=[]
        for ii, m in enumerate(M):
            basis.append(np.zeros([r[ii], m], dtype=dtype))
            basis[ii][:, ibeg[ii]:iend[ii]]=self.basis[ii]

        newOne=self.copy()
        newOne.basis=basis
        newOne.N=np.zeros((newOne.order,), dtype=int)
        for i in range(newOne.order):
            newOne.N[i]=newOne.basis[i].shape[1]
        # return CanoTensor(name=self.name, core=self.core, basis=basis, Fourier=self.Fourier)
        return newOne # this avoid using specific class name, e.g. canoTensor, so that can be shared by tucker and canoTensor

    def decrease(self, M):
        assert(self.Fourier is True)

        M=np.array(M, dtype=np.int)
        N=np.array(self.N)
        assert(np.all(np.less(M, N)))
        if self.fft_form in ['c']:
            ibeg=np.fix(np.array(N-M+(M%2), dtype=np.float)/2).astype(dtype=np.int)
            iend=np.fix(np.array(N+M+(M%2), dtype=np.float)/2).astype(dtype=np.int)
        elif self.fft_form in ['sr']:
            ibeg=np.zeros(N.shape).astype(dtype=np.int)
            iend=M

        basis=[]
        for ii in range(N.size):
            basis.append(self.basis[ii][:, ibeg[ii]:iend[ii]])

        newOne=self.copy()
        newOne.basis=basis
        newOne.N=np.zeros((newOne.order,), dtype=int)
        for i in range(newOne.order):
            newOne.N[i]=newOne.basis[i].shape[1]
        return newOne # this avoid using specific class name, e.g. canoTensor, so that can be shared by tucker and canoTensor
        # return Tucker(name=self.name, core=self.core, basis=basis, Fourier=self.Fourier)

    def __neg__(self):
        newOne=self.copy()
        newOne.core=-newOne.core
        return newOne # this avoid using specific class name, e.g. canoTensor, so that can be shared by tucker and canoTensor

    def __sub__(self, Y):
        return self.__add__(-Y)

    def __rmul__(self, X):

        if isinstance(X, np.float) or isinstance(X, np.int) :
            R=self.copy()
            R.core=X*self.core
        else:
            raise NotImplementedError()
        return R

    def conj(self):
        """Element-wise complex conjugate"""
        basis=[]
        for ii in range(self.order):
            basis.append(self.basis[ii].conj())
        res=self.copy()
        res.basis=basis
        return res

    def add(self, Y, tol=None, rank=None):
        return (self+Y).truncate(tol=tol, rank=rank)

    def multiply(self, Y, tol=None, rank=None):
        # element-wise multiplication
        return (self*Y).truncate(tol=tol, rank=rank)

    def scal(self, Y):
        X = self
        assert(X.Fourier==Y.Fourier)
        XY = X*Y
        return XY.mean()

    def repeat(self, M):
        """
        Enhance the tensor size from to N to M, by repeating all elements by M/N times.

        :param M: the new size .
        :type A: integer or list of integers

        :returns: Tucker -- a new tucker object with size M
        """
        if isinstance(M, int):
            M=M*np.ones((self.order,), dtype=int)

        if ((M.astype(float)/self.N)%1).any()!=0 :
            raise NotImplementedError("M is not a multiple of the old size N")

        res=self.copy()
        for i in range(self.order):
            res.basis[i]=np.repeat(res.basis[i], M[i]/self.N[i], axis=1)
            res.basis[i]/=np.sqrt(M[i]/self.N[i]) # restore original norm

        res.core*=np.prod(np.sqrt(M/self.N))
        res.N=M
        res.orthogonal=self.orthogonal

        return res

    def project(self, M):
        if self.Fourier:
            if all(M>=self.N):
                return self.enlarge(M)
            elif all(M<=self.N):
                return self.decrease(M)
        else:
            F=self.fourier()
            if all(M>=self.N):
                F=F.enlarge(M)
            elif all(M<=self.N):
                F=F.decrease(M)

            return F.fourier() # inverse Fourier

    def __repr__(self, full=False, detailed=False):
        keys=['name', 'N', 'Fourier', 'fft_form', 'orthogonal', 'r','norm','mean']
        return self._repr(keys)

    @property
    def size(self):
        "return the number of elements of the original full tensor"
        return np.prod(self.N)

    @property
    def memory(self):
        "return the number of floating point numbers that consist of the canonical tensor"
        return self.r +self.r*sum(self.N)

    def copy(self, **kwargs):
        keys=('name','core','basis','orthogonal','Fourier','r', 'N','fft_form')
        return self._copy(keys, **kwargs)

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
    N=10
    M=20
#    L= min(N,M)

    x=np.linspace(-np.pi, np.pi, M)
    y=np.linspace(-np.pi, 0.77*np.pi, N)
    # creat matrix for test
    S1=np.sin(x[np.newaxis, :]+y[:, np.newaxis])*(x[np.newaxis, :]+y[:, np.newaxis])
    S2=np.cos(2*x[np.newaxis, :]-y[:, np.newaxis])*(2*x[np.newaxis, :]-y[:, np.newaxis])
    #S1 = np.dot(np.reshape(x,(M,1)), np.reshape(y,(1,N))) + np.dot(np.sin(np.reshape(x,(M,1))), np.reshape(y,(1,N))**2)

    # factorize the matrix
    u1, s1, vt1=np.linalg.svd(S1, full_matrices=0)
    u2, s2, vt2=np.linalg.svd(S2, full_matrices=0)


    # construct  canoTensors with the normalized basis and the corresponding coefficients core
    a=CanoTensor(name='a', core=s1, basis=[u1.T, vt1])
    b=CanoTensor(name='b', core=s2, basis=[u2.T, vt2])

    print(a.norm(ord='fro'))
    print(a.norm(ord='core'))


#     N=[100,101]
#     a=CanoTensor(name='a', r=50, N=N, randomise=True)
#     b=CanoTensor(name='a', r=60, N=N, randomise=True)

    # addition
    c=a+b
    c2=a.add(b, tol=1e-5)
    c2=c.truncate(tol=1e-5)

    c_add=a.full()+b.full()
    print
    print "(a+b).full - (a.full+b.full)    = ", (np.linalg.norm(c.full()-c_add))
    print "add(a,b).full - (a.full+b.full) = ", (np.linalg.norm(c2.full()-c_add))


    # multiplication

    c=a*b
    c3=a.multiply(b, tol=0.001)

    c_mul=a.full()*b.full()
    print
    print  "                  (a*b).full - (a.full*b.full) = ", (np.linalg.norm(c.full()-c_mul))
    print  "truncated multiply(a,b).full - (a.full*b.full) = ", (np.linalg.norm(c3.full()-c_mul))
    print

    print('rank control on tensor product:')
    print "full product tensor rank=     ", c.r
    print "truncated product tensor rank=", c3.r
    print
    # truncation
    a_trunc=a.truncate(rank=4)

    print  "a.full  - a_trunc.full        = ", np.linalg.norm(a.full()-a_trunc.full())
    print

#    # DFT
#    print('testing DFT...')

#    from ffthompy.tensors.operators import DFT
#
#    Fa=a.fourier()
#    Fa2=DFT.fftnc(a.full(), a.N)
#
#    print(np.linalg.norm(Fa.full()-Fa2))
#
#    print('Comparing time cost of tensor of 1-D FFT and n-D FFT ...')
#    t1=timeit.timeit("a.fourier()", setup='from __main__ import a', number=10)
#    afull=a.full()
#    t2=timeit.timeit("DFT.fftnc(afull, a.N)", setup='from ffthompy.tensors.operators import DFT;from __main__ import a, afull', number=10)
#    # t1=timeit.timeit("aa=a.truncate(tol=0.05); aa.fourier()", setup='from __main__ import a', number=10000)
#    print
#    print "Tensor of 1D FFT costs: %f"%t1
#    print "n-D FFT costs         : %f"%t2


    N1=6
    N2=8

    T1=np.random.rand(N1,N2)
    T2=np.random.rand(N1,N2)

    a = CanoTensor(val=T1,name='a' )
    b = CanoTensor(val=T2,name='b' )
    c=a*b
    print(c)

    print

    m=c.mean()
#    k=N2
#    while k>2:
#        c_trunc=c.truncate(rank=k)
#        print "norm(c_truncated.full - c.full)/norm(c.full) = ", norm(c_trunc.full()-T2*T1)/norm(T2*T1)
#        k-=1

    print (np.mean(c.full().val) - c.mean())
    print (np.mean(c.full().val) - c.fourier().mean())


    cf=c.set_fft_form('c').fourier().set_fft_form('sr')

    cf2=c.set_fft_form('sr').fourier()

    print( (cf-cf2).norm())

    ### test Fourier Hadamard product #####
    af=a.set_fft_form('c').fourier()
    bf=b.set_fft_form('c').fourier()

    afbf=af*bf

    af2=a.set_fft_form('sr').fourier()
    bf2=b.set_fft_form('sr').fourier()

    afbf2=af2*bf2

    print( (afbf.fourier()-afbf2.fourier()).norm())

    print(c)
    print(c.fourier())

    cf=c.fourier().truncate(rank=5)
    print(cf)
    print(cf.fourier())



#    ### test enlarge#####
#    n=5
# #    T1= np.zeros((n,n, n))
# #    T1[n/3:2*n/3, n/3:2*n/3, n/3:2*n/3]=1
#
#    T1= np.zeros((n,n ))
#    T1[n/3:2*n/3, n/3:2*n/3 ]=1
#
#    u1, s1, vt1=np.linalg.svd(T1, full_matrices=0)
#
#    t=CanoTensor(name='a', core=s1, basis=[u1.T, vt1])
#    tf=t.fourier()
#    tfl=tf.enlarge([7,7])
#    tfli= tfl.fourier()
#
#    tfd=tf.decrease([3,3])
#    tfdi= tfd.fourier()
#
#    print(tfli.full().val)
#    print
#    print(tfdi.full().val)
#    print
#    t=t.set_fft_form('c')
#    tf=t.fourier()
#    tfl=tf.enlarge([7,7])
#    tfli= tfl.fourier()
#
#    print(tfli.full().val.real)




### test Fourier #####


    ### test full ###


#    print (np.linalg.norm(c.full()-T1*T2))
##    print (np.linalg.norm(c.full2()-T1*T2))
#
#    T=Tensor(val=T1*T2, order=0, Fourier=False)
#
#    print (np.linalg.norm(c.fourier().full(fft_form=full_fft_form_default).val- T.set_fft_form(fft_form=full_fft_form_default).fourier().val ))


    print('END')
