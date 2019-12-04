import numpy as np
from operator import mul

from numpy import reshape, dot
from numpy.linalg import qr, norm, svd
from scipy.linalg import rq
from ffthompy.tensors import Tensor
from ffthompy.tensorsLowRank.objects.tensors import LowRankTensorFuns
from tt.core.vector import vector
from ffthompy.tensorsLowRank.objects.tensors import fft_form_default
from functools import reduce


class TensorTrain(vector, LowRankTensorFuns):
    kind='tt'

    def __init__(self, val=None, core=None, eps=None, rmax=None, Fourier=False, name='unnamed',
                 vectorObj=None, fft_form=fft_form_default):

        if eps is None: eps=1e-14
        if rmax is None: rmax=999999

        if val is not None:
            vectorObject=vector(val, eps, rmax)
        elif core is not None:
            vectorObject=vector.from_list(core)
        elif vectorObj is not None: # cast a TTPY object to tensorTrain object
            vectorObject=vectorObj
        else: # a  3D zero tensor
            vectorObject=vector(np.zeros((3, 4, 5)), eps, rmax)

        for attr_name in vectorObject.__dict__:
            setattr(self, attr_name, getattr(vectorObject, attr_name))

        self.N=self.n # ttpy use n, we use N.
        self.name=name
        self.Fourier=Fourier
        self.fft_form=fft_form
        self._set_fft(fft_form)

    @staticmethod
    def from_list(a, name='unnamed', Fourier=False, fft_form=fft_form_default):
        """Generate TensorTrain object from given TT cores.

        :param a: List of TT cores.
        :type a: list
        :returns: TensorTrain object constructed from the given cores.

        """
        vectorObj=vector.from_list(a)

        ttObj=TensorTrain(vectorObj=vectorObj, name=name, Fourier=Fourier,fft_form=fft_form)

        return ttObj

    @property
    def dim(self):
        return self.d


    def fourier(self, real_output=False):
        "(inverse) discrete Fourier transform"

        if self.Fourier:
            fftfun=lambda Fx, N,real_output: self.ifft(Fx, N,real_output)
            name='Fi({})'.format(self.name)
        else:
            fftfun=lambda x, N,real_output: self.fft(x, N)
            name='F({})'.format(self.name)

        cl=self.to_list(self)
        clf=[None]*self.d
        for i in range(self.d):
            clf[i]=fftfun(cl[i], self.n[i],real_output)

        res=self.from_list(clf, name=name, Fourier=not self.Fourier, fft_form=self.fft_form)

        return res

    def set_fft_form(self, fft_form=fft_form_default, copy=False):
        if copy:
            R=self.copy()
        else:
            R=self

        if self.fft_form==fft_form:
            return R

        if R.Fourier:
            cl=R.to_list(R)
            for i in range(R.d):
                cl[i]= R.ifft(cl[i], R.N[i])

            R._set_fft(fft_form)
            for i in range(R.d):
                cl[i]= R.fft(cl[i], R.N[i])

            tt=R.from_list(cl, name=R.name, Fourier=True, fft_form=fft_form)
            for attr_name in tt.__dict__:
                setattr(R, attr_name, getattr(tt, attr_name))
            R.Fourier=True
        else:
            R._set_fft(fft_form)

        return R

    def repeat(self, M):
        """
        Enhance the tensor size from N to M, by repeating all elements by M/N times.
        """
        #if isinstance(M, int):
         #   M=M*np.ones((self.order,), dtype=int)
        M = np.array(M)
        if M is self.N:
            return self

        res=self.copy()
        cl = res.to_list(res)
        clf = [None] * self.d

        shift=lambda bas: np.fft.fftshift(bas, axes=1)
        ishift=lambda bas: np.fft.ifftshift(bas, axes=1)
        for i in range(self.d):
            clf[i] =ishift(np.repeat(shift(cl[i]), M[i]/self.n[i], axis=1))

        res=res.from_list(clf)

        return res

    def enlarge(self, M):
        assert(self.Fourier is True)

        M=np.array(M, dtype=np.int)
        N=np.array(self.n)

        if np.allclose(M, N):
            return self
        if self.fft_form in ['c']:
            ibeg=np.ceil(np.array(M-N, dtype=np.float)/2).astype(dtype=np.int)
            iend=np.ceil(np.array(M+N, dtype=np.float)/2).astype(dtype=np.int)
        elif self.fft_form in ['sr']:
            ibeg=np.zeros(N.shape).astype(dtype=np.int)
            iend=N

        cl=self.to_list(self)
        dtype=cl[0].dtype

        cl_new=[None]*self.d
        for i in range(self.d):
            cl_new[i]=np.zeros([self.r[i], M[i], self.r[i+1]], dtype=dtype)
            cl_new[i][:, ibeg[i]:iend[i], :]=cl[i]

        res=self.from_list(cl_new,  fft_form=self.fft_form)
        res.name=self.name+"_enlarged"
        res.Fourier=self.Fourier

        return res

    def decrease(self, M):
        assert(self.Fourier is True)

        M=np.array(M, dtype=np.int)
        N=np.array(self.n)
        assert(np.any(np.less(M, N)))
        if self.fft_form in ['c']:
            ibeg=np.fix(np.array(N-M+(M % 2), dtype=np.float)/2).astype(dtype=np.int)
            iend=np.fix(np.array(N+M+(M % 2), dtype=np.float)/2).astype(dtype=np.int)
        elif self.fft_form in ['sr']:
            ibeg=np.zeros(N.shape).astype(dtype=np.int)
            iend=M

        cl=self.to_list(self)

        cl_new=[None]*self.d
        for i in range(self.d):
            cl_new[i]=cl[i][:, ibeg[i]:iend[i], :]

        res=self.from_list(cl_new, fft_form=self.fft_form)
        res.name=self.name+"_decreased"
        res.Fourier=self.Fourier

        return res

    def mean(self, normal_domain=True):
        """
        compute the mean of all elements of the tensor
        """
        cl=self.to_list(self)
        cl_mean=[None]*self.d

        for i in range(self.d):
            shape=np.ones((self.d+1), dtype=np.int32)
            shape[i:i+2]=self.r[i:i+2]
            if self.Fourier and normal_domain:
                cl_mean[i]=(cl[i][:, self.mean_index()[i],:]).reshape(tuple(shape)).real
            else:
                cl_mean[i]=np.mean(cl[i], axis=1).reshape(tuple(shape))

        cl_mean_prod=reduce(mul, cl_mean, 1) # product of all arrays inside cl_sum

        return np.sum(cl_mean_prod)

    def full(self, **kwargs):
        """
        convert TT to a full tensor object
        """

        if self.Fourier:
            res=self.fourier()
        else:
            res=self

        val= vector.full(res)

        # Tensor with the default fft_form for full tensor
        T=Tensor(name=res.name, val=val, order=0, N=val.shape, Fourier=False, **kwargs)

        if self.Fourier:
            T.fourier()

        return T

    def scal(self, Y):
        X=self
        assert(X.Fourier==Y.Fourier)
        XY=X*Y
        return XY.mean()

    def inner(self, Y):
        X=self
        assert(X.Fourier==Y.Fourier)
#        XY=X*Y
        res_vec=vector.__mul__(X, Y)
        XY=TensorTrain(vectorObj=res_vec,Fourier=self.Fourier, fft_form=self.fft_form)
        return XY.mean(normal_domain=False)*np.prod(XY.N)

    def __mul__(self, other):

        name=self.name+'*'+(str(other) if isinstance(other, (int, float, complex)) or np.isscalar(other) else other.name)

        if self.fft_form=='sr' and not isinstance(other, (int, float, complex)):
            res_vec=vector.__mul__(self.set_fft_form('c',copy=True), other.set_fft_form('c',copy=True))
            res=TensorTrain(vectorObj=res_vec, name=name[:50],
                            Fourier=self.Fourier, fft_form='c').set_fft_form('sr')
        else:
            res_vec=vector.__mul__(self, other)
            res=TensorTrain(vectorObj=res_vec, name=name[:50],
                            Fourier=self.Fourier, fft_form=self.fft_form)
        return res

    def __add__(self, other):
        res_vec=vector.__add__(self, other)
        res=TensorTrain(vectorObj=res_vec,
                        name=self.name+'+'+(str(other) if isinstance(other, (int, float, complex)) else other.name),
                        Fourier=self.Fourier, fft_form=self.fft_form)
        return res

    def __kron__(self, other):
        res_vec=vector.__kron__(self, other)
        res=TensorTrain(vectorObj=res_vec,
                        name=self.name+' glued with '+(str(other) if isinstance(other, (int, float, complex)) else other.name),
                        Fourier=self.Fourier, fft_form=self.fft_form)
        return res

    def __sub__(self, other):

        res=self+(-other)
        res.name=self.name+'-'+(str(other) if isinstance(other, (int, float, complex)) else other.name)

        return res

    def multiply(self, Y, tol=None, rank=None):
        # element-wise multiplication
        return (self*Y).truncate(tol=tol, rank=rank)

    @property
    def size(self):
        "return the number of elements of the original full tensor"
        return np.prod(self.n)

    @property
    def memory(self):
        "return the number of floating point numbers that consist of the TT tensor"
        return self.core.shape[0]+self.ps.shape[0]

    def __repr__(self): # print statement
        keys=['name', 'Fourier', 'fft_form', 'dim', 'N', 'r']
        return self._repr(keys)

    def truncate(self, tol=None, rank=None, fast=False):
        if np.any(tol) is None and np.any(rank) is None:
            return self
        elif np.any(tol) is None and np.all(rank>=max(self.r))==True :
            return self
        else:
            if tol is None: tol=1e-14
            if rank is None: rank=int(1e6)

            if fast:  # do a preliminary truncation before optimal truncation
                r=self.r.copy()
                cr=self.to_list(self)
                cr_new=[None]*self.d

                ratio=10 # keep ratio-times more bases than the target rank before the optimal truncation
                for i in range(1,self.d):
                    if rank < r[i] or tol > 1e-14:
                        nrm=norm(cr[i-1],axis=1).squeeze()
                        if i>1: nrm=norm(nrm, axis=0).squeeze()
                        #print(nrm)
                        if tol > 1e-14:
                            select_ind = np.argwhere(nrm >= (1e-3)*tol*np.sum(nrm)).squeeze()
                        else:
                            keep_rank_num= np.minimum(ratio*rank, r[i])
                            select_ind=np.argpartition(-nrm, keep_rank_num-1)[:keep_rank_num]

                        cr_new[i-1] = np.take(cr[i-1], select_ind, axis=2)
                        cr[i] = np.take(cr[i], select_ind, axis=0)
                    else:
                        cr_new[i-1]=cr[i-1]

                cr_new[self.d-1]=cr[self.d-1]

                res_vec=self.from_list(cr_new)
                res_vec=res_vec.round(eps=tol, rmax=rank)
            else:
                res_vec=self.round(eps=tol, rmax=rank) # round() produces a TTPY vetcor object

            res=TensorTrain(vectorObj=res_vec, name=self.name+'_truncated', Fourier=self.Fourier,
                            fft_form=self.fft_form)
            return res

    def orthogonalise(self, direction='lr', r_output=False):
        """Orthogonalise the list of cores of a TT. If direction is 'lr' it does a left to right sweep of qr decompositions,
        makes the TT "left orthogonal", i.e. if the i-th core is reshaped to shape (r[i]*n[i],r[i+1]), it has orthogonal columns.
        While with direction 'rl' the TT would be made "right orthogonal", i.e. if the i-th core is reshaped to
        shape (r[i], n[i]*r[i+1]), it has orthogonal rows.

        If r_output=True, the r factor of the last qr decomposition would be also returned. This is useful when we
        orthogonalise only a section of the list of cores and the r can be multiplid to the adjacent core of the next section.
        If we are orthogonalising the whole TT, set r_output to be False. In this case the last core has an ending rank 1,
        i.e. only one column (or row) in the sense mentioned above, so the qr decomposition to the last core is saved.
        """
        d=self.d
        r=self.r.copy()
        n=self.n.copy()
        cr=self.to_list(self)
        cr_new=[None]*d

        if direction=='lr' or direction=='LR':
            # qr sweep from left to right
            for i in range(d-1):
                cr[i]=reshape(cr[i], (r[i]*n[i], r[i+1]))
                cr_new[i], ru=qr(cr[i], 'reduced')
                cr[i+1]=dot(ru, reshape(cr[i+1], (r[i+1], n[i+1]*r[i+2])))
                r[i+1]=cr_new[i].shape[1]
                cr_new[i]=reshape(cr_new[i], (r[i], n[i], r[i+1]))

            if r_output:
                cr[d-1]=reshape(cr[d-1], (r[d-1]*n[d-1], r[d]))
                cr_new[d-1], ru=qr(cr[d-1], 'reduced')
                r[d]=cr_new[d-1].shape[1]
                cr_new[d-1]=reshape(cr_new[d-1], (r[d-1], n[d-1], r[d]))
                return self.from_list(cr_new), ru
            else:
                cr_new[d-1]=cr[d-1].reshape(r[d-1], n[d-1], r[d])
                newObj=self.from_list(cr_new, fft_form=self.fft_form)
                newObj.Fourier=self.Fourier
                newObj.fft_form=self.fft_form
                return newObj

        elif direction=='rl' or direction=='RL':
            # rq sweep from right to left
            for i in range(d-1, 0,-1):
                cr[i]=reshape(cr[i], (r[i], n[i]*r[i+1]))
                ru, cr_new[i]=rq(cr[i], mode='economic')
                cr[i-1]=dot(reshape(cr[i-1], (r[i-1]*n[i-1], r[i])), ru)
                r[i]=cr_new[i].shape[0]
                cr_new[i]=reshape(cr_new[i], (r[i], n[i], r[i+1]))

            if r_output:
                cr[0]=reshape(cr[0], (r[0], n[0]*r[1]))
                ru, cr_new[0]=rq(cr[0], mode='economic')
                cr_new[0]=reshape(cr_new[0]*ru, (r[0], n[0], r[1]))
                return self.from_list(cr_new), ru
            else:
                cr_new[0]=cr[0].reshape(r[0], n[0], r[1])
                newObj=self.from_list(cr_new, fft_form=self.fft_form)
                newObj.Fourier=self.Fourier

                return newObj

        else:
            raise ValueError("Unexpected parameter '"+direction+"' at tt.vector.tt_qr")

    def tt_chunk(self, start, end):
        """
        Generate a new TT by taking out one section in the list of cores of the TT 'self'.
        start and end are python indices, indicating the starting and ending indices of the section.
        this is a subroutine for qtt_ftt function in which we need to do seperate ffts to the m sections
        of the list of cores of a qtt (each section corresponds to a physical variable).
        """
        assert(start<=end)

        cr=self.to_list(self)
        cr=cr[start: end+1]
        return self.from_list(cr)

    def qtt_fft(self, ds, tol=1e-14):
        """
        Multidimensional FFT to a tensor of qtt format.

        The object 'self' should be quantic tensor train, i.e. with n=2 on every dimension.

        :param ds: numbers of tensor dimensions corresponds to each physical dimensions.
                   e.g., Our original tensor has two dimensions x and y, with mode size 4 and 8.
                   This translates to a quantic tensor of shape 2*2*2*2*2, with the first 2 dims
                   corresponds to x, and the last 3 corresponds to y. The ds in this case is [2,3].
        :type ds: list of integers

        :param tol: error torlerance
        :type tol: float
        """
        assert((self.n==2).any())

        ds=np.array(ds)

        D=np.prod(ds.shape)
        x=self.orthogonalise(direction='rl')

        for i in range(D):
            cury=x.tt_chunk(0, ds[i]-1)
            if i<D-1:
                x=x.tt_chunk(ds[i], x.d-1)

            cury=cury.qtt_fft1(tol, bitReverse=False)

            if i<D-1:
                cury, ru=cury.orthogonalise(direction='lr', r_output=True)
                crx=x.to_list(x)
                crx[0]=reshape(crx[0], (x.r[0], x.n[0]*x.r[1]))
                crx[0]=np.dot(ru, crx[0])
                x.r[0]=crx[0].shape[0]
                crx[0]=reshape(crx[0], (x.r[0], x.n[0], x.r[1]))
                x=x.from_list(crx)

            if i==0:
                y=cury
            else:
                y=y.__kron__(cury)

        cr=y.to_list(y)
        cr2=[None]*y.d

        for i in range(y.d):
            cr2[y.d-i-1]=cr[i].T

        cr2[y.d-1]/=2**((sum(ds))/2.0) # divided by sqrt(prod(N)), another sqrt(prod(N)) was already divided in qtt_fft1()
        y=y.from_list(cr2)

        return y

    def copy(self, name=None):
        if name is None:
            name = 'copy({})'.format(self.name)
        cl=self.to_list(self)
        return TensorTrain(name=name, core=cl, Fourier=self.Fourier, fft_form=self.fft_form)

    def norm(self, normal_domain=True):
        if self.Fourier and self.fft_form=='sr':
            return vector.norm(self.set_fft_form('c',copy=True))
        else:
            return vector.norm(self)

if __name__=='__main__':




    print()
    print('----testing "Fourier" function ----')
    print()

    v1=np.random.rand(50,23,10)
    v2=np.random.rand(50,23,10)



#    v=np.reshape(v1, (2, 3, 4 , 5), order='F') # use 'F' to keep v the same as in matlab
    t1=TensorTrain(val=v1, rmax=99999)
    t2=TensorTrain(val=v2, rmax=99999)
    print(t1)

    t=t1*t2
    print(t)


#    t2=t.truncate(rank=5)
#    print(t2)
#
#    print((t-t2).norm())
#
#    t5=t.truncate(rank=5, fast=True)
#    print(t5)
#
#    print((t-t5).norm())
#
    t3=t.truncate(tol=4e-1)
    print(t3)

    print((t-t3).norm())

    t4=t.truncate(tol=4e-1, fast=True)
    print(t4)

    print((t-t4).norm())

    t4=t.truncate(rank=5, fast=True)
    print(t4)

    print((t-t4).norm())

#
##    vfft=np.fft.fftn(v)
##
## #    TT=t.to_list(t)
## #    TTfft=[None]*t.d
## #
## #    for i in range(t.d):
## #        TTfft[i]=np.fft.fft(TT[i], axis=1)  # the axis to compute fft is in the middle of 0,1,2, i.e. 1.
## #
## #    vTTfft=TensorTrain.from_list(TTfft)
## #    #print (vTTfft.full())
## #    print(np.linalg.norm(vfft-vTTfft.full()))
##
##    tf=t.fft(shift=False)
##    print(np.linalg.norm(vfft-tf.full()))
#
##    tf2=t.fourier()
##    vfft_shift=DFT.fftnc(v, v.shape)
##
##    print(np.linalg.norm(vfft_shift-tf2.full()))
#
##    tf2i=tf2.fourier()
##    print(np.linalg.norm(v-tf2i.full()))
##
##    ### test mean()#####
##    print
##    print np.linalg.norm(t.mean()-np.sum(t.full()))
##
##    print t
##
#    # ## test casting a vector obj to tensortrain obj
#    ty=vector(v)
#    ty2=TensorTrain(vectorObj=ty)
#
##    ##   test  enlarge  ###
##    n=3
## #    T1= np.zeros((n,n, n))
## #    T1[n/3:2*n/3, n/3:2*n/3, n/3:2*n/3]=1
##
##    T1= np.zeros((n,n ))
##    T1[n/3:2*n/3, n/3:2*n/3 ]=1
##
##    print(T1)
##    print
##
##    t=TensorTrain(T1)
##    tf=t.fourier()
##    tfl=tf.enlarge([7,7])
##    tfli= tfl.fourier()
##
##    print(tfli.full().real)
##    print
##
##    cl_tf=tf.to_list(tf)
##    cl_tfl=tfl.to_list(tfl)
##    cl_tfli=tfli.to_list(tfli)
##
##    tfld=tfl.decrease([5,5])
##    print(tfld.fourier().full())
#
## # test hardamard product
#
#    v1=np.random.rand(2, 3, 4)
#    t1=TensorTrain(v1)
#    v2=np.random.rand(2, 3, 4)
#    t2=TensorTrain(v2)
#
#    t3=t1.__dot__(t2)
#    print(t3)
#
#    c1=t1.__getitem__([0, [list(range(t1.N[1]))], [list(range(t1.N[2]))]])
#
#    print(c1)
#
#    t3=t1*t2
#    print(t3)
#    print((np.linalg.norm(t3.full()-v1*v2)))
#
#    t4=t1+t2
#    print((np.linalg.norm(t4.full()-v1-v2)))
#    # print t4
#
#    t4o=t4.orthogonalise('lr')
#    print((np.linalg.norm(t4o.full()-t4.full())))
#
#    t4o=t4.orthogonalise('rl')
#    print((np.linalg.norm(t4o.full()-t4.full())))
#
#    # print t5.size
#    v1=np.random.rand(1, 5)
#
#    v=np.reshape(v1, (1, 5), order='F') # use 'F' to keep v the same as in matlab
#    t=TensorTrain(val=v, rmax=3)
#
#    print(t3.r)
#
#    t5=t3.truncate(rank=3)
#    print((np.linalg.norm(t3.full()-t5.full())))
#
#    print()
#    print((np.mean(t1.full().val) - t1.mean()))
#
#
#    ### test Fourier Hadamard product #####
#    af=t1.set_fft_form('c').fourier()
#    bf=t2.set_fft_form('c').fourier()
#
#    afbf=af*bf
#
#    af2=t1.set_fft_form('sr').fourier()
#    bf2=t2.set_fft_form('sr').fourier()
#
#    afbf2=af2*bf2
#
#    print(( (afbf.fourier()-afbf2.fourier()).norm()))
#    print((t1.norm()))
#
##    v1=np.array([[1,0]])
##    t1=TensorTrain(val=v1)
##    v2=np.array([[14.2126 ,  -1.2689],[  0.0557,    0.6243]])
##    t2=TensorTrain(val=v2 )
##
##    t3=t1.__dot__(t2)
##
##    print t3.full()
##    print t3.full().shape
#
##    c11=t1.get_slice(0,0)
##    c12=t1.get_slice(0,1)
##
##    c21=t1.get_slice(1,0)
##
##    c34=t1.get_slice(2,3)
##
##    chunk1=t1.tt_chunk(1,1)
##
##    rs1=chunk1.get_rank_slice('left',0)
##
##    v1=np.array(range(1,121))
##    v=np.reshape(v1**2, (2, 3, 4, 5  ), order='F') # use 'F' to keep v the same as in matlab
##    t=TensorTrain(val=v)
##
##    chunk1=t.tt_chunk(1,1)
##
##    rs1=chunk1.get_rank_slice('left',0)
##
###    ## test norm###
## #    print t1.norm()
## #    print np.sqrt(np.sum(t1.full()**2))
## #
## #    print t3.norm()
## #    print np.sqrt(np.sum(t3.full()**2))
## #
## #    t3tr=t3.truncate(rank=3)
## #    print t3tr.norm()
## #    print np.sqrt(np.sum(t3tr.full()**2))
##    tp1=t1.tt_chunk(0,1)
##    tp1s1=tp1.get_rank_slice('right',0)
##
##    tp2=t1.tt_chunk(1,2)
##    #tp2s1=tp2.get_rank_slice('right',0)
#
#    print('END')
