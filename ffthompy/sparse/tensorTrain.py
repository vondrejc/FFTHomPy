#import sys
import numpy as np
from operator import mul

from numpy import reshape, dot
from numpy.linalg import qr
from scipy.linalg import rq

from ffthompy.tensors.operators import DFT
from tt.core.vector import vector

#import timeit

np.set_printoptions(precision=3)
np.set_printoptions(linewidth=999999)

class TensorTrain(vector):
    def __init__(self, val=None, core=None, eps=None, rmax=None, Fourier=False, name='unnamed', vectorObj=None):

        if eps is None:  eps = 1e-14
        if rmax is None: rmax=999999
            
        if val is not None:
            vector.__init__(self, val, eps, rmax)
            self.N=self.n # ttpy use n, we use N.
        elif core is not None:
            self = self.from_list(core,name=name, Fourier=Fourier)
        elif vectorObj is not None: # cast a TTPY object to tensorTrain object
            for attr_name in vectorObj.__dict__:
                setattr(self, attr_name, getattr(vectorObj, attr_name))
            self.N=self.n # ttpy use n, we use N.
        else: # a  3D zero tensor
            vector.__init__(self, np.zeros((3,4,5)), eps, rmax)
            self.N=self.n # ttpy use n, we use N.       
            
        self.name=name
        self.Fourier=Fourier
        
    @staticmethod
    def from_list(a, name='unnamed', Fourier=False):
        """Generate TT-vectorr object from given TT cores.

        :param a: List of TT cores.
        :type a: list
        :returns: vector -- TT-vector constructed from the given cores.

        """
        res_vec=vector.from_list(a)

        res=TensorTrain(vectorObj=res_vec, name=name, Fourier=Fourier)

        return res

    def fourier(self):
        "(inverse) discrete Fourier transform"
        if self.Fourier:
            fftfun=lambda Fx, N: np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Fx, axes=1), axis=1), axes=1).real*N
            name='Fi({})'.format(self.name)
        else:
            fftfun=lambda x, N: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=1), axis=1), axes=1)/N
            name='F({})'.format(self.name)

        cl=self.to_list(self)
        clf=[None]*self.d
        for i in range(self.d):
            clf[i]=fftfun(cl[i], self.n[i])

        res=self.from_list(clf, name=name, Fourier=not self.Fourier)

        return res

    def truncate(self, tol=None, rank=None):
        if np.any(tol) is None and np.any(rank) is None:
            return self
        elif np.any(tol) is None and np.all(rank>=max(self.r))==True :
            return self
        else:
            if tol is None:  tol=1e-14

            res_vec=self.round(eps=tol, rmax=rank) # round produces a TTPY vetcor object
            res=TensorTrain(vectorObj=res_vec, name=self.name+'_truncated', Fourier=self.Fourier)
            return res

    def enlarge(self, M):
        assert(self.Fourier is True)

        M=np.array(M, dtype=np.int)
        N=np.array(self.n)

        if np.allclose(M, N):
            return self

        ibeg=np.ceil(np.array(M-N, dtype=np.float)/2).astype(dtype=np.int)
        iend=np.ceil(np.array(M+N, dtype=np.float)/2).astype(dtype=np.int)

        cl=self.to_list(self)
        dtype=cl[0].dtype

        cl_new=[None]*self.d
        for i in range(self.d):
            cl_new[i]=np.zeros([self.r[i], M[i], self.r[i+1]], dtype=dtype)
            cl_new[i][:, ibeg[i]:iend[i], :]=cl[i]

        res=self.from_list(cl_new)
        res.name=self.name+"_enlarged"
        res.Fourier=self.Fourier
        return res

    def decrease(self, M):
        assert(self.Fourier is True)

        M=np.array(M, dtype=np.int)
        N=np.array(self.n)
        assert(np.any(np.less(M, N)))

        ibeg=np.fix(np.array(N-M+(M % 2), dtype=np.float)/2).astype(dtype=np.int)
        iend=np.fix(np.array(N+M+(M % 2), dtype=np.float)/2).astype(dtype=np.int)

        cl=self.to_list(self)

        cl_new=[None]*self.d
        for i in range(self.d):
            cl_new[i]=cl[i][:, ibeg[i]:iend[i], :]

        res=self.from_list(cl_new)
        res.name=self.name+"_decreased"
        res.Fourier=self.Fourier
        return res

    def mean(self):
        """
        compute the sum of all elements of the tensor
        """
        cl=self.to_list(self)
        cl_sum=[None]*self.d

        for i in range(self.d):
            shape=np.ones((self.d+1), dtype=np.int32)
            shape[i:i+2]=self.r[i:i+2]
            cl_sum[i]=np.sum(cl[i], axis=1).reshape(tuple(shape))

        cl_sum_prod=reduce(mul, cl_sum, 1) # product of all arrays inside cl_sum

        return np.sum(cl_sum_prod)

    def scal(self, Y):
        X=self
        assert(X.Fourier==Y.Fourier)
        XY=X*Y
        if X.Fourier:
            return XY.mean()
        else:
            return XY.mean()/np.prod(X.n)

    def __mul__(self, other):
        res_vec=vector.__mul__(self, other)
        res=TensorTrain(vectorObj=res_vec,
                          name=self.name+'*'+(str(other) if isinstance(other, (int, long, float, complex)) else other.name),
                          Fourier=self.Fourier)
        return res

    def __add__(self, other):
        res_vec=vector.__add__(self, other)
        res=TensorTrain(vectorObj=res_vec,
                          name=self.name+'+'+(str(other) if isinstance(other, (int, long, float, complex)) else other.name),
                          Fourier=self.Fourier)
        return res

    def __sub__(self, other):

        res=self+(-other)
        res.name=self.name+'-'+(str(other) if isinstance(other, (int, long, float, complex)) else other.name)

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
    # Print statement
    def __repr__(self):

        keys=['name', 'Fourier', 'n', 'r']
        ss="Class : {0}({1}) \n".format(self.__class__.__name__, self.d)
        skip=4*' '
        nstr=np.array([key.__len__() for key in keys]).max()

        for key in keys:
            attr=getattr(self, key)
            if callable(attr):
                ss+='{0}{1}{3} = {2}\n'.format(skip, key, str(attr()), (nstr-key.__len__())*' ')
            else:
                ss+='{0}{1}{3} = {2}\n'.format(skip, key, str(attr), (nstr-key.__len__())*' ')

        return ss+vector.__repr__(self)

    def orthogonalize(self, option='lr'):
        
        d=self.d
        r=self.r
        n=self.n
        cr=self.to_list(self)
        cr_new = [None]*d
        
        if option=='lr' or option=='LR':
            # qr sweep from left to right
            for i in range(d-1):            
                cr[i]=reshape(cr[i],(r[i]*n[i],r[i+1]) )
                cr_new[i], ru = qr(cr[i], 'reduced')
                cr_new[i]=reshape(cr_new[i],(r[i],n[i],r[i+1]) )
                cr[i+1]= dot(ru, reshape(cr[i+1],(r[i+1],n[i+1]*r[i+2]) ))
    
            cr[d-1]=reshape(cr[d-1],(r[d-1]*n[d-1],r[d]) )
            cr_new[d-1], ru = qr(cr[d-1], 'reduced')   
            cr_new[d-1]=reshape(cr_new[d-1]*ru,(r[d-1],n[d-1],r[d]) )
            
            return (self.from_list(cr_new))
        
        elif option=='rl' or option=='RL':
            # rq sweep from right to left
            cr[d-1]=reshape(cr[d-1],(r[d-1], n[d-1]*r[d]) )
            ru, cr_new[d-1] = rq(cr[d-1], mode='economic')   
            cr_new[d-1]=reshape(cr_new[d-1],(r[d-1],n[d-1],r[d]) )
            
            for i in range(d-2,-1,-1):   
                cr[i]= dot(reshape(cr[i],(r[i]*n[i],r[i+1]) ), ru)            
                ru, cr_new[i] = rq(reshape(cr[i],(r[i],n[i]*r[i+1] )), mode='economic')   
                cr_new[i]=reshape(cr_new[i],(r[i],n[i],r[i+1]) )        
            
            cr_new[i]*=ru
            return (self.from_list(cr_new))
        else:
            raise ValueError("Unexpected parameter '" + option +"' at tt.vector.tt_qr")
            
if __name__=='__main__':

    print
    print('----testing "Fourier" function ----')
    print

    v1=np.random.rand(120,)
    # v1=np.arange(24)
    n=v1.shape[0]

    v=np.reshape(v1, (2, 3, 4 , 5), order='F') # use 'F' to keep v the same as in matlab
    t=TensorTrain(v,rmax=3)
    

#    vfft=np.fft.fftn(v)
#
# #    TT=t.to_list(t)
# #    TTfft=[None]*t.d
# #
# #    for i in range(t.d):
# #        TTfft[i]=np.fft.fft(TT[i], axis=1)  # the axis to compute fft is in the middle of 0,1,2, i.e. 1.
# #
# #    vTTfft=TensorTrain.from_list(TTfft)
# #    #print (vTTfft.full())
# #    print(np.linalg.norm(vfft-vTTfft.full()))
#
#    tf=t.fft(shift=False)
#    print(np.linalg.norm(vfft-tf.full()))

    tf2=t.fourier()
    vfft_shift=DFT.fftnc(v, v.shape)

    print(np.linalg.norm(vfft_shift-tf2.full()))

    tf2i=tf2.fourier()
    print(np.linalg.norm(v-tf2i.full()))

    ### test mean()#####
    print
    print np.linalg.norm(t.mean()-np.sum(t.full()))

    print t
#
    # ## test casting a vector obj to tensortrain obj
    ty=vector(v)
    ty2=TensorTrain(vectorObj=ty)

#    ##   test  enlarge  ###
#    n=3
# #    T1= np.zeros((n,n, n))
# #    T1[n/3:2*n/3, n/3:2*n/3, n/3:2*n/3]=1
#
#    T1= np.zeros((n,n ))
#    T1[n/3:2*n/3, n/3:2*n/3 ]=1
#
#    print(T1)
#    print
#
#    t=TensorTrain(T1)
#    tf=t.fourier()
#    tfl=tf.enlarge([7,7])
#    tfli= tfl.fourier()
#
#    print(tfli.full().real)
#    print
#
#    cl_tf=tf.to_list(tf)
#    cl_tfl=tfl.to_list(tfl)
#    cl_tfli=tfli.to_list(tfli)
#
#    tfld=tfl.decrease([5,5])
#    print(tfld.fourier().full())

# # test hardamard product

    v1=np.random.rand(2, 3, 4)
    t1=TensorTrain(v1)
    v2=np.random.rand(2, 3, 4)
    t2=TensorTrain(v2)

    t3=t1*t2
    print t3
    print(np.linalg.norm(t3.full()-v1*v2))

    t4=t1+t2
    print(np.linalg.norm(t4.full()-v1-v2))
    #print t4

    t4o=t4.orthogonalize()
    print(np.linalg.norm(t4o.full()-t4.full()))

    #print t5.size

    print('END')
