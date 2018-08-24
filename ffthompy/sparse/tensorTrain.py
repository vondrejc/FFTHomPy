

#import sys
import numpy as np
from operator import mul
#sys.path.append("/home/disliu/fft_new/ffthompy-sparse")

#from ffthompy.sparse.tensors import SparseTensorFuns
#from ffthompy.tensors.operators import DFT
#from ffthompy.sparse.decompositions import HOSVD,nModeProduct

#from scipy.linalg import block_diag
#
#from numpy.linalg import svd, norm
#from numpy import dot, kron,newaxis, argsort, tensordot, rollaxis
from ffthompy.tensors.operators import DFT
from tt.core.vector import vector

#import timeit

np.set_printoptions(precision=2)
np.set_printoptions(linewidth=999999)

class TensorTrain(vector):
    def __init__(self, a=None, eps=1e-14, rmax=100000, Fourier=False,name=''):
        
        vector.__init__(self, a, eps, rmax)
        
        self.name = name
        self.Fourier = Fourier
        self.N=self.n    ## ttpy use n, we use N.
        
  
#    def fft(self, shift=False):    
#        """ Compute discrete fast Fourier Transform of the tensor.
#        :param shift: Shift the zero-frequency component to the center of the spectrum.
#        :type shift: Boolean 
#        :returns:  TT-vector of Fourier coefficients. 
#        """   
#        cl = self.to_list(self)   
#        clf=[None]*self.d 
#
#        for i in range(self.d ):
#            if shift:
#                pass
#            else:                    
#                clf[i]=np.fft.fft(cl[i], axis=1)  # the axis to compute fft is in the middle of 0,1,2, i.e. 1.
#        
#        res=vector.from_list(clf)
#        res.Fourier = True
#        return res
    
    @staticmethod
    def from_list(a, name='', Fourier=False, order='F'):
        """Generate TT-vectorr object from given TT cores.

        :param a: List of TT cores.
        :type a: list
        :returns: vector -- TT-vector constructed from the given cores.

        """
        d = len(a)  # Number of cores
        
        res = TensorTrain()
        
        n = np.zeros(d, dtype=np.int32)
        r = np.zeros(d+1, dtype=np.int32)
        cr = np.array([])
        for i in xrange(d):
            cr = np.concatenate((cr, a[i].flatten(order)))
            r[i] = a[i].shape[0]
            r[i+1] = a[i].shape[2]
            n[i] = a[i].shape[1]
        res.d = d
        res.n = res.N = n
        res.r = r
        res.core = cr
        res.get_ps()
        res.name=name
        res.Fourier = Fourier
        return res    
    
    def fourier(self):
        "(inverse) discrete Fourier transform"
        if self.Fourier:
            fftfun=lambda Fx, N: np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Fx, axes=1), axis=1), axes=1).real*N
            name='Fi({})'.format(self.name)
        else:
            fftfun=lambda x, N: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=1), axis=1), axes=1)/N
            name='F({})'.format(self.name)

        cl = self.to_list(self)   
        clf=[None]*self.d 
        for i in range(self.d ):
            clf[i]=fftfun(cl[i], self.n[i] )         
        
        res=self.from_list(clf,name=name,Fourier=not self.Fourier) 
        
        return res    
    
    def truncate(self, tol=None, rank=None):
        if  np.any(tol)  is None and np.any(rank) is None:
            return self
        elif  np.any(tol) is None and np.all(rank>=max(self.r))==True :
            return self
        else:
            if tol is None: 
                tol=1e-14
            return self.round(eps=tol,rmax=rank)
    
    def enlarge(self, M):
       
        assert(self.Fourier==True)

        M=np.array(M, dtype=np.int)
        N=np.array(self.n)

        if np.allclose(M, N):
            return self
   
        ibeg=np.ceil(np.array(M-N, dtype=np.float)/2).astype(dtype=np.int)
        iend=np.ceil(np.array(M+N, dtype=np.float)/2).astype(dtype=np.int)
        
        cl = self.to_list(self)   
        dtype=cl[0].dtype
        
        cl_new=[None]*self.d 
        for i in range(self.d ):
           cl_new[i]=  np.zeros([self.r[i], M[i],self.r[i+1]], dtype=dtype)     
           cl_new[i][:, ibeg[i]:iend[i], :]=cl[i]

        res=self.from_list(cl_new) 
        res.name = self.name + "_enlarged"
        res.Fourier = self.Fourier
        return res

    def decrease(self, M):
        assert(self.Fourier is True)

        M=np.array(M, dtype=np.int)
        N=np.array(self.n)
        assert(np.any(np.less(M, N)))

        ibeg=np.fix(np.array(N-M+(M%2), dtype=np.float)/2).astype(dtype=np.int)
        iend=np.fix(np.array(N+M+(M%2), dtype=np.float)/2).astype(dtype=np.int)

        cl = self.to_list(self)   
        
        cl_new=[None]*self.d 
        for i in range(self.d ):
           cl_new[i]=cl[i][:, ibeg[i]:iend[i], :]

        res=self.from_list(cl_new) 
        res.name = self.name + "_decreased"
        res.Fourier = self.Fourier
        return res
    
    def mean(self):
        """
        compute the sum of all elements of the tensor
        """
        cl = self.to_list(self)   
        cl_sum=[None]*self.d 
        
        for i in range(self.d):
            shape=np.ones((self.d+1),dtype=np.int32)
            shape[i:i+2]=self.r[i:i+2]           
            cl_sum[i]=np.sum(cl[i], axis=1).reshape(tuple(shape))
       
        cl_sum_prod= reduce(mul, cl_sum, 1) # product of all arrays inside cl_sum
        
        return np.sum(cl_sum_prod)

    def scal(self, Y):
        X = self
        assert(X.Fourier==Y.Fourier)
        XY = X*Y
        if X.Fourier:
            return XY.mean()
        else:
            return XY.mean()/np.prod(X.n)
        
    def __mul__(self, other):
        res_vec =vector.__mul__(self,other)
        res=TensorTrain()
        res.core= res_vec.core
        res.ps= res_vec.ps
        res.n= res.N = res_vec.n
        res.d= res_vec.d
        res.r= res_vec.r
        res.Fourier = self.Fourier
        #res.name=self.name+'*'+other.name
        return res    
    
    def multiply(self, Y, tol=None, rank=None):
        # element-wise multiplication
        return (self*Y).truncate(tol=tol, rank=rank)
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
         
        
        return ss + vector.__repr__(self)        
    
if __name__=='__main__':
    
    print
    print('----testing "Fourier" function ----')
    print
    
    v1=np.random.rand(120,)
    #v1=np.arange(24)
    n=v1.shape[0]
    
    v = np.reshape(v1,(2,3,4 ,5), order='F') # use 'F' to keep v the same as in matlab
    t=TensorTrain(v) 
    
#    vfft=np.fft.fftn(v)
#    
##    TT=t.to_list(t)
##    TTfft=[None]*t.d
##    
##    for i in range(t.d):
##        TTfft[i]=np.fft.fft(TT[i], axis=1)  # the axis to compute fft is in the middle of 0,1,2, i.e. 1.
##    
##    vTTfft=TensorTrain.from_list(TTfft)
##    #print (vTTfft.full())
##    print(np.linalg.norm(vfft-vTTfft.full()))  
#    
#    tf=t.fft(shift=False)
#    print(np.linalg.norm(vfft-tf.full()))  
    
    tf2 = t.fourier()
    vfft_shift = DFT.fftnc(v, v.shape)
    
    print(np.linalg.norm(vfft_shift -tf2.full())) 
    
    tf2i = tf2.fourier()
    print(np.linalg.norm(v -tf2i.full())) 
    
    ### test mean()#####
    print 
    print np.linalg.norm(t.mean()- np.sum(t.full()))
    
    print t
#  
#    ##   test  enlarge  ###
#    n=3
##    T1= np.zeros((n,n, n))
##    T1[n/3:2*n/3, n/3:2*n/3, n/3:2*n/3]=1
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

## test hardamard product 
    
    v1 = np.random.rand(2,3,4) 
    t1=TensorTrain(v1)     
    v2 = np.random.rand(2,3,4) 
    t2=TensorTrain(v2) 
    
    t3=t1*t2
    print t3
    print(np.linalg.norm( t3.full()-v1*v2))
    print('END')