

#import sys
import numpy as np
#sys.path.append("/home/disliu/fft_new/ffthompy-sparse")

from ffthompy.sparse.tensors import SparseTensorFuns
from ffthompy.tensors.operators import DFT
from ffthompy.sparse.decompositions import HOSVD,nModeProduct

from scipy.linalg import block_diag
import numpy.fft as fft

from numpy.linalg import svd, norm
from numpy import dot, kron,newaxis, argsort, tensordot, rollaxis

import timeit

np.set_printoptions(precision=2) 
 


class Tucker(SparseTensorFuns):

    def __init__(self, name='', core=None, basis=None, Fourier=False, orthogonal=False,
                 r=[3,3], N=[5,5], randomise=False):
        self.name=name
        self.Fourier=Fourier
        self.orthogonal = orthogonal
        if core is not None and basis is not None:
            self.order=basis.__len__()
            self.basis=basis
            self.core=core
            self.r=np.empty(self.order).astype(int)
            self.N=np.empty(self.order).astype(int)
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
        
        r_new = X.r + Y.r 
   
        if X.order==2:
            core= block_diag(X.core,Y.core)
        elif X.order==3:
            core=np.zeros((r_new[0],r_new[1],r_new[2] ))
            #  block_diag(X.core,Y.core) in 3d
            core[:X.r[0] ,:X.r[1],: X.r[2] ]=X.core
            core[ X.r[0]:, X.r[1]:, X.r[2]:]=Y.core 
        else:
            pass
                
        newBasis=[np.vstack([X.basis[ii],Y.basis[ii]]) for ii in range(X.order)]   
        
        result=Tucker(name=self.name+'+'+Y.name, core=core, basis=newBasis,orthogonal=False)
      
        return result.orthogonalize() 

#    def add(self, Y, tol=None, rank=None):
#        
#        X=self 
#        
#        r_new = X.r + Y.r 
#        core=np.zeros(r_new )
#        
#        x_id = [None] * X.order
#        y_id = [None] * Y.order
#        
#        for d in range(0, self.order):
#            x.id[d]= range(0, X.r[d])
#            y.id[d]= range(X.r[d], X.r[d] + Y.r[d])
#        
#        ixgrid = np.ix_([0, 1], [2, 4])

    def __neg__(self):
        return Tucker(core=-self.core, basis=self.basis)


    def __mul__(self, anotherTensor, tol=None, rank=None):
        """element-wise multiplication of two Tucker tensors"""
        
        # truncate X and Y before multiplication, so that the rank after multiplication 
        # roughly equals to the orginal rank.
        # how much to truncate could be made further adjustable in future versions.
#        tRankX = np.ceil(np.sqrt(self.r)).astype(int) 
#        tRankY = np.ceil(np.sqrt(anotherTensor.r)).astype(int) 
#         
#        X = self.truncate(rank=tRankX)
#        Y = anotherTensor.truncate(rank=tRankY) 
        X=self
        Y=anotherTensor
        #new_r=X.r*Y.r # this could explode    
       
        newCore=np.kron(X.core, Y.core)  
        
        newBasis = [None] * self.order
        for d in range(0, self.order):
            #newBasis[d]= np.zeros((new_r[d], X.N[d]))
            newBasis[d]= np.multiply(X.basis[d][:,newaxis, :], Y.basis[d][newaxis,:,:] )
            newBasis[d]= np.reshape(newBasis[d], (-1, X.N[d]))            
            #for i in range(0, X.r[d]):
                #for j in range(0, Y.r[d]):
                    #newBasis[d][i*Y.r[d]+j, :]=X.basis[d][i, :]*Y.basis[d][j, :]
             

        result= Tucker(name=X.name+'*'+Y.name, core=newCore, basis=newBasis)
        return  result.truncate(rank= X.N) 
        #return  result.truncate(rank=list(np.max(np.vstack([X.r,Y.r]),axis=0 ))  ) 
        
    def orthogonalize(self):
        """re-orthogonalize the basis""" 
        newBasis=[]
        R=[]
        # orthogonalize the basis
        for i in range(0, self.order):
            q, r1 =np.linalg.qr(self.basis[i].T)
            newBasis.append(q.T)
            R.append(r1)
        
        # transform the core in term of the new bases
        core=self.core
        for i in range(0, self.order):  
            core = nModeProduct(core, R[i], i)   
        
        return Tucker(name=self.name, core=core, basis=newBasis, orthogonal=True)   
          
    def sortBasis(self):
        """Sort the core in term of importance and sort the basis accordinglly"""  
        core=self.core
        basis=self.basis
        
        if self.order==2:
            # sort the 2-norm of the rows
            ind0 = argsort(norm(core, axis=1))[::-1]    
            core =core[ind0,:]           
            # sort the 2-norm of the columns
            ind1= argsort(norm(core, axis=0))[::-1]            
            core =core[:,ind1]
            
            basis[0]=basis[0][ind0,:]
            basis[1]=basis[1][ind1,:]
            
        elif self.order==3:
            # sort the 2-norm of the horizontal slices
            ind0 = argsort(norm(core, axis=(1,2)))[::-1]
            core =core[ind0,:,:]           
            # sort the 2-norm of the vertical slices
            ind1= argsort(norm(core, axis=(0,2)))[::-1]        
            core =core[:,ind1,:]
            # sort the 2-norm of the other way  slices
            ind2= argsort(norm(core, axis=(0,1)))[::-1]          
            core =core[:,:,ind2]
            
            basis[0]=basis[0][ind0,:]
            basis[1]=basis[1][ind1,:]                   
            basis[2]=basis[2][ind2,:]    
        
        return Tucker(name=self.name, core=core, basis=basis, orthogonal=self.orthogonal) 
             
    def full(self):
        """convert a tucker representation to a full tensor        
        A = CORE (*1) Basis1 (*2) Basis2 (*3) Basis3 ..., with (*n)  means n-mode product. 
        from paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE
        """
        d = self.N.shape[0]
        CBd= nModeProduct(self.core,self.basis[0].T,0)
        for i in range(1, d):
            CBd= nModeProduct(CBd,self.basis[i].T,i)
        return CBd  
 

    def truncate(self, tol=None, rank=None ):
        """return truncated tensor. tol, if presented, would override rank as truncation criteria.        
        """
        if  np.any(tol)  is None and np.any(rank) is None:
            print ("Warning: No truncation criteria input, truncation aborted!")
            return self             
        elif  np.any(tol) is None and np.any(rank >= self.r)==True :
            print ("Warning: Truncation rank not smaller than the original ranks, truncation aborted!")
            return self 
 
        # truncation
        
        if not self.orthogonal:
            self=self.orthogonalize()   
        
        self=self.sortBasis() 
        
        basis=list(self.basis)  
        core=self.core 
        
        # to determine the rank of truncation        
        if np.any(tol) is not None:
            if rank==None: rank=np.zeros((self.order),dtype=int)
            # determine the truncation rank so that (1.0-tol)*100% of the norm of the core in that direction  is perserved.
            if  self.order==2:
                sorted_dim0 = norm(core, axis=1)
                sorted_dim1 = norm(core, axis=0)
                
                rank[0]=np.searchsorted(np.cumsum(sorted_dim0)/np.sum(sorted_dim0), 1.0-tol[0])+1
                rank[1]=np.searchsorted(np.cumsum(sorted_dim1)/np.sum(sorted_dim1), 1.0-tol[1])+1
                
            elif self.order==3:
                sorted_dim0 = norm(core, axis=(1,2))
                sorted_dim1 = norm(core, axis=(0,2))
                sorted_dim2 = norm(core, axis=(0,1))
           
                rank[0]=np.searchsorted(np.cumsum(sorted_dim0)/np.sum(sorted_dim0), 1.0-tol[0])+1
                rank[1]=np.searchsorted(np.cumsum(sorted_dim1)/np.sum(sorted_dim1), 1.0-tol[1])+1
                rank[2]=np.searchsorted(np.cumsum(sorted_dim2)/np.sum(sorted_dim2), 1.0-tol[2])+1
                
        if self.order==2:
            core=core[:rank[0],:rank[1]]  
        elif self.order==3: 
            core=core[:rank[0],:rank[1], :rank[2]]  
        else:
            raise NotImplementedError("currently only support two and three dimensional tensor")
        
        for ii in range(self.order):
            basis[ii]=basis[ii][:rank[ii], :]

        return Tucker(name=self.name+'_truncated', core=core, basis=basis, orthogonal=True)
    
    def decrease(self, M):
        assert(self.Fourier is True)

        M = np.array(M, dtype=np.int)
        N = np.array(self.N)
        assert(np.all(np.less(M, N)))

        ibeg = np.fix(np.array(N-M+(M % 2), dtype=np.float)/2).astype(dtype=np.int)
        iend = np.fix(np.array(N+M+(M % 2), dtype=np.float)/2).astype(dtype=np.int)

        basis=[]
        for ii in range(N.size):
            basis.append(self.basis[ii][:,ibeg[ii]:iend[ii]])

        return Tucker(name=self.name, core=self.core, basis=basis, Fourier=self.Fourier)

    def __repr__(self, full=False, detailed=False):
        keys = ['name', 'Fourier', 'orthogonal', 'N', 'r']
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
     
    
    N=np.array([3,4])
    a = Tucker(name='a', r=np.array([3,4]), N=N, randomise=True)
    b = Tucker(name='b', r=np.array([5,6]), N=N, randomise=True)
    print(a)
    print(b)
    
    # addition
    c = a+b
    print(c)   
    
    c2 = a.full()+b.full()
    print('testing addition...')
    print(np.linalg.norm(c.full()-c2) / np.linalg.norm( c2))
    
    c_ortho = c.orthogonalize() 
    print('testing addition and then orthogonalize ...')
    print(np.linalg.norm(c.full() - c_ortho.full())/np.linalg.norm( c_ortho.full()) )
    print
 
    # multiplication
    c = a*b
    c2 = a.full()*b.full()
    print c
    print('testing multiplication...')
    print(np.linalg.norm(c.full()-c2) / np.linalg.norm( c2) )

    #DFT
    print('testing DFT...')
    
    
    
    Fa = a.fourier()
    print(Fa)
    Fa2 = DFT.fftnc(a.full(), a.N)
    print(np.linalg.norm(Fa.full()-Fa2))
#    
    
######### 3-d tenssor test #########
    print
    print('----testing 3d tucker ----')
    print
    
    N1=60  # warning: in 3d multiplication too large N number could kill the machine.
    N2=30
    N3=40 
    
    # creat 3d tensor for test

#    x=np.linspace(-np.pi, np.pi, N1)
#    y=np.linspace(-np.pi, np.pi, N2)
#    z=np.linspace(-np.pi, np.pi, N3)   
#    #this is a rank-1 tensor
#    T=np.sin(x[:,newaxis,newaxis]+y[newaxis,:, newaxis] + z[ newaxis, newaxis, :] ) 
    
    # this is a rank-2 tensor
#    T = np.arange(N1*N2*N3)
#    T = np.reshape(T,(N1,N2,N3))
    
    #a full rank tensor
    T = np.random.random((N1,N2   ))
    
#    #decompose the tensor into core and orthogonal basis by HOSVD
    #S2,U2 = HOSVD2(T)
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
     

   
    S,U = HOSVD(T)
    #creat tucker format tensor
     
    for i in range(0,len(U)):
        U[i] =U[i].T
    
    a = Tucker(name='a', core=S, basis=U, orthogonal=True )  
    print(a)    
    
    
    print('testing nd tucker representation error...')
    print "a.full - T = ",  norm(a.full()-T)
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
    #print sss
#    #decompose the tensor into core and orthogonal basis by HOSVD
#    T2=np.sin(T)
    
    # this is a rank-2 tensor
    T2= np.sqrt(T)
    S2,U2 = HOSVD(T2)
    
    basis = U2
    for i in range(0,len(basis)):
        basis[i] = basis[i].T
    #creat tucker format tensor
    b = Tucker(name='b', core=S2, basis=basis, orthogonal=True)   
    print(b)
    
    b_trunc= b.truncate(rank=[6, 8])  
    print(b_trunc)     
     
    print('testing 3d tucker core sorting and  rank-based truncation ...')
    print "b_truncated.full - b.full = ",  norm(b_trunc.full()-b.full()) 
    print "norm(b_truncated.full - b.full)/norm(b.full) = ",  norm(b_trunc.full()-b.full())/norm(b.full()) 
    print       
    
    b_trunc= b.truncate(tol=[1e-6, 1e-6])  
    print(b_trunc)     
     
    print('testing 3d tucker  tol-based truncation ...')
    print "b_truncated.full - b.full = ",  norm(b_trunc.full()-b.full()) 
    print "norm(b_truncated.full - b.full)/norm(b.full) = ",  norm(b_trunc.full()-b.full())/norm(b.full()) 
    print     
     
    c=a+b
    print(c)
    print
    
    print('testing 3d re-orthogonalization after addition ...')
    print "(a+b).full - (a.full+b.full) = ",  norm(c.full()-a.full()-b.full()) 
    print 
    
    
    # multiplication
    c = a*b
    print(c)
    
    print('testing 3d multiplication and re-orthogonalization...')
    print "(a*b).full - (a.full*b.full) = ",  norm(c.full()-a.full()*b.full())
    print "((a*b).full - (a.full*b.full))/|(a.full*b.full)| = ",  norm(c.full()-a.full()*b.full())/norm(a.full()*b.full())
    print "max((a*b).full - (a.full*b.full))/mean(a.full*b.full) = ",  np.max(c.full()-a.full()*b.full())/np.mean(a.full()*b.full())
    
 
    
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
#
#    print('Comparing time cost of tensor of 1-D FFT and n-D FFT ...')
#    t1=timeit.timeit("a.fourier()", setup='from __main__ import a', number=10)
#    afull=a.full()
#    t2=timeit.timeit("DFT.fftnc(afull, a.N)", setup='from ffthompy.tensors.operators import DFT;from __main__ import a, afull', number=10)
#    # t1=timeit.timeit("aa=a.truncate(tol=0.05); aa.fourier()", setup='from __main__ import a', number=10000)
#    print
#    print "Tensor of 1D FFT costs: %f"%t1
#    print "n-D FFT costs         : %f"%t2
#
#    print('END')   
   
    print('END')