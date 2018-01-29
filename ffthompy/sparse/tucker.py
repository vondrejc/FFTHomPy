import sys
import numpy as np
#sys.path.append("/home/disliu/fft_new/ffthompy-sparse")

from ffthompy.sparse.tensors import SparseTensorFuns
from scipy.linalg import block_diag
import numpy.fft as fft

from numpy.linalg import svd, norm
from numpy import dot, kron,newaxis, argsort, tensordot, rollaxis
 

def unfold(T, dim):
    """
    Unfolds a tensor T into a matrix, taking the dimension "dim" of T as the first dimension of the matrix, 
    and flattening all the other dimensions into the other one dimension of the matrix.
    
    dim starts from 0.
    """
    return np.rollaxis(T, dim, 0).reshape(T.shape[dim], -1)
    
#def nModeProduct(T, M, n):
#    """
#    n-Mode product of a tensor T(only 2d or 3d for now) and a matrix M,  multiplication and summation is made along the nth dim.
#    definition in paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE
#    
#    n takes value 1, 2, or 3, specify the 1st, 2nd or 3rd dim of the tensor T. 
#    For the matrix M, this function always take the second dimension.
#    """
#    # multiply  along the nth dimension
#    if  len(T.shape) ==2:  # this equals to matrix multiplication T*M or M*T
#        if n==1:
#            return  np.einsum('ik,li->lk', T, M)
#        elif n==2:
#            return  np.einsum('ki,li->kl', T, M) 
#        else:
#            pass 
#        
#    elif  len(T.shape)==3:
#        if n==1:
#            return  np.einsum('ijk,li->ljk', T, M)
#        elif n==2:
#            return  np.einsum('jik,li->jlk', T, M)
#        elif n==3:
#            return  np.einsum('jki,li->jkl', T, M)
#        else:
#            pass 
        
def nModeProduct(T, M, n):
    """
    n-Mode product of a tensor T  and a matrix M,  multiplication and summation is made along the nth dim.
    definition in paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE
    
    n takes value 0, 1, or 2, specify the 1st, 2nd or 3rd dim of the tensor T. 
    For the matrix M, this function always take the second dimension.
    """
 
    P = tensordot(T,M, axes=([n],[1])) 
    return rollaxis(P,len(T.shape)-1, n) 

def nModedivision(T, M, n):
    """
    Inverse function of n-Mode product of a tensor T(only 2d or 3d for now) and a matrix M. 
    i.e. This function returns a tensor T2 such that the n-mode product of T2 and M is T.
    """
     
  
def HOSVD(A):
    r"""
    High order svd of 3-dim tensor A. so that A = S (*1) u1 (*2) u2 (*3) u3, "(*n)" means n-mode product. S: core. u1,u2,u3: orthogonal basis.
    definition in paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE

    """ 
    A1=unfold(A, 0)
    
    u1,s1,vt1= svd( A1)
    u2,s2,vt2= svd( unfold(A, 1))
    u3,s3,vt3= svd( unfold(A, 2)) 
     
    S= dot(dot(u1.T,A1),kron(u2,u3))    
    S = np.reshape(S, A.shape)
    
    return S,u1,u2,u3

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
        
        #Y = self.basisCalibrate(Yo)
        
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
        
        #core = self.core + Y.core
        
        result=Tucker(name=self.name+'+'+Y.name, core=core, basis=newBasis,orthogonal=False)
      
        return result.orthogonalize() 

    def __neg__(self):
        return Tucker(core=-self.core, basis=self.basis)

    def __mul__(self, anotherTensor, tol=None, rank=None):
        "element-wise multiplication of two Tucker tensors" 
        
        # truncate X and Y before multiplication, so that the rank after multiplication 
        # roughly equals to the orginal rank.
        # how much to truncate could be made further adjustable in future versions.
        tRankX = np.ceil(np.sqrt(self.r)).astype(int) 
        tRankY = np.ceil(np.sqrt(anotherTensor.r)).astype(int) 
         
        X = self.truncate(rank=tRankX)
        Y = anotherTensor.truncate(rank=tRankY) 
        
        new_r=X.r*Y.r # this would explode without an truncation of X and Y. 
             
        newCore=np.kron(X.core, Y.core)  
        
        newBasis = [None] * self.order
        for d in range(0, self.order):
            newBasis[d]= np.zeros((new_r[d], X.N[d]))
            for i in range(0, X.r[d]):
                for j in range(0, Y.r[d]):
                    newBasis[d][i*Y.r[d]+j, :]=X.basis[d][i, :]*Y.basis[d][j, :]
             

        result= Tucker(name=X.name+'*'+Y.name, core=newCore, basis=newBasis)
        return  result.orthogonalize() 
       
 
    def orthogonalize(self):
        "re-orthogonalize the basis"  
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
        "Sort the core in term of importance and sort the basis accordinglly"  
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
        "return a full tensor"
        if self.order==2: 
            return np.einsum('ij,ik,jl->kl', self.core, self.basis[0],self.basis[1])
        elif self.order==3:
            # A = S (*1) u1 (*2) u2 (*3) u3, with (*n)  means n-mode product. 
            # from paper "A MULTILINEAR SINGULAR VALUE DECOMPOSITION" by LIEVEN DE LATHAUWER , BART DE MOOR , AND JOOS VANDEWALLE
            temp= nModeProduct(self.core,self.basis[0].T,0)
            temp= nModeProduct(temp,self.basis[1].T,1)
            return nModeProduct(temp,self.basis[2].T,2)
        else:
            raise NotImplementedError()

    def truncate(self, tol=None, rank=None ):
        "return truncated tensor. tol, if presented, would override rank as truncation criteria."       
    
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
        
        basis=list(self.basis) # this copying avoids perturbation to the original tensor object
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
    
    
    N=np.array([40,50])
    a = Tucker(name='a', r=np.array([20,30]), N=N, randomise=True)
    b = Tucker(name='b', r=np.array([40,50]), N=N, randomise=True)
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
    print('testing multiplication...')
    print(np.linalg.norm(c.full()-c2) / np.linalg.norm( c2) )

    #DFT
    print('testing DFT...')
    from ffthompy.operators import DFT
    Fa = a.fourier()
    print(Fa)
    Fa2 = DFT.fftnc(a.full(), a.N)
    print(np.linalg.norm(Fa.full()-Fa2))
#    
    
######### 3-d tenssor test #########
    print
    print('----testing 3d tucker ----')
    print
    
    N1=25  # warning: in 3d multiplication too large N number could kill the machine.
    N2=36
    N3=47 
    
    # creat 3d tensor for test

#    x=np.linspace(-np.pi, np.pi, N1)
#    y=np.linspace(-np.pi, np.pi, N2)
#    z=np.linspace(-np.pi, np.pi, N3)   
#    #this is a rank-1 tensor
#    T=np.sin(x[:,newaxis,newaxis]+y[newaxis,:, newaxis] + z[ newaxis, newaxis, :] ) 
    
    # this is a rank-2 tensor
    T = np.arange(N1*N2*N3)
    T = np.reshape(T,(N1,N2,N3))
    
    # a full rank tensor
    #T = np.random.random((N1,N2,N3))
    
    #decompose the tensor into core and orthogonal basis by HOSVD
    S,u1,u2,u3 = HOSVD(T)
    
    #creat tucker format tensor
    a = Tucker(name='a', core=S, basis=[u1.T, u2.T, u3.T], orthogonal=True )  
    print(a)    
    
    print('testing 3d tucker representation error...')
    print "a.full - T = ",  norm(a.full()-T)
    
#    #decompose the tensor into core and orthogonal basis by HOSVD
#    T2=np.sin(T)
    
    # this is a rank-2 tensor
    T2= np.sqrt(T)
    S2,u12,u22,u32 = HOSVD(T2)
    
    
    #creat tucker format tensor
    b = Tucker(name='b', core=S2, basis=[u12.T, u22.T, u32.T], orthogonal=True)   
    print(b)
    
    b_trunc= b.truncate(rank=[6, 8, 10])  
    print(b_trunc)     
     
    print('testing 3d tucker core sorting and  rank-based truncation ...')
    print "b_truncated.full - b.full = ",  norm(b_trunc.full()-b.full()) 
    print "norm(b_truncated.full - b.full)/norm(b.full) = ",  norm(b_trunc.full()-b.full())/norm(b.full()) 
    print       
    
    b_trunc= b.truncate(tol=[1e-8, 1e-8, 1e-8])  
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
    
   
    print('END')