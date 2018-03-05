from __future__ import division
import numpy as np
import scipy.sparse.linalg as sp
import itertools
import sys
from ffthompy import Timer
from ffthompy.trigpol import mean_index, Grid
import ffthompy.tensors.projection as proj
from ffthompy.tensors.operators import grad_tensor
from ffthompy.sparse.projection import grad_tensor as sgrad_tensor
from ffthompy.general.solver import CG
# from ffthompy.general.solver import richardson

# PARAMETERS ##############################################################
dim  = 2            # number of dimensions (works for 2D and 3D)
N    = dim*(105,)   # number of voxels (assumed equal for all directions)
Y    = np.ones(dim)
Amax = 10.          # material contrast

maxiter=1e2

tol=None
rank=25

calc_eigs = 1
debug=False

# auxiliary values
prodN = np.prod(np.array(N)) # number of grid points
ndof  = dim*prodN # number of degrees-of-freedom
vec_shape=(dim,)+N # shape of the vector for storing DOFs

# PROBLEM DEFINITION ######################################################
# A = np.einsum('ij,...->ij...',np.eye(dim),1.+10.*np.random.random(N)) # material coefficients
P=int(N[0]/3)
phase  = np.ones(N); phase[:P,:P] = Amax
A = np.einsum('ij,...->ij...', np.eye(dim), phase)
E = np.zeros(vec_shape); E[0] = 1. # set macroscopic loading

# PROJECTION IN FOURIER SPACE #############################################
_, Ghat, _ = proj.scalar(N, Y)

# OPERATORS ###############################################################
dot21  = lambda A,v: np.einsum('ij...,j...  ->i...',A,v)
fft    = lambda V: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(V),N))/np.prod(N)
ifft   = lambda V: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(V),N))*np.prod(N)
G_fun  = lambda V: np.real(ifft(dot21(Ghat.val,fft(V)))).reshape(-1)
A_fun  = lambda v: dot21(A,v.reshape(vec_shape))
GA_fun = lambda v: G_fun(A_fun(v))

# CONJUGATE GRADIENT SOLVER ###############################################
print('\n== CG solver for gradient field in real space =======================')
b = -GA_fun(E) # right-hand side
tic=Timer(name='scipy.cg (gradient)')
e, _=sp.cg(A=sp.LinearOperator(shape=(ndof, ndof), matvec=GA_fun, dtype='float'), b=b,
           tol=1e-10, maxiter=maxiter)
tic.measure()

aux = e+E.reshape(-1)
print('norm(residuum(e))={}'.format(np.linalg.norm(b-GA_fun(e))))
print('homogenised properties A11 = {}'.format(np.inner(A_fun(aux).reshape(-1), aux)/prodN))

### POTENTIAL SOLVER in real space #####################################################
print('\nGenerating operators for formulation with potential...')

# GRADIENT IN FOURIER SPACE #############################################
hGrad=grad_tensor(N, Y)
k2 = np.einsum('i...,i...', hGrad.val, np.conj(hGrad.val)).real
k2[mean_index(N)]=1.

# OPERATORS ###############################################################
FGrad = lambda Fu: np.einsum('i...,...->i...', hGrad.val, Fu)
FDiv = lambda Fe: -np.einsum('i...,i...->...', hGrad.val, Fe)
Grad = lambda u: ifft(FGrad(fft(u))).real
Div = lambda e: ifft(FDiv(fft(e))).real
DivAGrad_fun = lambda u: Div(dot21(A, Grad(u.reshape(N)))).ravel()
GFAFG_fun=lambda Fu:FDiv(fft(dot21(A, ifft(FGrad(Fu)))))
GFAFG_funvec=lambda Fu: GFAFG_fun(Fu.reshape(N)).ravel()
Pfun=lambda X: X/k2
PGFAFG_fun=lambda X: Pfun(GFAFG_fun(X))
B=-FDiv(fft(dot21(A, E))) # right-hand side

print('\n== CG solver with precond.for potential in Fourier space =======================')
k2r=k2**0.5
Nfun = lambda X: X/k2r
GnFAFGn_fun=lambda Fu:Nfun(FDiv(fft(dot21(A, ifft(FGrad(Nfun(Fu))))))) # minus necessary here
GnFAFGn_funvec=lambda Fu: GnFAFGn_fun(Fu.reshape(N)).ravel()
Bn=-Nfun(FDiv(fft(dot21(A, E))))

linoper=sp.LinearOperator(shape=(prodN, prodN), matvec=GnFAFGn_funvec, dtype=np.float)
tic=Timer(name='scipy.cg (potential - precond)')
Fu, _=sp.cg(A=linoper, b=Bn.ravel(), maxiter=maxiter, tol=1e-10)
tic.measure()
Fu = Fu.reshape(N)/k2r

e3=ifft(FGrad(Fu)).real
aux=e3+E
print('norm(residuum)={}'.format(np.linalg.norm(B-GFAFG_fun(Fu))))
print('norm(residuum(e))={}'.format(np.linalg.norm(b-GA_fun(e3))))
print('homogenised properties A11 = {}'.format(np.sum(dot21(A, aux)*aux)/prodN))
print('norm(e-e3)={}'.format(np.linalg.norm(e-e3.ravel())))

print('\n== CG - own solver =====')
scal = lambda X,Y: np.sum(X*Y.conj()).real

tic=Timer(name='own CG (potential)')
Fu, res=CG(Afun=GnFAFGn_fun, B=Bn, x0=np.zeros_like(B), par={'maxiter': maxiter,
                                                             'scal': scal})
tic.measure()

Fu=Fu/k2r
e=ifft(FGrad(Fu)).real
aux=e+E
print('norm(residuum)={}'.format(np.linalg.norm(B-GFAFG_fun(Fu.reshape(N)))))
print('norm(residuum(e))={}'.format(np.linalg.norm(b-GA_fun(e))))
print('homogenised properties A11 = {}'.format(np.sum(dot21(A, aux)*aux)/prodN))
print(res)

## RICHARDSON iteration #######################################################
from ffthompy.sparse.solver import richardson as richardson_s

### PRECONDITIONING in Fourier space #####################################################
print('\n== Richardson with preconditioning for potential field in Fourier space=====')
normfun=lambda X: X.norm()

parP={'alpha': (1.+Amax)/2.,
      'maxiter': maxiter,
      'tol': 1e-6,
      'norm': normfun}
B=-FDiv(fft(dot21(A, E))) # right-hand side
from ffthompy.general.solver import richardson
tic=Timer(name='Richardson (full)')
Fu3, res3=richardson(Afun=GnFAFGn_fun, B=Bn, x0=np.zeros_like(Bn), par=parP)
tic.measure()
Fu3 = Fu3.reshape(N)
print('norm(resP)={}'.format(np.linalg.norm(Bn-GnFAFGn_fun(Fu3))))
Fu3 = Fu3/k2r

print('iterations={}'.format(res3['kit']))
print('norm(dif)={}'.format(np.linalg.norm(Fu-Fu3)))
print('norm(resP)={}'.format(res3['norm_res']))
print('norm(res)={}'.format(np.linalg.norm(B-GFAFG_fun(Fu3))))

# print(sp.eigsh(linoper, k=2, which='LM', return_eigenvectors=False))
# print(sp.eigsh(linoper, k=2, which='SM', return_eigenvectors=False))

print('\n== Generating operators for SPARSE solver...')
from ffthompy.sparse import decompositions
from ffthompy.sparse.canoTensor import CanoTensor

# matrix A
Abas, k, max_err=decompositions.dCA_matrix_input(A[0, 0], k=3, tol=1e-14)
normAbas=np.linalg.norm(Abas, axis=0)
Abas=Abas/np.atleast_2d(normAbas)
Abas=Abas.T
As=CanoTensor(name='A', core=normAbas**2, basis=[Abas, Abas])

# Grad sparse
hGrad_s = sgrad_tensor(N, Y)

# R.H.S.
Es=CanoTensor(name='E', core=np.array([1.]),
               basis=[np.atleast_2d(np.ones(N[ii])) for ii in range(dim)], Fourier=False)

Bs=(hGrad_s[0]*(As*Es).fourier()) # two-times minus
print(np.linalg.norm(B-Bs.full()))

# linear operator
def GFAFG_fun_s(Fx, rank=rank, tol=tol):
    FGFx=[(hGrad_s[ii]*Fx).fourier() for ii in range(dim)]
    AFGFx=[As.multiply(FGFx[ii], rank=None, tol=None) for ii in range(dim)]
    AFGFx=[AFGFx[ii].truncate(rank=rank, tol=tol) for ii in range(dim)]
    FAFGFx=[AFGFx[ii].fourier() for ii in range(dim)]
    GFAFGFx=hGrad_s[0]*FAFGFx[0]
    for ii in range(1, dim):
        GFAFGFx+=hGrad_s[ii]*FAFGFx[ii]
    GFAFGFx=GFAFGFx.truncate(rank=rank, tol=tol)
    GFAFGFx.name='fun(x)'
    return -GFAFGFx

# testing
print('testing linear operator of system...')
x=CanoTensor(name='a', r=4, N=N, randomise=True)
Fx=x.fourier()
res1=GFAFG_fun_s(Fx, rank=None, tol=None)
res2=GFAFG_fun(fft(x.full()))
print(np.linalg.norm(res1.full()-res2))

norm=lambda X: np.linalg.norm(X.full())

print('\n== SPARSE Richardson solver with preconditioner =======================')
# preconditioner
from scipy.linalg import svd
Prank = 8
U, S, Vh=sp.svds(1./k2, k=Prank, which='LM')
# U,S,Vh = svd(1./k2)
print('singular values of P={}'.format(S))
Ps=CanoTensor(name='P', core=S[:Prank], basis=[U[:, :Prank].T, Vh[:Prank]], Fourier=True)
Ps.sort()
print('norm(P-Ps)={}'.format(np.linalg.norm(1./k2-Ps.full())))
# sys.exit()

def PGFAFG_fun_s(Fx, rank=rank, tol=tol):
    R=GFAFG_fun_s(Fx, rank=rank, tol=tol)
    R=Ps*R
    R=R.truncate(rank=rank, tol=tol)
    return R

# testing
print('testing precond. linear system...')
x=CanoTensor(name='a', r=4, N=N, randomise=True)
Fx=x.fourier()
res1=PGFAFG_fun_s(Fx, rank=None, tol=None)
res2=PGFAFG_fun(fft(x.full()))
print(np.linalg.norm(res1.full()-res2))
# sys.exit()

PBs=Ps*Bs
print('\nsolver results...')
parP['tol']=1e-8
tic=Timer(name='Richardson (sparse)')
Fus, ress=richardson_s(Afun=PGFAFG_fun_s, B=PBs, par=parP, norm=norm, rank=rank, tol=tol)
tic.measure()
Fus.name='Fus'
Fus = Fus

print(Fus)
print('iterations={}'.format(ress['kit']))
print('norm(dif)={}'.format(np.linalg.norm(Fu-Fus.full())))
print('norm(resP)={}'.format(ress['norm_res']))
print('norm(resP)={}'.format(np.linalg.norm((PBs-PGFAFG_fun_s(Fus)).full())))
print('norm(res)={}'.format(np.linalg.norm((Bs-GFAFG_fun_s(Fus, rank=None, tol=None)).full())))

print('memory efficiency = {0}/{1} = {2}'.format(Fus.size, Fu3.size, Fus.size/Fu3.size))

print('END')
