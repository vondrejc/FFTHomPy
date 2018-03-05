from __future__ import division
import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as sp
from scipy.linalg import svd
import itertools
import sys
from ffthompy import Timer
from ffthompy.materials import Material
from ffthompy.general.solver import linear_solver, richardson, CG
from ffthompy.trigpol import mean_index, Grid
from ffthompy.tensors import DFT, Operator, Tensor, grad_tensor, matrix2tensor
import ffthompy.tensors.projection as proj
from ffthompy.sparse import decompositions
from ffthompy.sparse.canoTensor import CanoTensor
from ffthompy.sparse.materials import SparseMaterial
from ffthompy.sparse.projection import grad_tensor as sgrad_tensor
from ffthompy.sparse.solver import richardson as richardson_s


# PARAMETERS ##############################################################
dim  = 2            # number of dimensions (works for 2D and 3D)
N    = dim*(55,)   # number of voxels (assumed equal for all directions)
Nbar = 2*np.array(N)-1
Y    = np.ones(dim)
Amax = 10.          # material contrast

maxiter=1e2

tol=None
rank=15

calc_eigs = 1
debug=False

# auxiliary values
prodN = np.prod(np.array(N)) # number of grid points
ndof  = dim*prodN # number of degrees-of-freedom
vec_shape=(dim,)+N # shape of the vector for storing DOFs

# PROBLEM DEFINITION ######################################################
E = np.zeros(vec_shape); E[0] = 1. # set macroscopic loading

# sparse A
mat_conf={'inclusions': ['square', 'otherwise'],
              'positions': [-0.2*np.ones(dim), ''],
              'params': [0.6*np.ones(dim), ''], # size of sides
              'vals': [10*np.eye(dim), 1.*np.eye(dim)],
              'Y': np.ones(dim),
              'P': N,
              'order': 0,
              }
mat=Material(mat_conf)
mats = SparseMaterial(mat_conf)

Agani = matrix2tensor(mat.get_A_GaNi(N, primaldual='primal'))
Aganis = mats.get_A_GaNi(N, primaldual='primal', k=2)
Aga = matrix2tensor(mat.get_A_Ga(Nbar, primaldual='primal'))
Agas = mats.get_A_Ga(Nbar, primaldual='primal', order=0, k=2)
print(np.linalg.norm(Agani.val[0,0]-Aganis.full()))
print(np.linalg.norm(Aga.val[0,0]-Agas.full()))

# OPERATORS ###############################################################
_, Ghat, _ = proj.scalar(N, Y)
# dot21  = lambda A,v: np.einsum('ij...,j...  ->i...',A,v)
# fft    = lambda V: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(V),N))/np.prod(N)
# ifft   = lambda V: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(V),N))*np.prod(N)
# G_fun  = lambda V: np.real(ifft(dot21(Ghat.val,fft(V)))).reshape(-1)
# A_fun  = lambda v: dot21(A.val,v.reshape(vec_shape))
# GA_fun = lambda v: G_fun(A_fun(v))

Ghat2 = Ghat.enlarge(Nbar)

F = DFT(name='FN', inverse=False, N=N) # discrete Fourier transform (DFT)
iF = DFT(name='FiN', inverse=True, N=N) # inverse DFT
F2 = DFT(name='FN', inverse=False, N=Nbar) # discrete Fourier transform (DFT)
iF2 = DFT(name='FiN', inverse=True, N=Nbar) # inverse DFT

print('\n== CG solver for gradient field ==============')
G1N = Operator(name='G1', mat=[[iF2, Ghat2, F2]]) # projection in original space
PAfun = Operator(name='FiGFA', mat=[[G1N, Aga]]) # lin. operator for a linear system
E = np.zeros(dim); E[0] = 1 # macroscopic load
EN = Tensor(name='EN', N=Nbar, shape=(dim,), Fourier=False) # constant trig. pol.
EN.set_mean(E)

x0 = Tensor(N=Nbar, shape=(dim,), Fourier=False) # initial approximation to solvers
B = PAfun(-EN) # right-hand side of linear system
X, info = linear_solver(solver='CG', Afun=PAfun, B=B,
                        x0=x0, par={}, callback=None)

print('homogenised properties (component 11) =', Aga(X + EN)*(X + EN))

print('\n== Generating operators for formulation with potential ===========')
hGrad = grad_tensor(N, Y)
k2 = np.einsum('i...,i...', hGrad.val, np.conj(hGrad.val)).real
k2[mean_index(N)]=1.
P = Tensor(name='P', val=1./k2**0.5, order=0, Fourier=True, multype=00)
iP= Tensor(name='P', val=k2**0.5, order=0, Fourier=True, multype=00)

print('\n== CG solver for potential ==============')
from ffthompy.tensors import grad, div
def DFAFGfun(X):
    assert(X.Fourier)
    FAX= F2(Aga*iF2(grad(X).enlarge(Nbar)))
    FAX=FAX.decrease(N)
    return -div(FAX)

B = -div(F2(Aga(-EN)).decrease(N))
x0 = Tensor(N=N, shape=(), Fourier=True) # initial approximation to solvers
# U, info = linear_solver(solver='CG', Afun=DFAFGfun, B=B,
#                         x0=x0, par={}, callback=None)

PDFAFGPfun = lambda Fx: P*DFAFGfun(P*Fx)
PB = P*B
iPU, info = linear_solver(solver='CG', Afun=PDFAFGPfun, B=PB,
                          x0=x0, par={tol:1e-10}, callback=None)
U = P*iPU

X2 = iF2(grad(U).enlarge(Nbar))
print('homogenised properties (component 11) =', Aga(X2 + EN)*(X2 + EN))

print('\n== Generating operators for SPARSE solver...')
hGrad_s = sgrad_tensor(N, Y)
print(np.linalg.norm(hGrad_s[0].full()-hGrad.val[0]))
# linear operator
def DFAFGfun_s(X, rank=rank, tol=tol):
    assert(X.Fourier)
    FGX=[((hGrad_s[ii]*X).enlarge(Nbar)).fourier() for ii in range(dim)]
    AFGFx=[Agas.multiply(FGX[ii], rank=None, tol=None) for ii in range(dim)]
    # or in following: Fourier, reduce, truncate
    AFGFx=[AFGFx[ii].truncate(rank=rank, tol=tol) for ii in range(dim)]
    FAFGFx=[AFGFx[ii].fourier() for ii in range(dim)]
    FAFGFx=[FAFGFx[ii].decrease(N) for ii in range(dim)]
    GFAFGFx=hGrad_s[0]*FAFGFx[0] # div
    for ii in range(1, dim):
        GFAFGFx+=hGrad_s[ii]*FAFGFx[ii]
    GFAFGFx=GFAFGFx.truncate(rank=rank, tol=tol)
    GFAFGFx.name='fun(x)'
    return -GFAFGFx

# testing
print('testing linear operator of system...')
x=CanoTensor(name='r', r=4, N=N, randomise=True)
Fx=x.fourier()
FX = Tensor(name='r', val=Fx.full(), order=0, Fourier=True)
y=CanoTensor(name='r', r=3, N=N, randomise=True)
Fy=y.fourier()
FY = Tensor(name='r', val=Fx.full(), order=0, Fourier=True)

res1=DFAFGfun_s(Fx, rank=None, tol=None)
res2=DFAFGfun(FX)
print(np.linalg.norm(res1.full()-res2.val))

# R.H.S.
Es=CanoTensor(name='E', core=np.array([1.]), Fourier=False,
              basis=[np.atleast_2d(np.ones(Nbar[ii])) for ii in range(dim)])

Bs=hGrad_s[0]*((Agas*Es).fourier()).decrease(N) # minus from B and from div
print(np.linalg.norm(B.val-Bs.full()))

# solver
print('\n== SPARSE Richardson solver with preconditioner =======================')
# preconditioner
Prank = np.min([8, N[0]-1])
u, s, vh=sp.svds(1./k2, k=Prank, which='LM')
# U,S,Vh = svd(1./k2)
print('singular values of P={}'.format(s))
Ps=CanoTensor(name='P', core=s[:Prank], basis=[u[:, :Prank].T, vh[:Prank]], Fourier=True)
Ps.sort()
print('norm(P-Ps)={}'.format(np.linalg.norm(1./k2-Ps.full())))

def PDFAFGfun_s(Fx, rank=rank, tol=tol):
    R=DFAFGfun_s(Fx, rank=rank, tol=tol)
    R=Ps*R
    R=R.truncate(rank=rank, tol=tol)
    return R

# testing
print('testing precond. linear system...')
res1=PDFAFGfun_s(Fx, rank=None, tol=None)
res2=P*(P*DFAFGfun(FX))
print(norm(res1.full()-res2.val))

print('\nsolver results...')
normfun=lambda X: X.norm()
normfun=lambda X: np.linalg.norm(X.full())

parP={'alpha': (1.+Amax)/2.,
      'maxiter': maxiter,
      'tol': 1e-10,
      'norm': normfun}

tic=Timer(name='Richardson (sparse)')
PBs=Ps*Bs
Fus, ress=richardson_s(Afun=PDFAFGfun_s, B=PBs, par=parP, norm=normfun, rank=rank, tol=tol)
tic.measure()
Fus.name='Fus'
Fus = Fus

print Fus
print('iterations={}'.format(ress['kit']))
print('norm(dif)={}'.format(np.linalg.norm(U.val-Fus.full())))
print('norm(resP)={}'.format(ress['norm_res']))
print('norm(resP)={}'.format(np.linalg.norm((PBs-PDFAFGfun_s(Fus)).full())))
print('norm(res)={}'.format(np.linalg.norm((Bs-DFAFGfun_s(Fus, rank=None, tol=None)).full())))
print('memory efficiency = {0}/{1} = {2}'.format(Fus.size, U.val.size, Fus.size/U.val.size))

Xs = iF2(grad(U).enlarge(Nbar))
FGX=[((hGrad_s[ii]*Fus).enlarge(Nbar)).fourier() for ii in range(dim)]
FGX[0] += Es # adding mean

# control
for ii in range(dim):
    print(norm(FGX[ii].full()-(X2+EN).val[ii])/np.prod(Nbar))

print(np.mean(FGX[0].full()))

AH11=0.
for ii in range(dim):
    AH11+=(Agas*FGX[ii]).scal(FGX[ii])

print('homogenised properties (component 11) = {}'.format(AH11))

AH11 = 0.
for ii, jj in itertools.product(range(dim), repeat=2):
    AH11+=np.sum(Aga.val[ii,jj]*(X.val[ii] + EN.val[ii])*(X.val[jj] + EN.val[jj]))/np.prod(Nbar)
print(AH11)

print('END')

