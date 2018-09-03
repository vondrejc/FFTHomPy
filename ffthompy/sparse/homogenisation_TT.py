import numpy as np
import scipy.sparse.linalg as sp
from ffthompy import Timer, Struct
#import ffthompy.tensors.projection as proj
from ffthompy.general.solver import linear_solver, richardson, CG
from ffthompy.tensors import DFT, Operator, Tensor, grad_tensor, grad, div
from ffthompy.trigpol import mean_index, Grid
from ffthompy.sparse.solver import richardson as richardson_s
from ffthompy.sparse.projection import grad_tensor as sgrad_tensor
from ffthompy.sparse.tensorTrain import TensorTrain



def homog_sparse(Agas, pars):
    Nbar = Agas.N
    N = np.array((np.array(Nbar)+1)/2, dtype=np.int)
    dim = Nbar.__len__()
    hGrad_s = sgrad_tensor(N, pars.Y, tensor=TensorTrain)
    # linear operator
    def DFAFGfun_s(X, rank=pars.rank, tol=pars.tol):
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
        return-GFAFGFx

    # R.H.S.
#    Es=CanoTensor(name='E', core=np.array([1.]), Fourier=False,
#                  basis=[np.atleast_2d(np.ones(Nbar[ii])) for ii in range(dim)])
#    Es=Tucker(name='E', core=np.array([1.]), Fourier=False,
#              basis=[np.atleast_2d(np.ones(Nbar[ii])) for ii in range(dim)])
    
    Es = TensorTrain(np.ones(Nbar), rmax=1)
    Bs=hGrad_s[0]*((Agas*Es).fourier()).decrease(N) # minus from B and from div
 
    # preconditioner

    N_ori=N
    reduce_factor=3.0 # this can be adjusted

    if np.prod(N_ori)>1e8: # this threshold can be adjusted
        N=np.ceil(N/reduce_factor).astype(int) #
        N[N%2==0]+=1 # make them odd numbers
        need_project=True
    else:
        need_project=False

    hGrad=grad_tensor(N, pars.Y)
    k2=np.einsum('i...,i...', hGrad.val, np.conj(hGrad.val)).real
    k2[mean_index(N)]=1.
    Prank=np.min([8, N[0]-1])
#    S, U=HOSVD (1./k2, k=Prank)
#    for i in range(len(U)):
#        U[i]=U[i].T
#
#    Ps=Tucker(name='P', core=S, basis=U, Fourier=True)
    
    Ps=TensorTrain(1./k2, rmax=Prank, Fourier=True)
    
    if need_project:
        Ps=Ps.project(N_ori) # approximation
        N=N_ori

    def PDFAFGfun_s(Fx, rank=pars.rank, tol=pars.tol):
        R=DFAFGfun_s(Fx, rank=rank, tol=tol)
        R=Ps*R
        R=R.truncate(rank=rank, tol=tol)
        return R

#     print('\nsolver results...')
    normfun=lambda X: X.norm()

    parP={'alpha': (1.+pars.Amax)/2.,
          'maxiter': pars.maxiter,
          'tol': 1e-5,
          'norm': normfun}

    tic=Timer(name='Richardson (sparse)')
    PBs=Ps*Bs
    Fu, ress=richardson_s(Afun=PDFAFGfun_s, B=PBs, par=parP,
                           norm=normfun, rank=pars.rank, tol=pars.tol)
    tic.measure()
    Fu.name='Fu'
    print('norm(resP)={}'.format(np.linalg.norm((PBs-PDFAFGfun_s(Fu)).full())))
    print('norm(res)={}'.format(np.linalg.norm((Bs-DFAFGfun_s(Fu, rank=None, tol=None)).full())))

    FGX=[((hGrad_s[ii]*Fu).enlarge(Nbar)).fourier() for ii in range(dim)]
    FGX[0] += Es # adding mean

#     AH=(Agas*FGX[0]).scal(Es) # homogenised coefficients A_11
    AH=0.
    for ii in range(dim):
        AH+=(Agas*FGX[ii]).scal(FGX[ii])
    return Struct(AH=AH, e=FGX, solver=ress, Fu=Fu)
