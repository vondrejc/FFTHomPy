import numpy as np
import scipy.sparse.linalg as sp
from ffthompy import Timer, Struct
import ffthompy.tensors.projection as proj
from ffthompy.general.solver import linear_solver, richardson, CG
from ffthompy.tensors import DFT, Operator, Tensor, grad_tensor, grad, div
from ffthompy.trigpol import mean_index, Grid
from ffthompy.sparse.solver import richardson as richardson_s
from ffthompy.sparse.projection import grad_tensor as sgrad_tensor
from ffthompy.sparse.canoTensor import CanoTensor

def homog_Ga_full(Aga, pars):
    Nbar = Aga.N
    N = np.array((np.array(Nbar)+1)/2, dtype=np.int)
    dim = Nbar.__len__()
    Y = np.ones(dim)

    _, Ghat, _ = proj.scalar(N, Y)
    Ghat2 = Ghat.enlarge(Nbar)

    F2 = DFT(name='FN', inverse=False, N=Nbar) # discrete Fourier transform (DFT)
    iF2 = DFT(name='FiN', inverse=True, N=Nbar) # inverse DFT

    G1N = Operator(name='G1', mat=[[iF2, Ghat2, F2]]) # projection in original space
    PAfun = Operator(name='FiGFA', mat=[[G1N, Aga]]) # lin. operator for a linear system
    E = np.zeros(dim); E[0] = 1 # macroscopic load
    EN = Tensor(name='EN', N=Nbar, shape=(dim,), Fourier=False) # constant trig. pol.
    EN.set_mean(E)

    x0 = Tensor(N=Nbar, shape=(dim,), Fourier=False) # initial approximation to solvers
    B = PAfun(-EN) # right-hand side of linear system
    tic=Timer(name='CG (gradient field)')
    X, info = linear_solver(solver='CG', Afun=PAfun, B=B,
                            x0=x0, par=pars.solver, callback=None)
    tic.measure()

    AH=Aga(X + EN)*(X + EN)
    return Struct(AH=AH, X=X)

def homog_Ga_full_potential(Aga, pars):
    Nbar = Aga.N # double grid number
    N = np.array((np.array(Nbar)+1)/2, dtype=np.int)
    dim = Nbar.__len__()
    Y = np.ones(dim) # cell size

    _, Ghat, _ = proj.scalar(N, Y)
    Ghat2 = Ghat.enlarge(Nbar)

    F2 = DFT(name='FN', inverse=False, N=Nbar) # discrete Fourier transform (DFT)
    iF2 = DFT(name='FiN', inverse=True, N=Nbar) # inverse DFT

    hGrad = grad_tensor(N, Y)
    k2=np.einsum('i...,i...', hGrad.val, np.conj(hGrad.val)).real
    k2[mean_index(N)]=1.
    P=Tensor(name='P', val=1./k2**0.5, order=0, Fourier=True, multype=00)
    iP=Tensor(name='P', val=k2**0.5, order=0, Fourier=True, multype=00)

    E=np.zeros(dim); E[0]=1 # macroscopic load
    EN=Tensor(name='EN', N=Nbar, shape=(dim,), Fourier=False) # constant trig. pol.
    EN.set_mean(E)

    def DFAFGfun(X):
        assert(X.Fourier)
        FAX=F2(Aga*iF2(grad(X).enlarge(Nbar)))
        FAX=FAX.decrease(N)
        return-div(FAX)

    B=-div(F2(Aga(-EN)).decrease(N))
    x0=Tensor(N=N, shape=(), Fourier=True) # initial approximation to solvers

    PDFAFGPfun=lambda Fx: P*DFAFGfun(P*Fx)
    PB=P*B
    tic=Timer(name='CG (potential)')
    iPU, info=linear_solver(solver='CG', Afun=PDFAFGPfun, B=PB,
                              x0=x0, par=pars.solver, callback=None)
    tic.measure()
    Fu=P*iPU

    X=iF2(grad(Fu).enlarge(Nbar))
    AH=Aga(X+EN)*(X+EN)
    iF=DFT(name='FiN', inverse=True, N=N)
    return Struct(AH=AH, e=X, Fu=Fu)

def homog_sparse(Agas, pars):
    Nbar = Agas.N
    N = np.array((np.array(Nbar)+1)/2, dtype=np.int)
    dim = Nbar.__len__()
    Y = np.ones(dim)
    hGrad_s = sgrad_tensor(N, Y)
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
    Es=CanoTensor(name='E', core=np.array([1.]), Fourier=False,
                  basis=[np.atleast_2d(np.ones(Nbar[ii])) for ii in range(dim)])

    Bs=hGrad_s[0]*((Agas*Es).fourier()).decrease(N) # minus from B and from div
    # print(np.linalg.norm(B.val-Bs.full()))

    # preconditioner
    hGrad=grad_tensor(N, Y)
    k2=np.einsum('i...,i...', hGrad.val, np.conj(hGrad.val)).real
    k2[mean_index(N)]=1.
    Prank=np.min([8, N[0]-1])
    u, s, vh=sp.svds(1./k2, k=Prank, which='LM')
    # U,S,Vh = svd(1./k2)
#     print('singular values of P={}'.format(s))
    Ps=CanoTensor(name='P', core=s[:Prank], basis=[u[:, :Prank].T, vh[:Prank]], Fourier=True)
    Ps.sort()
#     print('norm(P-Ps)={}'.format(np.linalg.norm(1./k2-Ps.full())))

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

    AH=0.
    for ii in range(dim):
        AH+=(Agas*FGX[ii]).scal(FGX[ii])
    return Struct(AH=AH, e=FGX, solver=ress, Fu=Fu)
