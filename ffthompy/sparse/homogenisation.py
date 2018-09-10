import numpy as np
import scipy.sparse.linalg as sp
from ffthompy import Timer, Struct
import ffthompy.tensors.projection as proj
from ffthompy.general.solver import linear_solver, richardson, CG
from ffthompy.tensors import DFT, Operator, Tensor, grad_tensor, grad, div
from ffthompy.trigpol import mean_index, Grid
from ffthompy.sparse.solver import richardson as richardson_s
# from ffthompy.sparse.solver import richardson_debug as richardson_s
from ffthompy.sparse.projection import grad_tensor as sgrad_tensor
from ffthompy.sparse.objects import SparseTensor

def homog_Ga_full(Aga, pars):
    Nbar=Aga.N
    N=np.array((np.array(Nbar)+1)/2, dtype=np.int)
    dim=Nbar.__len__()
    Y=np.ones(dim)
    _, Ghat, _=proj.scalar(N, Y)
    Ghat2=Ghat.enlarge(Nbar)
    F2=DFT(name='FN', inverse=False, N=Nbar) # discrete Fourier transform (DFT)
    iF2=DFT(name='FiN', inverse=True, N=Nbar) # inverse DFT

    G1N=Operator(name='G1', mat=[[iF2, Ghat2, F2]]) # projection in original space
    PAfun=Operator(name='FiGFA', mat=[[G1N, Aga]]) # lin. operator for a linear system
    E=np.zeros(dim); E[0]=1 # macroscopic load

    EN=Tensor(name='EN', N=Nbar, shape=(dim,), Fourier=False) # constant trig. pol.
    EN.set_mean(E)

    x0=Tensor(N=Nbar, shape=(dim,), Fourier=False) # initial approximation to solvers
    B=PAfun(-EN) # right-hand side of linear system
    tic=Timer(name='CG (gradient field)')
    X, info=linear_solver(solver='CG', Afun=PAfun, B=B,

                            x0=x0, par=pars.solver, callback=None)

    tic.measure()

    AH=Aga(X+EN)*(X+EN)
    return Struct(AH=AH, X=X)

def homog_Ga_full_potential(Aga, pars):

    Nbar=Aga.N # double grid number
    N=np.array((np.array(Nbar)+1)/2, dtype=np.int)

    dim=Nbar.__len__()
    Y=np.ones(dim) # cell size

    _, Ghat, _=proj.scalar(N, Y)
    Ghat2=Ghat.enlarge(Nbar)

    F2=DFT(name='FN', inverse=False, N=Nbar) # discrete Fourier transform (DFT)
    iF2=DFT(name='FiN', inverse=True, N=Nbar) # inverse DFT

    hGrad=grad_tensor(N, Y)

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
    print('iterations of CG={}'.format(info['kit']))
    print('norm of residuum={}'.format(info['norm_res']))

    Fu=P*iPU
    X=iF2(grad(Fu).enlarge(Nbar))

    AH=Aga(X+EN)*(X+EN)

    return Struct(AH=AH, e=X, Fu=Fu)

def homog_GaNi_full_potential(Agani, Aga, pars):

    N=Agani.N # double grid number
    dim=N.__len__()
    Y=np.ones(dim) # cell size
    _, Ghat, _=proj.scalar(N, Y)

    F=DFT(name='FN', inverse=False, N=N) # discrete Fourier transform (DFT)
    iF=DFT(name='FiN', inverse=True, N=N) # inverse DFT

    hGrad=grad_tensor(N, Y)
    k2=np.einsum('i...,i...', hGrad.val, np.conj(hGrad.val)).real
    k2[mean_index(N)]=1.

    P=Tensor(name='P', val=1./k2**0.5, order=0, Fourier=True, multype=00)
    iP=Tensor(name='P', val=k2**0.5, order=0, Fourier=True, multype=00)

    E=np.zeros(dim); E[0]=1 # macroscopic load
    EN=Tensor(name='EN', N=N, shape=(dim,), Fourier=False) # constant trig. pol.
    EN.set_mean(E)

    def DFAFGfun(X):
        assert(X.Fourier)
        FAX=F(Agani*iF(grad(X)))
        FAX=FAX
        return-div(FAX)

    B=-div(F(Agani(-EN)))
    x0=Tensor(N=N, shape=(), Fourier=True) # initial approximation to solvers

    PDFAFGPfun=lambda Fx: P*DFAFGfun(P*Fx)
    PB=P*B
    tic=Timer(name='CG (potential)')
    iPU, info=linear_solver(solver='CG', Afun=PDFAFGPfun, B=PB,
                            x0=x0, par=pars.solver, callback=None)
    tic.measure()
    print('iterations of CG={}'.format(info['kit']))
    print('norm of residuum={}'.format(info['norm_res']))

    Fu=P*iPU
    Nbar=2*np.array(N)-1

    iF2=DFT(name='FiN', inverse=True, N=Nbar) # inverse DFT
    XEN=iF2(grad(Fu).enlarge(Nbar))+EN.enlarge(Nbar)
    AH=Aga(XEN)*XEN
    return Struct(AH=AH, Fu=Fu)

def homog_Ga_sparse(Agas, pars):
    Nbar=Agas.N
    N=np.array((np.array(Nbar)+1)/2, dtype=np.int)
    dim=Nbar.__len__()
    hGrad_s=sgrad_tensor(N, pars.Y, kind=pars.kind)

    def DFAFGfun_s(X, rank=pars.rank, tol=pars.tol): # linear operator
        assert(X.Fourier)
        X=X.truncate(rank=rank, tol=tol)
        FGX=[((hGrad_s[ii]*X).enlarge(Nbar)).fourier() for ii in range(dim)]
        AFGFx=[Agas.multiply(FGX[ii], rank=rank, tol=tol) for ii in range(dim)]
        # or in following: Fourier, reduce, truncate
        FAFGFx=[AFGFx[ii].fourier() for ii in range(dim)]
        FAFGFx=[FAFGFx[ii].decrease(N) for ii in range(dim)]
        GFAFGFx=hGrad_s[0]*FAFGFx[0] # div
        for ii in range(1, dim):
            GFAFGFx+=hGrad_s[ii]*FAFGFx[ii]
        GFAFGFx=GFAFGFx.truncate(rank=rank, tol=tol)
        GFAFGFx.name='fun(x)'
        return-GFAFGFx

    # R.H.S.
    Es=SparseTensor(kind=pars.kind, val=np.ones(Nbar), rank=1)
    Bs=hGrad_s[0]*((Agas*Es).fourier()).decrease(N) # minus from B and from div

    # preconditioner
    N_ori=N
    reduce_factor=3.0 # this can be adjusted

    if np.prod(N_ori)>1e8: # this threshold can be adjusted
        N=np.ceil(N/reduce_factor).astype(int) #
        N[N % 2==0]+=1 # make them odd numbers
        need_project=True
    else:
        need_project=False

    hGrad=grad_tensor(N, pars.Y)
    k2=np.einsum('i...,i...', hGrad.val, np.conj(hGrad.val)).real
    k2[mean_index(N)]=1.
    Prank=np.min([8, N[0]-1])

    Ps=SparseTensor(kind=pars.kind, val=1./k2, rank=Prank, Fourier=True)

    if need_project:
        Ps=Ps.project(N_ori) # approximation
        N=N_ori

    def PDFAFGfun_s(Fx, rank=pars.rank, tol=pars.tol):
        R=DFAFGfun_s(Fx, rank=rank, tol=tol)
        R=Ps*R
        R=R.truncate(rank=rank, tol=tol)
        return R

    parP={'alpha': pars.alpha,
          'maxiter': pars.solver['maxiter'],
          'tol': pars.solver['tol']}

    tic=Timer(name='Richardson (sparse)')
    PBs=Ps*Bs
    Fu, ress=richardson_s(Afun=PDFAFGfun_s, B=PBs, par=parP,
                          rank=pars.rank, tol=pars.tol)
    tic.measure()
    print('iterations of solver={}'.format(ress['kit']))
    print('norm of residuum={}'.format(ress['norm_res'][-1]))
    Fu.name='Fu'
    print('norm(resP)={}'.format(np.linalg.norm((PBs-PDFAFGfun_s(Fu)).full())))
    print('norm(res)={}'.format(np.linalg.norm((Bs-DFAFGfun_s(Fu, rank=None, tol=None)).full())))

    FGX=[((hGrad_s[ii]*Fu).enlarge(Nbar)).fourier() for ii in range(dim)]
    FGX[0]+=Es # adding mean

#     AH=(Agas*FGX[0]).scal(Es) # homogenised coefficients A_11
    AH=0.
    for ii in range(dim):
        AH+=(Agas*FGX[ii]).scal(FGX[ii])
    return Struct(AH=AH, e=FGX, solver=ress, Fu=Fu)

def homog_GaNi_sparse(Aganis, Agas, pars):
    N=Aganis.N
    dim=N.__len__()
    hGrad_s=sgrad_tensor(N, pars.Y, kind=pars.kind)

    def DFAFGfun_s(X, rank=pars.rank, tol=pars.tol): # linear operator
        assert(X.Fourier)
        FGX=[(hGrad_s[ii]*X).fourier() for ii in range(dim)]
        AFGFx=[Aganis.multiply(FGX[ii], rank=rank, tol=tol) for ii in range(dim)]
        # or in following: Fourier, reduce, truncate
        FAFGFx=[AFGFx[ii].fourier() for ii in range(dim)]
        GFAFGFx=hGrad_s[0]*FAFGFx[0] # div
        for ii in range(1, dim):
            GFAFGFx+=hGrad_s[ii]*FAFGFx[ii]
        GFAFGFx=GFAFGFx.truncate(rank=rank, tol=tol)
        GFAFGFx.name='fun(x)'
        return -GFAFGFx

    # R.H.S.
    Es=SparseTensor(kind=pars.kind, val=np.ones(N), rank=1)
    Bs=hGrad_s[0]*(Aganis*Es).fourier() # minus from B and from div

    # preconditioner
    N_ori=N
    reduce_factor=3.0 # this can be adjusted

    if np.prod(N_ori)>1e8: # this threshold can be adjusted
        N=np.ceil(N/reduce_factor).astype(int) #
        N[N % 2==0]+=1 # make them odd numbers
        need_project=True
    else:
        need_project=False

    hGrad=grad_tensor(N, pars.Y)
    k2=np.einsum('i...,i...', hGrad.val, np.conj(hGrad.val)).real
    k2[mean_index(N)]=1.
    Prank=np.min([8, N[0]-1])

    Ps=SparseTensor(kind=pars.kind, val=1./k2, rank=Prank, Fourier=True)

    if need_project:
        Ps=Ps.project(N_ori) # approximation
        N=N_ori

    def PDFAFGfun_s(Fx, rank=pars.rank, tol=pars.tol):
        R=DFAFGfun_s(Fx, rank=rank, tol=tol)
        R=Ps*R
        R=R.truncate(rank=rank, tol=tol)
        return R

    parP={'alpha': pars.alpha,
          'maxiter': pars.solver['maxiter'],
          'tol': pars.solver['tol']}

    tic=Timer(name='Richardson (sparse)')
    PBs=Ps*Bs
    Fu, ress=richardson_s(Afun=PDFAFGfun_s, B=PBs, par=parP,
                          rank=pars.rank, tol=pars.tol)
    tic.measure()
    print('iterations of solver={}'.format(ress['kit']))
    print('norm of residuum={}'.format(ress['norm_res'][-1]))
    Fu.name='Fu'
    print('norm(resP)={}'.format(np.linalg.norm((PBs-PDFAFGfun_s(Fu)).full())))
    print('norm(res)={}'.format(np.linalg.norm((Bs-DFAFGfun_s(Fu, rank=None, tol=None)).full())))

    Nbar=2*np.array(N)-1

    FGX=[((hGrad_s[ii]*Fu).enlarge(Nbar)).fourier() for ii in range(dim)]
    Es=SparseTensor(kind=pars.kind, val=np.ones(Nbar), rank=1)
    FGX[0]+=Es # adding mean

    AH=0.
    for ii in range(dim):
        AH+=(Agas*FGX[ii]).scal(FGX[ii])
    return Struct(AH=AH, e=FGX, solver=ress, Fu=Fu)
