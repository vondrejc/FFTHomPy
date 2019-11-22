import numpy as np
from ffthompy import Timer, Struct
import ffthompy.tensors.projection as proj
from ffthompy.general.solver import linear_solver
from ffthompy.tensors import DFT, Operator, Tensor, grad_tensor, grad, div
from ffthompy.trigpol import mean_index
from ffthompy.sparse.solver import richardson as richardson_s
from ffthompy.sparse.solver import cheby2TERM as cheby2TERM_s
from ffthompy.sparse.solver import minimal_residual as minimal_residual_s
from ffthompy.sparse.solver import minimal_residual_debug
from ffthompy.sparse.projection import grad_tensor as sgrad_tensor
from ffthompy.sparse.objects import SparseTensor
import itertools


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
    return Struct(AH=AH, X=X, time=tic.vals[0][0], pars=pars)

def homog_Ga_full_potential(Aga, pars):

    Nbar=Aga.N # double grid number
    N=np.array((np.array(Nbar)+1)/2, dtype=np.int)

    dim=Nbar.__len__()
    Y=np.ones(dim) # cell size

    F2=DFT(name='FN', inverse=False, N=Nbar) # discrete Fourier transform (DFT)
    iF2=DFT(name='FiN', inverse=True, N=Nbar) # inverse DFT

    P = get_preconditioner(N, pars)

    E=np.zeros(dim); E[0]=1 # macroscopic load
    EN=Tensor(name='EN', N=Nbar, shape=(dim,), Fourier=False) # constant trig. pol.
    EN.set_mean(E)

    def DFAFGfun(X):
        assert(X.Fourier)
        FAX=F2(Aga*iF2(grad(X).enlarge(Nbar)))
        FAX=FAX.project(N)
        return -div(FAX)

    B=div(F2(Aga(EN)).decrease(N))
    x0=Tensor(N=N, shape=(), Fourier=True) # initial approximation to solvers

    PDFAFGPfun=lambda Fx: P*DFAFGfun(P*Fx)
    PB=P*B

    tic=Timer(name='CG (potential)')
    iPU, info=linear_solver(solver='CG', Afun=PDFAFGPfun, B=PB,
                            x0=x0, par=pars.solver, callback=None)
    tic.measure()

    print(('iterations of CG={}'.format(info['kit'])))
    print(('norm of residuum={}'.format(info['norm_res'])))
    R=PB-PDFAFGPfun(iPU)
    print(('norm of residuum={} (from definition)'.format(R.norm())))

    Fu=P*iPU
    X=iF2(grad(Fu).project(Nbar))

    AH=Aga(X+EN)*(X+EN)

    return Struct(AH=AH, e=X, Fu=Fu, info=info,  time=tic.vals[0][0])

def homog_GaNi_full_potential(Agani, Aga, pars):

    N=Agani.N # double grid number
    dim=N.__len__()
    Y=np.ones(dim) # cell size

    F=DFT(name='FN', inverse=False, N=N) # discrete Fourier transform (DFT)
    iF=DFT(name='FiN', inverse=True, N=N) # inverse DFT

    P = get_preconditioner(N, pars)

    E=np.zeros(dim); E[0]=1 # macroscopic load
    EN=Tensor(name='EN', N=N, shape=(dim,), Fourier=False) # constant trig. pol.
    EN.set_mean(E)

    def DFAFGfun(X):
        assert(X.Fourier)
        FAX=F(Agani*iF(grad(X)))
        return -div(FAX)

    B=div(F(Agani(EN)))
    x0=Tensor(N=N, shape=(), Fourier=True) # initial approximation to solvers

    PDFAFGPfun=lambda Fx: P*DFAFGfun(P*Fx)
    PB=P*B
    tic=Timer(name='CG (potential)')
    iPU, info=linear_solver(solver='CG', Afun=PDFAFGPfun, B=PB,
                            x0=x0, par=pars.solver, callback=None)
    tic.measure()
    print(('iterations of CG={}'.format(info['kit'])))
    print(('norm of residuum={}'.format(info['norm_res'])))

    Fu=P*iPU
    if Aga is None: # GaNi homogenised properties
        print('!!!!! homogenised properties are GaNi only !!!!!')
        XEN=iF(grad(Fu))+EN
        AH=Agani(XEN)*XEN
    else:
        Nbar=2*np.array(N)-1
        iF2=DFT(name='FiN', inverse=True, N=Nbar) # inverse DFT
        XEN=iF2(grad(Fu).project(Nbar))+EN.project(Nbar)
        AH=Aga(XEN)*XEN

    return Struct(AH=AH, Fu=Fu, info=info, time=tic.vals[0][0], pars=pars)

class Material_law():

    def __init__(self, Agas, Aniso, Es):
        self.Agas=Agas
        self.Aniso=Aniso
        self.dim=Aniso.shape[0]
        dim=self.dim

        if np.linalg.norm(Aniso) < 1e-12:
            self._call=self.material_isotropic
        else:
            self._call=self.material_anisotropic
            self.Aniso_fun=np.empty((dim, dim)).tolist()
            for i,j in itertools.product(range(dim), repeat=2):
                if i==j:
                    self.Aniso_fun[i][j]=Agas+Es*Aniso[i,j]
                else:
                    self.Aniso_fun[i][j]=Es*Aniso[i,j]

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def material_isotropic(self, X, rank=None, tol=None):
        return [self.Agas.multiply(X[i], rank=rank, tol=tol) for i in range(self.dim)]

    def material_anisotropic(self, X, rank=None, tol=None):
        AFGFx=self.dim*[None]
        for i in range(self.dim):
            for j in range(self.dim):
                AFGFx[i]+=self.Aniso_fun[i][j]*X[j]
            AFGFx[i]=AFGFx[i].truncate(rank=rank, tol=tol)
        return AFGFx

def homog_Ga_sparse(Agas, pars):
    Nbar=Agas.N
    N=np.array((np.array(Nbar)+1)/2, dtype=np.int)
    dim=Nbar.__len__()
    hGrad_s=sgrad_tensor(N, pars.Y, kind=pars.kind)
    if 'Aniso' in pars.keys():
        Aniso=pars.Aniso
    else:
        Aniso=np.zeros([dim,dim])

    # creating constant field in sparse tensor
    Es=SparseTensor(name='E', kind=pars.kind, val=np.ones(dim*(3,)), rank=1)
    Es=Es.fourier().enlarge(Nbar).fourier()

    material_law=Material_law(Agas, Aniso, Es)

    def DFAFGfun_s(X, rank=pars.rank, tol=pars.tol): # linear operator
        assert(X.Fourier)
        FGX=[((hGrad_s[ii]*X).enlarge(Nbar)).fourier() for ii in range(dim)]
        AFGFx=material_law(FGX, rank=pars.rank, tol=pars.tol)
        # or in following: Fourier, reduce, truncate
        FAFGFx=[AFGFx[ii].fourier() for ii in range(dim)]
        FAFGFx=[FAFGFx[ii].decrease(N) for ii in range(dim)]
        GFAFGFx=hGrad_s[0]*FAFGFx[0] # div
        for ii in range(1, dim):
            GFAFGFx+=hGrad_s[ii]*FAFGFx[ii]
        GFAFGFx=GFAFGFx.truncate(rank=rank, tol=tol, fast= pars.fast)
        GFAFGFx.name='fun(x)'
        return -GFAFGFx

    # R.H.S.
    Bs=hGrad_s[0]*((Agas*Es).fourier()).decrease(N) # minus from B and from div

    Ps=get_preconditioner_sparse(N, pars)

    def PDFAFGfun_s(Fx, rank=pars.rank, tol=pars.tol):
        R=DFAFGfun_s(Fx, rank=rank, tol=tol)
        R=Ps*R
        R=R.truncate(rank=rank, tol=tol, fast= pars.fast)
        return R

    parP=pars.solver

    tic=Timer(name=pars.solver['method'])

    PBs=Ps*Bs
    PBs2=PBs.truncate(tol=pars.rhs_tol)
#    print(('norm of r.h.s.= {}'.format(np.linalg.norm(PBs.full().val))))
#    print(('error in r.h.s. = {}'.format(np.linalg.norm(PBs.full().val-PBs2.full().val))))
    PBs=PBs2
    if pars.solver['method'] in ['Richardson','richardson','r','R']:
        Fu, ress=richardson_s(Afun=PDFAFGfun_s, B=PBs, par=parP,
                              rank=pars.rank, tol=pars.tol)
    elif pars.solver['method'] in ['Chebyshev','chebyshev','c','C']:
        Fu, ress=cheby2TERM_s(Afun=PDFAFGfun_s, B=PBs, par=parP,
                              rank=pars.rank, tol=pars.tol)
    elif pars.solver['method'] in ['minimal_residual','mr','m','M']:
        Fu, ress=minimal_residual_s(Afun=PDFAFGfun_s, B=PBs, par=parP,
                                    rank=pars.rank, tol=pars.tol, fast= pars.fast)
    elif pars.solver['method'] in ['minimal_residual_debug','mrd']:
        Fu, ress=minimal_residual_debug(Afun=PDFAFGfun_s, B=PBs, par=parP,
                                        rank=pars.rank, tol=pars.tol)
    tic.measure()

    print(('iterations of solver={}'.format(ress['kit'])))
    print(('norm of residuum={}'.format(ress['norm_res'][-1])))
    Fu.name='Fu'
    print(('norm(resP)={}'.format(np.linalg.norm((PBs-PDFAFGfun_s(Fu)).full()))))

    FGX=[((hGrad_s[ii]*Fu).enlarge(Nbar)).fourier() for ii in range(dim)]
    FGX[0]+=Es # adding mean

    AH = calculate_AH_sparse(Agas, Aniso, FGX, method='full')
    return Struct(AH=AH, e=FGX, solver=ress, Fu=Fu, time=tic.vals[0][0])

def homog_GaNi_sparse(Aganis, Agas, pars):

    N=Aganis.N
    dim=N.__len__()
    hGrad_s=sgrad_tensor(N, pars.Y, kind=pars.kind)

    if 'Aniso' in pars.keys():
        Aniso=pars.Aniso
    else:
        Aniso=np.zeros([dim,dim])

    # creating constant field in sparse tensor
    Es=SparseTensor(name='E', kind=pars.kind, val=np.ones(dim*(3,)), rank=1)
    Es=Es.fourier().enlarge(N).fourier()

    material_law=Material_law(Aganis, Aniso, Es)

    def DFAFGfun_s(X, rank=pars.rank, tol=pars.tol): # linear operator
        assert(X.Fourier)
        FGX=[(hGrad_s[ii]*X).fourier() for ii in range(dim)]
        AFGFx=material_law(FGX, rank=pars.rank, tol=pars.tol)
        # or in following: Fourier, reduce, truncate
        FAFGFx=[AFGFx[ii].fourier() for ii in range(dim)]
        GFAFGFx=hGrad_s[0]*FAFGFx[0] # div
        for ii in range(1, dim):
            GFAFGFx+=hGrad_s[ii]*FAFGFx[ii]
        GFAFGFx=GFAFGFx.truncate(rank=rank, tol=tol, fast= pars.fast)
        GFAFGFx.name='fun(x)'
        return -GFAFGFx

    # R.H.S.
    Bs=hGrad_s[0]*(Aganis*Es).fourier() # minus from B and from div

    Ps=get_preconditioner_sparse(N, pars)

    def PDFAFGfun_s(Fx, rank=pars.rank, tol=pars.tol):
        R=DFAFGfun_s(Fx, rank=rank, tol=tol)
        R=Ps*R
        R=R.truncate(rank=rank, tol=tol, fast= pars.fast)
        return R

    parP=pars.solver

    tic=Timer(name=pars.solver['method'])

    PBs=Ps*Bs
    #PBs2=PBs.truncate(tol=pars.rhs_tol, fast=True)
    PBs2=PBs.truncate(rank=pars.rank, fast=True)
#        print(('norm of r.h.s.= {}'.format(np.linalg.norm(PBs.full().val))))
#        print(('error in r.h.s. = {}'.format(np.linalg.norm(PBs.full().val-PBs2.full().val))))
    PBs=PBs2

    if pars.solver['method'] in ['Richardson','richardson','r','R']:
        Fu, ress=richardson_s(Afun=PDFAFGfun_s, B=PBs, par=parP,
                              rank=pars.rank, tol=pars.tol)
    elif pars.solver['method'] in ['Chebyshev','chebyshev','c','C']:
        Fu, ress=cheby2TERM_s(Afun=PDFAFGfun_s, B=PBs, par=parP,
                              rank=pars.rank, tol=pars.tol)
    elif pars.solver['method'] in ['minimal_residual','mr','m','M']:
        Fu, ress=minimal_residual_s(Afun=PDFAFGfun_s, B=PBs, par=parP,
                                    rank=pars.rank, tol=pars.tol, fast=pars.fast)
    elif pars.solver['method'] in ['minimal_residual_debug','mrd']:
        Fu, ress=minimal_residual_debug(Afun=PDFAFGfun_s, B=PBs, par=parP,
                                        rank=pars.rank, tol=pars.tol)
    tic.measure()
    print(('iterations of solver={}'.format(ress['kit'])))
    print(('norm of residuum={}'.format(ress['norm_res'][-1])))
    Fu.name='Fu'
    print(('norm(resP)={}'.format(np.linalg.norm((PBs-PDFAFGfun_s(Fu)).full()))))

    if Agas is None: # GaNi homogenised properties
        print('!!!!! homogenised properties are GaNi only !!!!!')
        FGX=[(hGrad_s[ii]*Fu).fourier() for ii in range(dim)]
        FGX[0]+=Es # adding mean
        AH = calculate_AH_sparse(Aganis, Aniso, FGX, method='full')
    else:
        Nbar=2*np.array(N)-1
        FGX=[((hGrad_s[ii]*Fu).enlarge(Nbar)).fourier() for ii in range(dim)]
        Es=SparseTensor(kind=pars.kind, val=np.ones(Nbar), rank=1)
        FGX[0]+=Es # adding mean
        AH = calculate_AH_sparse(Agas, Aniso, FGX, method='full')

    return Struct(AH=AH, e=FGX, solver=ress, Fu=Fu,  time=tic.vals[0][0] )

def get_preconditioner(N, pars):
    hGrad=grad_tensor(N, pars.Y)
    k2=np.einsum('i...,i...', hGrad.val, np.conj(hGrad.val)).real
    k2[mean_index(N)]=1.
    return Tensor(name='P', val=1./k2**0.5, order=0, N=N, Fourier=True, multype=00)

def get_preconditioner_sparse(N, pars):
    hGrad=grad_tensor(N, pars.Y, fft_form='c')
    k2=np.einsum('i...,i...', hGrad.val, np.conj(hGrad.val)).real
    k2[mean_index(N, fft_form='c')]=1.
    Prank=np.min([10, N[0]-1])
    val=1./k2
    Ps=SparseTensor(name='Ps', kind=pars.kind, val=val, rank=Prank, Fourier=True, fft_form='c')
    Ps.set_fft_form()
    return Ps

def calculate_AH_sparse(Agas, Aniso, FGX, method='full', rank=None, tol=None):
    tic=Timer(name='AH')
    AH=0.
    if method in ['full']:
        Aga=Agas.full(multype=00)
        FGXf=[T.full() for T in FGX]
        for i in range(FGX.__len__()):
            AH+=(Aga*FGXf[i])*FGXf[i]
            for j in range(FGX.__len__()):
                AH+=(Aniso[i,j]*FGXf[i])*FGXf[j]
    elif method in ['sparse']:
        assert(np.linalg.norm(Aniso)<1e-12)
        for ii in range(FGX.__len__()):
            AH+=(Agas*FGX[ii]).truncate(rank=rank, tol=tol).scal(FGX[ii])
    else:
        raise NotImplementedError()

    tic.measure()
    return AH
