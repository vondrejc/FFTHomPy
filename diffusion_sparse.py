import numpy as np
import scipy.sparse.linalg as sp
import itertools
import sys
# from ffthompy.general.solver import richardson

# PARAMETERS ##############################################################
dim  = 2            # number of dimensions (works for 2D and 3D)
N    = dim*(15,)    # number of voxels (assumed equal for all directions)
Amax = 10.          # material contrast

maxiter=10

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
Ghat = np.zeros((dim,dim)+ N) # zero initialize
freq = [np.arange(-(N[ii]-1)/2.,+(N[ii]+1)/2.) for ii in range(dim)]
for i,j in itertools.product(range(dim),repeat=2):
    for ind in itertools.product(*[range(n) for n in N]):
        q = np.empty(dim)
        for ii in range(dim):
            q[ii] = freq[ii][ind[ii]]  # frequency vector
        if not q.dot(q) == 0:          # zero freq. -> mean
            Ghat[i,j][ind] = -(q[i]*q[j])/(q.dot(q))

# OPERATORS ###############################################################
dot21  = lambda A,v: np.einsum('ij...,j...  ->i...',A,v)
fft    = lambda V: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(V),N))/np.prod(N)
ifft   = lambda V: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(V),N))*np.prod(N)
G_fun  = lambda V: np.real(ifft(dot21(Ghat,fft(V)))).reshape(-1)
A_fun  = lambda v: dot21(A,v.reshape(vec_shape))
GA_fun = lambda v: G_fun(A_fun(v))

# CONJUGATE GRADIENT SOLVER ###############################################
print('\n== CG solver for gradient field in real space =======================')
b = -GA_fun(E) # right-hand side
e, _=sp.cg(A=sp.LinearOperator(shape=(ndof, ndof), matvec=GA_fun, dtype='float'), b=b,
           tol=1e-10, maxiter=maxiter)

aux = e+E.reshape(-1)
# print('auxiliary field for macroscopic load E = {1}:\n{0}'.format(aux.reshape(vec_shape),
#                                                                   format((1,)+(dim-1)*(0,))))
print('norm(residuum(e))={}'.format(np.linalg.norm(b-GA_fun(e))))
print('homogenised properties A11 = {}'.format(np.inner(A_fun(aux).reshape(-1), aux)/prodN))

### POTENTIAL SOLVER in real space #####################################################
print('\nGenerating operators for formulation with potential...')
# GRADIENT IN FOURIER SPACE #############################################
hGrad = np.zeros((dim,)+ N) # zero initialize
freq = [np.arange(-(N[ii]-1)/2.,+(N[ii]+1)/2.) for ii in range(dim)]
for ind in itertools.product(*[range(n) for n in N]):
    for i in range(dim):
        hGrad[i][ind] = freq[i][ind[i]]
hGrad = -hGrad*2*np.pi*1j

k2 = np.einsum('i...,i...', hGrad, np.conj(hGrad)).real
from ffthompy.matvec import mean_index
k2[mean_index(N)]=1.

# OPERATORS ###############################################################
FGrad = lambda Fu: np.einsum('i...,...->i...', hGrad, Fu)
FDiv = lambda Fe: np.einsum('i...,i...->...', hGrad, Fe)
Grad = lambda u: ifft(FGrad(fft(u))).real
Div = lambda e: ifft(FDiv(fft(e))).real
DivAGrad_fun = lambda u: Div(dot21(A, Grad(u.reshape(N)))).ravel()
GFAFG_fun=lambda Fu:-FDiv(fft(dot21(A, ifft(FGrad(Fu)))))
GFAFG_funvec=lambda Fu: GFAFG_fun(Fu.reshape(N)).ravel()

# CONJUGATE GRADIENT SOLVER ###############################################
print('\n== CG solver for potential field in Fourier space =======================')
B=FDiv(fft(dot21(A, E))) # right-hand side

linoper=sp.LinearOperator(shape=(prodN, prodN), matvec=GFAFG_funvec, dtype='complex')
Fu, _=sp.cg(A=linoper, b=B.ravel(), maxiter=1e2)

e2=ifft(FGrad(Fu.reshape(N))).real
aux=e2+E
print('norm(residuum)={}'.format(np.linalg.norm(B-GFAFG_fun(Fu.reshape(N)))))
print('norm(residuum(e))={}'.format(np.linalg.norm(b-GA_fun(e2))))
print('homogenised properties A11 = {}'.format(np.sum(dot21(A, aux)*aux)/prodN))

print('\n== CG solver with preconditioner for potential field in Fourier space =======================')
k2r=k2**0.5
Nfun = lambda X: X/k2r
GnFAFGn_fun=lambda Fu:-Nfun(FDiv(fft(dot21(A, ifft(FGrad(Nfun(Fu)))))))
GnFAFGn_funvec=lambda Fu: GnFAFGn_fun(Fu.reshape(N)).ravel()
B=Nfun(FDiv(fft(dot21(A, E))))

linoper=sp.LinearOperator(shape=(prodN, prodN), matvec=GnFAFGn_funvec, dtype='complex')
Fu, _=sp.cg(A=linoper, b=B.ravel(), maxiter=maxiter)
Fu = Fu.reshape(N)/k2r

e3=ifft(FGrad(Fu)).real
aux=e3+E
print('norm(residuum)={}'.format(np.linalg.norm(B-GFAFG_fun(Fu))))
print('norm(residuum(e))={}'.format(np.linalg.norm(b-GA_fun(e3))))
print('homogenised properties A11 = {}'.format(np.sum(dot21(A, aux)*aux)/prodN))

print('norm(e-e3)={}'.format(np.linalg.norm(e-e3.ravel())))
Fu = Fu.reshape(N)
print('mean(Fu)={}'.format(Fu[mean_index(N)]))
print('mean(u)={}'.format(np.mean(ifft(FGrad(Fu.reshape(N))).real)))

print('\n== CG - own solver =====')
from ffthompy.general.solver import CG
# maxiter=1e1
scal = lambda X,Y: np.sum(X*Y.conj()).real

Fu4, res4=CG(Afun=GnFAFGn_fun, B=B, x0=np.zeros_like(B), scal=scal, par={'maxiter': maxiter})
Fu4=Fu4/k2r
e4=ifft(FGrad(Fu4)).real
aux=e4+E
print('norm(residuum)={}'.format(np.linalg.norm(B-GFAFG_fun(Fu4.reshape(N)))))
print('norm(residuum(e))={}'.format(np.linalg.norm(b-GA_fun(e4))))
print('homogenised properties A11 = {}'.format(np.sum(dot21(A, aux)*aux)/prodN))
print(res4)
print(np.linalg.norm(Fu-Fu4))

u=np.random.random(N)
v=np.random.random(N)

print(np.sum(GnFAFGn_fun(u)*v.conj()))
print(np.sum(GnFAFGn_fun(v)*u.conj()))

## RICHARDSON iteration #######################################################
def richardson(Afun, B, x0=None, par=None, norm=None):
    if isinstance(par['alpha'], float):
        omega=1./par['alpha']
    else:
        raise NotImplementedError()
    res={'norm_res': [],
           'kit': 0}
    if x0 is None:
        x=B*omega
    else:
        x=x0

    if norm is None:
        norm=lambda X: float(X.T*X)**0.5
    norm_res=1e15
    while (norm_res>par['tol'] and res['kit']<par['maxiter']):
        res['kit']+=1
        residuum=B-Afun(x)
        x=x+residuum*omega
        norm_res=norm(residuum)
        res['norm_res'].append(norm_res)
    res['norm_res'].append(norm(B-Afun(x)))
    return x, res

### PRECONDITIONING in Fourier space #####################################################
print('\n== Richardson with preconditioning for potential field in Fourier space=====')

parP={'alpha': (1.+Amax)/2.,
      'maxiter': maxiter,
      'tol': 1e-6}
B=FDiv(fft(dot21(A, E))) # right-hand side
Fu3, res3=richardson(Afun=PGFAFG_fun, B=Pfun(B), x0=None, par=parP, norm=np.linalg.norm)

print('iterations={}'.format(res3['kit']))
print('norm(dif)={}'.format(np.linalg.norm(Fu-Fu3.ravel())))
print('norm(resP)={}'.format(res3['norm_res']))
print('norm(resP)={}'.format(np.linalg.norm(Pfun(B)-PGFAFG_fun(Fu3))))
print('norm(res)={}'.format(np.linalg.norm(B-GFAFG_fun(Fu3))))

print('\n== Generating operators for SPARSE solver...')
from ffthompy.sparse import decompositions
from ffthompy.sparse.canoTensor import CanoTensor

# matrix A
Abas, k, max_err=decompositions.dCA_matrix_input(A[0, 0], k=3, tol=1e-14)
normAbas=np.linalg.norm(Abas, axis=0)
Abas=Abas/np.atleast_2d(normAbas)
Abas=Abas.T
As=CanoTensor(name='A', core=normAbas**2, basis=[Abas, Abas])

# Grad
hGrad_s=[]
for ii in range(dim):
    basis=[]
    for jj in range(dim):
        if ii==jj:
            basis.append(np.atleast_2d(freq[jj]*2*np.pi*1j))
        else:
            basis.append(np.atleast_2d(np.ones(N[jj])))
    hGrad_s.append(CanoTensor(name='hGrad({})'.format(ii), core=np.array([1]), basis=basis,
                              Fourier=True))

# R.H.S.
Es=CanoTensor(name='E', core=np.array([1.]),
               basis=[np.atleast_2d(np.ones(N[ii])) for ii in range(dim)], Fourier=False)

Bs=-(hGrad_s[0]*(As*Es).fourier())
print(np.linalg.norm(B-Bs.full()))


# linear operator
def GFAFG_fun_s(Fx, rank=rank, tol=tol):
    GFx=[hGrad_s[ii]*Fx for ii in range(dim)]
    FGFx=[GFx[ii].fourier() for ii in range(dim)]
    AFGFx=[As.multiply(FGFx[ii], rank=rank, tol=tol) for ii in range(dim)]
    FAFGFx=[AFGFx[ii].fourier() for ii in range(dim)]
    GFAFGFx=hGrad_s[0]*FAFGFx[0]
    for ii in range(1, dim):
        GFAFGFx+=hGrad_s[ii]*FAFGFx[ii]
    GFAFGFx=GFAFGFx.truncate(rank=rank, tol=tol)
    GFAFGFx.name='fun(x)'
    return-GFAFGFx

# testing
print('testing A...')
x=CanoTensor(name='a', r=4, N=N, randomise=True)
Fx=x.fourier()
res1=GFAFG_fun_s(Fx)
res2=GFAFG_fun(fft(x.full()))
print(np.linalg.norm(res1.full()-res2))

# solver
# norm = lambda X: X.norm(ord='core')
norm=lambda X: np.linalg.norm(X.full())

print('\n== SPARSE Richardson solver with preconditioner =======================')
# preconditioner
from scipy.linalg import svd
U, S, Vh=sp.svds(1./k2, k=4)
# U,S,Vh = svd(1./k2)

print('singular values of P={}'.format(S))

Ps=CanoTensor(name='P', core=S[:3], basis=[U[:, :3].T, Vh[:3]], Fourier=True)
print('norm(P-Ps)={}'.format(np.linalg.norm(1./k2-Ps.full())))

def PGFAFG_fun_s(Fx, rank=rank, tol=tol):
    R=GFAFG_fun_s(Fx, rank=rank, tol=tol)
    R=Ps*R
#     R=R.truncate(tol=1e-3)
    R=R.truncate(rank=rank, tol=tol)
    return R

PBs=Ps*Bs
Fus, ress=richardson(Afun=PGFAFG_fun_s, B=PBs, par=parP, norm=norm)
Fus.name='Fus'
Fus=Fus.truncate(rank=10)

print('solver results...')
print Fus
print('iterations={}'.format(ress['kit']))
print('norm(dif)={}'.format(np.linalg.norm(Fu-Fus.full().ravel())))
print('norm(resP)={}'.format(ress['norm_res']))
print('norm(resP)={}'.format(np.linalg.norm((PBs-PGFAFG_fun_s(Fus)).full())))
print('norm(res)={}'.format(np.linalg.norm((Bs-GFAFG_fun_s(Fus)).full())))

U, S, Vh=svd(Fus.full())
print('sing. vals of solution={}'.format(S))

print(Fus.core)

print('END')
