import numpy as np
import scipy.sparse.linalg as sp
import itertools

# PARAMETERS ##############################################################
ndim  = 2            # number of dimensions (works for 2D and 3D)
N     = ndim*(5,)    # number of voxels (assumed equal for all directions)

# auxiliary values
prodN = np.prod(np.array(N)) # number of grid points
ndof  = ndim*prodN # number of degrees-of-freedom
vec_shape=(ndim,)+N # shape of the vector for storing DOFs

# PROBLEM DEFINITION ######################################################
phase = np.zeros(N)
phase[:3, :3] = 1.
# A = np.einsum('ij,...->ij...',np.eye(ndim),1.+10.*np.random.random(N)) # material coefficients
A = np.einsum('ij,...->ij...',np.eye(ndim),1-phase) # material coefficients
A += np.einsum('ij,...->ij...',10*np.eye(ndim),phase) # material coefficients
E = np.zeros(vec_shape); E[0] = 1. # set macroscopic loading

# PROJECTION IN FOURIER SPACE #############################################
Ghat = np.zeros((ndim,ndim)+ N) # zero initialize
freq = [np.arange(-(N[ii]-1)/2.,+(N[ii]+1)/2.) for ii in range(ndim)]
for i,j in itertools.product(range(ndim),repeat=2):
    for ind in itertools.product(*[range(n) for n in N]):
        q = np.empty(ndim)
        for ii in range(ndim):
            q[ii] = freq[ii][ind[ii]]  # frequency vector
        if not q.dot(q) == 0:          # zero freq. -> mean
            Ghat[i,j][ind] = -(q[i]*q[j])/(q.dot(q))

# OPERATORS ###############################################################
dot21  = lambda A,v: np.einsum('ij...,j...  ->i...',A,v)
fft    = lambda V: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(V),N))
ifft   = lambda V: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(V),N))
G_fun  = lambda V: np.real(ifft(dot21(Ghat,fft(V)))).reshape(-1)
A_fun  = lambda v: dot21(A,v.reshape(vec_shape))
GA_fun = lambda v: G_fun(A_fun(v))

# CONJUGATE GRADIENT SOLVER ###############################################
b = -GA_fun(E) # right-hand side
e, _=sp.cg(A=sp.LinearOperator(shape=(ndof, ndof), matvec=GA_fun, dtype='float'), b=b)

aux = e+E.reshape(-1)
print('auxiliary field for macroscopic load E = {1}:\n{0}'.format(aux.reshape(vec_shape),
                                                                  format((1,)+(ndim-1)*(0,))))
print('homogenised properties A11 = {}'.format(np.inner(A_fun(aux).reshape(-1), aux)/prodN))

### POTENTIAL SOLVER in real space #####################################################################
print('\n== Potential solver in real space =======================')
# GRADIENT IN FOURIER SPACE #############################################
Ghat = np.zeros((ndim,)+ N) # zero initialize
freq = [np.arange(-(N[ii]-1)/2.,+(N[ii]+1)/2.) for ii in range(ndim)]
for ind in itertools.product(*[range(n) for n in N]):
    q = np.empty(ndim)
    for ii in range(ndim):
        q[ii] = freq[ii][ind[ii]]  # frequency vector
    if not q.dot(q) == 0:          # zero freq. -> mean
        for i in range(ndim):
            Ghat[i][ind] = - q[i]
Ghat = Ghat*2*np.pi*1j

# OPERATORS ###############################################################
FGrad = lambda Fu: np.einsum('i...,...->i...', Ghat, Fu)
FDiv = lambda Fe: np.einsum('i...,i...->...', Ghat, Fe)
Grad = lambda u: ifft(FGrad(fft(u))).real
Div = lambda e: ifft(FDiv(fft(e))).real
DivAGrad_fun = lambda u: Div(dot21(A, Grad(u.reshape(N)))).ravel()

# CONJUGATE GRADIENT SOLVER ###############################################
b = -Div(dot21(A,E)).ravel() # right-hand side
u, _=sp.cg(A=sp.LinearOperator(shape=(prodN, prodN), matvec=DivAGrad_fun, dtype='float'), b=b)

e = Grad(u.reshape(N))
aux = e+E
# print('auxiliary field for macroscopic load E = {1}:\n{0}'.format(aux.reshape(vec_shape),
#                                                                   format((1,)+(ndim-1)*(0,))))
print('homogenised properties A11 = {}'.format(np.sum(dot21(A,aux)*aux)/prodN))

### POTENTIAL SOLVER in Fourier space #####################################################################
print('\n== Potential solver in Fourier space =======================')
GFAFG_fun = lambda Fu: FDiv(fft(dot21(A, ifft(FGrad(Fu.reshape(N)))))).ravel()
b = -FDiv(fft(dot21(A,E))).ravel() # right-hand side

Fu, _=sp.cg(A=sp.LinearOperator(shape=(prodN, prodN), matvec=GFAFG_fun, dtype='complex'), b=b)

e = ifft(FGrad(Fu.reshape(N))).real
aux = e+E
print('auxiliary field for macroscopic load E = {1}:\n{0}'.format(aux.reshape(vec_shape),
                                                                   format((1,)+(ndim-1)*(0,))))
print('homogenised properties A11 = {}'.format(np.sum(dot21(A,aux)*aux)/prodN))

print('END')
