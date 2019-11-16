import numpy as np
import scipy.sparse.linalg as sp
import itertools
from functions import get_matinc, material_coef_at_grid_points, enlarge, square_weights

# PARAMETERS
dim   = 2 # dimension (works for 2D and 3D)
N     = 5*np.ones(dim, dtype=np.int) # number of grid points
phase = 10. # material contrast
assert(np.array_equal(N % 2, np.ones(dim, dtype=np.int)))

# auxiliary values
ndof  = dim*np.prod(N) # number of degrees-of-freedom
vec_shape=(dim,)+tuple(N) # shape of the vector for storing DOFs

# PROJECTION IN FOURIER SPACE
Ghat = np.zeros((dim,dim)+ tuple(N)) # zero initialize
freq = [np.arange(-(N[ii]-1)/2.,+(N[ii]+1)/2.) for ii in range(dim)]
for i,j in itertools.product(range(dim),repeat=2):
    for ind in itertools.product(*[range(n) for n in N]):
        q = np.array([freq[ii][ind[ii]] for ii in range(dim)]) # frequency vector
        if not q.dot(q) == 0: # zero freq. -> mean
            Ghat[i,j][ind] = -(q[i]*q[j])/(q.dot(q))

# OPERATORS
Agani  = material_coef_at_grid_points(N, phase)
dot    = lambda A,v: np.einsum('ij...,j...  ->i...',A,v)
fft    = lambda V, N: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(V),N)) / np.prod(N)
ifft   = lambda V, N: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(V),N)) * np.prod(N)
G_fun  = lambda V: np.real(ifft(dot(Ghat,fft(V, N)), N)).reshape(-1)
A_fun  = lambda v: dot(Agani,v.reshape(vec_shape))
GA_fun = lambda v: G_fun(A_fun(v))

# CONJUGATE GRADIENT SOLVER
E   = np.zeros(vec_shape); E[0] = 1. # macroscopic value
b   = -GA_fun(E)                     # right-hand side
e, _= sp.cg(A=sp.LinearOperator(shape=(ndof, ndof), matvec=GA_fun, dtype='float'), b=b)

# POSTPROCESSING to calculate energetic value influenced by numerical integration
aux  = e+E.reshape(-1)
AH11 = np.inner(A_fun(aux).reshape(-1), aux)/np.prod(N)
print('homogenised component AH11 = {} (FFTH-GaNi - nonconforming)'.format(AH11))

# POSTPROCESSING to calculate guaranteed bound
dN = 2*N-1
freq = [np.arange(-(dN[ii]-1)/2.,+(dN[ii]+1)/2.) for ii in range(dim)]
mat, inc = get_matinc(dim, phase)
h = 0.6*np.ones(dim) # size of square (rectangle) / cube
char_square = ifft(square_weights(h, dN, freq), dN).real
Aga = np.einsum('ij...,...->ij...', mat+inc,   char_square) \
    + np.einsum('ij...,...->ij...', mat, 1.-char_square)

# interpolation/projection of microscopic field on double grid
Fe = fft(np.reshape(e, vec_shape), N)
Fe2 = np.zeros((dim,)+tuple(dN), dtype=np.complex)
for di in range(dim):
    Fe2[di]=enlarge(Fe[di], dN)
e2 = ifft(Fe2, dN).real

# evaluation of homogenised property
E2 = np.zeros((dim,)+tuple(dN)); E2[0] = 1. # macroscopic value
aux2 = e2 + E2
AH11 = np.sum(dot(Aga, aux2)*aux2)/np.prod(dN)
print('homogenised component AH11 = {} (FFTH-GaNi - conforming - upper bound on homogenised properties)'.format(AH11))
print('END')
