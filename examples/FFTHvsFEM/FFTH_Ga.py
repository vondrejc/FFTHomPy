import numpy as np
import itertools
from scipy.sparse.linalg import cg, LinearOperator
from functions import material_coef_at_grid_points, get_matinc, square_weights

# PARAMETERS
dim   = 2 # dimension (works for 2D and 3D)
N     = 5*np.ones(dim, dtype=np.int) # number of grid points
phase = 10. # material contrast
assert(np.array_equal(N % 2, np.ones(dim, dtype=np.int)))

dN = 2*N-1 # grid value
vec_shape=(dim,)+tuple(dN) # shape of the vector for storing DOFs

# OPERATORS
Agani = material_coef_at_grid_points(N, phase)
dot   = lambda A, B: np.einsum('ij...,j...->i...', A, B)
fft   = lambda x, N: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), N)) / np.prod(N)
ifft  = lambda x, N: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x), N)) * np.prod(N)
freq  = [np.arange(np.fix(-n/2.), np.fix(n/2.+0.5)) for n in dN]

# SYSTEM MATRIX for Galerkin approximation with exact integration (FFTH-Ga)
mat, inc = get_matinc(dim, phase)
h = 0.6*np.ones(dim) # size of square (rectangle) / cube
char_square = ifft(square_weights(h, dN, freq), dN).real
Aga = np.einsum('ij...,...->ij...', mat+inc,   char_square) \
    + np.einsum('ij...,...->ij...', mat, 1.-char_square)

# PROJECTION
Ghat = np.zeros((dim,dim)+ tuple(dN)) # zero initialize
indices  = [range(int((dN[k]-N[k])/2), int((dN[k]-N[k])/2+N[k])) for k in range(dim)]
for i,j in itertools.product(range(dim),repeat=2):
    for ind in itertools.product(*indices):
        q = np.array([freq[ii][ind[ii]] for ii in range(dim)]) # frequency vector
        if not q.dot(q) == 0: # zero freq. -> mean
            Ghat[(i,j)+ind] = -(q[i]*q[j])/(q.dot(q))

# OPERATORS
G_fun  = lambda X: np.real(ifft(dot(Ghat, fft(X, dN)), dN)).reshape(-1)
A_fun  = lambda x: dot(Aga, x.reshape(vec_shape))
GA_fun = lambda x: G_fun(A_fun(x))

# CONJUGATE GRADIENT SOLVER
X = np.zeros((dim,) + tuple(dN), dtype=np.float)
E = np.zeros(vec_shape); E[0] = 1. # macroscopic value
b = -GA_fun(E.reshape(-1))

Alinoper = LinearOperator(shape=(X.size, X.size), matvec=GA_fun, dtype=np.float)
eE, info = cg(A=Alinoper, b=b, x0=X.reshape(-1)) # conjugate gradients
aux = eE.reshape(vec_shape) + E

# POSTPROCESSING to calculate guaranteed bound
AH_11 = np.sum(dot(Aga, aux)*aux)/np.prod(dN)
print('homogenised component AH11 = {} (FFTH-Ga)'.format(AH_11))

print('END')
