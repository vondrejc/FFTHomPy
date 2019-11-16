import numpy as np
import itertools
from fenics import SubDomain, near


def get_matinc(dim, phase):
    # set up the material coefficients for matrix phase (mat) and inclusion (inc)
    Rfun=lambda alp: np.array([[np.cos(alp), np.sin(alp)], [-np.sin(alp), np.cos(alp)]])
    if dim==2:
        R=Rfun(np.pi/3)
    elif dim==3:
        RA=np.eye(dim)
        RA[:2, :2]=Rfun(np.pi/3.)
        RB=np.eye(dim)
        RB[1:, 1:]=Rfun(np.pi/6.)
        R=RA.dot(RB)
    mat=R.dot(np.diag(np.arange(1, dim+1)).dot(R.T))
    inc=R.dot(phase*np.eye(dim).dot(R.T))
    return mat, inc

def material_coef_at_grid_points(N, phase):
    # calculates the material coefficients at grid points for a square inclusion
    dim = N.__len__()
    assert(np.array_equal(N % 5, np.zeros(dim)))
    mat, inc = get_matinc(dim, phase) # material coef. for matrix (mat) and inclusion (inc)
    topology = np.zeros(N)
    subindices=[slice(int(N[i]/5*1),int(N[i]/5*4)) for i in range(dim)]
    topology[tuple(subindices)]=1
    A = np.einsum('ij,...->ij...',mat,1.-topology) \
        + np.einsum('ij,...->ij...',mat+inc,topology) # material coefficients
    return A

def enlarge(xN, M):
    """
    Enlarge an array of Fourier coefficients by zeros.

    Parameters
    ----------
    xN : numpy.ndarray of shape = N
        input array that is to be enlarged

    Returns
    -------
    xM : numpy.ndarray of shape = M
        output array that is enlarged
    M : array like
        number of grid points
    """
    xM = np.zeros(M, dtype=xN.dtype)
    M = np.array(M, dtype=np.float)
    N = np.array(xN.shape, dtype=np.float)
    if np.allclose(M, N):
        return xN
    dim = N.size
    ibeg = np.ceil((M-N)/2).astype(dtype=np.int)
    iend = np.ceil((M+N)/2).astype(dtype=np.int)
    if dim == 3:
        xM[ibeg[0]:iend[0], ibeg[1]:iend[1], ibeg[2]:iend[2]] = xN
    elif dim == 2:
        xM[ibeg[0]:iend[0], ibeg[1]:iend[1]] = xN
    elif dim == 1:
        xM[ibeg[0]:iend[0]] = xN
    return xM

def square_weights(h, dN, freq):
    # calculation of integral weights of rectangular function for FFTH-Ga
    dim = h.size
    Wphi = np.zeros(dN) # integral weights
    for ind in itertools.product(*[range(n) for n in dN]):
        Wphi[ind] = np.prod(h)
        for ii in range(dim):
            Wphi[ind] *= np.sinc(h[ii]*freq[ii][ind[ii]])
    return Wphi

class PeriodicBoundary(SubDomain):
    # periodic boundary conditions for FEM
    def __init__(self, dim=2):
        SubDomain.__init__(self)
        self.dim = dim

    def inside(self, x, on_boundary):
        """ return True if on left or bottom boundary AND NOT on one of the
        two corners (0, 1) and (1, 0) """
        zero_boundary = False
        one_boundary = False
        corner = True
        for ii in range(self.dim):
            zero_boundary = zero_boundary or near(x[ii], 0.)
            one_boundary = one_boundary or near(x[ii], 1.)
            corner = corner and (near(x[ii], 0.) or near(x[ii], 1.))
        return bool(on_boundary and zero_boundary and not one_boundary)

    def map(self, x, y):
        for ii in range(self.dim):
            if near(x[ii], 1.):
                y[ii] = x[ii] - 1.
            else:
                y[ii] = x[ii]
