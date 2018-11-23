"""Basic methods for trigonometric polynomials."""

from __future__ import division
import numpy as np

fft_form_default='r' # real input data


class Grid():
    @staticmethod
    def get_ZNl(N, fft_form=fft_form_default):
        r"""
        it produces index set ZNl=\underline{\set{Z}}^d_N :
        ZNl[i][j]\in\set{Z} : -N[i]/2 <= ZNl[i] < N[i]/2
        """
        ZNl = []
        N = np.atleast_1d(np.array(N, dtype=np.int))
        for m in range(N.size):
            ZNl.append(np.arange(np.fix(-N[m]/2.), np.fix(N[m]/2.+0.5),
                                 dtype=np.int))
        if fft_form in ['r',0]:
            return [np.fft.ifftshift(val) for val in ZNl]
        else:
            return ZNl

    @staticmethod
    def get_xil(N, Y, fft_form=fft_form_default):
        """
        it produces discrete frequencies of Fourier series
        xil[i] = ZNl[i]/Y[i]
        """
        xil = []
        for m in np.arange(np.size(N)):
            xil.append(np.arange(np.fix(-N[m]/2.), np.fix(N[m]/2.+0.5))/Y[m])
        if fft_form in ['r',]:
            xil=[np.fft.ifftshift(xi) for xi in xil]
            xil[-1] = xil[-1][:int(np.fix(N[-1]/2)+1)]
        elif fft_form in [0]:
            xil = [np.fft.ifftshift(xi) for xi in xil]
        return xil

    @staticmethod
    def get_freq(N, Y, fft_form=fft_form_default):
        return Grid.get_xil(N, Y, fft_form=fft_form)

    @staticmethod
    def get_product(xi):
        xis = np.atleast_2d(xi[0])
        for ii in range(1, len(xi)):
            xis_new = np.tile(xi[ii], xis.shape[1])
            xis_old = np.repeat(xis, xi[ii].size, axis=1)
            xis = np.vstack([xis_old, xis_new])
        return xis

    @staticmethod
    def get_coordinates(N, Y):
        """
        It produces coordinates of the set of nodal points
        Coord[i][j] = x_N^{(i,j)}
        """
        d = np.size(N)
        ZNl = Grid.get_ZNl(N, fft_form='c')
        coord = np.zeros(np.hstack([d, N]))
        for ii in np.arange(d):
            x = Y[ii]*ZNl[ii]/N[ii]
            Nshape = np.ones(d, dtype=np.int)
            Nshape[ii] = N[ii]
            Nrep = np.copy(N)
            Nrep[ii] = 1
            coord[ii] = np.tile(np.reshape(x, Nshape), Nrep)
        return coord


class TrigPolBasis(Grid):
    """
    This represents a basis functions of trigonometric polynomials.
    """
    def eval_phi_k_N(self, x):
        val = np.zeros_like(x, dtype=np.complex128)
        coef = 1./np.prod(self.N)
        for ii in self.get_ZNl(self.N)[0]:
            val += coef*np.exp(2*1j*np.pi*ii*(x/self.Y - self.order/self.N))
        return val

    def eval_phi_k(self, x):
        val = np.exp(2*np.pi*1j*x*self.order/self.Y)
        return val

    def get_nodes(self):
        ZNl = self.get_ZNl(self.N)[0]
        x_nodes = ZNl*self.Y/self.N
        vals = np.zeros_like(x_nodes)
        ind = self.order + np.fix(self.N/2)
        vals[ind] = 1
        return x_nodes, vals

    def __init__(self, order, N=None, Y=None):
        self.order = order
        self.dim = np.size(order)
        self.N = np.array(N)

        if Y is None:
            self.Y = np.ones(self.dim)

        if N is None:
            self.Fourier = True
            self.eval = self.eval_phi_k
        else:
            self.Fourier = False
            self.eval = self.eval_phi_k_N

    def __repr__(self):
        if self.Fourier:
            ss = "Fourier basis function for k = %d" % (self.order,)
        else:
            ss = "Shape basis function for k = %d and N = %s" \
                % (self.order, str(self.N))
        return ss

def get_inverse(A):
    """
    It calculates the inverse of conductivity coefficients at grid points,
    i.e. of matrix A_GaNi

    Parameters
    ----------
    A : numpy.ndarray

    Returns
    -------
    invA : numpy.ndarray
    """
    B = np.copy(A)
    N = np.array(A.shape[2:])
    d = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise NotImplementedError("Non-square matrix!")

    invA = np.eye(d).tolist()
    for m in np.arange(d):
        Bdiag = np.copy(B[m][m])
        B[m][m] = np.ones(N)
        for n in np.arange(m+1, d):
            B[m][n] = B[m][n]/Bdiag
        for n in np.arange(d):
            invA[m][n] = invA[m][n]/Bdiag
        for k in np.arange(m+1, d):
            Bnull = np.copy(B[k][m])
            for l in np.arange(d):
                B[k][l] = B[k][l] - B[m][l]*Bnull
                invA[k][l] = invA[k][l] - invA[m][l]*Bnull
    for m in np.arange(d-1, -1, -1):
        for k in np.arange(m-1, -1, -1):
            Bnull = np.copy(B[k][m])
            for l in np.arange(d):
                B[k][l] = B[k][l] - B[m][l]*Bnull
                invA[k][l] = invA[k][l] - invA[m][l]*Bnull
    invA = np.array(invA)
    return invA


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
    M = np.array(M, dtype=np.float)
    N = np.array(xN.shape, dtype=np.float)
    if np.allclose(M, N):
        return xN

    ibeg = np.ceil((M-N)/2).astype(dtype=np.int)
    iend = np.ceil((M+N)/2).astype(dtype=np.int)

    slc=[slice(ibeg[i],iend[i],1) for i in range(N.size)]
    xM = np.zeros(M.astype(dtype=np.int), dtype=xN.dtype)
    xM[tuple(slc)]=xN
    return xM

def decrease(xN, M):
    """
    Decreases an array of Fourier coefficients by omitting the highest
    frequencies.

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
    M = np.array(M, dtype=np.float)
    N = np.array(xN.shape, dtype=np.float)
    ibeg = np.fix((N-M+(M % 2))/2).astype(dtype=np.int)
    iend = np.fix((N+M+(M % 2))/2).astype(dtype=np.int)

    slc=[slice(ibeg[i],iend[i],1) for i in range(N.size)]
    return xN[tuple(slc)]

def get_Nodd(N):
    Nodd = N - ((N + 1) % 2)
    return Nodd

def mean_index(N, fft_form=fft_form_default):
    if fft_form in [0, 'r']:
        return tuple(np.zeros_like(N, dtype=np.int))
    elif fft_form in ['c']:
        return tuple(np.array(np.fix(np.array(N)/2), dtype=np.int))
