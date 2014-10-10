import numpy as np

class TrigPolynomial():
    @staticmethod
    def get_ZNl(N):
        r"""
        it produces index set ZNl=\underline{\set{Z}}^d_N :
        ZNl[i][j]\in\set{Z} : -N[i]/2 <= ZNl[i] < N[i]/2
        """
        ZNl = []
        N = np.array(N)
        for m in np.arange(np.size(N)):
            ZNl.append(np.arange(np.fix(-N[m]/2.), np.fix(N[m]/2.+0.5)))
        return ZNl

    @staticmethod
    def get_xil(N, Y):
        """
        it produces discrete frequencies of Fourier series
        xil[i] = ZNl[i]/Y[i]
        """
        xil = []
        for m in np.arange(np.size(N)):
            xil.append(np.arange(np.fix(-N[m]/2.), np.fix(N[m]/2.+0.5))/Y[m])
        return xil

    @staticmethod
    def get_grid_coordinates(N, Y):
        """
        It produces coordinates of the set of nodal points
        Coord[i][j] = x_N^{(i,j)}
        """
        d = np.size(N)
        ZNl = TrigPolynomial.get_ZNl(N)
        coord = np.zeros(np.hstack([d, N]))
        for ii in np.arange(d):
            x = ZNl[ii]/N[ii]*Y[ii]
            Nshape = np.ones(d)
            Nshape[ii] = N[ii]
            Nrep = np.copy(N)
            Nrep[ii] = 1
            coord[ii] = np.tile(np.reshape(x, Nshape), Nrep)
        return coord


class TrigPolBasis(TrigPolynomial):
    """
    This represents a basis functions of trigonometric polynomials.
    """
    def eval_phi_k_N(self, x):
        val = np.zeros_like(x)
        coef = 1./np.prod(self.N)
        for ii in self.get_ZNl(self.N)[0]:
            val += coef*np.exp(1j*np.pi*ii*(x/self.Y - 2*self.order/self.N))
        return val

    def eval_phi_k(self, x):
        val = np.exp(np.pi*1j*x*self.order)
        return val

    def get_nodes(self):
        ZNl = self.get_ZNl(self.N)[0]
        x_nodes = ZNl*2*self.Y/self.N
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
    xM = np.zeros(M, dtype=xN.dtype)
    M = np.array(M)
    N = np.array(np.shape(xN))
    if np.allclose(M, N):
        return xN
    dim = np.size(N)
    ibeg = (M-N+(N % 2))/2
    iend = (M+N+(N % 2))/2
    if dim == 2:
        xM[ibeg[0]:iend[0], ibeg[1]:iend[1]] = xN
    elif dim == 3:
        xM[ibeg[0]:iend[0], ibeg[1]:iend[1], ibeg[2]:iend[2]] = xN
    else:
        raise NotImplementedError()
    return xM


def enlarge_M(xN, M):
    """
    Matrix representation of enlarge function.

    Parameters
    ----------
    xN : numpy.ndarray of shape = (dim, dim) + N
        input matrix that is to be enlarged

    Returns
    -------
    xM : numpy.ndarray of shape = (dim, dim) + M
        output matrix that is enlarged
    M : array like
        number of grid points
    """
    M = np.array(M, dtype=np.int32)
    N = np.array(xN.shape[2:])
    if np.allclose(M, N):
        return xN
    xM = np.zeros(np.hstack([xN.shape[0], xN.shape[1], M]))
    for m in np.arange(xN.shape[0]):
        for n in np.arange(xN.shape[1]):
            xM[m][n] = enlarge(xN[m][n], M)
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
    M = np.array(M, dtype=np.int32)
    N = np.array(xN.shape, dtype=np.int32)
    dim = N.size
    ibeg = (N-M+(M % 2))/2
    iend = (N+M+(M % 2))/2
    if dim == 2:
        xM = xN[ibeg[0]:iend[0], ibeg[1]:iend[1]]
    elif dim == 3:
        xM = xN[ibeg[0]:iend[0], ibeg[1]:iend[1], ibeg[2]:iend[2]]
    return xM

def curl_norm(e, Y):
    """
    it calculates curl-based norm,
    it controls that the fields are curl-free with zero mean as
    it is required of electric fields

    Parameters
    ----------
        e - electric field
        Y - the size of periodic unit cell

    Returns
    -------
        curlnorm - curl-based norm
    """
    N = np.array(np.shape(e[0]))
    d = np.size(N)
    xil = TrigPolynomial.get_xil(N, Y)
    xiM = []
    Fe = []
    for m in np.arange(d):
        Nshape = np.ones(d)
        Nshape[m] = N[m]
        Nrep = np.copy(N)
        Nrep[m] = 1
        xiM.append(np.tile(np.reshape(xil[m], Nshape), Nrep))
        Fe.append(DFT.fftnc(e[m], N)/np.prod(N))

    if d == 2:
        Fe.append(np.zeros(N))
        xiM.append(np.zeros(N))

    ind_mean = tuple(np.fix(N/2))
    curl = []
    e0 = []
    for m in np.arange(3):
        j = (m+1) % 3
        k = (j+1) % 3
        curl.append(xiM[j]*Fe[k]-xiM[k]*Fe[j])
        e0.append(np.real(Fe[m][ind_mean]))
    curl = np.array(curl)
    curlnorm = np.real(np.sum(curl[:]*np.conj(curl[:])))
    curlnorm = (curlnorm/np.prod(N))**0.5
    norm_e0 = np.linalg.norm(e0)
    if norm_e0 > 1e-10:
        curlnorm = curlnorm/norm_e0
    return curlnorm


def div_norm(j, Y):
    """
    it calculates divergence-based norm,
    it controls that the fields are divergence-free with zero mean as
    it is required of electric current

    Parameters
    ----------
        j - electric current
        Y - the size of periodic unit cell

    Returns
    -------
        divnorm - divergence-based norm
    """
    N = np.array(np.shape(j[0]))
    d = np.size(N)
    ind_mean = tuple(np.fix(N/2))
    xil = VecTri.get_xil(N, Y)
    R = 0
    j0 = np.zeros(d)
    for m in np.arange(d):
        Nshape = np.ones(d)
        Nshape[m] = N[m]
        Nrep = np.copy(N)
        Nrep[m] = 1
        xiM = np.tile(np.reshape(xil[m], Nshape), Nrep)
        Fj = DFT.fftnc(j[m], N)/np.prod(N)
        j0[m] = np.real(Fj[ind_mean])
        R = R + xiM*Fj
    divnorm = np.real(np.sum(R[:]*np.conj(R[:]))/np.prod(N))**0.5
    norm_j0 = np.linalg.norm(j0)
    if norm_j0 > 1e-10:
        divnorm = divnorm / norm_j0
    return divnorm




def get_Nodd(N):
    Nodd = N - ((N + 1) % 2)
    return Nodd
