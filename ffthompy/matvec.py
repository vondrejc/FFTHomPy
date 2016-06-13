#!/usr/bin/python
"""
This module contains classes and functions for trigonometric polynomials and
relating operators for homogenization.
"""

import numpy as np
import numpy.matlib as npmatlib
from ffthompy.matvec_fun import Grid, enlarge, enlarge_M, get_inverse
import copy
import itertools


class FieldFun():
    """
    General class that provides functions for VecTri and Matrix classes.
    """
    def dN(self):
        return np.hstack([self.d, self.N])

    def ddN(self, M=None):
        if M is None:
            M = self.N
        return np.hstack([self.d, self.d, M])

    def pN(self):
        return np.prod(self.N)

    def pdN(self):
        return np.prod(self.dN())

    def zeros_like(self):
        X = copy.deepcopy(self)
        X.val[:] = 0
        return X

    def mean_ind(self):
        return tuple(np.round(np.array(self.N)/2))

    def __getitem__(self, i):
        return self.val[i]

    def __repr__(self, full=False):
        ss = "Class : %s\n    name : %s\n" % (self.__class__.__name__,
                                              self.name)
        ss += '    Fourier = %s \n' % (self.Fourier)
        ss += '    dimension d = %g \n' % (self.d)
        ss += '    size N = %s \n' % str(self.N)
        ss += '    val.shape  = %s \n' % str(self.val.shape)
        ss += '    norm = %s\n' % str(self.norm())
        ss += '    mean = %s\n' % str(self.mean())
        if full:
            ss += 'val = \n'
            ss += str(self.val)
        return ss

    @staticmethod
    def get_mean_index(N):
        return tuple(np.round(np.array(N)/2))


class Scalar():
    """
    Scalar value that is used to multiply VecTri or Matrix classes
    """
    def __init__(self, val=None, name='c'):
        if val is not None:
            self.val = val
        else:
            self.val = 1.
        self.name = name

    def __call__(self, x):
        return x*self.val

    def __repr__(self):
        ss = "Class : %s\n" % (self.__class__.__name__)
        ss += 'val = \n'
        ss += str(self.val)
        return ss

    def transpose(self):
        return self


class VecTri(FieldFun, Grid):
    """
    Class representing the trigonometric polynomial using values at grid points
    or using Fourier coefficients

    Parameters
    ----------
    name : str
        name of a vector
    Fourier : boolean
        if True vector is in Fourier space, store Fourier coefficients
    elas : boolean
        if True it regards elasticity (not Implemented yet)
    val : numpy.ndarray of shape = (d, N)
        depending on Fourier, it stores the point values or
        Fourier coefficients of trigonometric polynomials
        the values are stored according to:
        kwargs['val'] : numpy.ndarray of shape (d,N)
        kwargs['macroval'] : numpy.ndarray of shape (d,)
        valtypes : str
            either of 'ones' or 'random'
    """
    def __init__(self, name='?', N=None, d=None, Fourier=False, valtype=None,
                 **kwargs):
        self.Fourier = Fourier

        if 'val' in kwargs:
            self.val = kwargs['val']
            self.N = np.array(self.val.shape[1:])
            self.d = self.val.shape[0]
        else:
            if N is not None:
                self.N = np.array(N, dtype=np.int32)
            else:
                raise ValueError("Parameter N is required!")
            if d is None:
                self.d = self.N.size
            else:
                self.d = d

            if 'macroval' in kwargs:
                self.name = 'macroval'
                self.d = kwargs['macroval'].size
                self.val = np.zeros(self.dN())
                for m in np.arange(self.d):
                        self.val[m] = kwargs['macroval'][m]
            elif valtype is 'ones':
                self.name = 'ones'
                self.val = np.ones(self.dN())
            elif valtype in ['random', 'rand']:
                self.name = 'random'
                self.val = np.random.random(self.dN())
            else:
                if self.Fourier:
                    dtype = np.complex128
                else:
                    dtype = np.float64
                self.name = '0'
                self.val = np.zeros(self.dN(), dtype=dtype)

        if 'Y' in kwargs:
            self.Y = np.array(kwargs['Y'])

        if name is not None:
            self.name = name

        self.valshape = self.val.shape
        self.size = np.size(self.val)

        topo_dim = self.N.size
        if self.d == topo_dim:
            self.physics = 'scalar'
        elif self.d == topo_dim*(topo_dim+1)/2:
            self.physics = 'elasticity'
        else:
            ValueError()

    def __mul__(self, x):
        if isinstance(x, VecTri):
            scal = np.real(np.sum(self.val[:]*np.conj(x.val[:])))
            if not self.Fourier:
                scal = scal / np.prod(self.N)
            return scal

        elif isinstance(x, np.ndarray):
            if np.size(x) == 1: # scalar value
                name = get_name(self.name, '*', 'c')
                scal = VecTri(name=name, val=np.array(x)*self.val)
                self.val *= x
                return self

            elif x.shape == (self.d, self.d):
                return self.mul_tensorR(x)

            elif all(self.val.shape == x.shape):
                return np.real(np.sum(self.val[:]*np.conj(x[:])))

        else:
            raise ValueError("The shape of vectors are not appropriate.")

    def __rmul__(self, x):
        if isinstance(x, Scalar):
            name = get_name('c', '*', self.name)
            return VecTri(name=name, val=x.val*self.val)

        elif np.size(x) == 1:
            name = get_name('c', '*', self.name)
            return VecTri(name=name, val=x*self.val, Fourier=self.Fourier)

        else:
            raise ValueError()


    def __add__(self, x):
        if isinstance(x, VecTri):
            name = get_name(self.name, '+', x.name)
            if self.Fourier != x.Fourier:
                raise ValueError("Mismatch in Fourier/shape coefficients!")
            summ = VecTri(name=name, val=self.val+x.val, Fourier=self.Fourier)
        else:
            summ = VecTri(name=self.name, val=self.val+x,
                          Fourier=self.Fourier)
        return summ

    def __radd__(self, x):
        return self+x

    def __neg__(self):
        return VecTri(name='-'+self.name, val=-self.val, Fourier=self.Fourier)

    def __sub__(self, x):
        return self.__add__(-x)

    def norm(self, ntype='L2'):
        if ntype == 'L2':
            scal = (self*self)**0.5
        elif ntype == 2:
            scal = (self*self)**0.5
        elif ntype == 1:
            scal = np.sum(np.abs(self.val))
        elif ntype == 'inf':
            scal = np.sum(np.abs(self.val))
        elif ntype == 'curl':
            scal = curl_norm(self.val, self.Y)
        elif ntype == 'div':
            scal = div_norm(self.val, self.Y)
        else:
            msg = "The norm (%s) of VecTri is not implemented!" % ntype
            raise NotImplementedError(msg)

        return scal

    def mean(self):
        """
        Mean of trigonometric polynomial of shape of macroscopic vector.
        """
        mean = np.zeros(self.d)
        if self.Fourier:
            ind = tuple(np.round(np.array(self.N)/2))
            for di in np.arange(self.d):
                mean[di] = np.real(self.val[di][ind])
        else:
            for di in np.arange(self.d):
                mean[di] = np.mean(self.val[di])
        return mean

    def __call__(self):
        return self.val

    def vec(self):
        """
        Returns one-dimensional vector (column) version of trigonometric
        polynomial.
        """
        return np.matrix(self.val.ravel()).transpose()

    def __eq__(self, x):
        """
        Check the equality with other objects comparable to trig. polynomials.
        """
        if isinstance(x, VecTri):
            res = (self-x).norm()
        elif np.shape(x) == self.get_shape():
            res = np.linalg.norm(self.val-x)
        else:
            res = False
        return res

    def enlarge(self, M):
        """
        It enlarges a trigonometric polynomial by adding zeros to the Fourier
        coefficients with high frequencies.
        """
        if np.allclose(self.N, M):
            return self
        val = np.zeros(np.hstack([self.d, M]), dtype=self.val.dtype)
        if self.Fourier is False:
            for m in np.arange(self.d):
                val[m] = enlargeF(self.val[m], M)
        else:
            for m in np.arange(self.d):
                val[m] = enlarge(self.val[m], M)
        return VecTri(name=self.name, val=val, Fourier=self.Fourier)

    def mulTri(self, y, resize=True):
        if isinstance(y, VecTri):
            if resize:
                M = np.max(np.vstack([self.N, y.N]), axis=0)
                x2 = self.resize(2*M)
                y2 = y.resize(2*M)
                return VecTri(name=self.name+y.name, val=x2.val*y2.val)

    def mul_elementwise(self, x):
        if self.Fourier != x.Fourier:
            msg = "Mismatch of Fourier coefficients and nodal values!"
            raise ValueError(msg)
        name = self.name + '*' + x.name
        return VecTri(name=name, val=self.val*x.val, Fourier=self.Fourier)

    def mul_tensorL(self, val):
        return VecTri(name=self.name,
                      val=np.einsum('ij...,j...->i...', val, self.val),
                      Fourier=self.Fourier)

    def mul_tensorR(self, val):
        return VecTri(name=self.name,
                      val=np.einsum('i...,ij...->j...', self.val, val),
                      Fourier=self.Fourier)

    def mul_gridwise(self, val):
        return VecTri(name=self.name,
                      val=np.einsum('...,i...->i...', val, self.val),
                      Fourier=self.Fourier)

    def scal_tensor(self, x):
        return np.einsum('i...,i...->...', self.val, x.val)

    def outer_tensor(self, x):
        return Matrix(val=np.einsum('i...,j...->ij...', self.val, x.val),
                      Fourier=self.Fourier)

    def get_S_subvector(self, ss=None):
        # NOT WORKING
        if ss is None:
            ss = np.zeros(self.d)
        else:
            ss = np.array(ss)
        ind0 = np.arange(0, self.N[0], 2) + 1
        ind1 = np.arange(0, self.N[1], 2) + 1
        subV = self.val[0]
        return subV[ind0-ss[0], :][:, ind1-ss[1]]

    def transpose(self):
        return self

    def fourier_transform(self):
        """
        Either discrete Fourier- or discrete inverse Fourier transform
        """
        DFT(inverse=not self.Fourier, N=self.N)
        return DFT(self)

    @property
    def T(self):
        return self


def get_name(x_name, oper, y_name):
    name = x_name + oper + y_name
    if len(name) > 20:
        name = 'oper(%s)' % oper
    return name


class Matrix(FieldFun):
    """
    Structured matrix storing the values of material or
    integral kernel in Fourier space (projection)

    parameters :
    Fourier : boolean
        information whether the values are in Fourier space or not
    Id : boolean
        if True it assemble identity matrix
    kwargs['homog'] : numpy.ndarray of shape N
        assemble the matrix to constant matrix
    kwargs['val'] : numpy.ndarray of shape (d,d,N)
        assemble the matrix to predefined values
    """
    def __init__(self, name='?', Fourier=False, valtype='val', **kwargs):
        self.Fourier = Fourier
        self.name = name
        self.valtype = valtype
        self.__dict__.update(kwargs)

        if valtype in ['val']:
            self.val = np.array(self.val)
            self.N = np.array(self.val.shape[2:])
            self.d = self.val.shape[0]
            self.dtype = self.val.dtype
            if self.val.shape[1] != self.d:
                raise ValueError("Improper dimension of values %s."
                                 % str(self.val.shape))
        else:
            if not hasattr(self, 'N'):
                raise ValueError("Argument 'N' has to be defined!")
            if not hasattr(self, 'd'):
                self.d = np.size(self.N)

            if self.Fourier:
                self.dtype = np.complex128
            else:
                self.dtype = np.float64

            if valtype in ['Id', 'id', 'identity']:
                self.val = np.zeros(self.ddN(), dtype=self.dtype)
                for m in np.arange(self.d):
                    self.val[m][m] = 1.

            elif valtype in ['random']:
                self.val = np.random.random(self.ddN())

            elif valtype in ['homog']:
                self.val = np.zeros(self.ddN(), dtype=self.dtype)
                for m in np.arange(self.d):
                    for n in np.arange(self.d):
                        self.val[m, n] = np.array(kwargs['val'][m, n])

    def __mul__(self, x):
        if isinstance(x, VecTri): # Matrix by VecTri multiplication
            name = get_name(self.name, '*', x.name)
            prod = VecTri(name=name,
                          val=np.einsum('ij...,j...->i...', self.val, x.val),
                          Fourier=x.Fourier)
        elif isinstance(x, Matrix): # Matrix by Matrix multiplication
            name = get_name(self.name, '*', x.name)
            prod = Matrix(name=name,
                          val=np.einsum('ij...,jk...->ik...', self.val, x.val))
        elif isinstance(x, LinOper) or isinstance(x, DFT):
            name = get_name(self.name, '*', x.name)
            prod = LinOper(name=name, mat=[[self, x]])
        elif isinstance(x, Scalar):
            name = get_name(self.name, '*', 'c')
            prod = Matrix(name=name, val=self.val*x.val)
        elif np.size(x) == 1: # Matrix by Constant multiplication
            name = get_name(self.name, '*', 'c')
            prod = Matrix(name=name, val=self.val*x)
        elif np.size(x) == self.pdN():
            val = np.einsum('ij...,j...->i...', self.val,
                            np.reshape(x, self.dN()))
            prod = np.reshape(val, self.pdN())
        else:
            name = get_name(self.name, '*', 'np.array')
            prod = VecTri(name=name,
                          val=np.einsum('ij...,j...->i...', self.val, x))
        return prod

    def __rmul__(self, x):
        if np.shape(x) == (self.d, self.d):
            # Matrix by (d,d)-array multiplication
            val = np.zeros(self.ddN())
            for m in np.arange(self.d):
                for n in np.arange(self.d):
                    val[m, n] = x[m, n]*self.val[m, n]
            return Matrix(val=val)
        else:
            return self*x

    def norm(self):
        return np.sum(self.val**2)**0.5

    def mean(self):
        res = np.zeros([self.d, self.d])
        if self.Fourier:
            ind = self.get_mean_index(self.N)
            for m in np.arange(self.d):
                for n in np.arange(self.d):
                    res[m, n] = self.val[m, n][ind]
        else:
            for m in np.arange(self.d):
                for n in np.arange(self.d):
                    res[m, n] = np.mean(self.val[m, n])
        return res

    def __add__(self, x):
        if isinstance(x, Matrix):
            name = get_name(self.name, '+', x.name)
            summ = Matrix(name=name, val=self.val+x.val)
        else:
            summ = Matrix(val=self.val+x)
        return summ

    def __call__(self, x):
        return self*x

    def __neg__(self):
        return Matrix(val=-self.val)

    def __sub__(self, x):
        if isinstance(x, Matrix):
            res = -x + self
        else:
            res = 'this type of operation is not supported'
        return res

    def __div__(self, x):
        return self*(1./x)

    def __getitem__(self, i):
        return self.val[i]

    def T(self):
        return self.transpose()

    def transpose(self):
        return Matrix(name=self.name,
                      val=np.einsum('ij...->ji...', self.val),
                      Fourier=self.Fourier)

    def inv(self):
        name = 'inv(%s)' % (self.name)
        if self.Fourier is False:
            return Matrix(name=name, val=get_inverse(self.val), Fourier=False)
        else:
            raise NotImplementedError("The inverse for Fourier coefficients!")

    def mul_tensorR(self, val):
        if val.shape == (self.d, self.d):
            return Matrix(name=self.name,
                          val=np.einsum('ij...,jk...->ik...', self.val, val),
                          Fourier=self.Fourier)
        elif val.shape == (self.d,):
            return VecTri(name=self.name,
                          val=np.einsum('ij...,j...->i...', self.val, val),
                          Fourier=self.Fourier)

    def mul_gridwise(self, val):
        return Matrix(name=self.name,
                      val=np.einsum('...,ij...->ij...', val, self.val),
                      Fourier=self.Fourier)

    def __eq__(self, x):
        if isinstance(x, Matrix):
            if self.val.shape == x.val.shape:
                res = (self-x).norm()
            else:
                res = 'same instance (Matrix); different shapes: %s, %s' \
                    % (str(self.val.shape), x.val.shape)
        elif all(self.get_shape() == np.shape(x)):
            res = 'different instances (Matrix vs numpy.array), norm = %f' \
                % (np.linalg.norm(np.reshape(self.val-x, self.ddN())))
        else:
            res = False
        return res

    def enlarge(self, M):
        if self.Fourier:
            val = enlarge_M(self.val, M)
        else:
            val = np.zeros(self.ddN())
            for ii in np.arange(self.d):
                for jj in np.arange(self.d):
                    val[ii, jj] = enlargeF(self.val[ii, jj], M)
        return Matrix(name=self.name, val=val, Fourier=self.Fourier)

    def get_shifted_submatrix(self, ss=None):
        if ss is None:
            ss = np.zeros(self.d, dtype=np.int)
        else:
            ss = np.array(ss, dtype=np.int)
        ind0 = np.arange(0, self.N[0], 2) + 1
        ind1 = np.arange(0, self.N[1], 2) + 1
        SM = Matrix(N=self.N/2)
        for ii in np.arange(self.d):
            for jj in np.arange(self.d):
                SM.val[ii, jj] = self.val[ii, jj][ind0-ss[0], :][:, ind1-ss[1]]
        return SM

    def matrix(self):
        """
        Returns full matrix version of Matrix class.
        """
        pN = np.prod(self.N)
        proddN = self.d*pN
        matrix = np.matrix(np.zeros([proddN, proddN], dtype=np.float64))
        for ii in np.arange(self.d):
            for jj in np.arange(self.d):
                submatrix = np.diag(self.val[ii, jj].ravel())
                matrix[pN*ii:pN**(ii+1), pN*jj:pN**(jj+1)] = submatrix
        return matrix


class ShiftMatrix():
    """
    Matrix object defining shift of Fourier coefficients.
    """
    @staticmethod
    def get_shift_matrix(N, ss=None):
        N = np.array(N)
        d = np.size(N)
        if ss is None:
            ss = np.zeros(d)
        else:
            ss = np.array(ss)

        def omeg2N(s, k, n):
            return np.exp(-2*np.pi*1j*(s*k/n))

        ZNl = VecTri.get_ZNl(N)
        SS = np.outer(omeg2N(ss[0], ZNl[0], 2*N[0]),
                      omeg2N(ss[1], ZNl[1], 2*N[1]))
        return SS

    def __init__(self, N):
        self.N = N
        self.d = np.size(N)

    def get_shift(self, ss, transpose=False):
        SS = Matrix(N=self.N, Fourier=True)
        S = self.get_shift_matrix(self.N, ss)
        if transpose:
            S = S.conj()
        for ii in np.arange(self.d):
            SS.val[ii, ii] = S
        return SS


class Id():
    """
    Identity operator.
    """
    def __init__(self, name='IdOper'):
        self.name = name

    def __call__(self, x):
        return x

    def __mul__(self, x):
        return x

    def __repr__(self):
        return 'Class : %s\n' % (self.__class__.__name__)


class DFT(FieldFun):
    """
    (inverse) Disrete Fourier Transform (DFT) to provide __call__
    by FFT routine.

    Parameters
    ----------
    inverse : boolean
        if True it provides inverse DFT
    N : numpy.ndarray
        N-sized (i)DFT,
    normalized : boolean
        version of DFT that is normalized by factor numpy.prod(N)
    """
    def __init__(self, inverse=False, N=None, normalized=True, **kwargs):
        self.__dict__.update(kwargs)
        if 'name' not in kwargs.keys():
            if inverse:
                self.name = 'iDFT'
            else:
                self.name = 'DFT'

        self.N = np.array(N, dtype=np.int32)
        self.inverse = inverse
        if normalized:
            self.norm_coef = np.prod(self.N)
        else:
            self.norm_coef = 1.

    def __mul__(self, x):
        return self.__call__(x)

    def __call__(self, x):
        if isinstance(x, VecTri):
            if not self.inverse:
                return VecTri(name=get_name('F', '*', x.name),
                              val=self.fftnc(x.val, self.N),
                              Fourier=not x.Fourier)
            else:
                return VecTri(name=get_name('Fi', '*', x.name),
                              val=np.real(self.ifftnc(x.val, self.N)),
                              Fourier=not x.Fourier)

        elif (isinstance(x, LinOper) or isinstance(x, Matrix)
                or isinstance(x, DFT)):
            return LinOper(mat=[[self, x]])

        else:
            if np.size(x) > np.prod(self.N):
                d = np.size(x)/np.prod(self.N)
                xre = np.reshape(x, np.hstack([d, self.N]))
            else:
                xre = np.reshape(x, self.N)
            if not self.inverse:
                Fxre = self.fftnc(xre, self.N)
            else:
                Fxre = np.real(self.ifftnc(xre, self.N))
            return np.reshape(Fxre, np.size(Fxre))

    def matrix(self):
        """
        This function returns the object as a matrix of DFT or iDFT resp.
        """
        N = self.N
        prodN = np.prod(N)
        proddN = self.d*prodN
        ZNl = Grid.get_ZNl(N)

        if self.inverse:
            DFTcoef = lambda k, l, N: np.exp(2*np.pi*1j*np.sum(k*l/N))
        else:
            DFTcoef = lambda k, l, N: np.exp(-2*np.pi*1j*np.sum(k*l/N))/np.prod(N)

        DTM = np.zeros([self.pN(), self.pN()], dtype=np.complex128)
        for ii, kk in enumerate(itertools.product(*tuple(ZNl))):
            for jj, ll in enumerate(itertools.product(*tuple(ZNl))):
                DTM[ii, jj] = DFTcoef(np.array(kk, dtype=np.float),
                                      np.array(ll), N)

        DTMd = npmatlib.zeros([proddN, proddN], dtype=np.complex128)
        for ii in range(self.d):
            DTMd[prodN*ii:prodN*(ii+1), prodN*ii:prodN*(ii+1)] = DTM
        return DTMd

    def __repr__(self):
        ss = "Class : %s\n" % (self.__class__.__name__,)
        ss += '    name : %s\n' % self.name
        ss += '    inverse = %s\n' % self.inverse
        ss += '    size N = %s\n' % str(self.N)
        return ss

    def transpose(self):
        return DFT(name=self.name+'^T', inverse=not(self.inverse), N=self.N)

    @staticmethod
    def fftnc(x, N):
        """
        centered n-dimensional FFT algorithm
        """
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), N))/np.prod(N)

    @staticmethod
    def ifftnc(Fx, N):
        """
        centered n-dimensional inverse FFT algorithm
        """
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(Fx), N))*np.prod(N)


class LinOper():
    """
    Linear operator composed of matrices or linear operators
    it is designed to provide __call__ function as a linear operation

    parameters :
        X : numpy.ndarray or VecTri or something else
            it represents the operand,
            it provides the information about size and shape of operand
        dtype : data type of operand, usually numpy.float64
    """
    def __init__(self, name='LinOper', dtype=None, X=None, **kwargs):
        self.name = name
        if 'mat_rev' in kwargs.keys():
            self.mat_rev = kwargs['mat_rev']
        elif 'mat' in kwargs.keys():
            self.mat_rev = []
            for summand in kwargs['mat']:
                no_oper = len(summand)
                summand_rev = []
                for m in np.arange(no_oper):
                    summand_rev.append(summand[no_oper-1-m])
                self.mat_rev.append(summand_rev)
        self.no_summands = len(self.mat_rev)

        if X is not None:
            self.define_operand(X)

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = np.float64

    def __mul__(self, x):
        if isinstance(x, VecTri):
            return self(x)
        elif (isinstance(x, Matrix) or isinstance(x, LinOper)
              or isinstance(x, DFT)):
            name = self.name + '*' + x.name
            return LinOper(name=name, mat=[[self, x]])

    def __add__(self, x):
        if isinstance(x, Matrix) or isinstance(x, LinOper):
            name = self.name + '+' + x.name
            return LinOper(name=name, mat=[[self], [x]])
        else:
            return 'This operation is not supported!'

    def __call__(self, x):
        res = 0.
        for summand in self.mat_rev:
            prod = x
            for matrix in summand:
                prod = matrix(prod)
            res = prod + res
        return res

    def __repr__(self):
        s = 'Class : %s\nname : %s\nexpression : ' % (self.__class__.__name__,
                                                      self.name)
        flag_sum = False
        no_sum = len(self.mat_rev)
        for isum in np.arange(no_sum):
            if flag_sum:
                    s += ' + '
            no_oper = len(self.mat_rev[isum])
            flag_mul = False
            for m in np.arange(no_oper):
                matrix = self.mat_rev[isum][no_oper-1-m]
                if flag_mul:
                    s += '*'
                s += matrix.name
                flag_mul = True
            flag_sum = True
        return s

    def define_operand(self, X):
        """
        This function defines the type of operand to correctly define linear
        operator.

        Parameters
        ----------
        X : any object
            operand of linear operator
        """
        if isinstance(X, VecTri):
            Y = self(X)
            self.shape = (Y.size, X.size)
            self.X_reshape = X.val.shape
            self.Y_reshape = Y.val.shape
        else:
            print 'LinOper : This operand is not implemented!'

    def matvec(self, x):
        """
        Provides the __call__ for operand recast into one-dimensional vector.
        This is suitable for e.g. iterative solvers when trigonometric
        polynomials are recast into one-dimensional numpy.arrays.

        Parameters
        ----------
        x : one-dimensional numpy.array
        """
        X = VecTri(val=self.revec(x))
        AX = self.__call__(X)
        return AX.vec()

    def vec(self, X):
        """
        Reshape the operand (VecTri) into one-dimensional vector (column)
        version.
        """
        return np.reshape(X, self.shape[1])

    def revec(self, x):
        """
        Reshape the one-dimensional vector of trig. pol. into shape occurring
        in class VecTri.
        """
        return np.reshape(x, self.Y_reshape)

    def transpose(self):
        """
        Transpose (adjoint) of linear operator.
        """
        mat = []
        for m in np.arange(self.no_summands):
            summand = []
            for n in np.arange(len(self.mat_rev[m])):
                summand.append(self.mat_rev[m][n].transpose())
            mat.append(summand)
        name = '(%s)^T' % self.name
        return LinOper(name=name, mat=mat)


class MultiVector():
    """
    MultiVector that is used for some mixed formulations.
    """
    def __init__(self, name='MultiVector', val=None):
        self.name = name
        self.val = val

        # parameters for vector like operations
        self.dim = len(self.val)
        self._iter = np.arange(self.dim)
        self.ltype = []
        self.ldtype = []
        self.lshape = []
        self.lsize = np.zeros(self.dim, dtype=np.int64)
        for m in self._iter:
            self.ltype.append(self.val[m].__class__.__name__)
            if self.ltype[-1] == 'VecTri':
                self.lshape.append(self.val[m].valshape)
                self.ldtype.append(self.val[m].dtype)
                self.lsize[m] = self.val[m].size

        self.size = np.sum(self.lsize)

    def __mul__(self, x):
        if isinstance(x, MultiVector):
            val = 0.
            for n in self._iter:
                val += self.val[n]*x.val[n]
            return val
        elif isinstance(x, Scalar):
            val = []
            for n in self._iter:
                val.append(self.val[n]*x)
            return MultiVector(val=val)
        elif np.size(x) == 1:
            val = self.val
            for n in self._iter:
                val[n] = val[n]*x
            return MultiVector(val=val)

    def __rmul__(self, x):
        return self*x

    def __call__(self, x):
        return self*x

    def __add__(self, x):
        val = []
        for n in self._iter:
            val.append(self.val[n] + x.val[n])
        return MultiVector(val=val)

    def __neg__(self):
        val = []
        for n in self._iter:
            val.append(-self.val[n])
        return MultiVector(val=val)

    def __sub__(self, x):
        return -x+self

    def __getitem__(self, m):
        return self.val[m]

    def __repr__(self):
        s = 'Class : %s\n    name : %s\n' % (self.__class__.__name__,
                                             self.name)
        s += '    dim = %d ; size = %d\n' % (self.dim, self.size)
        s += '    ltype = %s\n' % str(self.ltype)
        s += '    lshape = %s\n' % str(self.lshape)
        s += '    lsize = %s\n' % str(self.lsize)
        s += '    ldtype = %s\n' % str(self.ldtype)
        s += '    lnames : [ '
        flag_row = False
        for item in self._iter:
            if flag_row:
                s += ' , '
            s += self.val[item].name
            s += '(%s)' % self.val[item].__class__.__name__
            flag_row = True
        s += ' ]\n'
        s += '    val :\n'
        for m in self._iter:
            s += str(self[m])
        return s

    def vec(self):
        lx = []
        for m in self._iter:
            lx.append(self.val[m].vec())
        x = np.hstack(lx)
        return x

    def __eq__(self, x):
        ltype = []
        lval = []
        for m in self._iter:
            if self.ltype[m] == x[m].__class__.__name__:
                ltype.append(True)
                lval.append(self.val[m] == x.val[m])
            else:
                ltype.append(False)
                lval.append(False)
        return 'subvector types : %s; subvector equality : %s' % (str(ltype),
                                                                  str(lval))


class MultiOper():
    """
    MultiOperator used for some mixed formulations
    """
    def __init__(self, name='MultiOper', val=None):
        self.name = name
        self.val = val
        self.no_row = len(self.val)
        self.no_col = len(self.val[0])
        self.shape = (self.no_row, self.no_col)

    def __call__(self, x):
        if isinstance(x, MultiVector):
            val = list(np.zeros(self.no_row))
            for m in np.arange(self.no_row):
                for n in np.arange(self.no_col):
                    val[m] += self.val[m][n]*x[n]
        return MultiVector(val=val)

    def __mul__(self, x):
        return self(x)

    def __repr__(self):
        s = 'Class : %s\n    name : %s\n' % (self.__class__.__name__,
                                             self.name)
        s += '    expression :\n'
        for irow in np.arange(self.no_row):
            s += '        [ '
            flag_row = False
            for icol in np.arange(self.no_col):
                if flag_row:
                    s += ' , '
                s += self.val[irow][icol].name
                flag_row = True
            s += ' ]\n'
        return s

    def transpose(self):
        val = []
        for m in np.arange(self.no_col):
            row = []
            for n in np.arange(self.no_row):
                row.append(self.val[n][m].transpose())
            val.append(row)
        name = '(%s)^T' % self.name
        return MultiOper(name=name, val=val)


class ScipyOper():
    def __init__(self, name='ScipyLinOper', A=None, X=None, AT=None,
                 dtype=None):
        self.name = name
        self.A = A
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = np.float64

        if AT is not None:
            self.AT = AT

        Y = A(X)
        self.shape = (Y.size, X.size)
        self.X = X
        self.Y = Y

    def rmatvec(self, x):
        X = self.revecD(x)
        ATX = self.AT(X)
        return ATX.vec()

    def revec(self, x):
        val = []
        x_end = 0
        for m in self.X._iter:
            x_beg = x_end
            x_end = x_beg + self.X.lsize[m]
            xpart = x[x_beg:x_end]
            if self.X.ltype[m] == 'VecTri':
                comp = VecTri(val=np.reshape(xpart, self.X.lshape[m]))
            val.append(comp)
        return MultiVector(val=val)

    def revecD(self, x):
        return self.revec(x)

    def get_size(self, X):
            N = 0
            X_shape = []
            if isinstance(X, VecTri):
                N += X.size
                X_shape.append(X.shape())
            elif isinstance(X, MultiVector):
                for m in np.arange(X.lsize):
                    X_type = type(X)
                    N += X[m].size
            return N, X_type

    def matvec(self, x):
        X = self.revec(x)
        AX = self.A(X)
        return AX.vec()

    def __repr__(self):
        ss = 'Class : %s\n    name : %s\n' % (self.__class__.__name__,
                                              self.name)
        ss += '    shape = %s\n' % (str(self.shape))
        ss += '    A : %s\n' % (self.A.name)
        return ss

def enlargeF(xN, M):
    """
    It enlarges an array of grid values. First, Fourier coefficients are
    calculated and complemented by zeros. Then an inverse DFT provides
    the grid values on required grid.

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
    N = np.array(xN.shape, dtype=np.int)
    M = np.array(M, dtype=np.int)
    return np.real(DFT.ifftnc(enlarge(DFT.fftnc(xN, N), M), M))


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
    xil = Grid.get_xil(N, Y)
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


if __name__ == '__main__':
    execfile('../main_test.py')
