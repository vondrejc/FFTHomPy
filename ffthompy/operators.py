"""
This module contains operators working with Tensor from ffthompy.tensors
"""

import numpy as np
import numpy.matlib as npmatlib
from ffthompy.tensors import Tensor, TensorFuns, get_name
from ffthompy.matvec_fun import Grid
import itertools


class DFT(TensorFuns):
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
    def __init__(self, inverse=False, N=None, normalized=True, centered=True,
                 **kwargs):
        self.__dict__.update(kwargs)
        if 'name' not in list(kwargs.keys()):
            if inverse:
                self.name = 'iDFT'
            else:
                self.name = 'DFT'

        self.N = np.array(N, dtype=np.int32)
        self.inverse = inverse

    def __mul__(self, x):
        return self.__call__(x)

    def __call__(self, x):
        if isinstance(x, Tensor):
            if not self.inverse:
                return Tensor(name='F({0})'.format(x.name),
                              val=self.fftnc(x.val, self.N),
                              order=x.order, Fourier=not x.Fourier)
            else:
                return Tensor(name='iF({0})'.format(x.name),
                              val=np.real(self.ifftnc(x.val, self.N)),
                              order=x.order, Fourier=not x.Fourier)

        elif (isinstance(x, Operator) or isinstance(x, DFT)):
            return Operator(mat=[[self, x]])

        else:
            raise ValueError('DFT.__call__')

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

class Operator():
    """
    Linear operator composed of matrices or linear operators
    it is designed to provide __call__ function as a linear operation

    parameters :
        X : numpy.ndarray or VecTri or something else
            it represents the operand,
            it provides the information about size and shape of operand
        dtype : data type of operand, usually numpy.float64
    """
    def __init__(self, name='LinOper', mat_rev=None, mat=None, X=None):
        self.name = name
        if mat_rev is not None:
            self.mat_rev = mat_rev
        elif mat is not None:
            self.mat_rev = []
            for summand in mat:
                no_oper = len(summand)
                summand_rev = []
                for m in np.arange(no_oper):
                    summand_rev.append(summand[no_oper-1-m])
                self.mat_rev.append(summand_rev)
        self.no_summands = len(self.mat_rev)

        if X is not None:
            self.define_operand(X)

    def __call__(self, x):
        res = 0.
        for summand in self.mat_rev:
            prod = x
            for matrix in summand:
                prod = matrix(prod)
            res = prod + res
        res.name='%s(%s)' % (self.name, x.name)
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
        if isinstance(X, Tensor):
            Y = self(X)
            self.matshape = (Y.val.size, X.val.size)
            self.X_reshape = X.val.shape
            self.X_order=X.order
            self.Y_reshape = Y.val.shape
            self.Y_order = Y.order
        else:
            print('LinOper : This operand is not implemented!')

    def matvec(self, x):
        """
        Provides the __call__ for operand recast into one-dimensional vector.
        This is suitable for e.g. iterative solvers when trigonometric
        polynomials are recast into one-dimensional numpy.arrays.

        Parameters
        ----------
        x : one-dimensional numpy.array
        """
        X = Tensor(val=self.revec(x), order=self.X_order)
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
        in class Tensor.
        """
        return np.reshape(np.asarray(x), self.Y_reshape)

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
        return Operator(name=name, mat=mat)

def grad(X):
    if X.shape == (1,):
        shape = (X.dim,)
    else:
        shape = X.shape+(X.dim,)
    name='grad({0})'.format(X.name)
    gX = Tensor(shape=shape, N=X.N, Fourier=True)
    if X.Fourier:
        FX = X
    else:
        F = DFT(N=X.N)
        FX = F(X)

    dim = len(X.N)
    freq = Grid.get_freq(X.N, X.Y)
    strfreq = 'xyz'
    coef = -2*np.pi*1j
    val = np.empty((X.dim,)+X.shape+X.N, dtype=np.complex)

    for ii in range(X.dim):
        mul_str = '{0},...{1}->...{1}'.format(strfreq[ii], strfreq[:dim])
        val[ii] = np.einsum(mul_str, coef*freq[ii], FX.val, dtype=np.complex)

    if X.shape == (1,):
        gX.val = np.squeeze(val)
    else:
        gX.val = np.moveaxis(val, 0, X.order)

    if not X.Fourier:
        iF = DFT(N=X.N, inverse=True)
        gX = iF(gX)
    gX.name='grad({0})'.format(X.name)
    return gX

def symgrad(X):
    gX = grad(X)
    return 0.5 * (gX + gX.transpose())

def potential_scalar(x, freq, mean_ind):
    # get potential for scalar-valued function in Fourier space
    dim = x.shape[0]
    assert(dim==len(x.shape)-1)
    strfreq='xyz'
    coef = -2*np.pi*1j
    val = np.empty(x.shape[1:], dtype=np.complex)
    for d in range(0,dim):
        factor = np.zeros_like(freq[d], dtype=np.complex)
        inds = np.setdiff1d(np.arange(factor.size, dtype=np.int), mean_ind[d])
        factor[inds] = 1./(coef*freq[d][inds])
        val[mean_ind[:d]] = np.einsum('x,{0}->{0}'.format(strfreq[:dim-d]),
                                      factor, x[d][mean_ind[:d]], dtype=np.complex)
    return val

def potential(X, small_strain=False):
    if X.Fourier:
        FX = X
    else:
        F = DFT(N=X.N)
        FX = F(X)

    freq = Grid.get_freq(X.N, X.Y)
    if X.order == 1:
        assert(X.dim == X.shape[0])
        iX = Tensor(name='potential({0})'.format(X.name), shape=(1,), N=X.N, Fourier=True)
        iX.val[0] = potential_scalar(FX.val, freq=freq, mean_ind=FX.mean_ind())

    elif X.order == 2:
        assert(X.dim == X.shape[0])
        assert(X.dim == X.shape[1])
        iX = Tensor(name='potential({0})'.format(X.name),
                    shape=(X.dim,), N=X.N, Fourier=True)
        if not small_strain:
            for ii in range(X.dim):
                iX.val[ii] = potential_scalar(FX.val[ii], freq=freq, mean_ind=FX.mean_ind())

        else:
            assert((X-X.transpose()).norm()<1e-14) # symmetricity
            omeg = FX.zeros_like() # non-symmetric part of the gradient
            gomeg = Tensor(name='potential({0})'.format(X.name),
                           shape=FX.shape+(X.dim,), N=X.N, Fourier=True)
            grad_ep = grad(FX) # gradient of strain
            gomeg.val = np.einsum('ikj...->ijk...',grad_ep.val) - np.einsum('jki...->ijk...',grad_ep.val)
            for ij in itertools.product(range(X.dim), repeat=2):
                omeg.val[ij] = potential_scalar(gomeg.val[ij], freq=freq, mean_ind=FX.mean_ind())

            gradu = FX + omeg
            iX = potential(gradu, small_strain=False)

    if X.Fourier:
        return iX
    else:
        iF = DFT(N=X.N, inverse=True)
        return iF(iX)

def matrix2tensor(M):
    return Tensor(name=M.name, val=M.val, order=2, Fourier=M.Fourier, multype=21)

def vector2tensor(V):
    return Tensor(name=V.name, val=V.val, order=1, Fourier=V.Fourier)
