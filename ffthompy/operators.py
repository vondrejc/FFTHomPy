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
                return Tensor(name=get_name('F', '*', x.name),
                              val=self.fftnc(x.val, self.N),
                              order=x.order, Fourier=not x.Fourier)
            else:
                return Tensor(name=get_name('Fi', '*', x.name),
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
    def __init__(self, name='LinOper', dtype=None, X=None, **kwargs):
        self.name = name
        if 'mat_rev' in list(kwargs.keys()):
            self.mat_rev = kwargs['mat_rev']
        elif 'mat' in list(kwargs.keys()):
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
