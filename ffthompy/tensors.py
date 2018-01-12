"""
This module contains classes and functions representing tensors
of trigonometric polynomials and relating operators.
"""

import numpy as np
from ffthompy.matvec import Scalar, get_name, mean_index
from ffthompy.mechanics.matcoef import ElasticTensor


class TensorFuns():

    def mean_index(self):
        return mean_index(self.N)

    def __getitem__(self, ii):
        return self.val[ii]

    def pN(self):
        return np.prod(self.N)

    def point(self, ii):
        val = np.empty(self.shape)
        for ind in np.ndindex(*self.shape):
            val[ind] = self.val[ind][ii]
        return val

    def sub(self, ii):
        self.val[ii]

    def __repr__(self, full=False, detailed=False):
        keys = ['name', 'Y', 'shape', 'N', 'Fourier', 'norm']
        ss = "Class : {0}({1}) \n".format(self.__class__.__name__, self.order)
        skip = 4*' '
        nstr = np.array([key.__len__() for key in keys]).max()

        for key in keys:
            attr = getattr(self, key)
            if callable(attr):
                ss += '{0}{1}{3} = {2}\n'.format(skip, key, str(attr()), (nstr-key.__len__())*' ')
            else:
                ss += '{0}{1}{3} = {2}\n'.format(skip, key, str(attr), (nstr-key.__len__())*' ')

        if np.prod(np.array(self.shape)) < 20 or detailed:
            ss += '{0}norm component-wise =\n{1}\n'.format(skip, str(self.norm(componentwise=True)))
            ss += '{0}mean = \n{1}\n'.format(skip, str(self.mean()))
        if full:
            ss += '{0}val = \n{1}'.format(skip, str(self.val))
        return ss

class Tensor(TensorFuns):

    def __init__(self, name='', val=None, order=None, shape=None, N=None, Y=None,
                 Fourier=False, multype='scal'):

        self.name = name
        self.Fourier = Fourier

        if isinstance(val, np.ndarray): # define: val + order
            self.val = val
            if order is not None:
                self.order = int(order)
                self.shape = self.val.shape[:order]
                self.N = self.val.shape[order:]
            else:
                raise ValueError('order is not defined!')

        elif shape is not None and N is not None:  # define: shape + N
            self.N = tuple(np.array(N, dtype=np.int))
            self.shape = tuple(np.array(shape, dtype=np.int))
            self.order = len(self.shape)

            if self.Fourier:
                self.val = np.zeros(self.shape + self.N, dtype=np.complex)
            else:
                self.val = np.zeros(self.shape + self.N, dtype=np.float)
        else:
            raise ValueError('Initialization of Tensor.')

        self.dim = self.N.__len__()
        if Y is None:
            self.Y = np.ones(self.dim, dtype=np.float)
        else:
            self.Y = np.array(Y, dtype=np.float)

        # definition of __mul__ operation
        self.multype = multype
        if multype in ['scal', 'scalar']:
            self.mul_method = self.scalar_product
        elif multype in [21, '21']:
            self.mul_str = 'ij...,j...->i...'
            self.mul_method = self._mul
            self.call_method = self._mul
        elif multype in [42, '42']:
            self.mul_str = 'ijkl...,kl...->ij...'
            self.mul_method = self._mul
            self.call_method = self._mul
        else:
            self.mul_str = multype
            self.mul_method = self._mul
            self.call_method = self._mul

    def randomize(self):
        self.val = np.random.random(self.val.shape)
        if self.Fourier:
            self.val += 1j*np.random.random(self.val.shape)
        return self

    def __neg__(self):
        res = self.copy(name='-'+self.name)
        res.val = -res.val
        return res

    def __add__(self, x):
        if isinstance(x, Tensor):
            assert(self.Fourier == x.Fourier)
            assert(self.val.shape==x.val.shape)
            name = get_name(self.name, '+', x.name)
            return Tensor(name=name, val=self.val+x.val, Fourier=self.Fourier,
                          order=self.order, multype=self.multype)
        elif isinstance(x, np.ndarray) or isinstance(x, float):
            self.val += x
            return self
        else:
            raise ValueError('Tensor.__add__')

    def __sub__(self, x):
        return self.__add__(-x)

    def scalar_product(self, x):
        assert isinstance(x, Tensor)
        assert self.val.shape == x.val.shape
        scal = np.sum(self.val[:]*np.conj(x.val[:]))
        if not self.Fourier:
            scal = scal / np.prod(self.N)
        return scal

    def _mul(self, x):
        return self.einsum(self.mul_str, self, x)

    def __rmul__(self, x):
        if isinstance(x, Scalar):
            R = self.copy(name='c*'+'')
            R.val = x.val*self.val
        elif np.size(x) == 1 or isinstance(x, 'float'):
            R = self.copy(name='c*'+'')
            R.val = x*self.val
        else:
            raise ValueError()
        return R

    def __call__(self, *args, **kwargs):
        return self.call_method(*args, **kwargs)

    def __mul__(self,  *args, **kwargs):
        return self.mul_method(*args, **kwargs)

    @staticmethod
    def einsum(str_operator, x, y):
        assert(x.Fourier==y.Fourier)
        assert(x.N==y.N)
        val = np.einsum(str_operator, x.val, y.val)
        order = len(val.shape)-len(x.N)
        return Tensor(name='{0}({1})'.format(x.name, y.name),
                      val=val, order=order, Fourier=x.Fourier)

    @staticmethod
    def norm_fun(obj, ntype):
        if ntype in ['L2', 2]:
            scal = (obj.scalar_product(obj))**0.5
        elif ntype == 1:
            scal = np.sum(np.abs(obj.val))
        elif ntype == 'inf':
            scal = np.max(np.abs(obj.val))
        else:
            msg = "The norm ({}) of VecTri is not implemented!".format(ntype)
            raise NotImplementedError(msg)
        return scal

    def norm(self, ntype='L2', componentwise=False):
        if componentwise:
            scal = np.empty(self.shape)
            for ind in np.ndindex(*self.shape):
                obj = Tensor(name='aux', val=self.val[ind], order=0, Fourier=self.Fourier)
                scal[ind] = self.norm_fun(obj, ntype='L2')
            return scal
        else:
            return self.norm_fun(self, ntype=ntype)

    def mean(self):
        """
        Mean of trigonometric polynomial of shape of macroscopic vector.
        """
        mean = np.zeros(self.shape)
        if self.Fourier:
            ind = self.mean_index()
            for di in np.ndindex(*self.shape):
                mean[di] = np.real(self.val[di][ind])
        else:
            for di in np.ndindex(*self.shape):
                mean[di] = np.mean(self.val[di])
        return mean

    def add_mean(self, mean):
        assert(self.shape==mean.shape)

        if self.Fourier:
            ind = self.mean_index()
            for di in np.ndindex(*self.shape):
                self.val[di+ind] = mean[di]
        else:
            for di in np.ndindex(*self.shape):
                self.val[di] += mean[di]
        return self

    def set_mean(self, mean):
        assert(self.shape==mean.shape)
        self.add_mean(-self.mean()) # set mean to zero

        if self.Fourier:
            ind = self.mean_index()
            for di in np.ndindex(*self.shape):
                self.val[di+ind] = mean[di]
        else:
            for di in np.ndindex(*self.shape):
                self.val[di] += mean[di]
        return self

    def __eq__(self, x, full=True, tol=1e-13):
        """
        Check the equality with other objects comparable to trig. polynomials.
        """
        _bool = False
        res = None
        if isinstance(x, Tensor) and self.val.squeeze().shape==x.val.squeeze().shape:
            res = np.linalg.norm(self.val.squeeze()-x.val.squeeze())
            if res<tol:
                _bool = True
        if full:
            return _bool, res
        else:
            return _bool

    def set_shape(self):
        shape_size = self.val.ndim-self.N.size
        self.shape = np.array(self.val.shape[:shape_size])
        return self.shape

    def transpose(self):
        res = self.empty_like(name=self.name+'.T')
        if self.order == 2:
            res.val = np.einsum('ij...->ji...', self.val)
        elif self.order == 4:
            res.val = np.einsum('ijkl...->klij...', self.val)
        else:
            raise NotImplementedError()
        return res

    def transpose_left(self):
        res = self.empty_like(name=self.name+'.T')
        assert(self.order==4)
        res.val = np.einsum('ijkl...->jikl...', self.val)
        return res

    def transpose_right(self):
        res = self.empty_like(name=self.name+'.T')
        assert(self.order==4)
        res.val = np.einsum('ijkl...->ijlk...', self.val)
        return res

    def vec(self):
        """
        Returns one-dimensional vector (column) version of trigonometric
        polynomial.
        """
        return np.matrix(self.val.ravel()).transpose()

    def copy(self, name=None):
        if name is None:
            name = 'copy of' + self.name
        return Tensor(name=name, val=np.copy(self.val), order=self.order,
                      Fourier=self.Fourier)

    def zeros_like(self, name=None):
        if name is None:
            name = 'zeros like ' + self.name
        return Tensor(name=name, val=np.zeros_like(self.val), order=self.order,
                      Fourier=self.Fourier)

    def empty_like(self, name=None):
        if name is None:
            name = 'empty like ' + self.name
        return Tensor(name=name, val=np.empty_like(self.val), order=self.order,
                      Fourier=self.Fourier, multype=self.multype)

    def calc_eigs(self, sort=True, symmetric=False, mandel=False):
        if symmetric:
            eigfun = np.linalg.eigvalsh
        else:
            eigfun = np.linalg.eigvals

        if self.order == 2:
            eigs = np.zeros(self.shape[-1]+self.N)
            for ind in np.ndindex(self.N):
                mat = self.val[:,:][ind]
                eigs.append(eigfun(mat))
        elif self.order == 4:
            if mandel:
                matrixfun = lambda x: ElasticTensor.create_mandel(x)
                d = self.shape[2]
                eigdim = d*(d+1)/2
            else:
                matshape = (self.shape[0]*self.shape[1], self.shape[2]*self.shape[3])
                matrixfun = lambda x: np.reshape(val[ind], matshape)
                eigdim = self.shape[2]*self.shape[3]

            eigs = np.zeros(self.N + (eigdim,))
            val = np.copy(self.val)
            for ii in range(self.dim):
                val = np.rollaxis(val, self.val.ndim-self.dim+ii, ii)

            for ind in np.ndindex(*self.N):
                mat = matrixfun(val[ind])
                eigs[ind] = eigfun(mat)

        eigs = np.array(eigs)
        if sort:
            eigs = np.sort(eigs, axis=0)
        return eigs
