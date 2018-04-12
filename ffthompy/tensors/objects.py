"""
This module contains classes and functions representing tensors
of trigonometric polynomials and relating operators.
"""

import numpy as np
from ffthompy.trigpol import mean_index
from ffthompy.mechanics.matcoef import ElasticTensor
from ffthompy.trigpol import enlarge, decrease
from ffthompy.tensors.fft import fftnc, ifftnc
import itertools


class TensorFuns():

    def mean_index(self):
        return mean_index(self.N)

    def __getitem__(self, ii):
        return self.val[ii]

    def pN(self):
        return np.prod(self.N)

    def point(self, ii):
        val=np.empty(self.shape)
        for ind in np.ndindex(*self.shape):
            val[ind]=self.val[ind][ii]
        return val

    def sub(self, ii):
        self.val[ii]

    def __repr__(self, full=False, detailed=False):
        keys=['name', 'Y', 'shape', 'N', 'Fourier', 'norm']
        ss="Class : {0}({1}) \n".format(self.__class__.__name__, self.order)
        skip=4*' '
        nstr=np.array([key.__len__() for key in keys]).max()

        for key in keys:
            attr=getattr(self, key)
            if callable(attr):
                ss+='{0}{1}{3} = {2}\n'.format(skip, key, str(attr()), (nstr-key.__len__())*' ')
            else:
                ss+='{0}{1}{3} = {2}\n'.format(skip, key, str(attr), (nstr-key.__len__())*' ')

        if np.prod(np.array(self.shape))<=36 or detailed:
            ss+='{0}norm component-wise =\n{1}\n'.format(skip, str(self.norm(componentwise=True)))
            ss+='{0}mean = \n{1}\n'.format(skip, str(self.mean()))
        if full:
            ss+='{0}val = \n{1}'.format(skip, str(self.val))
        return ss

class Tensor(TensorFuns):

    def __init__(self, name='', val=None, order=None, shape=None, N=None, Y=None,
                 Fourier=False, multype='scal'):

        self.name=name
        self.Fourier=Fourier

        if isinstance(val, np.ndarray): # define: val + order
            self.val=val
            if order is not None:
                self.order=int(order)
                self.shape=self.val.shape[:order]
                self.N=self.val.shape[order:]
            else:
                raise ValueError('order is not defined!')

        elif shape is not None and N is not None: # define: shape + N
            self.N=tuple(np.array(N, dtype=np.int))
            self.shape=tuple(np.array(shape, dtype=np.int))
            self.order=len(self.shape)

            if self.Fourier:
                self.val=np.zeros(self.shape+self.N, dtype=np.complex)
            else:
                self.val=np.zeros(self.shape+self.N, dtype=np.float)
        else:
            raise ValueError('Initialization of Tensor.')

        self.dim=self.N.__len__()
        if Y is None:
            self.Y=np.ones(self.dim, dtype=np.float)
        else:
            self.Y=np.array(Y, dtype=np.float)

        # definition of __mul__ operation
        self.multype=multype

    def randomize(self):
        self.val=np.random.random(self.val.shape)
        if self.Fourier:
            self.val=self.val+1j*np.random.random(self.val.shape)
        return self

    def __neg__(self):
        return self.copy(name='-'+self.name[:10], val=-self.val)

    def __add__(self, x):
        if isinstance(x, Tensor):
            assert(self.Fourier==x.Fourier)
            assert(self.val.shape==x.val.shape)
            name='({0}+{1})'.format(self.name[:10], x.name[:10])
            return self.copy(name=name, val=self.val+x.val)

        elif isinstance(x, np.ndarray) or isinstance(x, float):
            return self.copy(val=self.val+x)
        else:
            raise ValueError('Tensor.__add__')

    def __sub__(self, x):
        return self.__add__(-x)

    def __rmul__(self, x):
        if isinstance(x, Scalar):
            return self.copy(val=x.val*self.val)
        elif np.size(x)==1 or isinstance(x, 'float'):
            return self.copy(val=x*self.val)
        else:
            raise ValueError()

    def __call__(self, *args, **kwargs):
        return self.__mul__(*args, **kwargs)

    def __mul__(self, Y, *args, **kwargs):
        multype=self.multype
        X = self
        if multype in ['scal', 'scalar']:
            return scalar_product(X, Y)
        elif multype in [21, '21']:
            return einsum('ij...,j...->i...', X, Y)
        elif multype in [42, '42']:
            return einsum('ijkl...,kl...->ij...', X, Y)
        elif multype in [00, 'elementwise', 'hadamard']:
            return einsum('...,...->...', X, Y)
        elif multype in ['grad']:
            return einsum('i...,...->i...', X, Y)
        elif multype in ['div']:
            return einsum('i...,i...->...', X, Y)
        else:
            raise ValueError()

    def norm(self, ntype='L2', componentwise=False):
        if componentwise:
            scal=np.empty(self.shape)
            for ind in np.ndindex(*self.shape):
                obj=Tensor(name='aux', val=self.val[ind], order=0, Fourier=self.Fourier)
                scal[ind]=norm_fun(obj, ntype='L2')
            return scal
        else:
            return norm_fun(self, ntype=ntype)

    def mean(self):
        """
        Mean of trigonometric polynomial of shape of macroscopic vector.
        """
        mean=np.zeros(self.shape)
        if self.Fourier:
            ind=self.mean_index()
            for di in np.ndindex(*self.shape):
                mean[di]=np.real(self.val[di][ind])
        else:
            for di in np.ndindex(*self.shape):
                mean[di]=np.mean(self.val[di])
        return mean

    def add_mean(self, mean):
        assert(self.shape==mean.shape)

        if self.Fourier:
            ind=self.mean_index()
            for di in np.ndindex(*self.shape):
                self.val[di+ind]=mean[di]
        else:
            for di in np.ndindex(*self.shape):
                self.val[di]+=mean[di]
        return self

    def set_mean(self, mean):
        assert(self.shape==mean.shape)
        self.add_mean(-self.mean()) # set mean to zero

        if self.Fourier:
            ind=self.mean_index()
            for di in np.ndindex(*self.shape):
                self.val[di+ind]=mean[di]
        else:
            for di in np.ndindex(*self.shape):
                self.val[di]+=mean[di]
        return self

    def __eq__(self, x, full=True, tol=1e-13):
        """
        Check the equality with other objects comparable to trig. polynomials.
        """
        _bool=False
        res=None
        if isinstance(x, Tensor) and self.val.squeeze().shape==x.val.squeeze().shape:
            res=np.linalg.norm(self.val.squeeze()-x.val.squeeze())
            if res<tol:
                _bool=True
        if full:
            return _bool, res
        else:
            return _bool

    def set_shape(self):
        shape_size=self.val.ndim-self.N.size
        self.shape=np.array(self.val.shape[:shape_size])
        return self.shape

    def transpose(self):
        res=self.empty_like(name=self.name[:10]+'.T')
        if self.order==2:
            val=np.einsum('ij...->ji...', self.val)
        elif self.order==4:
            val=np.einsum('ijkl...->klij...', self.val)
        else:
            raise NotImplementedError()
        return self.copy(name=self.name[:10]+'.T', val=val)

    def transpose_left(self):
        res=self.empty_like(name=self.name[:10]+'.T')
        assert(self.order==4)
        res.val=np.einsum('ijkl...->jikl...', self.val)
        return res

    def transpose_right(self):
        res=self.empty_like(name=self.name[:10]+'.T')
        assert(self.order==4)
        res.val=np.einsum('ijkl...->ijlk...', self.val)
        return res

    def identity(self):
        self.val[:] = 0.
        assert(self.order % 2 == 0)
        for ii in itertools.product(*tuple([range(n) for n in self.shape[:int(self.order/2)]])):
            self.val[ii + ii] = 1.

    def vec(self):
        """
        Returns one-dimensional vector (column) version of trigonometric
        polynomial.
        """
        return np.matrix(self.val.ravel()).transpose()

    def copy(self, **kwargs):
        data={'name': self.name, 'val': np.copy(self.val), 'order': self.order,
              'Fourier': self.Fourier, 'multype': self.multype, 'Y': self.Y}
        data.update(kwargs)
        return Tensor(**data)

    def zeros_like(self, name=None):
        if name is None:
            name='zeros({})'.format(self.name[:10])
        return self.copy(name=name, val=np.zeros_like(self.val))

    def empty_like(self, name=None):
        if name is None:
            name='empty({})'.format(self.name[:10])
        return self.copy(name=name, val=np.empty_like(self.val))

    def calc_eigs(self, sort=True, symmetric=False, mandel=False):
        if symmetric:
            eigfun=np.linalg.eigvalsh
        else:
            eigfun=np.linalg.eigvals

        if self.order==2:
            eigs=np.zeros(self.shape[-1]+self.N)
            for ind in np.ndindex(self.N):
                mat=self.val[:, :][ind]
                eigs.append(eigfun(mat))
        elif self.order==4:
            if mandel:
                matrixfun=lambda x: ElasticTensor.create_mandel(x)
                d=self.shape[2]
                eigdim=d*(d+1)/2
            else:
                matshape=(self.shape[0]*self.shape[1], self.shape[2]*self.shape[3])
                matrixfun=lambda x: np.reshape(val[ind], matshape)
                eigdim=self.shape[2]*self.shape[3]

            eigs=np.zeros(self.N+(eigdim,))
            val=np.copy(self.val)
            for ii in range(self.dim):
                val=np.rollaxis(val, self.val.ndim-self.dim+ii, ii)

            for ind in np.ndindex(*self.N):
                mat=matrixfun(val[ind])
                eigs[ind]=eigfun(mat)

        eigs=np.array(eigs)
        if sort:
            eigs=np.sort(eigs, axis=0)
        return eigs

    def enlarge(self, M):
        assert(self.Fourier)
        val=np.zeros(self.shape+tuple(M), dtype=self.val.dtype)
        for di in np.ndindex(*self.shape):
            val[di]=enlarge(self.val[di], M)
        return self.copy(val=val)

    def decrease(self, M):
        assert(self.Fourier)
        val=np.zeros(self.shape+tuple(M), dtype=self.val.dtype)
        for di in np.ndindex(*self.shape):
            val[di]=decrease(self.val[di], M)
        return self.copy(val=val)

    def fourier(self):
        if self.Fourier:
            return self.copy(val=ifftnc(self.val, self.N), Fourier=not self.Fourier)
        else:
            return self.copy(val=fftnc(self.val, self.N), Fourier=not self.Fourier)

    def project(self, M):
        """
        It enlarges a trigonometric polynomial by adding zeros to the Fourier
        coefficients with high frequencies.
        """
        if np.allclose(self.N, M):
            return self

        if self.Fourier:
            X=self
        else:
            X=self.fourier()

        if np.all(np.greater(M, self.N)):
            X=X.enlarge(M)
        elif np.all(np.less(M, self.N)):
            X=X.decrease(M)
        else:
            raise NotImplementedError()

        if self.Fourier:
            return X
        else:
            return X.fourier()

    def plot(self, ind=0, N=None, filen=None, ptype='surface'):
        dim=self.N.__len__()
        if dim!=2:
            raise ValueError("The plotting is suited only for dim=2!")
        if N is None:
            N=self.N

        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt
        from ffthompy.trigpol import Grid
        fig=plt.figure()
        coord=Grid.get_coordinates(N, self.Y)
        if np.all(np.greater(N, self.N)):
            Z=ifftnc(enlarge(fftnc(self.val[ind], self.N), N), N)
        elif np.all(np.less(N, self.N)):
            Z=ifftnc(decrease(fftnc(self.val[ind], self.N), N), N)
        elif np.allclose(N, self.N):
            Z=self.val[ind]

        if ptype in ['wireframe']:
            ax=fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(coord[0], coord[1], Z)
        elif ptype in ['surface']:
            from matplotlib import cm
            ax=fig.gca(projection='3d')
            surf=ax.plot_surface(coord[0], coord[1], Z,
                                   rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)

        if filen is None:
            plt.show()
        else:
            plt.savefig(filen)


class Scalar():
    """
    Scalar value that is used to multiply VecTri or Matrix classes
    """
    def __init__(self, val=None, name='c'):
        if val is not None:
            self.val=val
        else:
            self.val=1.
        self.name=name

    def __call__(self, x):
        return self*x

    def __repr__(self):
        ss="Class : {0}\n".format(self.__class__.__name__)
        ss+="    val = {0}".format(self.val)
        return ss

    def transpose(self):
        return self

# @staticmethod
def einsum(str_operator, x, y):
    assert(x.Fourier==y.Fourier)
    assert(np.all(x.N==y.N))
    val=np.einsum(str_operator, x.val, y.val)
    order=len(val.shape)-len(x.N)
    return y.copy(name='{0}({1})'.format(x.name, y.name), val=val, order=order)

# @staticmethod
def norm_fun(X, ntype):
    if ntype in ['L2', 2]:
        scal=(scalar_product(X, X))**0.5
    elif ntype==1:
        scal=np.sum(np.abs(X.val))
    elif ntype=='inf':
        scal=np.max(np.abs(X.val))
    else:
        msg="The norm ({}) of VecTri is not implemented!".format(ntype)
        raise NotImplementedError(msg)
    return scal

def scalar_product(y, x):
    assert(isinstance(x, Tensor))
    assert(y.val.shape==x.val.shape)
    scal=np.sum(y.val[:]*np.conj(x.val[:])).real
    if not y.Fourier:
        scal=scal/np.prod(y.N)
    return scal
