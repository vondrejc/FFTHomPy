"""
This module contains classes and functions representing tensors
of trigonometric polynomials and relating operators.
"""
from __future__ import print_function, division
import numpy as np
from ffthompy.general.base import Representation
from ffthompy.trigpol import mean_index, fft_form_default
from ffthompy.mechanics.matcoef import ElasticTensor
from ffthompy.trigpol import enlarge, decrease, get_inverse
from ffthompy.tensors.fft import fftn, ifftn, fftnc, icfftn, rfftn, irfftn, cfftnc, icfftnc
import itertools
from copy import copy


class TensorFuns(Representation):

    def mean_index(self):
        return mean_index(self.N, self.fft_form)

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

    def _set_fft(self, fft_form):
        assert(fft_form in ['c', 'r', 0])

        if fft_form in ['r']:
            self.N_fft=self.get_N_real(self.N)
            self.fftn=rfftn
            self.ifftn=irfftn
            self.fft_coef=np.prod(self.N)
        elif fft_form in [0]:
            self.N_fft=self.N
            self.fftn=fftn
            self.ifftn=ifftn
            self.fft_coef=1.
        elif fft_form in ['c']:
            self.N_fft=self.N
            self.fftn=fftnc
            self.ifftn=icfftn
            self.fft_coef=1.

        self.fft_form=fft_form

    def __repr__(self, full=False, detailed=False):
        keys=['order', 'name', 'Y', 'shape', 'N', 'Fourier', 'fft_form', 'origin', 'norm']
        ss=self._repr(keys)
        skip=4*' '
        if np.prod(np.array(self.shape))<=36 or detailed:
            ss+='{0}norm component-wise =\n{1}\n'.format(skip, str(self.norm(componentwise=True)))
            ss+='{0}mean = \n{1}\n'.format(skip, str(self.mean()))
        if full:
            ss+='{0}val = \n{1}'.format(skip, str(self.val))
        return ss

    def get_N_real(self, N):
        N_rfft=np.copy(N)
        N_rfft[-1]=int(np.fix(N[-1]/2)+1)
        return tuple(N_rfft)

    def get_N(self, N_rfft):
        N=np.copy(N_rfft)
        N[-1]=N_rfft[-1]*2-1
        return tuple(N)


class Tensor(TensorFuns):

    def __init__(self, name='', val=None, order=None, shape=None, N=None, Y=None,
                 multype='scal', Fourier=False, fft_form=fft_form_default, origin=0):

        self.name=name
        self.Fourier=Fourier
        self.origin=origin

        if isinstance(val, np.ndarray): # define: val + order
            self.val=val
            self.order=int(order)
            self.shape=self.val.shape[:order]

            if fft_form in ['r'] and Fourier:
                self.N=self.get_N(self.val.shape[order:])
            else:
                self.N=self.val.shape[order:]

            self._set_fft(fft_form)

        elif shape is not None and N is not None: # define: shape + N
            self.N=tuple(np.array(N, dtype=np.int))
            self._set_fft(fft_form)
            self.shape=tuple(np.array(shape, dtype=np.int))
            self.order=len(self.shape)

            if not self.Fourier:
                self.val=np.zeros(self.shape+self.N, dtype=np.float)
            else:
                self.val=np.zeros(self.shape+self.N_fft, dtype=np.complex)

        else:
            raise ValueError('Initialization of Tensor.')

        self.dim=self.N.__len__()
        if Y is None:
            self.Y=np.ones(self.dim, dtype=np.float)
        else:
            self.Y=np.array(Y, dtype=np.float)

        # definition of __mul__ operation
        self.multype=multype

    def set_fft_form(self, fft_form, copy=False):
        if copy:
            R=self.copy()
        else:
            R=self

        if self.fft_form==fft_form:
            return R

        fft_form_orig = self.fft_form
        if R.Fourier:
            if fft_form_orig in ['r']:
                R.fourier()
                R.set_fft_form(fft_form)
                R.fourier()
            elif fft_form_orig in ['c']:
                R.val=np.fft.ifftshift(R.val, axes=R.axes) # common for fft_form in [0,'r']
                if fft_form in ['r']:
                    R.val=R.val[...,:self.get_N_real(self.N)[-1]]*np.prod(self.N)
            elif fft_form_orig in [0]:
                if fft_form in ['c']:
                    R.val=np.fft.fftshift(R.val, axes=R.axes)
                else: # if fft_form in ['r']:
                    R.val=R.val[...,:self.get_N_real(self.N)[-1]]*np.prod(self.N)
        R._set_fft(fft_form)
        return R

    def shift(self, origin=None):
        """
        shift the values to the format (format)
        """
        assert(not self.Fourier)

        if origin==self.origin:
            return self
        elif origin is None:
            if self.origin in [0]:
                self.val = np.fft.fftshift(self.val, self.axes)
                self.origin='c'
            elif self.origin in ['c']:
                self.val = np.fft.ifftshift(self.val, self.axes)
                self.origin=0
            return self
        else:
            raise ValueError()

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
        X=self
        assert(X.Fourier==Y.Fourier)
        assert(X.fft_form==Y.fft_form)
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

    def inv(self):
        assert(self.Fourier is False)
        assert(self.order==2)
        assert(self.shape[0]==self.shape[1])
        return self.copy(name='inv({})'.format(self.name), val=get_inverse(self.val))

    def norm(self, ntype='L2', componentwise=False):
        if componentwise:
            scal=np.empty(self.shape)
            for ind in np.ndindex(*self.shape):
                obj=self.copy(name='aux', val=self.val[ind], order=0)
                scal[ind]=norm_fun(obj, ntype=ntype)
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
                mean[di]=np.real(self.val[di][ind])/self.fft_coef
        else:
            for di in np.ndindex(*self.shape):
                mean[di]=np.mean(self.val[di])
        return mean

    def add_mean(self, mean):
        assert(self.shape==mean.shape)

        if self.Fourier:
            ind=self.mean_index()
            for di in np.ndindex(*self.shape):
                self.val[di+ind]=mean[di]*self.fft_coef
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
                self.val[di+ind]=mean[di]*self.fft_coef
        else:
            for di in np.ndindex(*self.shape):
                self.val[di]+=mean[di]
        return self

    def __eq__(self, Y, full=True, tol=1e-13):
        """
        Check the equality with other objects comparable to trig. polynomials.
        """
        X=self
        _bool=False
        res=np.inf
        if (isinstance(X, Tensor) and X.fft_form==Y.fft_form and
                X.val.squeeze().shape==Y.val.squeeze().shape and X.Fourier==Y.Fourier):
            res=np.linalg.norm(X.val.squeeze()-Y.val.squeeze())
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
        self.val[:]=0.
        assert(self.order % 2 == 0)
        for ii in itertools.product(*tuple([range(n) for n in self.shape[:int(self.order/2)]])):
            self.val[ii+ii]=1.

    def vec(self):
        """
        Returns one-dimensional vector (column) version of trigonometric
        polynomial.
        """
        return np.matrix(self.val.ravel()).transpose()

    def copy(self, **kwargs):
        keys=('name','val','order','Y','multype','Fourier','fft_form','origin')
        data={k:copy(self.__dict__[k]) for k in keys}
        data.update(kwargs)
        return Tensor(**data)

    def update(self, **kwargs):
        return self.__dict__.update(**kwargs)

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

    @property
    def axes(self): # axes for Fourier transform
        return tuple(range(self.order, self.order+self.dim))

    def fourier(self, copy=False):
        assert(self.origin==0)
        if copy:
            if self.Fourier:
                return self.copy(val=self.ifftn(self.val, self.N), Fourier=not self.Fourier)
            else:
                return self.copy(val=self.fftn(self.val, self.N), Fourier=not self.Fourier)
        else:
            if self.Fourier:
                self.val=self.ifftn(self.val, self.N)
            else:
                self.val=self.fftn(self.val, self.N)
            self.Fourier=not self.Fourier
            return self

    def enlarge(self, M):
        """
        It enlarges a trigonometric polynomial by adding zeros to the Fourier
        coefficients with high frequencies.
        """
        assert(self.Fourier)
        if np.allclose(self.N, M):
            return self
        else:
            fft_form=self.fft_form
            self.set_fft_form(fft_form='c')

            val=np.zeros(self.shape+tuple(M), dtype=self.val.dtype)
            for di in np.ndindex(*self.shape):
                val[di]=enlarge(self.val[di], M)

            R=self.copy(val=val, fft_form='c')
            return R.set_fft_form(fft_form=fft_form)

    def decrease(self, M):
        """
        It enlarges a trigonometric polynomial by adding zeros to the Fourier
        coefficients with high frequencies.
        """
        assert(self.Fourier)
        if np.allclose(self.N, M):
            return self
        else:
            fft_form=self.fft_form
            self.set_fft_form(fft_form='c')

            val=np.zeros(self.shape+tuple(M), dtype=self.val.dtype)
            for di in np.ndindex(*self.shape):
                val[di]=enlarge(self.val[di], M)

            R=self.copy(val=val, fft_form='c')
            return R.set_fft_form(fft_form=fft_form)

    def project(self, M):
        """
        It projects a trigonometric polynomial to a polynomial with different grid.
        """

        if np.allclose(self.N, M):
            return self

        Fourier=self.Fourier
        if Fourier:
            Y=self.copy()
        else:
            Y=self.fourier(copy=copy)

        if np.all(np.greater(M, self.N)):
            Y=Y.enlarge(M)
        elif np.all(np.less(M, self.N)):
            Y=Y.decrease(M)
        else:
            raise NotImplementedError()

        if not Fourier:
            Y=Y.fourier()
        return Y

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
        Z = self.project(N)

        if ptype in ['wireframe']:
            ax=fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(coord[0], coord[1], Z)
        elif ptype in ['surface']:
            from matplotlib import cm
            ax=fig.gca(projection='3d')
            surf=ax.plot_surface(coord[0], coord[1], Z.val[ind],
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
    assert(y.fft_form==x.fft_form)

    if y.Fourier:
        if x.fft_form in ['r']:
            if x.N[-1] % 2 == 1:
                scal=(np.sum(y.val[...,0]*np.conj(x.val[...,0])).real +
                      2*np.sum(y.val[...,1:]*np.conj(x.val[...,1:])).real)/np.prod(y.N)**2
            else:
                scal=(np.sum(y.val[...,0]*np.conj(x.val[...,0])).real +
                      np.sum(y.val[...,-1]*np.conj(x.val[...,-1])).real +
                      2*np.sum(y.val[...,1:-1]*np.conj(x.val[...,1:-1])).real)/np.prod(y.N)**2
        else:
            scal=np.sum(y.val[:]*np.conj(x.val[:])).real
    else:
        scal=np.sum(y.val[:]*x.val[:])/np.prod(y.N)
    return scal

def get_N_fft_form_real(N):
    Nreal=np.copy(N)
    Nreal[-1]=Nreal[-1]/2+1
    return Nreal

if __name__=='__main__':
    N=np.array([5,5], dtype=np.int)
    M=2*N
    u=Tensor(name='test', shape=(2,), N=N, Fourier=False, fft_form='r')
    u.randomize()
    print(u)
    Fu=u.fourier(copy=True)
    print(Fu)
    print(Fu.norm())
    print(Fu.norm(componentwise=True))
    print(Fu.mean())

    print('end')
