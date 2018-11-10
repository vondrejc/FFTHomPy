"""
This module contains operators working with Tensor from ffthompy.tensors.objects
"""

import itertools
import numpy as np
import numpy.matlib as npmatlib
from ffthompy.trigpol import Grid, fft_form_default
from ffthompy.tensors.objects import Tensor, TensorFuns
from ffthompy.tensors.fft import *
from copy import copy


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
    fft_form : str or num
        determines the type of the Fourier transform and corresponding format in the Fourier domain
        the following values are considered:
        0 : standard numpy.fft.fftn algorithm
        'c' : centered version of numpy.fft.fftn algorithm with zero frequency in the middle
        'r' : version of numpy.fft.fftn suitable for real data
    """
    def __init__(self, inverse=False, N=None, fft_form=fft_form_default, **kwargs):
        self.__dict__.update(kwargs)
        if 'name' not in list(kwargs.keys()):
            if inverse:
                self.name='iDFT'
            else:
                self.name='DFT'

        self.N=np.array(N, dtype=np.int32)
        self.inverse=inverse
        self.set_fft(fft_form)

    def __mul__(self, x):
        return self.__call__(x)

    def __call__(self, x):
        if isinstance(x, Tensor):
            assert(x.Fourier==self.inverse)
            if self.inverse:
                return Tensor(name='iF({0})'.format(x.name[:10]), order=x.order,
                              val=self.ifftn(x.val, self.N).real, multype=x.multype,
                              Fourier=not x.Fourier, fft_form=self.fft_form)
            else:
                assert(x.fft_form==self.fft_form)
                return Tensor(name='F({0})'.format(x.name[:10]), order=x.order,
                              val=self.fftn(x.val, self.N), multype=x.multype,
                              Fourier=not x.Fourier, fft_form=self.fft_form)

        elif (isinstance(x, Operator) or isinstance(x, DFT)):
            return Operator(mat=[[self, x]])

        else:
            raise ValueError('DFT.__call__')

    def matrix(self, shape=None):
        """
        This function returns the object as a matrix of DFT or iDFT resp.
        """
        N=self.N
        prodN=np.prod(N)
        if shape is not None:
            dim=np.prod(np.array(shape))
        elif hasattr(self, 'shape'):
            dim=np.prod(np.array(shape))
        else:
            raise ValueError('Missing shape of the DFT.')

        proddN=dim*prodN
        ZN_input=Grid.get_ZNl(N, fft_form=0)
        ZN_output=Grid.get_ZNl(N, fft_form='c')

        if self.inverse:
            DFTcoef=lambda k, l, N: np.exp(2*np.pi*1j*np.sum(k*l/N))
        else:
            DFTcoef=lambda k, l, N: np.exp(-2*np.pi*1j*np.sum(k*l/N))/np.prod(N)

        DTM=np.zeros([self.pN(), self.pN()], dtype=np.complex128)
        for ii, kk in enumerate(itertools.product(*tuple(ZN_output))):
            for jj, ll in enumerate(itertools.product(*tuple(ZN_input))):
                DTM[ii, jj]=DFTcoef(np.array(kk, dtype=np.float),
                                      np.array(ll), N)

        DTMd=npmatlib.zeros([proddN, proddN], dtype=np.complex128)
        for ii in range(dim):
            DTMd[prodN*ii:prodN*(ii+1), prodN*ii:prodN*(ii+1)]=DTM
        return DTMd

    def __repr__(self):
        keys=['name','inverse','fft_form','N']
        return self._repr(keys)

    def transpose(self):
        kwargs = copy(self.__dict__)
        kwargs.update(dict(inverse=not self.inverse))
        return DFT(**kwargs)

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
    def __init__(self, name='Operator', mat_rev=None, mat=None, operand=None):
        self.name=name
        if mat_rev is not None:
            self.mat_rev=mat_rev
        elif mat is not None:
            self.mat_rev=[]
            for summand in mat:
                no_oper=len(summand)
                summand_rev=[]
                for m in np.arange(no_oper):
                    summand_rev.append(summand[no_oper-1-m])
                self.mat_rev.append(summand_rev)
        self.no_summands=len(self.mat_rev)

        if operand is not None:
            self.define_operand(operand)

    def __call__(self, x):
        res=0.
        for summand in self.mat_rev:
            prod=x
            for matrix in summand:
                prod=matrix(prod)
            res=prod+res
        res.name='{0}({1})'.format(self.name[:6], x.name[:10])
        return res

    def __repr__(self):
        s='Class : {0}\n    name : {1}\n    expression : '.format(self.__class__.__name__,
                                                                  self.name)
        flag_sum=False
        no_sum=len(self.mat_rev)
        for isum in np.arange(no_sum):
            if flag_sum:
                s+=' + '
            no_oper=len(self.mat_rev[isum])
            flag_mul=False
            for m in np.arange(no_oper):
                matrix=self.mat_rev[isum][no_oper-1-m]
                if flag_mul:
                    s+='*'
                s+=matrix.name
                flag_mul=True
            flag_sum=True
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
            Y=self(X)
            self.matshape=(Y.val.size, X.val.size)
            self.X_reshape=X.val.shape
            self.X_order=X.order
            self.Y_reshape=Y.val.shape
            self.Y_order=Y.order
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
        X=Tensor(val=self.revec(x), order=self.X_order)
        AX=self.__call__(X)
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
        mat=[]
        for m in np.arange(self.no_summands):
            summand=[]
            for n in np.arange(len(self.mat_rev[m])):
                summand.append(self.mat_rev[m][n].transpose())
            mat.append(summand)
        name='({0}).T'.format(self.name[:10])
        return Operator(name=name, mat=mat)

def grad(X):
    if X.shape==(1,):
        shape=(X.dim,)
    else:
        shape=X.shape+(X.dim,)
    name='grad({0})'.format(X.name[:10])
    gX=Tensor(name=name, shape=shape, N=X.N,
              Fourier=True, fft_form=X.fft_form)
    if X.Fourier:
        FX=X
    else:
        F=DFT(N=X.N, fft_form=X.fft_form)
        FX=F(X)

    dim=len(X.N)
    freq=Grid.get_freq(X.N, X.Y, fft_form=X.fft_form)
    strfreq='xyz'
    coef=2*np.pi*1j
    val=np.empty((X.dim,)+X.shape+X.N, dtype=np.complex)

    for ii in range(X.dim):
        mul_str='{0},...{1}->...{1}'.format(strfreq[ii], strfreq[:dim])
        val[ii]=np.einsum(mul_str, coef*freq[ii], FX.val, dtype=np.complex)

    if X.shape==(1,):
        gX.val=np.squeeze(val)
    else:
        gX.val=np.moveaxis(val, 0, X.order)

    if not X.Fourier:
        iF=DFT(N=X.N, inverse=True, fft_form=gX.fft_form)
        gX=iF(gX)
    gX.name='grad({0})'.format(X.name[:10])
    return gX

def div(X):
    if X.shape==(1,):
        shape=()
    else:
        shape=X.shape[:-1]
    assert(X.shape[-1]==X.dim)
    assert(X.order==1)

    dX=Tensor(shape=shape, N=X.N, Fourier=True, fft_form=X.fft_form)
    if X.Fourier:
        FX=X
    else:
        F=DFT(N=X.N, fft_form=X.fft_form)
        FX=F(X)

    dim=len(X.N)
    freq=Grid.get_freq(X.N, X.Y, fft_form=FX.fft_form)
    strfreq='xyz'
    coef=2*np.pi*1j

    for ii in range(X.dim):
        mul_str='{0},...{1}->...{1}'.format(strfreq[ii], strfreq[:dim])
        dX.val+=np.einsum(mul_str, coef*freq[ii], FX.val[ii], dtype=np.complex)

    if not X.Fourier:
        iF=DFT(N=X.N, inverse=True, fft_form=dX.fft_form)
        dX=iF(dX)
    dX.name='div({0})'.format(X.name[:10])
    return dX

def laplace(X):
    return div(grad(X))

def symgrad(X):
    gX=grad(X)
    return 0.5*(gX+gX.transpose())

def potential_scalar(x, freq, mean_index):
    # get potential for scalar-valued function in Fourier space
    dim=x.shape[0]
    assert(dim==len(x.shape)-1)
    strfreq='xyz'
    coef=2*np.pi*1j
    val=np.empty(x.shape[1:], dtype=np.complex)
    for d in range(0, dim):
        factor=np.zeros_like(freq[d], dtype=np.complex)
        inds=np.setdiff1d(np.arange(factor.size, dtype=np.int), mean_index[d])
        factor[inds]=1./(coef*freq[d][inds])
        val[mean_index[:d]]=np.einsum('x,{0}->{0}'.format(strfreq[:dim-d]),
                                      factor, x[d][mean_index[:d]], dtype=np.complex)
    return val

def potential(X, small_strain=False):
    if X.Fourier:
        FX=X
    else:
        F=DFT(N=X.N, fft_form=X.fft_form)
        FX=F(X)

    freq=Grid.get_freq(X.N, X.Y, fft_form=FX.fft_form)
    if X.order==1:
        assert(X.dim==X.shape[0])
        iX=Tensor(name='potential({0})'.format(X.name[:10]), shape=(1,), N=X.N,
                  Fourier=True, fft_form=FX.fft_form)
        iX.val[0]=potential_scalar(FX.val, freq=freq, mean_index=FX.mean_index())

    elif X.order==2:
        assert(X.dim==X.shape[0])
        assert(X.dim==X.shape[1])
        iX=Tensor(name='potential({0})'.format(X.name[:10]), shape=(X.dim,), N=X.N,
                  Fourier=True, fft_form=FX.fft_form)
        if not small_strain:
            for ii in range(X.dim):
                iX.val[ii]=potential_scalar(FX.val[ii], freq=freq, mean_index=FX.mean_index())

        else:
            assert((X-X.transpose()).norm()<1e-14) # symmetricity
            omeg=FX.zeros_like() # non-symmetric part of the gradient
            gomeg=Tensor(name='potential({0})'.format(X.name[:10]),
                           shape=FX.shape+(X.dim,), N=X.N, Fourier=True)
            grad_ep=grad(FX) # gradient of strain
            gomeg.val=np.einsum('ikj...->ijk...', grad_ep.val)-np.einsum('jki...->ijk...', grad_ep.val)
            for ij in itertools.product(range(X.dim), repeat=2):
                omeg.val[ij]=potential_scalar(gomeg.val[ij], freq=freq, mean_index=FX.mean_index())

            gradu=FX+omeg
            iX=potential(gradu, small_strain=False)

    if X.Fourier:
        return iX
    else:
        iF=DFT(N=X.N, inverse=True, fft_form=FX.fft_form)
        return iF(iX)

def matrix2tensor(M):
    return Tensor(name=M.name, val=M.val, order=2, multype=21,
                  Fourier=M.Fourier, fft_form=fft_form_default)

def vector2tensor(V):
    return Tensor(name=V.name, val=V.val, order=1, Fourier=V.Fourier)

def grad_div_tensor(N, Y=None, grad=True, div=True, fft_form=fft_form_default):
    if grad and div:
        return grad_tensor(N, Y, fft_form=fft_form), div_tensor(N, Y, fft_form=fft_form)
    elif grad:
        return grad_tensor(N, Y, fft_form=fft_form)
    elif div:
        return div_tensor(N, Y, fft_form=fft_form)

def grad_tensor(N, Y=None, fft_form=fft_form_default):
    if Y is None:
        Y = np.ones_like(N)
    # scalar valued versions of gradient and divergence
    N = np.array(N, dtype=np.int)
    dim = N.size
    hGrad = np.zeros((dim,)+ tuple(N)) # zero initialize
    freq = Grid.get_xil(N, Y, fft_form=fft_form)
    for ind in itertools.product(*[range(n) for n in N]):
        for i in range(dim):
            hGrad[i][ind] = freq[i][ind[i]]
    hGrad = hGrad*2*np.pi*1j
    return Tensor(name='hgrad', val=hGrad, order=1, multype='grad',
                  Fourier=True, fft_form=fft_form)

def div_tensor(N, Y=None, fft_form=fft_form_default):
    if Y is None:
        Y = np.ones_like(N)
    hGrad=grad_tensor(N, Y=Y, fft_form=fft_form)
    hGrad.multype='div'
    return hGrad
