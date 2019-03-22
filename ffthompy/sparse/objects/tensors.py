import numpy as np
from ffthompy.tensors.objects import TensorFuns
import itertools
from copy import deepcopy as copy
from ffthompy.sparse.fft1 import fft, ifft, fftc, icfft, cfftc, icfftc, srfft,sirfft

fft_form_default='sr' # scipy rfft

def multiply(A, B, *args, **kwargs):
    """element-wise (Hadamard) product of A and B"""
    dim=A.__len__()
    assert(dim==B.__len__())
    C=[]
    for ii in range(dim):
        shape=(A[ii].shape[0]*B[ii].shape[0], A[ii].shape[1])
        val=np.empty(shape)
        for iimn, (mm, nn) in enumerate(itertools.product(list(range(A[ii].shape[0])), list(range(B[ii].shape[0])))):
            val[iimn] = A[ii][mm]*B[ii][nn]
        C.append(val)
    return C


class SparseTensorFuns(TensorFuns):

    def mean_index(self):
        if self.fft_form in [0, 'sr']:
            return tuple(np.zeros_like(self.N, dtype=np.int))
        elif self.fft_form in ['c', 'cc']:
            return tuple(np.array(np.fix(np.array(self.N)/2), dtype=np.int))

    def _set_fft(self, fft_form):
        assert(fft_form in ['cc', 'c', 'sr', 0]) # 'sr' for scipy.fftpack.rfft
        if fft_form in [0]:
            self.N_fft=self.N
            self.fft=fft
            self.ifft=ifft
        elif fft_form in ['c']:
            self.N_fft=self.N
            self.fft=fftc
            self.ifft=icfft
        elif fft_form in ['cc']:
            self.N_fft=self.N
            self.fft=cfftc
            self.ifft=icfftc
        elif fft_form in ['sr']:
            self.N_fft=self.N
            self.fft=srfft
            self.ifft=sirfft
        self.fft_form=fft_form
        return self

    def fourier(self, real_output=False, copy=True):
        "(inverse) discrete Fourier transform"

        if self.Fourier:
            fftfun=lambda Fx, N, real_output: self.ifft(Fx, N, real_output)
            name='Fi({})'.format(self.name)
        else:
            fftfun=lambda x, N, real_output: self.fft(x, N)
            name='F({})'.format(self.name)

        basis=[]
        for ii in range(self.order):
            basis.append(fftfun(self.basis[ii], self.N[ii], real_output))

        if copy:
            return self.copy(name=name, basis=basis, Fourier=not self.Fourier, orthogonal=False)
        else:
            self.basis=basis
            self.Fourier=not self.Fourier
            self.orthogonal=False
            return self


if __name__=='__main__':
    # check multiplication
    r=2
    n=50
    m=55
    As = [np.random.random([r,n]), np.random.random([r, m])]
    A = np.einsum('ij,ik->jk', As[0], As[1])
    Bs = [np.random.random([r,n]), np.random.random([r, m])]
    B = np.einsum('ij,ik->jk', Bs[0], Bs[1])
    C0 = A*B
    C1s = multiply(As, Bs)
    C1 = np.einsum('ij,ik->jk', C1s[0], C1s[1])
    print((np.linalg.norm(C0-C1)))
    print('END')
