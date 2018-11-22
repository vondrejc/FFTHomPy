import numpy as np
import numpy.fft as fft

def cfftnc(x, N):
    """
    centered n-dimensional FFT algorithm
    """
    ax=tuple(np.setdiff1d(range(x.ndim), range(x.ndim-N.__len__()), assume_unique=True))
    return 1./np.prod(N)*fft.fftshift(fft.fftn(fft.ifftshift(x, ax), N), ax)

def icfftnc(Fx, N):
    """
    centered n-dimensional inverse FFT algorithm
    """
    ax=tuple(np.setdiff1d(range(Fx.ndim), range(Fx.ndim-N.__len__()), assume_unique=True))
    return fft.fftshift(fft.ifftn(fft.ifftshift(Fx, ax), N), ax).real*np.prod(N)

def fftnc(x, N):
    """
    centered n-dimensional FFT algorithm
    """
    ax=tuple(np.setdiff1d(range(x.ndim), range(x.ndim-N.__len__()), assume_unique=True))
    return 1./np.prod(N)*fft.fftshift(fft.fftn(x, N), ax)

def icfftn(Fx, N):
    """
    centered n-dimensional inverse FFT algorithm
    """
    ax=tuple(np.setdiff1d(range(Fx.ndim), range(Fx.ndim-N.__len__()), assume_unique=True))
    return fft.ifftn(fft.ifftshift(Fx, ax), N).real*np.prod(N)


def fftn(x, N):
    return 1./np.prod(N)*fft.fftn(x, N) # numpy.fft.fftn

def ifftn(x, N):
    return fft.ifftn(x, N).real*np.prod(N) # numpy.fft.fftn

def rfftn(x, N):
    return fft.rfftn(x, N) # real version of numpy.fft.fftn

def irfftn(x, N):
    return fft.irfftn(x, N) # real version of numpy.fft.fftn
