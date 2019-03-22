import numpy as np
import numpy.fft as fft

def cfftnc(x, N):
    """
    real and Fourier centered n-dimensional FFT algorithm
    """
    ax=tuple(np.setdiff1d(list(range(x.ndim)), list(range(x.ndim-N.__len__())), assume_unique=True))
    return 1./np.prod(N)*fft.fftshift(fft.fftn(fft.ifftshift(x, ax), N), ax)

def icfftnc(Fx, N):
    """
    real and Fourier centered n-dimensional inverse FFT algorithm
    """
    ax=tuple(np.setdiff1d(list(range(Fx.ndim)), list(range(Fx.ndim-N.__len__())), assume_unique=True))
    return fft.fftshift(fft.ifftn(fft.ifftshift(Fx, ax), N), ax).real*np.prod(N)

def fftnc(x, N):
    """
    Fourier centered n-dimensional FFT algorithm
    """
    ax=tuple(np.setdiff1d(list(range(x.ndim)), list(range(x.ndim-N.__len__())), assume_unique=True))
    return 1./np.prod(N)*fft.fftshift(fft.fftn(x, N), ax)

def icfftn(Fx, N):
    """
    Fourier centered n-dimensional inverse FFT algorithm
    """
    ax=tuple(np.setdiff1d(list(range(Fx.ndim)), list(range(Fx.ndim-N.__len__())), assume_unique=True))
    return fft.ifftn(fft.ifftshift(Fx, ax), N).real*np.prod(N)


def fftn(x, N): # normalised FFT
    return 1./np.prod(N)*fft.fftn(x, N)

def ifftn(x, N): # normalised FFT
    return fft.ifftn(x, N).real*np.prod(N)

def rfftn(x, N): # real-valued FFT
    return fft.rfftn(x, N)

def irfftn(x, N): # real-valued FFT
    return fft.irfftn(x, N)
