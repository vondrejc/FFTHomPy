"""
collection of 1-D FFTs for Fourier transform of basis, all done on the 2nd dimension of the basis.
"""

from scipy import fftpack
import numpy.fft as npfft


def cfftc(x, N):
    """
    centered 1-dimensional FFT algorithm
    """
    return npfft.fftshift(npfft.fft(npfft.ifftshift(x, axes=1), axis=1), axes=1)/N

def icfftc(Fx, N,real_output=False):
    """
    centered 1-dimensional inverse FFT algorithm
    """
    if real_output:
        return npfft.fftshift(npfft.ifft(npfft.ifftshift(Fx, axes=1), axis=1), axes=1).real*N
    else:
        return npfft.fftshift(npfft.ifft(npfft.ifftshift(Fx, axes=1), axis=1), axes=1)*N

def fftc(x, N):
    """
    centered 1-dimensional FFT algorithm
    """
    return npfft.fftshift(npfft.fft(x, axis=1),axes=1)/N

def icfft(Fx, N,real_output=False):
    """
    centered 1-dimensional inverse FFT algorithm
    """
    if real_output:
        return npfft.ifft(npfft.ifftshift(Fx, axes=1), axis=1).real*N
    else:
        return npfft.ifft(npfft.ifftshift(Fx, axes=1), axis=1)*N

def fft(x, N):
    return npfft.fft(x, axis=1)/N # numpy.fft.fft

def ifft(x, N,real_output=False):
    if real_output:
        return npfft.ifft(x, axis=1).real*N # numpy.fft.fft
    else:
        return npfft.ifft(x, axis=1)*N # numpy.fft.fft

def rfft(x, N):
    return npfft.rfft(x.real, axis=1)/N # real version of numpy.fft.fft

def irfft(x, N, real_output=True):
    return npfft.irfft(x, axis=1)*N # real version of numpy.fft.fft

def srfft(x, N):
    return fftpack.rfft(x.real, axis=1)/N  # 1-D real fft from scipy.fftpack.rfft

def sirfft(x, N, real_output=True):
    return fftpack.irfft(x.real, axis=1)*N  # 1-D real inverse fft from scipy.fftpack.irfft
