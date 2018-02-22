import numpy as np
import numpy.fft as fft

def fftnc(x, N):
    """
    centered n-dimensional FFT algorithm
    """
    return fft.fftshift(fft.fftn(fft.ifftshift(x), N))/np.prod(N)

def ifftnc(Fx, N):
    """
    centered n-dimensional inverse FFT algorithm
    """
    return fft.fftshift(fft.ifftn(fft.ifftshift(Fx), N))*np.prod(N)
