import numpy as np


class Tucker():
    def __init__(self):
        pass

    def __add__(self, Y, tol=None, rank=None):
        X=self
        pass

    def __mul__(self, Y, tol=None, rank=None):
        "element-wise multiplication of two Tucker tensors"
        X=self
        pass

    def fourier(self):
        "discrete Fourier transform"
        pass

    def ifourier(self):
        "inverse discrete Fourier transform"
        pass

    def full(self):
        "return a full tensor"
        pass

    def truncate(self, tol=None, rank=None):
        "return truncated tensor"
        pass

    def __repr__(self):
        pass

if __name__=='__main__':
    a = Tucker()
    b = Tucker()
    # addition
    c = a+b
    c2 = a.full()-b.full()
    print(np.linalg.norm(c.full()-c2))
    # multiplication
    c = a*b
    c2 = a.full()*b.full()
    print(np.linalg.norm(c.full()-c2))
    # DFT

    print('END')