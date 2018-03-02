print("""
This tutorial explains the usage of trigonometric polynomials and relating
operators for the use in FFT-based homogenization.

The basic classes, which are implemented in module "homogenize.matvec",
are listed here along with their important characteristics:
Grid : contains method "Grid.get_grid_coordinates", which returns coordinates
    of grid points
Tensor : this class represents a tensor-valued trigonometric polynomial and is
    thus the most important part of FFT-based homogenization
DFT : this class represents matrices of Discrete Fourier Transform, which is
    implemented via central version of FFT algorithm
----""")

import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(sys.path[0], '..')))
import numpy as np
from ffthompy.trigpol import Grid
from ffthompy.tensors import Tensor, DFT

print("""
The work with trigonometric polynomials is shown for""")
d = 2
N = 5*np.ones(d, dtype=np.int32)
print('dimension d =', d)
print('number of grid points N =', N)
print('which is implemented as a numpy.ndarray.')

print("""
Particularly, the vector-valued trigonometric polynomial is created as an instance 'xN' of class
'Tensor' and the random values are assigned.
""")

xN = Tensor(name='trigpol_rand', shape=(d,), N=N)
xN.randomize()

print("""
Basic properties of a trigonometric polynomials can be printed with a norm
corresponding to L2 norm of trigonometric polynomial, i.e.
xN =""")
print(xN)

print("""
The values of trigonometric polynomials are stored in atribute val of type
numpy.ndarray with shape = (self.d,) + tuple(self.N), i.e.
xN.val.shape =""")
print(xN.val.shape)
print("xN.val = xN[:] =")
print(xN.val)

print("""
In order to calculate Fourier coefficients of trigonometric polynomial,
we define DFT operators that are provided in class 'DFT'. The operation
is provided by central version of FFT algorithm and is implemented in method
'DFT.__call__' and/or 'DFT.__mul__'.
""")

FN = DFT(name='forward DFT', N=N, inverese=False)
FiN = DFT(name='inverse DFT', N=N, inverse=True)
print("FN = ")
print(FN)
print("FiN = ")
print(FiN)
print("""
The result of DFT is again the same trigonometric polynomial
with representation in Fourier domain (with Fourier coefficients);
FxN = FN*xN = FN(xN) =""")
FxN = FN*xN # Fourier coefficients of xN
print(FxN)

print("""
The forward and inverse DFT are mutually inverse operations that can
be observed by calculation of variable 'xN2':
xN2 = FiN(FxN) = FiN(FN(xN)) =""")
xN2 = FiN(FxN) # values of trigonometric polynomial at grid points
print(xN2)
print("and its comparison with initial trigonometric polynomial 'xN2'")
print("(xN == xN2) = ")
print(xN == xN2)

print("""
The norm of trigonometric polynomial calculated from Fourier
coefficients corresponds to L^2 norm and is the same like for values at grid
points, which is a consequence of Parseval's identity:
xN.norm() = np.linalg.norm(xN.val)/np.prod(xN.N)**0.5 =
= (np.sum(xN.val*xN.val)/np.prod(xN.N))**0.5 = """)
print(xN.norm())
print("""FxN.norm() = np.linalg.norm(FxN.val) =
= np.sum(FxN.val*np.conj(FxN.val)).real**0.5 =""")
print(FxN.norm())

print("""
The trigonometric polynomials can be also multiplied. The standard
multiplication with '*' operations corresponds to scalar product
leading to a square of norm, i.e.
FxN.norm() = xN.norm() = (xN*xN)**0.5 = (FxN*FxN)**0.5 =""")
print((xN*xN)**0.5)
print((FxN*FxN)**0.5)


print("""
The mean value of trigonometric polynomial is calculated independently for
each component of vector-field of trigonometric polynomial. In the real space,
it can be calculated as a mean of trigonometric polynomial at grid points,
while in Fourier space, it corresponds to zero frequency placed in the
center of grid, i.e.
xN.mean()[0] = xN[0].mean() = xN.val[0].mean() = FxN[0, 2, 2].real =""")
print(xN.mean()[0])
print(xN[0].mean())
print(xN.val[0].mean())
print(FxN[0, 2, 2].real)


print("""========================
Finally, we will plot the fundamental trigonometric polynomial, which
satisfies dirac-delta property at grid points and which
plays a major way in a theory of FFT-based homogenization.
phi =""")
phi = Tensor(name='phi_N,k', N=N, shape=())
phi.val[2, 2] = 1
print(phi)
print("phi.val =")
print(phi.val)

print("""
Fourier coefficients of phi
Fphi = FN*phi = FN(phi) =""")
Fphi = FN*phi
print(Fphi)
print("Fphi.val =")
print(Fphi.val)

print("""
In order to create a plot of this polynomial, it is
evaluated on a fine grid sizing
M = 16*N =""")
M = 16*N
print(M)

print("phi_fine = phi.project(M) =")
phi_fine = phi.project(M)
print(phi_fine)
print("""The procedure is provided by VecTri.enlarge(M) function, which consists of
a calculation of Fourier coefficients, putting zeros to Fourier coefficients
with high frequencies, and inverse FFT that evaluates the polynomial on
a fine grid.
""")


print("""In order to plot this polynomial, we also set a size of a cell
Y =""")
Y = np.ones(d) # size of a cell
print(Y)
print(""" and evaluate the coordinates of grid points, which are stored in
numpy.ndarray of following shape:
coord.shape =""")
coord = Grid.get_coordinates(M, Y)
print(coord.shape)

if __name__ == "__main__":
    print("""
    Now, the plot of fundamental trigonometric polynomial is shown:""")
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(coord[0], coord[1], phi_fine.val)
    plt.show()

print('END')
