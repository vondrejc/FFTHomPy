from __future__ import division, print_function

import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")


logger.info("""
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

from ffthompy.tensors import DFT, Tensor
from ffthompy.trigpol import Grid

logger.info("""
The work with trigonometric polynomials is shown for""")
d = 2
N = 5*np.ones(d, dtype=np.int32)
logger.info('dimension d = %s', d)
logger.info('number of grid points N = %s', N)
logger.info('which is implemented as a numpy.ndarray.')

logger.info("""
Particularly, the vector-valued trigonometric polynomial is created as an instance 'xN' of class
'Tensor' and the random values are assigned.
""")

xN = Tensor(name='trigpol_rand', shape=(d,), N=N)
xN.randomize()

logger.info("""
Basic properties of a trigonometric polynomials can be printed with a norm
corresponding to L2 norm of trigonometric polynomial, i.e.
xN =""")
logger.info('%s', xN)

logger.info("""
The values of trigonometric polynomials are stored in atribute val of type
numpy.ndarray with shape = (self.d,) + tuple(self.N), i.e.
xN.val.shape =""")
logger.info('%s', xN.val.shape)
logger.info("xN.val = xN[:] =")
logger.info('%s', xN.val)

logger.info("""
In order to calculate Fourier coefficients of trigonometric polynomial,
we define DFT operators that are provided in class 'DFT'. The operation
is provided by central version of FFT algorithm and is implemented in method
'DFT.__call__' and/or 'DFT.__mul__'.
""")

FN = DFT(name='forward DFT', N=N, inverese=False)
FiN = DFT(name='inverse DFT', N=N, inverse=True)
logger.info("FN = ")
logger.info('%s', FN)
logger.info("FiN = ")
logger.info('%s', FiN)
logger.info("""
The result of DFT is again the same trigonometric polynomial
with representation in Fourier domain (with Fourier coefficients);
FxN = FN*xN = FN(xN) =""")
FxN = FN*xN # Fourier coefficients of xN
logger.info('%s', FxN)

logger.info("""
The forward and inverse DFT are mutually inverse operations that can
be observed by calculation of variable 'xN2':
xN2 = FiN(FxN) = FiN(FN(xN)) =""")
xN2 = FiN(FxN) # values of trigonometric polynomial at grid points
logger.info('%s', xN2)
logger.info("and its comparison with initial trigonometric polynomial 'xN2'")
logger.info("(xN == xN2) = ")
logger.info('%s', xN == xN2)

logger.info("""
The norm of trigonometric polynomial calculated from Fourier
coefficients corresponds to L^2 norm and is the same like for values at grid
points, which is a consequence of Parseval's identity:
xN.norm() = np.linalg.norm(xN.val)/np.prod(xN.N)**0.5 =
= (np.sum(xN.val*xN.val)/np.prod(xN.N))**0.5 = """)
logger.info('%s', xN.norm())
logger.info("""FxN.norm() = np.linalg.norm(FxN.val) =
= np.sum(FxN.val*np.conj(FxN.val)).real**0.5 =""")
logger.info('%s', FxN.norm())

logger.info("""
The trigonometric polynomials can be also multiplied. The standard
multiplication with '*' operations corresponds to scalar product
leading to a square of norm, i.e.
FxN.norm() = xN.norm() = (xN*xN)**0.5 = (FxN*FxN)**0.5 =""")
logger.info('%s', (xN*xN)**0.5)
logger.info('%s', (FxN*FxN)**0.5)


logger.info("""
The mean value of trigonometric polynomial is calculated independently for
each component of vector-field of trigonometric polynomial. In the real space,
it can be calculated as a mean of trigonometric polynomial at grid points,
while in Fourier space, it corresponds to zero frequency placed in the
center of grid, i.e.
xN.mean()[0] = xN[0].mean() = xN.val[0].mean() = FxN[0, 2, 2].real =""")
logger.info('%s', xN.mean()[0])
logger.info('%s', xN[0].mean())
logger.info('%s', xN.val[0].mean())
logger.info('%s', FxN[0, 2, 2].real)


logger.info("""========================
Finally, we will plot the fundamental trigonometric polynomial, which
satisfies dirac-delta property at grid points and which
plays a major way in a theory of FFT-based homogenization.
phi =""")
phi = Tensor(name='phi_N,k', N=N, shape=())
phi.val[2, 2] = 1
logger.info('%s', phi)
logger.info("phi.val =")
logger.info('%s', phi.val)

logger.info("""
Fourier coefficients of phi
Fphi = FN*phi = FN(phi) =""")
Fphi = FN*phi
logger.info('%s', Fphi)
logger.info("Fphi.val =")
logger.info('%s', Fphi.val)

logger.info("""
In order to create a plot of this polynomial, it is
evaluated on a fine grid sizing
M = 16*N =""")
M = 16*N
logger.info('%s', M)

logger.info("phi_fine = phi.project(M) =")
phi_fine = phi.project(M)
logger.info('%s', phi_fine)
logger.info("""The procedure is provided by VecTri.enlarge(M) function, which consists of
a calculation of Fourier coefficients, putting zeros to Fourier coefficients
with high frequencies, and inverse FFT that evaluates the polynomial on
a fine grid.
""")


logger.info("""In order to plot this polynomial, we also set a size of a cell
Y =""")
Y = np.ones(d) # size of a cell
logger.info('%s', Y)
logger.info(""" and evaluate the coordinates of grid points, which are stored in
numpy.ndarray of following shape:
coord.shape =""")
coord = Grid.get_coordinates(M, Y)
logger.info('%s', coord.shape)

if __name__ == "__main__":
    logger.info("""
    Now, the plot of fundamental trigonometric polynomial is shown:""")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(coord[0], coord[1], phi_fine.val)
    plt.show()

logger.info('END')
