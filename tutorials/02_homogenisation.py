from __future__ import division, print_function

import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")


logger.info("""
This example shows how to interactively solve a problem of 
FFT-based homogenization for linear elasticity. This algorithm is equivalent
to original Moulinec-Suquet scheme, which is emphasized here.

In FFTHomPy program, the problem is defined using input file in folder
'examples' and then launched using module 'main.py' with an input file as
a parameter, e.g. in terminal launch
'./main.py examples/elasticity/linelas_3d.py'
or
'python main.py examples/elasticity/linelas_3d.py'

Here, we follow an example in examples/elasticity/linelas_3d.py using
an interactive mode with comments.
""")

import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(sys.path[0], '..')))

logger.info("""
The problem in topological dimension
dim = """)
dim = 2 # topological dimension of a problem
logger.info('%s', dim)
logger.info("""is defined on cell sizing
Y =""")
Y = np.ones(dim)
logger.info('%s', Y)

logger.info("""
The material coefficients of linear elastic medium are defined in terms of
bulk modulus
K =""")
K = np.array([1, 10.], dtype=np.float64)
logger.info('%s', K)
logger.info("""and shear modulus
G =""")
G = np.array([1, 5.], dtype=np.float64)
logger.info('%s', G)
logger.info("""where the first value of both 'K' and 'G' corresponds to the matrix
phase, while the second value corresponds to the cube inclusion of size 0.6.
The we express material coefficients as stiffness matrices in Mandel`s
engineering notation:
matcoefM.mandel =""")
from ffthompy.mechanics.matcoef import ElasticTensor

plane=None
if dim==2:
    plane='strain'
matcoefM = ElasticTensor(bulk=K[0], mu=G[0], plane=plane) # mat. coef. of matrix phase 
matcoefI = ElasticTensor(bulk=K[1], mu=G[1], plane=plane) # mat. coef. of inclusion
logger.info('%s', matcoefM.mandel)
logger.info("matcoefI.mandel =")
logger.info('%s', matcoefI.mandel)

logger.info("""The problem, which is discretized with following no. of grid points
N = """)
N = 5*np.ones(dim, dtype=np.int32) # no. of grid points
logger.info('%s', N)
logger.info("""is defined via problem definition with instance 'pb'
with a following keys:

material : dictionary
    there are two possibilities to define material:
        a) with a function that returns a material coefficients at grid points
        b) using 'inclusions' (square or circle) with given
            'positions',
            'params' (defining a size of inclusion),
            'vals' (defining the material parameters across inclusion)

solve : dictionary
    it defines what to solve with following keys:
         'kind' (stores the method of numerical integration)
         'N' (no. of grid points)
         'primaldual' (defines if it is to solve primal or dual formulation)

postprocess : list
    the way for evaluation of homogenized properties

solver : dictionary
    stores the parameters for linear solver
""")

pb = {'name': 'prob1',
      'physics': 'elasticity',
      'material': {'Y': Y, # size of cell
                   'inclusions': ['cube', 'otherwise'], # types of inclusions
                   'positions': [np.zeros(dim), ''], # position of inclusions
                   'params': [0.6*np.ones(dim), ''], # sizes of inclusions
                   'vals': [matcoefI.mandel, matcoefM.mandel], # material coef.
                   },
      'solve': {'kind': 'GaNi', # defines a way of numerical integration
                'N': N, # no. of grid points (order of trig. polynomials)
                'primaldual': ['primal']}, # distiguish primal and dual formul.
      'postprocess': [{'kind': 'GaNi'}],
      'solver': {'kind': 'CG',
                 'tol': 1e-8,
                 'maxiter': 1e3}}

logger.info("""
The material properties for FFT-based homogenization are stored at grid
points and are represented with class 'Matrix',
which has the same or similar attributes and methods as a VecTri class
from 'tutorial_01.py'. In order to show its application, we create a material
coefficients composed of random values, symmetrize them, and sum them with
a multiplication of identity to obtain positive definite matrix, i.e.
A =""")
from ffthompy.tensors import Tensor

D = int(dim*(dim+1)/2)
A = Tensor(N=N, shape=(D,D), Fourier=False, multype=21)
A.randomize()
A = A + A.transpose() # symmetrization
# adding a multiplication of identity matrix:
I = Tensor(N=N, shape=(D,D), Fourier=False, multype=21)
I.identity()
A += 3*I
A.name='material'
logger.info('%s', A)

logger.info("""
The values of material coefficients are stored in 'self.val', which == 'numpy.ndarray' object of shape = (D, D) + tuple(N)
here D=d*(d+1)/2 is a number of components of elasticity matrix in
engineering notation, so
A.val.shape =""")
logger.info('%s', A.val.shape)

del A

logger.info("""
Now, we create an instance of material coefficients defined in
a problem definition 'pb'. The methods, that evaluate the material coefficients
in accordance with the inclusion types, are implemented in
class 'Material' in module 'homogenize.materials'. Hence,
A =""")
# definition of material
from ffthompy.materials import Material

mat = Material(pb['material'])
A = mat.get_A_GaNi(pb['solve']['N'], 'primal')
logger.info('%s', A)

logger.info("""---------------------
Then, we define projection operators represented also with 'Matrix' class
and FFT operators represented with 'DFT' class.""")
import ffthompy.projections as proj

_, hG1hN, hG1sN, hG2hN, hG2sN = proj.elasticity(pb['solve']['N'], pb['material']['Y'],
                                                NyqNul=True, tensor=True)

from ffthompy.tensors import DFT, Operator

FN = DFT(name='FN', inverse=False, N=N)
FiN = DFT(name='FiN', inverse=True, N=N)

logger.info("""
The class 'Matrix' allows to check that the objects are projections""")
logger.info('(hG1hN*hG1hN-hG1hN).norm() = %s', (hG1hN*hG1hN-hG1hN).norm())
logger.info('(hG1sN*hG1sN-hG1sN).norm() = %s', (hG1sN*hG1sN-hG1sN).norm())
logger.info('(hG2hN*hG2hN-hG2hN).norm() = %s', (hG2hN*hG2hN-hG2hN).norm())
logger.info('(hG2sN*hG2sN-hG2sN).norm() = %s', (hG2sN*hG2sN-hG2sN).norm())
logger.info("""and check also for their ortogonality:""")
logger.info('(hG1hN*hG1sN).norm() = %s', (hG1hN*hG1sN).norm())
logger.info('(hG1hN*hG2hN).norm() = %s', (hG1hN*hG2hN).norm())
logger.info('(hG1hN*hG2sN).norm() = %s', (hG1hN*hG2sN).norm())
logger.info('(hG1sN*hG2hN).norm() = %s', (hG1sN*hG2hN).norm())
logger.info('(hG1sN*hG2sN).norm() = %s', (hG1sN*hG2sN).norm())
logger.info('(hG2hN*hG2sN).norm() = %s', (hG2hN*hG2sN).norm())

logger.info("""
Then we put all operators together to get a linear operator 'G1'
represented by 'LinOper' class and providing a projection on compatible fields.
Here, we also shows that both operators are orthogonal projection:""")
# projection on compatible fields
G1N = Operator(name='G1', mat=[[FiN, hG1hN + hG1sN, FN]])
# projection on divergence-free fields
G2N = Operator(name='G1', mat=[[FiN, hG2hN + hG2sN, FN]])
uN = Tensor(N=N, shape=(D,)).randomize()
vN = Tensor(N=N, shape=(D,)).randomize()
logger.info('(G1N(G1N(uN))-G1N(uN)).norm() = %s', (G1N(G1N(uN))-G1N(uN)).norm())
logger.info('(G2N*G2N*uN-G2N*uN).norm() = %s', (G2N(G2N(uN))-G2N(uN)).norm())
logger.info('G1N(uN)*G2N(vN) = %s', G1N(uN)*G2N(vN))
logger.info("""We note that different syntax has been used to show the usage of
class 'Matrix'. The important methods for 'LinOper', 'Matrix', and 'DFT'
classes are 'self.__call__' and 'self.__mul__', which enable the application to
trigonometric polynomials.""")

logger.info("""
We also show the possibility of linear operator ('LinOper') by showing one
iteration of Moulinec-Suquet scheme using three versions.
""")
# linear operator for solving a linear system
Afun = Operator(name='FiGFA', mat=[[G1N, A]])
# strain representation
epN = Tensor(name='strain', N=N, shape=(D,)).randomize()
# sufficiently small parameter corresponding to reference medium 
alp = 0.1

logger.info("#1")
logger.info("epN_new = epN - alp*Afun(epN)")
epN_new = epN - alp*Afun(epN)
logger.info("#2")
logger.info("""hG1 = hG1hN + hG1sN
epN_new2 = epN - alp * FiN(hG1(FN(A(epN))))""")
hG1 = hG1hN + hG1sN
epN_new2 = epN - alp * FiN(hG1(FN(A(epN))))

logger.info("""
Then the resulting vectors are compared to show they are the same:
(epN_new == epN_new2) = """)
logger.info('%s', epN_new == epN_new2)

logger.info("""==============================
Now, we show a solution of homogenization problem for material defined in
problem 'pb' and particular choice of macroscopic value
E =""")
# macroscopic load in Mandel's notation
E = np.zeros(D)
E[0] = 1
EN = Tensor(name='EN', N=N, shape=(D,), Fourier=False)
EN.set_mean(E)
logger.info('%s', EN)

# definition of reference media according to Moulinec-Suquet
Km = K.mean()
Gm = G.mean()
a = 1/(Km+4./3*Gm)
b = 1./(2*Gm)

# linear combination of projections corresponding to Moulinec-Suquet scheme
G1N_MS = Operator(name='G1', mat=[[FiN, a*hG1hN + b*hG1sN, FN]])
# linear system with scaled projection
Afun_MS = Operator(name='FiGFA', mat=[[G1N_MS, A]])

# linear system with orthogonal projection
Afun = Operator(name='FiGFA', mat=[[G1N, A]])

# initial approximation to solvers
x0 = Tensor(N=N, shape=(D,), Fourier=False)

B = Afun(-EN) # RHS
B_MS = Afun_MS(-EN) # RHS

logger.info("""
The linear system is solved with different algorithms:""")
logger.info("""
version #1 :
The solution by Conjugate gradients with zero initial vector; the macroscopic
value occurs on right-hand-side as a load.""")
from ffthompy.general.solver import linear_solver

X, info = linear_solver(solver='CG', Afun=Afun, B=B,
                        x0=x0, par=pb['solver'], callback=None)

logger.info('Homogenised properties (component 11) = %s', A(X+EN)*(X+EN))

logger.info("""
version #2 :
The solution by Conjugate gradients with initial vector corresponding to
macroscopic value; in this case, the right-hand-side remains zero.
The difference to previous solution is printed:
(X+EN == X2) =""")
X2, info2 = linear_solver(solver='CG', Afun=Afun, B=x0,
                          x0=EN, par=pb['solver'], callback=None)
logger.info('%s', X+EN == X2)

logger.info("""
version #3 :
The solution by Conjugate gradients as in #2 with initial vector
corresponding to macroscopic value. However, the projection operator is scaled
in accordance with Moulinec-Suquet scheme.
(X+EN == X3) =""")
X3, info3 = linear_solver(solver='CG', Afun=Afun_MS, B=x0,
                          x0=EN, par=pb['solver'], callback=None)
logger.info('%s', X+EN == X3)

logger.info("""
version #4 :
The solution by 'scipy.sparse.linalg.cg' with initial vector corresponding to
macroscopic value.
(X+EN == X4) =""")
# this helps to reshape the values to fit to scipy solvers
Afun.define_operand(B)
X4, info4 = linear_solver(solver='scipy_cg', Afun=Afun, B=x0,
                          x0=EN, par=pb['solver'], callback=None)
logger.info('%s', X+EN == X4)
logger.info("""
version #5 :
The solution by Richardson's iteration, which is exactly the same like
Moulinec-Suquet algorithm; the initial value is taken as macroscopic vector.
(X+EN == X5) =""")
X5 = EN
resn = 1.
iters = 0
while (resn > pb['solver']['tol'] and iters < pb['solver']['maxiter']):
        iters += 1
        X5_prev = X5
        X5 = X5 - Afun_MS(X5)
        resn = (X5_prev-X5).norm()

logger.info('%s', X+EN == X5)

logger.info('END')
