print """
This example shows how to interactively solve a problem of
FFT-based homogenization with exact integration, which is described in
J. Vondrejc, Improved guaranteed computable bounds on homogenized properties
of periodic media by FourierGalerkin method with exact integration,
Int. J. Numer. Methods Eng., 2016.
This publication will be referred as IJNME2016.
"""

print """Problem is defined via problem definition with instance 'pb'
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
"""

import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(sys.path[0], '..')))

import numpy as np
from ffthompy.materials import Material
import ffthompy.projections as proj
from ffthompy.matvec import DFT, VecTri, LinOper
from ffthompy.general.solver import linear_solver


dim = 2 # topological dimension of a problem
N = 25*np.ones(dim, dtype=np.int32) # no. of discretisation points
P = 5*np.ones(dim, dtype=np.int32) # resolution of material

pb = {'name': 'prob1',
      'physics': 'elasticity',
      'material': {'Y': np.ones(dim), # size of cell
                   'inclusions': ['square', 'otherwise'], # types of inclusions
                   'positions': [np.zeros(dim), ''], # position of inclusions
                   'params': [0.6*np.ones(dim), ''], # sizes of inclusions
                   'vals': [11*np.eye(dim), np.eye(dim)], # material coef.
                   'order': 1, # approximation order of material coef.
                   'P': P, # resolution of material
                   },
      'solve': {'kind': 'Ga', # defines a way of numerical integration
                'N': N, # no. of discretisation points (order of trig. polynomials)
                'primaldual': ['primal']}, # distiguish primal and dual formul.
      'postprocess': [{'kind': 'Ga'}],
      'solver': {'kind': 'CG',
                 'tol': 1e-8,
                 'maxiter': 1e3}}


# definition of material coefficients based on grid-based composite
mat = Material(pb['material'])
Nbar = 2*pb['solve']['N'] - 1
A = mat.get_A_Ga(Nbar=Nbar, primaldual=pb['solve']['primaldual'][0])

# projections in Fourier space
_, hG1N, _ = proj.scalar(pb['solve']['N'], pb['material']['Y'],
                         centered=True, NyqNul=True)
# increasing the projection with zeros to comply with a projection
# on double grid, see Definition 24 in IJNME2016
hG1N = hG1N.enlarge(Nbar)

FN = DFT(name='FN', inverse=False, N=Nbar) # discrete Fourier transform (DFT)
FiN = DFT(name='FiN', inverse=True, N=Nbar) # inverse DFT

G1N = LinOper(name='G1', mat=[[FiN, hG1N, FN]]) # projection in original space
Afun = LinOper(name='FiGFA', mat=[[G1N, A]]) # lin. operator for a linear system

E = np.zeros(dim); E[0] = 1 # macroscopic load
EN = VecTri(name='EN', macroval=E, N=Nbar, Fourier=False) # constant trig. pol.

x0 = VecTri(N=Nbar, d=dim, Fourier=False) # initial approximation to solvers
B = Afun(-EN) # right-hand side of linear system

X, info = linear_solver(solver='CG', Afun=Afun, B=B,
                        x0=x0, par=pb['solver'], callback=None)

print 'homogenised properties (component 11) =', A(X + EN)*(X + EN)

if __name__ == "__main__":
    ## plotting of local fields ##################
    X.plot(ind=0, N=N)

print 'END'
