from __future__ import division

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
import sys
from ffthompy import Timer, Struct
from ffthompy.materials import Material
from ffthompy.tensors import matrix2tensor
from ffthompy.sparse.materials_tucker import SparseMaterial
from ffthompy.sparse.homogenisation import homog_Ga_full_potential
from ffthompy.sparse.homogenisation_tucker import homog_sparse

# PARAMETERS ##############################################################
dim  = 3            # number of dimensions (works for 2D and 3D)
N    = dim*(31,)   # number of voxels (assumed equal for all directions)
Nbar = 2*np.array(N)-1
Y    = np.ones(dim)

pars=Struct(Amax=10.,          # material contrast
            maxiter=10,
            tol=None,
            rank=10,
            solver={'tol':1e-4}
            )

calc_eigs = 1
debug=False

# auxiliary values
prodN = np.prod(np.array(N)) # number of grid points
ndof  = dim*prodN # number of degrees-of-freedom
vec_shape=(dim,)+N # shape of the vector for storing DOFs

# PROBLEM DEFINITION ######################################################
E = np.zeros(vec_shape); E[0] = 1. # set macroscopic loading

# sparse A
mat_conf={'inclusions': ['square', 'otherwise'],
          'positions': [0.*np.ones(dim), ''],
          'params': [0.6*np.ones(dim), ''], # size of sides
          'vals': [10*np.eye(dim), 1.*np.eye(dim)],
          'Y': np.ones(dim),
          'P': N,
          'order': 0,}

mat=Material(mat_conf)
mats=SparseMaterial(mat_conf)

Agani = matrix2tensor(mat.get_A_GaNi(N, primaldual='primal'))
Aganis = mats.get_A_GaNi(N, primaldual='primal', k=2)
 
Aga = matrix2tensor(mat.get_A_Ga(Nbar, primaldual='primal'))
Agas = mats.get_A_Ga(Nbar, primaldual='primal', order=0, k=2)

print(np.linalg.norm(Agani.val[0,0]-Aganis.full()))
print(np.linalg.norm(Aga.val[0,0]-Agas.full()))

# OPERATORS ###############################################################
# print('\n== CG solver for gradient field ==============')
# res0 = homog_Ga_full(Aga, pars)
# print('homogenised properties (component 11) = {}'.format(res0.AH))

print('\n== Generating operators for formulation with potential ===========')
resP = homog_Ga_full_potential(Aga, pars)
print('homogenised properties (component 11) = {}'.format(resP.AH))


print('\n== SPARSE Richardson solver with preconditioner =======================')
resS = homog_sparse(Agas, pars)
print('homogenised properties (component 11) = {}'.format(resS.AH))

print(resS.Fu)
print('iterations={}'.format(resS.solver['kit']))
print('norm(dif)={}'.format(np.linalg.norm(resP.Fu.val-resS.Fu.full())))
print('norm(resP)={}'.format(resS.solver['norm_res']))
print('memory efficiency = {0}/{1} = {2}'.format(resS.Fu.size, resP.Fu.val.size, resS.Fu.size/resP.Fu.val.size))

print('END')

###  recorded results ###
#time (CG (potential)): [[148.31352600000002, 148.32072615623474]]
#homogenised properties (component 11) = 1.87953581074
#
#== SPARSE Richardson solver with preconditioner =======================
#time (Richardson (sparse)): [[77.45565899999997, 77.45629501342773]]

#dim  = 2            # number of dimensions (works for 2D and 3D)
#N    = dim*(505,)   # number of voxels (assumed equal for all directions)
#Nbar = 2*np.array(N)-1
#Y    = np.ones(dim)

####  with settings  #####
#pars=Struct(Amax=10.,          # material contrast
#            maxiter=10,
#            tol=None,
#            rank=10,
#            solver={'tol':1e-4}
#            )