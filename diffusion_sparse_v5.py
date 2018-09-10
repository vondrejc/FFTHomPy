from __future__ import division, print_function

import numpy as np

from ffthompy import Timer, Struct
from ffthompy.materials import Material
from ffthompy.tensors import matrix2tensor
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential, homog_GaNi_full_potential,
                                            homog_Ga_sparse, homog_GaNi_sparse)
from ffthompy.sparse.materials import SparseMaterial
from uq.decomposition import KL_Fourier

import os
import sys
os.nice(19)

# PARAMETERS ##############################################################
dim=3
N=95
material=2

pars=Struct(dim=dim, # number of dimensions (works for 2D and 3D)
            N=dim*(N,), # number of voxels (assumed equal for all directions)
            Y=np.ones(dim), # size of periodic cell
            recover_sparse=1, # recalculate full material coefficients from sparse one
            solver=dict(tol=1e-8,
                        maxiter=1e2),
            )

pars_sparse=pars.copy()
pars_sparse.update(Struct(kind='cano', # type of sparse tensor: 'cano', 'tucker', or 'tt'
                      rank=5, # rank of solution vector
                      tol=None,
                      solver=dict(tol=1e-4,
                                  maxiter=10), # no. of iterations for a solver
                      ))

if dim==2:
    pars_sparse.update(Struct(N=dim*(1*N,),
                              ))
elif dim==3:
    pars_sparse.update(Struct(N=dim*(1*N,),))

# auxiliary operator
Nbar=lambda N: 2*np.array(N)-1

# PROBLEM DEFINITION ######################################################
if material in [0]:
    mat_conf={'inclusions': ['square', 'otherwise'],
              'positions': [0.*np.ones(dim), ''],
              'params': [0.6*np.ones(dim), ''], # size of sides
              'vals': [10*np.eye(dim), 1.*np.eye(dim)],
              'Y': np.ones(dim),
              'P': pars.N,
              'order': 0, }
    pars_sparse.update(Struct(matrank=2))

elif material in [1]:
    mat_conf={'inclusions': ['pyramid', 'all'],
              'positions': [0.*np.ones(dim), ''],
              'params': [0.8*np.ones(dim), ''], # size of sides
              'vals': [10*np.eye(dim), 1.*np.eye(dim)],
              'Y': np.ones(dim),
              'P': pars.N,
              'order': 1, }
    pars_sparse.update(Struct(matrank=2))

elif material in [2]: # stochastic material
    pars_sparse.update(Struct(matrank=10))

    kl=KL_Fourier(covfun=2, cov_pars={'rho':0.15, 'sigma': 1.}, N=pars.N, puc_size=pars.Y,
                  transform=lambda x: 1e4*np.exp(x))
    kl.calc_modes(relerr=0.1)
    ip = np.random.random(kl.modes.n_kl)-0.5

    def mat_fun(coor):
        val = np.zeros_like(coor[0])
        for ii in range(kl.modes.n_kl):
            val += ip[ii]*kl.mode_fun(ii, coor)
        return np.einsum('ij,...->ij...', np.eye(dim), kl.transform(val))

    mat_conf={'fun':mat_fun,
              'Y': np.ones(dim),
              'P': pars.N,
              'order': 1,}

else:
    raise ValueError()

# generating material coefficients
mat=Material(mat_conf)
mats=SparseMaterial(mat_conf, pars_sparse.kind)

Agani=matrix2tensor(mat.get_A_GaNi(pars.N, primaldual='primal'))
Aganis=mats.get_A_GaNi(pars_sparse.N, primaldual='primal', k=pars_sparse.matrank)

Aga=matrix2tensor(mat.get_A_Ga(Nbar(pars.N), primaldual='primal'))
Agas=mats.get_A_Ga(Nbar(pars_sparse.N), primaldual='primal', k=pars_sparse.matrank)

if np.array_equal(pars.N, pars_sparse.N):
    print(np.linalg.norm(Agani.val[0, 0]-Aganis.full()))
    print(np.linalg.norm(Aga.val[0, 0]-Agas.full()))

if pars.recover_sparse:
    print('recovering full material tensors...')
    Agani.val = np.einsum('ij,...->ij...', np.eye(dim), Aganis.full())
    Aga.val = np.einsum('ij,...->ij...', np.eye(dim), Agas.full())

if np.array_equal(pars.N, pars_sparse.N):
    print(np.linalg.norm(Agani.val[0, 0]-Aganis.full()))
    print(np.linalg.norm(Aga.val[0, 0]-Agas.full()))

pars_sparse.update(Struct(alpha=0.5*(Agani[0,0].min()+Agani[0,0].max())))

#######OPERATORS ###############################################################
print('\n== Full solution with potential by CG (GaNi)===========')
resP=homog_GaNi_full_potential(Agani, Aga, pars)
print('homogenised properties (component 11) = {}'.format(resP.AH))
 
print('\n== Full solution with potential by CG (Ga) ===========')
resP=homog_Ga_full_potential(Aga, pars)
print('homogenised properties (component 11) = {}'.format(resP.AH))

print('\n== SPARSE Richardson solver with preconditioner (Ga) =======================')
resS=homog_Ga_sparse(Agas, pars_sparse)
print('homogenised properties (component 11) = {}'.format(resS.AH))
print(resS.Fu)
print('iterations={}'.format(resS.solver['kit']))
if np.array_equal(pars.N, pars_sparse.N):
    print('norm(dif)={}'.format(np.linalg.norm(resP.Fu.val-resS.Fu.full())))
print('norm(resP)={}'.format(resS.solver['norm_res']))
print('memory efficiency = {0}/{1} = {2}'.format(resS.Fu.memory, resP.Fu.val.size, resS.Fu.memory/resP.Fu.val.size))

print('\n== SPARSE Richardson solver with preconditioner (GaNi) =======================')
resS=homog_GaNi_sparse(Aganis, Agas, pars_sparse)
print('homogenised properties (component 11) = {}'.format(resS.AH))

print(resS.Fu)
print('iterations={}'.format(resS.solver['kit']))
if np.array_equal(pars.N, pars_sparse.N):
    print('norm(dif)={}'.format(np.linalg.norm(resP.Fu.val-resS.Fu.full())))
print('norm(resP)={}'.format(resS.solver['norm_res']))
print('memory efficiency = {0}/{1} = {2}'.format(resS.Fu.memory, resP.Fu.val.size, resS.Fu.memory/resP.Fu.val.size))

print('END')
