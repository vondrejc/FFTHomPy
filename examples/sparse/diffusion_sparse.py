from __future__ import division, print_function

import numpy as np

from ffthompy import Timer, Struct
from ffthompy.materials import Material
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential, homog_GaNi_full_potential,
                                            homog_Ga_sparse, homog_GaNi_sparse)
from ffthompy.sparse.materials import SparseMaterial
from .material_setting import getMat_conf,recover_Aga,recover_Agani

import os
import sys


# PARAMETERS ##############################################################
dim=2
N=5*3**1
material=0
kind=0
kind_list=['cano','tucker','tt']

pars=Struct(dim=dim, # number of dimensions (works for 2D and 3D)
            N=dim*(N,), # number of voxels (assumed equal for all directions)
            Y=np.ones(dim), # size of periodic cell
            recover_sparse=1, # recalculate full material coefficients from sparse one
            solver=dict(tol=1e-10,
                        maxiter=50),
            )
pars_sparse=pars.copy()
pars_sparse.update(Struct(kind=kind_list[kind], # type of sparse tensor: 'cano', 'tucker', or 'tt'
                          rank=25, # rank of solution vector
                          precond_rank=25,
                          tol=None,
                          N=dim*(1*N,),
                          solver=dict(method='mr', #  method could be 'Richardson'(r),'minimal_residual'(mr), or 'Chebyshev'(c)
                                      approx_omega=True, # inner product of tuckers could be so slow
                                                          # that using an approximate omega could gain.
                                      eigrange=[0.6,50], # for Chebyshev solver
                                      tol=1e-10,
                                      maxiter=50,# no. of iterations for a solver
                                      divcrit=False),
                          ))

print('== format={}, N={}, dim={}, material={} ===='.format(pars_sparse.kind,
                                                            N, dim, material))

### get material settings for experiment
pars, pars_sparse, mat_conf = getMat_conf( material, pars, pars_sparse)

# generating material coefficients
mat=Material(mat_conf)
mats=SparseMaterial(mat_conf, pars_sparse.kind)

Agani=mat.get_A_GaNi(pars.N, primaldual='primal')
Aga=mat.get_A_Ga(pars.Nbar(pars.N), primaldual='primal')

Aganis=mats.get_A_GaNi(pars_sparse.N, primaldual='primal', k=pars_sparse.matrank)
Agas=mats.get_A_Ga(pars_sparse.Nbar(pars_sparse.N), primaldual='primal', k=pars_sparse.matrank)
Agas.set_fft_form()

Aga.val=recover_Aga(Aga,Agas)
Agani.val=recover_Agani(Agani,Aganis)

pars_sparse.update(Struct(alpha=0.5*(Agani[0, 0].min()+Agani[0, 0].max())))

#######OPERATORS ###############################################################
print('\n== Full solution with potential by CG (GaNi)===========')
resP_GaNi=homog_GaNi_full_potential(Agani, Aga, pars)
print('mean of solution={}'.format(resP_GaNi.Fu.mean()))
print('homogenised properties (component 11) = {}'.format(resP_GaNi.AH))

print('\n== Full solution with potential by CG (Ga) ===========')
resP_Ga=homog_Ga_full_potential(Aga, pars)
print('mean of solution={}'.format(resP_Ga.Fu.mean()))
print('homogenised properties (component 11) = {}'.format(resP_Ga.AH))

print('\n== SPARSE solver with preconditioner (Ga) =======================')
resS_Ga=homog_Ga_sparse(Agas, pars_sparse)
print('mean of solution={}'.format(resS_Ga.Fu.mean()))
print('homogenised properties (component 11) = {}'.format(resS_Ga.AH))
print(resS_Ga.Fu)
print('iterations={}'.format(resS_Ga.solver['kit']))
#if np.array_equal(pars.N, pars_sparse.N):
#    print('norm(dif)={}'.format(np.linalg.norm(resP_Ga.Fu.fourier().val-resS_Ga.Fu.fourier().full().val)))
print('norm(resP)={}'.format(resS_Ga.solver['norm_res']))

#print('memory efficiency = {0}/{1} = {2}'.format(resS_Ga.Fu.memory, resP_Ga.Fu.val.size, resS_Ga.Fu.memory/resP_Ga.Fu.val.size))

print('\n== SPARSE  solver with preconditioner (GaNi) =======================')
resS_GaNi=homog_GaNi_sparse(Aganis, Agas, pars_sparse)
print('mean of solution={}'.format(resS_GaNi.Fu.mean()))
print('homogenised properties (component 11) = {}'.format(resS_GaNi.AH))

print(resS_GaNi.Fu)
print('iterations={}'.format(resS_GaNi.solver['kit']))
if np.array_equal(pars.N, pars_sparse.N):
    print('norm(dif)={}'.format(np.linalg.norm(resP_GaNi.Fu.fourier().val-resS_GaNi.Fu.fourier().full().val)))
print('norm(resP)={}'.format(resS_GaNi.solver['norm_res']))
#print('memory efficiency = {0}/{1} = {2}'.format(resS_GaNi.Fu.memory, resP_GaNi.Fu.val.size, resS_GaNi.Fu.memory/resP_GaNi.Fu.val.size))

print('END')
