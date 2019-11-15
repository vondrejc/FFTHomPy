from __future__ import division, print_function

import numpy as np

from ffthompy import Struct, Timer
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential, homog_GaNi_full_potential,
                                            homog_Ga_sparse, homog_GaNi_sparse)

from examples.sparse.material_setting import get_material_coef


# PARAMETERS ##############################################################
dim=3
N=5*3**1
material=0
kind=1
kind_list=['cano','tucker','tt']

pars=Struct(dim=dim, # number of dimensions (works for 2D and 3D)
            N=dim*(N,), # number of voxels (assumed equal for all directions)
            Y=np.ones(dim), # size of periodic cell
            recover_sparse=1, # recalculate full material coefficients from sparse one
            solver=dict(tol=1e-8,
                        maxiter=50),
            )
pars_sparse=pars.copy()
pars_sparse.update(Struct(kind=kind_list[kind], # type of sparse tensor: 'cano', 'tucker', or 'tt'
                          rank=10, # rank of solution vector
                          precond_rank=10,
                          tol=None,
                          N=dim*(N,),
                          rhs_tol=1e-8,
                          solver=dict(method='mrd', # method could be 'Richardson'(r),'minimal_residual'(mr), or 'Chebyshev'(c)
                                      approx_omega=False, # inner product of tuckers could be so slow
                                                          # that using an approximate omega could gain.
                                      tol=1e-4,
                                      maxiter=15, # no. of iterations for a solver
                                      divcrit=False), # stop if the norm of residuum fails to decrease
                          ))

print('== format={}, N={}, dim={}, material={} ===='.format(pars_sparse.kind,
                                                            N, dim, material))
print('dofs = {}'.format(N**dim))

# get material coefficients
Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse)

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
print('iterations={}'.format(resS_Ga.solver['kit']))
print('norm(resP)={}'.format(resS_Ga.solver['norm_res']))

print('\n== SPARSE solver with preconditioner (GaNi) =======================')
resS_GaNi=homog_GaNi_sparse(Aganis, Agas, pars_sparse)
print('mean of solution={}'.format(resS_GaNi.Fu.mean()))
print('homogenised properties (component 11) = {}'.format(resS_GaNi.AH))
print('iterations={}'.format(resS_GaNi.solver['kit']))
if np.array_equal(pars.N, pars_sparse.N):
    print('norm(dif)={}'.format(np.linalg.norm(resP_GaNi.Fu.fourier(Fourier=False).val-resS_GaNi.Fu.fourier().full().val)))
print('norm(resP)={}'.format(resS_GaNi.solver['norm_res']))

print('END')
