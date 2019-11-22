from __future__ import division, print_function

import numpy as np
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential, homog_GaNi_full_potential,
                                            homog_Ga_sparse, homog_GaNi_sparse)

from examples.sparse.setting import get_material_coef, get_default_parameters


# PARAMETERS ##############################################################
dim=2
N=5*3**2
material=0
kind=0 # from kind_list=['cano','tucker','tt']

pars, pars_sparse=get_default_parameters(dim, N, material, kind)
pars_sparse.debug=True

print('== format={}, N={}, dim={}, material={}, rank={} ===='.format(pars_sparse.kind, N, dim,
                                                                     material, pars_sparse.rank))
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
