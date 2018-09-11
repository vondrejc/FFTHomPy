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
N=15
# N=45
# N=135
material=0
kind=1

pars=Struct(dim=dim, # number of dimensions (works for 2D and 3D)
            N=dim*(N,), # number of voxels (assumed equal for all directions)
            Y=np.ones(dim), # size of periodic cell
            recover_sparse=1, # recalculate full material coefficients from sparse one
            solver=dict(tol=1e-4,
                        maxiter=10),
            )

pars_sparse=pars.copy()
kind_list=['cano','tucker','tt']
pars_sparse.update(Struct(kind=kind_list[kind], # type of sparse tensor: 'cano', 'tucker', or 'tt'
                          rank=10, # rank of solution vector
                          tol=None,
                          solver=dict(tol=1e-4,
                                      maxiter=10), # no. of iterations for a solver
                          ))

print('== format={}, N={}, dim={}, material={} ===='.format(pars_sparse.kind,
                                                            N, dim, material))

if dim==2:
    pars_sparse.update(Struct(N=dim*(1*N,),))
elif dim==3:
    pars_sparse.update(Struct(N=dim*(1*N,),))

if pars_sparse.kind in ['tt']:
    pars_sparse.update(Struct(tol=1e-14))

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
    if dim==2:
        kl.calc_modes(relerr=0.1)
    elif dim==3:
        kl.calc_modes(relerr=0.4)
    ip=np.random.random(kl.modes.n_kl)-0.5
    np.set_printoptions(precision=8)
    print('ip={}\n'.format(ip.__repr__()))
    if dim==2:
        ip=np.array([ 0.24995 , 0.009014,-0.004228, 0.266437, 0.345009,-0.29721 ,-0.291875,-0.125469, 0.495526,-0.452405,-0.333025, 0.208331, 0.045902,-0.441424,-0.274428,-0.243702,-0.146728, 0.239476, 0.404311, 0.214929])
#         ip=np.array([-0.39423835, -0.4389707 ,  0.4627532 ,  0.02490174, -0.49743839,  0.45896435, -0.11609073,  0.3577237 , -0.21840589, -0.19542385,  0.13852267, -0.00356906,  0.10412291,  0.21354277,  0.36524674, -0.02004963,  0.00604674,  0.17296395, -0.20326143,  0.49092423])
#         ip=np.array([-0.06137196,  0.12031872, -0.13422506,  0.34617451, -0.43839178, -0.45349221,  0.30368884, -0.47787331, -0.19005504,  0.32482465,  0.37039887, -0.36310423, -0.42458556, -0.38627275, -0.18097583, -0.36801909,  0.31939606,  0.46660035,  0.36363185, -0.0993647 ])
    elif dim==3:
        ip=np.array([-0.39561222,-0.37849801, 0.46069148,-0.0354164 , 0.04269214,-0.00624889, 0.18498634, 0.31043535,-0.14730729,-0.39756328, 0.48918557, 0.15098372,-0.11217825,-0.26506403, 0.2006125 ,-0.2596631 ,-0.16854476,-0.44617782,-0.19412459, 0.32968464,-0.18441118,-0.15455307, 0.1779399 ,-0.21214177, 0.18394519,-0.24561992])

    def mat_fun(coor):
        val=np.zeros_like(coor[0])
        for ii in range(kl.modes.n_kl):
            val+=ip[ii]*kl.mode_fun(ii, coor)
        return np.einsum('ij,...->ij...', np.eye(dim), kl.transform(val))

    mat_conf={'fun':mat_fun,
              'Y': np.ones(dim),
              'P': pars.N,
              'order': 1, }

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
    Agani.val=np.einsum('ij,...->ij...', np.eye(dim), Aganis.full())
    Aga.val=np.einsum('ij,...->ij...', np.eye(dim), Agas.full())

if np.array_equal(pars.N, pars_sparse.N):
    print(np.linalg.norm(Agani.val[0, 0]-Aganis.full()))
    print(np.linalg.norm(Aga.val[0, 0]-Agas.full()))

pars_sparse.update(Struct(alpha=0.5*(Agani[0, 0].min()+Agani[0, 0].max())))

#######OPERATORS ###############################################################
print('\n== Full solution with potential by CG (GaNi)===========')
resP_GaNi=homog_GaNi_full_potential(Agani, Aga, pars)
print('homogenised properties (component 11) = {}'.format(resP_GaNi.AH))

print('\n== Full solution with potential by CG (Ga) ===========')
resP_Ga=homog_Ga_full_potential(Aga, pars)
print('homogenised properties (component 11) = {}'.format(resP_GaNi.AH))

print('\n== SPARSE Richardson solver with preconditioner (Ga) =======================')
resS_Ga=homog_Ga_sparse(Agas, pars_sparse)
print('homogenised properties (component 11) = {}'.format(resS_Ga.AH))
print(resS_Ga.Fu)
print('iterations={}'.format(resS_Ga.solver['kit']))
if np.array_equal(pars.N, pars_sparse.N):
    print('norm(dif)={}'.format(np.linalg.norm(resP_Ga.Fu.val-resS_Ga.Fu.full())))
print('norm(resP)={}'.format(resS_Ga.solver['norm_res']))
print('memory efficiency = {0}/{1} = {2}'.format(resS_Ga.Fu.memory, resP_Ga.Fu.val.size, resS_Ga.Fu.memory/resP_Ga.Fu.val.size))

print('\n== SPARSE Richardson solver with preconditioner (GaNi) =======================')
resS_GaNi=homog_GaNi_sparse(Aganis, Agas, pars_sparse)
print('homogenised properties (component 11) = {}'.format(resS_GaNi.AH))

print(resS_GaNi.Fu)
print('iterations={}'.format(resS_GaNi.solver['kit']))
if np.array_equal(pars.N, pars_sparse.N):
    print('norm(dif)={}'.format(np.linalg.norm(resP_GaNi.Fu.val-resS_GaNi.Fu.full())))
print('norm(resP)={}'.format(resS_GaNi.solver['norm_res']))
print('memory efficiency = {0}/{1} = {2}'.format(resS_GaNi.Fu.memory, resP_GaNi.Fu.val.size, resS_GaNi.Fu.memory/resP_GaNi.Fu.val.size))

print('END')
