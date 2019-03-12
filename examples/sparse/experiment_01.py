from __future__ import division, print_function

import numpy as np
from numpy.core.multiarray import ndarray
from typing import List, Any, Union

from ffthompy import Timer, Struct
from ffthompy.materials import Material
from ffthompy.tensors import matrix2tensor
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential, homog_GaNi_full_potential,
                                            homog_Ga_sparse, homog_GaNi_sparse)
from ffthompy.sparse.materials import SparseMaterial
from uq.decomposition import KL_Fourier
from ffthompy.sparse.objects import SparseTensor
from material_setting import getMat_conf,recover_Aga,recover_Agani



import os
import sys
import pickle
import matplotlib.pyplot as plt

os.nice(19)

#######################################################
if not os.path.exists('data_for_plot'):
    os.makedirs('data_for_plot/dim_2/mat_0/')
    os.makedirs('data_for_plot/dim_2/mat_2/')
    os.makedirs('data_for_plot/dim_3/mat_0/')
    os.makedirs('data_for_plot/dim_3/mat_2/')

Ns = {'2': [81,243],#[45,81,243,729,1215]
      '3': [27,81]}
pickle.dump(Ns, open("data_for_plot/Ns.p", "wb"))

kinds = {'2': [0],
         '3': [1,2],
         }
pickle.dump(kinds, open("data_for_plot/kinds.p", "wb"))

material_list = [0,2]
pickle.dump(material_list, open("data_for_plot/material_list.p", "wb"))

sol_rank_range_set={'2': [1,5,10,20,30],#[1,5,10,20,30]
                    '3': [1,5,10,20]}
pickle.dump(sol_rank_range_set, open("data_for_plot/sol_rank_range_set.p", "wb"))

kind_list = ['cano', 'tucker', 'tt']
pickle.dump(kind_list, open("data_for_plot/kind_list.p", "wb"))

for dim in [2,3]:
    for kind in kinds['{}'.format(dim)]:
        for material in material_list:
            time_Ga = list()
            time_GaNi = list()

            time_Ga_Spar = [[] for x in range(len(sol_rank_range_set['{}'.format(dim)]))]
            time_GaNi_Spar = [[] for x in range(len(sol_rank_range_set['{}'.format(dim)]))]

            for grid in range(len(Ns['{}'.format(dim)])):
                N = Ns['{}'.format(dim)][grid]
                ## parameters for non-sparse solution
                pars = Struct(dim=dim,  # number of dimensions (works for 2D and 3D)
                              N=dim * (N,),  # number of voxels (assumed equal for all directions)
                              Y=np.ones(dim),  # size of periodic cell
                              recover_sparse=0,  # recalculate full material coefficients from sparse one
                              solver=dict(tol=1e-10,
                                          maxiter=1,
                                          method='mr'),
                              material=material,
                              )

                ## parameters for SPARSE solution
                pars_sparse = pars.copy()
                pars_sparse.update(Struct(kind=kind_list[kind],  # type of sparse tensor: 'cano', 'tucker', or 'tt'
                                          rank=1,  # rank of solution vector
                                          precond_rank=1,
                                          tol=None,
                                          solver=dict(method='mr',
                                                      # method could be 'Richardson'(r),'minimal_residual'(mr), or 'Chebyshev'(c)
                                                      approx_omega=False,  # inner product of tuckers could be slow
                                                      eigrange=[0.6, 50],
                                                      tol=1e-10,
                                                      maxiter=1,
                                                      divcrit=False),  # no. of iterations for a solver
                                          material=material
                                          ))
                print('== format={}, N={}, dim={}, material={} ===='.format(pars_sparse.kind,
                                                                            N, dim, material))

                ### get material settings for experiment
                pars, pars_sparse, mat_conf = getMat_conf( material, pars, pars_sparse)

                #if not os.path.exists('data_for_plot/dim_{}/mat_{}/mat_kind{}_N{}.p'.format(dim, material, kind, N)):

                mat = Material(mat_conf)
                mats = SparseMaterial(mat_conf, pars_sparse.kind)

                Aga = mat.get_A_Ga(pars.Nbar(pars.N), primaldual='primal')
                Agas = mats.get_A_Ga(pars_sparse.Nbar(pars_sparse.N), primaldual='primal',
                                     k=pars_sparse.matrank).set_fft_form()

                Agani = mat.get_A_GaNi(pars.N, primaldual='primal')
                Aganis = mats.get_A_GaNi(pars_sparse.N, primaldual='primal', k=pars_sparse.matrank)


                Aga.val = recover_Aga(Aga, Agas)
                Agani.val = recover_Agani(Agani, Aganis)
                pars_sparse.update(Struct(alpha=0.5 * (Agani[0, 0].min() + Agani[0, 0].max())))

                ### COMPUTING FULL SOLUTION ###
                ## compute full solutions ##
                resP_Ga = homog_Ga_full_potential(Aga, pars)
                resP_GaNi = homog_GaNi_full_potential(Agani, Aga, pars)

                time_Ga.append(resP_Ga.info['timeAx'][0])
                time_GaNi.append(resP_GaNi.info['timeAx'][0])

                pickle.dump(time_Ga,   open("data_for_plot/dim_{}/mat_{}/time_GaNew_{}.p".format(dim, material, kind), "wb"))
                pickle.dump(time_GaNi, open("data_for_plot/dim_{}/mat_{}/time_GaNiNew_{}.p".format(dim, material, kind), "wb"))

                ############ SPARSE SOLUTIONS ###############

                for sol_rank_i in range(0,len(sol_rank_range_set['{}'.format(dim)])):
                    pars_sparse.update(Struct(rank=sol_rank_range_set['{}'.format(dim)][sol_rank_i]))

                    time_Ga_SparSample=[]
                    time_GaNi_SparSample=[]

                    for sample in range(0, 5):
                        resS_Ga = homog_Ga_sparse(Agas, pars_sparse)
                        time_Ga_SparSample.append(resS_Ga.solver['timeAx'][0][0])

                        resS_GaNi = homog_GaNi_sparse(Aganis, Agas, pars_sparse)
                        time_GaNi_SparSample.append(resS_GaNi.solver['timeAx'][0][0])


                    time_Ga_Spar[sol_rank_i].append(np.mean(time_Ga_SparSample))
                    time_GaNi_Spar[sol_rank_i].append(np.mean(time_GaNi_SparSample))

                    print('         Done rank {}'.format(sol_rank_range_set['{}'.format(dim)][sol_rank_i]))
                print('             Done N    {}'.format(N))

            pickle.dump( time_Ga_Spar   , open("data_for_plot/dim_{}/mat_{}/time_Ga_SparNew_{}.p" .format(dim,material,kind), "wb"))
            pickle.dump( time_GaNi_Spar , open("data_for_plot/dim_{}/mat_{}/time_GaNi_SparNew_{}.p".format(dim, material, kind), "wb"))