

import numpy as np
import os
import pickle

from ffthompy import Struct
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential, homog_GaNi_full_potential,
                                            homog_Ga_sparse, homog_GaNi_sparse)
from examples.sparse.material_setting import get_material_coef
from examples.sparse.plots import plot_error, plot_memory, plot_residuals

os.nice(19)
#######################################################

Ns = {'2': [45, 135, 405, 1215],
      '3': [15, 25, 55, 135]}

kinds = {'2': [0,2],
         '3': [1,2],}

material_list = [0,2]

sol_rank_range_set={'2': [1,5,10,20,30],
                    '3': [1,5,10,20]}

kind_list = ['cano', 'tucker', 'tt']

if not os.path.exists('data_for_plot'):
    os.makedirs('data_for_plot/')
    os.makedirs('data_for_plot/dim_2/mat_2/')
    os.makedirs('data_for_plot/dim_3/mat_0/')
    os.makedirs('data_for_plot/dim_3/mat_2/')

pickle.dump(kind_list, open("data_for_plot/kind_list.p", "wb"))
pickle.dump(Ns, open("data_for_plot/Ns.p", "wb"))
pickle.dump(kinds, open("data_for_plot/kinds.p", "wb"))
pickle.dump(sol_rank_range_set, open("data_for_plot/sol_rank_range_set.p", "wb"))
pickle.dump(material_list, open("data_for_plot/material_list.p", "wb"))

for dim in [2,3]:
    for grid in range(len(Ns['{}'.format(dim)])):
        N = Ns['{}'.format(dim)][grid]

        for material in material_list:
            if not os.path.exists('data_for_plot/dim_{}/mat_{}/'.format(dim, material)):
                os.makedirs('data_for_plot/dim_{}/mat_{}/'.format(dim, material))
            for kind in kinds['{}'.format(dim)]:

                ################ MATERAL DATA AND SETTINGS ################
                ## parameters for non-sparse solution
                pars = Struct(dim=dim,  # number of dimensions (works for 2D and 3D)
                              N=dim * (N,),  # number of voxels (assumed equal for all directions)
                              Y=np.ones(dim),  # size of periodic cell
                              recover_sparse=1,  # recalculate full material coefficients from sparse one
                              solver=dict(tol=1e-10,
                                          maxiter=20,
                                          method='mr'),
                              material=material,
                              )

                ## parameters for SPARSE solution
                pars_sparse = pars.copy()
                pars_sparse.update(Struct(kind=kind_list[kind],  # type of sparse tensor: 'cano', 'tucker', or 'tt'
                                          rank=1,  # rank of solution vector
                                          precond_rank=15,
                                          tol=None,
                                          solver=dict(method='mr',
                                                      # method could be 'Richardson'(r),'minimal_residual'(mr), or 'Chebyshev'(c)
                                                      approx_omega=False,  # inner product of tuckers could be slow
                                                      eigrange=[0.6, 50],
                                                      tol=1e-10,
                                                      maxiter=20,
                                                      divcrit=False),  # no. of iterations for a solver
                                          material=material
                                          ))
                print('== format={}, N={}, dim={}, material={} ===='.format(pars_sparse.kind,
                                                                            N, dim, material))

                # get material settings for experiment
                Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse)

                pars_sparse.update(Struct(alpha=0.5 * (Agani[0, 0].min() + Agani[0, 0].max())))
                #######################################################################

                ### COMPUTING FULL SOLUTION ###
                sols_Ga = list()
                iter_Ga = list()
                time_Ga = list()

                sols_GaNi = list()
                iter_GaNi = list()
                time_GaNi = list()
                mem_GaNi = list()

                ## Compute Full solutions
                resP_Ga = homog_Ga_full_potential(Aga, pars)
                resP_GaNi = homog_GaNi_full_potential(Agani, Aga, pars)

                ############ SPARSE SOLUTIONS ###############
                sols_Ga_Spar = list()
                time_Ga_Spar = list()
                iter_Ga_Spar = list()
                mem_Ga_Spar = list()
                res_Ga_Spar = list()

                sols_GaNi_Spar = list()
                time_GaNi_Spar = list()
                iter_GaNi_Spar = list()
                mem_GaNi_Spar = list()
                res_GaNi_Spar = list()

                for sol_rank in sol_rank_range_set['{}'.format(dim)]: # rank of solution vector
                    sols_Ga.append(resP_Ga.AH)
                    iter_Ga.append(resP_Ga.info['kit'])
                    time_Ga.append(resP_Ga.info['time'][0])

                    sols_GaNi.append(resP_GaNi.AH)
                    iter_GaNi.append(resP_GaNi.info['kit'])
                    time_GaNi.append(resP_GaNi.info['time'][0])
                    mem_GaNi.append([1])

                    pars_sparse.update(Struct(rank=sol_rank))
                    pars_sparse.update(Struct(precond_rank=sol_rank))

                    resS_Ga = homog_Ga_sparse(Agas, pars_sparse)

                    res_Ga_Spar.append(resS_Ga.solver['norm_res'])
                    sols_Ga_Spar.append(resS_Ga.AH)
                    time_Ga_Spar.append(resS_Ga.time)
                    iter_Ga_Spar.append(resS_Ga.solver['kit'])
                    mem_Ga_Spar.append (resS_Ga.Fu.memory / resP_Ga.Fu.val.size)

                    resS_GaNi = homog_GaNi_sparse(Aganis, Agas, pars_sparse)

                    res_GaNi_Spar.append( resS_GaNi.solver['norm_res'])
                    sols_GaNi_Spar.append(resS_GaNi.AH)
                    time_GaNi_Spar.append(resS_GaNi.time)
                    iter_GaNi_Spar.append(resS_GaNi.solver['kit'])
                    mem_GaNi_Spar.append (resS_GaNi.Fu.memory/resP_GaNi.Fu.val.size)

                pickle.dump( res_Ga_Spar ,   open( "data_for_plot/dim_{}/mat_{}/res_Ga_Spar_{}_{}_{}.p" .format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )
                pickle.dump( sols_Ga_Spar ,  open( "data_for_plot/dim_{}/mat_{}/sols_Ga_Spar_{}_{}_{}.p".format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )
                pickle.dump( time_Ga_Spar ,  open( "data_for_plot/dim_{}/mat_{}/time_Ga_Spar_{}_{}_{}.p".format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )
                pickle.dump( iter_Ga_Spar,   open( "data_for_plot/dim_{}/mat_{}/iter_Ga_Spar_{}_{}_{}.p".format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )
                pickle.dump( mem_Ga_Spar ,   open( "data_for_plot/dim_{}/mat_{}/mem_Ga_Spar_{}_{}_{}.p" .format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )

                pickle.dump( res_GaNi_Spar , open( "data_for_plot/dim_{}/mat_{}/res_GaNi_Spar_{}_{}_{}.p" .format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )
                pickle.dump( sols_GaNi_Spar ,open( "data_for_plot/dim_{}/mat_{}/sols_GaNi_Spar_{}_{}_{}.p".format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )
                pickle.dump( time_GaNi_Spar ,open( "data_for_plot/dim_{}/mat_{}/time_GaNi_Spar_{}_{}_{}.p".format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )
                pickle.dump( iter_GaNi_Spar, open( "data_for_plot/dim_{}/mat_{}/iter_GaNi_Spar_{}_{}_{}.p".format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )
                pickle.dump( mem_GaNi_Spar , open( "data_for_plot/dim_{}/mat_{}/mem_GaNi_Spar_{}_{}_{}.p" .format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )


                pickle.dump( sols_Ga,   open( "data_for_plot/dim_{}/mat_{}/sols_Ga_{}.p".format(dim, material,N) , "wb"))
                pickle.dump( iter_Ga,   open( "data_for_plot/dim_{}/mat_{}/iter_Ga_{}.p".format(dim, material,N), "wb"))
                pickle.dump( time_Ga,   open( "data_for_plot/dim_{}/mat_{}/time_Ga_{}.p".format(dim, material,N), "wb"))

                pickle.dump( sols_GaNi, open("data_for_plot/dim_{}/mat_{}/sols_GaNi_{}.p".format(dim, material,N), "wb"))
                pickle.dump( iter_GaNi, open("data_for_plot/dim_{}/mat_{}/iter_GaNi_{}.p".format(dim, material,N), "wb"))
                pickle.dump( time_GaNi, open("data_for_plot/dim_{}/mat_{}/time_GaNi_{}.p".format(dim, material,N), "wb"))
                pickle.dump( mem_GaNi,  open("data_for_plot/dim_{}/mat_{}/mem_GaNi_{}.p".format(dim, material,N) , "wb"))


plot_error()
plot_memory()
plot_residuals()