import os
import pickle

from ffthompy.tensorsLowRank.homogenisation import (homog_Ga_full_potential, homog_GaNi_full_potential,
                                                    homog_Ga_sparse, homog_GaNi_sparse)
from examples.lowRankTensorApproximations.setting import get_material_coef, kind_list, get_default_parameters
from examples.lowRankTensorApproximations.plots import plot_residuals, save_experiment_settings

os.nice(19)
#######################################################

Ns = {'2': [1215],
      '3': [135]}

kinds = {'2': [0],
         '3': [1,2]}

material_list = [0,2]

sol_rank_range_set={'2': [2,5,10,20,30],
                    '3': [2,5,10,20]}

data_folder = "data_for_plot/residua"

save_experiment_settings(kind_list, Ns, kinds, sol_rank_range_set, material_list,
                         data_folder=data_folder)

for dim in [2,3]:
    for grid in range(len(Ns['{}'.format(dim)])):
        N = Ns['{}'.format(dim)][grid]

        for material in material_list:
            if not os.path.exists('{}/dim_{}/mat_{}/'.format(data_folder,dim, material)):
                os.makedirs('{}/dim_{}/mat_{}/'.format(data_folder,dim, material))
            for kind in kinds['{}'.format(dim)]:

                ################ MATERAL DATA AND SETTINGS ################
                ## parameters
                pars, pars_sparse=get_default_parameters(dim, N, material, kind)
                pars_sparse.solver.update(dict(rank=1,  # rank of solution vector
                                               minres_fails=int(1e6), # obtain info from all iters
                                               ))

                print('== format={}, N={}, dim={}, material={} ===='.format(pars_sparse.kind,
                                                                            N, dim, material))

                # get material settings for experiment
                Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse)

                #######################################################################

                ### COMPUTING FULL SOLUTION ###
                iter_Ga = list()
                iter_GaNi = list()

                ## Compute Full solutions
                resP_Ga = homog_Ga_full_potential(Aga, pars)
                resP_GaNi = homog_GaNi_full_potential(Agani, Aga, pars)

                ############ SPARSE SOLUTIONS ###############
                iter_Ga_Spar = list()
                res_Ga_Spar = list()

                iter_GaNi_Spar = list()
                res_GaNi_Spar = list()

                for sol_rank in sol_rank_range_set['{}'.format(dim)]: # rank of solution vector
                    pars_sparse.solver.update(dict(rank=sol_rank))

                    iter_Ga.append(resP_Ga.info['kit'])
                    iter_GaNi.append(resP_GaNi.info['kit'])

                    resS_Ga = homog_Ga_sparse(Agas, pars_sparse)
                    res_Ga_Spar.append(resS_Ga.solver['norm_res'])
                    iter_Ga_Spar.append(resS_Ga.solver['kit'])

                    resS_GaNi = homog_GaNi_sparse(Aganis, Agas, pars_sparse)
                    res_GaNi_Spar.append(resS_GaNi.solver['norm_res'])
                    iter_GaNi_Spar.append(resS_GaNi.solver['kit'])

                pickle.dump(res_Ga_Spar, open("{}/dim_{}/mat_{}/res_Ga_Spar_{}_{}_{}.p"
                                              .format(data_folder, dim, material, kind, N,
                                                      pars_sparse.solver['method']), "wb"))
                pickle.dump(iter_Ga_Spar, open("{}/dim_{}/mat_{}/iter_Ga_Spar_{}_{}_{}.p"
                                               .format(data_folder, dim, material, kind, N,
                                                       pars_sparse.solver['method']), "wb"))

                pickle.dump(res_GaNi_Spar, open("{}/dim_{}/mat_{}/res_GaNi_Spar_{}_{}_{}.p"
                                                .format(data_folder, dim, material, kind, N,
                                                        pars_sparse.solver['method']), "wb"))
                pickle.dump(iter_GaNi_Spar, open("{}/dim_{}/mat_{}/iter_GaNi_Spar_{}_{}_{}.p"
                                                 .format(data_folder, dim, material, kind, N,
                                                         pars_sparse.solver['method']), "wb"))

                pickle.dump(iter_Ga, open("{}/dim_{}/mat_{}/iter_Ga_{}.p"
                                          .format(data_folder, dim, material, N), "wb"))
                pickle.dump(iter_GaNi, open("{}/dim_{}/mat_{}/iter_GaNi_{}.p"
                                            .format(data_folder, dim, material, N), "wb"))

plot_residuals()
