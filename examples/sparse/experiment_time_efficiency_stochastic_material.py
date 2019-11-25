import os
import pickle

from ffthompy import Struct
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential,
                                            homog_GaNi_full_potential,
                                            homog_Ga_sparse,
                                            homog_GaNi_sparse)
from examples.sparse.setting import get_material_coef, kind_list, get_default_parameters
from examples.sparse.plots import plot_time

kinds = {'2': 0,
         '3': 2,}

N_lists = {'2': [45, 135, 320, 405, 640, 1215, 2560,3645,5120],
           '3': [5, 15, 45, 80, 135, 175, 225, 305, 375 ]}

#N_lists = {'2': [5, 15 ,25],
#           '3': [5, 15, 25]}

err_tol_list=[ 1e-4, 1e-6 ]
method=1 # 0-Ga, 1-GaNi

data_folder = "data_for_plot/time"

for material in [2, 4]:
    for dim in [2, 3]:

        if not os.path.exists('{}/dim_{}/mat_{}/'.format(data_folder,dim, material)):
            os.makedirs('{}/dim_{}/mat_{}/'.format(data_folder,dim, material))

        N_list = N_lists['{}'.format(dim)]
        kind=kinds['{}'.format(dim)]

        full_time_list = [None]*len(N_list)
        sparse_time_list = [[None]*len(N_list), [None]*len(N_list)]
        rank_list =  [[None]*len(N_list), [None]*len(N_list)]
        memory_list =  [[None]*len(N_list), [None]*len(N_list)]

        for i, N in enumerate(N_list):
            # PARAMETERS ##############################################################
            pars, pars_sparse=get_default_parameters(dim, N, material, kind)
            pars.solver.update(dict(tol=1e-6))

            # generating material coefficients
            if method in ['Ga',0]:
                Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse)
                print('\n== Full solution with potential by CG (Ga) ===========')
                resP_Ga=homog_Ga_full_potential(Aga, pars)
                print('mean of solution={}'.format(resP_Ga.Fu.mean()))
                print('homogenised properties (component 11) = {}'.format(resP_Ga.AH))
                full_time_list[i]=resP_Ga.time
            elif method in ['GaNi',1]:
                Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse, ga=False)
                print('\n== Full solution with potential by CG (GaNi)===========')
                resP=homog_GaNi_full_potential(Agani, Aga, pars)
                print('mean of solution={}'.format(resP.Fu.mean()))
                print('homogenised properties (component 11) = {}'.format(resP.AH))
            else:
                raise ValueError()

            full_time_list[i]=resP.time


            for counter, err_tol in enumerate(err_tol_list):

                for r in range(4, N+1, 2):
                    pars_sparse.update(Struct(rank=r)) # rank of solution vector

                    print('\n== format={}, N={}, dim={}, material={}, rank={}, err_tol={} ===='.format(pars_sparse.kind,
                        N, dim, material, pars_sparse.rank, err_tol))

                    # PROBLEM DEFINITION ######################################################
                    if method in ['Ga',0]:
                        print('\n== SPARSE solver with preconditioner (Ga) =======================')
                        resS=homog_Ga_sparse(Agas, pars_sparse)
                        print('mean of solution={}'.format(resS.Fu.mean()))
                        print('homogenised properties (component 11) = {}'.format(resS.AH))
                        print('norm(resP)={}'.format(resS.solver['norm_res']))
                    elif method in ['GaNi',1]:
                        print('\n== SPARSE solver with preconditioner (GaNi) =======================')
                        resS=homog_GaNi_sparse(Aganis, Agas, pars_sparse)
                        print('mean of solution={}'.format(resS.Fu.mean()))
                        print('homogenised properties (component 11) = {}'.format(resS.AH))
                        print('iterations={}'.format(resS.solver['kit']))
                        print('norm(resP)={}'.format(resS.solver['norm_res']))
                        print('memory efficiency = {0}/{1} = {2}'.format(resS.Fu.memory, resP.Fu.val.size, resS.Fu.memory/resP.Fu.val.size))
                        print("solution discrepancy", (resS.AH - resP.AH)/resP.AH)

                    if (resS.AH - resP.AH)/resP.AH <= err_tol:
                        rank_list[counter][i]=r
                        sparse_time_list[counter][i]=resS.time
                        memory_list[counter][i]=resS.Fu.memory/resP.Fu.val.size # memory efficiency
                        print("sparse solver time:",sparse_time_list[counter])
                        print("full solver time:",full_time_list)
                        print("rank:",rank_list[counter] )
                        break

        print("sparse solver time:",sparse_time_list)
        print("full solver time:",full_time_list)
        print("rank:",rank_list)




        pickle.dump(N_list, open("{}/dim_{}/mat_{}/N_list_{}.p".format(data_folder,dim, material,kind_list[kind]), "wb"))
        pickle.dump(full_time_list, open("{}/dim_{}/mat_{}/full_time_list_{}.p".format(data_folder,dim, material,kind_list[kind]), "wb"))
        pickle.dump(sparse_time_list[0], open(("{}/dim_{}/mat_{}/sparse_time_list_{}_"+"{:.0e}".format(err_tol_list[0])+'.p').format(data_folder,dim, material,kind_list[kind]), "wb"))
        pickle.dump(sparse_time_list[1], open(("{}/dim_{}/mat_{}/sparse_time_list_{}_"+"{:.0e}".format(err_tol_list[1])+'.p').format(data_folder,dim, material,kind_list[kind]), "wb"))
        pickle.dump(rank_list[0],        open(("{}/dim_{}/mat_{}/rank_list_{}_"+"{:.0e}".format(err_tol_list[0])+'.p').format(data_folder,dim, material,kind_list[kind]), "wb"))
        pickle.dump(rank_list[1],        open(("{}/dim_{}/mat_{}/rank_list_{}_"+"{:.0e}".format(err_tol_list[1])+'.p').format(data_folder,dim, material,kind_list[kind]), "wb"))

##### plot results ##############
#plot_time()
