import os
import pickle

from ffthompy import Struct
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential,
                                            homog_Ga_sparse,)
from examples.sparse.setting import get_material_coef, getMat_conf, get_default_parameters

from ffthompy.sparse.materials import SparseMaterial

kinds = {'2': 0,
         '3': 2,}

N_lists = {'2': [45,135,405,1215],
           '3': [5,15,45,135,175]}

kind_list=['cano','tucker','tt']
material=3 # 0 or 3

data_folder = "data_for_plot/time"

for dim in [2,3]:
    if not os.path.exists('{}/dim_{}/mat_{}/'.format(data_folder,dim, material)):
        os.makedirs('{}/dim_{}/mat_{}/'.format(data_folder,dim, material))

    N_list = N_lists['{}'.format(dim)]

    kind =kinds['{}'.format(dim)]

    full_time_list = [None]*len(N_list)
    sparse_time_list = [None]*len(N_list)
    rank_list = [None]*len(N_list)
    memory_list = [None]*len(N_list)

    for i, N in enumerate(N_list):
        # PARAMETERS ##############################################################
        pars, pars_sparse=get_default_parameters(dim, N, material, kind)
        pars.solver.update(dict(tol=1e-6))

        # generating material coefficients
        Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse)

        print('\n== Full solution with potential by CG (Ga) ===========')
        resP_Ga=homog_Ga_full_potential(Aga, pars)
        print('mean of solution={}'.format(resP_Ga.Fu.mean()))
        print('homogenised properties (component 11) = {}'.format(resP_Ga.AH))
        full_time_list[i]=resP_Ga.time

        # PARAMETERS FOR SPARSE SOLVER s#########################
        alp=3 # multiplier to increase the discretisation grid for sparse solver,
            # which enables to achieve the same level of accuracy as the full solver.
        pars.update(Struct(N=dim*(alp*N,),)) # number of voxels (assumed equal for all directions)
        # ----------------------------

        for r in range(4, N+1, 2):
            pars_sparse.solver.update(dict(rank=r)) # rank of solution vector

            print('== format={}, N={}, dim={}, material={} ===='.format(pars_sparse.kind,
                                                                        N, dim, material))

            # PROBLEM DEFINITION ######################################################
            # generating material coefficients
            pars, pars_sparse, mat_conf=getMat_conf(material, pars, pars_sparse)
            mats=SparseMaterial(mat_conf, pars_sparse.kind)
            Agas=mats.get_A_Ga(pars_sparse.Nbar(pars_sparse.N), primaldual='primal',
                               k=pars_sparse.matrank)

            print('\n== SPARSE solver with preconditioner (Ga) =======================')
            resS_Ga=homog_Ga_sparse(Agas, pars_sparse)
            print('mean of solution={}'.format(resS_Ga.Fu.mean()))
            print('homogenised properties (component 11) = {}'.format(resS_Ga.AH))
            print('norm(resP)={}'.format(resS_Ga.solver['norm_res']))
            print('memory efficiency = {0}/{1} = {2}'.format(resS_Ga.Fu.memory, resP_Ga.Fu.val.size, resS_Ga.Fu.memory/resP_Ga.Fu.val.size))
            print("solution discrepancy", resS_Ga.AH - resP_Ga.AH)

            if resS_Ga.AH - resP_Ga.AH <= 0:
                rank_list[i]=r
                sparse_time_list[i]=resS_Ga.time
                memory_list[i]=resS_Ga.Fu.memory/resP_Ga.Fu.val.size # memory efficiency
                print("sparse solver time:",sparse_time_list)
                print("full solver time:",full_time_list)
                print("rank:",rank_list)
                break

    print("sparse solver time:",sparse_time_list)
    print("full solver time:",full_time_list)
    print("rank:",rank_list)

    pickle.dump(N_list, open("{}/dim_{}/mat_{}/N_list_{}.p".format(data_folder,dim, material,kind_list[kind]), "wb"))
    pickle.dump(full_time_list, open("{}/dim_{}/mat_{}/full_time_list_{}.p".format(data_folder,dim, material,kind_list[kind]), "wb"))
    pickle.dump(sparse_time_list, open("{}/dim_{}/mat_{}/sparse_time_list_{}.p".format(data_folder,dim, material,kind_list[kind]), "wb"))