import itertools
import logging
import os
import pickle

from examples.lowRankTensorApproximations.setting import (
    get_default_parameters,
    get_material_coef,
    kind_list,
)
from ffthompy.tensorsLowRank.homogenisation import (
    homog_Ga_full_potential,
    homog_Ga_sparse,
    homog_GaNi_full_potential,
    homog_GaNi_sparse,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")



logger.info('running time-comparison for stochastic material...')

kinds = {'2': 0,
         '3': 2,}

N_lists = {'2': [45, 135, 320, 405, 640, 1215, 2560,3645,5120],
           '3': [5, 15, 45, 80, 135, 175, 225, 305, 375]}

err_tol_list=[1e-4, 1e-6]
method=1 # 0-Ga, 1-GaNi

data_folder = "data_for_plot/time"

for dim, material in itertools.product([2, 3], [2, 4]):
    if not os.path.exists('{}/dim_{}/mat_{}/'.format(data_folder,dim, material)):
        os.makedirs('{}/dim_{}/mat_{}/'.format(data_folder,dim, material))

    N_list = N_lists['{}'.format(dim)]
    kind=kinds['{}'.format(dim)]

    full_time_list = [None]*len(N_list)
    sparse_time_list = [[None]*len(N_list), [None]*len(N_list)]
    rank_list = [[None]*len(N_list), [None]*len(N_list)]
    memory_list = [[None]*len(N_list), [None]*len(N_list)]

    for i, N in enumerate(N_list):
        # PARAMETERS ##############################################################
        pars, pars_sparse=get_default_parameters(dim, N, material, kind)
        pars.solver.update(dict(tol=1e-6))

        # generating material coefficients
        if method in ['Ga',0]:
            Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse)
            logger.info('\n== Full solution with potential by CG (Ga) ===========')
            resP_Ga=homog_Ga_full_potential(Aga, pars)
            logger.info('mean of solution=%s', resP_Ga.Fu.mean())
            logger.info('homogenised properties (component 11) = %s', resP_Ga.AH)
            full_time_list[i]=resP_Ga.time
        elif method in ['GaNi',1]:
            Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse, ga=False)
            logger.info('\n== Full solution with potential by CG (GaNi)===========')
            resP=homog_GaNi_full_potential(Agani, Aga, pars)
            logger.info('mean of solution=%s', resP.Fu.mean())
            logger.info('homogenised properties (component 11) = %s', resP.AH)
        else:
            raise ValueError()

        full_time_list[i]=resP.time

        for counter, err_tol in enumerate(err_tol_list):

            for r in range(4, N+1, 2):
                pars_sparse.solver.update(dict(rank=r)) # rank of solution vector

                logger.info('\n== format=%s, N=%s, dim=%s, material=%s, rank=%s, err_tol=%s ====', pars_sparse.kind, N, dim, material, pars_sparse.solver['rank'], err_tol)

                # PROBLEM DEFINITION ######################################################
                if method in ['Ga',0]:
                    logger.info('\n== SPARSE solver with preconditioner (Ga) =======================')
                    resS=homog_Ga_sparse(Agas, pars_sparse)
                    logger.info('mean of solution=%s', resS.Fu.mean())
                    logger.info('homogenised properties (component 11) = %s', resS.AH)
                    logger.info('norm(resP)=%s', resS.solver['norm_res'])
                elif method in ['GaNi',1]:
                    logger.info('\n== SPARSE solver with preconditioner (GaNi) =======================')
                    resS=homog_GaNi_sparse(Aganis, Agas, pars_sparse)
                    logger.info('mean of solution=%s', resS.Fu.mean())
                    logger.info('homogenised properties (component 11) = %s', resS.AH)
                    logger.info('iterations=%s', resS.solver['kit'])
                    logger.info('norm(resP)=%s', resS.solver['norm_res'])
                    logger.info('memory efficiency = %s/%s = %s', resS.Fu.memory, resP.Fu.val.size, resS.Fu.memory/resP.Fu.val.size)
                    logger.info('solution discrepancy %s', (resS.AH - resP.AH)/resP.AH)

                if (resS.AH - resP.AH)/resP.AH <= err_tol:
                    rank_list[counter][i]=r
                    sparse_time_list[counter][i]=resS.time
                    memory_list[counter][i]=resS.Fu.memory/resP.Fu.val.size # memory efficiency
                    logger.info('tensorsLowRank solver time: %s', sparse_time_list[counter])
                    logger.info('full solver time: %s', full_time_list)
                    logger.info('rank: %s', rank_list[counter])
                    break

    logger.info('tensorsLowRank solver time: %s', sparse_time_list)
    logger.info('full solver time: %s', full_time_list)
    logger.info('rank: %s', rank_list)

    pickle.dump(N_list, open("{}/dim_{}/mat_{}/N_list_{}.p"
                             .format(data_folder,dim, material,kind_list[kind]), "wb"))
    pickle.dump(full_time_list, open("{}/dim_{}/mat_{}/full_time_list_{}.p"
                                     .format(data_folder,dim, material,kind_list[kind]), "wb"))
    pickle.dump(sparse_time_list[0], open(("{}/dim_{}/mat_{}/sparse_time_list_{}_"+"{:.0e}".format(err_tol_list[0])+'.p')
                                           .format(data_folder,dim, material,kind_list[kind]), "wb"))
    pickle.dump(sparse_time_list[1], open(("{}/dim_{}/mat_{}/sparse_time_list_{}_"+"{:.0e}".format(err_tol_list[1])+'.p')
                                           .format(data_folder,dim, material,kind_list[kind]), "wb"))
    pickle.dump(rank_list[0], open(("{}/dim_{}/mat_{}/rank_list_{}_"+"{:.0e}".format(err_tol_list[0])+'.p')
                                   .format(data_folder,dim, material,kind_list[kind]), "wb"))
    pickle.dump(rank_list[1], open(("{}/dim_{}/mat_{}/rank_list_{}_"+"{:.0e}".format(err_tol_list[1])+'.p')
                                   .format(data_folder,dim, material,kind_list[kind]), "wb"))

logger.info('END')
