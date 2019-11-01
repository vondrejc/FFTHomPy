import numpy as np
import matplotlib.pylab as plt
import os
import pickle

from ffthompy import Timer, Struct
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential,
                                            homog_Ga_sparse,)
from examples.sparse.material_setting import get_material_coef
from examples.sparse.plots import plot_time

k_list={'2': [2,3,4,5],
        '3': [4,3,2,1]}

kinds = {'2': 0,
         '3': 2,}

W_list = {'2': [45,135,405,1215],
          '3': [5,15,45,135,175]}

kind_list=['cano','tucker','tt']
material=0 # or 3

for dim in [2,3]:
    if not os.path.exists('data_for_plot/dim_{}/mat_{}/'.format(dim, material)):
        os.makedirs('data_for_plot/dim_{}/mat_{}/'.format(dim, material))

    if dim==2:
        N_list = W_list['{}'.format(dim)]
    else:
        N_list = W_list['{}'.format(dim)]

    kind =kinds['{}'.format(dim)]

    full_time_list = [None]*len(N_list)
    sparse_time_list = [None]*len(N_list)
    rank_list = [None]*len(N_list)
    memory_list = [None]*len(N_list)

    for i, N in enumerate(N_list):
        # PARAMETERS ##############################################################
        pars=Struct(dim=dim, # number of dimensions (works for 2D and 3D)
                    N=dim*(N,), # number of voxels (assumed equal for all directions)
                    Y=np.ones(dim), # size of periodic cell
                    recover_sparse=1, # recalculate full material coefficients from sparse one
                    solver=dict(tol=1e-6,
                                maxiter=50),
                    )

        pars_sparse = pars.copy()
        pars_sparse.update(Struct(kind=kind_list[kind],  # type of sparse tensor: 'cano', 'tucker', or 'tt'
                                  rank=1,  # rank of solution vector
                                  precond_rank=10,
                                  tol=None,
                                  N=dim*(N,),
                                  solver=dict(method='mr',
                                              # method could be 'Richardson'(r),'minimal_residual'(mr), or 'Chebyshev'(c)
                                              approx_omega=False,  # inner product of tuckers could be so slow
                                              # that using an approximate omega could gain.
                                              tol=1e-6,
                                              maxiter=30,  # no. of iterations for a solver
                                              divcrit=True),
                                  ))

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

        for r in range(5, N+1,5):
            pars_sparse.update(Struct(rank=r)) # rank of solution vector

            print('== format={}, N={}, dim={}, material={} ===='.format(pars_sparse.kind,
                                                                        N, dim, material))

            # PROBLEM DEFINITION ######################################################
            # generating material coefficients
            Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse)

            print('\n== SPARSE solver with preconditioner (Ga) =======================')
            resS_Ga=homog_Ga_sparse(Agas, pars_sparse)
            print('mean of solution={}'.format(resS_Ga.Fu.mean()))
            print('homogenised properties (component 11) = {}'.format(resS_Ga.AH))
            print(resS_Ga.Fu)
            print('norm(resP)={}'.format(resS_Ga.solver['norm_res']))
            print('memory efficiency = {0}/{1} = {2}'.format(resS_Ga.Fu.memory, resP_Ga.Fu.val.size, resS_Ga.Fu.memory/resP_Ga.Fu.val.size))
            print ("solution discrepancy", resS_Ga.AH - resP_Ga.AH)

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

    pickle.dump(N_list, open("data_for_plot/dim_{}/mat_{}/N_list_{}.p".format(dim, material,kind_list[kind]), "wb"))
    pickle.dump(full_time_list, open("data_for_plot/dim_{}/mat_{}/full_time_list_{}.p".format(dim, material,kind_list[kind]), "wb"))
    pickle.dump(sparse_time_list, open("data_for_plot/dim_{}/mat_{}/sparse_time_list_{}.p".format(dim, material,kind_list[kind]), "wb"))

##### plot results ##############
plot_time()
