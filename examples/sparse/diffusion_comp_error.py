

import numpy as np
import os
import pickle

from ffthompy import Struct
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential, homog_GaNi_full_potential,
                                            homog_Ga_sparse, homog_GaNi_sparse)
from examples.sparse.setting import get_material_coef, kind_list, get_default_parameters
from examples.sparse.plots import plot_error

os.nice(19)
#######################################################

Ns = {'2': [55],
      '3': [15]}

kinds = {'2': [0,2],
         '3': [1,2],}

material_list = [0,2]

sol_rank_range_set={'2': [1,5,10,20,30],
                    '3': [1,5,10,20]}

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

for dim in [3]:
    for grid in range(len(Ns['{}'.format(dim)])):
        N = Ns['{}'.format(dim)][grid]

        for material in material_list:
            if not os.path.exists('data_for_plot/dim_{}/mat_{}/'.format(dim, material)):
                os.makedirs('data_for_plot/dim_{}/mat_{}/'.format(dim, material))
            for kind in kinds['{}'.format(dim)]:

                ################ MATERAL DATA AND SETTINGS ################
                ## parameters
                pars, pars_sparse=get_default_parameters(dim, N, material, kind)
                pars_sparse.update(Struct(rank=1,  # rank of solution vector
                                          ))

                print('== format={}, N={}, dim={}, material={} ===='.format(pars_sparse.kind,
                                                                            N, dim, material))

                # get material settings for experiment
                Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse)

                #######################################################################

                ### COMPUTING FULL SOLUTION ###
                sols_Ga = list()
                sols_GaNi = list()

                ## Compute Full solutions
                resP_Ga = homog_Ga_full_potential(Aga, pars)
                resP_GaNi = homog_GaNi_full_potential(Agani, Aga, pars)

                ############ SPARSE SOLUTIONS ###############
                sols_Ga_Spar = list()
                sols_GaNi_Spar = list()

                for sol_rank in sol_rank_range_set['{}'.format(dim)]: # rank of solution vector
                    pars_sparse.update(Struct(rank=sol_rank))

                    sols_Ga.append(resP_Ga.AH)
                    sols_GaNi.append(resP_GaNi.AH)

                    resS_Ga = homog_Ga_sparse(Agas, pars_sparse)
                    sols_Ga_Spar.append(resS_Ga.AH)

                    resS_GaNi = homog_GaNi_sparse(Aganis, Agas, pars_sparse)
                    sols_GaNi_Spar.append(resS_GaNi.AH)

                pickle.dump( sols_Ga_Spar ,  open( "data_for_plot/dim_{}/mat_{}/sols_Ga_Spar_{}_{}_{}.p".format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )
                pickle.dump( sols_GaNi_Spar ,open( "data_for_plot/dim_{}/mat_{}/sols_GaNi_Spar_{}_{}_{}.p".format(dim,material,kind,N,pars_sparse.solver['method']), "wb"  ) )

                pickle.dump( sols_Ga,   open( "data_for_plot/dim_{}/mat_{}/sols_Ga_{}.p".format(dim, material,N) , "wb"))
                pickle.dump( sols_GaNi, open("data_for_plot/dim_{}/mat_{}/sols_GaNi_{}.p".format(dim, material,N), "wb"))

plot_error()
