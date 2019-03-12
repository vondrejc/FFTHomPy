import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from fig_pars import set_labels,set_pars

os.nice(19)

material_list = pickle.load(open("data_for_plot/material_list.p", "rb"))
sol_rank_range_set = pickle.load(open("data_for_plot/sol_rank_range_set.p", "rb"))
kinds = pickle.load(open("data_for_plot/kinds.p", "rb"))
Ns = pickle.load(open("data_for_plot/Ns.p", "rb"))
kind_list = pickle.load(open("data_for_plot/kind_list.p", "rb"))
solver ='mr'


def plot_error():
    ylimit = [10 ** -11, 10 ** 0]
    xlabel ='rank of solution'
    ylabel='norm of relative error'
    iter_rank_range_set=[1,5,10,15,20,30,40,50]


    for dim in [2]:
        N = max(Ns['{}'.format(dim)])
        xlimend = max(sol_rank_range_set['{}'.format(dim)])
        if not os.path.exists('figures'):
            os.makedirs('figures')

        ##### BEGIN: figure 1 resiguum(solution rank) ###########
        for material in material_list:
            parf=set_pars(mpl)
            lines,labels=set_labels()
            src='figures/' # source folder\
            plt.figure(num=None, figsize=parf['figsize'], dpi=parf['dpi'])
            plt.ylabel('relative error')
            plt.xlabel('rank of solution')

            sol_rank_range = sol_rank_range_set['{}'.format(dim)]
            i=0
            for kind in kinds['{}'.format(dim)]:

                sols_Ga        =   pickle.load(open( "data_for_plot/dim_{}/mat_{}/sols_Ga_{}.p".format(dim,material,N) , "rb"))
                sols_GaNi      =   pickle.load(open( "data_for_plot/dim_{}/mat_{}/sols_GaNi_{}.p".format(dim, material,N), "rb"))
                sols_Ga_Spar   =   pickle.load(open( "data_for_plot/dim_{}/mat_{}/sols_Ga_Spar_{}_{}_{}.p".format(dim,material,kind,N,solver), "rb"  ) )
                sols_GaNi_Spar =   pickle.load(open( "data_for_plot/dim_{}/mat_{}/sols_GaNi_Spar_{}_{}_{}.p".format(dim,material,kind,N,solver), "rb"  ) )

                plt.hold(True)
                plt.semilogy(sol_rank_range,[abs((sols_Ga_Spar[i]  -sols_Ga[1])  /sols_Ga[1])   for i in range(len(sols_Ga_Spar))], lines['Ga_{}'.format(kind_list[kind])][i],   label=labels['Ga{}'.format(kind_list[kind])] , markevery=1)
                plt.semilogy(sol_rank_range,[abs((sols_GaNi_Spar[i]-sols_GaNi[1])/sols_GaNi[1]) for i in range(len(sols_GaNi_Spar))],lines['GaNi_{}'.format(kind_list[kind])][i], label=labels['GaNi{}'.format(kind_list[kind])], markevery=1, markersize=7,markeredgewidth=1,markerfacecolor='None')
                i=i+1


            ax = plt.gca()
            plt.xlabel(xlabel)
            ax.set_xlim([0,xlimend])
            plt.xticks(sol_rank_range)

            plt.ylabel(ylabel)

            lg=plt.legend(loc='best')
            fname=src+'Error_dim{}_mat{}_{}{}'.format(dim,material,solver,'.pdf')
            print('create figure: {}'.format(fname))
            plt.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')
        print('END plot errors')
        ##### END: figure 1 resiguum(solution rank) ###########


    for dim in [3]:
        N = max(Ns['{}'.format(dim)])
        xlimend = max(sol_rank_range_set['{}'.format(dim)])
        if not os.path.exists('figures'):
            os.makedirs('figures')

        ##### BEGIN: figure 1 resiguum(solution rank) ###########
        for material in material_list:
            parf=set_pars(mpl)
            lines,labels=set_labels()
            src='figures/' # source folder\
            plt.figure(num=None, figsize=parf['figsize'], dpi=parf['dpi'])

            sol_rank_range = sol_rank_range_set['{}'.format(dim)]


            for kind in kinds['{}'.format(dim)]:

                sols_Ga        =   pickle.load(open( "data_for_plot/dim_{}/mat_{}/sols_Ga_{}.p".format(dim,material,N) , "rb"))
                sols_GaNi      =   pickle.load(open( "data_for_plot/dim_{}/mat_{}/sols_GaNi_{}.p".format(dim, material,N), "rb"))
                sols_Ga_Spar   =   pickle.load(open( "data_for_plot/dim_{}/mat_{}/sols_Ga_Spar_{}_{}_{}.p".format(dim,material,kind,N,solver), "rb"  ) )
                sols_GaNi_Spar =   pickle.load(open( "data_for_plot/dim_{}/mat_{}/sols_GaNi_Spar_{}_{}_{}.p".format(dim,material,kind,N,solver), "rb"  ) )

                plt.hold(True)
                plt.semilogy(sol_rank_range,[abs((sols_Ga_Spar[i]  -sols_Ga[1])  /sols_Ga[1])   for i in range(len(sols_Ga_Spar))], lines['Ga_{}'.format(kind_list[kind])][i],   label=labels['Ga{}'.format(kind_list[kind])] , markevery=1)
                plt.semilogy(sol_rank_range,[abs((sols_GaNi_Spar[i]-sols_GaNi[1])/sols_GaNi[1]) for i in range(len(sols_Ga_Spar))], lines['GaNi_{}'.format(kind_list[kind])][i], label=labels['GaNi{}'.format(kind_list[kind])], markevery=1, markersize=7,markeredgewidth=1,markerfacecolor='None')

            ax = plt.gca()
            plt.xlabel(xlabel)
            ax.set_xlim([0,xlimend])
            plt.xticks(sol_rank_range)

            plt.ylabel(ylabel)

            lg=plt.legend(loc='best')
            fname=src+'Error_dim{}_mat{}_{}{}'.format(dim,material,solver,'.pdf')
            print('create figure: {}'.format(fname))
            plt.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')
        print('END plot errors')
        ##### END: figure 1 resiguum(solution rank) ###########






