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

import os
import sys

import matplotlib.pyplot as plt
import pylab
from operator import add
os.nice(19)

import itertools

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()
def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
###############################################################
#   Experiment31 - combination of ex 3 a 1 to 1 picture
#
#######################################################
dim = 3
N = 11
material =2
stoch_mat_rank=10# only for material 2

maxiter=40
Sol_par_range=range(1,20,2)#

lines = ["-","--","-.",":"]
colours = ["k","b","r",":"]
counter = 0
for kind in [1,2]:

    print('%%%%%%%%%%%------------------------------ FORMAT {} -------------------------------------------%%%%%%%%%%'.format(kind))

    Ga_Sols = list()
    GaNi_Sols = list()
    Ga_Spar_Sols = list()
    GaNi_Spar_Sols = list()

    GaNi_mem = list()
    GaNi_Spar_mem = list()

    Ga_iter = list()
    GaNi_iter = list()
    Ga_Spar_iter = list()
    GaNi_Spar_iter = list()

    Ga_time= list()
    GaNi_time= list()
    Ga_Spar_time = list()
    GaNi_Spar_time = list()

    for sol_rank in Sol_par_range:
        print('%%%%%%%%%%%------------------ SOLUTION RANK {} ------------%%%%%%%%%%'.format(sol_rank))

        Ga_Spar_res = list()
        GaNi_Spar_res = list()

         ## parameters for non-sparse solution
        pars = Struct(dim=dim,  # number of dimensions (works for 2D and 3D)
                      N=dim * (N,),  # number of voxels (assumed equal for all directions)
                      Y=np.ones(dim),  # size of periodic cell
                      recover_sparse=1,  # recalculate full material coefficients from sparse one
                      solver=dict(tol=1e-8,
                                  maxiter=1e2),
                      )

        ## parameters for SPARSE solution
        pars_sparse = pars.copy()
        kind_list=['cano','tucker','tt']
        pars_sparse.update(Struct(kind=kind_list[kind],  # type of sparse tensor: 'cano', 'tucker', or 'tt'
                                  #rank=sol_rank,  # rank of solution vector
                                  tol=None,
                                  solver=dict(tol=1e-8,
                                              maxiter=maxiter),  # no. of iterations for a solver
                                  ))

        if dim == 2:
            pars_sparse.update(Struct(N=dim * (1 * N,),
                                      ))
        elif dim == 3:
            pars_sparse.update(Struct(N=dim * (1 * N,), ))

        if pars_sparse.kind in ['tt']:
            pars_sparse.update(Struct(tol=1e-30))

        # auxiliary operator
        Nbar = lambda N: 2 * np.array(N) - 1
        pars_sparse.update(Struct(rank=sol_rank))  # rank of solution vector


        # PROBLEM DEFINITION ######################################################
        if material in [0]:
            mat_conf = {'inclusions': ['square', 'otherwise'],
                        'positions': [0. * np.ones(dim), ''],
                        'params': [0.6 * np.ones(dim), ''],  # size of sides
                        'vals': [10 * np.eye(dim), 1. * np.eye(dim)],
                        'Y': np.ones(dim),
                        'P': pars.N,
                        'order': 0, }
            pars_sparse.update(Struct(matrank=2))

        elif material in [1]:
            mat_conf = {'inclusions': ['pyramid', 'all'],
                        'positions': [0. * np.ones(dim), ''],
                        'params': [0.8 * np.ones(dim), ''],  # size of sides
                        'vals': [10 * np.eye(dim), 1. * np.eye(dim)],
                        'Y': np.ones(dim),
                        'P': pars.N,
                        'order': 1, }
            pars_sparse.update(Struct(matrank=2))


        elif material in [2]: # stochastic material
            pars_sparse.update(Struct(matrank=stoch_mat_rank))
            #if counter ==0:
            kl=KL_Fourier(covfun=2, cov_pars={'rho':0.15, 'sigma': 1.}, N=pars.N, puc_size=pars.Y,
                      transform=lambda x: 1e4*np.exp(x))
            if dim==2:
                kl.calc_modes(relerr=0.1)
            elif dim==3:
                kl.calc_modes(relerr=0.4)
            ip=np.random.random(kl.modes.n_kl)-0.5
            np.set_printoptions(precision=8)
            #print('ip={}\n'.format(ip.__repr__()))

            if dim==2:
                #1
                ip=np.array([ 0.34948959, 0.216714  , 0.00621108, 0.00054696, 0.40259557, 0.06547262, 0.15660297, -0.27448685, 0.00058166, -0.24411106 , 0.37038647, -0.47713441, -0.12263014,  0.49500173, -0.17644054, -0.29632561, -0.29077766, -0.01179184, -0.3775561,  -0.11455433])
                #2
                #ip = np.array([0.24995, 0.009014, -0.004228, 0.266437, 0.345009, -0.29721, -0.291875, -0.125469, 0.495526,-0.452405, -0.333025, 0.208331, 0.045902, -0.441424, -0.274428, -0.243702, -0.146728, 0.239476, 0.404311, 0.214929])
            elif dim==3:
                ip=np.array([ 0.34948959, 0.216714  , 0.43621108, 0.46554696, 0.40259557, 0.06547262, 0.15660297, 0.27448685, 0.00058166, 0.24411106 , 0.37038647, 0.47713441, 0.12263014,  0.49500173, 0.17644054, 0.29632561, 0.29077766, 0.01179184, 0.3775561,  0.11455433,0.01179184, 0.3775561, 0.11455433, 0.01179184, 0.3775561,  0.11455433])
            # ip = np.array([-0.22055864, - 0.42053367, - 0.10792634,  0.26090105, - 0.46503888,  0.20162119, 0.17975708, - 0.48591457, - 0.36331554, 0.47545613, - 0.20418382,  0.4427007, 0.16336059, - 0.24336767, - 0.01019986,  0.23258972, - 0.00332544,  0.4476544,  0.49350152,  0.12309778, 0.28203263,- 0.06681636,  0.19054314, - 0.30960648, - 0.00899614, - 0.49683921])
                #ip = np.array([-0.22055864, - 0.42053367, - 0.10792634,  0.26090105, - 0.46503888,  0.20162119, 0.17975708 ,- 0.48591457, - 0.36331554,  0.47545613, - 0.20418382,  0.4427007,  0.16336059, - 0.24336767, - 0.01019986,  0.23258972, - 0.00332544,  0.4476544,   0.49350152,  0.12309778, 0.28203263, - 0.06681636,  0.19054314, - 0.30960648, - 0.00899614, - 0.49683921])

            def mat_fun(coor):
                val=np.zeros_like(coor[0])
                for ii in range(kl.modes.n_kl):
                    val+=ip[ii]*kl.mode_fun(ii, coor)
                return np.einsum('ij,...->ij...', np.eye(dim), kl.transform(val))

               # counter = 1

            mat_conf={'fun':mat_fun,
                      'Y': np.ones(dim),
                      'P': pars.N,
                      'order': 1, }



        else:
            raise ValueError()

    #########


    ########### generating material coefficients
        mat = Material(mat_conf)
        mats = SparseMaterial(mat_conf, pars_sparse.kind)

        Agani = matrix2tensor(mat.get_A_GaNi(pars.N, primaldual='primal'))
        Aganis = mats.get_A_GaNi(pars_sparse.N, primaldual='primal', k=pars_sparse.matrank)

        Aga = matrix2tensor(mat.get_A_Ga(Nbar(pars.N), primaldual='primal'))
        Agas = mats.get_A_Ga(Nbar(pars_sparse.N), primaldual='primal', k=pars_sparse.matrank)

        if np.array_equal(pars.N, pars_sparse.N):
            print(np.linalg.norm(Agani.val[0, 0] - Aganis.full()))
            print(np.linalg.norm(Aga.val[0, 0] - Agas.full()))

        if pars.recover_sparse:
            print('recovering full material tensors...')
            Agani.val = np.einsum('ij,...->ij...', np.eye(dim), Aganis.full())
            Aga.val = np.einsum('ij,...->ij...', np.eye(dim), Agas.full())

        if np.array_equal(pars.N, pars_sparse.N):
            print(np.linalg.norm(Agani.val[0, 0] - Aganis.full()))
            print(np.linalg.norm(Aga.val[0, 0] - Agas.full()))

        pars_sparse.update(Struct(alpha=0.5 * (Agani[0, 0].min() + Agani[0, 0].max())))
        #pars_sparse.update(Struct(alpha= Agani[0, 0].max()))
        #######################################################################33

        #pars(Struct(solver=alpha=0.5 * (Agani[0, 0].min() + Agani[0, 0].max())))
        pars.update(Struct(solver=dict(tol=1e-8,
                                       maxiter=maxiter,
                                       alpha=0.5 * (Agani[0, 0].min() + Agani[0, 0].max()))))# no. of iterations for a solver

        resP_GaNi = homog_GaNi_full_potential(Agani, Aga, pars)
        resS_GaNi = homog_GaNi_sparse(Aganis, Agas, pars_sparse)
    ####### OPERATORS #### SOLUTIONS ###############################################################
    #print('\n== Full solution with potential by CG (Ga) ===========')

        resP_Ga = homog_Ga_full_potential(Aga, pars)

        Ga_Sols.append(resP_Ga.AH)
        print('homogenised properties (component 11) = {}'.format(resP_Ga.AH))
        Ga_iter.append(resP_Ga.info['kit'])
        Ga_time.append(resP_Ga.info['time'][0])

    #print('\n== Full solution with potential by CG (GaNi)===========')
        resP_GaNi = homog_GaNi_full_potential(Agani, Aga, pars)
        GaNi_Sols.append(resP_GaNi.AH)
        print('homogenised properties (component 11) = {}'.format(resP_GaNi.AH))
        GaNi_mem.append(1)
        GaNi_iter.append(resP_GaNi.info['kit'])
        GaNi_time.append(resP_GaNi.info['time'][0])

    #print('\n== SPARSE Richardson solver with preconditioner (Ga) =======================')
        resS_Ga = homog_Ga_sparse(Agas, pars_sparse)
        Ga_Spar_Sols.append(resS_Ga.AH)
        print('homogenised properties (component 11) = {}'.format(resS_Ga.AH))
        Ga_Spar_iter.append(resS_Ga.solver['kit'])
        Ga_Spar_time.append(resS_Ga.time[0])
        Ga_Spar_res = resS_Ga.solver['norm_res']
    #print('\n== SPARSE Richardson solver with preconditioner (GaNi) =======================')
        resS_GaNi = homog_GaNi_sparse(Aganis, Agas, pars_sparse)
        GaNi_Spar_Sols.append(resS_GaNi.AH)
        print('homogenised properties (component 11) = {}'.format(resS_GaNi.AH))
        GaNi_Spar_mem.append(resS_GaNi.Fu.memory/resP_Ga.Fu.val.size)
        GaNi_Spar_iter.append(resS_GaNi.solver['kit'])
        GaNi_Spar_time.append(resS_GaNi.time[0])

        GaNi_Spar_res = resS_GaNi.solver['norm_res']
        ############### ERORS ###########
        E_Ga_Ga_Spar=[abs((Ga_Spar_Sols[i]-Ga_Sols[i])/Ga_Sols[i]) for i in range(len(Ga_Sols))]
        E_Ga_GaNi_Spar=[abs((GaNi_Spar_Sols[i]-GaNi_Sols[i])/GaNi_Sols[i]) for i in range(len(Ga_Sols))]
######################################### PLOTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        plt.figure(kind + 1)
        plt.subplot(2, 1,1)
        plt.semilogy(range(len(Ga_Spar_res)),Ga_Spar_res, label='Ga_solution_rank-{}'.format(sol_rank))
        plt.hold(True)
        plt.xlabel('Iteration')
        plt.ylabel('Residuum')
        plt.title('Format = {}, Material rank = {}, dim = {}, N = {}'.format(pars_sparse.kind, pars_sparse.matrank, dim , N),fontsize=20)
        legend = plt.legend(loc="upper right",bbox_to_anchor=(1.0, 1.0), framealpha=1)
        plt.subplot(2, 1,2)
        plt.semilogy(range(len(GaNi_Spar_res)),GaNi_Spar_res, label='GaNi_solution_rank- {}'.format(sol_rank),linestyle='--')



    ## Ploting #################

    plt.xlabel('Iteration')
    plt.ylabel('Residuum')
    #plt.title('material={},Mat rank={},dim={},format={}'.format(material,pars_sparse.matrank,dim,pars_sparse.kind))
    legend = plt.legend(loc="upper right",bbox_to_anchor=(1.0, 1.0), framealpha=1)

    #plt.show()

    plt.figure(0)
    plt.subplot(4, 1,1)
    #ax.plot(Sol_par_range,E_Ga_GaNi, label='Er-Ga_GaNi')
    plt.semilogy(Sol_par_range,E_Ga_Ga_Spar,'{}'.format(colours[kind]),label='Rel_Error(Sparse_Ga-Full_Ga)_{}'.format(pars_sparse.kind))
    plt.hold(True)
    plt.semilogy(Sol_par_range,E_Ga_GaNi_Spar,'{}'.format(colours[kind]), label='Rel_Error(Sparse_GaNi-Full_GaNi)_{}'.format(pars_sparse.kind),linestyle='--')

    plt.subplot(4, 1,2)
    plt.plot(Sol_par_range,GaNi_Spar_mem,'{}'.format(colours[kind]), label='Sparse_{}'.format(pars_sparse.kind),linestyle='--')

    plt.subplot(4, 1,3)
    plt.plot(Sol_par_range,Ga_Spar_iter,'{}'.format(colours[kind]),  label='Sparse_Ga_{}'.format(pars_sparse.kind))
    plt.plot(Sol_par_range,GaNi_Spar_iter,'{}'.format(colours[kind]),label='Sparse_GaNi_{}'.format(pars_sparse.kind),linestyle='--')

    plt.subplot(4, 1,4)
    plt.plot(Sol_par_range,list( map(add,list(zip(*Ga_Spar_time))[0],list(zip(*Ga_Spar_time))[1])),'{}'.format(colours[kind]),  label='Sparse_Ga_{}'.format(pars_sparse.kind))
    plt.plot(Sol_par_range,list( map(add,list(zip(*GaNi_Spar_time))[0],list(zip(*GaNi_Spar_time))[1])),'{}'.format(colours[kind]),label='Sparse_GaNi_{}'.format(pars_sparse.kind),linestyle='--')

plt.plot(Sol_par_range,list(zip(*Ga_time))[0],'{}'.format(colours[0]), label='Full_Ga')
plt.plot(Sol_par_range,list(zip(*GaNi_time))[0],'{}'.format(colours[0]), label='Full_GaNi',linestyle='--')

plt.subplot(4, 1,2)
plt.plot(Sol_par_range,GaNi_mem,'{}'.format(colours[0]),label='Full',linestyle='--')

plt.subplot(4, 1,3)
plt.plot(Sol_par_range,Ga_iter,'{}'.format(colours[0]),label='Full_Ga')
plt.plot(Sol_par_range,GaNi_iter,'{}'.format(colours[0]),label='Full_GaNi',linestyle='--')


################# PLOT OPTIONS ######################

plt.subplot(4, 1, 1)
#plt.title('N={},material={},Mat rank={},dim={}'.format(N,material,pars_sparse.matrank,dim), fontsize=20)
plt.title(' Material rank = {}, dim = {}, N = {}'.format( pars_sparse.matrank, dim , N),fontsize=20)
plt.xlabel('Rank of solution')
plt.ylabel('Error')
#plt.legend(bbox_to_anchor=(1.1, 1.05))
#plt.yticks(np.arange(0, 1e-5))
legend = plt.legend(loc="upper right",bbox_to_anchor=(1.0, 1.0), framealpha=1)

plt.subplot(4, 1, 2)
plt.xlabel('Rank of solution')
plt.ylabel('Memory efficiency')
legend = plt.legend(loc="upper right",bbox_to_anchor=(1.0, 1.0), framealpha=1)

plt.subplot(4, 1, 3)
plt.xlabel('Rank of solution')
plt.ylabel('Number of iterations')
legend = plt.legend(loc="upper right",bbox_to_anchor=(1.0, 1.0), framealpha=1)

plt.subplot(4, 1, 4)
plt.xlabel('Rank of solution')
plt.ylabel('Time consumption')
legend = plt.legend(loc="upper right",bbox_to_anchor=(1.0, 1.0), framealpha=1)


plt.show()
#plt.savefig('foo.png')
########################## material data plot ##################3
if dim==2:
    fig=plt.figure(6)
    plt.subplot(2, 1, 1)
    ax = fig.gca(projection='3d')
    x = np.arange(0, len(Agani.val[0, 0]))
    X, Y = np.meshgrid(x, x)
    ax.plot_surface(X, Y, Agani.val[0,0])
    #plt.imshowplot_surface(Aga.val[0,0])
    #plt.colorbar()
    fig=plt.figure(5)
    ax = fig.gca(projection='3d')
    x = np.arange(0,len(Agani.val[0, 0]))
    X, Y = np.meshgrid(x, x)
    ax.plot_surface(X, Y, Aganis.full())


if dim==3:
    fig=plt.figure(6)
    plt.subplot(2, 1, 1)
    multi_slice_viewer(Aga.val[0,0,:,:,:])
    plt.subplot(2, 1, 2)
    multi_slice_viewer(Agas.full())
#fig, ax = plt.subplots()
#ax.imshow(Aga.val[0,0,:,:,1])
#fig.canvas.mpl_connect('key_press_event', process_key)


plt.show()
#fig=plt.figure(6)
#ax = fig.gca(projection='3d')
#plt.imshow(Aga.val[0,0])
#QQ=Aga.val[0,0,0]
#plt.hold(True)
#x = np.arange(0, len(Aga.val[0, 0]))
#X, Y = np.meshgrid(x, x)
#plt.contourf(X, Y, Aga.val[0,0,0],30, cmap='RdGy')
#surf = ax.plot_surface(X, Y, Aga.val[0,0,:,:,1])
#for i in range(1,len(Aga.val[0, 0])):
    #plt.subplot(6,1, i)
 #   fig=plt.figure(i+5)
#    plt.hold(True)
 #   plt.imshow(Aga.val[0,0,:,:,i-1])