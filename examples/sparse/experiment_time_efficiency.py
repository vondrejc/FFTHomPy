

import numpy as np

from ffthompy import Timer, Struct
from ffthompy.materials import Material
from ffthompy.sparse.homogenisation import (homog_Ga_full_potential, homog_GaNi_full_potential,
                                            homog_Ga_sparse, homog_GaNi_sparse)
from ffthompy.sparse.materials import SparseMaterial

from examples.sparse.material_setting import getMat_conf,recover_Aga,recover_Agani

import matplotlib.pylab as plt
import os

dim=2
material=0
kind=0

k_list=[2,3]#,4,5,6 ]
N_list= 5*np.power(3, k_list)
#N_list= [15,45,75,125,215]

full_time_list=[None]*len(N_list)
sparse_time_list=[None]*len(N_list)
rank_list=[None]*len(N_list)
memory_list=[None]*len(N_list)
epsilon=1e-8
kind_list=['cano','tucker','tt']


for i in range(len(N_list)):
    # PARAMETERS ##############################################################
    N=N_list[i]

    pars=Struct(dim=dim, # number of dimensions (works for 2D and 3D)
                N=dim*(N,), # number of voxels (assumed equal for all directions)
                Y=np.ones(dim), # size of periodic cell
                recover_sparse=1, # recalculate full material coefficients from sparse one
                solver=dict(tol=1e-10,
                            maxiter=50),
                )
    pars_sparse = pars.copy()
    pars, pars_sparse, mat_conf = getMat_conf( material, pars, pars_sparse)

    # generating material coefficients
    mat=Material(mat_conf)
#    Agani=mat.get_A_GaNi(pars.N, primaldual='primal')
    Aga=mat.get_A_Ga(pars.Nbar(pars.N), primaldual='primal')

    print('\n== Full solution with potential by CG (Ga) ===========')
    resP_Ga=homog_Ga_full_potential(Aga, pars)
    print('mean of solution={}'.format(resP_Ga.Fu.mean()))
    print('homogenised properties (component 11) = {}'.format(resP_Ga.AH))
    full_time_list[i]=resP_Ga.time

    ###########################
    N=3*N
    ###########################
    pars.update(Struct(N=dim*(N,),)) # number of voxels (assumed equal for all directions)
    pars_sparse.update(Struct(kind=kind_list[kind],  # type of sparse tensor: 'cano', 'tucker', or 'tt'
                              rank=1,  # rank of solution vector
                              precond_rank=1,
                              tol=None,
                              N=dim * (1 * N,),
                              solver=dict(method='mr',
                                          # method could be 'Richardson'(r),'minimal_residual'(mr), or 'Chebyshev'(c)
                                          approx_omega=False,  # inner product of tuckers could be so slow
                                          # that using an approximate omega could gain.
                                          eigrange=[0.6, 50],  # for Chebyshev solver
                                          tol=1e-10,
                                          maxiter=50,  # no. of iterations for a solver
                                          divcrit=False),
                              ))

    # generating material coefficients
    mat=Material(mat_conf)
#    Agani=mat.get_A_GaNi(pars.N, primaldual='primal')
    Aga=mat.get_A_Ga(pars_sparse.Nbar(pars_sparse.N), primaldual='primal')

    for r in range(5, N+1,5):
        pars_sparse.update(Struct(rank=r, # rank of solution vector
                                  precond_rank=r,
                                  ))

        print('== format={}, N={}, dim={}, material={} ===='.format(pars_sparse.kind,
                                                                    N, dim, material))

        # PROBLEM DEFINITION ######################################################

        # generating material coefficients
        mats=SparseMaterial(mat_conf, pars_sparse.kind)

#        Aganis=mats.get_A_GaNi(pars_sparse.N, primaldual='primal', k=pars_sparse.matrank)
        Agas=mats.get_A_Ga(pars_sparse.Nbar(pars_sparse.N), primaldual='primal', k=pars_sparse.matrank)
        Agas.set_fft_form()

        Aga.val = recover_Aga(Aga, Agas)
        pars_sparse.update(Struct(alpha=0.5*(Aga[0, 0].min()+Aga[0, 0].max())))


        print('\n== SPARSE solver with preconditioner (Ga) =======================')
        resS_Ga=homog_Ga_sparse(Agas, pars_sparse)
        print('mean of solution={}'.format(resS_Ga.Fu.mean()))
        print('homogenised properties (component 11) = {}'.format(resS_Ga.AH))
        print(resS_Ga.Fu)
        #print('iterations={}'.format(resS_Ga.solver['kit']))
        #if np.array_equal(pars.N, pars_sparse.N):
        #    print('norm(dif)={}'.format(np.linalg.norm(resP_Ga.Fu.fourier().val-resS_Ga.Fu.fourier().full().val)))
        print('norm(resP)={}'.format(resS_Ga.solver['norm_res']))
        print('memory efficiency = {0}/{1} = {2}'.format(resS_Ga.Fu.memory, resP_Ga.Fu.val.size, resS_Ga.Fu.memory/resP_Ga.Fu.val.size))
        print ("solution discrepancy",resS_Ga.AH - resP_Ga.AH)

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

##figure( 1)
#fig, ax1 = plt.subplots()
#fig.set_size_inches(5 , 3.5 , forward=True)
#
#ax1.plot(N_list, memory_list,   linewidth=1 , marker='o' , markersize=2 )
#plt.title('sparse vs full memory ratio')
#plt.ylabel('Memory ratio S/F')
#picname = 'memory_efficiency_2D' +'.png'
#
#plt.savefig(picname)
#os.system('eog'+' '+picname +' '+ '&')
#####################################################
fig, ax2 = plt.subplots()
fig.set_size_inches(5 , 3.5 , forward=True)

ax2.plot(N_list, full_time_list,   linewidth=1 , marker='o' , markersize=2, label="Full")
ax2.plot(N_list, sparse_time_list,   linewidth=1 , marker='o' , markersize=2, label="Sparse")
plt.title('Time cost of full and sparse solvers')
plt.ylabel('Time cost(s)')
plt.xlabel('N')
plt.legend(loc='upper left')
picname = 'time_efficiency_2D_mat0_cano' +'.png'

plt.savefig(picname)
os.system('eog'+' '+picname +' '+ '&')