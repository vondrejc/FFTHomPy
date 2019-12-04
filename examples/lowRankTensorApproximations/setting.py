import numpy as np
from ffthompy import Struct, Timer
from ffthompy.materials import Material
from ffthompy.tensorsLowRank.materials import LowRankMaterial

try:
    from uq.decomposition import KL_Fourier
except:
    import warnings
    warnings.warn('Package StoPy (https://github.com/vondrejc/StoPy) is not available; required for Karhunen-Loeve decomposition.')


kind_list=['cano','tucker','tt']

def get_default_parameters(dim, N, material, kind):
    pars=Struct(dim=dim, # number of dimensions (works for 2D and 3D)
            N=dim*(N,), # number of voxels (assumed equal for all directions)
            Y=np.ones(dim), # size of periodic cell
            recover_sparse=1, # recalculate full material coefficients from tensorsLowRank one
            solver=dict(tol=1e-8,
                        maxiter=50),
            )
    pars_sparse=pars.copy()
    pars_sparse.update(Struct(debug=False,
                              kind=kind_list[kind], # type of tensorsLowRank tensor: 'cano', 'tucker', or 'tt'
                              precond_rank=10,
                              N=dim*(N,),
                              rhs_tol=1e-8,
                              solver=dict(method='mr', # method could be 'Richardson'(r),'minimal_residual'(mr), or 'Chebyshev'(c)
                                          rank=10, # rank of solution vector
                                          tol_truncate=None,
                                          tol=1e-8,
                                          maxiter=30, # no. of iterations for a solver
                                          minres_fails=6, # stop the solver when the residuum fails to decrease n times
                                          fast=True, # fast truncation
                                          approx_omega=False, # inner product of tuckers could be so slow
                                                              # that using an approximate omega could gain.
                                          ),
                              ))
    return pars, pars_sparse

def get_material_coef(material, pars, pars_sparse, ga=True):
    # get configuration settings
    pars, pars_sparse, mat_conf = getMat_conf(material, pars, pars_sparse)

    # generating material coefficients
    mat=Material(mat_conf)
    mats=LowRankMaterial(mat_conf, pars_sparse.kind)

    Agani=mat.get_A_GaNi(pars.N, primaldual='primal')
    Aganis=mats.get_A_GaNi(pars_sparse.N, primaldual='primal', k=pars_sparse.matrank)
    Agani.val=recover_Agani(Agani, Aganis)

    if ga:
        Aga=mat.get_A_Ga(pars.Nbar(pars.N), primaldual='primal')
        Agas=mats.get_A_Ga(pars_sparse.Nbar(pars_sparse.N), primaldual='primal', k=pars_sparse.matrank)
        Aga.val=recover_Aga(Aga, Agas)
    else:
        Aga=None
        Agas=None

    if 'Aniso' in mat_conf: # workaround for anisotropic material
        Aniso=mat_conf['Aniso']
        Agani.add_mean(Aniso)

        if ga:
            Aga.add_mean(Aniso)

        pars_sparse.update(Struct(Aniso=Aniso))

        tic=Timer('calc_eig')
        eigs=Agani.calc_eigs(symmetric=True)
        tic.measure()

        pars_sparse.solver['alpha']=0.5*(eigs.min()+eigs.max())
    else:
        pars_sparse.solver['alpha']=0.5*(Agani[0, 0].min()+Agani[0, 0].max())

    return Aga, Agani, Agas, Aganis

def getMat_conf(material, pars, pars_sparse):

    dim=pars.dim
    N=pars.N[0]

    if dim==2:
        pars_sparse.update(Struct(N=dim*(1*N,),))
    elif dim==3:
        pars_sparse.update(Struct(N=dim*(1*N,),))

    # ## auxiliary operator
    Nbar=lambda N: 2*np.array(N)-1
    pars.update(Struct(Nbar=lambda N: 2*np.array(N)-1))
    pars_sparse.update(Struct(Nbar=lambda N: 2*np.array(N)-1))

    # PROBLEM DEFINITION ######################################################
    if material in [0,3]: # square inclusion
        mat_conf={'inclusions': ['square', 'otherwise'],
                    'positions': [0. * np.ones(dim), ''],
                    'params': [0.6*np.ones(dim), ''], # size of sides
                    'vals': [10*np.eye(dim), 1. * np.eye(dim)],
                    'Y': np.ones(dim),
                    'P': dim*(5,),
                    'order': 0, }
        pars_sparse.update(Struct(matrank=2))

    elif material in [1]: # pyramid-like inclusion
        mat_conf={'inclusions': ['pyramid', 'all'],
                    'positions': [0. * np.ones(dim), ''],
                    'params': [0.8*np.ones(dim), ''], # size of sides
                    'vals': [10*np.eye(dim), 1. * np.eye(dim)],
                    'Y': np.ones(dim),
                    'P': pars.N,
                    'order': 1, }
        pars_sparse.update(Struct(matrank=2))

    elif material in [2,4]: # stochastic material
        pars_sparse.update(Struct(matrank=10))

        kl=KL_Fourier(covfun=2, cov_pars={'rho': 0.15, 'sigma': 1.}, N=pars.N, puc_size=pars.Y,
                      transform=lambda x: np.exp(x))
        if dim==2:
            kl.calc_modes(relerr=0.1)
        elif dim==3:
            kl.calc_modes(relerr=0.4)
        ip=np.random.random(kl.modes.n_kl)-0.5
        np.set_printoptions(precision=8)
        print('ip={}\n'.format(ip.__repr__()))
        if dim==2:
            ip=np.array(
                [0.24995, 0.009014,-0.004228, 0.266437, 0.345009,-0.29721,-0.291875,-0.125469,
                 0.495526,
                 -0.452405,-0.333025, 0.208331, 0.045902,-0.441424,-0.274428,-0.243702,-0.146728,
                 0.239476,
                 0.404311, 0.214929])
        if dim==3:
            ip=np.array(
                [-0.39561222,-0.37849801, 0.46069148,-0.0354164, 0.04269214,-0.00624889, 0.18498634,
                 0.31043535,-0.14730729,-0.39756328, 0.48918557, 0.15098372,-0.11217825,-0.26506403,
                 0.2006125,-0.2596631,-0.16854476,-0.44617782,-0.19412459, 0.32968464,-0.18441118,
                 -0.15455307, 0.1779399,-0.21214177, 0.18394519,-0.24561992])

        def mat_fun(coor, contrast=10):
            val=np.zeros_like(coor[0])
            for ii in range(kl.modes.n_kl):
                val+=ip[ii]*kl.mode_fun(ii, coor)
            val=(val-val.min())/(val.max()-val.min())*np.log(contrast)
            return np.einsum('ij,...->ij...', np.eye(dim), kl.transform(val))

        mat_conf={'fun': mat_fun,
                  'Y': np.ones(dim),
                  'P': pars.N,
                  'order': 1, }

    else:
        raise ValueError()

    if material in [3,4]: # adding anisotropic material
        if dim==2:
            Aniso=np.array([[ 5.5,-4.5],
                            [-4.5, 5.5]])
        elif dim==3:
            Aniso=np.array([[ 4.25,      -3.25,      -1.76776695],
                            [-3.25,       4.25,       1.76776695],
                            [-1.76776695, 1.76776695, 7.5       ]])

        mat_conf.update({'Aniso': Aniso})

    return pars, pars_sparse, mat_conf


def recover_Aga(Aga, Agas):

    print('recovering full material tensors for Ga...')
    Aga.val=np.einsum('ij,...->ij...', np.eye(Aga.dim), Agas.full().val)

    print('Norm of difference in mat properties: {}'.format(np.linalg.norm(Aga.val[0, 0]-Agas.full().val)))
    return Aga.val


def recover_Agani(Agani, Aganis):

    print('recovering full material tensors for GaNi...')
    Agani.val=np.einsum('ij,...->ij...', np.eye(Agani.dim), Aganis.full().val)

    print('Norm of difference in mat properties: {}'.format(np.linalg.norm(Agani.val[0, 0]-Aganis.full().val)))
    return Agani.val


def getGaData(mat, mats, pars, pars_sparse):

    Aga=mat.get_A_Ga(pars.Nbar(pars.N), primaldual='primal')
    Agas=mats.get_A_Ga(pars_sparse.Nbar(pars_sparse.N), primaldual='primal', k=pars_sparse.matrank).set_fft_form()

    #
    if np.array_equal(pars.N, pars_sparse.N):
        print(np.linalg.norm(Aga.val[0, 0]-Agas.full().val))

    if pars.recover_sparse:
        print('recovering full material tensors...')
        Aga.val=np.einsum('ij,...->ij...', np.eye(pars.dim), Agas.full().val)

    if np.array_equal(pars.N, pars_sparse.N):
        print(np.linalg.norm(Aga.val[0, 0]-Agas.full().val))

    return Aga, Agas


def getGaNiData(mat, mats, pars, pars_sparse):
    Agani=mat.get_A_GaNi(pars.N, primaldual='primal')
    Aganis=mats.get_A_GaNi(pars_sparse.N, primaldual='primal', k=pars_sparse.matrank)

    if np.array_equal(pars.N, pars_sparse.N):
        print(np.linalg.norm(Agani.val[0, 0]-Aganis.full().val))

    if pars.recover_sparse:
        print('recovering full material tensors...')
        Agani.val=np.einsum('ij,...->ij...', np.eye(pars.dim), Aganis.full().val)

    if np.array_equal(pars.N, pars_sparse.N):
        print(np.linalg.norm(Agani.val[0, 0]-Aganis.full().val))

    return Agani, Aganis
