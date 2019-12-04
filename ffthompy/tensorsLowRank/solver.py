import numpy as np
from ffthompy import Timer
from ffthompy.tensorsLowRank.objects import SparseTensor

def linear_solver(method, Afun, B, par):
    if method in ['Richardson','richardson','r','R']:
        Fu, ress=richardson(Afun=Afun, B=B, par=par)
    elif method in ['Chebyshev','chebyshev','c','C']:
        Fu, ress=cheby2TERM(Afun=Afun, B=B, par=par)
    elif method in ['minimal_residual','mr','m','M']:
        Fu, ress=minimal_residual(Afun=Afun, B=B, par=par)
    elif method in ['minimal_residual_debug','mrd']:
        Fu, ress=minimal_residual_debug(Afun=Afun, B=B, par=par)
    return Fu, ress

def cheby2TERM(Afun, B, x0=None, par={}, callback=None):
    """
    Chebyshev two-term iterative solver

    Parameters
    ----------
    Afun : a function, represnting linear function A in the system Ax =B
    B : tensorsLowRank tensor representing vector B in the right-hand side of linear system
    x0 : tensorsLowRank tensor representing initial approximation of solution of linear system
    par : dict
          parameters of the method
    callback :

    Returns
    -------
    x : resulting unknown vector
    res : dict
        results
    """

    if 'tol' not in par:
        par['tol'] = 1e-06
    if 'maxiter' not in par:
        par['maxiter'] = 1e7
    if 'eigrange' not in par:
        raise NotImplementedError("It is necessary to calculate eigenvalues.")
    else:
        Egv = par['eigrange']

    res={'norm_res': [],
         'kit': 0}

    bnrm2 = B.norm()
    Ib = 1.0/bnrm2
    if bnrm2 == 0:
        bnrm2 = 1.0

    if x0 is None:
        x=B
    else:
        x=x0

    r = B - Afun(x)
    r0=r.norm()
    res['norm_res'].append(Ib*r0)# For Normal Residue

    if res['norm_res'][-1] < par['tol']: # if errnorm is less than tol
        return x, res

    M=SparseTensor(kind=x.kind, val=np.ones(x.N.size*[3,]), rank=1) # constant field
    FM=M.fourier().enlarge(x.N)

    d = (Egv[1]+Egv[0])/2.0 # np.mean(par['eigrange'])
    c = (Egv[1]-Egv[0])/2.0 # par['eigrange'][1] - d
    v = x*0.0
    while (res['norm_res'][-1] > par['tol']) and (res['kit'] < par['maxiter']):
        res['kit'] += 1
        x_prev = x
        if res['kit'] == 1:
            p = 0
            w = 1/d
        elif res['kit'] == 2:
            p = -(1/2)*(c/d)*(c/d)
            w = 1/(d-c*c/2/d)
        else:
            p = -(c*c/4)*w*w
            w = 1/(d-c*c*w/4)
        v = (r - p*v).truncate(rank=par['rank'], tol=par['tol_truncate'])
        x = (x_prev + w*v)
        x=(-FM*x.mean()+x).truncate(rank=par['tol'], tol=par['tol_truncate']) # setting correct mean
        r = B - Afun(x)

        res['norm_res'].append((1.0/r0)*r.norm())

        if callback is not None:
            callback(x)

    if par['tol'] < res['norm_res']: # if tolerance is less than error norm
        print("Chebyshev solver does not converges!")
    else:
        print("Chebyshev solver converges.")

    if res['kit'] == 0:
        res['norm_res'] = 0
    return x, res

def minimal_residual(Afun, B, x0=None, par=None):
    fast=par.get('fast')

    res={'norm_res': [],
         'kit': 0}
    if x0 is None:
        x=B*(1./par['alpha'])
    else:
        x=x0
    x_sol=x # solution with minimal residuum

    if 'norm' not in par:
        norm=lambda X: X.norm(normal_domain=False)

    residuum=B-Afun(x)
    res['norm_res'].append(norm(residuum))
    beta=Afun(residuum.truncate(tol=par['tol_truncate'], fast=fast))

    M=SparseTensor(kind=x.kind, val=np.ones(x.N.size*[3,]), rank=1) # constant field
    FM=M.fourier().enlarge(x.N)
    minres_fail_counter=0

    while (res['norm_res'][-1] > par['tol'] and res['kit'] < par['maxiter']):
        res['kit']+=1

        if par['approx_omega']:
            omega=res['norm_res'][-1] / norm(beta) # approximate omega
        else:
            omega= beta.inner(residuum)/norm(beta)**2 # exact formula

        x=(x+residuum*omega)

        # setting correct mean
        x=(-FM*x.mean()+x).truncate(rank=par['rank'], tol=par['tol_truncate'], fast=fast)

        residuum=B-Afun(x)

        res['norm_res'].append(norm(residuum))

        if res['norm_res'][-1] <= np.min(res['norm_res'][:-1]):
            x_sol=x
        else:
            minres_fail_counter+=1
            if minres_fail_counter>=par['minres_fails']:
                print('Residuum has risen up {} times -> ending solver.'.format(par['minres_fails']))
                break

        beta=Afun(residuum.truncate(tol=min([res['norm_res'][-1]/1e1, par['tol']]), fast=fast))

    return x_sol, res


def minimal_residual_debug(Afun, B, x0=None, par=None):
    fast=par.get('fast')

    M=SparseTensor(kind=B.kind, val=np.ones(B.N.size*[3, ]), rank=1) # constant field
    FM=M.fourier().enlarge(B.N)

    res={'norm_res': [],
         'kit': 0}
    if x0 is None:
        x=B*(1./par['alpha'])
    else:
        x=x0

    if 'norm' not in par:
        norm=lambda X: X.norm(normal_domain=False)

    residuum=(B-Afun(x)).truncate(rank=None, tol=par['tol'], fast=fast)
    res['norm_res'].append(norm(residuum))
    beta=Afun(residuum)

    norm_res=res['norm_res'][res['kit']]

    while (norm_res>par['tol'] and res['kit']<par['maxiter']):
        res['kit']+=1
        print('iteration = {}'.format(res['kit']))

        if par['approx_omega']:
            omega=norm_res/norm(beta) # approximate omega
        else:
            omega=beta.inner(residuum)/norm(beta)**2 # exact formula

        x=(x+residuum*omega)
        x=(-FM*x.mean()+x).truncate(rank=par['rank'], tol=par['tol']) # setting correct mean

        tic=Timer('compute residuum')
        residuum=(B-Afun(x))
#         residuum=residuum.truncate(rank=2*rank, tol=tol)
#         residuum=(B-Afun(x)).truncate(rank=rank, tol=tol)
#         residuum=(B-Afun(x))
        tic.measure()
        tic=Timer('residuum norm')
        norm_res=norm(residuum)
        tic.measure()
        if par['divcrit'] and norm_res>res['norm_res'][-1]:
            break
        res['norm_res'].append(norm_res)

        tic=Timer('truncate residuum')
#         residuum_for_beta=residuum.truncate(rank=rank, tol=tol)
#         residuum_for_beta=residuum.truncate(rank=None, tol=1-4)
        tol=min([norm_res/1e1, par['tol']])
        residuum_for_beta=residuum.truncate(rank=None, tol=tol, fast=fast)
        tic.measure()
        print('tolerance={}, rank={}'.format(tol, residuum_for_beta.r))
        print('residuum_for_beta.r={}'.format(residuum_for_beta.r))
        tic=Timer('compute beta')
        beta=Afun(residuum_for_beta)
        tic.measure()
        pass
    return x, res

def richardson(Afun, B, x0=None, rank=None, tol=None, par=None, norm=None):
    if isinstance(par['alpha'], float):
        omega=1./par['alpha']
    else:
        raise ValueError()
    res={'norm_res': [],
         'kit': 0}
    if x0 is None:
        x=B*omega
    else:
        x=x0

    if norm is None:
        norm=lambda X: X.norm()

    res['norm_res'].append(norm(B))

    M=SparseTensor(kind=x.kind, val=np.ones(x.N.size*[3,]), rank=1) # constant field
    FM=M.fourier().enlarge(x.N)

    norm_res=1e15
    while (norm_res>par['tol'] and res['kit']<par['maxiter']):
        res['kit']+=1
        residuum= B-Afun(x)
        norm_res = norm(residuum)
        if par['divcrit'] and norm_res>res['norm_res'][res['kit']-1]:
            break

        x=(x+residuum*omega)
        x=(-FM*x.mean()+x).truncate(rank=rank, tol=tol, fast=True) # setting correct mean

        res['norm_res'].append(norm_res)

    return x, res
