#!/usr/bin/python
import numpy as np
import general.dbg as dbg
from homogenize.matvec import VecTri


def linear_solver(Afun=None, ATfun=None, B=None, x0=None, par=None,
                  solver=None, callback=None):
    if solver == 'CG':
        x, info = CG(Afun, B, x0=x0, par=par, callback=callback)
    elif solver == 'iterative':
        x, info = iterative(Afun, x0, par=par, callback=callback)
    else:
        solver = solver
        Afun.define_operand(B)
        ATfun.define_operand(B)
        from scipy.sparse.linalg import LinearOperator, cg, bicg
        Afunvec = LinearOperator(Afun.shape, matvec=Afun.matvec,
                                 rmatvec=ATfun.matvec, dtype=np.float64)
        if solver == 'scipy_cg':
            xcol, info = cg(Afunvec, B.vec(), x0=x0.vec(),
                            tol=par.solver['tol'],
                            maxiter=par.solver['maxiter'],
                            xtype=None, M=None, callback=callback)
        elif solver == 'scipy_bicg':
            xcol, info = bicg(Afunvec, B.vec(), x0=x0.vec(),
                              tol=par.solver['tol'],
                              maxiter=par.solver['maxiter'],
                              xtype=None, M=None, callback=callback)
        res = dict()
        res['info'] = info
        x = VecTri(val=np.reshape(xcol, B.dN()))
    return x, info


def iterative(Afun, x0, par=None, callback=None):
    alp = 1./par.solver['alpha']
    res = {'norm_res': 1.,
           'kit': 0}
    x = x0
    while (res['norm_res'] > par.solver['tol'] and
           res['kit'] < par.solver['maxit']):
        res['kit'] += 1
        x = x - alp*Afun(x)
        callback(x)
    return x, res


def CG(Afun, B, x0=None, par=None, callback=None):
    """
    Conjugate gradients solver.

    Parameters
    ----------
    Afun : Matrix, LinOper, or numpy.array of shape (n, n)
        it stores the matrix data of linear system and provides a matrix by
        vector multiplication
    B : VecTri or numpy.array of shape (n,)
        it stores a right-hand side of linear system
    x0 : VecTri or numpy.array of shape (n,)
        initial approximation of solution of linear system
    par : dict
        parameters of the method
    callback :

    Returns
    -------
    x : VecTri or numpy.array of shape (n,)
        resulting unknown vector
    res : dict
        results
    """
    if x0 is None:
        x0 = B
    if par is None:
        par = dict()
    if 'tol' not in par.keys():
        par['tol'] = 1e-6
    if 'maxit' not in par.keys():
        par['maxit'] = 1e3

    res = dict()
    res['time'] = dbg.start_time()
    xCG = x0
    Ax = Afun(x0)
    R = B - Ax
    P = R
    rr = R*R
    res['kit'] = 0
    res['norm_res'] = np.double(rr)**0.5 # /np.norm(E_N)
    norm_res_log = []
    norm_res_log.append(res['norm_res'])
    if callback is not None:
        callback(xCG)
    while (res['norm_res'] > par['tol']) and (res['kit'] < par['maxit']):
        res['kit'] += 1 # number of iterations
        AP = Afun(P)
        alp = rr/(P*AP)
        xCG = xCG + alp*P
        R = R - alp*AP
        rrnext = R*R
        bet = rrnext/rr
        rr = rrnext
        P = R + bet*P
        res['norm_res'] = np.double(rr)**0.5
        norm_res_log.append(res['norm_res'])
        if callback is not None:
            callback(xCG)
    res['time'] = dbg.get_time(res['time'])
    if res['kit'] == 0:
        res['norm_res'] = 0
    return xCG, res


def BiCG(Afun, ATfun, B, x0=None, par=None, callback=None):
    """
    BiConjugate gradient solver.

    Parameters
    ----------
    Afun : Matrix, LinOper, or numpy.array of shape (n, n)
        it stores the matrix data of linear system and provides a matrix by
        vector multiplication
    B : VecTri or numpy.array of shape (n,)
        it stores a right-hand side of linear system
    x0 : VecTri or numpy.array of shape (n,)
        initial approximation of solution of linear system
    par : dict
        parameters of the method
    callback :

    Returns
    -------
    x : VecTri or numpy.array of shape (n,)
        resulting unknown vector
    res : dict
        results
    """
    if x0 is None:
        x0 = B
    if par is None:
        par = dict()
    if 'tol' not in par:
        par['tol'] = 1e-6
    if 'maxit' not in par:
        par['maxit'] = 1e3

    res = dict()
    res['time'] = dbg.start_time()
    xCG = x0
    Ax = Afun(x0)
    R = B - Ax
    P = R
    rr = R*R
    res['kit'] = 0
    res['norm_res'] = np.double(rr)**0.5 # /np.norm(E_N)
    norm_res_log = []
    norm_res_log.append(res['norm_res'])
    if callback is not None:
        callback(xCG)
    while (res['norm_res'] > par['tol']) and (res['kit'] < par['maxit']):
        res['kit'] += 1 # number of iterations
        AP = Afun(P)
        alp = rr/(P*AP)
        xCG = xCG + alp*P
        R = R - alp*AP
        rrnext = R*R
        bet = rrnext/rr
        rr = rrnext
        P = R + bet*P
        res['norm_res'] = np.double(rr)**0.5
        norm_res_log.append(res['norm_res'])
        if callback is not None:
            callback(xCG)
    res['time'] = dbg.get_time(res['time'])
    if res['kit'] == 0:
        res['norm_res'] = 0
    return xCG, res

if __name__ == '__main__':
    execfile('main_test.py')
