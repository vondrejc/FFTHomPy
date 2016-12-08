#!/usr/bin/python
import numpy as np
from ffthompy.general.base import Timer
from ffthompy.matvec import VecTri


def linear_solver(Afun=None, ATfun=None, B=None, x0=None, par=None,
                  solver=None, callback=None):
    """
    Wraper for various linear solvers suited for FFT-based homogenization.
    """
    tim = Timer('Solving linsys by %s' % solver)
    if callback is not None:
        callback(x0)

    if solver.lower() in ['cg']: # conjugate gradients
        x, info = CG(Afun, B, x0=x0, par=par, callback=callback)
    elif solver.lower() in ['bicg']: # biconjugate gradients
        x, info = BiCG(Afun, ATfun, B, x0=x0, par=par, callback=callback)
    elif solver.lower() in ['iterative']: # iterative solver
        x, info = richardson(Afun, B, x0, par=par, callback=callback)
    elif solver.split('_')[0].lower() in ['scipy']: # solvers in scipy
        from scipy.sparse.linalg import LinearOperator, cg, bicg
        if solver == 'scipy_cg':
            Afun.define_operand(B)
            Afunvec = LinearOperator(Afun.shape, matvec=Afun.matvec,
                                     dtype=np.float64)
            xcol, info = cg(Afunvec, B.vec(), x0=x0.vec(),
                            tol=par['tol'],
                            maxiter=par['maxiter'],
                            xtype=None, M=None, callback=callback)
            info = {'info': info}
        elif solver == 'scipy_bicg':
            Afun.define_operand(B)
            ATfun.define_operand(B)
            Afunvec = LinearOperator(Afun.shape, matvec=Afun.matvec,
                                     rmatvec=ATfun.matvec, dtype=np.float64)
            xcol, info = bicg(Afunvec, B.vec(), x0=x0.vec(),
                              tol=par['tol'],
                              maxiter=par['maxiter'],
                              xtype=None, M=None, callback=callback)
        res = dict()
        res['info'] = info
        x = VecTri(val=np.reshape(xcol, B.dN()))
    else:
        msg = "This kind (%s) of linear solver is not implemented" % solver
        raise NotImplementedError(msg)
    tim.measure(print_time=False)
    info.update({'time': tim.vals})
    return x, info


def richardson(Afun, B, x0, par=None, callback=None):
    omega = 1./par['alpha']
    res = {'norm_res': 1.,
           'kit': 0}
    x = x0
    while (res['norm_res'] > par['tol'] and res['kit'] < par['maxiter']):
        res['kit'] += 1
        x_prev = x
        x = x - omega*(Afun*x - B)
        dif = x_prev-x
        res['norm_res'] = float(dif.T*dif)**0.5
        if callback is not None:
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
    if 'tol' not in list(par.keys()):
        par['tol'] = 1e-6
    if 'maxiter' not in list(par.keys()):
        par['maxiter'] = 1e3

    res = dict()
    xCG = x0
    Ax = Afun*x0
    R = B - Ax
    P = R
    rr = float(R.T*R)
    res['kit'] = 0
    res['norm_res'] = np.double(rr)**0.5 # /np.norm(E_N)
    norm_res_log = []
    norm_res_log.append(res['norm_res'])
    while (res['norm_res'] > par['tol']) and (res['kit'] < par['maxiter']):
        res['kit'] += 1 # number of iterations
        AP = Afun*P
        alp = float(rr/(P.T*AP))
        xCG = xCG + alp*P
        R = R - alp*AP
        rrnext = float(R.T*R)
        bet = rrnext/rr
        rr = rrnext
        P = R + bet*P
        res['norm_res'] = np.double(rr)**0.5
        norm_res_log.append(res['norm_res'])
        if callback is not None:
            callback(xCG)
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
    if 'maxiter' not in par:
        par['maxiter'] = 1e3

    res = dict()
    res['time'] = dbg.start_time()
    xBiCG = x0
    Ax = Afun*x0
    R = B - Ax
    Rs = R
    rr = float(R.T*Rs)
    P = R
    Ps = Rs
    res['kit'] = 0
    res['norm_res'] = rr**0.5 # /np.norm(E_N)
    norm_res_log = []
    norm_res_log.append(res['norm_res'])
    while (res['norm_res'] > par['tol']) and (res['kit'] < par['maxiter']):
        res['kit'] += 1 # number of iterations
        AP = Afun*P
        alp = rr/float(AP.T*Ps)
        xBiCG = xBiCG + alp*P
        R = R - alp*AP
        Rs = Rs - alp*ATfun*Ps
        rrnext = float(R.T*Rs)
        bet = rrnext/rr
        rr = rrnext
        P = R + bet*P
        Ps = Rs + bet*Ps
        res['norm_res'] = rr**0.5
        norm_res_log.append(res['norm_res'])
        if callback is not None:
            callback(xBiCG)
    res['time'] = dbg.get_time(res['time'])
    if res['kit'] == 0:
        res['norm_res'] = 0
    return xBiCG, res

if __name__ == '__main__':
    exec(compile(open('../main_test.py').read(), '../main_test.py', 'exec'))
