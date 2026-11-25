import logging

import numpy as np

import ffthompy.projections as proj
from ffthompy.general.base import Timer
from ffthompy.general.solver import linear_solver
from ffthompy.general.solver_pp import CallBack, CallBack_GA
from ffthompy.materials import Material
from ffthompy.postprocess import add_macro2minimizer, postprocess
from ffthompy.tensors import DFT, Operator, Tensor

logger = logging.getLogger(__name__)


def scalar(problem):
    """
    Homogenization of scalar elliptic problem.

    Parameters
    ----------
    problem : object
    """
    logger.info(' ')
    pb = problem
    logger.debug('%s', pb)

    # Fourier projections
    _, hG1N, hG2N = proj.scalar(pb.solve['N'], pb.Y, NyqNul=True, tensor=True)

    if pb.solve['kind'] == 'GaNi':
        Nbar = pb.solve['N']
    elif pb.solve['kind'] == 'Ga':
        Nbar = 2*pb.solve['N'] - 1
        hG1N = hG1N.enlarge(Nbar)
        hG2N = hG2N.enlarge(Nbar)

    FN = DFT(name='FN', inverse=False, N=Nbar)
    FiN = DFT(name='FiN', inverse=True, N=Nbar)

    G1N = Operator(name='G1', mat=[[FiN, hG1N, FN]])
    G2N = Operator(name='G2', mat=[[FiN, hG2N, FN]])

    for primaldual in pb.solve['primaldual']:
        tim = Timer(name='primal-dual')
        logger.info('%s', '\nproblem: ' + primaldual)
        solutions = np.zeros(pb.shape).tolist()
        results = np.zeros(pb.shape).tolist()

        # material coefficients
        mat = Material(pb.material)

        if pb.solve['kind'] == 'GaNi':
            A = mat.get_A_GaNi(pb.solve['N'], primaldual)
        elif pb.solve['kind'] == 'Ga':
            A = mat.get_A_Ga(Nbar=Nbar, primaldual=primaldual)

        if primaldual == 'primal':
            GN = G1N
        else:
            GN = G2N

        Afun = Operator(name='FiGFA', mat=[[GN, A]])

        for iL in np.arange(pb.dim): # iteration over unitary loads
            E = np.zeros(pb.dim)
            E[iL] = 1
            logger.debug('%s', 'macroscopic load E = ' + str(E))
            EN = Tensor(name='EN', N=Nbar, shape=(pb.dim,), Fourier=False)
            EN.set_mean(E)
            # initial approximation for solvers
            x0 = Tensor(name='x0', N=Nbar, shape=(pb.dim,), Fourier=False)

            B = Afun(-EN) # RHS

            if not hasattr(pb.solver, 'callback'):
                cb = CallBack(A=Afun, B=B)
            elif pb.solver['callback'] == 'detailed':
                cb = CallBack_GA(A=Afun, B=B, EN=EN, A_Ga=A, GN=GN)
            else:
                raise NotImplementedError("The solver callback (%s) is not \
                    implemented" % (pb.solver['callback']))

            logger.debug('solver : %s', pb.solver['kind'])
            X, info = linear_solver(solver=pb.solver['kind'], Afun=Afun, B=B,
                                    x0=x0, par=pb.solver, callback=cb)

            solutions[iL] = add_macro2minimizer(X, E)
            results[iL] = {'cb': cb, 'info': info}
            logger.info('%s', cb)
        tim.measure()

        # POSTPROCESSING
        del Afun, B, E, EN, GN, X
        postprocess(pb, A, mat, solutions, results, primaldual)


def elasticity(problem):
    """
    Homogenization of linear elasticity.

    Parameters
    ----------
    problem : object
    """
    logger.info(' ')
    pb = problem
    logger.debug('%s', pb)

    # Fourier projections
    _, hG1hN, hG1sN, hG2hN, hG2sN = proj.elasticity(pb.solve['N'], pb.Y, NyqNul=True)
    del _

    if pb.solve['kind'] == 'GaNi':
        Nbar = pb.solve['N']
    elif pb.solve['kind'] == 'Ga':
        Nbar = 2*pb.solve['N'] - 1
        hG1hN = hG1hN.enlarge(Nbar)
        hG1sN = hG1sN.enlarge(Nbar)
        hG2hN = hG2hN.enlarge(Nbar)
        hG2sN = hG2sN.enlarge(Nbar)

    FN = DFT(name='FN', inverse=False, N=Nbar)
    FiN = DFT(name='FiN', inverse=True, N=Nbar)

    G1N = Operator(name='G1', mat=[[FiN, hG1hN + hG1sN, FN]])
    G2N = Operator(name='G2', mat=[[FiN, hG2hN + hG2sN, FN]])

    for primaldual in pb.solve['primaldual']:
        tim = Timer(name='primal-dual')
        logger.info('%s', '\nproblem: ' + primaldual)
        solutions = np.zeros(pb.shape).tolist()
        results = np.zeros(pb.shape).tolist()

        # material coefficients
        mat = Material(pb.material)

        if pb.solve['kind'] == 'GaNi':
            A = mat.get_A_GaNi(pb.solve['N'], primaldual)
        elif pb.solve['kind'] == 'Ga':
            A = mat.get_A_Ga(Nbar=Nbar, primaldual=primaldual)

        if primaldual == 'primal':
            GN = G1N
        else:
            GN = G2N

        Afun = Operator(name='FiGFA', mat=[[GN, A]])

        D = int(pb.dim*(pb.dim+1)/2)
        for iL in range(D): # iteration over unitary loads
            E = np.zeros(D)
            E[iL] = 1
            logger.debug('%s', 'macroscopic load E = ' + str(E))
            EN = Tensor(name='EN', N=Nbar, shape=(D,), Fourier=False)
            EN.set_mean(E)
            # initial approximation for solvers
            x0 = EN.zeros_like(name='x0')

            B = Afun(-EN) # RHS

            if not hasattr(pb.solver, 'callback'):
                cb = CallBack(A=Afun, B=B)
            elif pb.solver['callback'] == 'detailed':
                cb = CallBack_GA(A=Afun, B=B, EN=EN, A_Ga=A, GN=GN)
            else:
                raise NotImplementedError("The solver callback (%s) is not \
                    implemented" % (pb.solver['callback']))

            logger.debug('%s', 'solver : %s' % pb.solver['kind'])
            X, info = linear_solver(solver=pb.solver['kind'], Afun=Afun, B=B,
                                    x0=x0, par=pb.solver, callback=cb)

            solutions[iL] = add_macro2minimizer(X, E)
            results[iL] = {'cb': cb, 'info': info}
            logger.info('%s', cb)
        tim.measure()

        # POSTPROCESSING
        del Afun, B, E, EN, GN, X
        postprocess(pb, A, mat, solutions, results, primaldual)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    exec(compile(open('../main_test.py').read(), '../main_test.py', 'exec'))
