import unittest
import numpy as np
from numpy.linalg import norm
from ffthompy import PrintControl
from ffthompy.tensors import Tensor, DFT, Operator
from ffthompy.tensors.projection import scalar_tensor
from ffthompy.projections import scalar
from ffthompy.general.solver import linear_solver

prt=PrintControl()


class Test_solvers(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_projections(self):
        print('\nChecking projections...')
        dim=2
        n=5
        N = n*np.ones(dim, dtype=np.int)
        hG0N, hG1N, hG2N = scalar(N, Y=np.ones(dim), centered=True, NyqNul=True)
        hG0Nt, hG1Nt, hG2Nt = scalar_tensor(N, Y=np.ones(dim), centered=True, NyqNul=True)

        self.assertAlmostEqual(0, norm(hG0N.val-hG0Nt.val), delta=1e-13)
        self.assertAlmostEqual(0, norm(hG1N.val-hG1Nt.val), delta=1e-13)
        self.assertAlmostEqual(0, norm(hG2N.val-hG2Nt.val), delta=1e-13)
        print('...ok')

    def test_solvers(self):
        print('\nChecking solvers...')
        dim=2
        n=5
        N = n*np.ones(dim, dtype=np.int)

        _, hG1Nt, _ = scalar_tensor(N, Y=np.ones(dim), centered=True, NyqNul=True)

        FN=DFT(name='FN', inverse=False, N=N)
        FiN=DFT(name='FiN', inverse=True, N=N)

        G1N=Operator(name='G1', mat=[[FiN, hG1Nt, FN]])

        A=Tensor(name='A', val=np.einsum('ij,...->ij...', np.eye(dim), 1.+10.*np.random.random(N)),
                 order=2, multype=21)

        E=np.zeros((dim,)+dim*(n,)); E[0] = 1. # set macroscopic loading
        E=Tensor(name='E', val=E, order=1)

        GAfun=Operator(name='GA', mat=[[G1N, A]])
        GAfun.define_operand(E)

        B=GAfun(-E)
        x0=E.copy(name='x0')
        x0.val[:]=0

        par={'tol': 1e-10,
             'maxiter': int(1e3),
             'alpha': 0.5*(1.+10.),
             'eigrange':[1., 10.]}

        # reference solution
        X,_=linear_solver(Afun=GAfun, B=B, x0=x0, par=par, solver='CG')

        prt.disable()
        for solver in ['scipy_cg', 'richardson', 'chebyshev']:
            x,_=linear_solver(Afun=GAfun, B=B, x0=x0, par=par, solver=solver)
            self.assertAlmostEqual(0, norm(X.val-x.val), delta=1e-8, msg=solver)
        prt.enable()

        print('...ok')


if __name__ == "__main__":
    unittest.main()
