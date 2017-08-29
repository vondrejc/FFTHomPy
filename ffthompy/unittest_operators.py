import unittest
import numpy as np
from ffthompy.tensors import Tensor
from ffthompy.operators import DFT, grad, symgrad, potential, Operator, matrix2tensor
from ffthompy.projections import scalar, elasticity_small_strain, elasticity_large_deformation

class Test_operators(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_operators(self):
        print('Checking operators...')
        for dim in [2, 3]:
            N = 5*np.ones(dim, dtype=np.int)
            F = DFT(N=N, inverse=False)

            # scalar problem
            u = Tensor(name='u', shape=(1,), N=N, Fourier=False).randomize()
            u.val -= np.mean(u.val)

            Fu = F(u)
            Fu2 = potential(grad(Fu))
            self.assertAlmostEqual(0, (Fu==Fu2)[1], delta=1e-13,
                                   msg='scalar problem, Fourier=True')

            u2 = potential(grad(u))
            self.assertAlmostEqual(0, (u==u2)[1], delta=1e-13,
                                   msg='scalar problem, Fourier=False')


            # vectorial problem
            u = Tensor(name='u', shape=(dim,), N=N, Fourier=False).randomize()
            u.add_mean(-u.mean())

            Fu = F(u)
            Fu2 = potential(grad(Fu))
            self.assertAlmostEqual(0, (Fu==Fu2)[1], delta=1e-13,
                                   msg='vectorial problem, Fourier=True')


            u2 = potential(grad(u))
            self.assertAlmostEqual(0, (u==u2)[1], delta=1e-13,
                                   msg='vectorial problem, Fourier=False')


            # 'vectorial problem - symetric gradient
            u = Tensor(name='u', shape=(dim,), N=N, Fourier=False).randomize()
            u.add_mean(-u.mean())

            Fu = F(u)
            Fu2 = potential(symgrad(Fu), small_strain=True)
            self.assertAlmostEqual(0, (Fu==Fu2)[1], delta=1e-13,
                                   msg='vectorial - sym, Fourier=True')


            u2 = potential(symgrad(u), small_strain=True)
            self.assertAlmostEqual(0, (u==u2)[1], delta=1e-13,
                                   msg='vectorial - sym, Fourier=False')

        print('...done')

    def test_compatibility(self):
        print('Checking compatibility...')
        for dim in [3]:
            N = 5*np.ones(dim, dtype=np.int)
            F = DFT(inverse=False, N=N)
            iF = DFT(inverse=True, N=N)

            # scalar problem
            _, G1l, G2l = scalar(N, Y=np.ones(dim), centered=True, NyqNul=True)
            P1 = Operator(name='P1', mat=[[iF, matrix2tensor(G1l), F]])
            P2 = Operator(name='P2', mat=[[iF, matrix2tensor(G2l), F]])
            u = Tensor(name='u', shape=(1,), N=N, Fourier=False).randomize()
            grad_u = grad(u)
            self.assertAlmostEqual(0, (P1(grad_u)-grad_u).norm(), delta=1e-13)
            self.assertAlmostEqual(0, P2(grad_u).norm(), delta=1e-13)

            e = P1(Tensor(name='u', shape=(dim,), N=N, Fourier=False).randomize())
            e2 = grad(potential(e))
            self.assertAlmostEqual(0, (e-e2).norm(), delta=1e-13)

            # vectorial problem
            hG = elasticity_large_deformation(N=N, Y=np.ones(dim), centered=True)
            P1 = Operator(name='P', mat=[[iF, hG, F]])
            u = Tensor(name='u', shape=(dim,), N=N, Fourier=False).randomize()
            grad_u = grad(u)
            val = (P1(grad_u)-grad_u).norm()
            self.assertAlmostEqual(0, val, delta=1e-13)

            e = P1(Tensor(name='F', shape=(dim,dim), N=N, Fourier=False).randomize())
            e2 = grad(potential(e))
            self.assertAlmostEqual(0, (e-e2).norm(), delta=1e-13)

            # vectorial problem - symetric gradient
            hG = elasticity_small_strain(N=N, Y=np.ones(dim), centered=True)
            P1 = Operator(name='P', mat=[[iF, hG, F]])
            u = Tensor(name='u', shape=(dim,), N=N, Fourier=False).randomize()
            grad_u = symgrad(u)
            val = (P1(grad_u)-grad_u).norm()
            self.assertAlmostEqual(0, val, delta=1e-13)

            e = P1(Tensor(name='strain', shape=(dim,dim), N=N, Fourier=False).randomize())
            e2 = symgrad(potential(e, small_strain=True))
            self.assertAlmostEqual(0, (e-e2).norm(), delta=1e-13)


if __name__ == "__main__":
    unittest.main()
