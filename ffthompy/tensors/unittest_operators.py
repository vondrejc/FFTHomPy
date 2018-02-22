import unittest
import numpy as np
from numpy.linalg import norm
from ffthompy.projections import scalar
from ffthompy.tensors import (Tensor, DFT, grad, div, symgrad, potential, Operator, matrix2tensor,
                              grad_div_tensor)
from ffthompy.tensors.projection import elasticity_small_strain, elasticity_large_deformation

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
            print(F) # checking representation

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

            hG, hD = grad_div_tensor(N)
            self.assertAlmostEqual(0, (hD(hG(Fu))==div(grad(Fu)))[1], delta=1e-13,
                                   msg='scalar problem, Fourier=True')

            # matrix version of DFT
            dft = F.matrix(shape=u.shape)
            Fu2 = dft.dot(u.val.ravel())
            self.assertAlmostEqual(0, norm(Fu.val.ravel()-Fu2), delta=1e-13)

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

        print('...ok')

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

            # transpose
            P1TT = P1.transpose().transpose()
            self.assertTrue(P1(grad_u)==P1TT(grad_u))

            self.assertTrue(hG==(hG.transpose_left().transpose_left()))
            self.assertTrue(hG==(hG.transpose_right().transpose_right()))

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

            # means
            Fu=F(u)
            E = np.random.random(u.shape)
            u.set_mean(E)
            self.assertAlmostEqual(0, norm(u.mean()-E), delta=1e-13)
            Fu.set_mean(E)
            self.assertAlmostEqual(0, norm(Fu.mean()-E), delta=1e-13)

            # __repr__
            print(P1)
            print(u)
            self.assertAlmostEqual(0, (P1==P1.transpose()), delta=1e-13)
        print('...ok')

if __name__ == "__main__":
    unittest.main()
