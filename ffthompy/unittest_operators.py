import unittest
import numpy as np
from ffthompy.tensors import Tensor
from ffthompy.operators import DFT, grad, symgrad, potential


class Test_operators(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_operators(self):
        print('Checking compatiblity...')
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


if __name__ == "__main__":
    unittest.main()
