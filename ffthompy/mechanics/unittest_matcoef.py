import unittest
import numpy as np
from numpy.linalg import norm
from .matcoef import ElasticTensor as ET


class Test_matcoef(unittest.TestCase):

    def setUp(self):
        print('\nChecking mechanics...')

    def tearDown(self):
        print('...ok')

    @staticmethod
    def get_rand_sym(dim, ndim):
        A = np.random.random(ndim*(dim,))
        if ndim == 2:
            A = A+A.T
        else:
            A = 0.5*(A+np.einsum('ijkl->klij',A))
            A = 0.5*(A+np.einsum('ijkl->ijlk',A))
            A = 0.5*(A+np.einsum('ijkl->jikl',A))
        return A

    def test_mandel(self):
        print("  Mandel's notation")
        for dim in [2, 3]:
            for ndim in [2, 4]:
                A = self.get_rand_sym(dim=dim, ndim=ndim)

                Am = ET.create_mandel(A, ndim=None)
                self.assertAlmostEqual(0, norm(Am-Am.T))
                A2 = ET.dispose_mandel(Am, ndim=None)
                Am2 = ET.create_mandel(A2, ndim=None)
                self.assertAlmostEqual(0, norm(Am-Am2),
                                       msg='mandel in dim={0} and ndim={1}'.format(dim, ndim),
                                       delta=1e-14)
                self.assertAlmostEqual(0, norm(A-A2),
                                       msg='mandel in dim={0} and ndim={1}'.format(dim, ndim),
                                       delta=1e-14)

    def test_plane(self):
        print('  plane strain and stress')
        A = self.get_rand_sym(dim=3, ndim=4)
        Am = ET.create_mandel(A)
        Amplane = ET.get_plane_in_engineering(Am).squeeze()

        Aplane = ET.get_plane_in_tensor(A)
        Aplanem = ET.create_mandel(Aplane).squeeze()
        self.assertAlmostEqual(0, norm(Aplanem - Amplane))


if __name__ == "__main__":
    unittest.main()
