import unittest
import numpy as np
from matvec import DFT, VecTri


class Test_matvec(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_matrix_versions(self):
        dim = 2
        for n in [5, 6]:
            N = n*np.ones(dim)
            ur = VecTri(name='rand', dim=2, N=N, valtype='rand')
            FN = DFT(name='FN', inverse=False, N=N, d=dim)
            FiN = DFT(name='FiN', inverse=True, N=N, d=dim)
            msg = 'Operations .matrix() .vec() do not work properly!'
            val = np.linalg.norm(FN(ur).vec()-FN.matrix().dot(ur.vec()))
            self.assertAlmostEqual(0, val, msg=msg, delta=1e-14)
            val = np.linalg.norm(FiN(ur).vec()-FiN.matrix().dot(ur.vec()))
            self.assertAlmostEqual(0, val, msg=msg, delta=1e-14)

if __name__ == "__main__":
    unittest.main()
