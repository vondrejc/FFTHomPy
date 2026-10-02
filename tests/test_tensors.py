import unittest
from ffthompy.tensors import Tensor
import numpy as np
import itertools

fft_forms=['r',0,'c']


class Test_tensors(unittest.TestCase):

    def test_even(self):
        print('\nChecking Tensors with even grid points...')

        for dim, n, fft_form in itertools.product([2,3], [4,5], fft_forms):
            msg='Tensors with: dim={}, n={}, fft_form={}'.format(dim, n, fft_form)

            N=dim*(n,)
            M=tuple(2*np.array(N))

            u=Tensor(name='test', shape=(), N=N, Fourier=False, fft_form=fft_form)
            u.randomize()
            Fu=u.fourier(copy=True)
            FuM=Fu.project(M)
            uM=FuM.fourier()

            if n%2 == 0:
                self.assertGreaterEqual(u.norm(), FuM.norm(), msg=msg)
                self.assertGreaterEqual(u.norm(componentwise=True), FuM.norm(componentwise=True),
                                       msg=msg)
                self.assertGreaterEqual(u.norm(), uM.norm(), msg=msg)
                self.assertGreaterEqual(u.norm(componentwise=True), uM.norm(componentwise=True),
                                        msg=msg)
            else:
                self.assertAlmostEqual(u.norm(), FuM.norm(), msg=msg)
                self.assertAlmostEqual(u.norm(componentwise=True), FuM.norm(componentwise=True),
                                       msg=msg)
                self.assertAlmostEqual(u.norm(), uM.norm(), msg=msg)
                self.assertAlmostEqual(u.norm(componentwise=True), uM.norm(componentwise=True),
                                        msg=msg)

            self.assertAlmostEqual(0, u.mean()-FuM.mean(), msg=msg)
            self.assertAlmostEqual(u.mean(), uM.mean(), msg=msg)

            # testing that that interpolation on double grid have exactly the same values
            slc=tuple(u.order*[slice(None),]+[slice(0,M[i],2) for i in range(dim)])
            self.assertAlmostEqual(0, np.linalg.norm(u.val-uM.val[slc]), msg=msg)
        print('...ok')


if __name__ == "__main__":
    unittest.main()
