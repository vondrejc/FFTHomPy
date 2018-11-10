import unittest
import numpy as np
from numpy.linalg import norm
from ffthompy import PrintControl
from ffthompy.tensors import Tensor, DFT, grad, div, symgrad, potential, Operator, grad_div_tensor
from ffthompy.tensors.projection import scalar, elasticity_small_strain, elasticity_large_deformation
import itertools

prt=PrintControl()
fft_forms=[0,'c']


class Test_operators(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_operators(self):
        print('\nChecking operators...')
        for dim, fft_form in itertools.product([2, 3], fft_forms):
            N = 3*np.ones(dim, dtype=np.int)
            F = DFT(N=N, inverse=False, fft_form=fft_form)
            iF = DFT(N=N, inverse=True, fft_form=fft_form)
            # Fourier transform
            prt.disable()
            print(F) # checking representation
            prt.enable()

            u = Tensor(name='u', shape=(1,), N=N, Fourier=False,
                       fft_form=fft_form).randomize()
            u2 = iF(F(u))
            self.assertAlmostEqual(0, (u==u2)[1], delta=1e-13,
                                   msg='Fourier tranform')

            # scalar problem
            u = Tensor(name='u', shape=(1,), N=N, Fourier=False,
                       fft_form=fft_form).randomize()
            u.val -= np.mean(u.val)
            Fu=F(u)
            Fu2 = potential(grad(Fu))
            self.assertAlmostEqual(0, (Fu==Fu2)[1], delta=1e-13,
                                   msg='scalar problem, Fourier=True')

            u2 = potential(grad(u))
            self.assertAlmostEqual(0, (u==u2)[1], delta=1e-13,
                                   msg='scalar problem, Fourier=False')

            hG, hD = grad_div_tensor(N, fft_form=fft_form)
            self.assertAlmostEqual(0, (hD(hG(Fu))==div(grad(Fu)))[1], delta=1e-13,
                                   msg='scalar problem, Fourier=True')

            # vectorial problem
            u = Tensor(name='u', shape=(dim,), N=N, Fourier=False, fft_form=fft_form)
            u.randomize()
            u.add_mean(-u.mean())

            Fu = F(u)
            Fu2 = potential(grad(Fu))
            self.assertAlmostEqual(0, (Fu==Fu2)[1], delta=1e-13,
                                   msg='vectorial problem, Fourier=True')

            u2 = potential(grad(u))
            self.assertAlmostEqual(0, (u==u2)[1], delta=1e-13,
                                   msg='vectorial problem, Fourier=False')

            # 'vectorial problem - symetric gradient
            Fu2 = potential(symgrad(Fu), small_strain=True)
            self.assertAlmostEqual(0, (Fu==Fu2)[1], delta=1e-13,
                                   msg='vectorial - sym, Fourier=True')

            u2 = potential(symgrad(u), small_strain=True)
            self.assertAlmostEqual(0, (u==u2)[1], delta=1e-13,
                                   msg='vectorial - sym, Fourier=False')

            # matrix version of DFT
            u = Tensor(name='u', shape=(1,), N=N, Fourier=False,
                       fft_form='c').randomize()
            F = DFT(N=N, inverse=False, fft_form='c')
            Fu=F(u)
            dft = F.matrix(shape=u.shape)
            Fu2 = dft.dot(u.val.ravel())
            self.assertAlmostEqual(0, norm(Fu.val.ravel()-Fu2), delta=1e-13)

        print('...ok')

    def test_compatibility(self):
        print('\nChecking compatibility...')
        for dim, fft_form in itertools.product([3],fft_forms):
            N = 5*np.ones(dim, dtype=np.int)
            F = DFT(inverse=False, N=N, fft_form=fft_form)
            iF = DFT(inverse=True, N=N, fft_form=fft_form)

            # scalar problem
            _, G1l, G2l = scalar(N, Y=np.ones(dim), NyqNul=True,
                                 fft_form=fft_form)
            P1 = Operator(name='P1', mat=[[iF, G1l, F]])
            P2 = Operator(name='P2', mat=[[iF, G2l, F]])
            u = Tensor(name='u', shape=(1,), N=N, Fourier=False, fft_form=fft_form)
            u.randomize()

            grad_u = grad(u)
            self.assertAlmostEqual(0, (P1(grad_u)-grad_u).norm(), delta=1e-13)
            self.assertAlmostEqual(0, P2(grad_u).norm(), delta=1e-13)

            e = P1(Tensor(name='u', shape=(dim,), N=N,
                          Fourier=False, fft_form=fft_form).randomize())
            e2 = grad(potential(e))
            self.assertAlmostEqual(0, (e-e2).norm(), delta=1e-13)

            # vectorial problem
            hG = elasticity_large_deformation(N=N, Y=np.ones(dim), fft_form=fft_form)
            P1 = Operator(name='P', mat=[[iF, hG, F]])
            u = Tensor(name='u', shape=(dim,), N=N, Fourier=False, fft_form=fft_form)
            u.randomize()
            grad_u = grad(u)
            val = (P1(grad_u)-grad_u).norm()
            self.assertAlmostEqual(0, val, delta=1e-13)

            e=Tensor(name='F', shape=(dim, dim), N=N, Fourier=False, fft_form=fft_form)
            e=P1(e.randomize())
            e2=grad(potential(e))
            self.assertAlmostEqual(0, (e-e2).norm(), delta=1e-13)

            # transpose
            P1TT = P1.transpose().transpose()
            self.assertTrue(P1(grad_u)==P1TT(grad_u))

            self.assertTrue(hG==(hG.transpose_left().transpose_left()))
            self.assertTrue(hG==(hG.transpose_right().transpose_right()))

            # vectorial problem - symetric gradient
            hG = elasticity_small_strain(N=N, Y=np.ones(dim), fft_form=fft_form)
            P1 = Operator(name='P', mat=[[iF, hG, F]])
            u = Tensor(name='u', shape=(dim,), N=N, Fourier=False, fft_form=fft_form)
            u.randomize()
            grad_u = symgrad(u)
            val = (P1(grad_u)-grad_u).norm()
            self.assertAlmostEqual(0, val, delta=1e-13)

            e=Tensor(name='strain', shape=(dim,dim), N=N,
                     Fourier=False, fft_form=fft_form)
            e = P1(e.randomize())
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
            prt.disable()
            print(P1)
            print(u)
            prt.enable()
            self.assertAlmostEqual(0, (P1==P1.transpose()), delta=1e-13)
        print('...ok')

if __name__ == "__main__":
    unittest.main()
