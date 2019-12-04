
import unittest
import numpy as np
from numpy.linalg import norm
from ffthompy.tensorsLowRank.objects import SparseTensor
from ffthompy.tensorsLowRank.objects import CanoTensor

from ffthompy import PrintControl, Timer
from ffthompy.tensors import Tensor
from examples.lowRankTensorApproximations.setting import get_default_parameters, get_material_coef
from ffthompy.tensorsLowRank.homogenisation import (homog_Ga_full_potential, homog_GaNi_full_potential,
                                                    homog_Ga_sparse, homog_GaNi_sparse)

import timeit

prt=PrintControl()


class Test_tensorsLowRank(unittest.TestCase):

    def setUp(self):
        self.T2d=np.random.rand(5, 10)
        self.T2dOther=np.random.rand(*self.T2d.shape)

        self.T3d=np.random.rand(10, 5, 20)
        self.T3dOther=np.random.rand(*self.T3d.shape)

    def tearDown(self):
        pass

    def test_sparse_solver(self):
        print('\nChecking homogenisation with low-rank tensor approximations...')

        tic=Timer('homogenisation')
        N=5
        for dim, material, kind in [(2,0,0), (2,3,1), (3,1,1), (3,0,2)]:
            print('dim={}, material={}, kind={}'.format(dim, material, kind))
            pars, pars_sparse=get_default_parameters(dim, N, material, kind)
            pars_sparse.debug=True
            pars_sparse.solver.update(dict(rank=5, maxiter=10))

            prt.disable()

            Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse)

            if material in [0, 3]:
                resP_Ga=homog_Ga_full_potential(Aga, pars)
                resS_Ga=homog_Ga_sparse(Agas, pars_sparse)

            if material in [1, 2, 4]:
                resP_GaNi=homog_GaNi_full_potential(Agani, Aga, pars)
                resS_GaNi=homog_GaNi_sparse(Aganis, Agas, pars_sparse)

            prt.enable()

            if material in [0, 3]:
                self.assertAlmostEqual(resP_Ga.Fu.mean(), 0)
                self.assertAlmostEqual(resS_Ga.Fu.mean(), 0)
                self.assertAlmostEqual(np.abs(resP_Ga.AH-resS_Ga.AH), 0, delta=5e-3)

            if material in [1, 2, 4]:
                self.assertAlmostEqual(resP_GaNi.Fu.mean(), 0)
                self.assertAlmostEqual(resS_GaNi.Fu.mean(), 0)
                self.assertAlmostEqual(np.abs(resP_GaNi.AH-resS_GaNi.AH), 0, delta=5e-3)

        tic.measure()
        print('...ok')

    def test_canoTensor(self):
        print('\nChecking canonical tensor...')

        u1, s1, vt1=np.linalg.svd(self.T2d, full_matrices=0)
        a=CanoTensor(name='a', core=s1, basis=[u1.T, vt1])
        self.assertAlmostEqual(norm(a.full()-self.T2d), 0)

        a=SparseTensor(kind='cano', val=self.T2d)
        self.assertAlmostEqual(norm(a.full()-self.T2d), 0)

        b=SparseTensor(kind='cano', val=self.T2dOther)

        self.assertAlmostEqual(norm((a+b).full()-self.T2d-self.T2dOther), 0)
        self.assertAlmostEqual(norm((a*b).full()-self.T2d*self.T2dOther), 0)

        print('...ok')

    def test_tucker(self):
        print('\nChecking tucker ...')

        a=SparseTensor(kind='tucker', val=self.T3d)
        self.assertAlmostEqual(norm(a.full()-self.T3d), 0)

        b=SparseTensor(kind='tucker', val=self.T3dOther)

        self.assertAlmostEqual(norm((a+b).full()-self.T3d-self.T3dOther), 0)
        self.assertAlmostEqual(norm((a*b).full()-self.T3d*self.T3dOther), 0)
        print('...ok')

    def test_tensorTrain(self):
        print('\nChecking TT ...')

        a=SparseTensor(kind='tt', val=self.T3d)
        self.assertAlmostEqual(norm(a.full()-self.T3d), 0)

        b=SparseTensor(kind='tt', val=self.T3dOther)

        self.assertAlmostEqual(norm((a+b).full()-self.T3d-self.T3dOther), 0)
        self.assertAlmostEqual(norm((a*b).full()-self.T3d*self.T3dOther), 0)

        print('...ok')

    def test_Fourier(self):
        print('\nChecking Fourier functions ...')

        for opt in [0,'c']:

            a=SparseTensor(kind='cano', val=self.T2d, fft_form=opt)
            T=Tensor(val=self.T2d, order=0, N=self.T2d.shape, Fourier=False, fft_form=opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=opt).val-T.fourier(copy=True).val), 0)
            self.assertEqual(norm(a.fourier().fourier(real_output=True).full().val.imag), 0)

            a=SparseTensor(kind='tucker', val=self.T3d, fft_form=opt)
            T=Tensor(val=self.T3d, order=0, N=self.T3d.shape, Fourier=False, fft_form=opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=opt)-T.fourier(copy=True)), 0)
            self.assertEqual(norm(a.fourier().fourier(real_output=True).full().val.imag), 0)

            a=SparseTensor(kind='tt', val=self.T3d, fft_form=opt)
            T=Tensor(val=self.T3d, order=0, N=self.T3d.shape, Fourier=False, fft_form=opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=opt)-T.fourier(copy=True).val), 0)
            self.assertEqual(norm(a.fourier().fourier(real_output=True).full().val.imag), 0)
        # checking shifting fft_forms
        sparse_opt='sr'
        for full_opt in [0, 'c']:

            a=SparseTensor(kind='cano', val=self.T2d, fft_form=sparse_opt)
            T=Tensor(val=self.T2d, order=0, N=self.T2d.shape, Fourier=False, fft_form=full_opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=full_opt)-T.fourier(copy=True)), 0)
            self.assertAlmostEqual((a.fourier().set_fft_form(full_opt)-a.set_fft_form(full_opt).fourier()).norm(), 0)

            a=SparseTensor(kind='tucker', val=self.T3d, fft_form=sparse_opt)
            T=Tensor(val=self.T3d, order=0, N=self.T3d.shape, Fourier=False, fft_form=full_opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=full_opt)-T.fourier(copy=True)), 0)
            self.assertAlmostEqual((a.fourier().set_fft_form(full_opt)-a.set_fft_form(full_opt).fourier()).norm(), 0)

            a=SparseTensor(kind='tt', val=self.T3d, fft_form=sparse_opt)
            T=Tensor(val=self.T3d, order=0, N=self.T3d.shape, Fourier=False, fft_form=full_opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=full_opt)-T.fourier(copy=True)), 0)
            self.assertAlmostEqual((a.fourier().set_fft_form(full_opt)-a.set_fft_form(full_opt).fourier()).norm(), 0)

        print('...ok')

    def test_Fourier_truncation(self):
        print('\nChecking TT truncation in Fourier domain ...')
        N = np.random.randint(20,50, size=3)
        a=np.arange(1,np.prod(N)+1).reshape(N)
        cases=[[None]*2,[None]*2]
        # first a random test case
        cases[0]=[np.random.random(N),np.random.random(N)]
        # this produces a "smooth", more realistic, tensor with modest rank
        cases[1]=[np.sin(a)/a, np.exp(np.sin(a)/a)]

        for i in range(len(cases)):
            for fft_form in [0,'c', 'sr']:

                a=cases[i][0]
                b=cases[i][1]
                ta=SparseTensor(kind='tt', val=a, fft_form=fft_form) # Fourier truncation works the best with option 'sr'
                tb=SparseTensor(kind='tt', val=b, fft_form=fft_form)
                tc=ta+tb
                k=tc.r[1:-1].max()/2-5

                tct=tc.truncate(rank=k)

                err_normal_truncate= (tct-tc).norm()
#                print("loss in normal  domain truncation:",norm(tct.full().val-(a+b) ))

                taf=ta.fourier()
                tbf=tb.fourier()
                tcf=taf+tbf
                tcft=tcf.truncate(rank=k)
                tcfti=tcft.fourier()
#                print("norm of imag part of F inverse tensor",norm(tcfti.full().val.imag))

                err_Fourier_truncate=(tcfti-tc).norm()
#                print("loss in Fourier domain truncation:",norm(tcfti.full().val-(a+b) ))

                # assert the two truncation errors are in the same order
                self.assertAlmostEqual(err_normal_truncate, err_Fourier_truncate, delta=err_normal_truncate*3)

        print('...ok')

    @unittest.skip("The testing is too slow.")
    def test_qtt_fft(self):
        print('\nChecking QTT FFT functions ...')
        L1=3
        L2=4
        L3=5
        tol=1e-6
        # v=np.random.rand(2**L1,2**L2)
        v=np.array(list(range(1, 2**(L1+L2+L3)+1)))
        v=np.sin(v)/v # to increase the rank

        v1=np.reshape(v, (2**L1, 2**L2, 2**L3), order='F')
        # vFFT= DFT.fftnc(v, [2**L1, 2**L2])
        # start = time.clock()
        v1fft=np.fft.fftn(v1)/2**(L1+L2+L3)
        # print("FFT time:     ", (time.clock() - start))

        vq=np.reshape(v, [2]*(L1+L2+L3), order='F') # a quantic tensor
        vqtt=SparseTensor(kind='tt', val=vq) # a qtt

        # start = time.clock()
        vqf=vqtt.qtt_fft([L1, L2, L3], tol=tol)
        # print("QTT_FFT time: ", (time.clock() - start))

        vqf_full=vqf.full().reshape((2**L3,2**L2,2**L1),order='F')

        print("discrepancy:  ", norm(vqf_full.T -v1fft)/norm(v1fft))
        print ("maximum rank of the qtt is:",np.max(vqtt.r))

        self.assertTrue(norm(vqf_full.T -v1fft)/norm(v1fft) < 3*tol)

#        qtt_fft_time= timeit.timeit('vqf= vqtt.fourier() ', number=50,
#              setup="from ffthompy.tensorsLowRank.objects import SparseTensor; import numpy as np; L1=9; L2=8; L3=7; tol=1e-6; v1=np.array(range(1,2**(L1+L2+L3)+1));  v1=np.sin(v1)/v1; vq= np.reshape(v1,[2]*(L1+L2+L3),order='F'); vqtt= SparseTensor(kind='tt', val=vq )")
#        print("QTT FFT time:",qtt_fft_time)

        tt_fft_time= timeit.timeit('v1f= v1tt.fourier()', number=10,
              setup="from ffthompy.tensorsLowRank.objects import SparseTensor; import numpy as np; L1=9; L2=8; L3=7; v1=np.array(range(1,2**(L1+L2+L3)+1)); v1=np.sin(v1)/v1; v1tt= SparseTensor(kind='tt', val=v1,eps=1e-6 )")
        print("  TT FFT time:",tt_fft_time)

        qtt_fft_time= timeit.timeit('vqf= vqtt.qtt_fft( [L1,L2,L3],tol= tol) ', number=10,
              setup="from ffthompy.tensorsLowRank.objects import SparseTensor; import numpy as np; L1=9; L2=8; L3=7; tol=1e-6; v1=np.array(range(1,2**(L1+L2+L3)+1)); v1=np.sin(v1)/v1; vq= np.reshape(v1,[2]*(L1+L2+L3),order='F'); vqtt= SparseTensor(kind='tt', val=vq )")
        print("QTT FFT time:",qtt_fft_time)

        self.assertTrue(qtt_fft_time < 0.1*tt_fft_time)

        print('...ok')

    def test_orthogonalise(self):
        print('\nChecking orthogonalization functions ...')

        a=SparseTensor(kind='cano', val=self.T2d)
        b=SparseTensor(kind='cano', val=self.T2dOther)
        c=a+b
        co=c.orthogonalise()
        for i in range(co.order):
            I=np.eye(co.N[i])
            self.assertAlmostEqual(np.dot(co.basis[i], co.basis[i].T).any(), I.any())

        a=SparseTensor(kind='tucker', val=self.T3d)
        b=SparseTensor(kind='tucker', val=self.T3dOther)
        c=a+b
        co=c.orthogonalise()
        for i in range(co.order):
            I=np.eye(co.N[i])
            self.assertAlmostEqual(np.dot(co.basis[i], co.basis[i].T).any(), I.any())

        a=SparseTensor(kind='tt', val=self.T3d)
        b=SparseTensor(kind='tt', val=self.T3dOther)
        c=a+b
        co=c.orthogonalise(direction='lr')
        cr=co.to_list(co)
        for i in range(co.d):
            cr[i]=np.reshape(cr[i], (-1, co.r[i+1]))
            I=np.eye(co.N[i])
            self.assertAlmostEqual(np.dot(cr[i].T, cr[i]).any(), I.any())

        co=c.orthogonalise(direction='rl')
        cr=co.to_list(co)
        for i in range(co.d):
            cr[i]=np.reshape(cr[i], (co.r[i],-1))
            I=np.eye(co.N[i])
            self.assertAlmostEqual(np.dot(cr[i], cr[i].T).any(), I.any())

        aSubTrain=c.tt_chunk(0,1)
        co, ru=aSubTrain.orthogonalise(direction='rl', r_output=True)
        cr=co.to_list(co)
        for i in range(co.d):
            cr[i]=np.reshape(cr[i], (co.r[i],-1))
            I=np.eye(co.N[i])
            self.assertAlmostEqual(np.dot(cr[i], cr[i].T).any(), I.any())

        print('...ok')

    def test_mean(self):
        print('\nChecking method mean() ...')
        a=SparseTensor(kind='cano', val=self.T2d)
        self.assertAlmostEqual(np.mean(self.T2d), a.mean())
        self.assertAlmostEqual(np.mean(self.T2d), a.fourier().mean())

        a=SparseTensor(kind='tucker', val=self.T3d)
        self.assertAlmostEqual(np.mean(self.T3d), a.mean())
        self.assertAlmostEqual(np.mean(self.T3d), a.fourier().mean())
#
        a=SparseTensor(kind='tt', val=self.T3d)
        self.assertAlmostEqual(np.mean(self.T3d), a.mean())
        self.assertAlmostEqual(np.mean(self.T3d), a.fourier().mean())
        print('...ok')

if __name__ == "__main__":
    unittest.main()
