from __future__ import print_function
import unittest
import numpy as np
from numpy.linalg import norm
from ffthompy.sparse.objects import SparseTensor
from ffthompy.sparse.objects import CanoTensor

from ffthompy import Struct
from ffthompy.materials import Material
from ffthompy.tensors import matrix2tensor
from ffthompy.sparse.homogenisation import homog_Ga_full_potential, homog_Ga_sparse
from ffthompy.sparse.materials import SparseMaterial
from ffthompy.tensors import Tensor

import timeit

def run_full_and_sparse_solver(kind='tt', N=15, rank=10):

    """
    Run  full CG solver and sparse solver and return two solutions
    kind: type of sparse tensor format.
    """
    # PARAMETERS ##############################################################
    dim=3
    N=N
    material=1
    pars=Struct(kind=kind, # type of sparse tensor
                dim=dim, # number of dimensions (works for 2D and 3D)
                N=dim*(N,), # number of voxels (assumed equal for all directions)
                Y=np.ones(dim),
                Amax=10., # material contrast
                maxiter=10,
                tol=None,
                rank=rank,
                solver={'tol':1e-4}
                )

    pars_sparse=pars.copy()
    if dim==2:
        pars_sparse.update(Struct(N=dim*(1*N,),))
    elif dim==3:
        pars_sparse.update(Struct(N=dim*(1*N,),))

    # auxiliary operator
    Nbar=lambda N: 2*np.array(N)-1

    # PROBLEM DEFINITION ######################################################
    if material in [0]:
        mat_conf={'inclusions': ['square', 'otherwise'],
                  'positions': [0.*np.ones(dim), ''],
                  'params': [0.6*np.ones(dim), ''], # size of sides
                  'vals': [10*np.eye(dim), 1.*np.eye(dim)],
                  'Y': np.ones(dim),
                  'P': pars.N,
                  'order': 0, }

    elif material in [1]:
        mat_conf={'inclusions': ['pyramid', 'all'],
                  'positions': [0.*np.ones(dim), ''],
                  'params': [0.8*np.ones(dim), ''], # size of sides
                  'vals': [10*np.eye(dim), 1.*np.eye(dim)],
                  'Y': np.ones(dim),
                  'P': pars.N,
                  'order': 1, }
        pars_sparse.update(Struct(matrank=2))
    else:
        raise

    mat=Material(mat_conf)
    mats=SparseMaterial(mat_conf, pars_sparse.kind)

    Agani=matrix2tensor(mat.get_A_GaNi(pars.N, primaldual='primal'))
#    Aganis=mats.get_A_GaNi(pars_sparse.N, primaldual='primal', k=pars_sparse.matrank)

    Aga=matrix2tensor(mat.get_A_Ga(Nbar(pars.N), primaldual='primal'))
    Agas=mats.get_A_Ga(Nbar(pars_sparse.N), primaldual='primal', k=2)

    pars_sparse.update(Struct(alpha=0.5*(Agani[0, 0].min()+Agani[0, 0].max())))

#    stdout_backup=sys.stdout
#    sys.stdout=open(os.devnull, "w") # stop screen output

    # print('\n== Full solution with potential by CG (Ga) ===========')
    resP=homog_Ga_full_potential(Aga, pars)
    # print('homogenised properties (component 11) = {}'.format(resP.AH))

    # print('\n== SPARSE Richardson solver with preconditioner =======================')
    # resS=homog_sparse(Agas, pars_sparse)
    # print('homogenised properties (component 11) = {}'.format(resS.AH))


#    print('\n== SPARSE Richardson solver with preconditioner =======================')
    resS=homog_Ga_sparse(Agas, pars_sparse)
#    print('homogenised properties (component 11) = {}'.format(resS.AH))

#    sys.stdout=stdout_backup # # restore screen output

    return resP.AH, resS.AH

class Test_sparse(unittest.TestCase):

    def setUp(self):
        self.T2d=np.random.rand(5, 10)
        self.T2dOther=np.random.rand(*self.T2d.shape)

        self.T3d=np.random.rand(10, 5, 20)
        self.T3dOther=np.random.rand(*self.T3d.shape)

    def tearDown(self):
        pass

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
            T = Tensor(val=self.T2d, order=0, Fourier=False, fft_form=opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=opt).val- T.fourier(copy=True).val), 0)

            a=SparseTensor(kind='tucker', val=self.T3d, fft_form=opt)
            T = Tensor(val=self.T3d, order=0, Fourier=False, fft_form=opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=opt)- T.fourier(copy=True)), 0)

            a=SparseTensor(kind='tt', val=self.T3d, fft_form=opt)
            T = Tensor(val=self.T3d, order=0, Fourier=False, fft_form=opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=opt)- T.fourier(copy=True).val), 0)

        # checking shifting fft_forms
        sparse_opt='sr'
        for full_opt in [0,'c']:

            a=SparseTensor(kind='cano', val=self.T2d, fft_form= sparse_opt)
            T = Tensor(val=self.T2d, order=0, Fourier=False, fft_form= full_opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=full_opt)- T.fourier(copy=True)), 0)
            self.assertAlmostEqual((a.fourier().set_fft_form(full_opt)-a.set_fft_form(full_opt).fourier()).norm() , 0)

            a=SparseTensor(kind='tucker', val=self.T3d, fft_form= sparse_opt)
            T = Tensor(val=self.T3d, order=0, Fourier=False, fft_form= full_opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=full_opt)- T.fourier(copy=True)), 0)
            self.assertAlmostEqual((a.fourier().set_fft_form(full_opt)-a.set_fft_form(full_opt).fourier()).norm() , 0)

            a=SparseTensor(kind='tt', val=self.T3d, fft_form= sparse_opt)
            T = Tensor(val=self.T3d, order=0, Fourier=False, fft_form= full_opt)
            self.assertAlmostEqual(norm(a.fourier().full(fft_form=full_opt)- T.fourier(copy=True)), 0)
            self.assertAlmostEqual((a.fourier().set_fft_form(full_opt)-a.set_fft_form(full_opt).fourier()).norm() , 0)

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
        #v=np.random.rand(2**L1,2**L2)
        v=np.array(range(1,2**(L1+L2+L3)+1))
        v=np.sin(v)/v # to increase the rank

        v1=np.reshape(v,(2**L1,2**L2,2**L3),order='F')
        #vFFT= DFT.fftnc(v, [2**L1, 2**L2])
        #start = time.clock()
        v1fft= np.fft.fftn(v1)/2**(L1+L2+L3)
        #print("FFT time:     ", (time.clock() - start))

        vq= np.reshape(v,[2]*(L1+L2+L3),order='F') # a quantic tensor
        vqtt= SparseTensor(kind='tt', val=vq) # a qtt

        #start = time.clock()
        vqf= vqtt.qtt_fft( [L1,L2,L3],tol= tol)
        #print("QTT_FFT time: ", (time.clock() - start))

        vqf_full=vqf.full().reshape((2**L3,2**L2,2**L1),order='F')

        print("discrepancy:  ", norm(vqf_full.T -v1fft)/norm(v1fft) )
        print ("maximum rank of the qtt is:",np.max(vqtt.r))

        self.assertTrue(norm(vqf_full.T -v1fft)/norm(v1fft) < 3*tol)

#        qtt_fft_time= timeit.timeit('vqf= vqtt.fourier() ', number=50,
#              setup="from ffthompy.sparse.objects import SparseTensor; import numpy as np; L1=9; L2=8; L3=7; tol=1e-6; v1=np.array(range(1,2**(L1+L2+L3)+1));  v1=np.sin(v1)/v1; vq= np.reshape(v1,[2]*(L1+L2+L3),order='F'); vqtt= SparseTensor(kind='tt', val=vq )")
#        print("QTT FFT time:",qtt_fft_time)

        tt_fft_time= timeit.timeit('v1f= v1tt.fourier()', number=10,
              setup="from ffthompy.sparse.objects import SparseTensor; import numpy as np; L1=9; L2=8; L3=7; v1=np.array(range(1,2**(L1+L2+L3)+1)); v1=np.sin(v1)/v1; v1tt= SparseTensor(kind='tt', val=v1,eps=1e-6 )")
        print("  TT FFT time:",tt_fft_time)

        qtt_fft_time= timeit.timeit('vqf= vqtt.qtt_fft( [L1,L2,L3],tol= tol) ', number=10,
              setup="from ffthompy.sparse.objects import SparseTensor; import numpy as np; L1=9; L2=8; L3=7; tol=1e-6; v1=np.array(range(1,2**(L1+L2+L3)+1)); v1=np.sin(v1)/v1; vq= np.reshape(v1,[2]*(L1+L2+L3),order='F'); vqtt= SparseTensor(kind='tt', val=vq )")
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
        co,ru=aSubTrain.orthogonalise(direction='rl',r_output=True)
        cr=co.to_list(co)
        for i in range(co.d):
            cr[i]=np.reshape(cr[i], (co.r[i],-1))
            I=np.eye(co.N[i])
            self.assertAlmostEqual(np.dot(cr[i], cr[i].T).any(), I.any())

        print('...ok')

#    def test_sparse_solver(self):
#        print('\nChecking sparse solver ...')
#
#        full_sol, sparse_sol=run_full_and_sparse_solver(kind='tt', N=11, rank=7)
#
#        self.assertTrue(abs(full_sol-sparse_sol)/full_sol<5e-3)
#
#        full_sol, sparse_sol=run_full_and_sparse_solver(kind='tucker', N=11, rank=7)
#
#        self.assertTrue(abs(full_sol-sparse_sol)/full_sol<5e-3)
#
#        full_sol, sparse_sol=run_full_and_sparse_solver(kind='tt', N=11, rank=10)
#
#        self.assertTrue(abs(full_sol-sparse_sol)/full_sol<1e-3)
#
#        full_sol, sparse_sol=run_full_and_sparse_solver(kind='tucker', N=11, rank=10)
#
#        self.assertTrue(abs(full_sol-sparse_sol)/full_sol<1e-3)
#
#        print('...ok')
    #@unittest.skip("The testing is too slow.")
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
