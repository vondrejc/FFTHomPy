import unittest
import numpy as np
from numpy.linalg import norm
from ffthompy.sparse.sparseTensorWrapper import SparseTensor
from ffthompy.sparse.canoTensor import CanoTensor

from ffthompy import Struct
from ffthompy.materials import Material
from ffthompy.tensors import matrix2tensor
from ffthompy.sparse.homogenisation import homog_Ga_full_potential, homog_GaNi_full_potential,homog_sparse
from ffthompy.sparse.materials import SparseMaterial

def run_full_and_sparse_solver(kind='tt', N=15, rank=10 ):
    """
    Run  full CG solver and sparse solver and return two solutions
    kind: type of sparse tensor format.
    """
    # PARAMETERS ##############################################################
    dim=3
    N=N
    material=1
    pars=Struct(kind=kind , ## type of sparse tensor
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
    else:
        raise
    
    mat=Material(mat_conf)
    mats=SparseMaterial(mat_conf, pars_sparse.kind )
     
     
    Aga=matrix2tensor(mat.get_A_Ga(Nbar(pars.N), primaldual='primal'))
    Agas=mats.get_A_Ga(Nbar(pars_sparse.N), primaldual='primal', k=2)
    
    print('\n== Full solution with potential by CG (Ga) ===========')
    resP=homog_Ga_full_potential(Aga, pars)
    print('homogenised properties (component 11) = {}'.format(resP.AH))
    
    print('\n== SPARSE Richardson solver with preconditioner =======================')
    resS=homog_sparse(Agas, pars_sparse)
    print('homogenised properties (component 11) = {}'.format(resS.AH))    

    return resP.AH, resS.AH

class Test_sparse(unittest.TestCase):

    def setUp(self):
        self.T2d=np.random.rand(10,20)
        self.T2dOther=np.random.rand(10,20)
        
        self.T3d=np.random.rand(5,7,10)
        self.T3dOther=np.random.rand(5,7,10)
        

    def tearDown(self):
        pass

    def test_canoTensor(self):
        print('\nChecking canonical tensor...')
  
        u1, s1, vt1=np.linalg.svd(self.T2d, full_matrices=0)
        a=CanoTensor(name='a', core=s1, basis=[u1.T, vt1])         
        self.assertAlmostEqual(norm(a.full()-self.T2d),0)
        
        a=SparseTensor(kind='cano',val=self.T2d)        
        self.assertAlmostEqual(norm(a.full()-self.T2d),0)

        b=SparseTensor(kind='cano',val=self.T2dOther)   
        
        self.assertAlmostEqual(norm((a+b).full()-self.T2d-self.T2dOther),0)
        self.assertAlmostEqual(norm((a*b).full()-self.T2d*self.T2dOther),0)        
        print('...ok')
        
    def test_tucker(self):
        print('\nChecking tucker ..') 
        
        a=SparseTensor(kind='tucker',val=self.T3d)        
        self.assertAlmostEqual(norm(a.full()-self.T3d),0)
        
        b=SparseTensor(kind='tucker',val=self.T3dOther)   
        
        self.assertAlmostEqual(norm((a+b).full()-self.T3d-self.T3dOther),0)
        self.assertAlmostEqual(norm((a*b).full()-self.T3d*self.T3dOther),0)
        print('...ok')

    def test_tensorTrain(self):
        print('\nChecking TT ..') 
        
        a=SparseTensor(kind='tt',val=self.T3d)        
        self.assertAlmostEqual(norm(a.full()-self.T3d),0)
        
        b=SparseTensor(kind='tt',val=self.T3dOther)   
        
        self.assertAlmostEqual(norm((a+b).full()-self.T3d-self.T3dOther),0)
        self.assertAlmostEqual(norm((a*b).full()-self.T3d*self.T3dOther),0)
        print('...ok')
        
    def test_sparse_solver(self):
        print('\nChecking sparse solver ..') 
        
        full_sol, sparse_sol=run_full_and_sparse_solver(kind='tt',N=15, rank=10) 

        self.assertTrue(abs(full_sol-sparse_sol)/full_sol < 5e-3 )

        full_sol, sparse_sol=run_full_and_sparse_solver(kind='tucker',N=15, rank=10) 

        self.assertTrue(abs(full_sol-sparse_sol)/full_sol < 5e-3 )   

        full_sol, sparse_sol=run_full_and_sparse_solver(kind='tt',N=15, rank=15) 

        self.assertTrue(abs(full_sol-sparse_sol)/full_sol < 1e-3 )

        full_sol, sparse_sol=run_full_and_sparse_solver(kind='tucker',N=15, rank=15) 

        self.assertTrue(abs(full_sol-sparse_sol)/full_sol < 1e-3 )     
        
        print('...ok')

if __name__ == "__main__":
    unittest.main() 
