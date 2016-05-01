"""
Input file for a scalar linear elliptic problems.
"""

import numpy as np
from ffthompy.mechanics.matcoef import ElasticTensor
import os
from ffthompy.general.base import get_base_dir

base_dir = get_base_dir()

dim = 3
N = 5*np.ones(dim, dtype=np.int32)

matcoefM = ElasticTensor(bulk=1, mu=1)
matcoefI = ElasticTensor(bulk=10, mu=5)

materials = {'square': {'inclusions': ['square', 'otherwise'],
                        'positions': [np.zeros(dim), ''],
                        'params': [0.6*np.ones(dim), ''], # size of sides
                        'vals': [matcoefI.mandel, matcoefM.mandel],
                        'Y': np.ones(dim),
                        'order': None,
                        'P': N,
                        },
             }


problems = [
    {'name': 'prob1',
     'physics': 'elasticity',
     'material': 'square',
     'solve': {'kind': 'GaNi',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'GaNi'},
                     {'kind': 'Ga',
                      'order': None},
                     {'kind': 'Ga',
                      'order': 0,
                      'P': N},
                     {'kind': 'Ga',
                      'order': 1,
                      'P': N}],
     'solver': {'kind': 'CG',
                'tol': 1e-5,
                'maxiter': 1e3},
     'save': {'filename': os.path.join(base_dir, 'temp/linelas_3d_prob1'),
              'data': 'all'},
     },
    {'name': 'prob2',
     'physics': 'elasticity',
     'material': 'square',
     'solve': {'kind': 'Ga',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'Ga',
                      'order': None}],
     'solver': {'kind': 'CG',
                'tol': 1e-2,
                'maxiter': 1e3},
     'save': {'filename': os.path.join(base_dir, 'temp/linelas_3d_prob2'),
              'data': 'all'},
     },
            ]

if __name__=='__main__':
    import subprocess
    subprocess.call(['../../main.py', __file__])
