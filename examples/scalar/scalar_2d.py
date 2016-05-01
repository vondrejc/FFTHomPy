"""
Input file for a scalar linear elliptic problems.
"""

import numpy as np
import os
from ffthompy.general.base import get_base_dir

base_dir = get_base_dir()

dim = 2
N = 5*np.ones(dim, dtype=np.int32)

materials = {'square': {'inclusions': ['square', 'otherwise'],
                        'positions': [np.zeros(dim), ''],
                        'params': [0.6*np.ones(dim), ''], # size of sides
                        'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                        'Y': np.ones(dim),
                        'order': None,
                        },
             'square_Ga': {'inclusions': ['square', 'otherwise'],
                           'positions': [np.zeros(dim), ''],
                           'params': [0.6*np.ones(dim), ''], # size of sides
                           'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                           'Y': np.ones(dim),
                           'order': 0,
                           'P': N,
                        },
             'square2': {'inclusions': ['square', 'otherwise'],
                         'positions': [np.zeros(dim), ''],
                         'params': [1.2*np.ones(dim), ''], # size of sides
                         'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                         'Y': 2.*np.ones(dim),
                         'order': None
                         },
             'ball': {'inclusions': ['ball', 'otherwise'],
                      'positions': [np.zeros(dim), ''],
                      'params': [1., ''], # diamater
                      'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                      'Y': 1.*np.ones(dim),
                      'order': None
                      },
             'ball2': {'inclusions': ['ball', 'otherwise'],
                       'positions': [np.zeros(dim), ''],
                       'params': [2., ''], # diamater
                       'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                       'Y': 2.*np.ones(dim),
                       'order': None
                       },
             'laminate': {'inclusions': ['square', 'otherwise'],
                          'positions': [np.zeros(dim), ''],
                          'params': [np.array([1., 0.5]), ''],
                          'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                          'Y': 1.*np.ones(dim),
                          'order': None
                          },
             'laminate2': {'inclusions': ['square', 'otherwise'],
                           'positions': [np.zeros(dim), ''],
                           'params': [np.array([2., 1.0]), ''],
                           'vals': [11.*np.eye(dim), 1.*np.eye(dim)],
                           'Y': 2.*np.ones(dim),
                           'order': None
                           },
             'pyramid': {'inclusions': ['pyramid', 'all'],
                         'positions': [np.zeros(dim), ''],
                         'params': [1.*np.ones(dim), ''], # size of sides
                         'vals': [10.*np.eye(dim), 1.*np.eye(dim)],
                         'Y': np.ones(dim),
                         'order': None,
                         },
             }


problems = [
    {'name': 'prob1',
     'physics': 'scalar',
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
                      'P': 27*N}],
     'solver': {'kind': 'CG',
                'tol': 1e-6,
                'maxiter': 1e3},
     'save': {'filename': os.path.join(base_dir, 'temp/scalar_2d_prob1'),
              'data': 'all'},
     },
    {'name': 'prob2',
     'physics': 'scalar',
     'material': 'square',
     'solve': {'kind': 'Ga',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'Ga',
                      }],
     'solver': {'kind': 'CG',
                'tol': 1e-2,
                'maxiter': 1e3},
     'save': {'filename': os.path.join(base_dir, 'temp/scalar_2d_prob2'),
              'data': 'all'},
     },
    {'name': 'prob3',
     'physics': 'scalar',
     'material': 'square_Ga',
     'solve': {'kind': 'Ga',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'Ga',
                      },],
     'solver': {'kind': 'CG',
                'tol': 1e-2,
                'maxiter': 1e3},
     'save': {'filename': os.path.join(base_dir, 'temp/scalar_2d_prob3'),
              'data': 'all'},
     },
    {'name': 'prob4',
     'physics': 'scalar',
     'material': 'pyramid',
     'solve': {'kind': 'Ga',
               'N': N,
               'primaldual': ['primal']},
     'postprocess': [{'kind': 'Ga',
                      'order': None},
                     {'kind': 'Ga',
                      'order': 0,
                      'P': N},
                     {'kind': 'Ga',
                      'order': 1,
                      'P': N}],
     'solver': {'kind': 'CG',
                'tol': 1e-6,
                'maxiter': 1e3},
     'save': {'filename': os.path.join(base_dir, 'temp/scalar_2d_prob4'),
              'data': 'all'},
     },
    {'name': 'prob5',
     'physics': 'scalar',
     'material': 'ball',
     'solve': {'kind': 'Ga',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'Ga',
                      'order': None},
                     {'kind': 'Ga',
                      'order': 0,
                      'P': N}],
     'solver': {'kind': 'CG',
                'tol': 1e-6,
                'maxiter': 1e3},
     'save': {'filename': os.path.join(base_dir, 'temp/scalar_2d_prob5'),
              'data': 'all'},
     },
            ]

if __name__=='__main__':
    import subprocess
    subprocess.call(['../../main.py', __file__])
