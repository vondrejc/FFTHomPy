"""
Input file for a scalar linear elliptic problems.
"""

import numpy as np
import os
from general.base import get_base_dir

base_dir = get_base_dir()

dim = 3
N = 5*np.ones(dim, dtype=np.int32)


materials = {'cube': {'inclusions': ['cube', 'otherwise'],
                      'positions': [np.zeros(dim), ''],
                      'params': [0.7*np.ones(dim), ''], # size of sides
                      'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                      'Y': np.ones(dim),
                      'order': None,
                      },
             'cube_Ga': {'inclusions': ['cube', 'otherwise'],
                       'positions': [np.zeros(dim), ''],
                       'params': [1.4*np.ones(dim), ''], # size of sides
                       'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                       'Y': 2.*np.ones(dim),
                       'order': 0,
                       'P': 2*N,
                       },
             'ball': {'inclusions': ['ball', 'otherwise'],
                      'positions': [np.zeros(dim), ''],
                      'params': [1., ''], # diamater
                      'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                      'Y': np.ones(dim),
                      },
             'ball2': {'inclusions': ['ball', 'otherwise'],
                       'positions': [np.zeros(dim), ''],
                       'params': [2., ''], # diamater
                       'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                       'Y': 2.*np.ones(dim),
                       },
             'laminate': {'inclusions': ['cube', 'otherwise'],
                          'positions': [np.zeros(dim), ''],
                          'params': [np.array([1., 1., 0.5]), ''],
                          'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                          'Y': np.ones(dim),
                          },
             'laminate2': {'inclusions': ['cube', 'otherwise'],
                           'positions': [np.zeros(dim), ''],
                           'params': [np.array([2., 2., 1.]), ''],
                           'vals': [11.*np.eye(dim), 1.*np.eye(dim)],
                           'Y': 2.*np.ones(dim),
                           },
             'prism': {'inclusions': ['cube', 'otherwise'],
                       'positions': [np.zeros(dim), ''],
                       'params': [np.array([1., 0.7, 0.7]), ''],
                       'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                       'Y': np.ones(dim),
                       },
             'prism2': {'inclusions': ['cube', 'otherwise'],
                        'positions': [np.zeros(dim), ''],
                        'params': [np.array([2., 1.4, 1.4]), ''],
                        'vals': [11.*np.eye(dim), 1.*np.eye(dim)],
                        'Y': 2.*np.ones(dim),
                        },
             }



problems = [
    {'name': 'prob1',
     'physics': 'scalar',
     'material': 'cube',
     'solve': {'kind': 'GaNi',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'GaNi'},
                     {'kind': 'Ga',
                      'order': None},
                     {'kind': 'Ga',
                      'order': 0,
                      'P': 3*N},
                     {'kind': 'Ga',
                      'order': 1,
                      'P': 3*N}],
     'solver': {'kind': 'CG',
                'tol': 1e-6,
                'maxiter': 1e3},
     'save': {'filename': os.path.join(base_dir, 'temp/scalar_3d_prob1'),
              'data': 'all'},
     },
    {'name': 'prob2',
     'physics': 'scalar',
     'material': 'cube',
     'solve': {'kind': 'Ga',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'Ga',
                      }],
     'solver': {'kind': 'CG',
                'tol': 1e-2,
                'maxiter': 1e3},
     'save': {'filename': os.path.join(base_dir, 'temp/scalar_3d_prob2'),
              'data': 'all'},
     },
    {'name': 'prob3',
     'physics': 'scalar',
     'material': 'cube_Ga',
     'solve': {'kind': 'Ga',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'Ga',
                      },],
     'solver': {'kind': 'CG',
                'tol': 1e-2,
                'maxiter': 1e3},
     'save': {'filename': os.path.join(base_dir, 'temp/scalar_3d_prob3'),
              'data': 'all'},
     },
            ]
