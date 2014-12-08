"""
Input file for a scalar linear elliptic problems.
"""

import numpy as np

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
                         },
             'ball': {'inclusions': ['ball', 'otherwise'],
                      'positions': [np.zeros(dim), ''],
                      'params': [1., ''], # diamater
                      'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                      'Y': 1.*np.ones(dim),
                      },
             'ball2': {'inclusions': ['ball', 'otherwise'],
                       'positions': [np.zeros(dim), ''],
                       'params': [2., ''], # diamater
                       'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                       'Y': 2.*np.ones(dim),
                       },
             'laminate': {'inclusions': ['square', 'otherwise'],
                          'positions': [np.zeros(dim), ''],
                          'params': [np.array([1., 0.5]), ''],
                          'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                          'Y': 1.*np.ones(dim),
                          },
             'laminate2': {'inclusions': ['square', 'otherwise'],
                           'positions': [np.zeros(dim), ''],
                           'params': [np.array([2., 1.0]), ''],
                           'vals': [11.*np.eye(dim), 1.*np.eye(dim)],
                           'Y': 2.*np.ones(dim),
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
                'maxiter': 1e3}
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
                'maxiter': 1e3}
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
                'maxiter': 1e3}
     },
            ]
