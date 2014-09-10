"""
Input file for a scalar linear elliptic problems.
"""

import numpy as np

dim = 3

materials = {'cube': {'inclusions': ['cube', 'otherwise'],
                      'positions': [np.zeros(dim), ''],
                      'params': [0.7*np.ones(dim), ''], # size of sides
                      'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                      'Y': np.ones(dim),
                      },
             'cube2': {'inclusions': ['cube', 'otherwise'],
                       'positions': [np.zeros(dim), ''],
                       'params': [1.4*np.ones(dim), ''], # size of sides
                       'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                       'Y': 2.*np.ones(dim),
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

N = 5*np.ones(dim, dtype=np.int32)

M = 10*np.ones(dim, dtype=np.int32)

problems = [
    {'name': 'prob1',
     'physics': 'scalar',
     # 'material': 'ball',
     'material': 'prism',
     # 'material': 'laminate',
     # 'material': 'cube',
     'solve': {'kind': 'GaNi',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'GaNi'},
                     {'kind': 'Ga',
                      'order': None},
                     {'kind': 'Ga',
                      'M': M,
                      'order': 0},
                     {'kind': 'Ga',
                      'M': 3*M,
                      'order': 1}
                     ],
     'solver': {'kind': 'CG',
                'tol': 1e-6,
                'maxiter': 1e3}},
    {'name': 'prob2',
     'physics': 'scalar',
     # 'material': 'ball2',
     'material': 'prism2',
     # 'material': 'laminate2',
     # 'material': 'cube2',
     'solve': {'kind': 'Ga',
               'N': N,
               'order': None,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'Ga',
                      'order': None}],
     'solver': {'kind': 'CG',
                'tol': 1e-6,
                'maxiter': 1e3}
     }, ]
