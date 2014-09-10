"""
Input file for a scalar linear elliptic problems.
"""

import numpy as np

dim = 2

materials = {'square': {'inclusions': ['square', 'otherwise'],
                        'positions': [np.zeros(dim), ''],
                        'params': [0.6*np.ones(dim), ''], # size of sides
                        'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                        'Y': np.ones(dim),
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

N = np.array([5, 5])

problems = [
    {'name': 'prob1',
     'physics': 'scalar',
     # 'material': 'ball',
     # 'material': 'laminate',
     'material': 'square',
     'solve': {'kind': 'GaNi',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'GaNi'},
                     {'kind': 'Ga',
                      'order': None},
                     {'kind': 'Ga',
                      'M': N,
                      'order': 0},
                     {'kind': 'Ga',
                      'M': 3**4*N,
                      'order': 1}
                     ],
     'solver': {'kind': 'CG',
                'tol': 1e-6,
                'maxiter': 1e3}},
    {'name': 'prob2',
     'physics': 'scalar',
     # 'material': 'ball2',
     # 'material': 'laminate2',
     'material': 'square2',
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
