"""
Input file for a scalar linear elliptic problems.
"""

import numpy as np

dim = 2

materials = {'square': {'inclusions': ['square', 'otherwise'],
                        'positions': [np.zeros(dim), ''],
                        'params': [1.4*np.ones(dim), ''], # size of sides
                        'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                        'Y': np.array([2., 2.]),
                        },
             'square2': {'inclusions': ['square', 'otherwise'],
                         'positions': [np.zeros(dim), ''],
                         'params': [2.8*np.ones(dim), ''], # size of sides
                         'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                         'Y': np.array([4., 4.]),
                         },
             'ball': {'inclusions': ['ball', 'otherwise'],
                      'positions': [np.zeros(dim), ''],
                      'params': [2., ''], # diamater
                      'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                      'Y': np.array([2., 2.]),
                      },
             'ball2': {'inclusions': ['ball', 'otherwise'],
                       'positions': [np.zeros(dim), ''],
                       'params': [4., ''], # diamater
                       'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                       'Y': np.array([4., 4.]),
                       },
             'laminate': {'inclusions': ['square', 'otherwise'],
                          'positions': [np.zeros(dim), ''],
                          'params': [np.array([2., 1.0]), ''],
                          'vals': [11*np.eye(dim), 1.*np.eye(dim)],
                          'Y': np.array([2., 2.]),
                          },
             'laminate2': {'inclusions': ['square', 'otherwise'],
                           'positions': [np.zeros(dim), ''],
                           'params': [np.array([4., 2.0]), ''],
                           'vals': [11.*np.eye(dim), 1.*np.eye(dim)],
                           'Y': np.array([4., 4.]),
                           },
             }

N = np.array([5, 5])

Nbar = np.array([145, 145])

problems = [
    {'name': 'prob1',
     'physics': 'scalar',
     'material': 'ball',
     # 'material': 'laminate',
     # 'material': 'square',
     'solve': {'kind': 'GaNi',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'GaNi',
                      'N': N},
                     {'kind': 'Ga',
                      'N': Nbar,
                      'order': None},
                     {'kind': 'Ga',
                      'N': Nbar,
                      'order': 0},
                     {'kind': 'Ga',
                      'N': Nbar,
                      'order': 1}
                     ],
     'solver': {'kind': 'CG',
                'tol': 1e-6,
                'maxiter': 1e3}},
    {'name': 'prob2',
     'physics': 'scalar',
     'material': 'ball2',
     #     'material': 'laminate2',
     #     'material': 'square2',
     'solve': {'kind': 'Ga',
               'N': N,
               'order': None,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'Ga',
                      'N': N,
                      'order': None}],
     'solver': {'kind': 'CG',
                'tol': 1e-6,
                'maxiter': 1e3}
     },
           ]
