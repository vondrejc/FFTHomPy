"""
Input file for a scalar linear elliptic problems.
"""

import numpy as np
from mechanics.matcoef import ElasticTensor


dim = 3

matcoefM = ElasticTensor(bulk=1, mu=1)
matcoefI = ElasticTensor(bulk=10, mu=5)

materials = {'square': {'inclusions': ['square', 'otherwise'],
                        'positions': [np.zeros(dim), ''],
                        'params': [0.6*np.ones(dim), ''], # size of sides
                        'vals': [matcoefI.mandel, matcoefM.mandel],
                        'Y': np.ones(dim),
                        },
             }

N = 5*np.ones(dim, dtype=np.int32)

problems = [
    {'name': 'prob1',
     'physics': 'elasticity',
     'material': 'square',
     'solve': {'kind': 'GaNi',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'GaNi'}],
     'solver': {'kind': 'CG',
                'tol': 1e-2,
                'maxiter': 1e3}}]
