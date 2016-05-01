"""
Input file for a scalar linear elliptic problems.
"""

import numpy as np
import os
from homogenize.general.base import get_base_dir

base_dir = get_base_dir()
input_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(input_dir, 'topologie.txt')


def get_topo():
    topo = np.loadtxt(file_name)
    return topo

topo = get_topo()
P = np.array(topo.shape) # image resolution
dim = P.size


def get_mat(coord=None):
    topo = get_topo()
    if coord is not None and not (topo.shape == coord.shape[1:]):
        raise ValueError()
    matrix_phase = np.eye(dim)
    inclusion = 11.*np.eye(dim)
    mat_vals = np.einsum('ij...,k...->ijk...', matrix_phase, topo == 0)
    mat_vals += np.einsum('ij...,k...->ijk...', inclusion, topo == 1)
    return mat_vals

materials = {'file': {'fun': get_mat,
                      'Y': np.ones(dim),
                      'order': 0,
                      'P': P}}

maxiter = 1e3
tol = 1e-6

problems = [
    {'name': 'prob1',
     'physics': 'scalar',
     'material': 'file',
     'solve': {'kind': 'GaNi',
               'N': P,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'GaNi'},
                     {'kind': 'Ga',
                      'order': 0,
                      'P': P},
                     {'kind': 'Ga',
                      'order': 1,
                      'P': P}],
     'solver': {'kind': 'CG',
                'tol': tol,
                'maxiter': maxiter},
     'save': {'filename': os.path.join(base_dir, 'temp/from_file_prob1'),
              'data': 'all'}},
    {'name': 'prob2',
     'physics': 'scalar',
     'material': 'file',
     'solve': {'kind': 'Ga',
               'N': P,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'Ga'}],
     'solver': {'kind': 'CG',
                'tol': tol,
                'maxiter': maxiter},
     'save': {'filename': os.path.join(base_dir, 'temp/from_file_prob2'),
              'data': 'all'}}
            ]

if __name__=='__main__':
    import subprocess
    subprocess.call(['../../main.py', __file__])
