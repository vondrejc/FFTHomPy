"""
Input file for a scalar linear elliptic problems.
"""

import numpy as np
import os
from general.base import get_base_dir

base_dir = get_base_dir()
input_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(input_dir, 'topologie.txt')


def get_topo():
    topo = np.loadtxt(file_name)
    return topo

topo = get_topo()
N = np.array(topo.shape)
dim = N.size


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
                      'Y': np.ones(dim)}}

maxiter = 1e3
tol = 1e-4
problems = [
    {'name': 'prob1',
     'physics': 'scalar',
     'material': 'file',
     'solve': {'kind': 'GaNi',
               'N': N,
               'primaldual': ['primal', 'dual']},
     'postprocess': [{'kind': 'GaNi'}],
     'solver': {'kind': 'CG',
                'tol': tol,
                'maxiter': maxiter},
     'save': {'filename': os.path.join(base_dir, 'output/from_file_gani'),
              'data': 'all'}}]
