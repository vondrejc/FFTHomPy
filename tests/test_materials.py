import numpy as np
import pytest

from ffthompy.materials import Material
from ffthompy.mechanics.matcoef import ElasticTensor


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mat", ["square", "pyramid"])
def test_operators(dim, mat):
    N = dim * (5,)
    print(('...checking dim={}; material="{}"'.format(dim, mat)))
    materials = dict(
        square0={
            "inclusions": ["square", "otherwise"],
            "positions": [np.zeros(dim), ""],
            "params": [0.6 * np.ones(dim), ""],
            "vals": [10 * np.eye(dim), 1.0 * np.eye(dim)],
            "Y": np.ones(dim),
            "order": None,
        },
        square1={
            "inclusions": ["square", "otherwise"],
            "positions": [np.zeros(dim), ""],
            "params": [0.6 * np.ones(dim), ""],
            "vals": [10 * np.eye(dim), 1.0 * np.eye(dim)],
            "Y": np.ones(dim),
            "P": np.array(N),
            "order": 0,
        },
        pyramid0={
            "inclusions": ["pyramid", "all"],
            "positions": [np.zeros(dim), ""],
            "params": [0.8 * np.ones(dim), ""],
            "vals": [10 * np.eye(dim), 1.0 * np.eye(dim)],
            "Y": np.ones(dim),
            "order": None,
        },
        pyramid1={
            "inclusions": ["pyramid", "all"],
            "positions": [np.zeros(dim), ""],
            "params": [0.8 * np.ones(dim), ""],
            "vals": [10 * np.eye(dim), 1.0 * np.eye(dim)],
            "Y": np.ones(dim),
            "P": np.array(N),
            "order": 1,
        },
    )

    mat0 = Material(materials[mat + "0"])
    Aga0 = mat0.get_A_Ga(N, primaldual="primal")
    mat1 = Material(materials[mat + "1"])
    Aga1 = mat1.get_A_Ga(N, primaldual="primal")

    msg = "dim={}; material={}".format(dim, mat)
    assert np.linalg.norm(Aga0.val[0, 0] - Aga1.val[0, 0]) == pytest.approx(
        0, rel=0, abs=1e-13
    ), msg


def test_eigenvalues():

    # SCALAR PROBLEM
    dim = 2
    scalar = {
        "inclusions": ["square", "otherwise"],
        "positions": [np.zeros(dim), ""],
        "params": [0.6 * np.ones(dim), ""],
        "vals": [10 * np.eye(dim), 1.0 * np.eye(dim)],
        "N": dim * (5,),
        "Y": np.ones(dim),
        "order": None,
    }

    # ELASTICITY
    dim = 3
    matcoefM = ElasticTensor(bulk=1, mu=1)
    matcoefI = ElasticTensor(bulk=10, mu=5)

    elastic = {
        "inclusions": ["square", "otherwise"],
        "positions": [np.zeros(dim), ""],
        "params": [0.6 * np.ones(dim), ""],  # size of sides
        "vals": [matcoefI.mandel, matcoefM.mandel],
        "Y": np.ones(dim),
        "order": None,
        "N": dim * (5,),
        "P": dim * (5,),
    }

    elastic2 = {
        "inclusions": ["square", "otherwise"],
        "positions": [np.zeros(dim), ""],
        "params": [0.6 * np.ones(dim), ""],  # size of sides
        "vals": [matcoefI.val, matcoefM.val],
        "Y": np.ones(dim),
        "order": None,
        "N": dim * (5,),
        "P": dim * (5,),
    }

    for mat_conf in [scalar, elastic, elastic2]:
        mat = Material(mat_conf)
        A = mat.get_A_GaNi(mat_conf["N"], primaldual="primal")
        _ = A.calc_eigs(sort=True, symmetric=True)
