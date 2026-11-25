"""Regression checks for the shipped examples and tutorials."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from ffthompy.problem import Problem, import_file

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    "examples/scalar/scalar_2d.py",
    "examples/scalar/scalar_3d.py",
    "examples/scalar/from_file.py",
    "examples/elasticity/linelas_3d.py",
]
TUTORIALS = [
    "tutorials/01_trig_pol.py",
    "tutorials/02_homogenisation.py",
    "tutorials/03_exact_integration_simple.py",
    "tutorials/04_exact_integration_fast.py",
]


@pytest.mark.parametrize("input_file", EXAMPLES)
def test_example_regression(input_file, monkeypatch):
    monkeypatch.chdir(ROOT)
    conf = import_file(input_file)
    for conf_problem in conf.problems:
        problem = Problem(conf_problem, conf)
        problem.calculate()
        problem.postprocessing()

        baseline = ROOT / "test_results" / "python3" / f"{Path(input_file).stem}_{problem.name}"
        with baseline.open("rb") as saved:
            expected = pickle.load(saved)
        for primaldual in problem.solve["primaldual"]:
            key = f"mat_{primaldual}"
            for quantity, value in problem.output[key].items():
                np.testing.assert_allclose(value, expected[key][quantity], rtol=0, atol=1e-9)


@pytest.mark.parametrize("tutorial", TUTORIALS)
def test_tutorial(tutorial, monkeypatch):
    monkeypatch.chdir(ROOT)
    source = ROOT / tutorial
    exec(compile(source.read_text(), str(source), "exec"), {"__name__": "test"})
