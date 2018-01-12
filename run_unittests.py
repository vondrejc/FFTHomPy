#!/usr/bin/python

import unittest
import numpy as np
from ffthompy.problem import Problem, import_file
import pickle as Pickle
import os
import sys

class Test_main(unittest.TestCase):

    def setUp(self):
        self.input_files = ['examples/scalar/scalar_2d.py',
                            'examples/scalar/scalar_3d.py',
                            'examples/scalar/from_file.py',
                            'examples/elasticity/linelas_3d.py']
        self.tutorial_files = ['tutorials/01_trig_pol.py',
                               'tutorials/02_homogenisation.py',
                               'tutorials/03_exact_integration_simple.py',
                               'tutorials/04_exact_integration_fast.py']

    def tearDown(self):
        pass

    def test_examples(self): # testing example files
        for input_file in self.input_files:
            self.examples(input_file)

    def examples(self, input_file): # test a particular example file
        basen = os.path.basename(input_file)
        conf = import_file(input_file)

        for conf_problem in conf.problems:
            prob = Problem(conf_problem, conf)
            prob.calculate()
            py_version = sys.version_info[0]
            file_res = 'test_results/python%d/%s_%s' \
                % (py_version, basen.split('.')[0], prob.name)
            if py_version == 2:
                with open(file_res, 'r') as frs:
                    res = Pickle.load(frs)
            elif py_version == 3:
                with open(file_res, 'rb') as frs:
                    res = Pickle.load(frs)
            else:
                raise NotImplementedError('Python version!')

            # check the homogenized matrices
            for primdual in prob.solve['primaldual']:
                kwpd = 'mat_'+primdual
                for kw in prob.output[kwpd]:
                    dif = prob.output[kwpd][kw]-res[kwpd][kw]
                    val = np.linalg.norm(dif.ravel(), np.inf)
                    msg = 'Incorrect (%s) in problem (%s)' % (kw, prob.name)
                    self.assertAlmostEqual(0, val, msg=msg, delta=1e-13)
        prob.postprocessing()

    def test_tutorials(self): # test tutorials
        for filen in self.tutorial_files:
            exec(compile(open(filen).read(), filen, 'exec'), {'__name__': 'test'})

if __name__ == "__main__":
    from ffthompy.unittest_matvec import Test_matvec
    from ffthompy.unittest_operators import Test_operators
    from ffthompy.mechanics.unittest_matcoef import Test_matcoef
    from ffthompy.general.unittest_solver import Test_solvers

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test_main))
    suite.addTest(unittest.makeSuite(Test_matvec))
    suite.addTest(unittest.makeSuite(Test_operators))
    suite.addTest(unittest.makeSuite(Test_matcoef))
    suite.addTest(unittest.makeSuite(Test_solvers))

    runner=unittest.TextTestRunner()
    runner.run(suite)
