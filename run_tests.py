#!/usr/bin/python

import unittest
import numpy as np
from homogenize.problem import Problem, import_file
import cPickle as Pickle
import os


class Test_main(unittest.TestCase):

    def setUp(self):
        self.input_files = ['examples/scalar/scalar_2d.py',
                            'examples/scalar/scalar_3d.py',
                            'examples/scalar/from_file.py',
                            'examples/elasticity/linelas_3d.py']

    def tearDown(self):
        pass

    def test_main(self):
        for input_file in self.input_files:
            self.main(input_file)

    def main(self, input_file):
        basen = os.path.basename(input_file)
        conf = import_file(input_file)

        for conf_problem in conf.problems:
            prob = Problem(conf_problem, conf)
            prob.calculate()
            prob.postprocessing()
            file_res = 'tests/results/%s_%s' % (basen.split('.')[0], prob.name)
            with open(file_res, 'r') as frs:
                res = Pickle.load(frs)

            # check the homogenized matrices
            for primdual in prob.solve['primaldual']:
                kwpd = 'mat_'+primdual
                for kw in prob.output[kwpd]:
                    val = np.linalg.norm(prob.output[kwpd][kw] - res[kwpd][kw])
                    msg = 'Incorrect (%s) in problem (%s)' % (kw, prob.name)
                    self.assertAlmostEqual(0, val, msg=msg, delta=1e-14)

if __name__ == "__main__":
    unittest.main()
