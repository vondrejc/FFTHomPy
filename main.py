#!/usr/bin/python3

import logging
from optparse import OptionParser

from ffthompy.problem import Problem, import_file

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")


parser = OptionParser()
_, args = parser.parse_args()

logger.info('###################################################')
logger.info('## FFT-based homogenization in Python (FFTHomPy) ##')
logger.info('###################################################')

if (len(args) == 1):
    input_file = args[0]
elif (len(args) == 0):
    raise ValueError("The input argument (input file name) is missing.")
else:
    raise ValueError("Too many input arguments")

conf = import_file(input_file)

for conf_problem in conf.problems:
    prob = Problem(conf_problem, conf)
    prob.calculate()
    prob.postprocessing()

logger.info('The calculation is finished!')
