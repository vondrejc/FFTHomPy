#!/usr/bin/python

from ffthompy.problem import Problem, import_file
from optparse import OptionParser

parser = OptionParser()
_, args = parser.parse_args()

print('###################################################')
print('## FFT-based homogenization in Python (FFTHomPy) ##')
print('###################################################')

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

print('The calculation is finished!')
