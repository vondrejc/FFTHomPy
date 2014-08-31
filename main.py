#!/usr/bin/python

from general.base import import_file
from homogenize.problem import Problem
from optparse import OptionParser

def main():
    parser = OptionParser()
    _, args = parser.parse_args()

    print '###################################################'
    print '## FFT-based homogenization in Python (FFTHomPy) ##'
    print '###################################################'

    if (len(args) == 1):
        input_file = args[0];
    elif (len(args) == 0):
        raise ValueError("The input argument (input file name) is missing.")
    else:
        raise ValueError("Two many input arguments")

    conf = import_file(input_file)

    for conf_problem in conf.problems:
        prob = Problem(conf_problem, conf)
        prob.calculate()
        prob.postprocessing()


if __name__ == '__main__':
    main()
    print 'END'
