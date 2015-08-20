import os
import sys


def get_base_dir():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(module_dir, '..'))
    return base_dir

def run_file(filen=''):
    base_dir = get_base_dir()
    main = base_dir + '/main.py'
    sys.argv = [main, filen]
    execfile(main)

def print_dict(d):
    print '-- print dictionary -------------------'
    for key, vals in d.items():
        print key + ' =', vals

def end():
    print 'end'
    sys.exit()

if __name__ == '__main__':
    run_file()
