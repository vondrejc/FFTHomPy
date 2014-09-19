import os


def get_base_dir():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(module_dir, '..'))
    return base_dir

if __name__ == '__main__':
    execfile('../main_test.py')
