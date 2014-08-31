import sys, os

def import_file(file_name):
    base_dir = os.path.dirname(os.path.normpath(os.path.realpath(__file__)))
    top_dir = os.path.normpath(os.path.join(base_dir, '..'))
    path = os.path.dirname(top_dir + '/' + file_name)

    if not path in sys.path:
        sys.path.append(path)
        remove_path = True
    else:
        remove_path = False

    name = os.path.splitext(os.path.basename(file_name))[0]

    module_name = name
    conf = __import__(module_name)

    if remove_path:
        sys.path.pop(-1)

    return conf

if __name__ == '__main__':
    execfile('../main_test.py')
