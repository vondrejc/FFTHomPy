import numpy as np
import os
import sys
from copy import copy, deepcopy


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


class Struct(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.__dict__.update(kwargs)


    def __str__(self):
        """Print instance class, name and items in alphabetical order.

        If the class instance has '_str_attrs' attribute, only the attributes
        listed there are taken into account. Other attributes are provided only
        as a list of attribute names (no values).

        For attributes that are Struct instances, if
        the listed attribute name ends with '.', the attribute is printed fully
        by calling str(). Otherwise only its class name/name are printed.

        Attributes that are NumPy arrays or SciPy sparse matrices are
        printed in a brief form.

        Only keys of dict attributes are printed. For the dict keys as
        well as list or tuple attributes only several edge items are
        printed if their length is greater than the threshold value 20.
        """
        return self._str()

    def _str(self, keys=None, threshold=20):
        ss = '%s' % self.__class__.__name__
        if hasattr(self, 'name'):
            ss += ':%s' % self.name
        ss += '\n'

        if keys is None:
            keys = self.__dict__.keys()

        str_attrs = sorted(Struct.get(self, '_str_attrs', keys))
        printed_keys = []
        for key in str_attrs:
            if key[-1] == '.':
                key = key[:-1]
                full_print = True
            else:
                full_print = False

            printed_keys.append(key)

            try:
                val = getattr(self, key)

            except AttributeError:
                continue

            if isinstance(val, Struct):
                if not full_print:
                    ss += '  %s:\n    %s' % (key, val.__class__.__name__)
                    if hasattr(val, 'name'):
                        ss += ':%s' % val.name
                    ss += '\n'

                else:
                    aux = '\n' + str(val)
                    aux = aux.replace('\n', '\n    ')
                    ss += '  %s:\n%s\n' % (key, aux[1:])

            elif isinstance(val, dict):
                sval = self._format_sequence(val.keys(), threshold)
                sval = sval.replace('\n', '\n    ')
                ss += '  %s:\n    dict with keys: %s\n' % (key, sval)

            elif isinstance(val, list):
                sval = self._format_sequence(val, threshold)
                sval = sval.replace('\n', '\n    ')
                ss += '  %s:\n    list: %s\n' % (key, sval)

            elif isinstance(val, tuple):
                sval = self._format_sequence(val, threshold)
                sval = sval.replace('\n', '\n    ')
                ss += '  %s:\n    tuple: %s\n' % (key, sval)

            elif isinstance(val, np.ndarray):
                ss += '  %s:\n    %s array of %s\n' \
                      % (key, val.shape, val.dtype)

            else:
                aux = '\n' + str(val)
                aux = aux.replace('\n', '\n    ')
                ss += '  %s:\n%s\n' % (key, aux[1:])

        other_keys = sorted(set(keys).difference(set(printed_keys)))
        if len(other_keys):
            ss += '  other attributes:\n    %s\n' \
                  % '\n    '.join(key for key in other_keys)

        return ss.rstrip()

    def __repr__(self):
        ss = "%s" % self.__class__.__name__
        if hasattr(self, 'name'):
            ss += ":%s" % self.name
        return ss

    def __add__(self, other):
        new = copy(self)
        for key, val in other.__dict__.iteritems():
            if hasattr(new, key):
                sval = getattr(self, key)
                if issubclass(sval.__class__, Struct) and \
                        issubclass(val.__class__, Struct):
                    setattr(new, key, sval + val)
                else:
                    setattr(new, key, val)
            else:
                setattr(new, key, val)
        return new

    def str_class(self):
        return self._str(self.__class__.__dict__.keys())

    def str_all(self):
        ss = "%s\n" % self.__class__
        for key, val in self.__dict__.iteritems():
            if issubclass(self.__dict__[key].__class__, Struct):
                ss += "  %s:\n" % key
                aux = "\n" + self.__dict__[key].str_all()
                aux = aux.replace("\n", "\n    ")
                ss += aux[1:] + "\n"
            else:
                aux = "\n" + str(val)
                aux = aux.replace("\n", "\n    ")
                ss += "  %s:\n%s\n" % (key, aux[1:])
        return(ss.rstrip())

    def to_dict(self):
        return copy(self.__dict__)

    def get(self, key, default=None, msg_if_none=None):
        out = getattr(self, key, default)

        if (out is None) and (msg_if_none is not None):
            raise ValueError(msg_if_none)

        return out

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def update(self, other, **kwargs):
        if other is None: return

        if not isinstance(other, dict):
            other = other.to_dict()
        self.__dict__.update(other, **kwargs)

    def set_default(self, key, default=None):
        return self.__dict__.setdefault(key, default)

    def copy(self, deep=False, name=None):
        if deep:
            other = deepcopy(self)
        else:
            other = copy(self)

        if hasattr(self, 'name') and name is not None:
            other.name = self.name + '_copy'

        return other

    def to_array(self):
        log = deepcopy(self)
        for key, val in log.__dict__.iteritems():
            try:
                log.__dict__.update({key: np.array(val)})
            except:
                pass
        return log

if __name__ == '__main__':
    run_file()
