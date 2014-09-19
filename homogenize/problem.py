import numpy as np
import homogenize.applications
from general.base import get_base_dir
import os
import sys


class Problem(object):
    def __init__(self, conf_problem=None, conf=None):
        self.__dict__.update(conf_problem)
        if isinstance(self.material, str):
            conf_material = conf.materials[self.material]
        else:
            conf_material = self.material

        self.material = self.parse_material(conf_material)

        self.Y = np.array(self.material['Y'], dtype=np.float64)
        self.dim = self.Y.size

        if self.physics == 'scalar':
            self.shape = (self.dim)
        elif self.physics == 'elasticity':
            self.shape = (self.dim*(self.dim+1)/2)

        self.output = {}

    @staticmethod
    def parse_material(conf_material):
        if 'fun' in conf_material:
            material = conf_material
        else:
            material = conf_material
            for incl in material['inclusions']:
                if incl in ['all', 'otherwise']:
                    n_incl = material['inclusions'].count(incl)
                    if n_incl == 0:
                        continue
                    elif n_incl == 1:
                        ind = material['inclusions'].index(incl)
                        if ind == n_incl:
                            continue
                        for _, vals in material.iteritems():
                            val = vals.pop(ind)
                            vals.append(val)
                    else:
                        raise ValueError()
        return material

    def calculate(self):
        print '\n=============================='
        if self.physics == 'scalar':
            homogenize.applications.scalar(self)
        elif self.physics == 'elasticity':
            homogenize.applications.elasticity(self)
        else:
            raise ValueError("Not implemented physics (%s)." % self.physics)

    def postprocessing(self):
        output = self.output
        if self.physics in ['scalar', 'elasticity']:
            print '\nHomogenized matrices'
            for primaldual in self.solve['primaldual']:
                for key, val in output['mat_'+primaldual].iteritems():
                    print key
                    print val

        if hasattr(self, 'save'):
            if 'data' not in self.save:
                self.save['data'] = 'all'

            if self.save['data'] == 'all':
                data = self.output
                data.update({'physics': self.physics,
                             'solve': self.solve,
                             'solver': self.solver,
                             'postprocess': self.postprocess,
                             'save': self.save,
                             'material': self.material})

            filename = self.save['filename']
            dirs = os.path.dirname(filename)
            if not os.path.exists(dirs) and dirs != '':
                os.makedirs(dirs)

            import cPickle
            filew = open(filename, 'w')
            cPickle.dump(self.output, filew)
            filew.close()

    def __repr__(self):
        ss = "Class : %s\n" % self.__class__.__name__
        ss += '    name : %s\n' % self.name
        ss += '    physics = %s\n' % (self.physics)
        ss += '    dim = %d (dimension)\n' % (self.dim)
        ss += '    Y = %s (PUC size)\n' % str(self.Y)
        ss += '    material:\n'
        for key, val in self.material.iteritems():
            ss += '        %s : %s\n' % (key, str(val))
        ss += '    solve:\n'
        for key, val in self.solve.iteritems():
            ss += '        %s : %s\n' % (key, str(val))
        return ss


def import_file(file_name):
    base_dir = get_base_dir()
    module_path = os.path.dirname(os.path.join(base_dir, file_name))

    if module_path not in sys.path:
        sys.path.append(module_path)
        remove_path = True
    else:
        remove_path = False

    module_name = os.path.splitext(os.path.basename(file_name))[0]

    conf = __import__(module_name)

    if remove_path:
        sys.path.pop(-1)

    return conf

if __name__ == '__main__':
    execfile('../main_test.py')
