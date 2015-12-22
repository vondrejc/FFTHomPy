import numpy as np
import homogenize.applications
from general.base import get_base_dir, Timer
import os
import sys

class Problem(object):
    """
    Class that parse input file, calculates the physical problem,
    and post-process the results (calculates the homogenized properties.)
    """
    def __init__(self, conf_problem=None, conf=None):
        """
        Parameters
        ----------
        conf_problem : dictionary
            particular problem from problems in input file; dictionary that
            usually contains following keywords: 'name', 'material', 'solve',
            'solver', 'postprocess'
        conf : module
            configuration data from input file
        """
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
        """
        Parse material from input file.
        """
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
                        msg = "Maximal one occurrence of inclusion \
                            'otherwise' or 'all' is allowed!"
                        raise ValueError(msg)
        return material

    def calculate(self):
        """
        Calculates the problem according to physical model.
        """
        print '\n=============================='
        tim = Timer(name='application')
        if hasattr(homogenize.applications, self.physics):
            eval('homogenize.applications.%s(self)' % self.physics)
        else:
            msg = 'Not implemented physics (%s).\n' \
                'Hint: Implement function (%s) into module' \
                ' homogenize.applications!' % (self.physics, self.physics)
            raise NotImplementedError(msg)
        tim.measure()

    def postprocessing(self):
        """
        Post-process the results. Usually consists of plotting of homogenized
        properties.
        """
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
