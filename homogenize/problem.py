import numpy as np
import homogenize.applications


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
            import cPickle
            import os
            if self.save['data'] == 'all':
                filename = self.save['file']
                dirs = os.path.dirname(filename)
                if not os.path.exists(dirs):
                    os.makedirs(dirs)

                filew = open(self.save['file'], 'w')
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

if __name__ == '__main__':
    execfile('../main_test.py')
