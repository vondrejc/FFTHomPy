import numpy as nm
from homogenize.matvec import VecTri


class CallBack():
    def __init__(self, A=None, B=None, E2N=None, **kwargs):
        self.iter = 0
        self.res_norm = []
        self.energy_norm = []
        self.A = A
        self.B = B
        if 'Aener' in kwargs.keys():
            self.Aener = kwargs['Aener']
        self.E2N = E2N

    def __call__(self, x):
        self.iter += 1
        if not isinstance(x, VecTri):
            X = VecTri(val=nm.reshape(x, self.B.dN()))
        else:
            X = x
        res = self.B - self.A(X)
        self.res_norm.append(res.norm())
        return

    def __repr__(self):
        try:
            ss = ''
            ss += '    iterations : %d\n' % self.iter
            ss += '    res_norm : %g' % self.res_norm[-1]
            ss += '\n'
        except:
            ss = 'the results are not initialized yet'
        return ss


class CallBack_GA():
    def __init__(self, A=None, B=None, E2N=None, **kwargs):
        self.iter = 0
        self.res_norm = []
        self.bound = []
        self.in_subspace_norm = []
        self.A = A
        self.B = B
        self.E2N = E2N
        self.Aex = kwargs['Aex']
        self.GN = kwargs['GN']
        self.primal = True

    def __call__(self, x):
        self.iter += 1
        if not isinstance(x, VecTri):
            X = VecTri(val=nm.reshape(x, self.E2N.dN()))
        else:
            X = x

        if nm.linalg.norm(X.mean() - self.E2N.mean()) < 1e-8:
            res = self.A(X)
            eN = X
        else:
            res = self.B-self.A(X)
            eN = X + self.E2N

        self.res_norm.append(res.norm())
        GeN = self.GN*eN + self.E2N
        self.bound.append(self.Aex*GeN*GeN)
        self.in_subspace_norm.append((GeN-eN).norm())
        return

    def __repr__(self):
        try:
            ss = ''
            ss += '    iterations : %d\n' % self.iter
            ss += '    res_norm : %g\n' % self.res_norm[-1]
            ss += '    bound : %g' % self.bound[-1]
        except:
            ss = 'no output'
        return ss
