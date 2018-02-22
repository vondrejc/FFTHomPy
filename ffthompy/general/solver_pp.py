import numpy as np
from ffthompy.matvecs import VecTri
from ffthompy.tensors import Tensor


class CallBack():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.iter = -1
        self.res_norm = []
        self.energy_norm = []

    def __call__(self, x):
        self.iter += 1
        if isinstance(x, np.ndarray):
            if isinstance(self.B, VecTri):
                X = VecTri(val=np.reshape(x, self.B.dN()))
            elif isinstance(self.B, VecTri):
                X = Tensor(val=np.reshape(x, self.B.dN()), shape=self.B.shape)
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
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.iter = -1
        self.res_norm = []
        self.bound = []
        self.nonconformity = []

    def __call__(self, x):
        self.iter += 1
        if not isinstance(x, VecTri):
            X = VecTri(val=np.reshape(x, self.E2N.dN()))
        else:
            X = x

        if np.linalg.norm(X.mean() - self.E2N.mean()) < 1e-8:
            res = self.A(X)
            eN = X
        else:
            res = self.B-self.A(X)
            eN = X + self.E2N

        self.res_norm.append(res.norm())
        GeN = self.GN*eN + self.E2N
        GeN_E = GeN + self.E2N
        self.bound.append(self.Aex*GeN_E*GeN_E)
        self.nonconformity.append((GeN-eN).norm())
        return

    def __repr__(self):
        try:
            ss = ''
            ss += '    iterations    : %d\n' % self.iter
            ss += '    res_norm      : %g\n' % self.res_norm[-1]
            ss += '    bound         : %g\n' % self.bound[-1]
            ss += '    nonconformity : %g' % self.bound[-1]
        except:
            ss = 'no output'
        return ss
