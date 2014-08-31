import numpy as np
import scipy.special as sp
from homogenize.matvec import DFT, VecTri, Matrix
from homogenize.trigonometric import TrigPolynomial

inclusion_keys = {'ball': ['ball', 'circle'],
                  'cube': ['cube', 'square']}


class Material():

    def __init__(self, problem_conf):
        self.conf = problem_conf.material
        self.Y = problem_conf.Y

    def get_A_Ga(self, N, order=None, primaldual='primal'):
        if order is None:
            shape_funs = self.get_shape_functions(N)
            val = np.zeros(self.conf['vals'][0].shape + shape_funs[0].shape)
            for ii in range(len(self.conf['inclusions'])):
                if primaldual is 'primal':
                    Aincl = self.conf['vals'][ii]
                elif primaldual is 'dual':
                    Aincl = np.linalg.inv(self.conf['vals'][ii])
                val += np.einsum('ij...,k...->ijk...', Aincl, shape_funs[ii])
            return Matrix(name='A_Ga', val=val)
        else:
            dim = np.size(N)
            coord = TrigPolynomial.get_grid_coordinates(N, self.Y)
            vals = self.evaluate(coord)
            if primaldual is 'dual':
                vals = vals.inv()

            h = self.Y/N
            if order in [0, 'constant']:
                Wraw = get_weights_con(h, N, self.Y)
            elif order in [1, 'bilinear']:
                Wraw = get_weights_lin(h, N, self.Y)

            Aapp = np.zeros(np.hstack([dim, dim, N]))
            for m in np.arange(dim):
                for n in np.arange(dim):
                    Aapp[m, n] = np.real(np.prod(N) * \
                        DFT.ifftnc(Wraw*DFT.fftnc(vals[m, n], N), N))
            name = 'Aexact%d' % order
            return Matrix(name=name, val=Aapp)

    def get_shape_functions(self, N):
        inclusions = self.conf['inclusions']
        params = self.conf['params']
        positions = self.conf['positions']
        chars = []
        for ii, incl in enumerate(inclusions):
            if incl in inclusion_keys['cube']:
                h = params[ii]
                Wraw = get_weights_con(h, N, self.Y)
                chars.append(np.real(DFT.ifftnc(Wraw, N))*np.prod(N))
            elif incl in inclusion_keys['ball']:
                r = params[ii]/2
                if r == 0:
                    Wraw = np.zeros(N)
                else:
                    Wraw = get_weights_circ(r, N, self.Y)
                chars.append(np.real(DFT.ifftnc(Wraw, N))*np.prod(N))
            elif incl == 'all':
                chars.append(np.ones(N))
            elif incl == 'otherwise':
                chars.append(np.ones(N))
                for ii in np.arange(len(inclusions)-1):
                    chars[-1] -= chars[ii]
            else:
                msg = 'The inclusion (%s) is not supported!' % (incl)
                raise NotImplementedError(msg)
        return chars

    def evaluate(self, coord):
        """
        Evaluate material at coordinates (coord).

        Parameters
        ----------
        material :
            material definition
        coord : numpy.array
            coordinates where material coefficients are evaluated

        Returns
        -------
        A : numpy.array
            material coefficients at coordinates (coord)
        """
        if hasattr(self.conf, '__call__'):
            A_val = self.conf(coord)

        else:
            A_val = np.zeros(self.conf['vals'][0].shape + coord.shape[1:])
            topos = self.get_topologies(coord, self.conf['inclusions'],
                                        self.conf['params'],
                                        self.conf['positions'])

            for ii in np.arange(len(self.conf['inclusions'])):
                A_val += np.einsum('ij...,k...->ijk...', self.conf['vals'][ii],
                                   topos[ii])

        return Matrix(name='material', val=A_val)

    @staticmethod
    def get_topologies(coord, inclusions, params, positions):
        dim = coord.shape[0]
        topos = []

        for ii, kind in enumerate(inclusions):

            if kind in inclusion_keys['cube']:
                param = np.array(params[ii], dtype=np.float64)
                position = np.array(positions[ii], dtype=np.float64)
                topos.append(np.ones(coord.shape[1:]))
                for jj in np.arange(dim):
                    topos[ii] *= ((coord[jj]-position[jj]) >= -param[jj]/2)
                    topos[ii] *= ((coord[jj]-position[jj]) <= param[jj]/2)
            elif kind in inclusion_keys['ball']:
                position = np.array(positions[ii], dtype=np.float64)
                norm2 = 0. # square of norm
                for jj in np.arange(dim):
                    norm2 += (coord[jj]-position[jj])**2
                topos.append(norm2**0.5 < params[ii]/2)
#             elif kind == 'all':
#                 topos.append(np.ones(coord.shape[1:]))
            elif kind == 'otherwise':
                topos.append(np.ones(coord.shape[1:]))
                for jj in np.arange(len(topos)-1):
                    topos[ii] -= topos[jj]
                if not (topos[ii]>=0).all():
                    raise NotImplementedError("Overlapping inclusions!")
            else:
                msg = "Inclusion (%s) is not implemented." % (kind)
                raise NotImplementedError(msg)
        return topos

def get_weights_con(h, Nbar, Y):
    """
    it evaluates integral weights,
    which are used for upper-lower bounds calculation,
    with constant rectangular inclusion

    Parameters
    ----------
        h - the parameter determining the size of inclusion
        Nbar - no. of points of regular grid where the weights are evaluated
        Y - the size of periodic unit cell
    Returns
    -------
        Wphi - integral weights at regular grid sizing Nbar
    """
    d = np.size(Y)
    meas_puc = np.prod(Y)
    ZN2l = VecTri.get_ZNl(Nbar)
    Wphi = np.ones(Nbar) / meas_puc
    for ii in np.arange(d):
        Nshape = np.ones(d)
        Nshape[ii] = Nbar[ii]
        Nrep = np.copy(Nbar)
        Nrep[ii] = 1
        Wphi *= h[ii]*np.tile(np.reshape(np.sinc(h[ii]*ZN2l[ii]/Y[ii]),
                                         Nshape), Nrep)
    return Wphi

def get_weights_lin(h, Nbar, Y):
    """
    it evaluates integral weights,
    which are used for upper-lower bounds calculation,
    with bilinear inclusion at rectangular area

    Parameters
    ----------
    h - the parameter determining the size of inclusion
    Nbar - no. of points of regular grid where the weights are evaluated
    Y - the size of periodic unit cell

    Returns
    -------
    Wphi - integral weights at regular grid sizing Nbar
    """
    d = np.size(Y)
    meas_puc = np.prod(Y)
    ZN2l = VecTri.get_ZNl(Nbar)
    Wphi = np.ones(Nbar) / meas_puc
    for ii in np.arange(d):
        Nshape = np.ones(d)
        Nshape[ii] = Nbar[ii]
        Nrep = np.copy(Nbar)
        Nrep[ii] = 1
        Wphi *= h[ii]*np.tile(np.reshape((np.sinc(h[ii]*ZN2l[ii]/Y[ii]))**2,
                                         Nshape), Nrep)
    return Wphi

def get_weights_circ(r, Nbar, Y):
    """
    it evaluates integral weights,
    which are used for upper-lower bounds calculation,
    with constant circular inclusion

    Parameters
    ----------
        r - the parameter determining the size of inclusion
        Nbar - no. of points of regular grid where the weights are evaluated
        Y - the size of periodic unit cell
    Returns
    -------
        Wphi - integral weights at regular grid sizing Nbar
    """
    d = np.size(Y)
    ZN2l = TrigPolynomial.get_ZNl(Nbar)
    meas_puc = np.prod(Y)
    circ = 0
    for m in np.arange(d):
        Nshape = np.ones(d)
        Nshape[m] = Nbar[m]
        Nrep = np.copy(Nbar)
        Nrep[m] = 1
        xi_p2 = np.tile(np.reshape((ZN2l[m]/Y[m])**2, Nshape), Nrep)
        circ += xi_p2
    circ = circ**0.5
    ind = tuple(np.round(Nbar/2))
    circ[ind] = 1.

    Wphi = r**2 * sp.jn(1, 2*np.pi*circ*r) / (circ*r)
    Wphi[ind] = np.pi*r**2
    Wphi = Wphi / meas_puc
    return Wphi

if __name__ == '__main__':
    execfile('../main_test.py')
