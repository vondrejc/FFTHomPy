import numpy as np
import scipy.special as sp
from ffthompy.matvec import DFT, VecTri, Matrix
from ffthompy.matvec_fun import Grid, decrease


inclusion_keys = {'ball': ['ball', 'circle'],
                  'cube': ['cube', 'square'],
                  'pyramid': ['bilinear_pyramid', 'pyramid']
                  }


class Material():

    def __init__(self, material_conf):
        self.conf = material_conf
        self.Y = material_conf['Y']

        # control the correctness of material definition
        if 'Y' not in self.conf:
            raise ValueError("The definition of PUC size (Y) is missing!")

        if 'fun' in self.conf:
            pass
        elif 'inclusions' in self.conf:
            n_incl = len(self.conf['inclusions'])
            for key in self.conf:
                if key not in ['inclusions', 'positions', 'params', 'vals']:
                    continue

                if len(self.conf[key]) != n_incl:
                    msg = "Improper no. of values in material for (%s)!" % key
                    raise ValueError(msg)

            for ii, incl in enumerate(self.conf['inclusions']):
                if incl in ['all', 'otherwise']:
                    continue

                param = self.conf['params'][ii]
                position = self.conf['positions'][ii]
                try:
                    if any(np.greater(param, self.Y)):
                        raise ValueError("Improper parameters of inclusion!")

                    self.conf['positions'][ii] = position % self.Y
                except:
                    raise ValueError("Improper material definition!")

        else:
            raise NotImplementedError("Improper material definition!")

    def get_A_Ga(self, Nbar, primaldual='primal', order=-1, P=None):
        """
        Returns stiffness matrix for scheme with exact integration.
        """
        if order == -1:
            if 'order' in self.conf:
                order = self.conf['order']
            else:
                raise ValueError('The material order is undefined!')
        elif order not in [None, 'exact', 0, 1]:
            raise ValueError('Wrong material order (%s)!' % str(order))

        if order in [None, 'exact']:
            shape_funs = self.get_shape_functions(Nbar)
            val = np.zeros(self.conf['vals'][0].shape + shape_funs[0].shape)
            for ii in range(len(self.conf['inclusions'])):
                if primaldual is 'primal':
                    Aincl = self.conf['vals'][ii]
                elif primaldual is 'dual':
                    Aincl = np.linalg.inv(self.conf['vals'][ii])
                val += np.einsum('ij...,k...->ijk...', Aincl, shape_funs[ii])
            name = 'A_Ga'

        else:
            if P is None and 'P' in self.conf:
                P = self.conf['P']
            coord = Grid.get_coordinates(P, self.Y)
            vals = self.evaluate(coord)
            dim = vals.d
            if primaldual is 'dual':
                vals = vals.inv()

            h = self.Y/P
            if order in [0, 'constant']:
                Wraw = get_weights_con(h, Nbar, self.Y)
            elif order in [1, 'bilinear']:
                Wraw = get_weights_lin(h, Nbar, self.Y)

            val = np.zeros(np.hstack([dim, dim, Nbar]))
            for m in np.arange(dim):
                for n in np.arange(dim):
                    hAM0 = DFT.fftnc(vals[m, n], P)
                    if np.allclose(P, Nbar):
                        hAM = hAM0
                    elif np.all(np.greater_equal(P, Nbar)):
                        hAM = decrease(hAM0, Nbar)
                    elif np.all(np.less(P, Nbar)):
                        factor = np.ceil(np.array(Nbar, dtype=np.float64) / P)
                        hAM0per = np.tile(hAM0,
                                          2*np.array(factor, dtype=np.int)-1)
                        hAM = decrease(hAM0per, Nbar)
                    else:
                        raise ValueError()

                    pNbar = np.prod(Nbar)
                    """ if DFT is normalized in accordance with articles there
                    should be np.prod(M) instead of np.prod(Nbar)"""
                    val[m, n] = np.real(pNbar*DFT.ifftnc(Wraw*hAM, Nbar))

            name = 'A_Ga_o%d_P%d' % (order, P.max())

        return Matrix(name=name, val=val, Fourier=False)

    def get_A_GaNi(self, N, primaldual='primal'):
        """
        Returns stiffness matrix for a scheme with trapezoidal quadrature rule.
        """
        coord = Grid.get_coordinates(N, self.Y)
        A = self.evaluate(coord)
        if primaldual is 'dual':
            A = A.inv()
        return A

    def get_shape_functions(self, N2):
        """
        Returns stiffness matrix for exact integration scheme for individual
        inclusions (square, circle, etc.).
        """
        N2 = np.array(N2, dtype=np.int)
        inclusions = self.conf['inclusions']
        params = self.conf['params']
        positions = self.conf['positions']

        chars = []
        for ii, incl in enumerate(inclusions):
            position = positions[ii]
            if isinstance(position, np.ndarray):
                SS = get_shift_inclusion(N2, position, self.Y)

            if incl in inclusion_keys['cube']:
                Wraw = get_weights_con(params[ii], N2, self.Y)
                chars.append(np.real(DFT.ifftnc(SS*Wraw, N2))*np.prod(N2))

            elif incl in inclusion_keys['ball']:
                r = params[ii]/2
                if r == 0:
                    Wraw = np.zeros(N2)
                else:
                    Wraw = get_weights_circ(r, N2, self.Y)
                chars.append(np.real(DFT.ifftnc(SS*Wraw, N2))*np.prod(N2))

            elif incl in inclusion_keys['pyramid']:
                Wraw = get_weights_lin(params[ii]/2., N2, self.Y)
                chars.append(np.real(DFT.ifftnc(SS*Wraw, N2))*np.prod(N2))

            elif incl == 'all':
                chars.append(np.ones(N2))

            elif incl == 'otherwise':
                chars.append(np.ones(N2))
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
        if 'fun' in self.conf:
            fun = self.conf['fun']
            A_val = fun(coord)
        else:
            A_val = np.zeros(self.conf['vals'][0].shape + coord.shape[1:])
            topos = self.get_topologies(coord)

            for ii in np.arange(len(self.conf['inclusions'])):
                A_val += np.einsum('ij...,k...->ijk...', self.conf['vals'][ii],
                                   topos[ii])

        return Matrix(name='A_GaNi', val=A_val, Fourier=False)

    def get_topologies(self, coord):
        """
        Returns topologies at coordinates (coord). By a topology, it is
        understood a characteristic function (ones and zeros at coord)
        of individual inclusions (square, circle, etc.) usually defined in
        input file.
        """
        inclusions = self.conf['inclusions']
        params = self.conf['params']
        positions = self.conf['positions']

        dim = coord.shape[0]
        topos = []

        # periodically enlarge coordinates
        N = np.array(coord.shape[1:], dtype=np.int)
        cop = np.empty(np.hstack([dim, 3*N]))
        mapY = np.array([-1, 0, 1])
        for dd in np.arange(dim):
            Nshape = np.ones(dim, dtype=np.int32)
            Nshape[dd] = 3*N[dd]
            Nrep = 3*N
            Nrep[dd] = 1
            vec = np.repeat(mapY*self.Y[dd], N[dd])
            Ymat = np.tile(np.reshape(vec, Nshape), Nrep)
            cop[dd] = np.tile(coord[dd], 3*np.ones(dim, dtype=np.int)) + Ymat

        mapY = np.array([-1, 0, 1])
        Yiter = mapY[np.newaxis]
        for ii in np.arange(1, dim):
            Yiter = np.vstack([np.repeat(Yiter, 3, axis=1),
                               np.tile(mapY, Yiter.shape[1])])

        for ii, kind in enumerate(inclusions):

            if kind in inclusion_keys['cube']:
                param = np.array(params[ii], dtype=np.float64)
                pos = np.array(positions[ii], dtype=np.float64)
                topos.append(np.zeros(cop.shape[1:]))
                for Ycoef in Yiter.T:
                    Ym = self.Y*Ycoef
                    topo_loc = np.ones(cop.shape[1:])
                    for dd in np.arange(dim):
                        topo_loc *= ((cop[dd]-pos[dd]+Ym[dd]) > -param[dd]/2)
                        topo_loc *= ((cop[dd]-pos[dd]+Ym[dd]) <= param[dd]/2)
                    topos[ii] += topo_loc
                topos[ii] = decrease(topos[ii], N)

            elif kind in inclusion_keys['ball']:
                pos = np.array(positions[ii], dtype=np.float64)
                topos.append(np.zeros(cop.shape[1:]))
                for Ycoef in Yiter.T:
                    Ym = self.Y*Ycoef
                    topo_loc = np.ones(cop.shape[1:])

                    norm2 = 0. # square of norm
                    for dd in np.arange(dim):
                        norm2 += (cop[dd]-pos[dd]-Ym[dd])**2
                    topos[ii] += (norm2**0.5 < params[ii]/2)
                topos[ii] = decrease(topos[ii], N)

            elif kind in inclusion_keys['pyramid']:
                param = np.array(params[ii], dtype=np.float64)
                pos = np.array(positions[ii], dtype=np.float64)
                topos.append(np.zeros(cop.shape[1:]))
                for Ycoef in Yiter.T:
                    Ym = self.Y*Ycoef
                    topo_loc = np.ones(cop.shape[1:])
                    shp = cop[0].shape
                    tri = lambda t, h: np.maximum(1-np.abs(t)/h, np.zeros(shp))
                    for dd in np.arange(dim):
                        topo_loc *= tri(cop[dd]-pos[dd]+2*Ym[dd], param[dd]/2.)
                    topos[ii] += topo_loc
                topos[ii] = decrease(topos[ii], N)

            elif kind == 'otherwise':
                topos.append(np.ones(coord.shape[1:]))
                for jj in np.arange(len(topos)-1):
                    topos[ii] -= topos[jj]
                if not (topos[ii] >= 0).all():
                    raise NotImplementedError("Overlapping inclusions!")

            elif kind == 'all':
                topos.append(np.ones(coord.shape[1:]))

            else:
                msg = "Inclusion (%s) is not implemented." % (kind)
                raise NotImplementedError(msg)

        return topos

    def savefig(self, fname='material.pdf', N=50*np.ones(2)):
        import pylab as pl
        pl.figure(num=None, figsize=(3,3), dpi=1000)
        coord = Grid.get_coordinates(N, self.Y)
        vals = self.evaluate(coord)[0, 0]
        pl.pcolor(coord[0], coord[1], -vals)
        pl.xlim([-self.Y[0]/2, self.Y[0]/2])
        pl.ylim([-self.Y[1]/2, self.Y[1]/2])
        pl.xlabel(r'coordinate $x_1$')
        pl.ylabel(r'coordinate $x_2$')
        pl.savefig(fname, pad_inches=0.02, bbox_inches='tight')
        pl.close()


def get_shift_inclusion(N, h, Y):
    N = np.array(N, dtype=np.int)
    Y = np.array(Y, dtype=np.float)
    dim = N.size
    ZN = Grid.get_ZNl(N)
    SS = np.ones(N, dtype=np.complex128)
    for ii in np.arange(dim):
        Nshape = np.ones(dim, dtype=np.int)
        Nshape[ii] = N[ii]
        Nrep = N
        Nrep[ii] = 1
        SS *= np.tile(np.reshape(np.exp(-2*np.pi*1j*(h[ii]*ZN[ii]/Y[ii])),
                                 Nshape), Nrep)
    return SS


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
    dim = np.size(Y)
    meas_puc = np.prod(Y)
    ZN2l = VecTri.get_ZNl(Nbar)
    Wphi = np.ones(Nbar) / meas_puc
    for ii in np.arange(dim):
        Nshape = np.ones(dim, dtype=np.int)
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
    h - the parameter determining the size of inclusion (half-size of support)
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
        Nshape = np.ones(d, dtype=np.int)
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
    ZN2l = Grid.get_ZNl(Nbar)
    meas_puc = np.prod(Y)
    circ = 0
    for m in range(d):
        Nshape = np.ones(d, dtype=np.int)
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
