import numpy as np


class ElasticTensor():
    """
    This class represents linear fourth-order tensor of elastic parameters
    for both stiffness and compliance. It also evaluates the tensor as a matrix
    using engineering Mandel's or Voight's notation. The plane stress or strain
    is also available.
    """

    def __init__(self, bulk=None, mu=None, stiffness=True, plane=None):
        """
        Parameters
        ----------
        bulk : float
            bulk modulus
        mu : float
            shear modulus
        stiffness : boolean
            determines whether the values are evaluated for stiffness (True)
            or for its inverse compliance (False)
        plane : None, 'strain', or 'stress'
            determines the physical problem; None represents three-dimensional
            problem and 'strain' or 'stress' represent reduced problem of plane
            strain or stress respectively
        """
        if stiffness:
            self.stiffness = stiffness
            self.val_type = 'stiffness'
        else:
            self.val_type = 'compliance'

        if plane is None:
            self.dim = 3
        elif plane in ['stress', 'strain']:
            self.dim = 2
        else:
            raise ValueError("This type of plane (%s) is not supported."
                             % str(plane))

        self.plane = plane

        self.sym = self.get_sym(self.dim)
        self.bulk = bulk
        self.mu = mu

        # set up values to tensor and matrices
        _, volumetric, deviatoric = self.get_decomposition()
        if stiffness:
            self.val = 3.*bulk*volumetric + 2.*mu*deviatoric
        else:
            self.val = 1./(3*bulk)*volumetric + 1./(2*mu)*deviatoric

        self.mandel = self.create_mandel(self.val)
        self.voight = self.create_voight(self.val)

        if plane is not None:
            self.val = self.val[0:2, 0:2, 0:2, 0:2]

        if (stiffness and plane == 'strain') or \
                (not stiffness and plane == 'stress'):
            self.mandel = self.get_plane(self.mandel)
            self.voight = self.get_plane(self.voight)
        elif (not stiffness and plane == 'strain') or \
                (stiffness and plane == 'stress'):
            inv = np.linalg.inv
            self.mandel = inv(self.get_plane(inv(self.mandel)))
            self.voight = inv(self.get_plane(inv(self.voight)))
        else:
            pass

    @staticmethod
    def get_plane(val, ind=None):
        if ind is None:
            ind = [0, 1]
        ind_shear = list(range(3))
        ind_shear.remove(ind[0])
        ind_shear.remove(ind[1])
        ind.append(ind_shear[0]+3)
        mat = val[ind][:, ind]
        return mat

    def __repr__(self):
        ss = "Class: %s\n" % (self.__class__.__name__)
        ss += '    stiffness = %s (%s)\n' % (self.stiffness, self.val_type)
        ss += '    dim = %d' % (self.dim)
        if self.plane is None:
            ss += '\n'
        else:
            ss += ' (plane %s)\n' % self.plane
        ss += '    bulk = %s\n' % str(self.bulk)
        ss += '    mu = %s\n' % str(self.mu)
        return ss

    @staticmethod
    def get_sym(dim):
        return dim*(dim+1)/2

    @staticmethod
    def get_decomposition():
        """
        It produces symmetrized fourth-order identity, hydrostatic, and
        deviatoric projections.

        Returns
        -------
        idsym : numpy.array of shape = (3, 3, 3, 3)
            symmetrized identity operator
        volumetric : numpy.array of shape = (3, 3, 3, 3)
            hydrostatic projection
        deviatoric : numpy.array of shape = (3, 3, 3, 3)
            deviatoric projection
        """
        idm = np.eye(3)
        volumetric = 1./3*np.einsum('ij,kl', idm, idm)
        idsym = 0.5*(np.einsum('ik,jl', idm, idm)
                     + np.einsum('il,jk', idm, idm))
        deviatoric = idsym - volumetric
        return idsym, volumetric, deviatoric

    @staticmethod
    def create_mandel(mat):
        """
        It transfer symmetric four-order tensor (or matrix) to
        second order tensor (or vector) using Mandel's notation.

        Parameters
        ----------
        mat : numpy.array of shape = (d, d, d, d) or (d, d) for dimension d
            fourth-order tensor of elastic parameters

        Returns
        -------
        vec : numpy.array of shape = (sym, sym) or (sym,) for sym = d*(d+1)/2
            second-order tensor of elastic parameters with Mandel's notation
        """
        dim = mat.shape[0]
        sym = dim*(dim+1)/2
        if mat.ndim == 4:
            vec = np.zeros([sym, sym], dtype=mat.dtype)
            for ii in np.arange(dim):
                for jj in np.arange(dim):
                    kk = list(range(dim))
                    kk.remove(ii)
                    ll = list(range(dim))
                    ll.remove(jj)
                    vec[ii, jj] = mat[ii, ii, jj, jj]
                    vec[ii, jj+dim] = 2**.5*mat[ii, ii, ll[0], ll[1]]
                    vec[jj+dim, ii] = vec[ii, jj+dim]
                    vec[ii+dim, jj+dim] = 2*mat[kk[0], kk[1], ll[0], ll[1]]
        elif mat.ndim == 2:
            vec = np.zeros(sym, dtype=mat.dtype)
            vec[:dim] = np.diag(mat)
            if dim == 2:
                vec[dim] = 2**.5*mat[0, 1]
            elif dim == 3:
                for ii in np.arange(sym-dim):
                    ind = list(range(sym-dim))
                    ind.remove(ii)
                    vec[dim+ii] = 2**.5*mat[ind[0], ind[1]]
            else:
                raise ValueError("Incorrect dimension (%d)" % dim)
        return vec

    @staticmethod
    def dispose_mandel(vec):
        vec = vec.squeeze()
        sym = vec.shape[0]
        dimfun = lambda sym: int((-1.+(1+8*sym)**.5)/2)
        dim = dimfun(sym)

        if vec.ndim == 2: # matrix
            if np.allclose(vec.shape[0], vec.shape[1]):
                raise ValueError()

            mat = np.zeros(dim*np.ones(2*vec.ndim))

            for ii in np.arange(dim):
                for jj in np.arange(dim):
                    kk = list(range(dim))
                    kk.remove(ii)
                    ll = list(range(dim))
                    ll.remove(jj)
                    mat[ii, ii, jj, jj] = vec[ii, jj]
                    mat[ii, ii, ll[0], ll[1]] = vec[ii, jj+dim] / 2**.5
                    mat[ll[0], ll[1], ii, ii] = mat[ii, ii, ll[0], ll[1]]
                    mat[kk[0], kk[1], ll[0], ll[1]] = vec[ii+dim, jj+dim] / 2.
                    mat[kk[1], kk[0], ll[0], ll[1]] = vec[ii+dim, jj+dim] / 2.
                    mat[kk[0], kk[1], ll[1], ll[0]] = vec[ii+dim, jj+dim] / 2.
                    mat[kk[1], kk[0], ll[1], ll[0]] = vec[ii+dim, jj+dim] / 2.

        elif vec.ndim == 1: # vector
            mat = np.diag(vec[:dim])

            if dim == 2:
                mat[0, 1] = vec[-1]/2**0.5
                mat[1, 0] = vec[-1]/2**0.5
            elif dim == 3:
                for ii in np.arange(sym-dim):
                    ind = list(range(sym-dim))
                    ind.remove(ii)
                    mat[ind[0], ind[1]] = vec[dim+ii]/2.**.5
                    mat[ind[1], ind[0]] = vec[dim+ii]/2.**.5
            else:
                raise ValueError("Incorrect dimension (%d)" % dim)

        return mat

    @staticmethod
    def create_voight(mat, valtype='strain'):
        """
        It transfer symmetric four-order tensor to second order tensor
        using Voight's notation.

        Parameters
        ----------
        mat : numpy.array of shape = (3, 3, 3, 3)
            fourth-order tensor of elastic parameters
        valtype : one of 'strain' or 'stress'
            this distinguish a engineering notation for strain and stress

        Returns
        -------
        vec : numpy.array of shape = (6, 6)
            second-order tensor of elastic parameters with Voight's notation
        """
        dim = mat.shape[0]
        sym = dim*(dim+1)/2
        if mat.ndim == 4:
            vec = np.zeros([sym, sym], dtype=mat.dtype)
            for ii in np.arange(dim):
                for jj in np.arange(dim):
                    kk = list(range(dim))
                    kk.remove(ii)
                    ll = list(range(dim))
                    ll.remove(jj)
                    vec[ii, jj] = mat[ii, ii, jj, jj]
                    vec[ii, jj+dim] = mat[ii, ii, ll[0], ll[1]]
                    vec[jj+dim, ii] = vec[ii, jj+dim]
                    vec[ii+dim, jj+dim] = mat[kk[0], kk[1], ll[0], ll[1]]

        elif mat.ndim == 2:
            vec = np.zeros(sym, dtype=mat.dtype)
            vec[:dim] = np.diag(mat)
            if valtype == 'strain':
                coef = 2.
            elif valtype == 'stress':
                coef = 1.
            else:
                msg = "Parameter valtype (%s) should be one of 'strain' or\
                     'stress'." % (str(valtype),)
                raise ValueError(msg)

            if dim == 2:
                vec[dim] = coef*mat[0, 1]
            elif dim == 3:
                for ii in np.arange(sym-dim):
                    ind = list(range(sym-dim))
                    ind.remove(ii)
                    vec[dim+ii] = coef*mat[ind[0], ind[1]]
            else:
                raise ValueError("Incorrect dimension (%d)" % dim)
        return vec