import numpy as np


class TrigPolynomial():
    @staticmethod
    def get_ZNl(N):
        r"""
        it produces index set ZNl=\underline{\set{Z}}^d_N :
        ZNl[i][j]\in\set{Z} : -N[i]/2 <= ZNl[i] < N[i]/2
        """
        ZNl = []
        N = np.array(N)
        for m in np.arange(np.size(N)):
            ZNl.append(np.arange(np.fix(-N[m]/2.), np.fix(N[m]/2.+0.5)))
        return ZNl

    @staticmethod
    def get_xil(N, Y):
        """
        it produces discrete frequencies of Fourier series
        xil[i] = ZNl[i]/Y[i]
        """
        xil = []
        for m in np.arange(np.size(N)):
            xil.append(np.arange(np.fix(-N[m]/2.), np.fix(N[m]/2.+0.5))/Y[m])
        return xil

    @staticmethod
    def get_grid_coordinates(N, Y):
        """
        It produces coordinates of the set of nodal points
        Coord[i][j] = x_N^{(i,j)}
        """
        d = np.size(N)
        ZNl = TrigPolynomial.get_ZNl(N)
        coord = np.zeros(np.hstack([d, N]))
        for ii in np.arange(d):
            x = ZNl[ii]/N[ii]*Y[ii]
            Nshape = np.ones(d)
            Nshape[ii] = N[ii]
            Nrep = np.copy(N)
            Nrep[ii] = 1
            coord[ii] = np.tile(np.reshape(x, Nshape), Nrep)
        return coord


class TrigPolBasis(TrigPolynomial):
    """
    This represents a basis functions of trigonometric polynomials.
    """
    def eval_phi_k_N(self, x):
        val = np.zeros_like(x)
        coef = 1./np.prod(self.N)
        for ii in self.get_ZNl(self.N)[0]:
            val += coef*np.exp(1j*np.pi*ii*(x/self.Y - 2*self.order/self.N))
        return val

    def eval_phi_k(self, x):
        val = np.exp(np.pi*1j*x*self.order)
        return val

    def get_nodes(self):
        ZNl = self.get_ZNl(self.N)[0]
        x_nodes = ZNl*2*self.Y/self.N
        vals = np.zeros_like(x_nodes)
        ind = self.order + np.fix(self.N/2)
        vals[ind] = 1
        return x_nodes, vals

    def __init__(self, order, N=None, Y=None):
        self.order = order
        self.dim = np.size(order)
        self.N = np.array(N)
        if Y is None:
            self.Y = np.ones(self.dim)
        if N is None:
            self.Fourier = True
            self.eval = self.eval_phi_k
        else:
            self.Fourier = False
            self.eval = self.eval_phi_k_N

    def __repr__(self):
        if self.Fourier:
            ss = "Fourier basis function for k = %d" % (self.order,)
        else:
            ss = "Shape basis function for k = %d and N = %s" \
                % (self.order, str(self.N))
        return ss

if __name__ == '__main__':
    execfile('../main_test.py')
