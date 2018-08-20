import numpy as np
from ffthompy.materials import Material
from ffthompy.sparse import decompositions
# from ffthompy.sparse.canoTensor import CanoTensor
from ffthompy.sparse.tucker import Tucker
from ffthompy.trigpol import Grid


class SparseMaterial(Material):
    def __init__(self, material_conf):
        Material.__init__(self, material_conf)
        self.mat=Material(material_conf)

    def get_A_GaNi(self, N, primaldual='primal', k=None, tol=None):
        A_GaNi=self.mat.get_A_GaNi(N, primaldual='primal')
        # # using canoTensor format
#        Abas, k, max_err=decompositions.dCA_matrix_input(A_GaNi.val[0, 0], k=k, tol=tol)
#        normAbas=np.linalg.norm(Abas, axis=0)
#        Abas=Abas/np.atleast_2d(normAbas)
#        Abas=Abas.T
#        return CanoTensor(name='Aganis', core=normAbas**2, basis=[Abas, Abas])
        # # using tucker format
        S, U=decompositions.HOSVD(A_GaNi.val[0, 0], k=k, tol=tol)
        for i in range(0, len(U)):
            U[i]=U[i].T

        return Tucker(name='A_GaNi', core=S, basis=U, orthogonal=True)

    def get_A_Ga(self, Nbar, primaldual='primal', order=-1, P=None, tol=None, k=None):
        if P is None and 'P' in self.conf:
            P=self.conf['P']
        As=self.get_A_GaNi(N=P, primaldual=primaldual, k=k, tol=tol)
        FAs=As.fourier()

        h=self.Y/P
        if order in [0, 'constant']:
            Wraw=get_weights_con(h, Nbar, self.Y)
        elif order in [1, 'bilinear']:
            Wraw=get_weights_lin(h, Nbar, self.Y)
        else:
            raise ValueError()

        FAs.core*=np.prod(P)
        if np.allclose(P, Nbar):
            hAM=FAs
        elif np.all(np.greater_equal(P, Nbar)):
            hAM=FAs.decrease(Nbar)
        elif np.all(np.less(P, Nbar)):
            factor=np.ceil(np.array(Nbar, dtype=np.float64)/P)
            hAM0per=tile(FAs, 2*np.array(factor, dtype=np.int)-1)
            hAM=hAM0per.decrease(Nbar)

        WFAs=(Wraw*hAM).fourier()
        WFAs.name='Agas'
        return WFAs

def tile(FAs, N):
    assert(FAs.Fourier is True)
    basis=[]
    for ii, n in enumerate(N):
        basis.append(np.tile(FAs.basis[ii], (1, n)))
    # return CanoTensor(name=FAs.name, core=FAs.core, basis=basis, Fourier=FAs.Fourier)
    return Tucker(name=FAs.name, core=FAs.core, basis=basis, Fourier=FAs.Fourier)

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
    dim=np.size(Y)
    meas_puc=np.prod(Y)
    ZN2l=Grid.get_ZNl(Nbar)
    Wphi=[]
    for ii in np.arange(dim):
        Nshape=np.ones(dim, dtype=np.int)
        Nshape[ii]=Nbar[ii]
        Nrep=np.copy(Nbar)
        Nrep[ii]=1
        Wphi.append(np.atleast_2d(h[ii]/meas_puc*np.sinc(h[ii]*ZN2l[ii]/Y[ii])))

    # W = CanoTensor(name='Wraw', core=np.array([1.]), basis=Wphi, Fourier=True)
    W=Tucker(name='Wraw', core=np.array([1.]), basis=Wphi, Fourier=True)
    return W


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
    d=np.size(Y)
    meas_puc=np.prod(Y)
    ZN2l=Grid.get_ZNl(Nbar)
    Wphi=[]
    for ii in np.arange(d):
        Nshape=np.ones(d, dtype=np.int)
        Nshape[ii]=Nbar[ii]
        Nrep=np.copy(Nbar)
        Nrep[ii]=1
        Wphi.append(np.atleast_2d(h[ii]/meas_puc*np.sinc(h[ii]*ZN2l[ii]/Y[ii])**2))
    # W = CanoTensor(name='Wraw', core=np.array([1.]), basis=Wphi, Fourier=True)
    W=Tucker(name='Wraw', core=np.array([1.]), basis=Wphi, Fourier=True)
    return W
