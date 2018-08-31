import numpy as np
from ffthompy.materials import Material
from ffthompy.sparse import decompositions
# from ffthompy.sparse.canoTensor import CanoTensor
from ffthompy.sparse.tucker import Tucker
from ffthompy.sparse.tensorTrain import TensorTrain
from ffthompy.trigpol import Grid


class SparseMaterial(Material):
    def __init__(self, material_conf):
        Material.__init__(self, material_conf)
        self.mat=Material(material_conf)

    def get_A_GaNi(self, N, primaldual='primal', k=None, tol=None):
        A_GaNi=self.mat.get_A_GaNi(N, primaldual='primal')

        return TensorTrain(A_GaNi.val[0, 0], rmax=k, name='A_GaNi')


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

        FAs*=np.prod(P)
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

    cl=FAs.to_list(FAs)
    cl_new=[None]*FAs.d
    for i in range(FAs.d):
        cl_new[i]=np.tile(cl[i], (1, N[i], 1))

    return FAs.from_list(cl_new, name=FAs.name+'_tiled', Fourier=FAs.Fourier)

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
        Wphi.append(np.reshape(h[ii]/meas_puc*np.sinc(h[ii]*ZN2l[ii]/Y[ii]), (1,-1, 1))) # since it is rank 1

    return TensorTrain.from_list(Wphi, Fourier=True)


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
        Wphi.append(np.reshape(h[ii]/meas_puc*np.sinc(h[ii]*ZN2l[ii]/Y[ii])**2, (1,-1, 1)))
    # W = CanoTensor(name='Wraw', core=np.array([1.]), basis=Wphi, Fourier=True)
    # W=Tucker(name='Wraw', core=np.array([1.]), basis=Wphi, Fourier=True)
    return TensorTrain.from_list(Wphi, Fourier=True)
