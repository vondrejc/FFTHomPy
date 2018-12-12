import numpy as np
from ffthompy import Timer
from ffthompy.sparse.objects import SparseTensor

def richardson(Afun, B, x0=None, rank=None, tol=None, par=None, norm=None):
    if isinstance(par['alpha'], float):
        omega=1./par['alpha']
    else:
        raise ValueError()
    res={'norm_res': [],
           'kit': 0}
    if x0 is None:
        x=B*omega
    else:
        x=x0

    if norm is None:
        norm=lambda X: X.norm()

    res['norm_res'].append(norm(B))

    M=SparseTensor(kind=x.kind, val=np.ones(x.N.size*[3,]), rank=1) # constant field
    FM=M.fourier().enlarge(x.N)

    norm_res=1e15
    while (norm_res>par['tol'] and res['kit']<par['maxiter']):
        res['kit']+=1
        residuum=B-Afun(x)
        norm_res = norm(residuum)
        if par['divcrit'] and norm_res>res['norm_res'][res['kit']-1]:
            break

        x=(x+residuum*omega)
        x=(-FM*x.mean()+x).truncate(rank=rank, tol=tol) # setting correct mean

        res['norm_res'].append(norm_res)
    return x, res

def richardson_debug(Afun, B, x0=None, rank=None, tol=None, par=None, norm=None):
    if isinstance(par['alpha'], float):
        omega=1./par['alpha']
    else:
        raise ValueError()
    res={'norm_res': [],
           'kit': 0}
    if x0 is None:
        x=B*omega
    else:
        x=x0
    x=x.truncate(rank=rank, tol=tol)


    if norm is None:
        norm=lambda X: X.norm()

    norm_res=1e15
    while (norm_res>par['tol'] and res['kit']<par['maxiter']):
        res['kit']+=1
        tic=Timer(name='Afun(x)')
        Afunx=Afun(x)
        tic.measure()
        tic=Timer(name='residuum')
        residuum=B-Afunx
        tic.measure()
        tic=Timer(name='iteration')
        x=(x+residuum*omega).truncate(rank=rank, tol=tol)
        tic.measure()
        tic=Timer(name='norm_residuum')
        norm_res=norm(residuum)
        tic.measure()
        res['norm_res'].append(norm_res)

    res['norm_res'].append(norm(B-Afun(x)))
    return x, res
