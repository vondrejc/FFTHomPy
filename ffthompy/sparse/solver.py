

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
        norm=lambda X: float(X.T*X)**0.5
    norm_res=1e15
    while (norm_res>par['tol'] and res['kit']<par['maxiter']):
        res['kit']+=1
        residuum=B-Afun(x)
        x=(x+residuum*omega).truncate(rank=rank, tol=tol)
        norm_res=norm(residuum)
        norm_res2=residuum.norm()
        res['norm_res'].append(norm_res)
    res['norm_res'].append(norm(B-Afun(x)))
    return x, res
