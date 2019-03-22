import numpy as np
from ffthompy.general.base import Timer
import itertools


def postprocess(pb, A, mat, solutions, results, primaldual):
    """
    The function post-process the results.
    """
    tim = Timer(name='postprocessing')
    print('\npostprocessing')
    matrices = {}
    for pp in pb.postprocess:
        if pp['kind'] in ['GaNi', 'gani']:
            order_name = ''
            Nname = ''
            if A.name is not 'A_GaNi':
                A = mat.get_A_GaNi(pb.solve['N'], primaldual)

        elif pp['kind'] in ['Ga', 'ga']:
            if 'order' in pp:
                Nbarpp = tuple(2*np.array(pb.solve['N']) - 1)
                if pp['order'] is None:
                    Nname = ''
                    order_name = ''
                    A = mat.get_A_Ga(Nbar=Nbarpp, primaldual=primaldual, order=pp['order'])
                else:
                    order_name = '_o' + str(pp['order'])
                    Nname = '_P%d' % np.mean(pp['P'])
                    A = mat.get_A_Ga(Nbar=Nbarpp, primaldual=primaldual,
                                     order=pp['order'], P=pp['P'])
            else:
                order_name = ''
                Nname = ''
        else:
            ValueError()

        name = 'AH_%s%s%s_%s' % (pp['kind'], order_name, Nname, primaldual)
        print(('calculating: ' + name))

        AH = assembly_matrix(A, solutions)

        if primaldual is 'primal':
            matrices[name] = AH
        else:
            matrices[name] = np.linalg.inv(AH)
    tim.measure()

    pb.output.update({'sol_' + primaldual: solutions,
                      'res_' + primaldual: results,
                      'mat_' + primaldual: matrices})

def assembly_matrix(Afun, solutions):
    """
    The function assembles the homogenized matrix from minimizers (corrector
    functions).
    """
    dim = len(solutions)
    if not np.allclose(Afun.N, solutions[0].N):
        Nbar = Afun.N
        sol = []
        for ii in np.arange(dim):
            sol.append(solutions[ii].project(Nbar))
    else:
        sol = solutions

    AH = np.zeros([dim, dim])
    for ii, jj in itertools.product(list(range(dim)), repeat=2):
        AH[ii, jj] = Afun(sol[ii]) * sol[jj]
    return AH


def add_macro2minimizer(X, E):
    """
    The function takes the minimizers (corrector function with zero-mean
    property or equaling to macroscopic value) and returns a corrector function
    with mean that equals to macroscopic value E.
    """
    if np.allclose(X.mean(), E):
        return X
    elif np.allclose(X.mean(), np.zeros_like(E)):
        EN = X.zeros_like(name='EN')
        EN.set_mean(E)
        return X + EN
    else:
        raise ValueError("Field is neither zero-mean nor E-mean.")
