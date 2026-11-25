from __future__ import division, print_function

import logging

import numpy as np

from examples.lowRankTensorApproximations.setting import (
    get_default_parameters,
    get_material_coef,
)
from ffthompy.tensorsLowRank.homogenisation import (
    homog_Ga_full_potential,
    homog_Ga_sparse,
    homog_GaNi_full_potential,
    homog_GaNi_sparse,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")



# PARAMETERS ##############################################################
dim=2
N=5*3**2
material=0
kind=0 # from kind_list=['cano','tucker','tt']

pars, pars_sparse=get_default_parameters(dim, N, material, kind)
pars_sparse.debug=True
pars_sparse.solver.update(dict(rank=10))

logger.info('== format=%s, N=%s, dim=%s, material=%s, rank=%s ====', pars_sparse.kind, N, dim, material, pars_sparse.solver['rank'])
logger.info('dofs = %s', N**dim)

# get material coefficients
Aga, Agani, Agas, Aganis=get_material_coef(material, pars, pars_sparse)

logger.info('\n== Full solution with potential by CG (GaNi)===========')
resP_GaNi=homog_GaNi_full_potential(Agani, Aga, pars)
logger.info('mean of solution=%s', resP_GaNi.Fu.mean())
logger.info('homogenised properties (component 11) = %s', resP_GaNi.AH)

logger.info('\n== Full solution with potential by CG (Ga) ===========')
resP_Ga=homog_Ga_full_potential(Aga, pars)
logger.info('mean of solution=%s', resP_Ga.Fu.mean())
logger.info('homogenised properties (component 11) = %s', resP_Ga.AH)

logger.info('\n== SPARSE solver with preconditioner (Ga) =======================')
resS_Ga=homog_Ga_sparse(Agas, pars_sparse)
logger.info('mean of solution=%s', resS_Ga.Fu.mean())
logger.info('homogenised properties (component 11) = %s', resS_Ga.AH)
logger.info('iterations=%s', resS_Ga.solver['kit'])
logger.info('norm(resP)=%s', resS_Ga.solver['norm_res'])

logger.info('\n== SPARSE solver with preconditioner (GaNi) =======================')
resS_GaNi=homog_GaNi_sparse(Aganis, Agas, pars_sparse)
logger.info('mean of solution=%s', resS_GaNi.Fu.mean())
logger.info('homogenised properties (component 11) = %s', resS_GaNi.AH)
logger.info('iterations=%s', resS_GaNi.solver['kit'])
if np.array_equal(pars.N, pars_sparse.N):
    logger.info('norm(dif)=%s', np.linalg.norm(resP_GaNi.Fu.fourier(Fourier=False).val-resS_GaNi.Fu.fourier().full().val))
logger.info('norm(resP)=%s', resS_GaNi.solver['norm_res'])

logger.info('END')
