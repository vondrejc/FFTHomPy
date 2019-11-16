import numpy as np
from functions import get_matinc, PeriodicBoundary
from fenics import (Constant, FunctionSpace, UnitSquareMesh, UnitCubeMesh, inner,
                    MeshFunction, SubDomain, between, Measure, DirichletBC,
                    TrialFunction, TestFunction, Function, grad, assemble, solve)

# PARAMETERS
dim   = 2   # dimension (works for 2D and 3D)
N     = 5*np.ones(dim, dtype=np.int) # number of voxels (assumed equal for all directions)
phase = 10. # material contrast
order = 1   # polynomial order in FE space

# auxiliary values
prodN = np.prod(np.array(N)) # number of grid points
vec_shape=(dim,)+tuple(N) # shape of the vector for storing DOFs

# PROBLEM DEFINITION
_mat, _inc = get_matinc(dim, phase) # material coef. for matrix (mat) and inclusion (inc)
mat=Constant(_mat)
inc=Constant(_mat+_inc)

class Inclusion_2d(SubDomain): # square inclusion
    def inside(self, x, on_boundary):
        return (between(x[1], (0.2, 0.8)) and between(x[0], (0.2, 0.8)))

class Inclusion_3d(SubDomain): # cube inclusion
    def inside(self, x, on_boundary):
        return (between(x[2], (0.2, 0.8)) and between(x[1], (0.2, 0.8)) and between(x[0], (0.2, 0.8)))

E=np.zeros(dim); E[0]=1. # macroscopic value
E=Constant(E)

if dim==2:
    mesh=UnitSquareMesh(*N) # generation of mesh
    inclusion=Inclusion_2d()
    point0="near(x[0], 0) && near(x[1], 0)"
elif dim==3:
    mesh=UnitCubeMesh(*N) # generation of mesh
    inclusion=Inclusion_3d()
    point0="near(x[0], 0) && near(x[1], 0) && near(x[2], 0)"

V=FunctionSpace(mesh, "CG", order, constrained_domain=PeriodicBoundary(dim))

# setting the elements that lies in inclusion and in matrix phase
domains=MeshFunction("size_t", mesh, dim)
domains.set_all(0)
inclusion.mark(domains, 1)
dx=Measure('dx', subdomain_data=domains)

def bilf(up, vp): # bilinear form
    return inner(mat*up, vp)*dx(0)+inner(inc*up, vp)*dx(1)

bc0=DirichletBC(V, Constant(0.), point0, method='pointwise')

# SOLVER
u=TrialFunction(V)
v=TestFunction(V)
uE = Function(V)

solve(bilf(grad(u), grad(v)) == -bilf(E, grad(v)), uE, bcs=[bc0])

# POSTPROCESSING evaluation of guaranteed bound
AH11 = assemble(bilf(grad(uE)+E, grad(uE)+E))
print('homogenised component A11 = {} (FEM)'.format(AH11))

print('END')
