import numpy as np
import sys
import os
from pathlib import Path
import copy

# ------------------------------------------------------- #

import matplotlib.pyplot as plt
from flatiron_tk.mesh import Mesh
from flatiron_tk.physics import SteadyScalarTransport
from meshprep.features import Holes
import dolfinx
import meshio
from dolfinx import fem
import ufl
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
from meshprep.io import read_mesh, write_mesh

from mesh_harmonics_solver import MeshHarmonicsSolver

def create_msh(stl_file):
    geo_str = 'Mesh.MshFileVersion = 2.0;\nGeneral.Verbosity = 1;\nMerge "%s";\nSurface Loop(1) = {1};\nPhysical Surface(1) = {1};'%(stl_file.split("/")[-1])
    with open(stl_file[:-3]+'geo', 'w') as fid:
        fid.write(geo_str)
    os.system('gmsh -v 1 -2 %s -o %s'%(stl_file[:-3]+'geo', stl_file[:-3]+'msh'))
    return stl_file[:-3]+'msh'

def get_dfx_surface_mesh(dfx_msh):
    facets = dfx_msh.topology.connectivity(2, 0).array.reshape((-1, 3))
    vertices = dfx_msh.geometry.x[:]
    return vertices, facets


# stl_file = 'mesh/sphere_clipped_rm.stl'
stl_file = 'mesh/sphere_0.0500.stl'
msh_file = create_msh(stl_file)
mesh = Mesh(mesh_file=msh_file, gdim=3)
num_modes = 36
mht = MeshHarmonicsSolver(mesh, num_modes=num_modes, lump_mass=False)
# mht.build_eigen_problem(homogeneous_bnds=[1,2,3,4,5])
mht.build_eigen_problem()
mht.solve()

U = mht.get_eigen_functions()

mesh_V, mesh_F = get_dfx_surface_mesh(mesh.msh)
point_data = {}
for i in range(U.shape[1]):
    point_data['U%03d'%i] = U[:,i]
smesh = meshio.Mesh(mesh_V, [("triangle", mesh_F)], point_data=point_data)
smesh.write('U.vtu')

# zero = flatiron_tk.constant(mesh, 0.)
# one = flatiron_tk.constant(mesh, 1.)
# bc_dict = {
#     1: {'type': 'dirichlet', 'value': zero},
#     2: {'type': 'dirichlet', 'value': zero},
#     3: {'type': 'dirichlet', 'value': one},
#     4: {'type': 'dirichlet', 'value': zero},
#     5: {'type': 'dirichlet', 'value': zero},
# }
# stp.set_bcs(bc_dict)
# problem = NonLinearProblem(stp)
# solver = NonLinearSolver(mesh.msh.comm, problem)
# solver.solve()
# phi = stp.solution.x.array
# pp = phi * (1-phi)
# psi = pp/np.max(pp)
# # stp.solution.x.array[:] = psi
# return psi, phi


