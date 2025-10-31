import numpy as np
import sys
import os
from pathlib import Path
import copy

# ------------------------------------------------------- #

from meshprep.io import read_mesh, write_mesh
from meshprep.process import remesh
from mesh_harmonics_solver import MeshHarmonicsSolver
from surfmop import SurfaceMesh
from flatiron_tk.mesh import Mesh
import meshio
import matplotlib.pyplot as plt
from dolfinx import fem
from flatiron_tk.physics import PhysicsProblem
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver
import dolfinx

def set_weak_form(self):
    b = self.external_function('b')
    u = self.get_solution_function()
    w = self.get_test_function()
    self.weak_form = ufl.inner(u-b, w)*self.dx

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

smesh_tgt = SurfaceMesh(stl_file='la_003.stl')
smesh_tgt.build_normal(True)

msh_file = create_msh('steps/tmp_mesh_39.stl')
mesh = Mesh(mesh_file=msh_file, gdim=3)

# V, F = get_dfx_surface_mesh(mesh.msh)
# smesh = SurfaceMesh(vertices=V, facets=F)
# sdf, nsdf = smesh_tgt.signed_distance(V)
# u = -nsdf*sdf[:,np.newaxis]
# smesh.set_point_data('u', u)
# smesh.write('sdf.vtu')
