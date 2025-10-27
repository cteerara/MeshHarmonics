import numpy as np
import sys
import os
from pathlib import Path
import copy

# ------------------------------------------------------- #

from meshprep.io import read_mesh, write_mesh
from surfmop import SurfaceMesh
from flatiron_tk.mesh import CuboidMesh
from dolfinx.fem import functionspace

# V, F = read_mesh('mesh/sphere_0.0500.stl')
stl_file = 'mesh/sphere_0.0500.stl'

h = 1e-1
box_mesh = CuboidMesh(0,0,0,1,1,1,h)
func_space = functionspace(box_mesh.msh, ("CG", 1))
xdof = func_space.tabulate_dof_coordinates()

smesh = SurfaceMesh(stl_file=stl_file)
smesh.build_normal()

