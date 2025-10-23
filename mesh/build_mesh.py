import numpy as np
import sys
import os
from pathlib import Path
import copy

# ------------------------------------------------------- #

from meshprep.io import read_mesh, write_mesh
from meshprep.process import remesh

V, F = read_mesh('sphere_ref.stl')


mesh_lens = [0.1, 0.05, 0.025, 0.0125]
for mesh_len in mesh_lens:
    print(mesh_len)
    V_, F_ = remesh(V, F, target_len=mesh_len, remesh_nitr=100)
    write_mesh(V_, F_, 'sphere_%.04f.stl'%mesh_len)
