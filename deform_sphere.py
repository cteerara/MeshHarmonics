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

def _abs_pow(x, p):
    return np.sign(x)*np.abs(x)**p

def _transform_sph_to_sq(V_sph, rx, ry, rz, e1=1, e2=1):
    x_sph = V_sph[:,0]
    y_sph = V_sph[:,1]
    z_sph = V_sph[:,2]
    theta = np.arctan2(np.sqrt(x_sph**2 + y_sph**2), z_sph)
    phi = np.arctan2(y_sph, x_sph)
    V_sq = np.zeros(V_sph.shape)
    V_sq[:,0] = rx * _abs_pow(np.sin(theta), e1) \
                * _abs_pow(np.cos(phi), e2)
    V_sq[:,1] = ry * _abs_pow(np.sin(theta), e1) \
                * _abs_pow(np.sin(phi), e2)
    V_sq[:,2] = rz * _abs_pow(np.cos(theta), e1)
    return V_sq

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

def fMHT_vector(mht, vec):
    vec_hat = []
    for i in range(vec.shape[1]):
        vec_hat.append(mht.fMHT(vec[:,i]))
    return np.array(vec_hat).T

def iMHT_vector(mht, vec_hat):
    vec = []
    for i in range(vec_hat.shape[1]):
        vec.append(mht.iMHT(vec_hat[:,i]))
    return np.array(vec).T


def main():

    src_mesh = sys.argv[1]
    dest_mesh = sys.argv[2]
    num_modes = int(sys.argv[3])
    out_mesh = sys.argv[4]

    msh_file = create_msh(src_mesh)
    mesh = Mesh(mesh_file=msh_file, gdim=3)
    V, F = get_dfx_surface_mesh(mesh.msh)

    smesh_tgt = SurfaceMesh(stl_file=dest_mesh)
    smesh_tgt.build_normal(True)

    sdf, nsdf = smesh_tgt.signed_distance(V)
    u = -nsdf*sdf[:,np.newaxis] * 0.1

    mht = MeshHarmonicsSolver(mesh, num_modes=num_modes)
    mht.build_eigen_problem()
    mht.solve()
    u_hat = fMHT_vector(mht, u)
    u_T = iMHT_vector(mht, u_hat)

    new_V = V + u_T
    V, F = remesh(new_V, F, remesh_nitr=20, target_len_percent=1)
    write_mesh(V, F, out_mesh)

if __name__ == '__main__':

    main()
