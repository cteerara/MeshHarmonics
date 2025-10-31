import meshio
import vtk
import os
import sys

from meshprep.features import Holes
from meshprep.process import remesh, laplacian_smoothing, build_flat_cap
from meshprep.io import read_mesh, write_mesh
from meshprep.features import Holes

def remesh_LA(V, F, smooth=True):
    if smooth:
        V, F = laplacian_smoothing(V, F, smooth_type='hc')
    holes = Holes(V, F)
    V, F = holes.flatten_holes()
    V, F = remesh(V, F,
                  target_len_percent=1, remesh_nitr=10,
                  smooth_on=True, smooth_nitr=10)
    return V, F

def cap_LA(V, F):
    holes = Holes(V, F)
    V, F = holes.flatten_holes() # need to do this again after remesh
    V_cap, F_cap = build_flat_cap(V, F)
    return V_cap, F_cap

def _get_zero_padding_format(width):
    fmt = '%%s%%0%dd.%%s'%width
    return fmt

def write_series_to_pvd(prefix_dir, mesh_prefix, mesh_suffix, num_mesh_file, zero_pad_width=6):

    fmt = _get_zero_padding_format(zero_pad_width)
    assert(mesh_suffix in ['stl', 'vtu'])
    # mesh_names = [os.path.join(prefix_dir, fmt%(mesh_prefix, i, mesh_suffix)) for i in range(num_mesh_file)]
    mesh_names = [os.path.join(prefix_dir, "%s%d.%s"%(mesh_prefix, i, mesh_suffix)) for i in range(num_mesh_file)]

    pvd_str = '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n  <Collection>\n'
    for i, mesh_name in enumerate(mesh_names):
        print("Writing mesh %s to pvd"%mesh_name)
        if mesh_suffix == 'stl':
            vtu_name = mesh_name[:-3]+'vtu'
            mesh = meshio.read(mesh_name)
            mesh.write(vtu_name, binary=True)
        else:
            vtu_name = mesh_name
        pvd_str += '    <DataSet timestep="%d" group="" part="0" file="%s"/>\n'%(i, vtu_name.split("/")[-1])
    pvd_str+='  </Collection>\n</VTKFile>'

    pvd_prefix = mesh_prefix
    while pvd_prefix.endswith('_'):
        pvd_prefix = pvd_prefix[:-1]

    with open(os.path.join(prefix_dir, '%s.pvd'%pvd_prefix), 'w') as fid:
        fid.write(pvd_str)
    print("%s is written!"%("%s.pvd"%pvd_prefix))

def build_3d_mesh_from_stl(stl_file):

    prefix = stl_file[:-4]
    geo_file = prefix+'.geo'
    with open(geo_file, 'w') as fid:
        fid.write("Mesh.MshFileVersion = 2.0;\n")
        fid.write('Merge "%s";\n'%stl_file)
        fid.write('Surface Loop(1) = {1};\nPhysical Surface(1) = {1};Volume(1) = {1};\nPhysical Volume(1) = {1};')
    os.system("geo2h5 -d 3 -m %s -o %s"%(geo_file, prefix))
    return prefix+'.h5'

def write_pc(points, fname, func=None):
    import vtk
    from vtkmodules.util import numpy_support
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    if func is not None:
        scalars_vtk = numpy_support.numpy_to_vtk(func)
        polydata.GetPointData().SetScalars(scalars_vtk)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(polydata)
    writer.Write()


