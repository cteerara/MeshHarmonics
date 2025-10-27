import gmsh

def build_surface_mesh_from_stl(stl_file, output_file=None):
    if output_file is None:
        output_file = stl_file[:-3]+'msh'
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.merge(stl_file)
    n = gmsh.model.getDimension()
    s = gmsh.model.getEntities(n)
    l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(n)
    gmsh.write(output_file)
    gmsh.finalize()

def build_volume_mesh_from_stl(stl_file, output_file=None):
    if output_file is None:
        output_file = stl_file[:-3]+'msh'
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.merge(stl_file)
    n = gmsh.model.getDimension()
    s = gmsh.model.getEntities(n)
    l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
    gmsh.model.geo.addVolume([1])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(output_file)
    gmsh.finalize()



# # build_surface_mesh_from_stl('sphere_clipped_rm.stl')
# sphere_stl = 'mesh/sphere_0.1000.stl'
# sphere_stl = 'sphere_clipped_rm.stl'
# build_volume_mesh_from_stl(sphere_stl, output_file='sphere.vtk')
