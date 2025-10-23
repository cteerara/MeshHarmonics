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

class MeshHarmonicsTransform(SteadyScalarTransport):
    '''
    This problem solve for the Eigenfunction of the Laplacian operator on a manifold mesh.

    This is essentially equivalent to solving the Helmholtz equation with an arbitrary Eigenvalue k:

    Au=kMu

    where A and M are the discrete Laplacian and Mass matrix respectively
    '''
    def __init__(self, *args, **kwargs):

        self._lump_mass = kwargs.pop("lump_mass", True)

        assert("num_modes" in kwargs)
        self._num_modes = kwargs.pop("num_modes")

        super().__init__(*args, **kwargs)

    def get_num_modes(self):
        return self._num_modes

    def build_eigen_problem(self, bc_type='neumann', bc_dofs=[], space_family='CG', space_deg=1):

        '''
        bc_type is either neumann or dirichlet
        '''

        assert(bc_type == 'neumann' or bc_type == 'dirichlet')
        self._bc_type = bc_type

        self.set_element(space_family, space_deg)
        self.build_function_space()
        self.set_diffusivity(1.)

        V = self.get_function_space()

        D = self.get_diffusivity()
        u = ufl.TrialFunction(V)
        w = self.get_test_function()

        a_form = self._diffusive_form(u, D) # Laplacian matrix
        m_form = u*w*self.dx # Mass matrix

        self._A = self._assemble_form(a_form)
        self._M = self._assemble_form(m_form)

        if self._lump_mass:
            M_row_sum = self._lump_matrix(self._M)
            M_lumped = PETSc.Mat().createAIJ(self._M.getSize(), nnz=1)
            M_lumped.setDiagonal(M_row_sum)
            self._M = M_lumped

    def _apply_neumann_bcs(self):
        '''
        Here, we handle a case where no Dirichlet boundary condition is applied to the matrix.
        The Laplacian operator has rank n-1 when no Dirichlet BC is applied and we will get
        an Eigenvalue=0 mode with a garbage values in the Eigenfunction. 
        Here, we will deflate the space a constant eigenvector representing the 0th mode 
        to remove the one nullspace from the Eigenvalue problem
        '''
        n, _ = self._A.getSize()
        self.vec_one = self._A.createVecLeft()
        self.vec_one.set(1.0)
        self.vec_one.assemble()
        self.vec_one.normalize()
        self._EPS.setDeflationSpace([self.vec_one])

    def _apply_dirichlet_bcs(self):
        raise ValueError("Not implemented yet!")

    def solve(self):

        # Solve eigenvalue problem
        nev = self._num_modes
        self._EPS = SLEPc.EPS().create(comm=self.mesh.msh.comm)
        self._EPS.setOperators(self._A, self._M)
        self._EPS.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        if self._bc_type == 'dirichlet':
            self._apply_dirichlet_bcs()
        else:
            self._apply_neumann_bcs()

        # Smallest eigenvalues
        # We should only expect positive eigenvalues since Laplacian is SPSD
        self._EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        self._EPS.setTarget(1e-10)

        self._EPS.setDimensions(nev=nev)
        self._EPS.setFromOptions()
        self._EPS.solve()
        return self._EPS

    def get_eigen_pair(self, mode):
        '''
        Return Eigenvalue, RealEigenvector, ComplexEigenvector at the specified mode
        '''
        vr, vi = self._A.getVecs()
        eigval = self._EPS.getEigenpair(mode, vr, vi)
        return eigval, vr[:], vi[:]

    def _assemble_form(self, form, bcs=[]):
        F = fem.petsc.assemble_matrix(fem.form(form), bcs)
        F.assemble()
        return F

    def _lump_matrix(self, M):
        '''
        Lump each row of a PETSc matrix M into its diagonal
        returns a PETSc vector of the lumped value
        '''
        M_row_sum = M.createVecLeft()
        M.getRowSum(M_row_sum)
        M_row_sum = M.getDiagonal()
        return M_row_sum

def get_dfx_surface_mesh(dfx_msh):
    facets = dfx_msh.topology.connectivity(2, 0).array.reshape((-1, 3))
    vertices = dfx_msh.geometry.x[:]
    return vertices, facets

def create_msh(stl_file):
    geo_str = 'Mesh.MshFileVersion = 2.0;\nMerge "%s";\nSurface Loop(1) = {1};\nPhysical Surface(1) = {1};'%(stl_file.split("/")[-1])
    with open(stl_file[:-3]+'geo', 'w') as fid:
        fid.write(geo_str)
    os.system('gmsh -2 %s -o %s'%(stl_file[:-3]+'geo', stl_file[:-3]+'msh'))
    return stl_file[:-3]+'msh'

# V_LA, F_LA = read_mesh('/mnt/d/CTeeraratkul/ExternalData/2018_UTAH_MICCAI/Testing Set/4URSJYI2QUH1T5S5PP47/Segmentation_Segment_1.stl')
# xc_LA = np.mean(V_LA, axis=0)
# r_LA = np.max([np.linalg.norm(p-xc_LA) for p in V_LA])
# V, F = read_mesh('mesh/sphere_0.0500.stl')

num_modes = 100
msh_file = create_msh('mesh/sphere_0.0500.stl')
mesh = Mesh(mesh_file=msh_file, gdim=3)
mht = MeshHarmonicsTransform(mesh, num_modes=num_modes, lump_mass=True)
mht.build_eigen_problem()
mht.solve()

conv_modes = min(mht.get_num_modes(), mht._EPS.getConverged())
# conv_modes = mht.get_num_modes()
# U = np.zeros((mesh.msh.geometry.x.shape[0], conv_modes))
# for i in range(conv_modes):
#     lam, vr, vi = mht.get_eigen_pair(i)
#     U[:,i] = vr
#     print(i, lam)

U = np.zeros((mesh.msh.geometry.x.shape[0], conv_modes+1))
U[:,0] = mht.vec_one[:]
for i in range(1, conv_modes):
    lam, vr, vi = mht.get_eigen_pair(i-1)
    U[:,i] = vr
    print(i, lam)


V = mht.get_function_space()
x = ufl.SpatialCoordinate(mesh.msh)
r = ufl.sqrt(x[0]*x[0] + x[1]*x[1] * x[2]*x[2])
theta = ufl.atan2(ufl.sqrt(x[0]*x[0] + x[1]*x[1]), x[2])
phi = ufl.atan2(x[1], x[0])
expr = dolfinx.fem.Expression(ufl.exp(-(theta - ufl.pi/2)/1), V.element.interpolation_points())
g = fem.Function(V)
g.interpolate(expr)
area = fem.assemble_scalar(fem.form(g*mht.dx))
expr = dolfinx.fem.Expression(ufl.exp(-(theta - ufl.pi/2)/1)/area, V.element.interpolation_points())
g.interpolate(expr)
g = g.x.array[:]

point_data = {}

# U^T.U = I, so we can just do this instead of a least square
# U = U[:,1:]
# U[:,0] = 1
# g_hat = U.T @ g
# g_hat, _, _, _ =
# g_hat, res, rank, s = np.linalg.lstsq(U, g, rcond=None)
# g2 = U @ g_hat
# print("Lsq rank:", rank)
# print("Lsq res:", res)

mesh_V, mesh_F = get_dfx_surface_mesh(mesh.msh)

for i in range(U.shape[1]):
    point_data['U%03d'%i] = U[:,i]

# point_data['g'] = g
# point_data['g2'] = g2
smesh = meshio.Mesh(mesh_V, [("triangle", mesh_F)], point_data=point_data)
smesh.write('f.vtu')
sys.exit()

plt.semilogy(np.abs(g_hat))
plt.grid(True)
plt.show()

sys.exit()

# f = fem.Function(V)
# f_expr = dolfinx.fem.Expression(x[0]+x[1]+x[2], V.element.interpolation_points())
# f.interpolate(f_expr)
# f = f.x.array[:]

# eps = 1e-1
# f = f + np.random.uniform(-eps, eps, len(f))
# point_data['f_noisy'] = f

# f_hat, _, _, _ = np.linalg.lstsq(U, f)

# f = U @ f_hat
# point_data['f_fit'] = f

# f = U @ (f_hat*g_hat)
# point_data['f_conv'] = f

# smesh = meshio.Mesh(mesh_V, [("triangle", mesh_F)], point_data=point_data)
# smesh.write('f.vtu')
