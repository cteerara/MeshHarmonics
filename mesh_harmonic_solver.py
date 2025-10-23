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
        self._num_modes -= 1 # reduce requested modes because we will add the trivial constant mode to the list
        n, _ = self._A.getSize()
        self._const_vec = self._A.createVecLeft()
        self._const_vec.set(1.0)
        self._const_vec.assemble()
        # self._const_vec.normalize()

        # Normalize const_vec in an M-norm sense.
        # Define constant vector: x = _self.const_vec
        # We want <x, Mx> = 1
        y = self._M.createVecLeft()
        self._M.mult(self._const_vec, y) # y = Mx
        scale = 1/(self._const_vec.dot(y))**0.5 # scale = 1/sqrt(x.Mx)
        self._const_vec.scale(scale)

        # Deflate the space with the constant vector
        self._EPS.setDeflationSpace([self._const_vec])

    def _apply_dirichlet_bcs(self):
        raise ValueError("Not implemented yet!")

    def solve(self):

        # Solve eigenvalue problem
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

        self._EPS.setDimensions(nev=self._num_modes)
        self._EPS.setFromOptions()
        self._EPS.solve()

        self._set_transform_matrix()
        return self._EPS

    def _set_transform_matrix(self):
        if self._bc_type == 'neumann':
            self._set_transform_matrix_neumann()
        elif self._bc_type == 'dirichlet':
            self._set_transform_matrix_dirichlet()

    def _set_transform_matrix_neumann(self):
        eigvecs = [self._const_vec]
        for i in range(self._num_modes):
            _, vr, _ = self.get_eigen_pair(i)
            eigvecs.append(vr)

        n = eigvecs[0].getSize()
        self._H = PETSc.Mat().createDense([n, len(eigvecs)], 
                                          comm=self.mesh.msh.comm)
        self._H.setUp()
        for _j, v in enumerate(eigvecs):
            self._H.setValues(range(n), [_j], v.getArray(), addv=PETSc.InsertMode.INSERT_VALUES)
        self._H.assemble()

        self._MH = self._M.matMult(self._H)

        # # Verify that our matrix is orthonormal
        # C = self._M.matMult(self._H)
        # D = self._H.transposeMatMult(C)
        # D.view()

    def fMHT(self, f):
        P, B = self._MH.getSize()

        # Set PETSc vector of the point-wise value
        f_petsc = PETSc.Vec().createMPI(P, comm=self.mesh.msh.comm)
        f_petsc.setArray(f)

        # Create basis-space as a target 
        f_hat = PETSc.Vec().createMPI(B, comm=self.mesh.msh.comm)
        self._MH.multTranspose(f_petsc, f_hat)
        return f_hat

    def iMHT(self, f_hat):
        P, B = self._MH.getSize()

        f_hat_petsc = PETSc.Vec().createMPI(B, comm=self.mesh.msh.comm)
        f_hat_petsc.setArray(f_hat)

        f = PETSc.Vec().createMPI(P, comm=self.mesh.msh.comm)
        self._H.mult(f_hat, f)
        return f

    def _set_transform_matrix_dirichlet(self):
        raise ValueError("Not implemented yet!!")

    def get_eigen_pair(self, mode):
        '''
        Return Eigenvalue, RealEigenvector, ComplexEigenvector at the specified mode
        '''
        vr, vi = self._A.getVecs()
        eigval = self._EPS.getEigenpair(mode, vr, vi)
        return eigval, vr, vi

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

from meshprep.io import read_mesh, write_mesh

# V_LA, F_LA = read_mesh('/mnt/d/CTeeraratkul/ExternalData/2018_UTAH_MICCAI/Testing Set/4URSJYI2QUH1T5S5PP47/Segmentation_Segment_1.stl')
# xc_LA = np.mean(V_LA, axis=0)
# r_LA = np.max([np.linalg.norm(p-xc_LA) for p in V_LA])
# V, F = read_mesh('mesh/sphere_0.0500.stl')

num_modes = 100
msh_file = create_msh('mesh/sphere_0.0500.stl')
mesh = Mesh(mesh_file=msh_file, gdim=3)
mht = MeshHarmonicsTransform(mesh, num_modes=num_modes, lump_mass=False)
mht.build_eigen_problem()
mht.solve()

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

g_hat = mht.fMHT(g)
g2 = mht.iMHT(g_hat)
plt.semilogy(np.abs(g_hat))
plt.grid(True)
plt.show()

point_data = {}
mesh_V, mesh_F = get_dfx_surface_mesh(mesh.msh)
# U = mht._H.getDenseArray()
# for i in range(U.shape[1]):
#     point_data['U%03d'%i] = U[:,i]
point_data['g'] = g
point_data['g2'] = g2
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
