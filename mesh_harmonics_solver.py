import numpy as np
import sys
import os
from pathlib import Path
import copy

# ------------------------------------------------------- #

import matplotlib.pyplot as plt
from flatiron_tk.mesh import Mesh
from flatiron_tk.physics import SteadyScalarTransport
import flatiron_tk
from meshprep.features import Holes
import dolfinx
import meshio
from dolfinx import fem
import ufl
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

class MeshHarmonicsSolver(SteadyScalarTransport):
    '''
    This problem solve for the Eigenfunction of the Laplacian operator on a manifold mesh.

    This is essentially equivalent to solving the Helmholtz equation with an arbitrary Eigenvalue k:

    Au=kMu

    where A and M are the discrete Laplacian and Mass matrix respectively
    '''
    def __init__(self, *args, **kwargs):

        self._lump_mass = kwargs.pop("lump_mass", True)

        assert("num_modes" in kwargs)
        self._requested_num_modes = kwargs.pop("num_modes")

        super().__init__(*args, **kwargs)

        self._num_bnds = self._mark_manifold_boundaries()

    def get_num_modes(self):
        _, B = self._H.getSize()
        return B

    def build_eigen_problem(self, homogeneous_bnds=[], space_family='CG', space_deg=1):

        '''
        homogeneous_bnds is a list of dirichlet bnds used
        '''

        self._homogeneous_bnds = homogeneous_bnds
        for bnd_idx in homogeneous_bnds:
            assert(bnd_idx <= self._num_bnds) # bnd_idx starts at 1

        # Detect whether this is a pure Neumann problem
        # We do this because we need to account for the single nullspace
        # that wrise in a pure Neumann problem
        self._pure_neumann = False
        if len(homogeneous_bnds)==0:
            self._pure_neumann = True

        # Set function space
        self.set_element(space_family, space_deg)
        self.build_function_space()
        self.set_diffusivity(1.)

        V = self.get_function_space()
        D = self.get_diffusivity()
        u = ufl.TrialFunction(V)
        w = self.get_test_function()

        # Set laplacian and mass form
        a_form = self._diffusive_form(u, D) # Laplacian matrix
        m_form = u*w*self.dx # Mass matrix
        
        # Set dirichlet boundary if any exist
        zero = flatiron_tk.constant(self.mesh, 0.)
        bc_dict = {}
        for bnd_idx in self._homogeneous_bnds:
            bc_dict[bnd_idx] = {'type': 'dirichlet', 'value': zero}
        self.set_bcs(bc_dict)

        self._A = self._assemble_form(a_form, self.dirichlet_bcs)
        self._M = self._assemble_form(m_form, self.dirichlet_bcs)

        if self._lump_mass:
            M_row_sum = self._lump_matrix(self._M)
            M_lumped = PETSc.Mat().createAIJ(self._M.getSize(), nnz=1)
            M_lumped.setDiagonal(M_row_sum)
            self._M = M_lumped

    def _deflate_nullspace(self):
        '''
        Here, we handle a case where no Dirichlet boundary condition is applied to the matrix.
        The Laplacian operator has rank n-1 when no Dirichlet BC is applied and we will get
        an Eigenvalue=0 mode with a garbage values in the Eigenfunction.
        Here, we will deflate the space a constant eigenvector representing the 0th mode
        to remove the one nullspace from the Eigenvalue problem
        '''
        self._requested_num_modes -= 1 # reduce requested modes because we will add the trivial constant mode to the list
        n, _ = self._A.getSize()
        self._const_vec = self._A.createVecLeft()
        self._const_vec.set(1.0)
        self._const_vec.assemble()

        # Normalize const_vec in an M-norm sense.
        # Define constant vector: x = _self.const_vec
        # We want <x, Mx> = 1
        y = self._M.createVecLeft()
        self._M.mult(self._const_vec, y) # y = Mx
        scale = 1/(self._const_vec.dot(y))**0.5 # scale = 1/sqrt(x.Mx)
        self._const_vec.scale(scale)

        # Deflate the space with the constant vector
        self._EPS.setDeflationSpace([self._const_vec])

    def solve(self):

        # Solve eigenvalue problem
        self._EPS = SLEPc.EPS().create(comm=self.mesh.msh.comm)
        self._EPS.setOperators(self._A, self._M)
        self._EPS.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        if self._pure_neumann:
            self._deflate_nullspace()

        # Smallest eigenvalues
        # We should only expect positive eigenvalues since Laplacian is SPSD
        self._EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        self._EPS.setTarget(1e-10)

        self._EPS.setDimensions(nev=self._requested_num_modes)
        self._EPS.setFromOptions()
        self._EPS.solve()

        self._set_transform_matrix()
        return self._EPS

    def get_eigen_functions(self):
        return self._H.getDenseArray()

    def _set_transform_matrix(self):
        eigvecs = []
        if self._pure_neumann:
            eigvecs.append(self._const_vec)

        for i in range(self._requested_num_modes):
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

        # Verify that our matrix is orthonormal
        C = self._M.matMult(self._H)
        D = self._H.transposeMatMult(C)
        D = D.getDenseArray()
        print(D)
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
        self._H.mult(f_hat_petsc, f)
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

    def _mark_manifold_boundaries(self):

        mesh = self.mesh
        tp = mesh.msh.topology
        tp.create_connectivity(mesh.get_fdim(), mesh.get_tdim())
        tp.create_connectivity(2, 0)
        tp.create_connectivity(0, 1)
        tp.create_connectivity(1, 0)
        v2e = tp.connectivity(0, 1)
        num_edges = len(tp.connectivity(1,0).offsets)-1
        F = tp.connectivity(2, 0).array.reshape((-1, 3))
        vertices = mesh.msh.geometry.x[:]
        _holes = Holes(vertices, F)
        bnd_edges = []
        edge_tag = []
        for hole_idx, vertices_list in enumerate(_holes.boundary_ids()):
            for vi in vertices_list:
                for ei in self._get_row_data(v2e, vi):
                    if not self._is_boundary_edge(tp, ei): continue
                    bnd_edges.append(ei)
                    edge_tag.append(hole_idx+1)
        bnd = dolfinx.mesh.meshtags(mesh.msh, 1, bnd_edges, edge_tag)
        mesh.boundary = bnd

        return _holes.num_holes()

    def _is_boundary_edge(self, tp, ei):
        conn = tp.connectivity(1, 2)
        connected_facet = self._get_row_data(conn, ei)
        return (len(connected_facet) == 1)

    def _get_row_data(self, conn, i):
        array = conn.array
        offsets = conn.offsets
        start = offsets[i]
        end = offsets[i+1]
        return array[start:end]


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


if __name__ == '__main__':

    from meshprep.io import read_mesh, write_mesh

    # V_LA, F_LA = read_mesh('/mnt/d/CTeeraratkul/ExternalData/2018_UTAH_MICCAI/Testing Set/4URSJYI2QUH1T5S5PP47/Segmentation_Segment_1.stl')
    # xc_LA = np.mean(V_LA, axis=0)
    # r_LA = np.max([np.linalg.norm(p-xc_LA) for p in V_LA])
    # V, F = read_mesh('mesh/sphere_0.0500.stl')

    V, F = read_mesh('mesh/sphere_0.0500.stl')
    eps = 2e-2
    V = V + np.random.uniform(-eps, eps, V.shape)
    write_mesh(V, F, 'tmp_mesh.stl')

    num_modes = 36
    msh_file = create_msh('tmp_mesh.stl')
    mesh = Mesh(mesh_file=msh_file, gdim=3)
    mht = MeshHarmonicsSolver(mesh, num_modes=num_modes, lump_mass=False)
    mht.build_eigen_problem()
    mht.solve()

    point_data = {}
    mesh_V, mesh_F = get_dfx_surface_mesh(mesh.msh)

    x_hat = mht.fMHT(mesh_V[:,0])
    y_hat = mht.fMHT(mesh_V[:,1])
    z_hat = mht.fMHT(mesh_V[:,2])

    new_mesh_V = np.zeros(mesh_V.shape)
    new_mesh_V[:,0] = mht.iMHT(x_hat)
    new_mesh_V[:,1] = mht.iMHT(y_hat)
    new_mesh_V[:,2] = mht.iMHT(z_hat)

    write_mesh(mesh_V, mesh_F, 'mesh_original.stl')
    write_mesh(new_mesh_V, mesh_F, 'mesh_T.stl')

    sys.exit()

    x = mesh_V[:,0]
    x_noisy = x + np.random.uniform(-eps, eps, len(x))
    x_hat = mht.fMHT(x_noisy)
    plt.semilogy(np.abs(x_hat))
    plt.grid(True)
    plt.show()

    x_T = mht.iMHT(x_hat)
    x_f = mht.iMHT(x_hat*g_hat)

    # U = mht._H.getDenseArray()
    # for i in range(U.shape[1]):
    #     point_data['U%03d'%i] = U[:,i]
    point_data['g'] = g
    point_data['g2'] = g2
    point_data['x_noisy'] = x_noisy
    point_data['x_T'] = x_T
    point_data['x_filt'] = x_f

    smesh = meshio.Mesh(mesh_V, [("triangle", mesh_F)], point_data=point_data)
    smesh.write('f.vtu')
