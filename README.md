# MeshHarmonics

This code computes the harmonic functions on a mesh. These values are computed by solving for the Eigenfunctions of the Laplace-Beltrami operator on a manifold mesh. This code can handle pure Neumann Laplacian and Dirichlet Laplacian (Eigenfunctions in the homogeneous BC space). 

This code requires the following libraries and their dependencies

1. [MeshPrep](https://github.com/Cardiovascular-Imaging-Resarch-Lab/MeshPrep) : This is a close source code to CVIRL
2. [fenicsx 0.9.0](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/index.html)
3. [FLATiron](https://github.com/flowlabcu/FLATiron) : This is a FEniCSx wrapper for the physics problem definitions

