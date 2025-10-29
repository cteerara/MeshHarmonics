import numpy as np
import sys
import os
from pathlib import Path
import copy

# ------------------------------------------------------- #

from meshprep.io import read_mesh, write_mesh

V, F = read_mesh('mesh/sphere_0.0500.stl')

def cart2sph(p):
    xc = np.mean(p, axis=0)
    x = p[:,0] - xc[0]
    y = p[:,1] - xc[1]
    z = p[:,2] - xc[2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.atan2(np.sqrt(x**2 + y**2), z)
    phi = np.atan2(y, x)
    return r, theta, phi


r, theta, phi = cart2sph(V)
