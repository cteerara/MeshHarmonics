import numpy as np
import sys
import os
from pathlib import Path
import copy

# ------------------------------------------------------- #

from geomdl import BSpline, multi
from geomdl.visualization import VisMPL

# --- Define first surface ---
surf1 = BSpline.Surface()

# Degrees
surf1.degree_u = 2
surf1.degree_v = 2

# Control points grid for surf1 (3x3)
ctrlpts1 = [
    [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0],
    [1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [1.0, 2.0, 0.5],
    [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]
]
surf1.set_ctrlpts(ctrlpts1, 3, 3)

# Knot vectors
surf1.knotvector_u = [0, 0, 0, 1, 1, 1]
surf1.knotvector_v = [0, 0, 0, 1, 1, 1]


# --- Define second surface ---
surf2 = BSpline.Surface()
surf2.degree_u = 2
surf2.degree_v = 2

# Control points grid for surf2 (3x3)
# Notice that the first row matches the last row of surf1
ctrlpts2 = [
    [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0],
    [3.0, 0.0, -0.5], [3.0, 1.0, -0.5], [3.0, 2.0, -0.5],
    [4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 2.0, 0.0]
]
surf2.set_ctrlpts(ctrlpts2, 3, 3)

surf2.knotvector_u = [0, 0, 0, 1, 1, 1]
surf2.knotvector_v = [0, 0, 0, 1, 1, 1]


# --- Stitch surfaces into a multi-surface container ---
msurf = multi.SurfaceContainer()
msurf.add(surf1)
msurf.add(surf2)

# Visualization
msurf.vis = VisMPL.VisSurface()
msurf.render()

