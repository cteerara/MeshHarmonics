import numpy as np
import sys
import os
from pathlib import Path
import copy

# ------------------------------------------------------- #

from meshing_utils import *

prefix_dir = sys.argv[1]
mesh_prefix = sys.argv[2]
num_steps = int(sys.argv[3])

write_series_to_pvd(prefix_dir, mesh_prefix, 'stl', num_steps)

