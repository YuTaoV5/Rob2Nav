"""Conservative local box around the proximal calf mesh at each knee."""
from itertools import product
from functools import lru_cache

import numpy as np

# Union of the official AS2-W calf meshes within 50 mm below the knee.
# The box rounds the mesh extrema outwards and includes the complete slice.
KNEE_ENVELOPE_CORNERS = tuple(product((-.024,.017),(-.019,.019),(-.050,.026)))
MIN_KNEE_SURFACE_M = .015

# Rounded union of the four official calf meshes below local z = -0.18 m.
DISTAL_CALF_ENVELOPE_CORNERS = tuple(product((-.041, .042), (-.038, .038), (-.252, -.180)))
@lru_cache(maxsize=4)
def calf_hull_vertices(mesh_dir):
    from scipy.spatial import ConvexHull

    triangle_dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    hulls = []
    for leg in ("FL", "FR", "RL", "RR"):
        path = mesh_dir / f"{leg}_calf.STL"
        triangles = np.fromfile(path, dtype=triangle_dtype, offset=84)
        vertices = np.unique(triangles["vertices"].reshape(-1, 3), axis=0)
        hulls.append(vertices[ConvexHull(vertices).vertices])
    return hulls


