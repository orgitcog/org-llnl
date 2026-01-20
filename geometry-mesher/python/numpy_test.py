import numpy as np
import meshio # for mesh output
from geometry_bindings import universal_mesh

def func(x, y, z):
    return x*x + y*y + z*z - 0.75

# note: try to take cell_size < cylinder radius,
# or the underlying mesh can get too coarse to capture geometry
cell_size = 0.05

bounds = np.array([[-1.1, -1.1, -1.1], [1.1, 1.1, 1.1]], dtype=float)

(vertices, tris, tets) = universal_mesh(func, bounds, cell_size)
meshio.write_points_cells("foo.stl", vertices, [("triangle", tris)])
meshio.write_points_cells("foo.vtu", vertices, [("tetra", tets)])


#    [-1.5, -0.8, 0,   1.5, -0.8, 0,   0.1, 0.1],
#       ^     ^   ^     ^     ^   ^     ^    ^
#           p1              p2          r1   r2
cylinders = np.array([
    # along x-direction
    [-1.5, -0.8, 0,   1.5, -0.8, 0,   0.1, 0.1],
    [-1.5, -0.4, 0,   1.5, -0.4, 0,   0.1, 0.1],
    [-1.5,  0.0, 0,   1.5,  0.0, 0,   0.1, 0.1],
    [-1.5,  0.4, 0,   1.5,  0.4, 0,   0.1, 0.1],
    [-1.5,  0.8, 0,   1.5,  0.8, 0,   0.1, 0.1],

    # along y-direction
    [-0.8, -1.5, 0.18,   -0.8, 1.5, 0.18,   0.1, 0.1],
    [-0.4, -1.5, 0.18,   -0.4, 1.5, 0.18,   0.1, 0.1],
    [ 0.0, -1.5, 0.18,    0.0, 1.5, 0.18,   0.1, 0.1],
    [ 0.4, -1.5, 0.18,    0.4, 1.5, 0.18,   0.1, 0.1],
    [ 0.8, -1.5, 0.18,    0.8, 1.5, 0.18,   0.1, 0.1]
], dtype=float)

# note: try to take cell_size < cylinder radius,
# or the underlying mesh can get too coarse to capture geometry
cell_size = 0.05

(vertices, tris, tets) = universal_mesh(cylinders, bounds, cell_size)
meshio.write_points_cells("bar.stl", vertices, [("triangle", tris)])
meshio.write_points_cells("bar.vtu", vertices, [("tetra", tets)])
