Overview
=====

A small collection of various geometry tools:
  - "universal" meshing: (see these papers for more info: [2D](https://www.dropbox.com/s/7icl393gmo6xj02/universal_meshes_2D.pdf?dl=0) and [3D](https://www.dropbox.com/s/hh6zyx6941et7no/universal_meshing_3D.pdf?dl=0]))
  - bounding volume hierarchy construction and traversal (based on LBVH algorithm presented here: [part 1](https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/), [part 2](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/), [part 3](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/))
  - some basic geometric primitives and their signed distance functions (see https://iquilezles.org/articles/distfunctions/)
  - Marching Triangles / Tetrahedra
  - Naive Isosurface stuffing (https://people.eecs.berkeley.edu/~jrs/papers/stuffing.pdf)

Note: this library may change rapidly, so do not expect any form of API stability.

LLNL-CODE-2004362

Requirements
========

- A C++17 compatible compiler (on LC, call `module load gcc/12.1.1`)
- If enabling the OpenGL tools, install OpenGL first (run `sudo apt-get install xorg-dev libglu1-mesa-dev` on ubuntu)

Building the library and its dependencies
=================

1. Clone the repo:

`git clone ssh://git@czgitlab.llnl.gov:7999/topopt/geometry-mesher.git`

---------

2. Configure with `cmake`

with no OpenGL stuff enabled
  
`cd path/to/geometry-mesher && cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=True`

OR

with OpenGL and OpenGL examples enabled
  
`cd path/to/geometry-mesher && cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENGL=True -DBUILD_TESTS=True -DBUILD_EXAMPLES=True`

Note: 
- the first configure w/ OpenGL takes a moment to fetch and build all of its dependencies
- if CMake finds libtiff, it will build the `import_tiff` and `universal_mesh` overload for tiff files

---------

3. build

`cmake --build build --parallel`


Integration into an existing CMake project 
=================

Add this project as subdirectory

`add_subdirectory(path/to/geometry-mesher)`

and link against the `geometry` or `opengl_tools` targets

`target_link_libraries(my_target PUBLIC geometry)`

Using the meshgen executable
============================

After making geometry-mesher, you will have the `build/meshgen` executable. The intent of this code is to mesh the
union of "capsule" geometric primitives that are specified at runtime via a binary input file that provides the
endpoints and radii of each capsule. This allows one to define a relatively large class of structures, including
truss lattices like the octet truss, Isotruss, etc., as well as structures formed by depositing continuous 
filaments, including common additive manufacturing proceeses like fused deposition modeling (FDM) and direct ink
writing (DIW). This executable produces a tetrahedral mesh, and can output it, or its boundary (e.g., for visualization)
in several differerent file formats, including VTK, VTU, GMSH, and STL.

The binary input file consists of eight single-precision floating-point numbers per capsule; in order, these are
x1, y1, z1, x2, y2, z2, r1, and r2, where x, y, and z are coordinates in three-dimensional space, r is a radius, and
the trailing numeral represents the endpoint index of the capsule.

For additional details, call `build/meshgen --help`.

An example invocation is:

`build/meshgen -i data/icosahedron_capsules.bin -o icosahedron.stl --cellsize 0.1`

which meshes capsules laid along the edges of a regular icosahedron and writes a binary STL file with about 77k facets.
This process takes somewhere between a quarter and half a second.
