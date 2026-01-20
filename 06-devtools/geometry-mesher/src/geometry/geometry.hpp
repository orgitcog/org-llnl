#pragma once

#include "fm/types/vec.hpp"
#include "fm/types/AABB.hpp"
#include "fm/types/matrix.hpp"

#include <array>
#include <tuple>
#include <vector>
#include <cstdint>
#include <cstring>
#include <functional>

namespace geometry {

using namespace fm;

namespace tpms {
  float gyroid(vec3f x);
  float schwarz_p(vec3f x);
  float schwarz_d(vec3f x);
  float neovius(vec3f x);
  float schoen_iwp(vec3f x);
  float fischer_koch_s(vec3f x);
  float fischer_koch_y(vec3f x);
  float fischer_koch_cp(vec3f x);
}

struct Ball {
  vec3f c;
  float r;
  float SDF(vec3f x) const;
};

struct Capsule {
  vec3f p1, p2;
  float r1, r2;
  float SDF(vec3f x) const;
};

struct Filament {
  std::vector<Ball> nodes;
  float SDF(vec3f x) const;
};

// revolves a polygon defined by v in the xy plane about the y axis
struct RevolvedPolygon {
  std::vector<vec2f> v;
  float SDF(vec3f x) const;
};

struct Line {
  vec3f vertices[2];
  float SDF(vec3f x) const;
};

struct Triangle {
  vec3f vertices[3];
  float SDF(vec3f x) const;
};

struct Quad {

  //
  //  4 ---- 3
  //  |      |
  //  |      |
  //  0 ---- 1
  //
  Quad(mat4x3f corners);
  float SDF(vec3f p) const;
  float interior_SDF(vec3f p) const;

  mat4x3f X; // vertices
  mat4x3f N; // normals

 private:
  vec3f bilinear_interpolate(mat4x3f X, vec2f s) const;
  vec3f d_dxi(mat4x3f X, vec2f s) const;
  vec3f d_deta(mat4x3f X, vec2f s) const;
};

float area(const Triangle & tri);

// computes (4 / sqrt(3)) * (A / (L1 * L2 * L3)^(2/3)) 
float quality(const Triangle & tri);

struct Tetrahedron {
  vec3f vertices[4];
};

// signed volume (negative implies an inverted element)
float volume(const Tetrahedron & tet);

// computes (6 * sqrt(2)) * (V / (L_rms)^(3)) 
float quality(const Tetrahedron & tet);

std::tuple<float, mat4x3f > quality_and_gradient(const Tetrahedron & tet);

float volume(const AABB<3> & box);

inline AABB<3> bounding_box(Ball b) {
  return AABB<3>{
    {b.c[0] - b.r, b.c[1] - b.r, b.c[2] - b.r}, 
    {b.c[0] + b.r, b.c[1] + b.r, b.c[2] + b.r}, 
  };
}

inline AABB<3> bounding_box(vec3f a, vec3f b) {
  return AABB<3>{
    { std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2]) }, 
    { std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2]) }
  };
}

inline AABB<3> bounding_box(Triangle t) {
  return AABB<3>{
    {
      std::min(std::min(t.vertices[0][0], t.vertices[1][0]), t.vertices[2][0]),
      std::min(std::min(t.vertices[0][1], t.vertices[1][1]), t.vertices[2][1]),
      std::min(std::min(t.vertices[0][2], t.vertices[1][2]), t.vertices[2][2])
    }, {
      std::max(std::max(t.vertices[0][0], t.vertices[1][0]), t.vertices[2][0]),
      std::max(std::max(t.vertices[0][1], t.vertices[1][1]), t.vertices[2][1]),
      std::max(std::max(t.vertices[0][2], t.vertices[1][2]), t.vertices[2][2])
    }
  };
}

inline AABB<3> bounding_box(Capsule cap) {
  return bounding_box(bounding_box(Ball{cap.p1, cap.r1}), bounding_box(Ball{cap.p2, cap.r2}));
}

vec3f closest_point_projection(const Line & a, const vec3f & p);

vec3f closest_point_projection(const Triangle & a, const vec3f & p);

/**
 * these indices correspond to a "tetragonal disphenoid honeycomb" of identical
 * tetrahedra. Mathematica code below for visualization:
 *
 *    TetrahedronIds[i_, j_, k_, l_] := (# + {i, j, 2 k - 1}) & /@ {
 *        {{0, 0, 1}, {1, 0, 1}, {1, 0, 0}, {1, 1, 0}},
 *        {{0, 0, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 2}},
 *        {{0, 0, 1}, {1, 0, 1}, {1, 1, 2}, {1, 0, 2}},
 *        {{0, 0, 1}, {1, 0, 1}, {1, 0, 2}, {1, 0, 0}},
 *
 *        {{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 1, 2}},
 *        {{0, 0, 1}, {0, 1, 1}, {0, 1, 2}, {1, 1, 2}},
 *        {{0, 0, 1}, {0, 1, 1}, {1, 1, 2}, {1, 1, 0}},
 *        {{0, 0, 1}, {0, 1, 1}, {1, 1, 0}, {0, 1, 0}},
 *
 *        {{0, 0, 1}, {0, 0, 3}, {0, 0, 2}, {1, 0, 2}},
 *        {{0, 0, 1}, {0, 0, 3}, {1, 0, 2}, {1, 1, 2}},
 *        {{0, 0, 1}, {0, 0, 3}, {1, 1, 2}, {0, 1, 2}},
 *        {{0, 0, 1}, {0, 0, 3}, {0, 1, 2}, {0, 0, 2}}
 *        }[[l]]
 *
 *    n = 4;
 *    \[CapitalDelta]x = 2/n;
 *    offset = {0.0, 0.0, 0.0};
 *    points = Table[
 *       ({i - 1, j - 1, 1/2 (k - 1)} + {1/2 Boole[EvenQ[k]],
 *            1/2 Boole[EvenQ[k]], 0}) \[CapitalDelta]x + offset,
 *       {i, 1, n}, {j, 1, n}, {k, 1, 2 n}
 *       ];
 *    Graphics3D[{PointSize[0.02], Point /@ Flatten[points, 2],
 *      Table[Tetrahedron[Extract[points, TetrahedronIds[i, j, k, l]]], {i,
 *        1, n - 1}, {j, 1, n - 1}, {k, 1, n - 1}, {l, 1, 12}],
 *      Red, Arrow[{{0, 0, 0}, {1, 0, 0}}],
 *      Green, Arrow[{{0, 0, 0}, {0, 1, 0}}],
 *      Blue, Arrow[{{0, 0, 0}, {0, 0, 1}}]
 *    }]
 *
 */
inline std::array<std::array<int, 3>, 4> TetrahedronGridIndices(int x,
                                                                int y,
                                                                int z,
                                                                int w) {
  static constexpr int offsets[12][4][3] = {
      {{0, 0, 1}, {1, 0, 1}, {1, 0, 0}, {1, 1, 0}},
      {{0, 0, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 2}},
      {{0, 0, 1}, {1, 0, 1}, {1, 1, 2}, {1, 0, 2}},
      {{0, 0, 1}, {1, 0, 1}, {1, 0, 2}, {1, 0, 0}},

      {{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 1, 2}},
      {{0, 0, 1}, {0, 1, 1}, {0, 1, 2}, {1, 1, 2}},
      {{0, 0, 1}, {0, 1, 1}, {1, 1, 2}, {1, 1, 0}},
      {{0, 0, 1}, {0, 1, 1}, {1, 1, 0}, {0, 1, 0}},

      {{0, 0, 1}, {0, 0, 3}, {0, 0, 2}, {1, 0, 2}},
      {{0, 0, 1}, {0, 0, 3}, {1, 0, 2}, {1, 1, 2}},
      {{0, 0, 1}, {0, 0, 3}, {1, 1, 2}, {0, 1, 2}},
      {{0, 0, 1}, {0, 0, 3}, {0, 1, 2}, {0, 0, 2}}};

  std::array<std::array<int, 3>, 4> indices;
  for (int i = 0; i < 4; i++) {
    indices[i] = {x + offsets[w][i][0], y + offsets[w][i][1],
                  2 * z + offsets[w][i][2]};
  }
  return indices;
}

void MarchingTriangles(vec3f p[3], float v[3], std::vector<Line>& lines);
void MarchingTetrahedra(vec3f p[4], float v[4], std::vector<Triangle>& tris);
void MarchingTetrahedra(vec3f p[4], float v[4], std::vector<Tetrahedron>& tets);

/**
 * @brief 
 *
 * @param f an implicit function that defines the surface to discretize
 * @param bounds the domain of where to sample the level set function
 * @param n the number of gridpoints along the shortest dimension
 * @param snap_threshold a number between 0.0 and 0.35 used to characterize when
 * a grid point is "close enough" to warrant snapping it to the surface.
 * @return a list of triangles
 */
std::vector<Triangle> GenerateSurfaceMesh(const std::function<float(vec3f)>& f,
                                          AABB<3> bounds = AABB<3>{{-1, -1, -1}, {1, 1, 1}},
                                          int n = 16,
                                          float snap_threshold = 0.25);

std::vector<Tetrahedron> GenerateVolumeMesh(const std::function<float(vec3f)>& f,
                                            AABB<3> bounds = AABB<3>{{-1, -1, -1}, {1, 1, 1}},
                                            int n = 16,
                                            float snap_threshold = 0.25);

static constexpr int T_n[4] = {0, 1, 3, 6};

template < int dim >
struct BackgroundGrid {
  BackgroundGrid(AABB<dim> box, float cell_size) {

    bounds = box;

    uint64_t num_grid_points = 1;

    for (int i = 0; i < dim; i++) {
      float width = bounds.max[i] - bounds.min[i];

      // note: since the background grid in 2D is made up of
      // equilateral triangles, the unit cell isn't square.
      // So, effective cell size in the y-direction is smaller
      if (dim == 2 && i == 1) {
        constexpr float sqrt3_over_2 = 0.8660254037844386f;
        n[i] = round(width / (cell_size * sqrt3_over_2));
      } else {
        n[i] = round(width / cell_size);
      }

      scale[i] = 0.5f * width / n[i];
      offset[i] = bounds.min[i];
      num_grid_points *= (2 * n[i] + 1);
    }

    if (dim == 2) {
      bounds.max[0] += 0.5f * scale[0];
    }

    if (dim == 3) {
      bounds.max += 0.5f * scale;
    }

    values.resize(num_grid_points);
  };

  auto vertex(std::array<int, dim> i) const {
    if constexpr (dim == 2) {
      return vec2f{
        scale[0] * (i[0] + 0.5f * (i[1] % 2 == 1)), 
        scale[1] * i[1] 
      } + offset;
    }
    if constexpr (dim == 3) {
      return vec3f{
        scale[0] * (i[0] + 0.5f * ((i[2] % 2 == 0) * (i[1] % 2 == 0))),
        scale[1] * (i[1] + 0.5f * ((i[2] % 2 == 1) * (i[0] % 2 == 1))),
        scale[2] * (i[2] + 0.5f * ((i[1] % 2 == 1) * (i[0] % 2 == 0)))
      } + offset;
    }
  }

  uint64_t vertex_id(std::array<int, dim> i) const {
    if constexpr (dim == 2) {
      return i[1] * (2 * n[0] + 1) + i[0];
    }
    if constexpr (dim == 3) {
      return (i[2] * (2 * n[1] + 1) + i[1]) * (2 * n[0] + 1) + i[0];
    }
  }

  // note: this value can differ slightly from the value passed in
  // to the constructor, since the grid needs to fit an integer number
  // of cells into the sampling region.
  // 
  // Additionally, since the cell size can differ slightly in each coordinate
  // direction, this function returns the average cell sizes in each direction
  float cell_size() const {
    float value = 0.0f;
    for (int i = 0; i < dim; i++) {
      value += scale[i] * 2;
    }
    return value / dim;
  }

  float & operator()(std::array<int, dim> i) {
    return values[vertex_id(i)];
  }

  const float & operator()(std::array<int, dim> i) const {
    return values[vertex_id(i)];
  }

  vecf<dim> scale;
  vecf<dim> offset;
  AABB<dim> bounds;

  std::array< int, dim > n;
  std::vector< float > values;
};

template <int dim>
using Simplex =
    typename std::conditional<dim == 2, Triangle, Tetrahedron>::type;

template < int dim >
struct SimplexMesh {
  std::vector < vec3f > vertices;
  std::vector < std::array< uint64_t, dim+1 > > elements;
  std::vector < std::array< uint64_t, dim > > boundary_elements;

  std::vector < vec3f > quadratic_nodes;
  std::vector < std::array< uint64_t, T_n[dim] > > elements_quadratic_ids;
  std::vector < std::array< uint64_t, T_n[dim-1] > > boundary_elements_quadratic_ids;
};

std::function<float(vec3f)> SDF(const SimplexMesh<3> & mesh, float dx);

template < int dim >
void dvr(SimplexMesh<dim> & mesh, float alpha, float step, int ndvr, int num_threads);

template < int dim >
void boundary_dvr(SimplexMesh<dim> & mesh, float alpha, float step, int ndvr, int num_threads);

void improve_boundary(
  const std::function< float(vec2f) > & func,
  SimplexMesh<2> & mesh, float alpha, int steps, int num_threads);

void improve_boundary(
  const std::function< float(vec3f) > & func,
  SimplexMesh<3> & mesh, float alpha, int steps, int num_threads);

SimplexMesh<2> universal_mesh(const std::function<float(vec2f)>& sdf,
                              float cell_size,
                              AABB<3> bounds = AABB<3>{{-1, -1, -1}, {1, 1, 1}},
                              float interior_snap_threshold = 0.5f, 
                              float dvr_step = 0.05f,
                              int ndvr = 3,
                              int num_threads = -1);

SimplexMesh<3> universal_mesh(const std::function<float(vec3f)>& sdf,
                              float cell_size,
                              AABB<3> bounds = AABB<3>{{-1, -1, -1}, {1, 1, 1}},
                              float interior_snap_threshold = 0.5f, 
                              float dvr_step = 0.05f,
                              int ndvr = 3,
                              int num_threads = -1);

SimplexMesh<3> universal_boundary_mesh(const std::function<float(vec3f)>& sdf,
                                       float cell_size,
                                       AABB<3> bounds = AABB<3>{{-1, -1, -1}, {1, 1, 1}},
                                       float interior_snap_threshold = 0.5f, 
                                       float dvr_step = 0.05f,
                                       int ndvr = 3,
                                       int num_threads = -1);

SimplexMesh<2> universal_mesh(const BackgroundGrid<2> & grid,
                              float interior_snap_threshold = 0.5f, 
                              float dvr_step = 0.05f,
                              int ndvr = 3,
                              int num_threads = -1);

SimplexMesh<3> universal_mesh(const BackgroundGrid<3> & grid,
                              float interior_snap_threshold = 0.5f, 
                              float dvr_step = 0.05f,
                              int ndvr = 3,
                              int num_threads = -1);

void promote_to_quadratic(SimplexMesh<3> & mesh, const std::function<float(vec3f)>& sdf, float cell_size);

std::vector<int> cell_values(const SimplexMesh<3> & mesh, 
                             const std::vector< Capsule > & capsules, 
                             const std::vector<int> capsule_values,
                             const float cell_size);

std::vector<float> vertex_values(const SimplexMesh<3> & mesh, 
                                 const std::vector< Capsule > & capsules, 
                                 const std::vector< vec2f > capsule_values,
                                 const float cell_size);

#ifdef UM_TIFF_SUPPORT
SimplexMesh<3> universal_mesh(std::vector< std::string > tiff_filenames,
                              float boundary_threshold = 0.5f, 
                              float snapping_distance = 0.5f, 
                              float dvr_step = 0.05f,
                              int ndvr = 3,
                              int num_threads = -1);
#endif

struct HexLattice {
  HexLattice() = default;
  HexLattice(std::vector< std::string > mask);

  void save_to_gmsh(std::string filename);
  static HexLattice load_from_gmsh(std::string filename);

  SimplexMesh<3> fluid_mesh(const std::vector<float> & radii, float cell_size);
  SimplexMesh<3> capsule_mesh(const std::vector<float> & radii, float cell_size);

  AABB<3> bounds;
  std::vector< vec3f > vertices;
  std::vector< uint32_t > boundary_faces;
  std::vector< std::array< uint32_t, 2 > > edges;
  std::vector< std::array< uint32_t, 4 > > faces;
  std::vector< std::array< uint32_t, 8 > > hexes;
};

template < typename T >
std::vector<T> combine(const std::vector < T > containers[], int n) {
  size_t total_size = 0;
  for (int i = 0; i < n; i++) {
    total_size += containers[i].size();
  }

  std::vector<T> output;
  output.reserve(total_size);
  for (int i = 0; i < n; i++) {
    output.insert(output.end(), containers[i].begin(), containers[i].end());
  }
  return output;
}

template < typename T >
std::vector<T> combine(const std::vector< std::vector < T > > & containers) {
  uint32_t n = containers.size();
  size_t total_size = 0;
  for (uint32_t i = 0; i < n; i++) {
    total_size += containers[i].size();
  }

  std::vector<T> output;
  output.reserve(total_size);
  for (uint32_t i = 0; i < n; i++) {
    output.insert(output.end(), containers[i].begin(), containers[i].end());
  }
  return output;
}

std::string file_extension(std::string filename);

// note: this only operates on the boundary of the mesh
void laplace_smoothing(SimplexMesh<3> & mesh, float lambda);

SimplexMesh<3> import_stl(std::string filename);
SimplexMesh<3> import_gmsh22(std::string filename);

bool export_stl(const SimplexMesh<2> & mesh, std::string filename);
bool export_stl(const SimplexMesh<3> & mesh, std::string filename);
bool export_vtk(const SimplexMesh<2> & mesh, std::string filename);
bool export_vtk(const SimplexMesh<3> & mesh, std::string filename);
bool export_vtu(const SimplexMesh<2> & mesh, std::string filename);
bool export_vtu(const SimplexMesh<3> & mesh, std::string filename, const std::vector<int32_t> & attr = {});
bool export_gmsh(const SimplexMesh<2> & mesh, std::string filename);
bool export_gmsh(const SimplexMesh<3> & mesh, std::string filename);

// detects which format, based on file extension
bool export_mesh(const SimplexMesh<2> & mesh, std::string filename);
bool export_mesh(const SimplexMesh<3> & mesh, std::string filename);

bool export_mesh_boundary(const SimplexMesh<3> & mesh, std::string filename);

void quality_summary(const SimplexMesh<3> & mesh);

bool verify(const SimplexMesh<2> & mesh);
bool verify(const SimplexMesh<3> & mesh);

// TODO delete
template < int dim >
void snap_vertices_to_boundary(SimplexMesh<dim> & mesh, std::vector<float> & values, const std::vector< uint64_t> & partially_inside, float max_snap_distance, int num_threads, bool boundary_only);

}