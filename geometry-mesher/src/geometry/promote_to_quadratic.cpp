#include <array>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include "geometry.hpp"

namespace geometry {

template < std::size_t n >
static auto small_sort(const std::array< uint64_t, n > & values) { 
  auto copy = values;
  std::sort(copy.begin(), copy.end()); 
  return copy;
}

struct array_hasher {
  template< std::size_t n>
  std::size_t operator()(const std::array< uint64_t, n > & arr) const {
    auto sorted = small_sort(arr);
    uint64_t seed = 0;
    for(const auto elem : sorted) {
      seed ^= std::hash<uint64_t>()(elem) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
    return seed;
  }
};

struct array_equality {
  template< std::size_t n>
  bool operator()(const std::array< uint64_t, n > & a, 
                  const std::array< uint64_t, n > & b) const {
    return small_sort(a) == small_sort(b);
  }
};

template < std::size_t n >
using unordered_map_of_arrays = std::unordered_map< std::array< uint64_t, n>, 
                                                    uint64_t, 
                                                    array_hasher,
                                                    array_equality >;
template < size_t n >
using unordered_set_of_arrays = std::unordered_set< std::array< uint64_t, n>, 
                                                    array_hasher, 
                                                    array_equality >;

#if 0
static vec3f compute_unit_normal(const std::vector< vec3f > & v, const std::array< uint64_t, 2 > & edge) {
  return normalize(cross(v[edge[1]] - v[edge[0]], vec3f{0, 0, 1}));
}
#endif

static vec3f compute_unit_normal(const std::vector< vec3f > & v, const std::array< uint64_t, 3 > & tri) {
  return normalize(cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]]));
}

void promote_to_quadratic(SimplexMesh<2> & mesh, const std::function<float(vec2f)>& f, float cell_size) {

}

void promote_to_quadratic(SimplexMesh<3> & mesh, const std::function<float(vec3f)>& f, float cell_size) {

  static constexpr uint64_t tri_edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
  static constexpr uint64_t tet_edges[6][2] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {2, 3}, {1, 3}}; // gmsh numbering

  mesh.elements_quadratic_ids.resize(mesh.elements.size());
  mesh.boundary_elements_quadratic_ids.resize(mesh.boundary_elements.size());

  uint64_t edge_id = 0;
  unordered_map_of_arrays<2> edges;
  for (int i = 0; i < mesh.elements.size(); i++) {
    auto tet = mesh.elements[i];
    for (int j = 0; j < 6; j++) {
      auto e = tet_edges[j];
      std::array<uint64_t, 2> edge{uint64_t(tet[e[0]]), uint64_t(tet[e[1]])};
      if (edges.count(edge) == 0) {
        edges[edge] = edge_id++;
        mesh.quadratic_nodes.push_back(0.5f * (mesh.vertices[edge[0]] + mesh.vertices[edge[1]]));
      }

      mesh.elements_quadratic_ids[i][j] = edges[edge];
    }
  }

  std::unordered_set<uint64_t> boundary_vertex_set;
  std::unordered_set<uint64_t> boundary_edge_node_set;
  std::vector< vec3f > edge_normals(edges.size(), vec3f{});
  std::vector< vec3f > vertex_normals(mesh.vertices.size(), vec3f{});
  for (int i = 0; i < mesh.boundary_elements.size(); i++) {
    auto tri = mesh.boundary_elements[i];
    vec3f n = compute_unit_normal(mesh.vertices, tri);
    for (int j = 0; j < 3; j++) {
      auto e = tri_edges[j];
      uint64_t edge_id = edges[{uint64_t(tri[e[0]]), uint64_t(tri[e[1]])}];
      mesh.boundary_elements_quadratic_ids[i][j] = edge_id;
      boundary_edge_node_set.insert(edge_id);

      edge_normals[edge_id] += n;
      vertex_normals[tri[j]] += n;
      boundary_vertex_set.insert(tri[j]);
    }
  }

  std::vector< uint64_t > boundary_vertex_ids(boundary_vertex_set.begin(), 
                                              boundary_vertex_set.end());

  std::vector< uint64_t > boundary_edge_node_ids(boundary_edge_node_set.begin(), 
                                                 boundary_edge_node_set.end());


  for (auto i : boundary_vertex_ids) {
    vec3f x = mesh.vertices[i];
    vec3f n = normalize(vertex_normals[i]);
    float epsilon = 0.01f * cell_size;
    for (int k = 0; k < 4; k++) {
      float r[2] = {f(x), f(x + epsilon * n)};
      float dr_dx = (r[1] - r[0]) / epsilon;
      if (fabs(r[0] / dr_dx) < 0.5f * cell_size) {
        x -= (r[0] / dr_dx) * n;
      } else {
        epsilon *= 0.25;
      }
    }
    mesh.vertices[i] = x;
  }

  for (auto i : boundary_edge_node_ids) {
    vec3f x = mesh.quadratic_nodes[i];
    vec3f n = normalize(edge_normals[i]);
    float epsilon = 0.01f * cell_size;
    for (int k = 0; k < 4; k++) {
      float r[2] = {f(x), f(x + epsilon * n)};
      float dr_dx = (r[1] - r[0]) / epsilon;
      if (fabs(r[0] / dr_dx) < 0.5f * cell_size) {
        x -= (r[0] / dr_dx) * n;
      } else {
        epsilon *= 0.25;
      }
    }
    mesh.quadratic_nodes[i] = x;
  }

}

}