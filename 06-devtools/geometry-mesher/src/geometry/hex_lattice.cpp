#include "geometry.hpp"

#include "BVH.hpp"
#include "mesh/io.hpp"

#include "fm/operations/dot.hpp"
#include "fm/operations/inverse.hpp"

#include <array>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>

using namespace fm;

namespace geometry {

using u32 = uint32_t;
using vec8f = vec<8,float>;
using mat8x3f = mat<8,3,float>;

constexpr u32 dx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
constexpr u32 dy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
constexpr u32 dz[8] = {0, 0, 0, 0, 1, 1, 1, 1};

constexpr u32 local_edge_ids[12][2] = {{0, 1},{1, 2},{3, 2},{0, 3},{0, 4},{1, 5},{2, 6},{3, 7},{4, 5},{5, 6},{7, 6},{4, 7}};
constexpr u32 local_face_ids[6][4] = {{1, 0, 3, 2},{0, 1, 5, 4},{1, 2, 6, 5},{2, 3, 7, 6},{3, 0, 4, 7},{4, 5, 6, 7}};

bool point_inside_hex(mat8x3f X, vec3f p) {

  auto residual = [&](vec3f s) {
    vec3f t = vec3f{1.0, 1.0, 1.0} - s;
    return t[0]*t[1]*t[2]*X[0] + 
           s[0]*t[1]*t[2]*X[1] + 
           s[0]*s[1]*t[2]*X[2] + 
           t[0]*s[1]*t[2]*X[3] + 
           t[0]*t[1]*s[2]*X[4] + 
           s[0]*t[1]*s[2]*X[5] + 
           s[0]*s[1]*s[2]*X[6] + 
           t[0]*s[1]*s[2]*X[7] - p;
  };

  auto jacobian = [&](vec3f s) {
    vec3f t = vec3f{1.0, 1.0, 1.0} - s;
    return mat3f{
        -t[1]*t[2]*X[0] + t[1]*t[2]*X[1] + s[1]*t[2]*X[2] - s[1]*t[2]*X[3] - t[1]*s[2]*X[4] + t[1]*s[2]*X[5] + s[1]*s[2]*X[6] - s[1]*s[2]*X[7],
        -t[0]*t[2]*X[0] - s[0]*t[2]*X[1] + s[0]*t[2]*X[2] + t[0]*t[2]*X[3] - t[0]*s[2]*X[4] - s[0]*s[2]*X[5] + s[0]*s[2]*X[6] + t[0]*s[2]*X[7],
        -t[0]*t[1]*X[0] - s[0]*t[1]*X[1] - s[0]*s[1]*X[2] - t[0]*s[1]*X[3] + t[0]*t[1]*X[4] + s[0]*t[1]*X[5] + s[0]*s[1]*X[6] + t[0]*s[1]*X[7]
    };
  };

  vec3f xi = {0.5f, 0.5f, 0.5f};
  vec3f r = residual(xi);
  for (int k = 0; k < 6; k++) {
    if (dot(r,r) < 1.0e-10f) {
      return true;
    } else {
      mat3f JT = jacobian(xi);
      xi = clamp(xi - dot(r, inv(JT)), 0.0f, 1.0f);
      r = residual(xi);
    }
  }
  return dot(r,r) < 1.0e-10f;

}

template < size_t n >
auto sort(const std::array< u32, n > & values) { 
  auto copy = values;
  std::sort(copy.begin(), copy.end()); 
  return copy;
}

struct array_hasher {
  template<size_t n>
  std::size_t operator()(const std::array< u32, n > & arr) const {
    u32 seed = 0;
    for(const auto elem : arr) {
      seed ^= std::hash<u32>()(elem) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
    return seed;
  }
};

template < size_t n >
using key = std::array< u32, n >;

template < size_t n, typename T >
using array_map = std::unordered_map< key<n>, T, array_hasher >;

struct EdgeInfo {
    std::array< u32, 2 > vertex_ids;
    u32 id;
};

struct FaceInfo {
    std::array< u32, 4 > vertex_ids;
    u32 id;
    bool boundary;
};

void edge_and_face_connectivity(std::vector< key<2> > & edges, 
                                std::vector< key<4> > & faces,
                                std::vector< u32 > & boundary_faces,
                                const std::vector< key<8> > & hexes) {

    array_map< 2, EdgeInfo > edge_LUT;
    array_map< 4, FaceInfo > face_LUT;

    for (auto hex : hexes) {
        for (auto local_edge : local_edge_ids) {
            key<2> edge = {hex[local_edge[0]], hex[local_edge[1]]};
            key<2> sorted_edge = sort(edge);

            if (edge_LUT.count(sorted_edge) == 0) {
                u32 edge_id = edge_LUT.size();
                edge_LUT[sorted_edge] = EdgeInfo{edge, edge_id};
            } 
        }

        for (auto local_face : local_face_ids) {
            key<4> face = {hex[local_face[0]], hex[local_face[1]], hex[local_face[2]], hex[local_face[3]]};
            key<4> sorted_face = sort(face);

            if (face_LUT.count(sorted_face) == 0) {
                u32 face_id = face_LUT.size();
                face_LUT[sorted_face] = FaceInfo{face, face_id, true};
            } else {
                face_LUT[sorted_face].boundary = false;
            }
        }
    }

    edges.resize(edge_LUT.size());
    for (auto [_, edge] : edge_LUT) {
        edges[edge.id] = edge.vertex_ids;
    }

    boundary_faces = std::vector<u32>(0);
    faces.resize(face_LUT.size());
    for (auto [_, face] : face_LUT) {
        faces[face.id] = face.vertex_ids;
        if (face.boundary) {
            boundary_faces.push_back(face.id);
        }
    }

}

////////////////////////////////////////////////////////////////////////////////

HexLattice::HexLattice(std::vector< std::string > mask) {

    u32 rows = mask.size();
    if (rows == 0) return;

    // first, verify that all the strings are the same length
    u32 columns = mask[0].size();
    for (u32 i = 1; i < mask.size(); i++) {
        if (columns != mask[i].size()) {
            std::cout << "error: all strings must be the same length in HexLattice(std::vector<std::string>)" << std::endl;
        }
    }

    bounds.min = { 1.0e30f,  1.0e30f,  1.0e30f};
    bounds.max = {-1.0e30f, -1.0e30f, -1.0e30f};

    // use a std::unordered map to figure out if a vertex has been seen before
    array_map< 3, u32 > vertex_id_LUT;
    vertices = std::vector< vec3f >();
    auto insert_vertex = [&](const key<3> & k) {
        if (vertex_id_LUT.count(k) == 0) {
            u32 id = vertex_id_LUT.size();
            vec3f v = vec3f{float(k[0]), float(k[1]), float(k[2])};
            vertices.push_back(v);
            for (u32 i = 0; i < 3; i++) {
                bounds.min[i] = std::min(bounds.min[i], v[i]);
                bounds.max[i] = std::max(bounds.max[i], v[i]);
            }
            vertex_id_LUT[k] = id;
            return id;
        } else {
            return vertex_id_LUT[k];
        }
    };

    hexes = std::vector< key<8> >();
    u32 z = 0; // in xy-plane
    for (u32 x = 0; x < rows; x++) {
        for (u32 y = 0; y < columns; y++) {
            if (mask[x][y] == 'X') {
                key<8> hex;
                for (u32 i = 0; i < 8; i++) {
                    hex[i] = insert_vertex({x + dx[i], y + dy[i], z + dz[i]});
                }
                hexes.push_back(hex);
            } else {
                if (mask[x][y] != ' ') {
                    std::cout << "error: expected only ' ' or 'X' characters in this HexLattice ctor" << std::endl;
                    return;
                }
            }
        }
    }

    edge_and_face_connectivity(edges, faces, boundary_faces, hexes);

}

SimplexMesh<3> HexLattice::capsule_mesh(const std::vector<float> & radii, float cell_size) {

  const auto & v = vertices;
  std::vector < Capsule > capsules;

  std::cout << edges.size() << std::endl;

  for (auto [i, j] : edges) {
    capsules.push_back({v[i], v[j], radii[i], radii[j]});
  }

  std::vector< AABB<3> > bounding_boxes(capsules.size());
  for (uint32_t i = 0; i < capsules.size(); i++) {
    bounding_boxes[i] = bounding_box(capsules[i]);
  }
  BVH<3> bvh(bounding_boxes);

  vec3f widths = bvh.global.max - bvh.global.min;

  auto sampling_bounds = bvh.global;

  sampling_bounds.max += 0.15f * widths;
  sampling_bounds.min -= 0.15f * widths;

  float blend_distance = 0.0; // currently unused for "hard" min
  float dx = 1.5 * cell_size + 2 * blend_distance;

  std::function<float(vec3f)> f = [&](vec3f x) -> float {
    AABB<3>box{
      {x[0] - dx, x[1] - dx, x[2] - dx}, 
      {x[0] + dx, x[1] + dx, x[2] + dx}
    };

    float value = 2 * dx;
    bvh.query(box, [&](int i) {
      value = std::min(value, capsules[i].SDF(x));
    });
    return value;
  };

  return universal_mesh(f, cell_size, sampling_bounds);

}

SimplexMesh<3> HexLattice::fluid_mesh(const std::vector<float> & radii, float cell_size) {

  const auto & v = vertices;
  std::vector < Capsule > capsules;
  float r_min = radii[0];
  for (auto r : radii) {
    r_min = std::min(r, r_min);
  }

  std::cout << edges.size() << std::endl;

  for (auto [i, j] : edges) {
    capsules.push_back({v[i], v[j], radii[i], radii[j]});
  }

  std::vector< AABB<3> > capsule_bounding_boxes(capsules.size());
  for (uint32_t i = 0; i < capsules.size(); i++) {
    capsule_bounding_boxes[i] = bounding_box(capsules[i]);
  }
  BVH<3> capsule_bvh(capsule_bounding_boxes);

  std::vector< AABB<3> > hex_bounding_boxes(hexes.size());
  for (uint32_t i = 0; i < hexes.size(); i++) {
    auto hex = hexes[i];
    vec3f min = v[hex[0]];
    vec3f max = v[hex[0]];
    for (uint32_t j = 1; j < 8; j++) {
      vec3f vj = v[hex[j]];
      min[0] = std::min(min[0], vj[0]);
      min[1] = std::min(min[1], vj[1]);
      min[2] = std::min(min[2], vj[2]);

      max[0] = std::max(max[0], vj[0]);
      max[1] = std::max(max[1], vj[1]);
      max[2] = std::max(max[2], vj[2]);
    }
    hex_bounding_boxes[i] = AABB<3>{min, max};
  };
  BVH<3> hex_bvh(hex_bounding_boxes);

  std::vector< Quad > quads;
  quads.reserve(boundary_faces.size());
  std::vector< AABB<3> > quad_bounding_boxes(boundary_faces.size());
  for (uint32_t i = 0; i < boundary_faces.size(); i++) {
    auto quad = faces[boundary_faces[i]];
    vec3f min = v[quad[0]];
    vec3f max = v[quad[0]];
    for (uint32_t j = 1; j < 4; j++) {
      vec3f vj = v[quad[j]];
      min[0] = std::min(min[0], vj[0]);
      min[1] = std::min(min[1], vj[1]);
      min[2] = std::min(min[2], vj[2]);

      max[0] = std::max(max[0], vj[0]);
      max[1] = std::max(max[1], vj[1]);
      max[2] = std::max(max[2], vj[2]);
    }
    quads.push_back(Quad{{v[quad[0]], v[quad[1]], v[quad[2]], v[quad[3]]}});
    quad_bounding_boxes[i] = AABB<3>{min, max};
  };
  BVH<3> quad_bvh(quad_bounding_boxes);

  vec3f widths = capsule_bvh.global.max - capsule_bvh.global.min;

  auto sampling_bounds = capsule_bvh.global;

  sampling_bounds.max += 0.15f * widths;
  sampling_bounds.min -= 0.15f * widths;

  float dx = 1.5 * cell_size;

  std::function<float(vec3f)> f = [&](vec3f x) -> float {
    AABB<3>box{
      {x[0] - dx, x[1] - dx, x[2] - dx}, 
      {x[0] + dx, x[1] + dx, x[2] + dx}
    };

    float capsule_distance = 2 * dx;
    capsule_bvh.query(box, [&](int i) {
      capsule_distance = std::min(capsule_distance, capsules[i].SDF(x));
    });

    bool inside_a_hex = false;
    hex_bvh.query(box, [&](int i) {
      if (inside_a_hex) { return; }
      auto hex = hexes[i];
      mat8x3f hex_vertices = {v[hex[0]], v[hex[1]], v[hex[2]], v[hex[3]], v[hex[4]], v[hex[5]], v[hex[6]], v[hex[7]]};
      inside_a_hex = point_inside_hex(hex_vertices, x);
    });

    float quad_distance = 2 * dx;
    quad_bvh.query(box, [&](int i) {
      quad_distance = std::min(quads[i].SDF(x), quad_distance);
    });

    if (inside_a_hex) { quad_distance *= -1; }

    // this std::max implements the boolean difference 
    // hex mesh - edge capsules
    return std::max(quad_distance, -capsule_distance);
  };

  return universal_mesh(f, cell_size, sampling_bounds);

}

void HexLattice::save_to_gmsh(std::string filename) { 
  io::Mesh mesh;

  mesh.nodes.reserve(vertices.size());
  for (uint32_t i = 0; i < vertices.size(); i++) {
      mesh.nodes.push_back({vertices[i][0], vertices[i][1], vertices[i][2]});
  }

  mesh.elements.resize(hexes.size());
  for (uint32_t i = 0; i < mesh.elements.size(); i++) {
      mesh.elements[i].type = io::Element::Type::Hex8;
      mesh.elements[i].node_ids.resize(8);
      for (uint32_t j = 0; j < 8; j++) {
          mesh.elements[i].node_ids[j] = hexes[i][j];
      }
  }

  io::export_gmsh_v22(mesh, filename, io::FileEncoding::Binary);
}

HexLattice HexLattice::load_from_gmsh(std::string filename) { 

  HexLattice lattice;

  io::Mesh mesh = io::import_gmsh_v22(filename);

  lattice.bounds.min = { 1.0e30f,  1.0e30f,  1.0e30f};
  lattice.bounds.max = {-1.0e30f, -1.0e30f, -1.0e30f};

  lattice.vertices.resize(mesh.nodes.size());
  for (uint32_t i = 0; i < lattice.vertices.size(); i++) {
    auto node = mesh.nodes[i];
    lattice.vertices[i] = vec3f{float(node[0]), float(node[1]), float(node[2])};

    for (u32 j = 0; j < 3; j++) {
      lattice.bounds.min[j] = std::min(lattice.bounds.min[j], float(node[j]));
      lattice.bounds.max[j] = std::max(lattice.bounds.max[j], float(node[j]));
    }
  }

  lattice.hexes.resize(mesh.elements.size());
  for (uint32_t i = 0; i < mesh.elements.size(); i++) {
    if (mesh.elements[i].type != io::Element::Type::Hex8) {
      std::cout << "error: trying to import gmsh file with incompatible element types" << std::endl;
      exit(1);
    }
    for (uint32_t j = 0; j < 8; j++) {
        lattice.hexes[i][j] = mesh.elements[i].node_ids[j];
    }
  }

  edge_and_face_connectivity(lattice.edges, lattice.faces, lattice.boundary_faces, lattice.hexes);

  return lattice;

}

}