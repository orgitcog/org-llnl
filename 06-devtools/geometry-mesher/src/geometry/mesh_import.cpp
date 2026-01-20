#include "mesh/io.hpp"
#include "BVH.hpp"
#include "geometry.hpp"

#include <iostream>
#include <filesystem>

namespace geometry {

struct array_hasher {
  template<size_t n>
  std::size_t operator()(const std::array< uint64_t, n > & arr) const {
    uint64_t seed = 0;
    for(const auto elem : arr) {
      seed ^= std::hash<uint64_t>()(elem) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
    return seed;
  }
};

template < size_t n >
auto sort(const std::array< uint64_t, n > & values) { 
  auto copy = values;
  std::sort(copy.begin(), copy.end()); 
  return copy;
}

template < size_t n >
using unordered_map_of_arrays = std::unordered_map< std::array< uint64_t, n >, std::array<uint64_t , n>, array_hasher >;

SimplexMesh<3> convert(io::Mesh mesh) {

    SimplexMesh<3> output;

    output.vertices.resize(mesh.nodes.size());
    for (uint32_t i = 0; i < mesh.nodes.size(); i++) {
        for (uint32_t j = 0; j < 3; j++) {
            output.vertices[i][j] = mesh.nodes[i][j];
        }
    }

    static constexpr int local_triangle_ids[4][3] = {{2, 1, 0}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
    unordered_map_of_arrays<3> boundary_triangles{};

    output.elements.resize(mesh.elements.size());
    for (uint32_t i = 0; i < mesh.elements.size(); i++) {
        auto & e = output.elements[i];

        if (mesh.elements[i].type != io::Element::Type::Tet4) {
            std::cout << "import error: only Tet4 elements are currently supported" << std::endl;
        }
        for (uint32_t j = 0; j < 4; j++) {
            e[j] = mesh.elements[i].node_ids[j];
        }

        for (auto [v1,v2,v3] : local_triangle_ids) {
            std::array<uint64_t, 3> tri{e[v1], e[v2], e[v3]};
            auto sorted_tri = sort(tri);
            if (boundary_triangles.count(sorted_tri) == 1) {
                boundary_triangles.erase(sorted_tri);
            } else {
                boundary_triangles[sorted_tri] = tri;
            }
        }

    }

    output.boundary_elements.resize(boundary_triangles.size());
    uint32_t count = 0;
    for (auto & [k, v] : boundary_triangles) {
        output.boundary_elements[count++] = v;
    }

    return output;
}

SimplexMesh<3> convert_STL(io::Mesh mesh) {

    SimplexMesh<3> output;

    float r = 1.0e-4;

    std::cout << "original vertices: " << mesh.nodes.size() << std::endl;

    std::vector<AABB<3>> vertex_boxes(mesh.nodes.size());
    for (uint32_t i = 0; i < mesh.nodes.size(); i++) {
        vertex_boxes[i] = {
            {float(mesh.nodes[i][0] - r), float(mesh.nodes[i][1] - r), float(mesh.nodes[i][2] - r)},
            {float(mesh.nodes[i][0] + r), float(mesh.nodes[i][1] + r), float(mesh.nodes[i][2] + r)}
        };
    }

    BVH<3> bvh(vertex_boxes);

    // deduplicate vertices
    std::vector< uint32_t > new_ids(mesh.nodes.size());
    for (int i = 0; i < mesh.nodes.size(); i++) {
        vec3f p = {float(mesh.nodes[i][0]), float(mesh.nodes[i][1]), float(mesh.nodes[i][2])};

        AABB<3> vbox = {{p[0] - r, p[1] - r, p[2] - r}, {p[0] + r, p[1] + r, p[2] + r}};

        int new_id = i;
        bvh.query(vbox, [&](int i) { new_id = std::min(new_id, i); });

        // if this vertex is not coincident with any other vertex
        if (new_id == i) { 
            new_ids[i] = output.vertices.size();
            output.vertices.push_back(0.5f * (vbox.min + vbox.max)); 
        } else {
            new_ids[i] = new_ids[new_id];
        }
    }

    std::cout << "deduplicated vertices: " << output.vertices.size() << std::endl;

    output.boundary_elements.resize(mesh.elements.size());
    for (uint32_t i = 0; i < mesh.elements.size(); i++) {
        auto & e = output.elements[i];

        if (mesh.elements[i].type != io::Element::Type::Tri3) {
            std::cout << "import STL error: expected only Tri3 elements" << std::endl;
        }

        output.boundary_elements[i][0] = new_ids[mesh.elements[i].node_ids[0]]; 
        output.boundary_elements[i][1] = new_ids[mesh.elements[i].node_ids[1]]; 
        output.boundary_elements[i][2] = new_ids[mesh.elements[i].node_ids[2]]; 
    }

    return output;
}

SimplexMesh<3> import_gmsh22(std::string filename) {
    return convert(io::import_gmsh_v22(filename));
}

SimplexMesh<3> import_stl(std::string filename) {
    return convert_STL(io::import_stl(filename));
}

}