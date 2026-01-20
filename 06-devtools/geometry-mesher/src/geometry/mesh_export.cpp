#include "mesh/io.hpp"
#include "geometry.hpp"

#include <iostream>
#include <filesystem>

namespace geometry {

io::Mesh convert(const SimplexMesh<2> & mesh) {
    io::Mesh output;

    output.nodes.reserve(mesh.vertices.size());
    for (uint32_t i = 0; i < mesh.vertices.size(); i++) {
        output.nodes.push_back({mesh.vertices[i][0], mesh.vertices[i][1], 0.0});
    }

    output.elements.resize(mesh.elements.size());
    for (uint32_t i = 0; i < mesh.elements.size(); i++) {
        output.elements[i].type = io::Element::Type::Tri3;
        output.elements[i].node_ids.resize(3);
        for (uint32_t j = 0; j < 3; j++) {
            output.elements[i].node_ids[j] = mesh.elements[i][j];
        }
    }
    
    return output;
}

io::Mesh convert(const SimplexMesh<3> & mesh, const std::vector<int32_t> & attr = {}) {
    io::Mesh output;

    bool has_attributes = attr.size() > 0;
    bool mesh_is_quadratic = mesh.quadratic_nodes.size() > 0;

    output.nodes.reserve(mesh.vertices.size());
    for (uint32_t i = 0; i < mesh.vertices.size(); i++) {
        output.nodes.push_back({mesh.vertices[i][0], mesh.vertices[i][1], mesh.vertices[i][2]});
    }
    if (mesh_is_quadratic) {
        for (uint32_t i = 0; i < mesh.quadratic_nodes.size(); i++) {
            output.nodes.push_back({mesh.quadratic_nodes[i][0], mesh.quadratic_nodes[i][1], mesh.quadratic_nodes[i][2]});
        }
    }

    output.elements.resize(mesh.elements.size());

    if (mesh_is_quadratic) {
        uint64_t num_vertices = mesh.vertices.size();
        for (uint32_t i = 0; i < mesh.elements.size(); i++) {
            output.elements[i].type = io::Element::Type::Tet10;
            output.elements[i].node_ids.reserve(10);
            for (uint32_t j = 0; j < 4; j++) {
                output.elements[i].node_ids.push_back(mesh.elements[i][j]);
            }
            for (uint32_t j = 0; j < 6; j++) {
                output.elements[i].node_ids.push_back(mesh.elements_quadratic_ids[i][j] + num_vertices);
            }
        }
    } else {
        for (uint32_t i = 0; i < mesh.elements.size(); i++) {
            output.elements[i].type = io::Element::Type::Tet4;
            output.elements[i].node_ids.resize(4);
            for (uint32_t j = 0; j < 4; j++) {
                output.elements[i].node_ids[j] = mesh.elements[i][j];
            }
        }
    }

    if (has_attributes) {
        if (attr.size() != output.elements.size()) {
            std::cout << "error: attribute array size is not equal to number of elements in mesh" << std::endl;
            exit(1);
        }
        for (uint32_t i = 0; i < mesh.elements.size(); i++) {
            output.elements[i].tags.push_back(attr[i]);
        }
    }
    return output;
}

io::Mesh convert_boundary(const SimplexMesh<3> & mesh) {
    io::Mesh output;

    output.nodes.resize(mesh.vertices.size());
    for (uint32_t i = 0; i < mesh.vertices.size(); i++) {
        for (uint32_t j = 0; j < 3; j++) {
            output.nodes[i][j] = mesh.vertices[i][j];
        }
    }

    output.elements.resize(mesh.boundary_elements.size());
    for (uint32_t i = 0; i < mesh.boundary_elements.size(); i++) {
        output.elements[i].type = io::Element::Type::Tri3;
        output.elements[i].node_ids.resize(3);
        for (uint32_t j = 0; j < 3; j++) {
            output.elements[i].node_ids[j] = mesh.boundary_elements[i][j];
        }
    }

    return output;
}

bool export_stl(const SimplexMesh<2> & mesh, std::string filename) {
    return io::export_stl(convert(mesh), filename);
}

bool export_stl(const SimplexMesh<3> & mesh, std::string filename) {
    return io::export_stl(convert_boundary(mesh), filename);
}

bool export_vtk(const SimplexMesh<2> & mesh, std::string filename) {
    return io::export_vtk(convert(mesh), filename, io::FileEncoding::Binary);
}

bool export_vtk(const SimplexMesh<3> & mesh, std::string filename) {
    return io::export_vtk(convert(mesh), filename, io::FileEncoding::Binary);
}

bool export_vtu(const SimplexMesh<2> & mesh, std::string filename) {
    return io::export_vtu(convert(mesh), filename);
}

bool export_vtu(const SimplexMesh<3> & mesh, std::string filename, const std::vector<int32_t> & attr) {
    return io::export_vtu(convert(mesh, attr), filename);
}

bool export_gmsh(const SimplexMesh<2> & mesh, std::string filename) {
    return io::export_gmsh_v22(convert(mesh), filename, io::FileEncoding::Binary);
}

bool export_gmsh(const SimplexMesh<3> & mesh, std::string filename) {
    return io::export_gmsh_v22(convert(mesh), filename, io::FileEncoding::Binary);
}

// returns file extension in lower case characters
std::string file_extension(std::string filename) {
  std::string ext = std::filesystem::path(filename).extension().string();
  for(auto& c : ext) { c = tolower(c); }
  return ext;
}

bool export_mesh(const SimplexMesh<2> & mesh, std::string filename) {
  std::string ext = file_extension(filename);
  if (ext == ".stl") {
    return export_stl(mesh, filename);
  } else if (ext == ".vtk") {
    return export_vtk(mesh, filename);
  } else if (ext == ".vtu") {
    return export_vtu(mesh, filename);
  } else if (ext == ".msh") {
    return export_gmsh(mesh, filename);
  } else {
    std::cout << "unrecognized extension: " << ext << std::endl;
    exit(1);
  }
}

bool export_mesh(const SimplexMesh<3> & mesh, std::string filename) {
  std::string ext = file_extension(filename);
  if (ext == ".stl") {
    return export_stl(mesh, filename);
  } else if (ext == ".vtk") {
    return export_vtk(mesh, filename);
  } else if (ext == ".vtu") {
    return export_vtu(mesh, filename);
  } else if (ext == ".msh") {
    return export_gmsh(mesh, filename);
  } else {
    std::cout << "unrecognized extension: " << ext << std::endl;
    exit(1);
  }
}

}