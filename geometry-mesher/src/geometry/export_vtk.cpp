#include "geometry.hpp"

#include <fstream>

// taken from mfem mesh/vtk.hpp
bool systemIsBigEndian() {
  int64_t x16 = 1;
  int32_t *x8 = reinterpret_cast<int32_t*>(&x16);
  return !*x8;
}

const bool bigEndian = systemIsBigEndian();

template <typename T>
T makeBigEndian(T value) {
  if (!bigEndian) {
    auto it = reinterpret_cast<uint8_t*>(&value);
    std::reverse(it, it + sizeof(T)); // value is now in swapped from little to big endianness
  }
  return value;
}

template <typename T>
void write_binary(std::ofstream &outfile, T value) {
  T be_value = makeBigEndian(value);
  const char *converted_be_value = reinterpret_cast<const char*>(&be_value);
  outfile.write(converted_be_value, sizeof(T));
}

static constexpr uint32_t VTK_TRIANGLE = 5;
static constexpr uint32_t VTK_TETRAHEDRON = 10;
static constexpr uint32_t VTK_QUADRATIC_TRIANGLE = 22;
static constexpr uint32_t VTK_QUADRATIC_TETRAHEDRON = 24;

template < int dim >
void export_vtk_impl(const SimplexMesh<dim> & mesh, std::string filename, VTKFileType filetype) {

  bool is_quadratic = mesh.quadratic_nodes.size() > 0;

  std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);

  outfile << "# vtk DataFile Version 3.0\n";
  outfile << "--------------------------\n";
  std::string outputType = filetype == VTKFileType::ASCII ? "ASCII\n" : "BINARY\n";
  outfile << outputType;
  outfile << "DATASET UNSTRUCTURED_GRID\n";

  if (is_quadratic) {
    outfile << "POINTS " << mesh.vertices.size() + mesh.quadratic_nodes.size() << " float\n";
  }
  else {
    outfile << "POINTS " << mesh.vertices.size() << " float\n";
  }
  if (filetype == VTKFileType::ASCII) {
    for (auto p : mesh.vertices) {
      outfile << p[0] << " " << p[1] << " " << p[2] << '\n';
    }
    if (is_quadratic) {
      for (auto p : mesh.quadratic_nodes) {
        outfile << p[0] << " " << p[1] << " " << p[2] << '\n';
      }
    }
  }
  else { // binary
    for (auto p : mesh.vertices) {
      write_binary<float>(outfile, p[0]);
      write_binary<float>(outfile, p[1]);
      write_binary<float>(outfile, p[2]);
    }
    if (is_quadratic) {
      for (auto p : mesh.quadratic_nodes) {
        write_binary<float>(outfile, p[0]);
        write_binary<float>(outfile, p[1]);
        write_binary<float>(outfile, p[2]);
      }
    }
    outfile << '\n';
  }

  int32_t nelems = mesh.elements.size();
  int32_t entries_per_elem;
  if (dim == 2 && !is_quadratic) entries_per_elem = 3;
  if (dim == 3 && !is_quadratic) entries_per_elem = 4;
  if (dim == 2 &&  is_quadratic) entries_per_elem = 6;
  if (dim == 3 &&  is_quadratic) entries_per_elem = 10;
  outfile << "CELLS " << nelems << " " << nelems * (entries_per_elem + 1) << '\n';
  if (filetype == VTKFileType::ASCII) {
    for (int i = 0; i < mesh.elements.size(); i++) {
      outfile << entries_per_elem;
      for (auto id : mesh.elements[i]) { outfile << " " << id; }
      if (is_quadratic) {
        for (auto id : mesh.elements_quadratic_ids[i]) { outfile << " " << id + mesh.vertices.size(); }
      }
      outfile << '\n';
    }
  }
  else { // binary
    for (int i = 0; i < mesh.elements.size(); i++) {
      write_binary<int32_t>(outfile, entries_per_elem);
      for (auto id : mesh.elements[i]) {
        write_binary<int32_t>(outfile, id);
      }
      if (is_quadratic) {
        for (auto id : mesh.elements_quadratic_ids[i]) {
          write_binary<int32_t>(outfile, id + mesh.vertices.size());
        }
      }
    }
    outfile << '\n';
  }

  outfile << "CELL_TYPES " << nelems << '\n';
  int32_t type;
  if (dim == 2 && !is_quadratic) type = VTK_TRIANGLE;
  if (dim == 3 && !is_quadratic) type = VTK_TETRAHEDRON;
  if (dim == 2 &&  is_quadratic) type = VTK_QUADRATIC_TRIANGLE;
  if (dim == 3 &&  is_quadratic) type = VTK_QUADRATIC_TETRAHEDRON;
  if (filetype == VTKFileType::ASCII) {
    for ([[maybe_unused]] auto e: mesh.elements) {
      outfile << type << '\n';
    }
  }
  else { // binary
    for ([[maybe_unused]] auto e : mesh.elements) {
      write_binary<int32_t>(outfile, type);
    }
    outfile << '\n';
  }

  outfile.close();

}

void export_vtk(const SimplexMesh<2> & mesh, std::string filename,VTKFileType filetype) {
  export_vtk_impl(mesh, filename, filetype);
}

void export_vtk(const SimplexMesh<3> & mesh, std::string filename, VTKFileType filetype) {
  export_vtk_impl(mesh, filename, filetype);
}
