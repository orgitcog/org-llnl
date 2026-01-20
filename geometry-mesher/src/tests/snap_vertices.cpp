#include "geometry/geometry.hpp"

#include "gtest/gtest.h"

#include "fm/operations/print.hpp"

using namespace geometry;

template < int dim > 
std::vector< float > calculate_values(SimplexMesh<dim> mesh) {
  std::vector< float > values(mesh.vertices.size());
  for (int i = 0; i < mesh.vertices.size(); i++) {
    values[i] = mesh.vertices[i][0];
  }
  return values;
}

template < int dim >
std::vector< uint64_t > classify_elements(const SimplexMesh<dim> & mesh, const std::vector< float > & values) {
  std::vector< uint64_t > partially_inside;
  for (int i = 0; i < mesh.elements.size(); i++) {
    auto element = mesh.elements[i];
    float min_value, max_value;
    min_value = max_value = values[element[0]];
    for (int j = 1; j < dim + 1; j++) {
      min_value = std::min(min_value, values[element[j]]);
      max_value = std::max(max_value, values[element[j]]);
    }
    if (min_value * max_value <= 0.0f){
      partially_inside.push_back(i);
    }
  }
  return partially_inside;
}

template < int dim >
void run_test(std::vector<vec3f> vertices) {
  SimplexMesh<dim> mesh = {vertices, {{0, 1, 2}, {1, 3, 2}}};
  std::vector< float > values = calculate_values(mesh);
  std::vector< uint64_t > partially_inside = classify_elements(mesh, values);

  std::cout << partially_inside.size() << " partially-inside elements" << std::endl;

  snap_vertices_to_boundary(mesh, values, partially_inside, 0.25f, 1, false);

  EXPECT_EQ(mesh.boundary_elements.size(), 1);
  auto & v = mesh.vertices;
  for (auto belem : mesh.boundary_elements) {
    std::cout << v[belem[0]] << " " << v[belem[1]] << std::endl;
  }
}

TEST(two_dim, 2_in) { 
  run_test<2>({{-1.1f, 0.5f, 0.0f}, {-0.1f, 0.0f, 0.0f}, {-0.1f, 1.0f, 0.0f}, { 0.9f, 0.5f, 0.0f}});
}

TEST(two_dim, 1_far_in_1_on_boundary) { 
  run_test<2>({{-1.0f, 0.0f, 0.0f}, { 0.0f, 0.0f, 0.0f}, {-0.5f, 1.0f, 0.0f}, { 0.5f, 1.0f, 0.0f}});
}

TEST(two_dim, 1_in_1_on_boundary) { 
  run_test<2>({{-1.0f, 0.5f, 0.0f}, {-0.1f, 0.0f, 0.0f}, { 0.0f, 1.0f, 0.0f}, { 1.0f, 0.5f, 0.0f}});
}

TEST(two_dim, 2_on_boundary) { 
  run_test<2>({{-1.0f, 0.5f, 0.0f}, { 0.0f, 0.0f, 0.0f}, { 0.0f, 1.0f, 0.0f}, { 1.0f, 0.5f, 0.0f}});
}

TEST(two_dim, 1_on_boundary_1_out) { 
  run_test<2>({{-1.0f, 0.5f, 0.0f}, {+0.1f, 0.0f, 0.0f}, { 0.0f, 1.0f, 0.0f}, { 1.0f, 0.5f, 0.0f}});
}

TEST(two_dim, 1_on_boundary_1_far_out) { 
  run_test<2>({{-0.5f, 1.0f, 0.0f}, { 0.0f, 0.0f, 0.0f}, { 0.5f, 1.0f, 0.0f}, { 1.0f, 0.0f, 0.0f}});
}

TEST(two_dim, 2_out) { 
  run_test<2>({{-0.9f, 0.5f, 0.0f}, {+0.1f, 0.0f, 0.0f}, {+0.1f, 1.0f, 0.0f}, { 1.1f, 0.5f, 0.0f}});
}