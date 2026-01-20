#include "geometry/geometry.hpp"

using namespace geometry;

int main() {
  std::function<float(vec3f)> sdf = [](vec3f x) { return dot(x, x) - 2.0f; };
  float cell_size = 0.05f;
  AABB<3> bounds = {{-1.1, -1.1, -1.1}, {1.1, 1.1, 1.1}};
  auto mesh = universal_mesh(sdf, cell_size, bounds);
  export_mesh(mesh, "cpp_test.msh");
}
