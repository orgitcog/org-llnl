#include "geometry/geometry.hpp"
#include "binary_io.hpp"

#include "fm/operations/print.hpp"

#include "BVH.hpp"

using namespace geometry;

void voronoi_2D() {

  float r = 0.05f;
  int n = 128;

  std::vector<mat2f> edges = read_binary<mat2f>(GEOMETRY_DATA_DIR"voronoi_edges.bin");

  std::vector< Capsule > capsules;

  for (auto edge : edges) {
    vec3f p = {edge[0][0], edge[0][1], 0.0f};
    vec3f q = {edge[1][0], edge[1][1], 0.0f};
    capsules.push_back(Capsule{p, q, r, r});
  }

  std::vector< AABB<3> > bounding_boxes(capsules.size());
  for (uint32_t i = 0; i < capsules.size(); i++) {
    bounding_boxes[i] = bounding_box(capsules[i]);
  }
  BVH<3> bvh(bounding_boxes);

  AABB<2> bounds{
    {bvh.global.min[0], bvh.global.min[1]}, 
    {bvh.global.max[0], bvh.global.max[1]} 
  };

  std::cout << bounds.min << ", " << bounds.max << std::endl;

  vec3f widths = bvh.global.max - bvh.global.min;
  float cell_size = std::max(widths[0], widths[1]) / n;

  BackgroundGrid<2> grid(bounds, cell_size);

  float dx = 1.5 * cell_size;

  std::function<float(vec2f)> sdf = [&](vec2f x) -> float {
    AABB<3>box{
      {x[0] - dx, x[1] - dx, 0.0f}, 
      {x[0] + dx, x[1] + dx, 0.0f}
    };

    float value = 2 * dx;
    bvh.query(box, [&](int i) {
      value = std::min(value, capsules[i].SDF(vec3f{x[0], x[1], 0.0f}));
    });
    return value;
  };

  for (int j = 0; j < 2 * grid.n[1] + 1; j++) {
    for (int i = 0; i < 2 * grid.n[0] + 1; i++) {
      grid({i,j}) = sdf(grid.vertex({i,j}));
    }
  }

  auto mesh = universal_mesh(grid);

  std::cout << "generated mesh with:" << std::endl;
  std::cout << "  " << mesh.vertices.size() << " vertices" << std::endl;
  std::cout << "  " << mesh.elements.size() << " elements" << std::endl;
  std::cout << "  " << mesh.boundary_elements.size() << " boundary_elements" << std::endl;

  export_stl(mesh, "voronoi.stl");

}

int main() {

  {
    BackgroundGrid<2> mesh({{0, 0}, {1, 2}}, 0.25f);

    std::vector< vec2f > points2D;

    for (int j = 0; j < mesh.n[1] * 2 + 1; j++) {
      for (int i = 0; i < mesh.n[0] * 2 + 1; i++) {
        points2D.push_back(mesh.vertex({i,j}));
      }
    }

    std::cout << mesh.cell_size() << std::endl;

    write_binary(points2D, "points2D.bin");
  }

  {
    BackgroundGrid<3> mesh({{0, 0, 0}, {1, 2, 3}}, 0.25f);

    std::vector< vec3f > points3D;

    for (int k = 0; k < mesh.n[2] * 2 + 1; k++) {
      for (int j = 0; j < mesh.n[1] * 2 + 1; j++) {
        for (int i = 0; i < mesh.n[0] * 2 + 1; i++) {
          points3D.push_back(mesh.vertex({i,j,k}));
        }
      }
    }

    std::cout << mesh.cell_size() << std::endl;

    write_binary(points3D, "points3D.bin");
  }

  voronoi_2D();

}