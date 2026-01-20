#include "geometry.hpp"

#include "BVH.hpp"

#include <iostream>
#include <functional>

namespace geometry {

std::vector<int> cell_values(const SimplexMesh<3> & mesh, 
                             const std::vector< Capsule > & capsules, 
                             const std::vector<int> capsule_values,
                             const float cell_size) {

  constexpr int dim = 3;

  std::vector< AABB<3> > bounding_boxes(capsules.size());
  for (uint32_t i = 0; i < capsules.size(); i++) {
    bounding_boxes[i] = bounding_box(capsules[i]);
  }
  BVH<3> bvh(bounding_boxes);

  std::function<int(vec3f, float)> closest_capsule = [&](vec3f x, float dx) -> float {

    AABB<3>box{
      {x[0] - dx, x[1] - dx, x[2] - dx}, 
      {x[0] + dx, x[1] + dx, x[2] + dx}
    };

    int min_index = -1;
    float min_distance = 2 * dx;
    bvh.query(box, [&](int i) {
      float distance = capsules[i].SDF(x);
      if (distance < min_distance) {
        min_distance = distance;
        min_index = i;
      }
    });
    return min_index;
  };

  float scale = 1.0 / (dim + 1);
  std::vector< int > output(mesh.elements.size());
  for (int i = 0; i < mesh.elements.size(); i++) {
    auto element = mesh.elements[i];

    vec3f center{};
    for (int j = 0; j < dim+1; j++) {
      center += mesh.vertices[element[j]];
    }
    center *= scale;

    int capsule_index = closest_capsule(center, cell_size);
    if (capsule_index == -1) {
      std::cout << "error: unable to find capsule containing element " << i << std::endl;
    }
    output[i] = capsule_values[capsule_index];
  }

  return output;
}

std::vector<float> vertex_values(const SimplexMesh<3> & mesh, 
                                 const std::vector< Capsule > & capsules, 
                                 const std::vector<vec2f> capsule_values,
                                 const float cell_size) {
           
  constexpr int dim = 3;

  std::vector< AABB<3> > bounding_boxes(capsules.size());
  for (uint32_t i = 0; i < capsules.size(); i++) {
    bounding_boxes[i] = bounding_box(capsules[i]);
  }
  BVH<3> bvh(bounding_boxes);

  std::function<int(vec3f, float)> closest_capsule = [&](vec3f x, float dx) -> float {

    AABB<3>box{
      {x[0] - dx, x[1] - dx, x[2] - dx}, 
      {x[0] + dx, x[1] + dx, x[2] + dx}
    };

    int min_index = -1;
    float min_distance = 2 * dx;
    bvh.query(box, [&](int i) {
      float distance = capsules[i].SDF(x);
      if (distance < min_distance) {
        min_distance = distance;
        min_index = i;
      }
    });
    return min_index;
  };

  float scale = 1.0 / (dim + 1);
  std::vector< float > output(mesh.elements.size());
  for (int i = 0; i < mesh.vertices.size(); i++) {

    vec3f x = mesh.vertices[i];

    int capsule_index = closest_capsule(x, cell_size);
    if (capsule_index == -1) {
      std::cout << "error: unable to find capsule containing element " << i << std::endl;
    }

    Capsule c = capsules[capsule_index];
    vec2f v = capsule_values[capsule_index];

    vec3f e1 = normalize(c.p2 - c.p1);
    float t = dot(x - c.p1, e1);

    output[i] = v[0] * (1.0 - t) + v[1] * t;
  }

  return output;
}

}