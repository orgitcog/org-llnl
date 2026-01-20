#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/function.h>

#include <array>
#include <vector>
#include <thread>
#include <iostream>

#include "BVH.hpp"
#include "geometry/geometry.hpp"

namespace nb = nanobind;

using namespace geometry;
using namespace nb::literals;

namespace impl {
  template < typename T, typename I >
  struct shape;

  template < typename T, int ... i >
  struct shape< T, std::integer_sequence<int, i...> >{
    using type = nb::shape< (std::extent_v<T, i> ? std::extent_v<T, i> : -1) ... >;
  };
}

template < typename T >
struct shape : public impl::shape<T, std::make_integer_sequence< int, std::rank_v<T> > > {};

template < typename T, uint32_t n >
auto to_ndarray(const std::vector< vec< n, T > > & arr) {

  // Delete 'data' when the 'owner' capsule expires
  T * data = new T[arr.size() * n];
  nb::capsule owner(data, [](void *p) noexcept {
     delete[] (T *) p;
  });

  nb::ndarray<nb::numpy, T, nb::shape< -1, n > > out(data, {arr.size(), n}, owner);
  for (uint64_t i = 0; i < arr.size(); i++) {
    for (uint64_t j = 0; j < n; j++) {
      out(i, j) = arr[i][j];
    }
  }
  return out;

}

template < typename T, std::size_t n >
auto to_ndarray(const std::vector< std::array< T, n > > & arr) {

  // Delete 'data' when the 'owner' capsule expires
  T * data = new T[arr.size() * n];
  nb::capsule owner(data, [](void *p) noexcept {
     delete[] (T *) p;
  });

  nb::ndarray<nb::numpy, T, nb::shape< -1, n > > out(data, {arr.size(), n}, owner);
  for (uint64_t i = 0; i < arr.size(); i++) {
    for (uint64_t j = 0; j < n; j++) {
      out(i, j) = arr[i][j];
    }
  }
  return out;

}

NB_MODULE(geometry_bindings, m) {
  m.def("foo", [](const std::function<float(float)> & func) { 
    return func(4.2f);
  });

  m.def("universal_mesh", [](
    const std::function<float(float, float, float)> & func, 
    nb::ndarray< nb::numpy, const float, nb::shape<2, 3> > bounds,
    float cell_size) { 

    std::function< float(vec3f) > wrapped_func = [=](const vec3f & x) { return func(x[0], x[1], x[2]); };

    AABB<3> bounds_aabb = {
        {bounds(0, 0), bounds(0, 1), bounds(0, 2)}, 
        {bounds(1, 0), bounds(1, 1), bounds(1, 2)}
    };

    auto mesh = universal_mesh(wrapped_func, cell_size, bounds_aabb, 0.5f, 0.05f, 3, 1);

    return std::tuple{
      to_ndarray(mesh.vertices),
      to_ndarray(mesh.boundary_elements),
      to_ndarray(mesh.elements)
    };
  });

  m.def("universal_mesh", [](
    nb::ndarray< nb::numpy, const float, nb::shape<-1, 8> > np_capsules,
    nb::ndarray< nb::numpy, const float, nb::shape<2, 3> > np_bounds,
    float cell_size) { 

    static constexpr int dim = 3;

    AABB<dim> bounds{};
    for (int i = 0; i < dim; i++) {
      bounds.min[i] = np_bounds(0, i);
      bounds.max[i] = np_bounds(1, i);
    }

    int num_capsules = np_capsules.shape(0);
    std::vector < Capsule > capsules(num_capsules);
    for (int i = 0; i < num_capsules; i++) {
      vec3f p1{};
      vec3f p2{};
      for (int j = 0; j < dim; j++) {
        p1[j] = np_capsules(i, dim * 0 + j);
        p2[j] = np_capsules(i, dim * 1 + j);
      }
      float r1 = np_capsules(i, dim * 2);
      float r2 = np_capsules(i, dim * 2 + 1);
      capsules[i] = Capsule{p1, p2, r1, r2};
    }

    std::vector<AABB<dim>> bounding_boxes(capsules.size(), AABB<dim>{});
    for (uint32_t i = 0; i < capsules.size(); i++) {
      AABB<dim> box = bounding_box(capsules[i]);
      for (int j = 0; j < dim; j++) {
        bounding_boxes[i].min[j] = box.min[j];
        bounding_boxes[i].max[j] = box.max[j];
      }
    }

    // this uses the literal value "3" instead of dim,
    // to work around a bug in MSVC compiler, where it
    // can't figure out that dim == 3 ...
    BVH<3> bvh(bounding_boxes);

    std::function<float(vecf<dim>)> sdf = [&](vecf<dim> x) -> float {

      float dx = 1.5 * cell_size;

      vec3f x3{};
      AABB<3> box{}; // see above for explanation of literal "3"
      for (int j = 0; j < dim; j++) {
        x3[j] = x[j];
        box.min[j] = x[j] - dx;
        box.max[j] = x[j] + dx;
      };

      float value = 2 * dx;
      bvh.query(box, [&](int i) {
        value = std::min(value, capsules[i].SDF(x3));
      });
      return value;
    };

    auto mesh = universal_mesh(sdf, cell_size, bounds, 0.5f, 0.04f, 5);

    return std::tuple{
      to_ndarray(mesh.vertices),
      to_ndarray(mesh.boundary_elements),
      to_ndarray(mesh.elements)
    };

  });

}
