#include "geometry.hpp"

#include "timer.hpp"
#include "parallel_for.hpp"

#include <set>
#include <iostream>

namespace geometry {

namespace impl {

bool print_timings = true;

vec3f compute_unit_normal(const std::vector< vec3f > & v, const std::array< uint64_t, 2 > & edge) {
  return normalize(cross(v[edge[1]] - v[edge[0]], vec3f{0, 0, 1}));
}

vec3f compute_unit_normal(const std::vector< vec3f > & v, const std::array< uint64_t, 3 > & tri) {
  return normalize(cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]]));
}

vec2f center(const mat2f & edge) {
  return (edge[0] + edge[1]) / 2.0;
}

vec3f center(const mat3f & tri) {
  return (tri[0] + tri[1] + tri[2]) / 3.0;
}

std::tuple< float, std::array< vec3f, 2 > > area_and_gradient(const mat2f & edge) {
    vec2f u = edge[1] - edge[0];
    float norm_u = norm(u);
    vec2f uhat = u / norm_u; // unit vector

    float length = norm_u;
    std::array< vec3f, 2 > dlength_dX = {xyz(-uhat), xyz(uhat)};
    return {length, dlength_dX};
}

std::tuple< float, mat3f > area_and_gradient(const mat3f & tri) {
    vec3f n = cross(tri[1] - tri[0], tri[2] - tri[0]);
    float norm_n = norm(n);
    vec3f nhat = n / norm(n); // unit normal

    float area = 0.5f * norm_n;
    mat3f darea_dX = {
        0.5 * cross(nhat, tri[2] - tri[1]),
        0.5 * cross(nhat, tri[0] - tri[2]),
        0.5 * cross(nhat, tri[1] - tri[0])
    };

    return {area, darea_dX};
}

std::tuple< float, mat3f > quality_and_gradient(const mat3f &x) {

  vec3f L01 = x[1] - x[0];
  vec3f L12 = x[2] - x[1];
  vec3f L20 = x[0] - x[2];

  auto [top, dtop_dx] = area_and_gradient(x); 

  float bot = dot(L01, L01) + dot(L12, L12) + dot(L20, L20);
  auto dbot_dx =
      mat3f{{{4 * x[0][0] - 2 * x[1][0] - 2 * x[2][0],
              4 * x[0][1] - 2 * x[1][1] - 2 * x[2][1],
              4 * x[0][2] - 2 * x[1][2] - 2 * x[2][2]},
             {-2 * x[0][0] + 4 * x[1][0] - 2 * x[2][0],
              -2 * x[0][1] + 4 * x[1][1] - 2 * x[2][1],
              -2 * x[0][2] + 4 * x[1][2] - 2 * x[2][2]},
             {-2 * x[0][0] - 2 * x[1][0] + 4 * x[2][0],
              -2 * x[0][1] - 2 * x[1][1] + 4 * x[2][1],
              -2 * x[0][2] - 2 * x[1][2] + 4 * x[2][2]}}};

  constexpr float scale = 3.4641016151377543864;  // 2 * sqrt(3)

  return {
    scale * (top / bot),
    scale * (dtop_dx / bot - (top / (bot * bot)) * dbot_dx)
  };
}

template < int dim >
void improve_boundary(
  const std::function< float(vecf<dim>) > & f,
  SimplexMesh<dim> & mesh, float stepsize, int steps, int num_threads) {

  timer stopwatch;

  float eps = 1.0e-4;
  auto I = Identity<dim, float>();
  std::function< std::tuple<float, vec3f>(vecf<dim>) > level_set_value_and_gradient = [&](vecf<dim> x) {
    vec3f gradient{};
    float value = f(x);
    for (int i = 0; i < dim; i++) {
      gradient[i] = (f(x + eps * I[i]) - value) / eps;
    }
    return std::tuple{value, gradient};
  };

  std::set< uint64_t > boundary_vertex_id_set;
  for (auto & belem : mesh.boundary_elements) {
    for (uint32_t j = 0; j < dim; j++) {
      boundary_vertex_id_set.insert(belem[j]);
    }
  }

  std::vector< uint64_t > boundary_vertex_ids(boundary_vertex_id_set.begin(), boundary_vertex_id_set.end());

  uint64_t num_boundary_vertices = boundary_vertex_ids.size();
  uint64_t num_boundary_elements = mesh.boundary_elements.size();

  threadpool pool(num_threads);

  stopwatch.start();
  for (int k = 0; k < steps; k++) {

    float objective = 0.0f;
    std::vector< float > quality_scale(mesh.vertices.size(), 0.0f);
    std::vector< vec3f > quality_gradient(mesh.vertices.size(), vec3f{});
    std::vector< vec3f > vertex_gradient(mesh.vertices.size(), vec3f{});

    // compute surface normals and mark boundary nodes

    pool.parallel_for(num_boundary_elements, [&](uint32_t i, uint32_t /*tid*/) {
      auto bdr_elem = mesh.boundary_elements[i];

      mat<dim,dim,float> elem;
      for (int j = 0; j < dim; j++) {
        if constexpr (dim == 2)  {
            elem[j] = xy(mesh.vertices[bdr_elem[j]]);
        } else {
            elem[j] = mesh.vertices[bdr_elem[j]];
        }
      }

      // one point quadrature
      auto c = center(elem);

      auto [f, dfdx] = level_set_value_and_gradient(c);
      auto [A, dAdX] = area_and_gradient(elem);

      #if 0
      if constexpr (dim == 3) {
        auto [Q, dQdX] = quality_and_gradient(elem);
        float expQ = expf(-10.0 * Q);
        dQdX *= expQ;
        for (int j = 0; j < dim; j++) {
          vec3f g = xyz(dQdX[j]);
          quality_gradient[bdr_elem[j]] += g;
          quality_scale[bdr_elem[j]] += expQ;
        }
      }
      #endif

      objective += f * f * A;
      for (uint32_t j = 0; j < dim; j++) {
        vertex_gradient[bdr_elem[j]] += (2 * A * dfdx + f * dAdX[j]) * f / 3.0;
      }
 
    });

    // does this benefit from multiple threads (?)
    for (uint64_t i = 0; i < num_boundary_vertices; i++) {
      uint64_t id = boundary_vertex_ids[i];

      vecf<dim> xd;
      for (uint32_t j = 0; j < dim; j++) {
        xd[j] = mesh.vertices[id][j];
      }
      auto [_, n] = level_set_value_and_gradient(xd);
      vec3f g = vertex_gradient[id];
      n = normalize(n);

      vec3f u = g - dot(g, n) * n;

      mesh.vertices[id] -= stepsize * u;
    }

    std::cout << objective << std::endl;

  }
  stopwatch.stop();

  if (print_timings) {
    std::cout << "improve_boundary iterations: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

}

struct BasisR3 {
  mat3f A;
  int count;

  void gram_schmidt(vec3f & n) {
    for (int i = 0; i < count; i++) {
      n -= dot(A[i], n) * A[i];
    }
  }

  void insert(vec3f n /* unit vector */) {
    if (count < 3) {
      gram_schmidt(n);
      if (norm(n) > 1.0e-6) {
        A[count++] = normalize(n);
      }
    }
  }

  vec3f project_out(vec3f n) {
    return n - dot(dot(A, n), A);
  }
};

#if 0
template < int dim >
void boundary_dvr(SimplexMesh<dim> & mesh, float alpha, float stepsize, int steps, int num_threads) {

  timer stopwatch;

  std::set< uint64_t > boundary_vertex_id_set;
  for (auto & belem : mesh.boundary_elements) {
    for (uint32_t j = 0; j < dim; j++) {
      boundary_vertex_id_set.insert(belem[j]);
    }
  }

  std::vector< uint64_t > boundary_vertex_ids(boundary_vertex_id_set.begin(), boundary_vertex_id_set.end());

  uint64_t num_boundary_vertices = boundary_vertex_ids.size();
  uint64_t num_boundary_elements = mesh.boundary_elements.size();

  stopwatch.start();
  for (int k = 0; k < steps; k++) {

    float objective = 0.0f;
    std::vector< float > quality_scale(mesh.vertices.size(), 0.0f);
    std::vector< vec3f > quality_gradient(mesh.vertices.size(), vec3f{});

    // compute surface normals and mark boundary nodes
    for (uint64_t i = 0; i < num_boundary_elements; i++) {
      auto bdr_elem = mesh.boundary_elements[i];

      tensor<float, dim, dim> elem;
      for (int j = 0; j < dim; j++) {
        if constexpr (dim == 2)  {
            elem[j] = xy(mesh.vertices[bdr_elem[j]]);
        } else {
            elem[j] = mesh.vertices[bdr_elem[j]];
        }
      }

      if constexpr (dim == 3) {
        auto [Q, dQdX] = quality_and_gradient(elem);
        float expQ = expf(-10.0 * Q);
        dQdX *= expQ;
        for (int j = 0; j < dim; j++) {
          vec3f g = xyz(dQdX[j]);
          quality_gradient[bdr_elem[j]] += g;
          quality_scale[bdr_elem[j]] += expQ;
        }
      }
    }

    // does this benefit from multiple threads (?)
    for (uint64_t i = 0; i < num_boundary_vertices; i++) {
      uint64_t id = boundary_vertex_ids[i];

      vec3f g = vertex_gradient[id];
      n = normalize(n);

      vec3f u = g - dot(g, n) * n;

      mesh.vertices[id] -= stepsize * u;
    }

    std::cout << objective << std::endl;

  }
  stopwatch.stop();

  if (print_timings) {
    std::cout << "improve_boundary iterations: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

}
#endif

}

void improve_boundary(const std::function< float(vec2f) > & func, SimplexMesh<2> & mesh, float stepsize, int steps, int num_threads) {
    impl::improve_boundary<2>(func, mesh, stepsize, steps, num_threads);
};

void improve_boundary(const std::function< float(vec3f) > & func, SimplexMesh<3> & mesh, float stepsize, int steps, int num_threads) {
    impl::improve_boundary<3>(func, mesh, stepsize, steps, num_threads);
};

#if 0
template <>
void boundary_dvr(SimplexMesh<2> & mesh, float alpha, float stepsize, int steps, int num_threads) {
    impl::boundary_dvr(mesh, alpha, stepsize, steps, num_threads);
};

template <>
void boundary_dvr(SimplexMesh<3> & mesh, float alpha, float stepsize, int steps, int num_threads) {
    impl::boundary_dvr(mesh, alpha, stepsize, steps, num_threads);
};
#endif

}