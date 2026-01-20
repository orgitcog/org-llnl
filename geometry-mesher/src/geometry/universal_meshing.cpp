#include <cmath>
#include <mutex>
#include <map>
#include <set>
#include <thread>
#include <tuple>
#include <bitset>
#include <unordered_set>
#include <unordered_map>

#include "parasort.h"

#include "binary_io.hpp"  // TODO: remove
#include "geometry.hpp"
#include "timer.hpp"
#include "parallel_for.hpp"
#include "morton.hpp"

#ifdef UM_TIFF_SUPPORT
#include "geometry/image.hpp"
#endif

#define BOUNDS_CHECKING false

namespace geometry {

// static constexpr bool debug_output = false;
static constexpr bool print_timings = true;

template < typename T, size_t n >
std::ostream & operator<<(std::ostream& out, std::array< T, n > arr) {
  out << "{ ";
  for (auto value : arr) { out << value << " "; }
  out << "}";
  return out;
}

std::tuple<float, mat3x2f> quality_and_gradient(
    const Triangle &tri) {
  auto &x = tri.vertices;

  vec2f L01 = xy(x[1] - x[0]);
  vec2f L12 = xy(x[2] - x[1]);
  vec2f L20 = xy(x[0] - x[2]);

  float top = det(mat2f{L20, L01});
  auto dtop_dx = mat3x2f{{{x[1][1] - x[2][1], -x[1][0] + x[2][0]},
                          {-x[0][1] + x[2][1], x[0][0] - x[2][0]},
                          {x[0][1] - x[1][1], -x[0][0] + x[1][0]}}};

  float bot = dot(L01, L01) + dot(L12, L12) + dot(L20, L20);
  auto dbot_dx =
      mat3x2f{{{4 * x[0][0] - 2 * x[1][0] - 2 * x[2][0],
                4 * x[0][1] - 2 * x[1][1] - 2 * x[2][1]},
               {-2 * x[0][0] + 4 * x[1][0] - 2 * x[2][0],
                -2 * x[0][1] + 4 * x[1][1] - 2 * x[2][1]},
               {-2 * x[0][0] - 2 * x[1][0] + 4 * x[2][0],
                -2 * x[0][1] - 2 * x[1][1] + 4 * x[2][1]}}};

  constexpr float scale = 3.4641016151377543864;  // 2 * sqrt(3)

  return {
    scale * (top / bot),
    scale * (dtop_dx / bot - (top / (bot * bot)) * dbot_dx)
  };
}

std::tuple<float, mat4x3f > quality_and_gradient(const Tetrahedron & tet) {
  auto & x = tet.vertices;

  vec3f L01 = x[1]-x[0];
  vec3f L02 = x[2]-x[0];
  vec3f L03 = x[3]-x[0];
  vec3f L12 = x[2]-x[1];
  vec3f L13 = x[3]-x[1];
  vec3f L23 = x[3]-x[2];
  float tmp = sqrt(dot(L01,L01)+dot(L02,L02)+dot(L03,L03)+dot(L12,L12)+dot(L13,L13)+dot(L23,L23)); 

  float top = det(mat3f{L01, L02, L03});

  auto dtop_dx = mat4x3f{{
    {
       x[1][2] * x[2][1] - x[1][1] * x[2][2] - x[1][2] * x[3][1] + x[2][2] * x[3][1] + x[1][1] * x[3][2] - x[2][1] * x[3][2], 
      -x[1][2] * x[2][0] + x[1][0] * x[2][2] + x[1][2] * x[3][0] - x[2][2] * x[3][0] - x[1][0] * x[3][2] + x[2][0] * x[3][2], 
       x[1][1] * x[2][0] - x[1][0] * x[2][1] - x[1][1] * x[3][0] + x[2][1] * x[3][0] + x[1][0] * x[3][1] - x[2][0] * x[3][1]
    }, {
      -x[0][2] * x[2][1] + x[0][1] * x[2][2] + x[0][2] * x[3][1] - x[2][2] * x[3][1] - x[0][1] * x[3][2] + x[2][1] * x[3][2], 
       x[0][2] * x[2][0] - x[0][0] * x[2][2] - x[0][2] * x[3][0] + x[2][2] * x[3][0] + x[0][0] * x[3][2] - x[2][0] * x[3][2], 
      -x[0][1] * x[2][0] + x[0][0] * x[2][1] + x[0][1] * x[3][0] - x[2][1] * x[3][0] - x[0][0] * x[3][1] + x[2][0] * x[3][1]
    }, {
       x[0][2] * x[1][1] - x[0][1] * x[1][2] - x[0][2] * x[3][1] + x[1][2] * x[3][1] + x[0][1] * x[3][2] - x[1][1] * x[3][2], 
      -x[0][2] * x[1][0] + x[0][0] * x[1][2] + x[0][2] * x[3][0] - x[1][2] * x[3][0] - x[0][0] * x[3][2] + x[1][0] * x[3][2], 
       x[0][1] * x[1][0] - x[0][0] * x[1][1] - x[0][1] * x[3][0] + x[1][1] * x[3][0] + x[0][0] * x[3][1] - x[1][0] * x[3][1]
    }, {
      -x[0][2] * x[1][1] + x[0][1] * x[1][2] + x[0][2] * x[2][1] - x[1][2] * x[2][1] - x[0][1] * x[2][2] + x[1][1] * x[2][2], 
       x[0][2] * x[1][0] - x[0][0] * x[1][2] - x[0][2] * x[2][0] + x[1][2] * x[2][0] + x[0][0] * x[2][2] - x[1][0] * x[2][2], 
      -x[0][1] * x[1][0] + x[0][0] * x[1][1] + x[0][1] * x[2][0] - x[1][1] * x[2][0] - x[0][0] * x[2][1] + x[1][0] * x[2][1]
    }
  }};

  float bot = tmp * tmp * tmp;
  auto dbot_dx = 1.5f * tmp * mat4x3f{{
    { 
      6.0f * x[0][0] - 2.0f * (x[1][0] + x[2][0] + x[3][0]), 
      6.0f * x[0][1] - 2.0f * (x[1][1] + x[2][1] + x[3][1]), 
      6.0f * x[0][2] - 2.0f * (x[1][2] + x[2][2] + x[3][2])
    }, {
      -2.0f * (x[0][0] - 3.0f * x[1][0] + x[2][0] + x[3][0]), 
      -2.0f * (x[0][1] - 3.0f * x[1][1] + x[2][1] + x[3][1]), 
      -2.0f * (x[0][2] - 3.0f * x[1][2] + x[2][2] + x[3][2])
    }, {
      -2.0f * (x[0][0] + x[1][0] - 3.0f * x[2][0] + x[3][0]), 
      -2.0f * (x[0][1] + x[1][1] - 3.0f * x[2][1] + x[3][1]), 
      -2.0f * (x[0][2] + x[1][2] - 3.0f * x[2][2] + x[3][2])
    }, {
      -2.0f * (x[0][0] + x[1][0] + x[2][0] - 3.0f * x[3][0]), 
      -2.0f * (x[0][1] + x[1][1] + x[2][1] - 3.0f * x[3][1]), 
      -2.0f * (x[0][2] + x[1][2] + x[2][2] - 3.0f * x[3][2])
    }
  }};

  constexpr float scale = 20.784609690826527522; // 12 * sqrt(3)

  return {
    scale * (top / bot),
    scale * (dtop_dx / bot - (top / (bot * bot)) * dbot_dx)
  };
}

template <typename T>
void only_keep_marked_values(std::vector<T> &values,
                             const std::vector<char> &keep) {
  int index = 0;
  for (uint32_t i = 0; i < keep.size(); i++) {
    values[index] = values[i];
    index += (keep[i] != 0);
  }
  values.erase(values.begin() + index, values.end());
}

template <typename T>
std::unordered_set<T> set_combine(
    const std::vector<std::vector<T>> &containers) {
  uint32_t n = containers.size();
  size_t total_size = 0;
  for (uint32_t i = 0; i < n; i++) {
    total_size += containers[i].size();
  }

  std::unordered_set<T> output;
  output.reserve(total_size);
  for (uint32_t i = 0; i < n; i++) {
    output.insert(containers[i].begin(), containers[i].end());
  }
  return output;
}

template <typename T, int m, int n>
bool all_zeros(mat<m, n, T> &A) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (A(i, j) != 0.0) return false;
    }
  }
  return true;
}

template <typename T, typename index_t, std::size_t n>
auto gather(const std::vector<T> &arr, const std::array<index_t, n> &indices) {
  std::array<T, n> output{};
  for (int i = 0; i < n; i++) {
    output[i] = arr[indices[i]];
  }
  return output;
}


template <int dim>
struct Slice {
  vec3f vertices[2 * (dim - 1)];
};

inline float sign(float val) { return (0.0f < val) - (val < 0.0f); }

template <typename X>
auto find_root(const X (&x)[2], const float (&f)[2]) {
  if (sign(f[0] * f[1]) >= 0.0) {
    return 0.5f * (x[0] + x[1]);
  } else {
    return (x[1] * f[0] - x[0] * f[1]) / (f[0] - f[1]);
  }
}

// map the values {-1, 0, 1} -> {0, 1, 2}
// and then concatenate the digits to make a 3-digit number (base 10)
constexpr uint32_t id(int a, int b, int c) {
  return 10 * (10 * (a + 1) + (b + 1)) + (c + 1);
}

// map the values {-1, 0, 1} -> {0, 1, 2}
// and then concatenate the digits to make a 4-digit number (base 10)
constexpr uint32_t id(int a, int b, int c, int d) {
  return 10 * (10 * (10 * (a + 1) + (b + 1)) + (c + 1)) + (d + 1);
}

Slice<2> SliceSimplex(Simplex<2> simplex, float v[3]) {
  auto p = simplex.vertices;

  auto s = id(sign(v[0]), sign(v[1]), sign(v[2]));

  // a triangle must contain:
  //  - at least 1 negative vertex
  //  - at least 1 positive vertex
  // or the zero isocontour won't be inside of it

  vec3f m0 = find_root({p[0], p[1]}, {v[0], v[1]});
  vec3f m1 = find_root({p[1], p[2]}, {v[1], v[2]});
  vec3f m2 = find_root({p[2], p[0]}, {v[2], v[0]});

  // clang-format off
  switch (s) {
    case id(-1, -1, -1): return {};
    case id(-1, -1,  0): return {};
    case id(-1, -1, +1): return {m1, m2};
    case id(-1,  0, -1): return {}; break;
    case id(-1,  0,  0): return {}; break;
    case id(-1,  0, +1): return {p[1], m2}; break;
    case id(-1, +1, -1): return {m0, m1}; break;
    case id(-1, +1,  0): return {m0, p[2]}; break;
    case id(-1, +1, +1): return {m0, m2}; break;

    case id( 0, -1, -1): return {}; break;
    case id( 0, -1,  0): return {}; break;
    case id( 0, -1, +1): return {m1, p[0]}; break;
    case id( 0,  0, -1): return {}; break;
    case id( 0,  0,  0): return {}; break;
    case id( 0,  0, +1): return {}; break;
    case id( 0, +1, -1): return {p[0], m1}; break;
    case id( 0, +1,  0): return {}; break;
    case id( 0, +1, +1): return {}; break;

    case id(+1, -1, -1): return {m2, m0}; break;
    case id(+1, -1,  0): return {p[2], m0}; break;
    case id(+1, -1, +1): return {m1, m0}; break;
    case id(+1,  0, -1): return {m2, p[1]}; break;
    case id(+1,  0,  0): return {}; break;
    case id(+1,  0, +1): return {}; break;
    case id(+1, +1, -1): return {m2, m1}; break;
    case id(+1, +1,  0): return {}; break;
    case id(+1, +1, +1): return {}; break;
    
  }
  // clang-format on

  // unreachable
  return {};

}

Slice<3> SliceSimplex(Simplex<3> simplex, float v[4]) {

  auto p = simplex.vertices;

  auto s = id(sign(v[0]), sign(v[1]), sign(v[2]), sign(v[3]));

  // clang-format off
  if ((s == id( 0,  0,  0,  0)) || 
      (s == id(-1, -1,  0,  0)) || 
      (s == id(-1,  0, -1,  0)) || 
      (s == id(-1,  0,  0, -1)) ||
      (s == id( 0, -1, -1,  0)) || 
      (s == id( 0, -1,  0, -1)) || 
      (s == id( 0,  0, -1, -1)) || 
      (s == id( 1, +1,  0,  0)) ||
      (s == id( 1,  0, +1,  0)) || 
      (s == id( 1,  0,  0, +1)) || 
      (s == id( 0, +1, +1,  0)) || 
      (s == id( 0, +1,  0, +1)) ||
      (s == id( 0,  0, +1, +1)) || 
      (s == id( 1, +1, +1, +1)) || 
      (s == id( 0, -1, -1, -1)) || 
      (s == id( 0, +1, +1, +1)) ||
      (s == id(-1,  0, -1, -1)) || 
      (s == id( 1,  0, +1, +1)) || 
      (s == id(-1, -1,  0, -1)) || 
      (s == id( 1, +1,  0, +1)) ||
      (s == id(-1, -1, -1,  0)) || 
      (s == id( 1, +1, +1,  0))) return {};

  if (s == id(-1,  0,  0,  0)) return {p[1], p[2], p[3], p[3]};
  if (s == id( 0, -1,  0,  0)) return {p[0], p[3], p[2], p[2]};
  if (s == id( 0,  0, -1,  0)) return {p[0], p[1], p[3], p[3]};
  if (s == id( 0,  0,  0, -1)) return {p[2], p[1], p[0], p[0]};
                                                       
  if (s == id(+1,  0,  0,  0)) return {p[3], p[2], p[1], p[1]};
  if (s == id( 0, +1,  0,  0)) return {p[2], p[3], p[0], p[0]};
  if (s == id( 0,  0, +1,  0)) return {p[3], p[1], p[0], p[0]};
  if (s == id( 0,  0,  0, +1)) return {p[0], p[1], p[2], p[2]};

  vec3f m01 = find_root({p[0], p[1]}, {v[0], v[1]});
  vec3f m02 = find_root({p[0], p[2]}, {v[0], v[2]});
  vec3f m03 = find_root({p[0], p[3]}, {v[0], v[3]});
  vec3f m12 = find_root({p[1], p[2]}, {v[1], v[2]});
  vec3f m13 = find_root({p[1], p[3]}, {v[1], v[3]});
  vec3f m23 = find_root({p[2], p[3]}, {v[2], v[3]});

  if (s == id(+1, -1,  0,  0)) return {m01, p[3], p[2], p[2]};
  if (s == id(-1, +1,  0,  0)) return {m01, p[2], p[3], p[3]};
  if (s == id(+1,  0, -1,  0)) return {m02, p[1], p[3], p[3]};
  if (s == id(-1,  0, +1,  0)) return {m02, p[3], p[1], p[1]};
  if (s == id(+1,  0,  0, -1)) return {m03, p[2], p[1], p[1]};
  if (s == id(-1,  0,  0, +1)) return {m03, p[1], p[2], p[2]};
  if (s == id( 0, +1, -1,  0)) return {m12, p[3], p[0], p[0]};
  if (s == id( 0, -1, +1,  0)) return {m12, p[0], p[3], p[3]};
  if (s == id( 0, +1,  0, -1)) return {m13, p[0], p[2], p[2]};
  if (s == id( 0, -1,  0, +1)) return {m13, p[2], p[0], p[0]};
  if (s == id( 0,  0, +1, -1)) return {m23, p[1], p[0], p[0]};
  if (s == id( 0,  0, -1, +1)) return {m23, p[0], p[1], p[1]};

  if (s == id( 0, -1, -1, +1)) return {m13, m23, p[0], p[0]};
  if (s == id( 0, -1, +1, -1)) return {m23, m12, p[0], p[0]};
  if (s == id( 0, -1, +1, +1)) return {m13, m12, p[0], p[0]};
  if (s == id( 0, +1, -1, -1)) return {m12, m13, p[0], p[0]};
  if (s == id( 0, +1, -1, +1)) return {m12, m23, p[0], p[0]};
  if (s == id( 0, +1, +1, -1)) return {m23, m13, p[0], p[0]};

  if (s == id(-1,  0, -1, +1)) return {m23, m03, p[1], p[1]};
  if (s == id(-1,  0, +1, -1)) return {m02, m23, p[1], p[1]};
  if (s == id(-1,  0, +1, +1)) return {m02, m03, p[1], p[1]};
  if (s == id(+1,  0, -1, -1)) return {m03, m02, p[1], p[1]};
  if (s == id(+1,  0, -1, +1)) return {m23, m02, p[1], p[1]};
  if (s == id(+1,  0, +1, -1)) return {m03, m23, p[1], p[1]};
                                                     
  if (s == id(-1, -1,  0, +1)) return {m03, m13, p[2], p[2]};
  if (s == id(-1, +1,  0, -1)) return {m13, m01, p[2], p[2]};
  if (s == id(-1, +1,  0, +1)) return {m03, m01, p[2], p[2]};
  if (s == id(+1, -1,  0, -1)) return {m01, m03, p[2], p[2]};
  if (s == id(+1, -1,  0, +1)) return {m01, m13, p[2], p[2]};
  if (s == id(+1, +1,  0, -1)) return {m13, m03, p[2], p[2]};
                                                     
  if (s == id(-1, -1, +1,  0)) return {m12, m02, p[3], p[3]};
  if (s == id(-1, +1, -1,  0)) return {m01, m12, p[3], p[3]};
  if (s == id(-1, +1, +1,  0)) return {m01, m02, p[3], p[3]};
  if (s == id(+1, -1, -1,  0)) return {m02, m01, p[3], p[3]};
  if (s == id(+1, -1, +1,  0)) return {m12, m01, p[3], p[3]};
  if (s == id(+1, +1, -1,  0)) return {m02, m12, p[3], p[3]};

  if (s == id(-1, +1, +1, +1)) return {m01, m02, m03, m03};
  if (s == id(+1, -1, +1, +1)) return {m01, m13, m12, m12};
  if (s == id(+1, +1, -1, +1)) return {m02, m12, m23, m23};
  if (s == id(+1, +1, +1, -1)) return {m03, m23, m13, m13};
                                                    
  if (s == id(+1, -1, -1, -1)) return {m03, m02, m01, m01};
  if (s == id(-1, +1, -1, -1)) return {m12, m13, m01, m01};
  if (s == id(-1, -1, +1, -1)) return {m23, m12, m02, m02};
  if (s == id(-1, -1, -1, +1)) return {m13, m23, m03, m03};

  if (s == id(+1, -1, -1, +1)) return {m23, m01, m13, m02};
  if (s == id(-1, +1, +1, -1)) return {m01, m23, m13, m02};
  if (s == id(+1, -1, +1, -1)) return {m01, m23, m12, m03};
  if (s == id(-1, +1, -1, +1)) return {m23, m01, m12, m03};
  if (s == id(+1, +1, -1, -1)) return {m12, m03, m02, m13};
  if (s == id(-1, -1, +1, +1)) return {m03, m12, m02, m13};
  // clang-format on

  // unreachable
  return {};

}

vec3f closest_point_projection(const Slice<2> & s, const vec3f & p){
  return closest_point_projection(Line{s.vertices[0], s.vertices[1]}, p);
}

vec3f closest_point_projection(const Slice<3> & s, const vec3f & p){
  vec3f q1 = closest_point_projection(Triangle{s.vertices[0], s.vertices[1], s.vertices[2]}, p);
  if (s.vertices[2] == s.vertices[3]) {
    return q1;
  } else {
    vec3f q2 = closest_point_projection(Triangle{s.vertices[1], s.vertices[2], s.vertices[3]}, p);
    return (norm(q1 - p) < norm(q2 - p)) ? q1 : q2;
  }
}

template < std::size_t n, typename T >
std::array < T, 2 > min_max(std::array< T, n > values) {
  std::array< T, 2 > output{values[0], values[0]};
  for (std::size_t i = 1; i < n; i++) {
    output[0] = std::min(output[0], values[i]);
    output[1] = std::max(output[1], values[i]);
  }
  return output;
}

void sample_implicit_function(SimplexMesh<2> &mesh, 
            std::vector<float> &values,
            std::vector< uint64_t > &partially_inside,
            const std::function<float(vec2f)> & f,
            float cell_size,
            AABB<3> bounds,
            int num_threads,
            bool boundary_only)
{

  static constexpr int block_size = 4;
  constexpr float sqrt3_over_2 = 0.8660254037844386f;

  const int nx = floor((bounds.max[0] - bounds.min[0]) / cell_size);
  const int ny = floor((bounds.max[1] - bounds.min[1]) / (sqrt3_over_2 * cell_size));

  vec3f offset{bounds.min[0], bounds.min[1], 0.0f};

  auto gridpoint = [&](int i, int j) {
    return vec3f{0.5f * i + 0.25f * (j % 2 == 1), 0.5f * sqrt3_over_2 * j, 0.0} * cell_size + offset;
  };

  static constexpr int tris_per_tile = 8;
  static constexpr int tile[tris_per_tile][3] = {
    {0, 1, 3}, {1, 4, 3}, {1, 2, 4}, {2, 5, 4},
    {3, 7, 6}, {3, 4, 7}, {4, 8, 7}, {4, 5, 8}
  };

  auto vertex_id = [&](int i, int j) { 
    return j * (2 * nx + 1) + i;
  };

  num_threads = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;

  timer stopwatch;
  stopwatch.start();

  std::unordered_map < uint64_t, float > vertex_info;

  for (int ty = 0; ty < ny; ty += block_size) {
    for (int tx = 0; tx < nx; tx += block_size) {
      
      float block_min_value = 1.0e10;
      float block_values[2 * block_size + 1][2 * block_size + 1]; 

      // evaluate the implicit function at each of the vertices in the local block
      for (int dy = 0; dy <= 2 * block_size; dy++) {
        int vy = 2 * ty + dy;
        for (int dx = 0; dx <= 2 * block_size; dx++) {
          int vx = 2 * tx + dx;

          // don't evaluate `f` outside of the sampling domain
          if ((vx <= 2 * nx) && (vy <= 2 * ny)) {
            float value = f(xy(gridpoint(vx, vy)));
            block_values[dx][dy] = value;
            block_min_value = std::min(block_min_value, value);
          }

        }
      }

      // there is no work to be done if the entire block is outside
      if (block_min_value >= 0.0) continue;

      // for each tile in this block, identify triangles
      // that are either partially or entirely inside
      bool keep[2 * block_size + 1][2 * block_size + 1]{}; 
      for (int by = 0; by < block_size; by++) {
        for (int bx = 0; bx < block_size; bx++) {

          if (tx + bx >= nx || ty + by >= ny) continue;

          float tile_values[9];
          uint64_t tile_ids[9];
          for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
              tile_values[j * 3 + i] = block_values[2 * bx + i][2 * by + j];
              tile_ids[j * 3 + i] = vertex_id(2 * (tx + bx) + i, 2 * (ty + by)+ j);
            }
          }

          for (auto tri : tile) {
            std::array tri_values = {tile_values[tri[0]], tile_values[tri[1]], tile_values[tri[2]]};
            auto [min_value, max_value] = min_max(tri_values);

            // discard triangles that are entirely outside
            if (min_value >= 0.0) continue;

            // note which triangles are partially inside
            if (min_value * max_value <= 0) {
              partially_inside.push_back(mesh.elements.size());
            }

            mesh.elements.push_back({tile_ids[tri[0]], tile_ids[tri[1]], tile_ids[tri[2]]});

            // mark which vertices in the tile belong to at least one active triangle
            keep[2 * bx + (tri[0] % 3)][2 * by + (tri[0] / 3)] = true;
            keep[2 * bx + (tri[1] % 3)][2 * by + (tri[1] / 3)] = true;
            keep[2 * bx + (tri[2] % 3)][2 * by + (tri[2] / 3)] = true;
          }

        }
      }

      for (int dy = 0; dy <= 2 * block_size; dy++) {
        int vy = 2 * ty + dy;
        for (int dx = 0; dx <= 2 * block_size; dx++) {
          int vx = 2 * tx + dx;
          if ((vx <= 2 * nx) && (vy <= 2 * ny)) {
            if (keep[dx][dy]) {
              vertex_info[vertex_id(vx, vy)] = block_values[dx][dy];
            }
          }
        }
      }

    }
  }

  values.resize(vertex_info.size());
  mesh.vertices.resize(vertex_info.size());
  std::unordered_map< uint64_t, uint64_t > new_ids(vertex_info.size());

  uint64_t new_id = 0;
  for (auto [vertex_id, value] : vertex_info) {
    values[new_id] = value;
    uint32_t i = vertex_id % (2 * nx + 1); 
    uint32_t j = vertex_id / (2 * nx + 1); 
    mesh.vertices[new_id] = gridpoint(i, j);
    new_ids[vertex_id] = new_id++;
  }

  for (auto & tri : mesh.elements) {
    tri[0] = new_ids[tri[0]];
    tri[1] = new_ids[tri[1]];
    tri[2] = new_ids[tri[2]];
  }

  stopwatch.stop();

  if (print_timings) {
    std::cout << "evaluating sdf over grid: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

}

void sample_implicit_function(SimplexMesh<2> &mesh, 
            std::vector<float> &values,
            std::vector< uint64_t > &partially_inside,
            const BackgroundGrid<2> & grid,
            int num_threads,
            bool boundary_only)
{

  static constexpr int block_size = 4;

  const int nx = grid.n[0];
  const int ny = grid.n[1];

  static constexpr int tris_per_tile = 8;
  static constexpr int tile[tris_per_tile][3] = {
    {0, 1, 3}, {1, 4, 3}, {1, 2, 4}, {2, 5, 4},
    {3, 7, 6}, {3, 4, 7}, {4, 8, 7}, {4, 5, 8}
  };

  num_threads = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;

  timer stopwatch;
  stopwatch.start();

  std::unordered_map < uint64_t, float > vertex_info;

  for (int ty = 0; ty < ny; ty += block_size) {
    for (int tx = 0; tx < nx; tx += block_size) {
      
      float block_min_value = 1.0e10;
      float block_values[2 * block_size + 1][2 * block_size + 1]; 

      // evaluate the implicit function at each of the vertices in the local block
      for (int dy = 0; dy <= 2 * block_size; dy++) {
        int vy = 2 * ty + dy;
        for (int dx = 0; dx <= 2 * block_size; dx++) {
          int vx = 2 * tx + dx;

          // don't evaluate `f` outside of the sampling domain
          if ((vx <= 2 * nx) && (vy <= 2 * ny)) {
            float value = grid({vx, vy});
            block_values[dx][dy] = value;
            block_min_value = std::min(block_min_value, value);
          }

        }
      }

      // there is no work to be done if the entire block is outside
      if (block_min_value >= 0.0) continue;

      // for each tile in this block, identify triangles
      // that are either partially or entirely inside
      bool keep[2 * block_size + 1][2 * block_size + 1]{}; 
      for (int by = 0; by < block_size; by++) {
        for (int bx = 0; bx < block_size; bx++) {

          if (tx + bx >= nx || ty + by >= ny) continue;

          float tile_values[9];
          uint64_t tile_ids[9];
          for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
              tile_values[j * 3 + i] = block_values[2 * bx + i][2 * by + j];
              tile_ids[j * 3 + i] = grid.vertex_id({2 * (tx + bx) + i, 2 * (ty + by)+ j});
            }
          }

          for (auto tri : tile) {
            std::array tri_values = {tile_values[tri[0]], tile_values[tri[1]], tile_values[tri[2]]};
            auto [min_value, max_value] = min_max(tri_values);

            // discard triangles that are entirely outside
            if (min_value >= 0.0) continue;

            // note which triangles are partially inside
            if (min_value * max_value <= 0) {
              partially_inside.push_back(mesh.elements.size());
            }

            mesh.elements.push_back({tile_ids[tri[0]], tile_ids[tri[1]], tile_ids[tri[2]]});

            // mark which vertices in the tile belong to at least one active triangle
            keep[2 * bx + (tri[0] % 3)][2 * by + (tri[0] / 3)] = true;
            keep[2 * bx + (tri[1] % 3)][2 * by + (tri[1] / 3)] = true;
            keep[2 * bx + (tri[2] % 3)][2 * by + (tri[2] / 3)] = true;
          }

        }
      }

      for (int dy = 0; dy <= 2 * block_size; dy++) {
        int vy = 2 * ty + dy;
        for (int dx = 0; dx <= 2 * block_size; dx++) {
          int vx = 2 * tx + dx;
          if ((vx <= 2 * nx) && (vy <= 2 * ny)) {
            if (keep[dx][dy]) {
              vertex_info[grid.vertex_id({vx, vy})] = block_values[dx][dy];
            }
          }
        }
      }

    }
  }

  values.resize(vertex_info.size());
  mesh.vertices.resize(vertex_info.size());
  std::unordered_map< uint64_t, uint64_t > new_ids(vertex_info.size());

  uint64_t new_id = 0;
  for (auto [vertex_id, value] : vertex_info) {
    values[new_id] = value;
    int32_t i = vertex_id % (2 * nx + 1); 
    int32_t j = vertex_id / (2 * nx + 1); 
    vec2f v = grid.vertex({i, j});
    mesh.vertices[new_id] = {v[0], v[1], 0.0f};
    new_ids[vertex_id] = new_id++;
  }

  for (auto & tri : mesh.elements) {
    tri[0] = new_ids[tri[0]];
    tri[1] = new_ids[tri[1]];
    tri[2] = new_ids[tri[2]];
  }

  stopwatch.stop();

  if (print_timings) {
    std::cout << "evaluating sdf over grid: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

}

template < int dim >
void combine(SimplexMesh< dim > & combined_mesh,
             const std::vector< SimplexMesh< dim > > & meshes, 
             std::vector< float > & combined_vertex_values,
             const std::vector< std::vector< float > > & vertex_values,
             std::vector< uint64_t > & combined_partially_inside,
             const std::vector< std::vector< uint64_t > > & partially_inside,
             AABB<3> bounds, int num_threads) {

  threadpool pool(num_threads);

  // figure out how many vertices / elements 
  // there will be in the combined mesh
  uint64_t total_vertices = 0;
  uint64_t total_elements = 0;
  uint64_t total_partials = 0;
  std::vector< uint64_t > voffsets;
  std::vector< uint64_t > eoffsets;
  std::vector< uint64_t > poffsets;
  for (auto & mesh : meshes) {
    voffsets.push_back(total_vertices);
    total_vertices += mesh.vertices.size();

    eoffsets.push_back(total_elements);
    total_elements += mesh.elements.size();
  }

  if (total_vertices == 0 || total_elements == 0) return;

  for (int i = 0; i < partially_inside.size(); i++) {
    poffsets.push_back(total_partials);
    total_partials += partially_inside[i].size();
  }

  // combine the vertex arrays into a single std::vector
  std::vector< vec3f > all_vertices;
  all_vertices.reserve(total_vertices);
  std::vector< float > all_values;
  all_values.reserve(total_vertices);
  for (uint32_t i = 0; i < meshes.size(); i++) {
    all_vertices.insert(all_vertices.end(), meshes[i].vertices.begin(), meshes[i].vertices.end());
    all_values.insert(all_values.end(), vertex_values[i].begin(), vertex_values[i].end());
  }

  // calculate a morton code for each vertex
  // this accomplishes two things:
  //   1. makes it easy to detect duplicated vertices
  //   2. spatial ordering improves data locality
  vec3f scale = 1.0f / (bounds.max - bounds.min);
  std::vector< std::array< uint64_t, 2 > > morton_code_ids(total_vertices);
  pool.parallel_for(total_vertices, [&](uint64_t i, uint32_t /*tid*/){
    vec3f u = (all_vertices[i] - bounds.min) * scale;

    //if (u[0] < 0 || u[0] > 1.0f || 
    //    u[1] < 0 || u[1] > 1.0f || 
    //    u[2] < 0 || u[2] > 1.0f) {
    //  std::cout << i << " " << u << std::endl;
    //}

    if constexpr (dim == 2) {
      morton_code_ids[i] = {morton::encode({u(0), u(1)}), i}; 
    } else {
      morton_code_ids[i] = {morton::encode({u(0), u(1), u(2)}), i};
    }
  });

  parasort(morton_code_ids.size(), &morton_code_ids[0], num_threads);

  // scan the sorted list of morton codes for duplicates and remove them
  std::vector< uint64_t > new_ids(all_vertices.size());
  new_ids[0] = 0;
  uint64_t id = 0;
  for (uint64_t i = 1; i < morton_code_ids.size(); i++) {
    if (morton_code_ids[i][0] != morton_code_ids[i-1][0]) {
      id++;
    }
    new_ids[morton_code_ids[i][1]] = id;
  }

  // copy the deduplicated nodes into an appropriately sized container
  combined_mesh.vertices = std::vector< vec3f >(id+1);
  combined_vertex_values = std::vector< float >(id+1);
  pool.parallel_for(morton_code_ids.size(), [&](uint64_t i, uint32_t /*tid*/){
    combined_mesh.vertices[new_ids[i]] = all_vertices[i];
    combined_vertex_values[new_ids[i]] = all_values[i];
  });

  // combine the element lists, and update their vertex ids
  combined_mesh.elements = std::vector< std::array< uint64_t, dim + 1 > >(total_elements);
  combined_partially_inside = std::vector< uint64_t >(total_partials);
  pool.parallel_for(meshes.size(), [&](uint64_t i, uint32_t /*tid*/){

    auto & mesh = meshes[i];
    for (int j = 0; j < mesh.elements.size(); j++) {
      auto elem = mesh.elements[j];
      for (auto & id : elem) { 
        id = new_ids[id + voffsets[i]]; 
      }
      combined_mesh.elements[eoffsets[i] + j] = elem;
    }

    for (int j = 0; j < partially_inside[i].size(); j++) {
      combined_partially_inside[poffsets[i] + j] = partially_inside[i][j] + eoffsets[i];
    }

  });

}

void sample_implicit_function(SimplexMesh<3> &mesh, 
            std::vector<float> &values,
            std::vector< uint64_t > &partially_inside,
            const std::function<float(vec3f)> &f,
            float cell_size,
            AABB<3> bounds,
            int num_threads,
            bool boundary_only)
{

  static constexpr int dim = 3;
  static constexpr int block_size = 3;
  const int nx = floor((bounds.max[0] - bounds.min[0]) / cell_size);
  const int ny = floor((bounds.max[1] - bounds.min[1]) / cell_size);
  const int nz = floor((bounds.max[2] - bounds.min[2]) / cell_size);

  // vertex locations for A15 lattice structure
  auto gridpoint = [cell_size, offset = bounds.min](int i, int j, int k) {
    return vec3f{0.5f * i + 0.25f * ((k % 2 == 0) * (j % 2 == 0)),
                 0.5f * j + 0.25f * ((k % 2 == 1) * (i % 2 == 1)),
                 0.5f * k + 0.25f * ((j % 2 == 1) * (i % 2 == 0))} * cell_size + offset;
  };

  // connectivity of A15 lattice structure
  static constexpr int tets_per_tile = 46;
  static constexpr int tile[tets_per_tile][4] = {
    {12, 10, 18, 9},  {18, 12, 21, 22}, {10, 12, 18, 22}, {19, 10, 18, 22},
    {13, 12, 10, 22}, {14, 19, 20, 23}, {19, 14, 10, 22}, {14, 19, 10, 11},
    {14, 13, 10, 22}, {19, 14, 20, 11}, {19, 23, 14, 22}, {0, 1, 4, 10},
    {0, 10, 3, 9},    {3, 10, 12, 9},   {12, 13, 10, 3},  {4, 3, 0, 10},
    {13, 3, 4, 10},   {10, 5, 1, 4},    {5, 10, 1, 11},   {5, 13, 4, 10},
    {10, 14, 11, 5},  {2, 11, 5, 1},    {13, 5, 14, 10},  {16, 25, 24, 13},
    {24, 12, 22, 21}, {13, 25, 24, 22}, {13, 12, 24, 15}, {12, 13, 24, 22},
    {24, 16, 13, 15}, {25, 13, 14, 22}, {25, 23, 22, 14}, {23, 25, 26, 14},
    {14, 25, 26, 17}, {25, 16, 17, 13}, {13, 25, 14, 17}, {13, 12, 15, 3},
    {6, 16, 15, 13},  {3, 6, 13, 4},    {6, 3, 13, 15},   {6, 7, 16, 13},
    {7, 6, 4, 13},    {7, 16, 13, 17},  {13, 5, 4, 7},    {7, 13, 5, 17},
    {13, 14, 5, 17},  {17, 8, 5, 7}
  };

  auto vertex_id = [&](uint64_t i, uint64_t j, uint64_t k) { 
    #if BOUNDS_CHECKING
    if (i >= (2 * nx + 1) || j >= (2 * ny + 1) || k > (2 * nz + 1)) {
      std::cout << "out of bounds" << std::endl;
    }
    #endif

    return (k * (2 * ny + 1) + j) * (2 * nx + 1) + i;
  };

  num_threads = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;

  timer stopwatch;
  stopwatch.start();

  std::vector< SimplexMesh< dim > > thr_meshes(num_threads);
  std::vector< std::vector< float > > thr_vertex_values(num_threads);
  std::vector< std::vector< uint64_t > > thr_partially_inside(num_threads);

  threadpool pool(num_threads);

  int x_blocks = (nx + block_size - 1) / block_size;
  int y_blocks = (ny + block_size - 1) / block_size;
  int z_blocks = (nz + block_size - 1) / block_size;

  pool.parallel_for(x_blocks, y_blocks, z_blocks,
    [&](int ix, int iy, int iz) {

    int tid = ix % num_threads;

    int tx = ix * block_size;
    int ty = iy * block_size;
    int tz = iz * block_size;

    float block_min_value = +1.0e10;
    float block_max_value = -1.0e10;
    float block_values[2 * block_size + 1][2 * block_size + 1][2 * block_size + 1];

    // evaluate the implicit function at each of the vertices in the local block
    for (int dz = 0; dz <= 2 * block_size; dz++) {
      int vz = 2 * tz + dz;
      for (int dy = 0; dy <= 2 * block_size; dy++) {
        int vy = 2 * ty + dy;
        for (int dx = 0; dx <= 2 * block_size; dx++) {
          int vx = 2 * tx + dx;

          // don't evaluate `f` outside of the sampling domain
          if ((vx <= 2 * nx) && (vy <= 2 * ny) && (vz <= 2 * nz)) {
            float value = f(gridpoint(vx, vy, vz));
            block_values[dx][dy][dz] = value;
            block_min_value = std::min(block_min_value, value);
            block_max_value = std::max(block_max_value, value);
          }
        }
      }
    }

    // exit early if the entire block is outside or doesn't contain the boundary
    if (!boundary_only && block_min_value >= 0.0) return;
    if (boundary_only && (block_min_value > 0.0 || block_max_value < 0.0)) return;

    // for each tile in this block, identify triangles/tetrahedra
    // that are either partially or entirely inside
    bool keep[2 * block_size + 1][2 * block_size + 1][2 * block_size + 1]{};
    uint64_t block_ids[2 * block_size + 1][2 * block_size + 1][2 * block_size + 1]{};

    for (int bz = 0; bz < block_size; bz++) {
      for (int by = 0; by < block_size; by++) {
        for (int bx = 0; bx < block_size; bx++) {
          if ((tx + bx >= nx) || (ty + by >= ny) || (tz + bz >= nz))
            continue;

          float tile_values[27];
          uint64_t tile_ids[27];
          for (int k = 0; k < 3; k++) {
            for (int j = 0; j < 3; j++) {
              for (int i = 0; i < 3; i++) {
                tile_values[(k * 3 + j) * 3 + i] =
                    block_values[2 * bx + i][2 * by + j][2 * bz + k];
                tile_ids[(k * 3 + j) * 3 + i] =
                    vertex_id(2 * (tx + bx) + i, 2 * (ty + by) + j,
                              2 * (tz + bz) + k);
              }
            }
          }

          for (auto tet : tile) {
            std::array tet_values = {
                tile_values[tet[0]], tile_values[tet[1]],
                tile_values[tet[2]], tile_values[tet[3]]};
            auto [min_value, max_value] = min_max(tet_values);

            // discard tets that are entirely outside
            if (!boundary_only && min_value >= 0.0) continue;
            if (boundary_only && (min_value > 0.0 || max_value < 0.0)) continue;

            // note which tets are partially inside
            if (min_value * max_value <= 0) {
              thr_partially_inside[tid].push_back(thr_meshes[tid].elements.size());
            }

            std::array<uint64_t, 4> tet_ids;
            for (int i = 0; i < 4; i++) {
              int dx = tet[i] % 3;
              int dy = (tet[i] % 9) / 3;
              int dz = tet[i] / 9;

              if (!keep[2 * bx + dx][2 * by + dy][2 * bz + dz]) {
                keep[2 * bx + dx][2 * by + dy][2 * bz + dz] = true;
                block_ids[2 * bx + dx][2 * by + dy][2 * bz + dz] = thr_meshes[tid].vertices.size();
                thr_vertex_values[tid].push_back(tet_values[i]);
                thr_meshes[tid].vertices.push_back(gridpoint(2 * (tx + bx) + dx, 
                                                             2 * (ty + by) + dy,
                                                             2 * (tz + bz) + dz));
              } 

              tet_ids[i] = block_ids[2 * bx + dx][2 * by + dy][2 * bz + dz];
            }

            thr_meshes[tid].elements.push_back(tet_ids);

          }
        }
      }
    }

  });

  stopwatch.stop();

  if (print_timings) {
    std::cout << "sdf evaluation: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

  stopwatch.start();

  combine(mesh, thr_meshes, 
          values, thr_vertex_values, 
          partially_inside, thr_partially_inside, 
          bounds, num_threads);

  stopwatch.stop();

  if (print_timings) {
    std::cout << "combining: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

}

void sample_implicit_function(SimplexMesh<3> &mesh, 
            std::vector<float> &values,
            std::vector< uint64_t > &partially_inside,
            const BackgroundGrid<3> & grid,
            int num_threads,
            bool boundary_only)
{

  static constexpr int dim = 3;
  static constexpr int block_size = 3;

  const int nx = grid.n[0];
  const int ny = grid.n[1];
  const int nz = grid.n[2];

  // connectivity of A15 lattice structure
  static constexpr int tets_per_tile = 46;
  static constexpr int tile[tets_per_tile][4] = {
    {12, 10, 18, 9},  {18, 12, 21, 22}, {10, 12, 18, 22}, {19, 10, 18, 22},
    {13, 12, 10, 22}, {14, 19, 20, 23}, {19, 14, 10, 22}, {14, 19, 10, 11},
    {14, 13, 10, 22}, {19, 14, 20, 11}, {19, 23, 14, 22}, {0, 1, 4, 10},
    {0, 10, 3, 9},    {3, 10, 12, 9},   {12, 13, 10, 3},  {4, 3, 0, 10},
    {13, 3, 4, 10},   {10, 5, 1, 4},    {5, 10, 1, 11},   {5, 13, 4, 10},
    {10, 14, 11, 5},  {2, 11, 5, 1},    {13, 5, 14, 10},  {16, 25, 24, 13},
    {24, 12, 22, 21}, {13, 25, 24, 22}, {13, 12, 24, 15}, {12, 13, 24, 22},
    {24, 16, 13, 15}, {25, 13, 14, 22}, {25, 23, 22, 14}, {23, 25, 26, 14},
    {14, 25, 26, 17}, {25, 16, 17, 13}, {13, 25, 14, 17}, {13, 12, 15, 3},
    {6, 16, 15, 13},  {3, 6, 13, 4},    {6, 3, 13, 15},   {6, 7, 16, 13},
    {7, 6, 4, 13},    {7, 16, 13, 17},  {13, 5, 4, 7},    {7, 13, 5, 17},
    {13, 14, 5, 17},  {17, 8, 5, 7}
  };

  auto vertex_id = [&](uint64_t i, uint64_t j, uint64_t k) { 
    #if BOUNDS_CHECKING
    if (i >= (2 * nx + 1) || j >= (2 * ny + 1) || k > (2 * nz + 1)) {
      std::cout << "out of bounds" << std::endl;
    }
    #endif

    return (k * (2 * ny + 1) + j) * (2 * nx + 1) + i;
  };

  num_threads = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;

  timer stopwatch;
  stopwatch.start();

  std::vector< SimplexMesh< dim > > thr_meshes(num_threads);
  std::vector< std::vector< float > > thr_vertex_values(num_threads);
  std::vector< std::vector< uint64_t > > thr_partially_inside(num_threads);

  threadpool pool(num_threads);

  int x_blocks = (nx + block_size - 1) / block_size;
  int y_blocks = (ny + block_size - 1) / block_size;
  int z_blocks = (nz + block_size - 1) / block_size;

  pool.parallel_for(x_blocks, y_blocks, z_blocks,
    [&](int ix, int iy, int iz) {

    int tid = ix % num_threads;

    int tx = ix * block_size;
    int ty = iy * block_size;
    int tz = iz * block_size;

    float block_min_value = +1.0e10;
    float block_max_value = -1.0e10;
    float block_values[2 * block_size + 1][2 * block_size + 1][2 * block_size + 1];

    // evaluate the implicit function at each of the vertices in the local block
    for (int dz = 0; dz <= 2 * block_size; dz++) {
      int vz = 2 * tz + dz;
      for (int dy = 0; dy <= 2 * block_size; dy++) {
        int vy = 2 * ty + dy;
        for (int dx = 0; dx <= 2 * block_size; dx++) {
          int vx = 2 * tx + dx;

          // don't evaluate `f` outside of the sampling domain
          if ((vx <= 2 * nx) && (vy <= 2 * ny) && (vz <= 2 * nz)) {
            float value = grid({vx, vy, vz});
            block_values[dx][dy][dz] = value;
            block_min_value = std::min(block_min_value, value);
            block_max_value = std::max(block_max_value, value);
          }
        }
      }
    }

    // exit early if the entire block is outside or doesn't contain the boundary
    if (!boundary_only && block_min_value >= 0.0) return;
    if (boundary_only && (block_min_value > 0.0 || block_max_value < 0.0)) return;

    // for each tile in this block, identify triangles/tetrahedra
    // that are either partially or entirely inside
    bool keep[2 * block_size + 1][2 * block_size + 1][2 * block_size + 1]{};
    uint64_t block_ids[2 * block_size + 1][2 * block_size + 1][2 * block_size + 1]{};

    for (int bz = 0; bz < block_size; bz++) {
      for (int by = 0; by < block_size; by++) {
        for (int bx = 0; bx < block_size; bx++) {
          if ((tx + bx >= nx) || (ty + by >= ny) || (tz + bz >= nz))
            continue;

          float tile_values[27];
          uint64_t tile_ids[27];
          for (int k = 0; k < 3; k++) {
            for (int j = 0; j < 3; j++) {
              for (int i = 0; i < 3; i++) {
                tile_values[(k * 3 + j) * 3 + i] =
                    block_values[2 * bx + i][2 * by + j][2 * bz + k];
                tile_ids[(k * 3 + j) * 3 + i] =
                    vertex_id(2 * (tx + bx) + i, 2 * (ty + by) + j,
                              2 * (tz + bz) + k);
              }
            }
          }

          for (auto tet : tile) {
            std::array tet_values = {
                tile_values[tet[0]], tile_values[tet[1]],
                tile_values[tet[2]], tile_values[tet[3]]};
            auto [min_value, max_value] = min_max(tet_values);

            // discard tets that are entirely outside
            if (!boundary_only && min_value >= 0.0) continue;
            if (boundary_only && (min_value > 0.0 || max_value < 0.0)) continue;

            // note which tets are partially inside
            if (min_value * max_value <= 0) {
              thr_partially_inside[tid].push_back(thr_meshes[tid].elements.size());
            }

            std::array<uint64_t, 4> tet_ids;
            for (int i = 0; i < 4; i++) {
              int dx = tet[i] % 3;
              int dy = (tet[i] % 9) / 3;
              int dz = tet[i] / 9;

              if (!keep[2 * bx + dx][2 * by + dy][2 * bz + dz]) {
                keep[2 * bx + dx][2 * by + dy][2 * bz + dz] = true;
                block_ids[2 * bx + dx][2 * by + dy][2 * bz + dz] = thr_meshes[tid].vertices.size();
                thr_vertex_values[tid].push_back(tet_values[i]);
                thr_meshes[tid].vertices.push_back(grid.vertex({2 * (tx + bx) + dx, 
                                                                2 * (ty + by) + dy,
                                                                2 * (tz + bz) + dz}));
              } 

              tet_ids[i] = block_ids[2 * bx + dx][2 * by + dy][2 * bz + dz];
            }

            thr_meshes[tid].elements.push_back(tet_ids);

          }
        }
      }
    }

  });

  stopwatch.stop();

  if (print_timings) {
    std::cout << "sdf evaluation: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

  stopwatch.start();

  combine(mesh, thr_meshes, 
          values, thr_vertex_values, 
          partially_inside, thr_partially_inside, 
          grid.bounds, num_threads);

  stopwatch.stop();

  if (print_timings) {
    std::cout << "combining: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

}

template < int dim >
constexpr auto boundary_elem_ids() {
  if constexpr (dim == 2) {
    return std::array< std::array< uint64_t, 2 >, 3 >{{{0, 1}, {1, 2}, {2, 0}}};
  }
  if constexpr (dim == 3) {
    return std::array< std::array< uint64_t, 3 >, 4 >{{{0, 2, 1}, {1, 2, 3}, {2, 0, 3}, {0, 1, 3}}};
  }
}

constexpr auto reverse(const std::array< uint64_t, 2 > & arr) {
  return std::array< uint64_t, 2 >{arr[1], arr[0]};
}

constexpr auto reverse(const std::array< uint64_t, 3 > & arr) {
  return std::array< uint64_t, 3 >{arr[2], arr[1], arr[0]};
}

template < int dim >
void snap_vertices_to_boundary(SimplexMesh<dim> & mesh, std::vector<float> & values, const std::vector< uint64_t> & partially_inside, float max_snap_distance, int num_threads, bool boundary_only) {

  timer stopwatch;
  stopwatch.start();

  std::vector < float > updated_values = values;
  std::vector < Slice<dim> > slices(partially_inside.size());
  std::vector< vec3f > displacements(mesh.vertices.size(), vec3f{});

  threadpool pool(num_threads);
  constexpr int nmutex = 128;
  std::mutex mtx[nmutex];

  // move interior vertices 
  pool.parallel_for(partially_inside.size(), [&](uint32_t i, uint32_t /*tid*/) {

    auto element = mesh.elements[partially_inside[i]];

    Simplex<dim> s;
    float local_values[dim + 1];

    for (int j = 0; j < dim + 1; j++) {
      local_values[j] = values[element[j]];
      s.vertices[j] = mesh.vertices[element[j]];
    }

    auto slice = slices[i] = SliceSimplex(s, local_values);

    for (int j = 0; j < dim + 1; j++) {
      if (local_values[j] < 0) {
        vec3f p = s.vertices[j];
        vec3f u = closest_point_projection(slice, p) - p;
        float u_norm = norm(u);

        int which = element[j] % nmutex;
        mtx[which].lock();
        float old_u_norm = norm(displacements[element[j]]);

        bool overwrite = (u_norm < max_snap_distance) &&
                         (u_norm < old_u_norm || old_u_norm == 0.0f);

        if (overwrite) {
          displacements[element[j]] = u;
          updated_values[element[j]] = 0;
        }
        mtx[which].unlock();
      }
    }

  });

  stopwatch.stop();
  if (print_timings) {
    std::cout << "move interior vertices " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  } 

  std::vector < float > final_values = updated_values;
  std::vector < char > keep(mesh.elements.size(), 1);

  stopwatch.start();

  std::vector< std::vector< std::array< uint64_t, dim > > > thr_boundary_elements(num_threads);

  // move exterior vertices 
  pool.parallel_for(partially_inside.size(), [&](uint32_t i, uint32_t tid) {

    auto element = mesh.elements[partially_inside[i]];

    Simplex<dim> s;
    float local_values[dim + 1];

    int num_zeroes = 0;
    float min_value = +1.0e10;
    for (int j = 0; j < dim + 1; j++) {
      local_values[j] = updated_values[element[j]];
      s.vertices[j] = mesh.vertices[element[j]];
      min_value = std::min(min_value, local_values[j]);
      num_zeroes += (local_values[j] == 0);
    }

    if (min_value >= 0.0) {
      keep[partially_inside[i]] = false;

      if (boundary_only && num_zeroes == dim) {
        auto conditionally_insert_boundary_elem = [&](const std::array< uint64_t, dim > & j) {
          if constexpr (dim == 2) {
            if (local_values[j[0]] == 0 && local_values[j[1]] == 0) {
              thr_boundary_elements[tid].push_back({element[j[0]], element[j[1]]});
            }
          }
          if constexpr (dim == 3) {
            if (local_values[j[0]] == 0 && local_values[j[1]] == 0 && local_values[j[2]] == 0) {
              thr_boundary_elements[tid].push_back({element[j[0]], element[j[1]], element[j[2]]});
            }
          }
        };

        for (const auto & belem : boundary_elem_ids<dim>()) {
          conditionally_insert_boundary_elem(belem);
        } 
      }

      return;
    }

    auto slice = slices[i];

    for (int j = 0; j < dim + 1; j++) {
      if (local_values[j] > 0) {
        vec3f p = s.vertices[j];
        vec3f u = closest_point_projection(slice, p) - p;
        float u_norm = norm(u);

        int which = element[j] % nmutex;
        mtx[which].lock();
        float old_u_norm = norm(displacements[element[j]]);

        bool overwrite = (u_norm < old_u_norm || old_u_norm == 0.0f);
        if (overwrite) {
          displacements[element[j]] = u;
          final_values[element[j]] = 0;
        }
        mtx[which].unlock();

        local_values[j] = 0;
      }
    }

  });

  stopwatch.stop();
  if (print_timings) {
    std::cout << "move exterior vertices " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  } 

  stopwatch.start();

  mesh.boundary_elements = combine(thr_boundary_elements);

  // remove elements that were pushed entirely out of the domain
  int index = 0;
  for (uint32_t i = 0; i < keep.size(); i++) {

    auto & element = mesh.elements[i];

    if (keep[i]) {
      int num_zeroes = 0;
      float local_values[dim + 1];
      for (int j = 0; j < dim + 1; j++) {
        local_values[j] = final_values[element[j]];
        num_zeroes += (local_values[j] == 0);
      }

      if (num_zeroes == dim) {
        auto conditionally_insert_boundary_elem = [&](const std::array< uint64_t, dim > & j) {
          if constexpr (dim == 2) {
            if (local_values[j[0]] == 0 && local_values[j[1]] == 0) {
              mesh.boundary_elements.push_back({element[j[0]], element[j[1]]});
            }
          }
          if constexpr (dim == 3) {
            if (local_values[j[0]] == 0 && local_values[j[1]] == 0 && local_values[j[2]] == 0) {
              mesh.boundary_elements.push_back({element[j[0]], element[j[1]], element[j[2]]});
            }
          }
        };

        for (const auto & belem : boundary_elem_ids<dim>()) {
          conditionally_insert_boundary_elem(belem);
        } 
      }
    }

    mesh.elements[index] = mesh.elements[i];
    index += (keep[i] != 0);
  }

  if (boundary_only) {

    mesh.elements.clear();

    stopwatch.stop();
    if (print_timings) {
      std::cout << "remove exterior elements " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
    } 

    stopwatch.start();

    // remove any vertices that were orphaned by element removal
    std::vector < uint64_t > new_vertex_ids(mesh.vertices.size(), 0);

    for (const auto & bdr_element : mesh.boundary_elements) {
      for (auto i : bdr_element) {
        new_vertex_ids[i] = 1; // mark the vertices to keep
      }
    }

    index = 0;
    for (uint64_t i = 0; i < mesh.vertices.size(); i++) {
      if (new_vertex_ids[i] == 1) {
        mesh.vertices[index] = mesh.vertices[i] + displacements[i];
        new_vertex_ids[i] = index++;
      }
    }
    mesh.vertices.erase(mesh.vertices.begin() + index, mesh.vertices.end());

    stopwatch.stop();
    if (print_timings) {
      std::cout << "remove orphaned vertices " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
    } 

    stopwatch.start();

    // update the mesh connectivity to use the new vertex ids
    for (auto & boundary_element : mesh.boundary_elements) {
      for (int i = 0; i < dim; i++) {
        boundary_element[i] = new_vertex_ids[boundary_element[i]];
      }
    }

    stopwatch.stop();

    if (print_timings) {
      std::cout << "renumbering elements: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
    } 

  } else {

    mesh.elements.erase(mesh.elements.begin() + index, mesh.elements.end());

    stopwatch.stop();
    if (print_timings) {
      std::cout << "remove exterior elements " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
    } 

    stopwatch.start();

    // remove any vertices that were orphaned by element removal
    std::vector < uint64_t > new_vertex_ids(mesh.vertices.size(), 0);

    for (const auto & element : mesh.elements) {
      for (auto i : element) {
        new_vertex_ids[i] = 1; // mark the vertices to keep
      }
    }

    index = 0;
    for (uint64_t i = 0; i < mesh.vertices.size(); i++) {
      if (new_vertex_ids[i] == 1) {
        mesh.vertices[index] = mesh.vertices[i] + displacements[i];
        new_vertex_ids[i] = index++;
      }
    }
    mesh.vertices.erase(mesh.vertices.begin() + index, mesh.vertices.end());

    stopwatch.stop();
    if (print_timings) {
      std::cout << "remove orphaned vertices " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
    } 

    stopwatch.start();

    // update the mesh connectivity to use the new vertex ids
    for (auto & element : mesh.elements) {
      for (int i = 0; i < dim + 1; i++) {
        element[i] = new_vertex_ids[element[i]];
      }
    }

    for (auto & boundary_element : mesh.boundary_elements) {
      for (int i = 0; i < dim; i++) {
        boundary_element[i] = new_vertex_ids[boundary_element[i]];
      }
    }

    stopwatch.stop();

    if (print_timings) {
      std::cout << "renumbering elements: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
    } 
  }

}

vec3f compute_unit_normal(const std::vector< vec3f > & v, const std::array< uint64_t, 2 > & edge) {
  return normalize(cross(v[edge[1]] - v[edge[0]], vec3f{0, 0, 1}));
}

vec3f compute_unit_normal(const std::vector< vec3f > & v, const std::array< uint64_t, 3 > & tri) {
  return normalize(cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]]));
}

template < int dim >
void dvr(SimplexMesh<dim> & mesh,
         float alpha,
         float step,
         int ndvr,
         int num_threads) {

  timer stopwatch;

  stopwatch.start();

  if (ndvr > 0) {

    constexpr int nmutex = 128;
    std::mutex mtx[nmutex];
    
    //std::vector< char > marker(mesh.vertices.size(), 0);
    std::vector< char > marker(mesh.vertices.size(), 1);
    std::vector< vec3f > normals(mesh.vertices.size(), vec3f{});

    // compute surface normals and mark boundary nodes
    for (uint64_t i = 0; i < mesh.boundary_elements.size(); i++) {
      auto bdr_elem = mesh.boundary_elements[i];
      vec3f n = compute_unit_normal(mesh.vertices, bdr_elem);
      for (uint32_t j = 0; j < dim; j++) {
        normals[bdr_elem[j]] += n;
        marker[bdr_elem[j]] = 1;
      }
    }

    for (uint64_t i = 0; i < mesh.vertices.size(); i++) {
      normals[i] = normalize(normals[i]);
    }

    //// propagate markers to nodes neighboring the boundary
    //for (uint64_t i = 0; i < mesh.elements.size(); i++) {
    //  char value = 0;
    //  for (uint32_t j = 0; j < dim + 1; j++) {
    //    std::cout << int(marker[mesh.elements[i][j]]) << " ";
    //    value = std::max(value, marker[mesh.elements[i][j]]);
    //  }
    //  std::cout << int(value) << std::endl;
    //  for (uint32_t j = 0; j < dim + 1; j++) {
    //    marker[mesh.elements[i][j]] = value;
    //  }
    //}

    // select the elements that contain at least one marked node
    std::vector< uint64_t > active_set;
    for (uint64_t i = 0; i < mesh.elements.size(); i++) {
      char value = 0;
      for (uint32_t j = 0; j < dim + 1; j++) {
        value = std::max(value, marker[mesh.elements[i][j]]);
      }
      if (value > 0) {
        active_set.push_back(i);
      }
    }

    threadpool pool(num_threads);
    std::vector< std::thread > threads;

    for (int k = 0; k < ndvr; k++) {

      std::vector< float > scale(mesh.vertices.size(), 0.0);
      std::vector< vec3f > grad(mesh.vertices.size(), vec3f{});

      pool.parallel_for(active_set.size(), [&](uint32_t i, uint32_t /*tid*/) {
        auto elem_ids = mesh.elements[active_set[i]];

        Simplex<dim> elem;
        for (int j = 0; j < (dim + 1); j++) {
          elem.vertices[j] = mesh.vertices[elem_ids[j]];
        }

        auto [Q, dQdX] = quality_and_gradient(elem);

        float expQ = expf(-alpha * Q);

        dQdX *= expQ;
        for (int j = 0; j < (dim + 1); j++) {
          vec3f g = vec3f{dQdX[j][0], dQdX[j][1], (dim == 2) ? 0.0f : dQdX[j][2]};

          int which = elem_ids[j] % nmutex;
          mtx[which].lock();
          grad[elem_ids[j]] += g;
          scale[elem_ids[j]] += expQ;
          mtx[which].unlock();
        }
      });

      // does this benefit from multiple threads (?)
      for (uint64_t i = 0; i < grad.size(); i++) {
        if (scale[i] != 0.0) {
          vec3f n = normals[i];
          vec3f g = grad[i];
          vec3f u = g - dot(g, n) * n;

          mesh.vertices[i] += step * u / scale[i];
        }
      }

    }

  }

  stopwatch.stop();

  if (print_timings) {
    std::cout << "dvr iterations: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  } 

}

template < int dim >
SimplexMesh<dim> universal_mesh_impl(const std::function<float(vec<dim,float>)>& f,
                                     float cell_size,
                                     AABB<3> bounds,
                                     float threshold,
                                     float dvr_step,
                                     int ndvr,
                                     int num_threads,
                                     bool boundary_only) {

  // allocate enough threads to saturate the hardware, unless otherwise specified
  num_threads = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;

  SimplexMesh< dim > mesh;
  std::vector< float > values;
  std::vector< uint64_t > partially_inside;

  // sample the domain specified by `bounds` and return a mesh
  // of the elements that are either entirely or partially inside
  sample_implicit_function(mesh, values, partially_inside, f, cell_size, bounds, num_threads, boundary_only);

  // move the vertices that are already "close" to the boundary
  // onto the (approximate) boundary by performing a discrete closest-point projection
  const float max_snap_distance = threshold * (0.5f * cell_size);
  snap_vertices_to_boundary(mesh, values, partially_inside, max_snap_distance, num_threads, boundary_only);

  // iteratively move vertices to improve element quality
  const float step = dvr_step * cell_size * cell_size;
  dvr(mesh, 8.0f, step, ndvr, num_threads);

  return mesh;

}

template < int dim >
SimplexMesh<dim> universal_mesh_impl(const BackgroundGrid<dim> & grid,
                                     float threshold,
                                     float dvr_step,
                                     int ndvr,
                                     int num_threads,
                                     bool boundary_only) {

  // allocate enough threads to saturate the hardware, unless otherwise specified
  num_threads = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;

  SimplexMesh< dim > mesh;
  std::vector< float > values;
  std::vector< uint64_t > partially_inside;

  // sample the domain specified by `bounds` and return a mesh
  // of the elements that are either entirely or partially inside
  sample_implicit_function(mesh, values, partially_inside, grid, num_threads, boundary_only);

  // move the vertices that are already "close" to the boundary
  // onto the (approximate) boundary by performing a discrete closest-point projection
  float cell_size = grid.cell_size();
  const float max_snap_distance = threshold * (0.5f * cell_size);
  snap_vertices_to_boundary(mesh, values, partially_inside, max_snap_distance, num_threads, boundary_only);

  // iteratively move vertices to improve element quality
  const float step = dvr_step * cell_size * cell_size;
  dvr(mesh, 8.0f, step, ndvr, num_threads);

  return mesh;

}

SimplexMesh<2> universal_mesh(const std::function<float(vec2f)>& sdf,
                                     float cell_size,
                                     AABB<3> bounds,
                                     float threshold,
                                     float dvr_step,
                                     int ndvr,
                                     int num_threads) {

  bool boundary_only = false;
  return universal_mesh_impl<2>(sdf, cell_size, bounds, threshold, dvr_step, ndvr, num_threads, boundary_only);

}

SimplexMesh<3> universal_mesh(const std::function<float(vec3f)>& sdf,
                                     float cell_size,
                                     AABB<3> bounds,
                                     float threshold,
                                     float dvr_step,
                                     int ndvr,
                                     int num_threads) {

  bool boundary_only = false;
  return universal_mesh_impl<3>(sdf, cell_size, bounds, threshold, dvr_step, ndvr, num_threads, boundary_only);

}

SimplexMesh<2> universal_mesh(const BackgroundGrid<2> & grid,
                              float threshold, 
                              float dvr_step,
                              int ndvr,
                              int num_threads) {

  bool boundary_only = false;
  return universal_mesh_impl<2>(grid, threshold, dvr_step, ndvr, num_threads, boundary_only);

};

SimplexMesh<3> universal_mesh(const BackgroundGrid<3> & grid,
                              float threshold, 
                              float dvr_step,
                              int ndvr,
                              int num_threads) {

  bool boundary_only = false;
  return universal_mesh_impl<3>(grid, threshold, dvr_step, ndvr, num_threads, boundary_only);

};

SimplexMesh<2> universal_boundary_mesh(const std::function<float(vec2f)>& sdf,
                                     float cell_size,
                                     AABB<3> bounds,
                                     float threshold,
                                     float dvr_step,
                                     int ndvr,
                                     int num_threads) {

  bool boundary_only = true;
  return universal_mesh_impl<2>(sdf, cell_size, bounds, threshold, dvr_step, ndvr, num_threads, boundary_only);

}

SimplexMesh<3> universal_boundary_mesh(const std::function<float(vec3f)>& sdf,
                                     float cell_size,
                                     AABB<3> bounds,
                                     float threshold,
                                     float dvr_step,
                                     int ndvr,
                                     int num_threads) {

  bool boundary_only = true;
  return universal_mesh_impl<3>(sdf, cell_size, bounds, threshold, dvr_step, ndvr, num_threads, boundary_only);

}

#ifdef UM_TIFF_SUPPORT

void sample_images(SimplexMesh<3> &mesh, 
            std::vector<float> &values,
            float threshold,
            std::vector< uint64_t > &partially_inside,
            const std::vector<std::string> tiff_filenames,
            int num_threads)
{

  constexpr int dim = 3;

  if (tiff_filenames.size() < 6) {
    std::cout << "error: more images required to create a mesh" << std::endl;
    exit(1);
  }

  Image images[6];

  // put these in slots 4 and 5 since the 
  // sampling algorithm starts by shifting spots 
  // [4,5]->[0,1] and reading the next 4 tif files
  images[4] = import_tiff(tiff_filenames[0]);
  images[5] = import_tiff(tiff_filenames[1]);

  const uint32_t nx = images[4].height;
  const uint32_t ny = images[4].width;
  const uint32_t nz = tiff_filenames.size();

  // vertex locations for A15 lattice structure
  auto gridpoint = [](int i, int j, int k) {
    return std::array<int, 3>{2 * i + ((k % 2 == 0) * (j % 2 == 0)),
                              2 * j + ((k % 2 == 1) * (i % 2 == 1)),
                              2 * k + ((j % 2 == 1) * (i % 2 == 0))};
  };

  // connectivity of A15 lattice structure
  static constexpr int tets_per_tile = 46;
  static constexpr int tile[tets_per_tile][4] = {
    {12, 10, 18, 9},  {18, 12, 21, 22}, {10, 12, 18, 22}, {19, 10, 18, 22},
    {13, 12, 10, 22}, {14, 19, 20, 23}, {19, 14, 10, 22}, {14, 19, 10, 11},
    {14, 13, 10, 22}, {19, 14, 20, 11}, {19, 23, 14, 22}, {0, 1, 4, 10},
    {0, 10, 3, 9},    {3, 10, 12, 9},   {12, 13, 10, 3},  {4, 3, 0, 10},
    {13, 3, 4, 10},   {10, 5, 1, 4},    {5, 10, 1, 11},   {5, 13, 4, 10},
    {10, 14, 11, 5},  {2, 11, 5, 1},    {13, 5, 14, 10},  {16, 25, 24, 13},
    {24, 12, 22, 21}, {13, 25, 24, 22}, {13, 12, 24, 15}, {12, 13, 24, 22},
    {24, 16, 13, 15}, {25, 13, 14, 22}, {25, 23, 22, 14}, {23, 25, 26, 14},
    {14, 25, 26, 17}, {25, 16, 17, 13}, {13, 25, 14, 17}, {13, 12, 15, 3},
    {6, 16, 15, 13},  {3, 6, 13, 4},    {6, 3, 13, 15},   {6, 7, 16, 13},
    {7, 6, 4, 13},    {7, 16, 13, 17},  {13, 5, 4, 7},    {7, 13, 5, 17},
    {13, 14, 5, 17},  {17, 8, 5, 7}
  };

  auto vertex_id = [&](uint64_t i, uint64_t j, uint64_t k) { 
    return (k * (2 * ny + 1) + j) * (2 * nx + 1) + i;
  };

  num_threads = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;

  constexpr int nmutex = 128;
  std::mutex mtx[nmutex];

  timer stopwatch;
  stopwatch.start();

  std::unordered_map<uint64_t, float> vertex_info[nmutex];

  std::vector < std::vector< uint64_t > > thr_partially_inside(num_threads);
  std::vector < std::vector < std::array< uint64_t, dim+1 > > > thr_elements(num_threads);

  threadpool pool(num_threads);

  int tid = 0;

  for (int iz = 0; iz < nz - 6; iz += 4) {

    images[0] = images[4];
    images[1] = images[5];
    for (int dz = 2; dz < 6; dz++) {
      images[dz] = import_tiff(tiff_filenames[iz + dz]);
    }

    for (int iy = 0; iy < ny - 4; iy += 4) {
      for (int ix = 0; ix < nx - 4; ix += 4) {

        bool keep[27];
        float tile_values[27];
        uint64_t tile_ids[27];
        for (int dz = 0; dz < 3; dz++) {
          for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
              auto [x, y, z] = gridpoint(dx, dy, dz);
              tile_values[(dz * 3 + dy) * 3 + dx] = threshold - images[z](ix + x, iy + y);
              tile_ids   [(dz * 3 + dy) * 3 + dx] = vertex_id(ix + x, iy + y, iz + z);
            }
          }
        }

        for (auto tet : tile) {
          std::array tet_values = {
              tile_values[tet[0]], tile_values[tet[1]],
              tile_values[tet[2]], tile_values[tet[3]]};
          auto [min_value, max_value] = min_max(tet_values);

          // discard tets that are entirely outside
          if (min_value >= 0.0) continue;

          // note which tets are partially inside
          if (min_value * max_value <= 0) {
            thr_partially_inside[tid].push_back(thr_elements[tid].size());
          }

          thr_elements[tid].push_back({tile_ids[tet[0]], tile_ids[tet[1]],
                                       tile_ids[tet[2]], tile_ids[tet[3]]});

          // mark which vertices in the tile belong to at least one active tetrahedron
          for (int i = 0; i < 4; i++) {
            keep[tet[i]] = true;
          }
        }

        for (int i = 0; i < 27; i++) {
          int index = tile_ids[i];
          int which = index % nmutex;
          mtx[which].lock();
          vertex_info[which][index] = tile_values[i];
          mtx[which].unlock();
        }

      }
    }


  }

  stopwatch.stop();

  if (print_timings) {
    std::cout << "sdf evaluation: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

  stopwatch.start();

  uint64_t total_size = 0;
  for (auto container : vertex_info) {
    total_size += container.size();
  }

  values.resize(total_size);
  mesh.vertices.resize(total_size);

  std::vector< uint64_t > offsets(num_threads, 0);
  for (int i = 1; i < num_threads; i++) {
    offsets[i] = offsets[i-1] + thr_elements[i-1].size();
  }

  pool.parallel_for(num_threads, [&](uint64_t i, uint32_t /*tid*/){
    for (auto & idx : thr_partially_inside[i]) {
      idx += offsets[i];
    }
  });

  stopwatch.stop();

  if (print_timings) {
    std::cout << "renumbering: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

  stopwatch.start();

  mesh.elements = combine(thr_elements);
  partially_inside = combine(thr_partially_inside);

  stopwatch.stop();

  if (print_timings) {
    std::cout << "combining: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

  stopwatch.start();

  std::unordered_map< uint64_t, uint64_t > new_ids;

  uint64_t new_id = 0;
  for (auto container : vertex_info) {
    for (auto [vertex_id, value] : container) {
      values[new_id] = value;
      uint32_t i = vertex_id % (2 * nx + 1); 
      uint32_t j = (vertex_id % ((2 * nx + 1) * (2 * ny + 1))) / (2 * nx + 1); 
      uint32_t k = vertex_id / ((2 * nx + 1) * (2 * ny + 1));
      auto p = gridpoint(i,j,k);
      mesh.vertices[new_id] = {float(p[0]), float(p[1]), float(p[2])};
      new_ids[vertex_id] = new_id++;
    }
  }

  stopwatch.stop();

  if (print_timings) {
    std::cout << "filling std::unordered_map: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }

  stopwatch.start();

  pool.parallel_for(mesh.elements.size(), [&](uint64_t i, uint32_t /*tid*/){
    auto & tet = mesh.elements[i];
    tet[0] = new_ids.at(tet[0]);
    tet[1] = new_ids.at(tet[1]);
    tet[2] = new_ids.at(tet[2]);
    tet[3] = new_ids.at(tet[3]);
  });

  stopwatch.stop();

  if (print_timings) {
    std::cout << "renumbering elements: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;
  }
}

SimplexMesh<3> universal_mesh(std::vector< std::string > tiff_filenames,
                              float threshold,
                              float snapping_distance,
                              float dvr_step,
                              int ndvr,
                              int num_threads) {

  // allocate enough threads to saturate the hardware, unless otherwise specified
  num_threads = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;

  SimplexMesh< 3 > mesh;
  std::vector< float > values;
  std::vector< uint64_t > partially_inside;

  sample_images(mesh, values, threshold, partially_inside, tiff_filenames, num_threads);

  // move the vertices that are already "close" to the boundary
  // onto the (approximate) boundary by performing a discrete closest-point projection
  const float cell_size = 4.0f;
  const float max_snap_distance = snapping_distance * (0.5f * cell_size);
  const bool boundary_only = false;
  snap_vertices_to_boundary(mesh, values, partially_inside, max_snap_distance, num_threads, boundary_only);

  // iteratively move vertices to improve element quality
  const float step = dvr_step * cell_size * cell_size;
  dvr(mesh, 8.0f, step, ndvr, num_threads);

  return mesh;

}
#endif

} // namespace geometry