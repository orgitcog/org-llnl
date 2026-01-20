#include "geometry.hpp"

#include <cmath>
#include <tuple>
#include <thread>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include "timer.hpp"
#include "heap_array.hpp"

namespace geometry {

#if 0
static constexpr int A15vertices[27][3] = {
  {1, 0, 4}, {3, 0, 4}, {5, 0, 4}, 
  {0, 0, 2}, {2, 1, 2}, {4, 0, 2}, 
  {1, 0, 0}, {3, 0, 0}, {5, 0, 0}, 
  {0, 2, 5}, {2, 2, 4}, {4, 2, 5}, 
  {0, 2, 3}, {2, 3, 2}, {4, 2, 3}, 
  {0, 2, 1}, {2, 2, 0}, {4, 2, 1}, 
  {1, 4, 4}, {3, 4, 4}, {5, 4, 4}, 
  {0, 4, 2}, {2, 5, 2}, {4, 4, 2}, 
  {1, 4, 0}, {3, 4, 0}, {5, 4, 0}
};

static constexpr int A15Tets[46][4] = {
  {12,  4,  0,  3}, { 0, 12,  9, 10}, { 4, 12,  0, 10}, { 1,  4,  0, 10}, 
  {13, 12,  4, 10}, {14,  1,  2, 11}, { 1, 14,  4, 10}, {14,  1,  4,  5}, 
  {14, 13,  4, 10}, { 1, 14,  2,  5}, { 1, 11, 14, 10}, { 6,  7, 16,  4}, 
  { 6,  4, 15,  3}, {15,  4, 12,  3}, {12, 13,  4, 15}, {16, 15,  6,  4}, 
  {13, 15, 16,  4}, { 4, 17,  7, 16}, {17,  4,  7,  5}, {17, 13, 16,  4}, 
  { 4, 14,  5, 17}, { 8,  5, 17,  7}, {13, 17, 14,  4}, {22, 19, 18, 13}, 
  {18, 12, 10,  9}, {13, 19, 18, 10}, {13, 12, 18, 21}, {12, 13, 18, 10}, 
  {18, 22, 13, 21}, {19, 13, 14, 10}, {19, 11, 10, 14}, {11, 19, 20, 14}, 
  {14, 19, 20, 23}, {19, 22, 23, 13}, {13, 19, 14, 23}, {13, 12, 21, 15}, 
  {24, 22, 21, 13}, {15, 24, 13, 16}, {24, 15, 13, 21}, {24, 25, 22, 13}, 
  {25, 24, 16, 13}, {25, 22, 13, 23}, {13, 17, 16, 25}, {25, 13, 17, 23}, 
  {13, 14, 17, 23}, {23, 26, 17, 25}
};
#endif

inline vec3f GridPoint(int i, int j, int k) {
  return vec3f{i + 0.5f * (k % 2 == 1), j + 0.5f * (k % 2 == 1), 0.5f * k};
}

inline auto GridPoint(std::array<int, 3> i) {
  return GridPoint(i[0], i[1], i[2]);
}

enum class LatticeType { BCC, A15 };

enum class VertexType : uint32_t { Negative, Zero, Positive, Unused };

constexpr VertexType classify(float value) {
  if (value == 0) { return VertexType::Zero; }
  if (value > 0) { return VertexType::Positive; }
  if (value < 0) { return VertexType::Negative; }
  return VertexType::Unused;
};

struct VertexId {
  uint32_t x : 10;
  uint32_t y : 10;
  uint32_t z : 10;
  VertexType type : 2;
  constexpr bool operator<(const VertexId & other) const {
    return std::tuple{type, z, y, x} < std::tuple{other.type, other.z, other.y, other.x};
  }

  constexpr bool operator==(const VertexId & other) const {
    return std::tuple{type, z, y, x} == std::tuple{other.type, other.z, other.y, other.x};
  }

  constexpr bool operator!=(const VertexId & other) const {
    return std::tuple{type, z, y, x} != std::tuple{other.type, other.z, other.y, other.x};
  }

};

constexpr VertexId unused {0, 0, 0, VertexType::Unused};

struct EdgeOrVertexId {
  VertexId ids[2];
  constexpr EdgeOrVertexId() : ids{unused, unused} {}
  constexpr EdgeOrVertexId(VertexId id) : ids{id, unused} {}
  constexpr EdgeOrVertexId(VertexId id0, VertexId id1) : ids{id0, id1} {}
  
  constexpr bool operator<(const EdgeOrVertexId & other) const {
    return std::tuple{ids[0], ids[1]} < std::tuple{other.ids[0], other.ids[1]};
  }

  constexpr bool operator==(const EdgeOrVertexId & other) const {
    return std::tuple{ids[0], ids[1]} == std::tuple{other.ids[0], other.ids[1]};
  }

  constexpr bool operator!=(const EdgeOrVertexId & other) const {
    return std::tuple{ids[0], ids[1]} != std::tuple{other.ids[0], other.ids[1]};
  }
};

void MarchingTetrahedra(VertexId v[4], std::vector< std::array < EdgeOrVertexId, 4 > >& tets);

constexpr int TetrahedraPerCell(LatticeType type) {
  switch (type) {
    case LatticeType::BCC: return 12;
    case LatticeType::A15: return 46;
  }
  return 0;
}

float sign(float val) {
  return (0.0f < val) - (val < 0.0f);
}

float clamp(float val, float minval, float maxval) {
  return std::min(std::max(val, minval), maxval);
}

float Line::SDF(vec3f p) const {
  vec3f t = vertices[1] - vertices[0];
  float h = clamp(dot(p - vertices[0],t)/dot(t,t), 0.0f, 1.0f);
  return norm(p - vertices[0] - t*h);
}

float Ball::SDF(vec3f p) const {
  return norm(p - c) - r;
}

// taken from https://iquilezles.org/articles/distfunctions/
float Capsule::SDF(vec3f p) const {
  // sampling independent computations (only depend on shape)
  vec3f ba = p2 - p1;
  float l2 = norm_squared(ba);
  float rr = r1 - r2;
  float a2 = l2 - rr * rr;
  float il2 = 1.0 / l2;

  // sampling dependant computations
  vec3f pa = p - p1;
  float y = dot(pa, ba);
  float z = y - l2;
  float x2 = norm_squared(pa * l2 - ba * y);
  float y2 = y * y * l2;
  float z2 = z * z * l2;

  // single square root!
  float k = sign(rr) * rr * rr * x2;
  if (sign(z) * a2 * z2 > k)
    return sqrt(x2 + z2) * il2 - r2;
  if (sign(y) * a2 * y2 < k)
    return sqrt(x2 + y2) * il2 - r1;
  return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

float Triangle::SDF(vec3f p) const {
  vec3f ba = vertices[1] - vertices[0]; 
  vec3f cb = vertices[2] - vertices[1]; 
  vec3f ac = vertices[0] - vertices[2]; 
  vec3f pa = p - vertices[0];
  vec3f pb = p - vertices[1];
  vec3f pc = p - vertices[2];
  vec3f n = cross(ba, ac);

  return sqrt(
    (sign(dot(cross(ba,n),pa)) +
     sign(dot(cross(cb,n),pb)) +
     sign(dot(cross(ac,n),pc))<2.0)
     ?
     std::min(std::min(
     norm_squared(ba*clamp(dot(ba,pa)/dot(ba,ba),0.0,1.0)-pa),
     norm_squared(cb*clamp(dot(cb,pb)/dot(cb,cb),0.0,1.0)-pb)),
     norm_squared(ac*clamp(dot(ac,pc)/dot(ac,ac),0.0,1.0)-pc))
     :
     (dot(n,pa)*dot(n,pa))/dot(n,n));
}

Quad::Quad(mat4x3f corners) {
  X = corners;
  N = {
    cross(X[1]-X[0], X[3]-X[0]), 
    cross(X[2]-X[1], X[0]-X[1]), 
    cross(X[3]-X[2], X[1]-X[2]), 
    cross(X[0]-X[3], X[2]-X[3])
  };
}

float Quad::SDF(vec3f p) const {
  return std::min(Line{X[0], X[1]}.SDF(p), 
         std::min(Line{X[1], X[2]}.SDF(p), 
         std::min(Line{X[2], X[3]}.SDF(p), 
         std::min(Line{X[3], X[0]}.SDF(p), interior_SDF(p)))));
}

float Quad::interior_SDF(vec3f p) const {
  vec3f xi = {0.5f, 0.5f, 0.0f};
  vec3f r = bilinear_interpolate(X, vec2f{xi[0], xi[1]}) - p;
  for (int k = 0; k < 6; k++) {
    if (dot(r,r) < 1.0e-10f) {
      break;
    } else {
      vec2f s{xi[0], xi[1]};
      float z = xi[2];
      mat4x3f Y = X + z * N;
      mat3f JT = {d_dxi(Y, s), d_deta(Y, s), bilinear_interpolate(N, s)}; 
      xi -= dot(r, inv(JT));
      xi[0] = clamp(xi[0], 0.0f, 1.0f);
      xi[1] = clamp(xi[1], 0.0f, 1.0f);
      r = bilinear_interpolate(X + xi[2] * N, vec2f{xi[0], xi[1]}) - p;
      
      // TODO: terminate when xi doesn't change
      //std::cout << dot(r,r) << std::endl;
    }
  }
  //std::cout << dot(r,r) << std::endl;
  //std::cout << std::endl;

  return norm(bilinear_interpolate(X, vec2f{xi[0], xi[1]}) - p);
}

vec3f Quad::bilinear_interpolate(mat4x3f Y, vec2f s) const {
  vec2f t = vec2f{1.0f - s[0], 1.0f - s[1]};
  return t[0]*t[1]*Y[0] + s[0]*t[1]*Y[1] + s[0]*s[1]*Y[2] + t[0]*s[1]*Y[3];
}
vec3f Quad::d_dxi(mat4x3f Y, vec2f s) const {
  vec2f t = vec2f{1.0f - s[0], 1.0f - s[1]};
  return t[1]*(Y[1] - Y[0]) + s[1]*(Y[2] - Y[3]);
}

vec3f Quad::d_deta(mat4x3f Y, vec2f s) const {
  vec2f t = vec2f{1.0f - s[0], 1.0f - s[1]};
  return t[0]*(Y[3] - Y[0]) + s[0]*(Y[2] - Y[1]);
}

// taken from from https://iquilezles.org/articles/distfunctions/ and /distfunctions2d/
float RevolvedPolygon::SDF(vec3f p) const {
  vec2f q { sqrtf(p[0]*p[0] + p[2]*p[2]), p[1] };
  float d = dot(q-v[0], q-v[0]);
  float s = 1.0;
  for (std::size_t i=0, j=v.size()-1; i<v.size(); j=i, i++) {
    // distance calculation
    const vec2f e = v[j] - v[i];
    const vec2f w = q    - v[i];
    const float clamped = std::fmax(0.0, std::fmin(dot(w,e)/dot(e,e), 1.0));
    const vec2f b = w - clamped * e;
    d = std::fmin(d, dot(b,b));
    // winding number from http://geomalgorithms.com/a03-_inclusion.html
    const bool cond1 = q[1]      >= v[i][1];
    const bool cond2 = q[1]      <  v[j][1];
    const bool cond3 = e[0]*w[1] >  e[1]*w[0];
    if ((cond1 && cond2 && cond3) || (!cond1 && !cond2 && !cond3)) {
      s *= -1.0;
    }
  }
  return s * sqrt(d);
}

float area(const Triangle & tri) {
  auto e1 = tri.vertices[1] - tri.vertices[0];
  auto e2 = tri.vertices[2] - tri.vertices[0];
  return 0.5 * norm(cross(e1, e2));
}

float quality(const Triangle & tri) {
  float L1 = norm(tri.vertices[1] - tri.vertices[0]);
  float L2 = norm(tri.vertices[2] - tri.vertices[1]);
  float L3 = norm(tri.vertices[0] - tri.vertices[2]);
  float A = 0.5 * norm(cross(tri.vertices[1] - tri.vertices[0], tri.vertices[2] - tri.vertices[0]));
  return (4.0 / sqrt(3.0)) * (A / pow(L1 * L2 * L3, 2.0 / 3.0));
}

float volume(const AABB<3> & box) {
  return (box.max[0] - box.min[0]) * (box.max[1] - box.min[1]) * (box.max[2] - box.min[2]);
}

float volume(const Tetrahedron & tet) {
  auto e1 = tet.vertices[1] - tet.vertices[0];
  auto e2 = tet.vertices[2] - tet.vertices[0];
  auto e3 = tet.vertices[3] - tet.vertices[0];
  return dot(cross(e1, e2), e3) / 6.0;
}

float quality(const Tetrahedron & tet) {
  float L_rms = sqrt((norm_squared(tet.vertices[1] - tet.vertices[0]) + 
                      norm_squared(tet.vertices[2] - tet.vertices[1]) + 
                      norm_squared(tet.vertices[0] - tet.vertices[2]) + 
                      norm_squared(tet.vertices[0] - tet.vertices[3]) + 
                      norm_squared(tet.vertices[1] - tet.vertices[3]) + 
                      norm_squared(tet.vertices[2] - tet.vertices[3])) / 6.0); 
  float V = volume(tet);
  return 6.0 * sqrt(2.0) *  V / (L_rms * L_rms * L_rms);
}

template < typename X >
auto find_root(const X (&x)[2], const float (&f)[2]) {
  if (sign(f[0] * f[1]) >= 0.0) {
    return 0.5f * (x[0] + x[1]);
  } else {
    return (x[1] * f[0] - x[0] * f[1]) / (f[0] - f[1]);
  }
}

vec3f closest_point_projection(const Line & a, const vec3f & p) {
  vec3f delta = a.vertices[1] - a.vertices[0];
  float bot = dot(delta, delta);
  float top = dot(p - a.vertices[0], delta);
  float t = (bot == 0) ? 0.5f : std::min(std::max(top / bot, 0.0f), 1.0f); 
  return (1.0 - t) * a.vertices[0] + t * a.vertices[1];
}

//https://www.shadertoy.com/view/4sXXRN
vec3f closest_point_projection(const Triangle & a, const vec3f & p) {

  vec3f e1 = a.vertices[1] - a.vertices[0];
  vec3f e3 = a.vertices[0] - a.vertices[2];
  vec3f n  = normalize(cross(e3, e1));

  mat3f A = {{
    {e1[0], -e3[0], n[0]},
    {e1[1], -e3[1], n[1]},
    {e1[2], -e3[2], n[2]}
  }};

  vec3f x = dot(inv(A), p - a.vertices[0]);

  float u = x[0];
  float v = x[1];
  float w = 1.0f - u - v;

  // if the point's normal projection 
  // lands inside the triangle
  if (0.0f <= u && u <= 1.0f &&
      0.0f <= v && v <= 1.0f &&
      0.0f <= w && w <= 1.0f) {

    return a.vertices[0] * w + a.vertices[1] * u + a.vertices[2] * v;

  // otherwise, check the distances to
  // the closest edge of the triangle
  } else {

    vec3f q[3] = {
      closest_point_projection(Line{a.vertices[0], a.vertices[1]}, p),
      closest_point_projection(Line{a.vertices[1], a.vertices[2]}, p),
      closest_point_projection(Line{a.vertices[2], a.vertices[0]}, p)
    };

    vec3f closest = q[0];
    if (norm_squared(p - q[1]) < norm_squared(p - closest)) { closest = q[1]; }
    if (norm_squared(p - q[2]) < norm_squared(p - closest)) { closest = q[2]; }

    return closest;
    
  }

}

// map the values {-1, 0, 1} -> {0, 1, 2}
// and then concatenate the digits to make a 3-digit number (base 10)
constexpr uint32_t id(int a, int b, int c) {
  return 10 * (10 * (a + 1) + (b + 1)) + (c + 1);
}

void MarchingTriangles(vec3f p[3], float v[3], std::vector<Line>& lines) {

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
    case id(-1, -1, -1): lines = {}; break;
    case id(-1, -1,  0): lines = {}; break;
    case id(-1, -1, +1): lines = {Line{m1, m2}}; break;
    case id(-1,  0, -1): lines = {}; break;
    case id(-1,  0,  0): lines = {}; break;
    case id(-1,  0, +1): lines = {Line{p[1], m2}}; break;
    case id(-1, +1, -1): lines = {Line{m0, m1}}; break;
    case id(-1, +1,  0): lines = {Line{m0, p[2]}}; break;
    case id(-1, +1, +1): lines = {Line{m0, m2}}; break;

    case id( 0, -1, -1): lines = {}; break;
    case id( 0, -1,  0): lines = {}; break;
    case id( 0, -1, +1): lines = {Line{m1, p[0]}}; break;
    case id( 0,  0, -1): lines = {}; break;
    case id( 0,  0,  0): lines = {}; break;
    case id( 0,  0, +1): lines = {}; break;
    case id( 0, +1, -1): lines = {Line{p[0], m1}}; break;
    case id( 0, +1,  0): lines = {}; break;
    case id( 0, +1, +1): lines = {}; break;

    case id(+1, -1, -1): lines = {Line{m2, m0}}; break;
    case id(+1, -1,  0): lines = {Line{p[2], m0}}; break;
    case id(+1, -1, +1): lines = {Line{m1, m0}}; break;
    case id(+1,  0, -1): lines = {Line{m2, p[1]}}; break;
    case id(+1,  0,  0): lines = {}; break;
    case id(+1,  0, +1): lines = {}; break;
    case id(+1, +1, -1): lines = {Line{m2, m1}}; break;
    case id(+1, +1,  0): lines = {}; break;
    case id(+1, +1, +1): lines = {}; break;
  }
}

// map the values {-1, 0, 1} -> {0, 1, 2}
// and then concatenate the digits to make a 4-digit number (base 10)
constexpr uint32_t id(int a, int b, int c, int d) {
  return 10 * (10 * (10 * (a + 1) + (b + 1)) + (c + 1)) + (d + 1);
}

void MarchingTetrahedra(vec3f p[4], float v[4], std::vector<Triangle>& tris) {
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
      (s == id( 1, +1, +1,  0))) return;

//if (s == id( 0,  0,  0,  0)) return; // is this the right thing to do in this case?

  if (s == id(-1,  0,  0,  0)) tris.push_back({p[1], p[2], p[3]});
  if (s == id( 0, -1,  0,  0)) tris.push_back({p[0], p[3], p[2]});
  if (s == id( 0,  0, -1,  0)) tris.push_back({p[0], p[1], p[3]});
  if (s == id( 0,  0,  0, -1)) tris.push_back({p[2], p[1], p[0]});

  if (s == id(+1,  0,  0,  0)) tris.push_back({p[3], p[2], p[1]});
  if (s == id( 0, +1,  0,  0)) tris.push_back({p[2], p[3], p[0]});
  if (s == id( 0,  0, +1,  0)) tris.push_back({p[3], p[1], p[0]});
  if (s == id( 0,  0,  0, +1)) tris.push_back({p[0], p[1], p[2]});

//if (s == id(-1, -1,  0,  0)) return;
//if (s == id(-1,  0, -1,  0)) return;
//if (s == id(-1,  0,  0, -1)) return;
//if (s == id( 0, -1, -1,  0)) return;
//if (s == id( 0, -1,  0, -1)) return;
//if (s == id( 0,  0, -1, -1)) return;

//if (s == id(+1, +1,  0,  0)) return;
//if (s == id(+1,  0, +1,  0)) return;
//if (s == id(+1,  0,  0, +1)) return;
//if (s == id( 0, +1, +1,  0)) return;
//if (s == id( 0, +1,  0, +1)) return;
//if (s == id( 0,  0, +1, +1)) return;

  vec3f m01 = find_root({p[0], p[1]}, {v[0], v[1]});
  vec3f m02 = find_root({p[0], p[2]}, {v[0], v[2]});
  vec3f m03 = find_root({p[0], p[3]}, {v[0], v[3]});
  vec3f m12 = find_root({p[1], p[2]}, {v[1], v[2]});
  vec3f m13 = find_root({p[1], p[3]}, {v[1], v[3]});
  vec3f m23 = find_root({p[2], p[3]}, {v[2], v[3]});

  if (s == id(+1, -1,  0,  0)) tris.push_back({m01, p[3], p[2]});
  if (s == id(-1, +1,  0,  0)) tris.push_back({m01, p[2], p[3]});
  if (s == id(+1,  0, -1,  0)) tris.push_back({m02, p[1], p[3]});
  if (s == id(-1,  0, +1,  0)) tris.push_back({m02, p[3], p[1]});
  if (s == id(+1,  0,  0, -1)) tris.push_back({m03, p[2], p[1]});
  if (s == id(-1,  0,  0, +1)) tris.push_back({m03, p[1], p[2]});
  if (s == id( 0, +1, -1,  0)) tris.push_back({m12, p[3], p[0]});
  if (s == id( 0, -1, +1,  0)) tris.push_back({m12, p[0], p[3]});
  if (s == id( 0, +1,  0, -1)) tris.push_back({m13, p[0], p[2]});
  if (s == id( 0, -1,  0, +1)) tris.push_back({m13, p[2], p[0]});
  if (s == id( 0,  0, +1, -1)) tris.push_back({m23, p[1], p[0]});
  if (s == id( 0,  0, -1, +1)) tris.push_back({m23, p[0], p[1]});

//if (s == id( 0, -1, -1, -1)) return;
  if (s == id( 0, -1, -1, +1)) tris.push_back({m13, m23, p[0]});
  if (s == id( 0, -1, +1, -1)) tris.push_back({m23, m12, p[0]});
  if (s == id( 0, -1, +1, +1)) tris.push_back({m13, m12, p[0]});
  if (s == id( 0, +1, -1, -1)) tris.push_back({m12, m13, p[0]});
  if (s == id( 0, +1, -1, +1)) tris.push_back({m12, m23, p[0]});
  if (s == id( 0, +1, +1, -1)) tris.push_back({m23, m13, p[0]});
//if (s == id( 0, +1, +1, +1)) return;

//if (s == id(-1,  0, -1, -1)) return;
  if (s == id(-1,  0, -1, +1)) tris.push_back({m23, m03, p[1]});
  if (s == id(-1,  0, +1, -1)) tris.push_back({m02, m23, p[1]});
  if (s == id(-1,  0, +1, +1)) tris.push_back({m02, m03, p[1]});
  if (s == id(+1,  0, -1, -1)) tris.push_back({m03, m02, p[1]});
  if (s == id(+1,  0, -1, +1)) tris.push_back({m23, m02, p[1]});
  if (s == id(+1,  0, +1, -1)) tris.push_back({m03, m23, p[1]});
//if (s == id(+1,  0, +1, +1)) return;

//if (s == id(-1, -1,  0, -1)) return;
  if (s == id(-1, -1,  0, +1)) tris.push_back({m03, m13, p[2]});
  if (s == id(-1, +1,  0, -1)) tris.push_back({m13, m01, p[2]});
  if (s == id(-1, +1,  0, +1)) tris.push_back({m03, m01, p[2]});
  if (s == id(+1, -1,  0, -1)) tris.push_back({m01, m03, p[2]});
  if (s == id(+1, -1,  0, +1)) tris.push_back({m01, m13, p[2]});
  if (s == id(+1, +1,  0, -1)) tris.push_back({m13, m03, p[2]});
//if (s == id(+1, +1,  0, +1)) return;

//if (s == id(-1, -1, -1,  0)) return;
  if (s == id(-1, -1, +1,  0)) tris.push_back({m12, m02, p[3]});
  if (s == id(-1, +1, -1,  0)) tris.push_back({m01, m12, p[3]});
  if (s == id(-1, +1, +1,  0)) tris.push_back({m01, m02, p[3]});
  if (s == id(+1, -1, -1,  0)) tris.push_back({m02, m01, p[3]});
  if (s == id(+1, -1, +1,  0)) tris.push_back({m12, m01, p[3]});
  if (s == id(+1, +1, -1,  0)) tris.push_back({m02, m12, p[3]});
//if (s == id(+1, +1, +1,  0)) return;

  if (s == id(-1, +1, +1, +1)) tris.push_back({m01, m02, m03});
  if (s == id(+1, -1, +1, +1)) tris.push_back({m01, m13, m12});
  if (s == id(+1, +1, -1, +1)) tris.push_back({m02, m12, m23});
  if (s == id(+1, +1, +1, -1)) tris.push_back({m03, m23, m13});

  if (s == id(+1, -1, -1, -1)) tris.push_back({m03, m02, m01});
  if (s == id(-1, +1, -1, -1)) tris.push_back({m12, m13, m01});
  if (s == id(-1, -1, +1, -1)) tris.push_back({m23, m12, m02});
  if (s == id(-1, -1, -1, +1)) tris.push_back({m13, m23, m03});

  if (s == id(+1, -1, -1, +1)) { tris.push_back({m01, m13, m23}); tris.push_back({m01, m23, m02}); }
  if (s == id(-1, +1, +1, -1)) { tris.push_back({m23, m13, m01}); tris.push_back({m02, m23, m01}); }
  if (s == id(+1, -1, +1, -1)) { tris.push_back({m01, m23, m12}); tris.push_back({m01, m03, m23}); }
  if (s == id(-1, +1, -1, +1)) { tris.push_back({m12, m23, m01}); tris.push_back({m23, m03, m01}); }
  if (s == id(+1, +1, -1, -1)) { tris.push_back({m12, m03, m02}); tris.push_back({m12, m13, m03}); }
  if (s == id(-1, -1, +1, +1)) { tris.push_back({m02, m03, m12}); tris.push_back({m03, m13, m12}); }

//if (s == id(+1, +1, +1, +1)) return;
  // clang-format on


/* The following Mathematica code is very helpful for visualizing the different orientations
   and figuring out what to do in each case:

verts = {{0, 0, 
    Sqrt[2/3] - 1/(2 Sqrt[6])}, {-(1/(2 Sqrt[3])), -(1/2), -(1/(
     2 Sqrt[6]))}, {-(1/(2 Sqrt[3])), 1/2, -(1/(2 Sqrt[6]))}, {1/Sqrt[
    3], 0, -(1/(2 Sqrt[6]))}};
Graphics3D[{
  FaceForm[None], Tetrahedron[verts],
  Text[Style["\!\(\*SubscriptBox[\(p\), \(0\)]\)", Large], 
   1.3 verts[[1]]],
  Text[Style["\!\(\*SubscriptBox[\(p\), \(1\)]\)", Large], 
   1.3 verts[[2]]],
  Text[Style["\!\(\*SubscriptBox[\(p\), \(2\)]\)", Large], 
   1.3 verts[[3]]],
  Text[Style["\!\(\*SubscriptBox[\(p\), \(3\)]\)", Large], 
   1.3 verts[[4]]],
  {Text[Style[
       Subscript[m, ToString[#[[1]] - 1] <> ToString[#[[2]] - 1]], 
       Large], 1.3 Mean[verts[[#]]]],
     Point[Mean[verts[[#]]]]} & /@ {{1, 2}, {1, 3}, {1, 4}, {2, 
     3}, {2, 4}, {3, 4}}
  }, Boxed -> False] 
*/

}

// clang-format off
void MarchingTetrahedra(vec3f p[4], float v[4], std::vector<Tetrahedron>& tets) {

  auto s = id(sign(v[0]), sign(v[1]), sign(v[2]), sign(v[3]));

  if ((s == 2222) || 
      (s == 1222) || (s == 2122) || (s == 2212) || (s == 2221) || 
      (s == 1122) || (s == 1212) || (s == 1221) || (s == 2112) || (s == 2121) || (s == 2211) || 
      (s == 1112) || (s == 1121) || (s == 1211) || (s == 2111) || 
      (s == 1111)) { return; }

  if (s == id(-1, -1, -1, -1) || 

      s == id( 0, -1, -1, -1) || 
      s == id(-1,  0, -1, -1) || 
      s == id(-1, -1,  0, -1) || 
      s == id(-1, -1, -1,  0) ||

      s == id( 0,  0, -1, -1) ||
      s == id( 0, -1,  0, -1) ||
      s == id( 0, -1, -1,  0) ||
      s == id(-1,  0,  0, -1) ||
      s == id(-1,  0, -1,  0) ||
      s == id(-1, -1,  0,  0) ||

      s == id(-1,  0,  0,  0) || 
      s == id( 0, -1,  0,  0) || 
      s == id( 0,  0, -1,  0) || 
      s == id( 0,  0,  0, -1)) {
    tets.push_back({p[0], p[1], p[2], p[3]});
  }

  vec3f m01 = find_root({p[0], p[1]}, {v[0], v[1]});
  vec3f m02 = find_root({p[0], p[2]}, {v[0], v[2]});
  vec3f m03 = find_root({p[0], p[3]}, {v[0], v[3]});
  vec3f m12 = find_root({p[1], p[2]}, {v[1], v[2]});
  vec3f m13 = find_root({p[1], p[3]}, {v[1], v[3]});
  vec3f m23 = find_root({p[2], p[3]}, {v[2], v[3]});

  // 1 in, 3 out => 1 tet
  if (s == id(+1, +1, +1, -1)) tets.push_back({m03, m13, m23, p[3]});
  if (s == id(+1, +1, -1, +1)) tets.push_back({m02, m12, p[2], m23});
  if (s == id(+1, -1, +1, +1)) tets.push_back({m01, p[1], m12, m13});
  if (s == id(-1, +1, +1, +1)) tets.push_back({p[0], m01, m02, m03});

  auto prism = [&tets](const vec3f &  p0, const vec3f &  p1,
                       const vec3f & m01, const vec3f & m12, 
                       const vec3f & m02, const vec3f & m03, const vec3f & m13) {
    tets.push_back({p0, m01, m02, m03});
    tets.push_back({m02, m01, m12, m03});
    tets.push_back({p1, m12, m01, m13});
    tets.push_back({m01, m13, m12, m03});
  };

  // 2 in, 2 out => truncated triangular prism => 4 tets
  if (s == id(-1, -1, +1, +1)) prism(p[0], p[1], m01, m12, m02, m03, m13);
  if (s == id(-1, +1, -1, +1)) prism(p[2], p[0], m02, m01, m12, m23, m03);
  if (s == id(-1, +1, +1, -1)) prism(p[0], p[3], m03, m13, m01, m02, m23);
  if (s == id(+1, -1, -1, +1)) prism(p[1], p[2], m12, m02, m01, m13, m23);
  if (s == id(+1, -1, +1, -1)) prism(p[1], p[3], m13, m23, m12, m01, m03);
  if (s == id(+1, +1, -1, -1)) prism(p[2], p[3], m23, m03, m02, m12, m13);

  auto frustum = [&tets](const vec3f &  p0, const vec3f &  p1, const vec3f &  p2, 
                         const vec3f & m01, const vec3f & m12, const vec3f & m02, 
                         const vec3f & m03, const vec3f & m13, const vec3f & m23) {
    tets.push_back({p0, m01, m02, m03});
    tets.push_back({m02, m01, m12, m03});
    tets.push_back({p1, m12, m01, m13});
    tets.push_back({m01, m13, m12, m03});
    tets.push_back({p2, m02, m12, m23});
    tets.push_back({m12, m23, m02, m03});
    tets.push_back({m13, m03, m23, m12});
  };

  // 3 in, 1 out => frustum => 7 tets
  if (s == id(-1, -1, -1, +1)) { frustum(p[0], p[1], p[2], m01, m12, m02, m03, m13, m23); }; 
  if (s == id(-1, -1, +1, -1)) { frustum(p[0], p[3], p[1], m03, m13, m01, m02, m23, m12); }; 
  if (s == id(-1, +1, -1, -1)) { frustum(p[0], p[2], p[3], m02, m23, m03, m01, m12, m13); }; 
  if (s == id(+1, -1, -1, -1)) { frustum(p[1], p[3], p[2], m13, m23, m12, m01, m03, m02); }; 

  // 1 in, 2 boundary, 1 out => 1 tet
  if (s == id( 0,  0, -1, +1)) { tets.push_back({p[0], p[1], p[2], m23}); }
  if (s == id( 0,  0, +1, -1)) { tets.push_back({p[0], p[1], m23, p[3]}); }
  if (s == id( 0, -1,  0, +1)) { tets.push_back({p[0], p[2], m13, p[1]}); }
  if (s == id( 0, +1,  0, -1)) { tets.push_back({p[0], p[2], p[3], m13}); }
  if (s == id( 0, -1, +1,  0)) { tets.push_back({p[0], p[3], p[1], m12}); }
  if (s == id( 0, +1, -1,  0)) { tets.push_back({p[0], p[3], m12, p[2]}); }
  if (s == id(-1,  0,  0, +1)) { tets.push_back({p[1], p[2], p[0], m03}); }
  if (s == id(+1,  0,  0, -1)) { tets.push_back({p[1], p[2], m03, p[3]}); }
  if (s == id(-1,  0, +1,  0)) { tets.push_back({p[1], p[3], m02, p[0]}); }
  if (s == id(+1,  0, -1,  0)) { tets.push_back({p[1], p[3], p[2], m02}); }
  if (s == id(-1, +1,  0,  0)) { tets.push_back({p[2], p[3], p[0], m01}); }
  if (s == id(+1, -1,  0,  0)) { tets.push_back({p[2], p[3], m01, p[1]}); }

  // written from perspective of (-1, -1, +1, 0)
  auto pyramid = [&tets](const vec3f &  p0, const vec3f &  p1, const vec3f &  p3,
                         const vec3f & m01, const vec3f & m12, const vec3f & m02) {
    tets.push_back({ p0, m01, m02, p3});
    tets.push_back({m01, m12, m02, p3});
    tets.push_back({m01,  p1, m12, p3});
  };

  // 2 in, 1 boundary, 1 out => trapezoidal-base pyramid => 3 tets
  if (s == id(-1, -1, +1,  0)) { pyramid(p[0], p[1], p[3], m01, m12, m02); }
  if (s == id(-1, +1, -1,  0)) { pyramid(p[2], p[0], p[3], m02, m01, m12); }
  if (s == id(+1, -1, -1,  0)) { pyramid(p[1], p[2], p[3], m12, m02, m01); }

  if (s == id(-1, -1,  0, +1)) { pyramid(p[1], p[0], p[2], m01, m03, m13); }
  if (s == id(-1, +1,  0, -1)) { pyramid(p[0], p[3], p[2], m03, m13, m01); }
  if (s == id(+1, -1,  0, -1)) { pyramid(p[3], p[1], p[2], m13, m01, m03); }

  if (s == id(-1,  0, -1, +1)) { pyramid(p[0], p[2], p[1], m02, m23, m03); }
  if (s == id(-1,  0, +1, -1)) { pyramid(p[3], p[0], p[1], m03, m02, m23); }
  if (s == id(+1,  0, -1, -1)) { pyramid(p[2], p[3], p[1], m23, m03, m02); }

  if (s == id( 0, -1, -1, +1)) { pyramid(p[2], p[1], p[0], m12, m13, m23); }
  if (s == id( 0, -1, +1, -1)) { pyramid(p[1], p[3], p[0], m13, m23, m12); }
  if (s == id( 0, +1, -1, -1)) { pyramid(p[3], p[2], p[0], m23, m12, m13); }

  // 1 in, 1 boundary, 2 out => 1 tet
  if (s == id(+1, +1, -1,  0)) { tets.push_back({p[2], m02, m12, p[3]}); }
  if (s == id(+1, -1, +1,  0)) { tets.push_back({p[1], m12, m01, p[3]}); }
  if (s == id(-1, +1, +1,  0)) { tets.push_back({p[0], m01, m02, p[3]}); }

  if (s == id(+1, +1,  0, -1)) { tets.push_back({p[3], m13, m03, p[2]}); }
  if (s == id(+1, -1,  0, +1)) { tets.push_back({p[1], m01, m13, p[2]}); }
  if (s == id(-1, +1,  0, +1)) { tets.push_back({p[0], m03, m01, p[2]}); }

  if (s == id(+1,  0, +1, -1)) { tets.push_back({p[3], m03, m23, p[1]}); }
  if (s == id(+1,  0, -1, +1)) { tets.push_back({p[2], m23, m02, p[1]}); }
  if (s == id(-1,  0, +1, +1)) { tets.push_back({p[0], m02, m03, p[1]}); }

  if (s == id( 0, +1, +1, -1)) { tets.push_back({p[3], m23, m13, p[0]}); }
  if (s == id( 0, +1, -1, +1)) { tets.push_back({p[2], m12, m23, p[0]}); }
  if (s == id( 0, -1, +1, +1)) { tets.push_back({p[1], m13, m12, p[0]}); }

}
// clang-format on

constexpr uint32_t id(VertexId a, VertexId b, VertexId c, VertexId d) {
  return 10 * (10 * (10 * uint32_t(a.type) + uint32_t(b.type)) + uint32_t(c.type)) + uint32_t(d.type);
}

bool swap(VertexId & i, VertexId & j) { 
  if (j < i) { 
    std::swap(i,j); 
    return true;
  } else {
    return false;
  }
};

bool sort(VertexId v[4]) {
  return swap(v[0],v[2]) ^ swap(v[1],v[3]) ^ swap(v[0],v[1]) ^ swap(v[2],v[3]) ^ swap(v[1],v[2]);
}

// clang-format off
void MarchingTetrahedra(VertexId v[4], std::vector< std::array < EdgeOrVertexId, 4 > >& tets) {

  // sorting the vertex ids first lets us only
  // have to consider 15 cases, instead of 81
  bool flip = sort(v);

  switch (id(v[0], v[1], v[2], v[3])) {

    // ignore any case where there are no interior vertices
    default: 
    case id(+1, +1, +1, +1):
    case id( 0, +1, +1, +1): 
    case id( 0,  0, +1, +1): 
    case id( 0,  0,  0, +1): 
    case id( 0,  0,  0,  0): break;

    // any case with at least one interior vertex and no exterior vertices
    // does not need to be cut at all, and we just emit the whole tetrahedron
    case id(-1,  0,  0,  0):
    case id(-1, -1,  0,  0):
    case id(-1, -1, -1,  0):
    case id(-1, -1, -1, -1): {
      if (flip) { std::swap(v[0], v[1]); }
      tets.push_back({v[0], v[1], v[2], v[3]});
      break; 
    }

    // 1 interior and 3 exterior vertices emits one tet
    case id(-1, +1, +1, +1): {
      if (flip) { std::swap(v[2], v[3]); }
      tets.push_back({v[0], {v[0], v[1]}, {v[0], v[2]}, {v[0], v[3]}});
      break; 
    }

    case id(-1,  0, +1, +1): {
      if (flip) { std::swap(v[2], v[3]); }
      tets.push_back({v[0], v[1], {v[0], v[2]}, {v[0], v[3]}});
      break; 
    }

    case id(-1, -1,  0, +1): {
      if (flip) { 
        tets.push_back({v[1], {v[0], v[3]}, v[2], {v[1], v[3]}});
        tets.push_back({v[1], v[0], v[2], {v[0], v[3]}});
      } else {
        tets.push_back({v[1], {v[0], v[3]}, {v[1], v[3]}, v[2]});
        tets.push_back({v[1], v[0], {v[0], v[3]}, v[2]});
      }
      break;
    }

    case id(-1,  0,  0, +1): {
      if (flip) { std::swap(v[1], v[2]); }
      tets.push_back({v[0], v[1], v[2], {v[0], v[3]}});
      break;
    }

    case id(-1, -1, +1, +1): {
      if (flip) { std::swap(v[2], v[3]); }
      tets.push_back({v[0], v[1], {v[1], v[2]}, {v[1], v[3]}});
      tets.push_back({v[0], {v[1], v[2]}, {v[0], v[2]}, {v[1], v[3]}});
      tets.push_back({v[0], {v[0], v[2]}, {v[0], v[3]}, {v[1], v[3]}});
      break;
    }

    case id(-1, -1, -1, +1): {
      if (flip) { 
        tets.push_back({v[0], v[2], v[1], {v[2], v[3]}});
        tets.push_back({v[0], {v[1], v[3]}, {v[0], v[3]}, {v[2], v[3]}});
        tets.push_back({v[0], {v[2], v[3]}, v[1], {v[1], v[3]}});
      } else {
        tets.push_back({v[0], v[1], v[2], {v[2], v[3]}});
        tets.push_back({v[0], {v[0], v[3]}, {v[1], v[3]}, {v[2], v[3]}});
        tets.push_back({v[0], v[1], {v[2], v[3]}, {v[1], v[3]}});
      }
      break;
    }
  }
  
}

template < typename cell_type >
std::vector < cell_type > GenerateMesh(const std::function< float(vec3f) > & f,
AABB<3> bounds, int n, float snap_threshold) {

  vec3f offset = bounds.min;
  vec3f width = bounds.max - bounds.min;

  float dx = min(width) / n;

  int n0 = ceil(width[0] / dx);
  int n1 = ceil(width[1] / dx);
  int n2 = ceil(width[2] / dx);

  float epsilon = 1.0e-4 * dx;
  vec3f e1{epsilon, 0.0, 0.0};
  vec3f e2{0.0, epsilon, 0.0};
  vec3f e3{0.0, 0.0, epsilon};

  auto g = [&](auto ... args) { return GridPoint(args...) * dx + offset; };

  heap_array< float, 3 > values(n0, n1, 2 * n2);
  heap_array< vec3f, 3 > nodes(n0, n1, 2 * n2);

  std::cout << "evaluating implicit function at gridpoints... " << std::flush;

  timer stopwatch;
  stopwatch.start();

  int num_threads = std::thread::hardware_concurrency();

  std::vector< std::thread > threads;
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread([&](int i0) {
      for (int i = i0; i < n0; i+= num_threads) {
        for (int j = 0; j < n1; j++) {
          for (int k = 0; k < 2 * n2; k++) {
            vec3f x = g(i,j,k);
            float v = f(x);
            auto dv_dx = vec3f{f(x+e1)-v, f(x+e2)-v, f(x+e3)-v} / epsilon;

            // if a node is close to the zero isocontour, then we'll
            // nudge it toward zero by doing a few steps of gradient descent
            if (fabs(v) < (norm(dv_dx) * snap_threshold * dx)) {
              for (int m = 0; m < 3; m++) {
                x -= (v * dv_dx) / dot(dv_dx, dv_dx);
                v = f(x);
              }
              v = 0.0;
            }

            nodes(i,j,k) = x;
            values(i,j,k) = v;
          }
        }
      }
    }, i));
  }
  for (auto & thread : threads) { thread.join(); }

  stopwatch.stop();
  std::cout << " done in " << stopwatch.elapsed() << "s" << std::endl;

  std::vector < cell_type > cells;
  cells.reserve((n0 - 1) * (n1 - 1) * (n2 - 1));

  std::cout << "applying Marching Tetrahedron algorithm...    " << std::flush;
  stopwatch.start();

  std::vector< std::vector < cell_type > > cells_per_thread(num_threads);
  threads.clear();
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread([&](int x0) {
      for (int x = x0; x < n0 - 1; x += num_threads) {
        for (int y = 0; y < n1 - 1; y++) {
          for (int z = 0; z < n2 - 1; z++) {

            for (int w = 0; w < TetrahedraPerCell(LatticeType::BCC); w++) {
              auto indices = TetrahedronGridIndices(x, y, z, w);

              float tet_values[4];
              vec3f tet_vertices[4];
              for (int m = 0; m < 4; m++) {
                auto [i, j, k] = indices[m];
                tet_values[m] = values(i, j, k);
                tet_vertices[m] = nodes(i, j, k);
              }

              MarchingTetrahedra(tet_vertices, tet_values, cells_per_thread[x0]);
            }

          }
        }
      }
    }, i));
  }
  for (int i = 0; i < num_threads; i++) {
    threads[i].join(); 
    cells.reserve(cells.size() + cells_per_thread[i].size());
    cells.insert(cells.end(), cells_per_thread[i].begin(), cells_per_thread[i].end());  
  }
  stopwatch.stop();
  std::cout << " done in " << stopwatch.elapsed() << "s" << std::endl;

  return cells;
}

std::vector < Triangle > GenerateSurfaceMesh(const std::function< float(vec3f) > & f,
                                             AABB<3> bounds, int n, float snap_threshold) {
  return GenerateMesh<Triangle>(f, bounds, n, snap_threshold);
}

std::vector < Tetrahedron > GenerateVolumeMesh(const std::function< float(vec3f) > & f,
                                             AABB<3> bounds, int n, float snap_threshold) {
  return GenerateMesh<Tetrahedron>(f, bounds, n, snap_threshold);
}

}