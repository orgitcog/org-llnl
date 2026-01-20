#include "geometry.hpp"

#include "BVH.hpp"

#include <algorithm>

namespace geometry {

float TrianglePointDistance(Triangle tri, vec3f p)
{
  // actually implement
  vec3f a = tri.vertices[0];
  vec3f b = tri.vertices[1];
  vec3f c = tri.vertices[2];
  vec3f v0 = b - a;
  vec3f v1 = c - a;
  vec3f n = cross(v0, v1);
  vec3f v2 = p - a;
  float d = dot(v2, n) / norm(n);
  vec3f proj = p - (d * normalize(n));
  vec3f vbp = proj - a;
  float d00 = dot(v0, v0);
  float d01 = dot(v0, v1);
  float d11 = dot(v1, v1);
  float d20 = dot(vbp, v0);
  float d21 = dot(vbp, v1);
  float denom = (d00 * d11) - (d01 * d01);
  float v = ((d11 * d20) - (d01 * d21)) / denom;
  float w = ((d00 * d21) - (d01 * d20)) / denom;
  float u = 1 - v - w;
  vec3f closestPoint;
  if (1 >= u && u >= 0 && 1 >= v && v >= 0 && 1 >= w && w >= 0)
  {
    closestPoint = proj;
  }
  else
  {
    vec3f vecLastSide = c - b;
    vec3f vap = proj - a;
    vec3f vcp = proj - b;
    float t1 = std::clamp(dot(vap, v0) / dot(v0, v0), 0.0f, 1.0f);
    float t2 = std::clamp(dot(vap, v1) / dot(v1, v1), 0.0f, 1.0f);
    float t3 = std::clamp(dot(vcp, vecLastSide) / dot(vecLastSide, vecLastSide), 0.0f, 1.0f);

    vec3f ptAB = a + t1 * v0;
    vec3f ptAC = a + t2 * v1;
    vec3f ptBC = b + t3 * vecLastSide;

    float distAB = norm(ptAB - p);
    float distAC = norm(ptAC - p);
    float distBC = norm(ptBC - p);

    if (distAB <= distBC && distAB <= distAC)
    {
      closestPoint = ptAB;
    }
    else if (distBC <= distAC)
    {
      closestPoint = ptBC;
    }
    else
    {
      closestPoint = ptAC;
    }
  }
  float distance = norm(p - closestPoint);

  return distance;
}

float Determinant3x3(float mat[3][3])
{
  return (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]));
}

float tetVol(vec3f a, vec3f b, vec3f c, vec3f d)
{
  float mat[3][3] =
      {{(b - a)[0], (b - a)[1], (b - a)[2]},
       {(c - a)[0], (c - a)[1], (c - a)[2]},
       {(d - a)[0], (d - a)[1], (d - a)[2]}};
  return std::abs(Determinant3x3(mat)) / 6.0f;
}

bool PointInsideTetrahedron(Tetrahedron tet, vec3f p)
{
  // actually implement
  vec3f a = tet.vertices[0];
  vec3f b = tet.vertices[1];
  vec3f c = tet.vertices[2];
  vec3f d = tet.vertices[3];

  float volo = tetVol(a, b, c, d);
  float vol1 = tetVol(p, b, c, d);
  float vol2 = tetVol(a, p, c, d);
  float vol3 = tetVol(a, b, p, d);
  float vol4 = tetVol(a, b, c, p);
  if (std::abs(volo - (vol1 + vol2 + vol3 + vol4)) < ((1e-5f) * volo))
  {
    return true;
  }
  else
  {
    return false;
  }
}


std::function<float(vec3f)> SDF(const SimplexMesh<3> & mesh, float dx) {

  // first create arrays for the tets and tris, as well as their associated BVHs
  auto & v = mesh.vertices;

  std::vector< Tetrahedron > tets;
  tets.reserve(mesh.elements.size());
  std::vector< AABB<3> > tet_bounding_boxes(mesh.elements.size());
  for (uint32_t i = 0; i < mesh.elements.size(); i++) {
    auto tet = mesh.elements[i];
    vec3f min = v[tet[0]];
    vec3f max = v[tet[0]];
    for (uint32_t j = 1; j < 4; j++) {
      vec3f vj = v[tet[j]];
      min[0] = std::min(min[0], vj[0]);
      min[1] = std::min(min[1], vj[1]);
      min[2] = std::min(min[2], vj[2]);

      max[0] = std::max(max[0], vj[0]);
      max[1] = std::max(max[1], vj[1]);
      max[2] = std::max(max[2], vj[2]);
    }
    tets.push_back({{v[tet[0]], v[tet[1]], v[tet[2]], v[tet[3]]}});
    tet_bounding_boxes[i] = AABB<3>{min, max};
  };
  BVH<3> tet_bvh(tet_bounding_boxes);

  std::vector< Triangle > tris;
  tris.reserve(mesh.boundary_elements.size());
  std::vector< AABB<3> > tri_bounding_boxes(mesh.boundary_elements.size());
  for (uint32_t i = 0; i < mesh.boundary_elements.size(); i++) {
    auto tri = mesh.boundary_elements[i];
    vec3f min = v[tri[0]];
    vec3f max = v[tri[0]];
    for (uint32_t j = 1; j < 3; j++) {
      vec3f vj = v[tri[j]];
      min[0] = std::min(min[0], vj[0]);
      min[1] = std::min(min[1], vj[1]);
      min[2] = std::min(min[2], vj[2]);

      max[0] = std::max(max[0], vj[0]);
      max[1] = std::max(max[1], vj[1]);
      max[2] = std::max(max[2], vj[2]);
    }
    tris.push_back({{v[tri[0]], v[tri[1]], v[tri[2]]}});
    tri_bounding_boxes[i] = AABB<3>{min, max};
  };
  BVH<3> tri_bvh(tri_bounding_boxes);

////////////////////////////////////////////////////////////////////////////////

  return [v = mesh.vertices, tets, tris, tet_bvh, tri_bvh, dx] (vec3f x) {
    vec3f min = {x[0]-dx,x[1]-dx,x[2]-dx};
    vec3f max = {x[0]+dx,x[1]+dx,x[2]+dx};
AABB<3> pointBox= {min,max};
float min_dist=dx;
bool inside = false;

tet_bvh.query(pointBox, [&](int i){
  if (inside) return;
  if(PointInsideTetrahedron(tets[i],x)){
    inside =true;
  }
});

tri_bvh.query(pointBox,[&](int i){
  min_dist = std::min(min_dist, TrianglePointDistance(tris[i],x));
});

if(inside){
  min_dist= -min_dist;
}

    return min_dist;
  };

}

}