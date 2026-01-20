#include "BVH.hpp"
#include "geometry/geometry.hpp"
#include "wll_interface.h"

#include "binary_io.hpp"

using namespace geometry;

SimplexMesh<2> mesh2D{};
SimplexMesh<3> mesh3D{};

template <typename T>
std::vector<T> convert_from_wll(const wll::tensor<T, 1> &wll, T offset = 0) {
  std::vector<T> output(wll.dimension(0));
  for (int i = 0; i < output.size(); i++) {
    output[i] = wll(i) + offset;
  }
  return output;
}

template <typename T, typename S>
std::vector<T> convert_from_wll(const wll::tensor<S, 2> &wll, S offset = 0) {
  constexpr size_t n = T{}.size();
  static_assert(std::is_same<typename T::value_type, S>::value);
  std::vector<T> output(wll.dimension(0));
  for (int i = 0; i < output.size(); i++) {
    for (int j = 0; j < n; j++) {
      output[i][j] = wll(i, j) + offset;
    }
  }
  return output;
}

template <typename T>
wll::tensor<T, 1> convert_to_wll(const std::vector<T> &vec, T offset = 0) {
  wll::tensor<T, 1> output({vec.size()});
  for (int i = 0; i < vec.size(); i++) {
    output(i) = vec[i] + offset;
  }
  return output;
}

template <typename T, size_t n>
wll::tensor<T, 2> convert_to_wll(const std::vector<std::array<T, n>> &vec,
                                 T offset = 0) {
  wll::tensor<T, 2> output({vec.size(), n});
  for (int i = 0; i < vec.size(); i++) {
    for (int j = 0; j < n; j++) {
      output(i, j) = vec[i][j] + offset;
    }
  }
  return output;
}

template <typename T, size_t n>
wll::tensor<T, n> convert_to_wll(const std::vector<T> &input,
                                 std::array<size_t, n> dimensions,
                                 T offset = 0) {
  wll::tensor<T, n> output(dimensions);
  for (std::size_t i = 0; i < output.size(); i++) {
    output[i] = input[i] + offset;
  }
  return output;
}

template <uint64_t dim>
struct structured_mesh {
  structured_mesh(const wll::tensor<double, dim> & v, AABB<3> bounds) : values(v) {
    skip = false;
    for (int i = 0; i < dim; i++) {
      n[i] = values.dimension(i) - 1;
      offset[i] = bounds.min[i];
      scale[i] = (bounds.max[i] - bounds.min[i]) / n[i];
      skip |= (n[i] == 0);
    }
  }

  double operator()(vecf<dim> x) const {
    if (skip) return 0.0;

    vecf<dim> y = (x - offset) / scale;
    if constexpr (dim == 2) {
      uint32_t i = std::min(uint32_t(floor(x[0] - offset[0] * scale[0])), n[0]-1);
      uint32_t j = std::min(uint32_t(floor(x[1] - offset[1] * scale[1])), n[1]-1);

      float s = y[0] - i;
      float t = y[1] - j;
      
      return (1.0f - s) * (1.0f - t) * values(i  , j  ) + 
                     s  * (1.0f - t) * values(i+1, j  ) + 
             (1.0f - s) *         t  * values(i  , j+1) + 
                     s  *         t  * values(i+1, j+1);
    }

    if constexpr (dim == 3) {
      uint32_t i = std::min(uint32_t(floor(x[0] - offset[0] * scale[0])), n[0]-1);
      uint32_t j = std::min(uint32_t(floor(x[1] - offset[1] * scale[1])), n[1]-1);
      uint32_t k = std::min(uint32_t(floor(x[2] - offset[2] * scale[2])), n[2]-1);

      float s = y[0] - i;
      float t = y[1] - j;
      float u = y[2] - k;

      double v[4] = {
        (1.0f - u) * values(i  , j  , k) + u * values(i  , j  , k+1),
        (1.0f - u) * values(i+1, j  , k) + u * values(i+1, j  , k+1),
        (1.0f - u) * values(i  , j+1, k) + u * values(i  , j+1, k+1),
        (1.0f - u) * values(i+1, j+1, k) + u * values(i+1, j+1, k+1)
      };

      return (1.0f - s) * (1.0f - t) * v[0] + 
                     s  * (1.0f - t) * v[1] + 
             (1.0f - s) *         t  * v[2] + 
                     s  *         t  * v[3];
    }
  }

  bool skip;
  uint32_t n[dim];
  vecf<dim> offset;
  vecf<dim> scale;
  const wll::tensor<double, dim> & values; 
};

template <size_t dim>
void create_mesh(const wll::tensor<double, 2> & wll_capsules,
                 const wll::tensor<double, 2> & wll_bounds,
                 const wll::tensor<double, dim> & wll_offsets, 
                 double cell_size,
                 double blend_radius = 0.0) {

  AABB<3> bounds{};
  AABB<3> inflated_bounds = bounds;
  for (int i = 0; i < dim; i++) {
    bounds.min[i] = wll_bounds(0, i);
    bounds.max[i] = wll_bounds(1, i);

    inflated_bounds.min[i] = wll_bounds(0, i) - cell_size;
    inflated_bounds.max[i] = wll_bounds(1, i) + cell_size;
  }

  int num_capsules = wll_capsules.dimension(0);
  std::vector < Capsule > capsules(num_capsules);
  for (int i = 0; i < num_capsules; i++) {
    vec3f p{};
    vec3f q{};
    for (int j = 0; j < dim; j++) {
      p[j] = wll_capsules(i, dim * 0 + j);
      q[j] = wll_capsules(i, dim * 1 + j);
    }
    float r = wll_capsules(i, dim * 2);
    capsules[i] = Capsule{p, q, r, r};
  }

  std::vector<AABB<dim>> bounding_boxes(capsules.size(), AABB<dim>{});
  for (uint32_t i = 0; i < capsules.size(); i++) {
    AABB<3> box = bounding_box(capsules[i]);
    for (int j = 0; j < dim; j++) {
      bounding_boxes[i].min[j] = box.min[j];
      bounding_boxes[i].max[j] = box.max[j];
    }
  }

  BVH<dim> bvh(bounding_boxes);

  structured_mesh<dim> distance_offset(wll_offsets, bounds); 

  std::function<float(vecf<dim>)> sdf = [&](vecf<dim> x) -> float {

    float d = distance_offset(x);

    float dx = 1.5 * cell_size + 2.0 * blend_radius + fabs(d);

    vec3f x3{};
    AABB<dim> box{};
    for (int j = 0; j < dim; j++) {
      x3[j] = x[j];
      box.min[j] = x[j] - dx;
      box.max[j] = x[j] + dx;
    };

    if (blend_radius == 0) {
      float value = 2 * dx;
      bvh.query(box, [&](int i) {
        value = std::min(value, capsules[i].SDF(x3));
      });
      return std::max(value - d, bounds.SDF(x3));
    } else {
      double value = 0.0;
      bvh.query(box, [&](int i) {
        value += exp(-capsules[i].SDF(x3) / blend_radius);
      });
      return std::max(float(-blend_radius * log(value) - d), bounds.SDF(x3));
    }
  };

  if constexpr (dim == 2) {
    mesh2D = universal_mesh(sdf, cell_size, inflated_bounds);
  }

  if constexpr (dim == 3) {
    mesh3D = universal_mesh(sdf, cell_size, inflated_bounds);
  }

}

void create_mesh_2D(const wll::tensor<double, 2> &capsules,
                    const wll::tensor<double, 2> &bounds,
                    const wll::tensor<double, 2> &offsets, 
                    double cell_size,
                    double blend_radius) {
  create_mesh(capsules, bounds, offsets, cell_size, blend_radius);
}

void create_mesh_3D(const wll::tensor<double, 2> &capsules,
                    const wll::tensor<double, 2> &bounds,
                    const wll::tensor<double, 3> &offsets, 
                    double cell_size,
                    double blend_radius) {
  create_mesh(capsules, bounds, offsets, cell_size, blend_radius);
}

wll::tensor<double, 2> get_mesh_vertices(uint32_t dim) {
  if (dim == 2) {
    uint32_t num_vertices = mesh2D.vertices.size();
    wll::tensor<double, 2> output({num_vertices, dim});
    for (int i = 0; i < num_vertices; i++) {
      for (int j = 0; j < dim; j++) {
        output(i, j) = mesh2D.vertices[i][j];
      }
    }
    return output;
  } 
  if (dim == 3) {
    uint32_t num_vertices = mesh3D.vertices.size();
    wll::tensor<double, 2> output({num_vertices, dim});
    for (int i = 0; i < num_vertices; i++) {
      for (int j = 0; j < dim; j++) {
        output(i, j) = mesh3D.vertices[i][j];
      }
    }
    return output;
  }

  return {}; // unreachable
}

wll::tensor<uint32_t, 2> get_mesh_elements(uint32_t dim) {
  if (dim == 2) {
    uint32_t num_elements = mesh2D.elements.size();
    wll::tensor<uint32_t, 2> output({num_elements, dim + 1});
    for (int i = 0; i < num_elements; i++) {
      for (int j = 0; j < dim + 1; j++) {
        output(i, j) = mesh2D.elements[i][j] + 1;
      }
    }
    return output;
  } 

  if (dim == 3) {
    uint32_t num_elements = mesh3D.elements.size();
    wll::tensor<uint32_t, 2> output({num_elements, dim + 1});
    for (int i = 0; i < num_elements; i++) {
      for (int j = 0; j < dim + 1; j++) {
        output(i, j) = mesh3D.elements[i][j] + 1;
      }
    }
    return output;
  }

  return {}; // unreachable
}

wll::tensor<uint32_t, 2> get_mesh_boundary_elements(uint32_t dim) {
  if (dim == 2) {
    uint32_t num_boundary_elements = mesh2D.boundary_elements.size();
    wll::tensor<uint32_t, 2> output({num_boundary_elements, dim});
    for (int i = 0; i < num_boundary_elements; i++) {
      for (int j = 0; j < dim; j++) {
        output(i, j) = mesh2D.boundary_elements[i][j] + 1;
      }
    }
    return output;
  } 

  if (dim == 3) {
    uint32_t num_boundary_elements = mesh3D.boundary_elements.size();
    wll::tensor<uint32_t, 2> output({num_boundary_elements, dim});
    for (int i = 0; i < num_boundary_elements; i++) {
      for (int j = 0; j < dim; j++) {
        output(i, j) = mesh3D.boundary_elements[i][j] + 1;
      }
    }
    return output;
  }

  return {}; // unreachable
}

AABB<3> bounding_box(std::vector<vec3f> points){
  AABB<3> box = {points[0],points[0]};
  for(int i =1; i < points.size(); i++){
    box.min[0]=std::min(box.min[0],points[i][0]);
    box.min[1]=std::min(box.min[1],points[i][1]);
    box.min[2]=std::min(box.min[2],points[i][2]);

    box.max[0]=std::max(box.max[0],points[i][0]);
    box.max[1]=std::max(box.max[1],points[i][1]);
    box.max[2]=std::max(box.max[2],points[i][2]);
  }
  return box;
}

void remesh_3D(const wll::tensor<double, 2> &vertex_coordinates,
               const wll::tensor<uint32_t, 2> &tets,
               const wll::tensor<uint32_t, 2> &bdr_tris, 
               double cell_size) {

  SimplexMesh<3> mesh{};
  int numverts= vertex_coordinates.dimension(0);
  mesh.vertices.resize(numverts);
  for(int i =0;i<numverts;i++){
  mesh.vertices[i]={float(vertex_coordinates(i,0)),float(vertex_coordinates(i,1)),float(vertex_coordinates(i,2))};
  }

  AABB<3> bounds = bounding_box(mesh.vertices);
  bounds.min[0] -= cell_size;
  bounds.min[1] -= cell_size;
  bounds.min[2] -= cell_size;

  bounds.max[0] += cell_size;
  bounds.max[1] += cell_size;
  bounds.max[2] += cell_size;

  int numtets = tets.dimension(0);
  mesh.elements.resize(numtets);
  for(int i =0;i<numtets;i++){
    mesh.elements[i]={uint64_t(tets(i,0)-1),uint64_t(tets(i,1)-1),uint64_t(tets(i,2)-1),uint64_t(tets(i,3)-1)};
  }

  int numtris = bdr_tris.dimension(0);
  mesh.boundary_elements.resize(numtris);
  for(int i =0;i<numtris;i++){
    mesh.boundary_elements[i]={uint64_t(bdr_tris(i,0)-1),uint64_t(bdr_tris(i,1)-1),uint64_t(bdr_tris(i,2)-1)};
  }

  mesh3D = universal_mesh(SDF(mesh,cell_size),cell_size,bounds);

}

vec3f normalVector(vec3f p0, vec3f p1, vec3f p2){
return normalize(cross(p1-p0,p2-p0));
}

vec3f getAngles(vec3f p0, vec3f p1, vec3f p2){
vec3f e1=normalize(p1-p0);
vec3f e2=normalize(p2-p1);
vec3f e3=normalize(p0-p2);
return {acosf(dot(-e3,e1)),acosf(dot(-e1,e2)),acosf(dot(-e3,e2))};
}

wll::tensor<double, 2> apply_normal_displacement(const wll::tensor<double, 2> &vertex_coordinates,
                                                 const wll::tensor<uint32_t, 2> &mesh_elements,
                                                 const wll::tensor<uint32_t, 2> &bdr_tris,
                                                 const wll::tensor<double, 1> &normal_displacement,
                                                 float cell_size,
                                                 int ndvr,
                                                 const wll::tensor<double, 1> &bounds) {

  constexpr int dim = 3;

  // convert wll::tensor types into native C++ types
  uint32_t nverts= vertex_coordinates.dimension(0);
  std::vector< vec3f > v(nverts);
  for(uint32_t i =0; i < nverts; i++){
    v[i] = {
      float(vertex_coordinates(i,0)),
      float(vertex_coordinates(i,1)),
      float(vertex_coordinates(i,2))
    };
  }

  uint32_t ntris = bdr_tris.dimension(0);
  std::vector< std::array<uint32_t, 3> > tris(ntris);
  for(uint32_t i =0; i < ntris; i++){
    tris[i] = {bdr_tris(i,0)-1, bdr_tris(i,1)-1, bdr_tris(i,2)-1};
  }

  std::vector< vec3f > v_new(nverts);
  std::vector<vec3f> vertexNormals(nverts,{0,0,0});
  for(uint32_t i = 0; i < tris.size(); i++){
  vec3f p0=v[tris[i][0]];
  vec3f p1=v[tris[i][1]];
  vec3f p2=v[tris[i][2]];
  mat3f scaledNormals = outer(getAngles(p0,p1,p2), normalVector(p0,p1,p2));

  vertexNormals[tris[i][0]]+=scaledNormals[0];
  vertexNormals[tris[i][1]]+=scaledNormals[1];
  vertexNormals[tris[i][2]]+=scaledNormals[2];
  }
for(uint32_t i =0; i < vertexNormals.size(); i++){
vertexNormals[i]=normalize(vertexNormals[i]);
  }

for (uint32_t i = 0; i < nverts; i++) {
   v_new[i] = v[i]-float(normal_displacement[i])*vertexNormals[i];
  }
 
float xBound=bounds[0];
float yBound=bounds[1];
float zBound=bounds[2];

for(uint32_t i = 0; i < nverts; i++){
if(v_new[i][0]<-xBound){v_new[i][0]=-xBound;}
if(v_new[i][0]>xBound){v_new[i][0]=xBound;}

if(v_new[i][1]<-yBound){v_new[i][1]=-yBound;}
if(v_new[i][1]>yBound){v_new[i][1]=yBound;}

if(v_new[i][2]<-zBound){v_new[i][2]=-zBound;}
if(v_new[i][2]>zBound){v_new[i][2]=zBound;}
  }

//maybe here do a couple rounds of vertex relaxation 
//need to add the mesh elements to the input and then 
//need to make a simplex mesh from the mesh elements and new coordinates and then call the ndvr vertex relaxation function to keep the mesh quality up
 SimplexMesh<3> mesh{};

 mesh.vertices.resize(vertex_coordinates.size());
 mesh.vertices=v_new;

 uint32_t nelem= mesh_elements.dimension(0);
 mesh.elements.resize(nelem);
for(int i = 0; i < nelem;i++){
  mesh.elements[i]={
    mesh_elements(i,0)-1,
    mesh_elements(i,1)-1,
    mesh_elements(i,2)-1,
    mesh_elements(i,3)-1
  };
}

 uint32_t nbelem= bdr_tris.dimension(0);
  mesh.boundary_elements.resize(nbelem);
for(int i = 0; i < nbelem;i++){
  mesh.boundary_elements[i]={
    bdr_tris(i,0)-1,
    bdr_tris(i,1)-1,
    bdr_tris(i,2)-1
  };
}
//the mesh is now fully built and we can do the vertex relaxation

dvr(mesh, 7.5f,(0.05f*cell_size*cell_size),ndvr,8);

  // convert v_new back to wll::tensor type that Mathematica can understand
  wll::tensor<double, 2> output({nverts, dim});
  for (int i = 0; i < nverts; i++) {
    for (int j = 0; j < dim; j++) {
      output(i, j) = mesh.vertices[i][j];
    }
  }
  return output;

}

wll::tensor<double, 2> fit_edges_to_bounds(const wll::tensor<double, 2> &coords,
                                           double margin,
                                           const wll::tensor<double, 3> &bounds){

  constexpr int dim = 3;
//this moves points near (or beyond) the given bounds to the bounds.
// margin is how close a point can be to the edge before it gets pushed to the edge.

uint32_t nverts= coords.dimension(0);
  std::vector< vec3f > v_new(nverts);
  for(uint32_t i =0; i < nverts; i++){
    v_new[i] = {
      float(coords(i,0)),
      float(coords(i,1)),
      float(coords(i,2))
    };
  }

float xBound=bounds[0];
float yBound=bounds[1];
float zBound=bounds[2];


for(uint32_t i = 0; i < nverts; i++){
if(v_new[i][0]<float(-xBound+margin)){v_new[i][0]=-xBound;}
if(v_new[i][0]>float(xBound-margin)){v_new[i][0]=xBound;}

if(v_new[i][1]<float(-yBound+margin)){v_new[i][1]=-yBound;}
if(v_new[i][1]>float(yBound-margin)){v_new[i][1]=yBound;}

if(v_new[i][2]<float(-zBound+margin)){v_new[i][2]=-zBound;}
if(v_new[i][2]>float(zBound-margin)){v_new[i][2]=zBound;}
}




  // convert v_new back to wll::tensor type that Mathematica can understand
  wll::tensor<double, 2> output({nverts, dim});
  for (int i = 0; i < nverts; i++) {
    for (int j = 0; j < dim; j++) {
      output(i, j) = v_new[i][j];
    }
  }

  return output;

}

DEFINE_WLL_FUNCTION(remesh_3D)

DEFINE_WLL_FUNCTION(create_mesh_2D)
DEFINE_WLL_FUNCTION(create_mesh_3D)

DEFINE_WLL_FUNCTION(get_mesh_vertices)
DEFINE_WLL_FUNCTION(get_mesh_elements)
DEFINE_WLL_FUNCTION(get_mesh_boundary_elements)

DEFINE_WLL_FUNCTION(apply_normal_displacement)
DEFINE_WLL_FUNCTION(fit_edges_to_bounds)