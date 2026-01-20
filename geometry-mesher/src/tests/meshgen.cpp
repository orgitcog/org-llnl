#include "BVH.hpp"
#include "geometry/geometry.hpp"
#include "geometry/parse_dat.hpp"

#include "binary_io.hpp"
#include "CLI11.hpp"

#include "fm/operations/print.hpp"

#include <algorithm>

using namespace geometry;

// map 2D lamina into 3D spherical shells
vec3f hemispherical_mapping(vec3f x, float R) {
  float theta_max = 1.57079632679;
  float phi = atan2(x[1], x[0]);
  float theta = theta_max * norm(vec2f{x[0], x[1]});
  vec3f p = {sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta)};
  vec3f n = p; // only true for spherical symmetry
  return R * (p + x[2] * n);
}

int main(int argc, char **argv) {

  CLI::App app(
    "                                                                     \n"
    "         command line 'universal meshing' tool for print data        \n"
    "the input file is expected to be a binary file of \"capsule\" shapes,\n"
    "each of which is made up of 8 single-precision floating point values:\n"
    "          [ x1, y1, z1, x2, y2, z2, r1, r2 ]                         \n"
    "where (x1,y1,z1), (x2,y2,z3) make up the endpoints of the capsule and\n"
    "r1 and r2 are the radii at the repsective endpoints.                 \n"
  );

  // input file is expected to contain [ x1, y1, z1, x2, y2, z2, r1, r2 ] for each capsule 1, 2, ..., n-1, n
  std::string input_filename;
  app.add_option("-i", input_filename, "input filename (see above for format specification)")->required();

  std::string output_filename;
  app.add_option("-o", output_filename, "output filename")->required();

  std::string attr_filename;
  app.add_option("--attr", attr_filename, "attribute values filename");

  int ndvr = 15;
  app.add_option("--ndvr", ndvr, "how many dvr iterations to use (default 15)");

  bool punchout;
  std::vector< float > punchout_bounds;
  app.add_option("--punchout", punchout_bounds, "manually specify AABB dimensions (ex \"--punchout min_x min_y min_z max_x max_y max_z\")")->expected(6);

  float cell_size = -1.0f;
  app.add_option("--cellsize", cell_size, "approximate characteristic length for sampling (small value => finer mesh)");

  bool quadratic = false;
  app.add_option("--quadratic", quadratic, "use quadratic tetrahedra");

  bool hemispherical = false;
  app.add_option("--hemi", hemispherical, "use hemispherical sampling");

  CLI11_PARSE(app, argc, argv);

  std::vector<Capsule> capsules = read_binary<Capsule>(input_filename);

  if (capsules.size() < 1) {
    std::cout << "invalid input file: requires at least 1 capsule definition" << std::endl;
    exit(1);
  } else {
    std::cout << "read in " << capsules.size() << " capsules" << std::endl;
  }

  std::vector< int32_t > capsule_attributes;
  if (!attr_filename.empty()) {
    std::string ext = file_extension(output_filename); 
    if (ext != ".vtu") {
      std::cout << "error: attribute output only supported with .vtu files" << std::endl;
      exit(1);
    }
    capsule_attributes = read_binary<int32_t>(attr_filename);
    if (capsule_attributes.size() != capsules.size()) {
      std::cout << "error: attribute array length does not match number of capsules" << std::endl;
      exit(1);
    }
  }

  std::vector< AABB<3>> bounding_boxes(capsules.size());
  for (uint32_t i = 0; i < capsules.size(); i++) {
    bounding_boxes[i] = bounding_box(capsules[i]);
  }
  BVH<3> bvh(bounding_boxes);

  SimplexMesh<3> mesh;

  if (hemispherical) {

    float R_min = +1.0e+10f; // hemisphere inner radius
    float R_max = -1.0e+10f; // hemisphere outer radius
    float r_max = -1.0e+6f;  // filament maximal radius
    for (const auto &c : capsules) {
      std::vector<vec3f> points {c.p1, c.p2};
      for (const auto &p : points) {
        float R = norm(p);
        R_min = std::min(R_min, R);
        R_max = std::max(R_max, R);
      }
      r_max = std::max(r_max, std::max(c.r1, c.r2));
    }
    R_min -= r_max;
    R_max += r_max;

    std::cout << "min, max hemisphere radii:" << R_min << " " << R_max << std::endl;

    float relative_thickness = (R_max / R_min) - 1.0f;
    AABB<3> bounds{
      {-1.1f, -1.1f, -1.1f*relative_thickness}, 
      {+1.1f, +1.1f, +0.1f*relative_thickness}
    };

    if (punchout_bounds.size() == 6) {
      bounds.min[0] = punchout_bounds[0];
      bounds.min[1] = punchout_bounds[1];
      bounds.min[2] = punchout_bounds[2];
      bounds.max[0] = punchout_bounds[3];
      bounds.max[1] = punchout_bounds[4];
      bounds.max[2] = punchout_bounds[5];
    }

    std::cout << "bounding box corners (spherical coordinates): " << bounds.min << ", " << bounds.max << std::endl;

    if (cell_size == -1) {
      cell_size = 2.0 / 100.0;
      std::cout << "--cellsize not specified, using " << R_max * cell_size << std::endl; 
    }

    float dy = 1.5 * R_max * cell_size;

    std::function<float(vec3f)> sdf = [&](vec3f x) -> float {
      vec3f y = hemispherical_mapping(x, R_max);

      AABB<3>box{
        {y[0] - dy, y[1] - dy, y[2] - dy}, 
        {y[0] + dy, y[1] + dy, y[2] + dy}
      };

      float value = dy;
      bvh.query(box, [&](int i) {
        value = std::min(value, capsules[i].SDF(y));
      });
      return value;
    };

    mesh = universal_mesh(sdf, cell_size, bounds, 0.5, 0.05f, ndvr);

    for (auto & x : mesh.vertices) { 
      x = hemispherical_mapping(x, R_max); 
    }

    cell_size *= R_max;

  } else {

    vec3f widths = bvh.global.max - bvh.global.min;
    auto bounds = bvh.global;
    bounds.max += 0.05f * widths;
    bounds.min -= 0.05f * widths;

    if (punchout_bounds.size() == 6) {
      bounds.min[0] = punchout_bounds[0];
      bounds.min[1] = punchout_bounds[1];
      bounds.min[2] = punchout_bounds[2];
      bounds.max[0] = punchout_bounds[3];
      bounds.max[1] = punchout_bounds[4];
      bounds.max[2] = punchout_bounds[5];
    }

    std::cout << "bounding box corners: " << bounds.min << ", " << bounds.max << std::endl;

    if (cell_size == -1) {
      cell_size = std::max(std::max(widths[0], widths[1]), widths[2]) / 100;
      std::cout << "--cellsize not specified, using " << cell_size << std::endl; 
    }

    float dx = 1.5 * cell_size;

    std::function<float(vec3f)> sdf = [&](vec3f x) -> float {

      AABB<3>box{
        {x[0] - dx, x[1] - dx, x[2] - dx}, 
        {x[0] + dx, x[1] + dx, x[2] + dx}
      };

      float value = 2 * dx;
      bvh.query(box, [&](int i) {
        value = std::min(value, capsules[i].SDF(x));
      });
      return value;
    };

    mesh = universal_mesh(sdf, cell_size, bounds, 0.5, 0.05f, ndvr);

    if (quadratic) {
      promote_to_quadratic(mesh, sdf, cell_size);
    }

  }

  std::cout << "generated mesh with:" << std::endl;
  std::cout << "  " << mesh.vertices.size() << " vertices" << std::endl;
  std::cout << "  " << mesh.elements.size() << " elements" << std::endl;
  std::cout << "  " << mesh.boundary_elements.size() << " boundary_elements" << std::endl;

  std::cout << std::endl;
  quality_summary(mesh);
  std::cout << std::endl;

  if (attr_filename.empty()) {
    export_mesh(mesh, output_filename);
  } else {
    auto attributes = cell_values(mesh, capsules, capsule_attributes, cell_size);
    export_vtu(mesh, output_filename, attributes);
  }

}
