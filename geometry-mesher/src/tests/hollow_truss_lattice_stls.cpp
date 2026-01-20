#include <vector>

#include "BVH.hpp"
#include "geometry/geometry.hpp"
#include "geometry/unit_cells.hpp"

using namespace geometry;

struct OrientedOctetTruss {

  vec3f dimensions;
  float thickness;
  std::vector< Capsule > capsules;

  OrientedOctetTruss(int orientation, float r_in, float r_out) {

    thickness = r_out - r_in;

    switch (orientation) {
      case 110:
        dimensions = {0.7071067812f, 1.000000000f, 0.7071067812f};
        capsules = {
            {{-0.35355340,0.50000000,-0.35355339},{0.35355339,0.50000000,-0.35355339}, r_out, r_out},
            {{-0.35355340,0.50000000,0.35355339},{-0.35355340,0.50000000,-0.35355339}, r_out, r_out},
            {{0,0,0},{0.35355339,0.50000000,-0.35355339}, r_out, r_out},
            {{0,0,0},{-0.35355340,0.50000000,0.35355339}, r_out, r_out},
            {{0,0,0},{-0.35355340,0.50000000,-0.35355339}, r_out, r_out},
            {{-0.35355340,0.50000000,0.35355339},{0,1.0000000,0}, r_out, r_out},
            {{0.35355339,0.50000000,-0.35355339},{0,1.0000000,0}, r_out, r_out},
            {{-0.35355340,0.50000000,-0.35355339},{0,1.0000000,0}, r_out, r_out},
            {{0,0,0.70710678},{-0.35355340,0.50000000,0.35355339}, r_out, r_out},
            {{-0.35355340,0.50000000,0.35355339},{0,1.0000000,0.70710678}, r_out, r_out},
            {{-0.35355340,0.50000000,0.35355339},{0.35355339,0.50000000,0.35355339}, r_out, r_out},
            {{0.35355339,0.50000000,0.35355339},{0.35355339,0.50000000,-0.35355339}, r_out, r_out},
            {{0.70710678,0,0},{0.35355339,0.50000000,-0.35355339}, r_out, r_out},
            {{0,0,0},{0.35355339,0.50000000,0.35355339}, r_out, r_out},
            {{0.70710678,0,0.70710678},{0.35355339,0.50000000,0.35355339}, r_out, r_out},
            {{0,0,0.70710678},{0.35355339,0.50000000,0.35355339}, r_out, r_out},
            {{0.70710678,0,0},{0.35355339,0.50000000,0.35355339}, r_out, r_out},
            {{0,0,0.70710678},{0.70710678,0,0.70710678}, r_out, r_out},
            {{0.70710678,0,0.70710678},{0.70710678,0,0}, r_out, r_out},
            {{0.70710678,0,0},{0,0,0}, r_out, r_out},
            {{0,0,0},{0,0,0.70710678}, r_out, r_out},
            {{0.35355339,0.50000000,-0.35355339},{0.70710678,1.0000000,0}, r_out, r_out},
            {{0.35355339,0.50000000,0.35355339},{0,1.0000000,0}, r_out, r_out},
            {{0.35355339,0.50000000,0.35355339},{0.70710678,1.0000000,0.70710678}, r_out, r_out},
            {{0.35355339,0.50000000,0.35355339},{0,1.0000000,0.70710678}, r_out, r_out},
            {{0.35355339,0.50000000,0.35355339},{0.70710678,1.0000000,0}, r_out, r_out},
            {{0,1.0000000,0.70710678},{0.70710678,1.0000000,0.70710678}, r_out, r_out},
            {{0.70710678,1.0000000,0.70710678},{0.70710678,1.0000000,0}, r_out, r_out},
            {{0.70710678,1.0000000,0},{0,1.0000000,0}, r_out, r_out},
            {{0,1.0000000,0},{0,1.0000000,0.70710678}, r_out, r_out}
        };
        break;
      case 111:
        dimensions = {sqrtf(3.0f / 2.0f), sqrtf(2.0f) / 2.0f, sqrtf(3.0f)};
        capsules = {
            {{0.61237244,0.35355339,0},{0.40824829,0,0.57735030}, r_out, r_out},
            {{0,0.70710678,0},{0.61237244,0.35355339,0}, r_out, r_out},
            {{0.61237244,0.35355339,0},{1.2247449,0,0}, r_out, r_out},
            {{0,0,0},{0.40824829,0,0.57735030}, r_out, r_out},
            {{0,0,0},{0,0.70710678,0}, r_out, r_out},
            {{0,0,0},{0.61237244,0.35355339,0}, r_out, r_out},
            {{1.0206207,1.0606602,0.57735030},{0.81649658,0.70710678,1.1547005}, r_out, r_out},
            {{1.2247449,0,0},{1.2247449,0.70710678,0}, r_out, r_out},
            {{0,0.70710678,0},{0.40824829,0.70710678,0.57735030}, r_out, r_out},
            {{0.40824829,0.70710678,0.57735030},{0.81649658,0.70710678,1.1547005}, r_out, r_out},
            {{0.40824829,0,0.57735030},{0.40824829,0.70710678,0.57735030}, r_out, r_out},
            {{0,0.70710678,0},{0.61237244,1.0606602,0}, r_out, r_out},
            {{1.2247449,0,0},{1.0206207,0.35355339,0.57735030}, r_out, r_out},
            {{1.0206207,0.35355339,0.57735030},{0.81649658,0.70710678,1.1547005}, r_out, r_out},
            {{0.40824829,0,0.57735030},{1.0206207,0.35355339,0.57735030}, r_out, r_out},
            {{1.2247449,0.70710678,0},{1.0206207,1.0606602,0.57735030}, r_out, r_out},
            {{0.40824829,0.70710678,0.57735030},{1.0206207,1.0606602,0.57735030}, r_out, r_out},
            {{0.61237244,1.0606602,0},{1.0206207,1.0606602,0.57735030}, r_out, r_out},
            {{1.0206207,0.35355339,0.57735030},{1.0206207,1.0606602,0.57735030}, r_out, r_out},
            {{0.61237244,0.35355339,0},{1.2247449,0.70710678,0}, r_out, r_out},
            {{0.61237244,0.35355339,0},{0.40824829,0.70710678,0.57735030}, r_out, r_out},
            {{0.61237244,0.35355339,0},{0.61237244,1.0606602,0}, r_out, r_out},
            {{0.61237244,0.35355339,0},{1.0206207,0.35355339,0.57735030}, r_out, r_out},
            {{0.61237244,1.0606602,0},{0.40824829,0.70710678,0.57735030}, r_out, r_out},
            {{0.40824829,0.70710678,0.57735030},{1.0206207,0.35355339,0.57735030}, r_out, r_out},
            {{1.0206207,0.35355339,0.57735030},{1.2247449,0.70710678,0}, r_out, r_out},
            {{1.2247449,0.70710678,0},{0.61237244,1.0606602,0}, r_out, r_out},
            {{0.40824829,0,0.57735030},{0.81649658,0,1.1547005}, r_out, r_out},
            {{0.81649658,0,1.1547005},{1.2247449,0,1.7320508}, r_out, r_out},
            {{0.81649658,0,1.1547005},{0.81649658,0.70710678,1.1547005}, r_out, r_out},
            {{1.0206207,0.35355339,0.57735030},{0.81649658,0,1.1547005}, r_out, r_out},
            {{0.81649658,0.70710678,1.1547005},{1.2247449,0.70710678,1.7320508}, r_out, r_out},
            {{1.2247449,0,1.7320508},{1.2247449,0.70710678,1.7320508}, r_out, r_out},
            {{0,0.70710678,0},{-0.20412410,0.35355339,0.57735030}, r_out, r_out},
            {{-0.20412410,0.35355339,0.57735030},{-0.40824830,0,1.1547005}, r_out, r_out},
            {{-0.20412410,0.35355339,0.57735030},{0.40824829,0,0.57735030}, r_out, r_out},
            {{0,0,0},{-0.20412410,0.35355339,0.57735030}, r_out, r_out},
            {{0.20412415,1.0606602,1.1547005},{0,0.70710678,1.7320508}, r_out, r_out},
            {{0.20412415,1.0606602,1.1547005},{0.81649658,0.70710678,1.1547005}, r_out, r_out},
            {{-0.40824830,0.70710678,1.1547005},{0,0.70710678,1.7320508}, r_out, r_out},
            {{-0.40824830,0,1.1547005},{-0.40824830,0.70710678,1.1547005}, r_out, r_out},
            {{0,0.70710678,0},{-0.20412410,1.0606602,0.57735030}, r_out, r_out},
            {{0.40824829,0,0.57735030},{0.20412415,0.35355339,1.1547005}, r_out, r_out},
            {{0.20412415,0.35355339,1.1547005},{0,0.70710678,1.7320508}, r_out, r_out},
            {{-0.40824830,0,1.1547005},{0.20412415,0.35355339,1.1547005}, r_out, r_out},
            {{0.20412415,0.35355339,1.1547005},{0.81649658,0.70710678,1.1547005}, r_out, r_out},
            {{0.40824829,0.70710678,0.57735030},{0.20412415,1.0606602,1.1547005}, r_out, r_out},
            {{-0.40824830,0.70710678,1.1547005},{0.20412415,1.0606602,1.1547005}, r_out, r_out},
            {{-0.20412410,1.0606602,0.57735030},{0.20412415,1.0606602,1.1547005}, r_out, r_out},
            {{0.20412415,0.35355339,1.1547005},{0.20412415,1.0606602,1.1547005}, r_out, r_out},
            {{-0.20412410,0.35355339,0.57735030},{0.40824829,0.70710678,0.57735030}, r_out, r_out},
            {{-0.20412410,0.35355339,0.57735030},{-0.40824830,0.70710678,1.1547005}, r_out, r_out},
            {{-0.20412410,0.35355339,0.57735030},{-0.20412410,1.0606602,0.57735030}, r_out, r_out},
            {{-0.20412410,0.35355339,0.57735030},{0.20412415,0.35355339,1.1547005}, r_out, r_out},
            {{-0.20412410,1.0606602,0.57735030},{-0.40824830,0.70710678,1.1547005}, r_out, r_out},
            {{-0.40824830,0.70710678,1.1547005},{0.20412415,0.35355339,1.1547005}, r_out, r_out},
            {{0.20412415,0.35355339,1.1547005},{0.40824829,0.70710678,0.57735030}, r_out, r_out},
            {{0.40824829,0.70710678,0.57735030},{-0.20412410,1.0606602,0.57735030}, r_out, r_out},
            {{0.81649658,0.70710678,1.1547005},{0.61237244,1.0606602,1.7320508}, r_out, r_out},
            {{0,0.70710678,1.7320508},{0.61237244,1.0606602,1.7320508}, r_out, r_out},
            {{0.20412415,1.0606602,1.1547005},{0.61237244,1.0606602,1.7320508}, r_out, r_out},
            {{0.81649658,0.70710678,1.1547005},{0.61237244,0.35355339,1.7320508}, r_out, r_out},
            {{0,0.70710678,1.7320508},{0.61237244,0.35355339,1.7320508}, r_out, r_out},
            {{0.61237244,0.35355339,1.7320508},{1.2247449,0,1.7320508}, r_out, r_out},
            {{-0.40824830,0,1.1547005},{0,0,1.7320508}, r_out, r_out},
            {{0,0,1.7320508},{0,0.70710678,1.7320508}, r_out, r_out},
            {{0.81649658,0,1.1547005},{0.61237244,0.35355339,1.7320508}, r_out, r_out},
            {{0,0,1.7320508},{0.61237244,0.35355339,1.7320508}, r_out, r_out},
            {{0.20412415,0.35355339,1.1547005},{0.61237244,0.35355339,1.7320508}, r_out, r_out},
            {{0.20412415,0.35355339,1.1547005},{0,0,1.7320508}, r_out, r_out},
            {{0.81649658,0,1.1547005},{0.20412415,0.35355339,1.1547005}, r_out, r_out},
            {{0.61237244,0.35355339,1.7320508},{1.2247449,0.70710678,1.7320508}, r_out, r_out},
            {{0.61237244,0.35355339,1.7320508},{0.61237244,1.0606602,1.7320508}, r_out, r_out},
            {{1.2247449,0.70710678,1.7320508},{0.61237244,1.0606602,1.7320508}, r_out, r_out}
        };
        break;
    }
  }
};
        
void mesh_and_export_stl(OrientedOctetTruss octet, int n, std::array<int, 3> blocks, std::string filename) {

    std::vector< Capsule > cylinders;
    std::vector< AABB<3>> bounding_boxes;
    cylinders.reserve(blocks[0] * blocks[1] * blocks[2] * octet.capsules.size());
    bounding_boxes.reserve(cylinders.size());

    vec3f dim = octet.dimensions;

    for (int i = 0; i < blocks[0]; i++) {
        for (int j = 0; j < blocks[1]; j++) {
            for (int k = 0; k < blocks[2]; k++) {
                vec3f offset{i * dim[0], j * dim[1], k * dim[2]};

                for (auto copy : octet.capsules) {
                    copy.p1 += offset;
                    copy.p2 += offset;
                    cylinders.push_back(copy);
                    bounding_boxes.push_back(bounding_box(cylinders.back()));
                }
            }
        }
    }

    BVH bvh(bounding_boxes);

    AABB clipping_box = bvh.global;
    clipping_box.min[2] = 0.0;
    clipping_box.max[2] -= cylinders[0].r2;

    auto widths = bvh.global.max - bvh.global.min;

    float dx = 1.25 * std::max(widths[0], std::max(widths[1], widths[2])) / n;

    std::function<float(vec3f)> f = [&](vec3f x) { 
      AABB<3>box{
        {x[0] - dx, x[1] - dx, x[2] - dx}, 
        {x[0] + dx, x[1] + dx, x[2] + dx}
      };

      #if 1
      float value = 1.0e10;
      bvh.query(box, [&](int i) {
        value = std::min(cylinders[i].SDF(x), value);
      });
      #else
      float blend_distance = 0.02 * dx;
      float value = 0.0;
      bvh.query(box, [&](int i) {
        value += exp(-cylinders[i].SDF(x) / blend_distance);
      });
      value = -blend_distance * logf(value);
      #endif

      //return std::max(clipping_box.SDF(x), std::max(value, -(value + octet.thickness)));
      return std::max(value, -(value + octet.thickness));
    };

    AABB<3>sampling_bounds = bvh.global;
    sampling_bounds.min -= {1.5f * dx, 1.5f * dx, 1.5f * dx};
    sampling_bounds.max += {1.5f * dx, 1.5f * dx, 1.5f * dx};

    auto mesh = universal_mesh(f, dx, sampling_bounds);

    export_stl(mesh, filename);

}

int main() {

    // rho = 0.15, diameter ratio = 0.0, outer_diameter = 0.1163
    {
      float r_out = 0.1163f / 2.0;
      float r_in = 0.0f;
      mesh_and_export_stl(OrientedOctetTruss(110, r_in, r_out), 120, {2, 2, 2}, "rho_15_orientation_110_solid");

      //mesh_and_export_stl(OrientedOctetTruss(111, r_in, r_out), 257, {3, 4, 2}, "rho_15_orientation_111_solid.stl");
      //mesh_and_export_stl(OrientedOctetTruss(110, r_in, r_out), 257, {4, 4, 4}, "rho_15_orientation_110_solid.stl");
      //mesh_and_export_stl(OrientedOctetTruss(111, r_in, r_out), 257, {3, 4, 2}, "rho_15_orientation_111_solid.stl");
    }

#if 0
    // rho = 0.15, diameter ratio = 0.7, outer_diameter = 0.1821
    {
      float r_out = 0.1821f / 2.0;
      float r_in = 0.7f * r_out;

      mesh_and_export_stl(OrientedOctetTruss(110, r_in, r_out), 400, {4, 4, 4}, "rho_15_orientation_110_hollow_07_coarse.stl");
      mesh_and_export_stl(OrientedOctetTruss(111, r_in, r_out), 300, {3, 4, 2}, "rho_15_orientation_111_hollow_07_coarse.stl");

      mesh_and_export_stl(OrientedOctetTruss(110, r_in, r_out), 500, {4, 4, 4}, "rho_15_orientation_110_hollow_07.stl");
      mesh_and_export_stl(OrientedOctetTruss(111, r_in, r_out), 400, {3, 4, 2}, "rho_15_orientation_111_hollow_07.stl");
    }

    // rho = 0.05, diameter ratio = 0.0, outer_diameter = 0.0643
    //{
    //  float r_out = 0.0643f / 2.0;
    //  float r_in = 0.0f;
    //  mesh_and_export_stl(OrientedOctetTruss(110, r_in, r_out), 257, {4, 4, 4}, "rho_05_orientation_110_solid.stl");
    //  mesh_and_export_stl(OrientedOctetTruss(111, r_in, r_out), 257, {3, 4, 2}, "rho_05_orientation_111_solid.stl");
    //}

    // rho = 0.05, diameter ratio = 0.8, outer_diameter = 0.1158
    {
      float r_out = 0.1158f / 2.0;
      float r_in = 0.8f * r_out;
      mesh_and_export_stl(OrientedOctetTruss(110, r_in, r_out), 384, {4, 4, 4}, "rho_05_orientation_110_hollow_08_coarse.stl");
      mesh_and_export_stl(OrientedOctetTruss(111, r_in, r_out), 384, {3, 4, 2}, "rho_05_orientation_111_hollow_08_coarse.stl");

      mesh_and_export_stl(OrientedOctetTruss(110, r_in, r_out), 512, {4, 4, 4}, "rho_05_orientation_110_hollow_08.stl");
      mesh_and_export_stl(OrientedOctetTruss(111, r_in, r_out), 512, {3, 4, 2}, "rho_05_orientation_111_hollow_08.stl");
    }

#if 0
    // rho = 0.05, diameter ratio = 0.9, outer_diameter = 0.1748
    {
      float r_out = 0.1748f / 2.0;
      float r_in = 0.9f * r_out;
      mesh_and_export_stl(OrientedOctetTruss(110, r_in, r_out), 256, {4, 4, 4}, "rho_05_orientation_110_solid.stl");
      mesh_and_export_stl(OrientedOctetTruss(111, r_in, r_out), 256, {3, 4, 2}, "rho_05_orientation_111_solid.stl");
    }
#endif
#endif

}
