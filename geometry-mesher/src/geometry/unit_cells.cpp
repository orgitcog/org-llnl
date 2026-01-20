#include "unit_cells.hpp"

namespace geometry {

UnitCell::UnitCell(Type type, float outer_radius, float inner_radius) {

  r_out = outer_radius;
  r_in = inner_radius;

  switch (type) {

    case Type::SC:
        capsules = {
            {{0.0, 0.5, 0.5}, {1.0, 0.5, 0.5}, r_out, r_out}, 
            {{0.5, 0.0, 0.5}, {0.5, 1.0, 0.5}, r_out, r_out}, 
            {{0.5, 0.5, 0.0}, {0.5, 0.5, 1.0}, r_out, r_out}
        };
        break;

    case Type::ISO:
        capsules = {
            {{0.0,1.0,1.0},{1.0,1.0,1.0}, r_out, r_out},
            {{1.0,0.0,1.0},{1.0,1.0,1.0}, r_out, r_out},
            {{1.0,1.0,0.0},{1.0,1.0,1.0}, r_out, r_out},
            {{0.0,0.0,0.0},{1.0,1.0,1.0}, r_out, r_out},
            {{1.0,0.0,0.0},{0.0,1.0,1.0}, r_out, r_out},
            {{1.0,1.0,0.0},{0.0,0.0,1.0}, r_out, r_out},
            {{0.0,1.0,0.0},{1.0,0.0,1.0}, r_out, r_out},
            {{0.0,0.5,0.5},{1.0,0.5,0.5}, r_out, r_out},
            {{0.5,0.0,0.5},{0.5,1.0,0.5}, r_out, r_out},
            {{0.5,0.5,0.0},{0.5,0.5,1.0}, r_out, r_out}
        };
        break;

    case Type::OCTET:
        capsules = {
            {{0.0,0.0,1.0},{1.0,1.0,1.0}, r_out, r_out},
            {{1.0,0.0,1.0},{0.0,1.0,1.0}, r_out, r_out},
            {{1.0,0.0,0.0},{1.0,1.0,1.0}, r_out, r_out},
            {{1.0,1.0,0.0},{1.0,0.0,1.0}, r_out, r_out},
            {{0.0,1.0,0.0},{1.0,1.0,1.0}, r_out, r_out},
            {{1.0,1.0,0.0},{0.0,1.0,1.0}, r_out, r_out},
            {{1.0,0.5,0.5},{0.5,0.5,1.0}, r_out, r_out},
            {{0.5,1.0,0.5},{0.5,0.5,1.0}, r_out, r_out},
            {{1.0,0.5,0.5},{0.5,1.0,0.5}, r_out, r_out}
        };
        break;

    case Type::ORC:
        capsules = {
            {{1.0,1.0,0.5},{1.0,0.5,1.0}, r_out, r_out},
            {{0.5,1.0,1.0},{1.0,1.0,0.5}, r_out, r_out},
            {{1.0,0.5,1.0},{0.5,1.0,1.0}, r_out, r_out}
        };
        break;
    case Type::RD:
        capsules = {
            {{0.5,0.5,1.0},{0.8,0.8,0.8}, r_out, r_out},
            {{0.5,1.0,0.5},{0.8,0.8,0.8}, r_out, r_out},
            {{1.0,0.5,0.5},{0.8,0.8,0.8}, r_out, r_out},
            {{0.8,0.8,0.8},{1.0,1.0,1.0}, r_out, r_out}
        };
        break;
    case Type::TO:
        capsules = {
            {{0.5,0.8,1.0},{0.8,0.5,1.0}, r_out, r_out},
            {{1.0,0.5,0.8},{1.0,0.8,0.5}, r_out, r_out},
            {{0.5,1.0,0.8},{0.8,1.0,0.5}, r_out, r_out},
            {{0.5,0.8,1.0},{0.5,1.0,0.8}, r_out, r_out},
            {{0.8,0.5,1.0},{1.0,0.5,0.8}, r_out, r_out},
            {{0.8,1.0,0.5},{1.0,0.8,0.5}, r_out, r_out}
        };
        break;
    }

}

float UnitCell::SDF(vec3f p) const {
    vec3f x = {
      std::abs(p[0] - 0.5f) + 0.5f,  
      std::abs(p[1] - 0.5f) + 0.5f,  
      std::abs(p[2] - 0.5f) + 0.5f
    };

    float output = 10e10;
    for (const auto & cap : capsules) {
        output = std::min(cap.SDF(x), output);
    }
    return std::max(output, -(output + (r_out - r_in)));
};

}