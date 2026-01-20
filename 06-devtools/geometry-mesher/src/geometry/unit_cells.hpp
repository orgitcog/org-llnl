#include "geometry.hpp"

namespace geometry {

struct UnitCell {

    enum Type { SC, ISO, OCTET, ORC, RD, TO };

    UnitCell(Type t, float outer_radius, float inner_radius = 0.0);

    float SDF(vec3f p) const;

    float r_in;
    float r_out;
    std::vector< Capsule > capsules;

};

}