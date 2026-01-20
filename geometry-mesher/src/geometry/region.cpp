#include "region.hpp"

namespace geometry {

////////////////////////////////////////////////////////////////////////////////

MeshRegion::MeshRegion(const std::vector< vec3f > & vertex_coordinates, const std::vector< tet > & tetrahedra) : Region(true) {
}

MeshRegion::MeshRegion(const std::vector< vec3f > & vertex_coordinates, const std::vector< hex > & hexahedra) : Region(true) {
}

float MeshRegion::SDF(vec3f x, float rmax) const {
    return 1.0;
};

void MeshRegion::factor() const {
    // TODO
}

////////////////////////////////////////////////////////////////////////////////

PrimitiveRegion::PrimitiveRegion(const std::vector< Ball > & b) : Region(true), balls(b), boxes{}, capsules{} {}
PrimitiveRegion::PrimitiveRegion(const std::vector< AABB<3> > & b) : Region(true), balls{}, boxes{b}, capsules{} {}
PrimitiveRegion::PrimitiveRegion(const std::vector< Capsule > & c)  : Region(true), balls{}, boxes{}, capsules{c} {}

float PrimitiveRegion::SDF(vec3f x, float rmax) const {
    float sdf = rmax;

    // TODO
    // TODO
    for (const auto & ball : balls) {
      sdf = std::min(ball.SDF(x), sdf);
    }

    for (const auto & cap : capsules) {
      sdf = std::min(cap.SDF(x), sdf);
    }

    for (const auto & box : boxes) {
      sdf = std::min(box.SDF(x), sdf);
    }
    // TODO
    // TODO

    return sdf;
};

void PrimitiveRegion::factor() const {
    // TODO
}

////////////////////////////////////////////////////////////////////////////////

BooleanRegion::BooleanRegion(BooleanOperation op, const Region & A, const Region & B) : Region(false) {


}

float BooleanRegion::SDF(vec3f x, float rmax) const {
    float sdf = children[0]->SDF(x, rmax);
    for (int i = 1; i < children.size(); i++) {
        float value = children[i]->SDF(x, rmax);
        switch (ops[i-1]) {
            case BooleanOperation::DIFFERENCE: return std::max(sdf, -value);
            case BooleanOperation::INTERSECTION: return std::max(sdf, value);
            case BooleanOperation::UNION: return std::min(sdf, -value);
        }
    }
    return sdf;
};

void BooleanRegion::factor() const {
    for (auto & child : children) { child->factor(); }
}

////////////////////////////////////////////////////////////////////////////////

BooleanRegion operator+(const Region & A, const Region & B) {
    return BooleanRegion{BooleanOperation::UNION, A, B};
}

BooleanRegion operator-(const Region & A, const Region & B) {
    return BooleanRegion{BooleanOperation::DIFFERENCE, A, B};
}

BooleanRegion operator*(const Region & A, const Region & B) {
    return BooleanRegion{BooleanOperation::INTERSECTION, A, B};
}

////////////////////////////////////////////////////////////////////////////////

}