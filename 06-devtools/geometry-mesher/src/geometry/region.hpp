#pragma once

#include <array>
#include <vector>
#include <cinttypes>

#include "geometry.hpp"

namespace geometry {

struct Region {
    Region(bool leaf) : is_leaf(leaf) {}
    virtual float SDF(vec3f p, float rmax) const = 0;
    virtual void factor() const = 0;
    bool is_leaf;
};
 
struct MeshRegion : public Region {
    using edge = std::array< uint32_t, 2 >;
    using tri = std::array< uint32_t, 3 >;
    using quad = std::array< uint32_t, 4 >;
    using tet = std::array< uint32_t, 4 >;
    using hex = std::array< uint32_t, 8 >;

    std::vector < vec3f > vertices;

    std::vector < tet > tets;
    std::vector < hex > hexes;

    std::vector < tri > bdr_tris;
    std::vector < quad > bdr_quads;

    std::vector < edge > edges;

    MeshRegion(const std::vector< vec3f > & vertex_coordinates, const std::vector< tet > & tetrahedra);
    MeshRegion(const std::vector< vec3f > & vertex_coordinates, const std::vector< hex > & hexahedra);

    float SDF(vec3f x, float rmax) const final;
    void factor() const final;
};

struct PrimitiveRegion : public Region {
    std::vector < Ball > balls;
    std::vector < AABB<3> > boxes;
    std::vector < Capsule > capsules;

    PrimitiveRegion(const std::vector< Ball > & primitives);
    PrimitiveRegion(const std::vector< AABB<3> > & primitives);
    PrimitiveRegion(const std::vector< Capsule > & primitives);

    float SDF(vec3f x, float rmax) const final;
    void factor() const final;
};

enum class BooleanOperation {DIFFERENCE, INTERSECTION, UNION};

struct BooleanRegion : public Region {
    std::vector< BooleanOperation > ops;
    std::vector< Region * > children;

    BooleanRegion(BooleanOperation o, const Region & A, const Region & B);

    float SDF(vec3f x, float rmax) const final;
    void factor() const final;
};

BooleanRegion operator+(const Region & A, const Region & B); // union
BooleanRegion operator-(const Region & A, const Region & B); // difference
BooleanRegion operator*(const Region & A, const Region & B); // intersection

}