#include "geometry.hpp"

#include <iostream>
#include <functional> // for std::hash
#include <unordered_set>

namespace geometry {

// for some reason (even in 2023), C++ still makes us 
// define custom hash functions for std::array, std::tuple, etc 
struct hashfunc {
    template < typename T, size_t n >
    std::size_t operator()(const std::array< T, n > & arr) const {
        std::size_t seed = std::hash<uint32_t>()(arr[0]);
        for (int i = 1; i < n; i++) {
            seed ^= std::hash<uint32_t>()(arr[i]) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
        return seed;
    }
};

bool verify(const SimplexMesh<2> & mesh) {

    using key = std::array<uint64_t, 2>;

    static constexpr std::array< key, 3 > edges = {{{0, 1}, {1, 2}, {2, 0}}};

    // fill the set with our boundary elements
    std::unordered_set<key, hashfunc> bdr(mesh.boundary_elements.begin(), mesh.boundary_elements.end());

    uint32_t count = 0;

    // query the set to find out which tets contain boundary elements
    for (int i = 0; i < mesh.elements.size(); i++) {
        const auto & tri = mesh.elements[i];
        for (int j = 0; j < 3; j++) {
            if (bdr.count(key{tri[edges[j][0]], tri[edges[j][1]]})) {
                count++;
            }
        }
    }

    if (count != mesh.boundary_elements.size()) {
        std::cout << "mesh verification error: expected " << mesh.boundary_elements.size() << " edges, got " << count << std::endl;
        return false;
    } else {
        return true;
    }

}

bool verify(const SimplexMesh<3> & mesh) {

    using key = std::array<uint64_t, 3>;

    static constexpr std::array< key, 4 > faces = {{{2, 1, 0}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}}};

    // fill the set with our boundary elements
    std::unordered_set<key, hashfunc> bdr(mesh.boundary_elements.begin(), mesh.boundary_elements.end());

    uint32_t count = 0;

    // query the set to find out which tets contain boundary elements
    for (int i = 0; i < mesh.elements.size(); i++) {
        const auto & tet = mesh.elements[i];
        for (int j = 0; j < 4; j++) {
            if (bdr.count(key{tet[faces[j][0]], tet[faces[j][1]], tet[faces[j][2]]})) {
                count++;
            }
        }
    }

    if (count != mesh.boundary_elements.size()) {
        std::cout << "mesh verification error: expected " << mesh.boundary_elements.size() << " tris, got " << count << std::endl;
        return false;
    } else {
        return true;
    }

}

}