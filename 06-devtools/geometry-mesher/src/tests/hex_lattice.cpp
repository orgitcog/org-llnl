#include "gtest/gtest.h"

#include "geometry/geometry.hpp"

using namespace geometry;

TEST(hex_lattice, string_constructor_0) { 

    HexLattice lattice({"X"});

    std::cout << lattice.vertices.size() << std::endl;
    std::cout << lattice.edges.size() << std::endl;
    std::cout << lattice.faces.size() << std::endl;
    std::cout << lattice.hexes.size() << std::endl;
    std::cout << lattice.boundary_faces.size() << std::endl;

}

TEST(hex_lattice, string_constructor_1) { 

    HexLattice lattice({
        "   XXX     ",
        "XXXX X XXXX",
        "     XXX   "
    });

    std::cout << lattice.vertices.size() << std::endl;
    std::cout << lattice.edges.size() << std::endl;
    std::cout << lattice.faces.size() << std::endl;
    std::cout << lattice.hexes.size() << std::endl;
    std::cout << lattice.boundary_faces.size() << std::endl;

}

