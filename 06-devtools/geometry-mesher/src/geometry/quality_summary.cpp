#include "geometry.hpp"

#include <iostream>

namespace geometry {

void quality_summary(const SimplexMesh<3> & mesh) {

    int negative_elements = 0;
    double qmin = 1.0;
    double qmax = 0.0;
    double qavg = 0.0;
    double qdev = 0.0;

    auto & x = mesh.vertices;

    for (auto [i,j,k,l] : mesh.elements) {
        double q = quality(Tetrahedron{x[i], x[j], x[k], x[l]});
        qmin = std::min(q, qmin);
        qmax = std::max(q, qmax);
        qavg += q;
        qdev += q*q;

        negative_elements += q < 0;
    }

    uint32_t n = mesh.elements.size();
    qavg /= n;
    qdev = sqrt(qdev/n - qavg * qavg);

    std::cout << "mesh quality report: " << std::endl;
    std::cout << negative_elements << " inverted elements" << std::endl;
    std::cout << "min: " << qmin << std::endl;
    std::cout << "avg±dev: " << qavg << " ± " << qdev << std::endl;
    std::cout << "max: " << qmax << std::endl;

}

}