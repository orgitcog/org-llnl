#include "geometry.hpp"

namespace geometry {

vec3f cotan_weights(const Triangle & tri) {
    vec3f e0 = tri.vertices[1] - tri.vertices[0];
    vec3f e1 = tri.vertices[2] - tri.vertices[1];
    vec3f e2 = tri.vertices[0] - tri.vertices[2];

    // note: cot(theta) = cos(theta) / sin(theta)
    //               but since
    //     dot(a, b)  === |a| |b| cos(theta)
    //  |cross(a, b)| === |a| |b| sin(theta) 
    // 
    //  we can take a ratio of dot/cross and 
    //  completely avoid expensive trig functions
    return {
        -dot(e0, e2) / norm(cross(e0, e2)),
        -dot(e1, e0) / norm(cross(e1, e0)),
        -dot(e2, e1) / norm(cross(e2, e1))
    };
}

void laplace_smoothing(SimplexMesh<3> & mesh, float lambda) {

    auto & x = mesh.vertices;

    std::vector< float > w(mesh.vertices.size(), 0.0);
    std::vector< vec3f > s(mesh.vertices.size(), vec3f{});

    for (auto & [i, j, k] : mesh.boundary_elements) {
        auto c = cotan_weights(Triangle{x[i], x[j], x[k]});

        s[i] += (c[1] * x[k] + c[2] * x[j]);
        w[i] += (c[1] + c[2]);

        s[j] += (c[0] * x[k] + c[2] * x[i]);
        w[j] += (c[0] + c[2]);

        s[k] += (c[0] * x[j] + c[1] * x[i]);
        w[k] += (c[0] + c[1]);
    }

    for (uint32_t i = 0; i < mesh.vertices.size(); i++) {
        if (w[i] > 0) {
            x[i] += lambda * ((s[i] / w[i]) - x[i]);
        }
    }

}

}