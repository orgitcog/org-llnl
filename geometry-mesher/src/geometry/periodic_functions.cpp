#include "fm/types/vec.hpp"

#include <cmath>

namespace geometry {

using namespace fm;

// see https://www.smish.dev/math/triply_periodic_surfaces/
namespace tpms {
  float gyroid(vec3f x) {
    return 0.652121f * (sinf(x[0]) * cosf(x[1]) + sinf(x[1]) * cosf(x[2]) + sinf(x[2]) * cosf(x[0]));
  }

  float schwarz_p(vec3f x) {
    return 0.747041f * cosf(x[0]) + cosf(x[1]) + cosf(x[2]);
  }

  float schwarz_d(vec3f x) {
    float c[3] = {cosf(x[0]), cosf(x[1]), cosf(x[2])};
    float s[3] = {sinf(x[0]), sinf(x[1]), sinf(x[2])};
    return 0.667712f * (s[0] * s[1] * s[2] + 
                        s[0] * c[1] * c[2] + 
                        c[0] * s[1] * c[2] + 
                        c[0] * c[1] * s[2]);
  }

  float neovius(vec3f x) {
    float c[3] = {cosf(x[0]), cosf(x[1]), cosf(x[2])};
    return 1.0404f * (c[0] + c[1] + c[2]) + 1.3872f * c[0] * c[1] * c[2];
  }

  float schoen_iwp(vec3f x) {
    float c[3] = {cosf(x[0]), cosf(x[1]), cosf(x[2])};
    float c2[3] = {cosf(2.0f * x[0]), cosf(2.0f * x[1]), cosf(2.0f * x[2])};
    return 0.460268f * (c[0]*c[1] + c[1]*c[2] + c[2]*c[0]) - 0.230134f * (c2[0] + c2[1] + c2[2]);
  }

  float fischer_koch_s(vec3f x) {
    float s[3] = {sinf(x[0]), sinf(x[1]), sinf(x[2])};
    float c[3] = {cosf(x[0]), cosf(x[1]), cosf(x[2])};
    float c2[3] = {c[0]*c[0]-s[0]*s[0], c[1]*c[1]-s[1]*s[1], c[2]*c[2]-s[2]*s[2]};
    return 0.554025f * (c2[0] *  s[1] *  c[2] + 
                         c[0] * c2[1] *  s[2] + 
                         s[0] *  c[1] * c2[2]);
  }

  float fischer_koch_y(vec3f x) {
    float s[3] = {sinf(x[0]), sinf(x[1]), sinf(x[2])};
    float c[3] = {cosf(x[0]), cosf(x[1]), cosf(x[2])};
    float s2[3] = {2*c[0]*s[0], 2*c[1]*s[1], 2*c[2]*s[2]};
    return 0.885124f * c[0]*c[1]*c[2] + 0.442562f * (s2[0]*s[1] + s2[1]*s[2] + s2[2]*s[0]);
  }

  float fischer_koch_cp(vec3f x) {
    float c[3] = {cosf(x[0]), cosf(x[1]), cosf(x[2])};
    return 0.409236f * (c[0]+c[1]+c[2]) + 1.63694 * c[0]*c[1]*c[2];
  }
}

}