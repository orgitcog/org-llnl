#include "gtest/gtest.h"

#include "geometry/geometry.hpp"

using namespace geometry;

#define check(cell, p, expected) \
EXPECT_LT(norm(closest_point_projection(cell, p) - expected), 1.0e-8)

TEST(line, nondegenerate_and_degenerate) { 

  for (int i = 0; i < 9; i++) {

    float length = std::pow(10, -i);

    Line line{{{0.0f, 0.0f, 0.0f}, {length, 0.0f, 0.0f}}};

    // closest point is before the first vertex in the line segment
    {
      vec3f projection = closest_point_projection(line, vec3f{-0.5f, 1.0f, 2.0f});
      vec3f expected = {0.0f, 0.0f, 0.0f};
      EXPECT_LT(norm(projection - expected), 1.0e-8);
    }

    // closest point is on the interior of the line segment
    {
      vec3f projection = closest_point_projection(line, vec3f{0.5f * length, 1.0f, 0.0f});
      vec3f expected = {0.5f * length, 0.0f, 0.0f};
      EXPECT_LT(norm(projection - expected), 1.0e-8);
    }

    // closest point is after the second vertex in the line segment
    {
      vec3f projection = closest_point_projection(line, vec3f{1.5f * length, 1.0f, -1.0f});
      vec3f expected = {length, 0.0f, 0.0f};
      EXPECT_LT(norm(projection - expected), 1.0e-8);
    }

  }

}

// the different regions under test for triangles:
//       
//    6      .
//         .     
//   . . o
//       | .   5  
//   3   |   .       .
//       | 4   .  .  
//   . . o------o
//       .      .
//   0   .  1   .     2

TEST(triangle, nondegenerate) { 

  Triangle tri{{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}};

  // index corresponds to region numbers indicated above
  vec3f p[7] = {
    {-0.5f, -0.5f, 1.0f},
    { 0.5f, -0.5f, 1.0f},
    { 1.5f, -0.5f, 1.0f},
    {-0.5f,  0.5f, 1.0f},
    { 0.2f,  0.2f, 1.0f},
    { 1.0f,  1.0f, 1.0f},
    {-0.5f,  1.5f, 1.0f}
  };

  vec3f expected[7] = {
    {0.0f, 0.0f, 0.0f},
    {0.5f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f},
    {0.0f, 0.5f, 0.0f},
    {0.2f, 0.2f, 0.0f},
    {0.5f, 0.5f, 0.0f},
    {0.0f, 1.0f, 0.0f}
  };

  for (int region = 0; region < 7; region++) {
    check(tri, p[region], expected[region]) << "error in region " << region;
  }

}

#if 1
TEST(triangle, degenerate_line) { 

  for (int i = 0; i < 10; i++) {

    float length = std::pow(10, -i);

    Triangle tri = {{{0.0f, 0.0f, 0.0f}, {length, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}};

    // index corresponds to region numbers indicated above
    vec3f expected[7] = {
      {0.0f         , 0.0f, 0.0f},
      {0.5f * length, 0.0f, 0.0f},
      {1.0f * length, 0.0f, 0.0f},
      {0.0f         , 0.5f, 0.0f},
      {0.2f * length, 0.2f, 0.0f},
      {0.5f * length, 0.5f, 0.0f},
      {0.0f         , 1.0f, 0.0f}
    };

    vec3f p[7] = {
      expected[0] + vec3f{-1.0f, -1.0f, 1.5f},
      expected[1] + vec3f{ 0.0f, -1.0f, 1.5f},
      expected[2] + vec3f{ 1.0f, -1.0f, 1.5f},
      expected[3] + vec3f{-1.0f,  0.0f, 1.5f},
      expected[4] + vec3f{ 0.0f,  0.0f, 1.5f},
      expected[5] + vec3f{ 1.0f, length, 1.5f},
      expected[6] + vec3f{-1.0f,  1.0f, 1.5f}
    };

    for (int k = 0; k < 3; k++) {
      for (int region = 0; region < 7; region++) {
        check(tri, p[region], expected[region])
          << "error in region " << region
          << ", orientation " << k
          << ", length = " << length;
      }

      // rotate the triangle to make sure the projection is invariant wrt orientation
      tri = Triangle{tri.vertices[1], tri.vertices[2], tri.vertices[0]};
    }

  }

}

TEST(triangle, degenerate_point) { 

  for (int i = 0; i < 10; i++) {

    float length = std::pow(10, -i);

    Triangle tri = {{{0.0f, 0.0f, 0.0f}, {length, 0.0f, 0.0f}, {0.0f, length, 0.0f}}};

    // index corresponds to region numbers indicated above
    vec3f expected[7] = {
      {0.0f         , 0.0f         , 0.0f},
      {0.5f * length, 0.0f         , 0.0f},
      {1.0f * length, 0.0f         , 0.0f},
      {0.0f         , 0.5f * length, 0.0f},
      {0.2f * length, 0.2f * length, 0.0f},
      {0.5f * length, 0.5f * length, 0.0f},
      {0.0f         , 1.0f * length, 0.0f}
    };

    vec3f p[7] = {
      expected[0] + vec3f{-1.0f, -1.0f, 1.5f},
      expected[1] + vec3f{ 0.0f, -1.0f, 1.5f},
      expected[2] + vec3f{ 1.0f, -1.0f, 1.5f},
      expected[3] + vec3f{-1.0f,  0.0f, 1.5f},
      expected[4] + vec3f{ 0.0f,  0.0f, 1.5f},
      expected[5] + vec3f{ 1.0f,  1.0f, 1.5f},
      expected[6] + vec3f{-1.0f,  1.0f, 1.5f}
    };

    for (int k = 0; k < 3; k++) {
      for (int region = 0; region < 7; region++) {
        check(tri, p[region], expected[region])
          << "error in region " << region
          << ", orientation " << k
          << ", length = " << length;
      }

      // rotate the triangle to make sure the projection is invariant wrt orientation
      tri = Triangle{tri.vertices[1], tri.vertices[2], tri.vertices[0]};
    }

  }

}
#endif