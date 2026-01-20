#pragma once

#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "opengl/Shader.hpp"
#include "opengl/RGBColor.hpp"
#include "geometry/geometry.hpp"

using namespace geometry;

struct Palette {
  glm::vec3 colors[4];
};

struct VertexWithColor {
  vec3f position;
  rgbcolor color;
};

struct VertexWithValue {
  vec3f position;
  float value;
};

struct LineWithColors {
  VertexWithColor vertices[2];
};

constexpr LineWithColors add_color(Line L, rgbcolor color) {
  return {{{L.vertices[0], color}, {L.vertices[1], color}}};
}

struct LineWithValue {
  VertexWithValue vertices[2];
};

struct TriangleWithValues {
  VertexWithValue vertices[3];
};

struct TriangleWithColors {
  VertexWithColor vertices[3];
};

constexpr TriangleWithColors add_color(Triangle T, rgbcolor color) {
  return {{{T.vertices[0], color}, {T.vertices[1], color}, {T.vertices[2], color}}};
}

struct Scene {

  template < typename T >
  struct Primitive {
    Primitive(ShaderProgram & program) : VAO(0), VBO(0), dirty(false) {
      glGenVertexArrays(1, &VAO);
      glBindVertexArray(VAO);
      glGenBuffers(1, &VBO);
      glBindBuffer(GL_ARRAY_BUFFER, VBO);
      program.setAttribute("vert", 3, sizeof(VertexWithColor), 0);
      program.setAttribute("rgba", 4, sizeof(VertexWithColor), offsetof(VertexWithColor, color), GL_TRUE, GL_UNSIGNED_BYTE);
    }

    GLuint VAO, VBO;
    std::vector < T > data;
    bool dirty;
  };

  Palette palette;
  rgbcolor color;

  ShaderProgram program;

  Primitive< LineWithColors > lines;
  Primitive< TriangleWithColors > tris;

  Scene(); 

  void push_back(const Line & line);
  void push_back(const std::vector< Line > & more_lines);

  void push_back(const LineWithColors & line);
  void push_back(const std::vector< LineWithColors > & more_lines);

  void push_back(const Triangle & tri);
  void push_back(const std::vector< Triangle > & more_tris);

  void push_back(const TriangleWithColors & tri);
  void push_back(const std::vector< TriangleWithColors > & more_tris);

  void clear();

  void draw(const glm::mat4 & proj);
  void draw_wireframe(const glm::mat4 & proj);

};