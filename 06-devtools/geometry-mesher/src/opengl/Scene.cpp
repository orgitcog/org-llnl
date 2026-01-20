#include "Scene.hpp"

[[maybe_unused]] static Palette grayscale = {
  glm::vec3(240.0f / 255.0f, 240.0 / 255.0f, 240.0f / 255.0f),
  glm::vec3(160.0f / 255.0f, 160.0 / 255.0f, 160.0f / 255.0f),
  glm::vec3( 80.0f / 255.0f,  80.0 / 255.0f,  80.0f / 255.0f),
  glm::vec3(  0.0f / 255.0f,   0.0 / 255.0f,   0.0f / 255.0f)
};

[[maybe_unused]] static Palette light_gray = {
  glm::vec3(120.0f / 255.0f, 120.0 / 255.0f, 120.0f / 255.0f),
  glm::vec3(120.0f / 255.0f, 120.0 / 255.0f, 120.0f / 255.0f),
  glm::vec3(120.0f / 255.0f, 120.0 / 255.0f, 120.0f / 255.0f),
  glm::vec3(120.0f / 255.0f, 120.0 / 255.0f, 120.0f / 255.0f)
};

static Palette dark_gray = {
  glm::vec3( 40.0f / 255.0f,  40.0 / 255.0f,  40.0f / 255.0f),
  glm::vec3( 40.0f / 255.0f,  40.0 / 255.0f,  40.0f / 255.0f),
  glm::vec3( 40.0f / 255.0f,  40.0 / 255.0f,  40.0f / 255.0f),
  glm::vec3( 40.0f / 255.0f,  40.0 / 255.0f,  40.0f / 255.0f)
};

const std::string vert_shader_src(R"vert(
#version 150

uniform mat4 proj;
uniform float lighten;

in vec3 vert;
in vec4 rgba;

out vec3 color;
out float alpha;

void main() {
  gl_Position = proj * vec4(vert,1);
  color = (1 - lighten) * rgba.xyz + lighten * vec3(1, 1, 1);
  //color = vec3(1, 0, 0);
  alpha = rgba.w; 
}
)vert");

const std::string frag_shader_src(R"frag(
#version 150

in vec3 color;
in float alpha;

out vec4 finalColor;

void main() {
  finalColor = vec4(color, alpha);
}
)frag");

Scene::Scene() : palette{dark_gray}, 
  program({
    Shader::fromString(vert_shader_src, GL_VERTEX_SHADER),
    Shader::fromString(frag_shader_src, GL_FRAGMENT_SHADER)
  }),
  lines(program),
  tris(program) {}

void Scene::push_back(const Line & line) { 
  lines.data.push_back(add_color(line, color));
  lines.dirty = true;
}

void Scene::push_back(const std::vector< Line > & more_lines) { 
  lines.data.reserve(lines.data.size() + more_lines.size());
  for (auto & line : more_lines) {
    lines.data.push_back(add_color(line, color));
  }
  lines.dirty = true;
}

void Scene::push_back(const LineWithColors & line) { 
  lines.data.push_back(line);
  lines.dirty = true;
}

void Scene::push_back(const std::vector< LineWithColors > & more_lines) { 
  lines.data.reserve(lines.data.size() + more_lines.size());
  lines.data.insert(lines.data.end(), more_lines.begin(), more_lines.end());  
  lines.dirty = true;
}

void Scene::push_back(const Triangle & tri) { 
  tris.data.push_back(add_color(tri, color));
  tris.dirty = true;
}

void Scene::push_back(const std::vector< Triangle > & more_tris) { 
  tris.data.reserve(tris.data.size() + more_tris.size());
  for (auto tri : more_tris) {
    tris.data.push_back(add_color(tri, color));
  }
  tris.dirty = true;
}

void Scene::push_back(const TriangleWithColors & tri) { 
  tris.data.push_back(tri);
  tris.dirty = true;
}

void Scene::push_back(const std::vector< TriangleWithColors > & more_tris) { 
  tris.data.reserve(tris.data.size() + more_tris.size());
  tris.data.insert(tris.data.end(), more_tris.begin(), more_tris.end());  
  tris.dirty = true;
}

void Scene::clear() { 
  lines.data.clear();
  lines.dirty = true;

  tris.data.clear();
  tris.dirty = true;
}

void Scene::draw(const glm::mat4 & proj) { 

  if (lines.data.size() == 0 && tris.data.size() == 0) { return; }

  program.use();

  if (lines.data.size()) {
    glBindVertexArray(lines.VAO);
    glUniform1f(program.uniform("lighten"), 0.0f);
    glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, glm::value_ptr(proj));
    glBindBuffer(GL_ARRAY_BUFFER, lines.VBO); 

    if (lines.dirty) {
      glBufferData(GL_ARRAY_BUFFER, sizeof(LineWithColors) * lines.data.size(), &lines.data[0], GL_STATIC_DRAW);
      lines.dirty = false;
    }

    glDrawArrays(GL_LINES, 0, lines.data.size() * 2);
  }

  if (tris.data.size()) {
    glBindVertexArray(tris.VAO);
    glUniform1f(program.uniform("lighten"), 0.0f);
    glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, glm::value_ptr(proj));
    glBindBuffer(GL_ARRAY_BUFFER, tris.VBO); 

    if (tris.dirty) {
      glBufferData(GL_ARRAY_BUFFER, sizeof(TriangleWithColors) * tris.data.size(), &tris.data[0], GL_STATIC_DRAW);
      tris.dirty = false;
    }

    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1, 1);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    bool cull_back = false;
    if (cull_back) {
      glEnable(GL_CULL_FACE);
      glCullFace(GL_BACK);
    }
    glDrawArrays(GL_TRIANGLES, 0, tris.data.size() * 3);
  }

  program.unuse();

}

void Scene::draw_wireframe(const glm::mat4 & proj) { 

  if (lines.data.size() == 0 && tris.data.size() == 0) { return; }

  program.use();

  if (lines.data.size()) {
    glBindVertexArray(lines.VAO);
    glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, glm::value_ptr(proj));
    glBindBuffer(GL_ARRAY_BUFFER, lines.VBO); 

    if (lines.dirty) {
      glBufferData(GL_ARRAY_BUFFER, sizeof(LineWithColors) * lines.data.size(), &lines.data[0], GL_STATIC_DRAW);
      lines.dirty = false;
    }

    glDrawArrays(GL_LINES, 0, lines.data.size() * 2);
  }

  if (tris.data.size()) {
    glBindVertexArray(tris.VAO);
    glUniform1f(program.uniform("lighten"), 0.30f);
    glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, glm::value_ptr(proj));
    glBindBuffer(GL_ARRAY_BUFFER, tris.VBO); 

    if (tris.dirty) {
      glBufferData(GL_ARRAY_BUFFER, sizeof(TriangleWithColors) * tris.data.size(), &tris.data[0], GL_STATIC_DRAW);
      tris.dirty = false;
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glDrawArrays(GL_TRIANGLES, 0, tris.data.size() * 3);
  }

  program.unuse();

}

#if 0

  //initialization code for element array buffer
  glGenVertexArrays(1, &gVAO);
  glBindVertexArray(gVAO);

  // make and bind the VBO
  glGenBuffers(1, &gVBO);
  glBindBuffer(GL_ARRAY_BUFFER, gVBO); 
  glBufferData(GL_ARRAY_BUFFER,
               sizeof(glm::vec4) * vertices.size(),
               &vertices[0],
               GL_STATIC_DRAW);

  glGenBuffers(1, &gEBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
               sizeof(glm::ivec3) * triangles.size(), 
               &triangles[0],
               GL_STATIC_DRAW);

  // drawing from element array buffers
  gProgram->use();

    glBindVertexArray(gVAO);

    glUniform3fv(gProgram->uniform("colors"), 4, glm::value_ptr(dark_gray[0]));
  
    glUniformMatrix4fv(gProgram->uniform("proj"), 1, GL_FALSE, glm::value_ptr(gCamera.matrix()));

    glBindBuffer(GL_ARRAY_BUFFER, gVBO); 
    glEnableVertexAttribArray(gProgram->attribute("vert"));
    glVertexAttribPointer(gProgram->attribute("vert"), 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
    
    glEnableVertexAttribArray(gProgram->attribute("value"));
    glVertexAttribPointer(gProgram->attribute("value"), 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)12);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBO);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, 0);

    glUniform3fv(gProgram->uniform("colors"), 4, glm::value_ptr(light_gray[0]));
  
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, 0);

  gProgram->unuse();

#endif