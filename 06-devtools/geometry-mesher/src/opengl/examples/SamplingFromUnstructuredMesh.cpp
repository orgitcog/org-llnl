#include "BVH.hpp"
#include "geometry/geometry.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <iomanip>

//////////////////////////////////////////////////////////
//
// a minimal implementation of the finite element parts we care about

// quad shape functions
vec4f N(vec2f xi) { 
  return { 
    (1.0f - xi[0]) * (1.0f - xi[1]),
            xi[0]  * (1.0f - xi[1]),
    (1.0f - xi[0]) *         xi[1],
            xi[0]  *         xi[1]
  };
}

// quad shape function gradients
mat4x2f dN(vec2f xi) { 
  return {{
    {-1 + xi[1], -1 + xi[0]}, 
    {1 - xi[1], -xi[0]}, 
    {-xi[1], 1 - xi[0]}, 
    {xi[1], xi[0]}
  }};
}

// for a quad element, output is restricted to [0, 1]x[0, 1]
vec2f find_isoparametric_coords(mat2x4f vT, vec2f x) {

  // a few newton iterations to solve $ x_p(\xi) - x == 0 $
  // starting with guess at element center
  vec2f xi = {0.5f, 0.5f};
  for (int k = 0; k < 6; k++) {
    vec2f r = x - dot(vT, N(xi));
    mat2f J = dot(vT, dN(xi));
    xi = clamp(xi + dot(inv(J), r), 0.0f, 1.0f);
  }

  return xi;

}

// edge shape functions
vec2f N(float xi) { return {1.0f - xi, xi}; }

// closest point projection for an edge element, output is restricted to [0, 1]
double find_isoparametric_coords(mat2f vT, vec2f x) {
  auto v = transpose(vT);
  return dot(x - v[0], v[1] - v[0]) / dot(v[1] - v[0], v[1] - v[0]);
}

struct QuadMesh {
  std::vector< vec2f > vertices;
  std::vector< std::array< uint32_t, 4 > > quads;
  std::vector< std::array< uint32_t, 2 > > boundary_edges;
};

QuadMesh CartesianMesh(int n) {
  uint32_t num_vertices = (n + 1) * (n + 1);
  uint32_t num_elements = n * n;
  uint32_t num_boundary_edges = 4 * n;

  QuadMesh mesh;
  mesh.vertices.resize(num_vertices);
  mesh.quads.resize(num_elements);
  mesh.boundary_edges.resize(num_boundary_edges);

  auto vertex_id = [&](uint32_t i, uint32_t j) { return j * (n + 1) + i; };

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      uint32_t elem_id = j * n + i;
      mesh.quads[elem_id] = {vertex_id(i,j), vertex_id(i+1,j), vertex_id(i,j+1), vertex_id(i+1,j+1)};
    }
  }

  for (int i = 0; i < n; i++) {
    mesh.boundary_edges[0*n + i] = {vertex_id(i,0), vertex_id(i+1,0)};
    mesh.boundary_edges[1*n + i] = {vertex_id(n,i), vertex_id(n,i+1)};
    mesh.boundary_edges[2*n + i] = {vertex_id(n-i,n), vertex_id(n-i-1,n)};
    mesh.boundary_edges[3*n + i] = {vertex_id(0,n-i), vertex_id(0,n-i-1)};
  }

  // scale and offset to map the domain to [-1, 1] x [-1, 1]
  float scale = 2.0 / n;
  vec2f offset = {0.5f * n, 0.5f * n};

  for (int i = 0; i < n + 1; i++) {
    for (int j = 0; j < n + 1; j++) {
      mesh.vertices[vertex_id(i,j)] = (vec2f{float(i), float(j)} - offset) * scale;
    }
  }

  return mesh;
}

struct rosenbrock {
  float a, b;
  float operator()(vec2f x) const {
    return (a - x[0]) * (a - x[0]) + b * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
  }
};

template < std::size_t n >
AABB<3>bounding_box(std::array< vec2f, n > points) {
  AABB<3>box{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++) {
      box.min[j] = (i == 0) ? points[i][j] : std::min(box.min[j], points[i][j]);
      box.max[j] = (i == 0) ? points[i][j] : std::max(box.max[j], points[i][j]);
    }
  }
  return box;
}

struct UniversalMesher : public Application {

  UniversalMesher();

  void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
  void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
  void mouse_motion_callback(GLFWwindow* window, double xpos, double ypos);
  void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

  void loop();

 private: 
  Scene scene;
  Scene background_mesh;
  Camera camera;

  float camera_speed;
  bool keys_down[256];
  double mouse_x, mouse_y;

  bool lmb_down = false;
  bool rmb_down = false;

  int n_um;
  int n_mesh;

  float mesh_scale;
  float mesh_rotation;

  float threshold;
  float a;
  float b;

  QuadMesh mesh;
  BVH<3> bvh;
  BVH<3> bdr_bvh;

  std::string example;

  void remesh();
  void update_camera_position();

};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (UniversalMesher*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (UniversalMesher*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (UniversalMesher*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (UniversalMesher*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

void UniversalMesher::key_callback(GLFWwindow* window,
                          int key,
                          int scancode,
                          int action,
                          int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);

  // clang-format off
  if (key == GLFW_KEY_W){ keys_down[uint8_t('w')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_A){ keys_down[uint8_t('a')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_S){ keys_down[uint8_t('s')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_D){ keys_down[uint8_t('d')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_Q){ keys_down[uint8_t('q')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_E){ keys_down[uint8_t('e')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_SPACE){ keys_down[uint8_t(' ')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  // clang-format on
};

void UniversalMesher::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void UniversalMesher::mouse_motion_callback(GLFWwindow* window,
                                   double xpos,
                                   double ypos) {
}

void UniversalMesher::mouse_button_callback(GLFWwindow* window,
                                   int button,
                                   int action,
                                   int mods) {
  if (button == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS) {
    lmb_down = true;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);
  }

  if (button == GLFW_MOUSE_BUTTON_2 && action == GLFW_PRESS) {
    rmb_down = true;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);
  }

  if (button == GLFW_MOUSE_BUTTON_1 && action == GLFW_RELEASE) {
    lmb_down = false;
  }
  if (button == GLFW_MOUSE_BUTTON_2 && action == GLFW_RELEASE) {
    rmb_down = false;
  }
}

void UniversalMesher::update_camera_position() {
  // clang-format off
  float scale = 1.0f;
  if (keys_down[uint8_t(' ')]) { scale = 0.1f; }
  if (keys_down[uint8_t('w')]) { camera.move_up(scale * camera_speed); }
  if (keys_down[uint8_t('s')]) { camera.move_up(-scale * camera_speed); }
  if (keys_down[uint8_t('a')]) { camera.move_left(scale * camera_speed); }
  if (keys_down[uint8_t('d')]) { camera.move_right(scale * camera_speed); }
  if (keys_down[uint8_t('q')]) { camera.move_down(scale * camera_speed); }
  if (keys_down[uint8_t('e')]) { camera.move_up(scale * camera_speed); }
  // clang-format on
}

void draw_quad(Scene & scene, vec2f v0, vec2f v1, vec2f v2, vec2f v3, float z = 0.0f) {
  scene.push_back(Line{vec3f{v0[0], v0[1], z}, vec3f{v1[0], v1[1], z}});
  scene.push_back(Line{vec3f{v1[0], v1[1], z}, vec3f{v3[0], v3[1], z}});
  scene.push_back(Line{vec3f{v3[0], v3[1], z}, vec3f{v2[0], v2[1], z}});
  scene.push_back(Line{vec3f{v2[0], v2[1], z}, vec3f{v0[0], v0[1], z}});
}

void draw_AABB2D(Scene & scene, AABB<3>box) {
  vec2f v[4] = {
    {box.min[0], box.min[1]},
    {box.max[0], box.min[1]},
    {box.min[0], box.max[1]},
    {box.max[0], box.max[1]},
  };
  draw_quad(scene, v[0], v[1], v[2], v[3]);
}

void UniversalMesher::remesh() {

  rosenbrock fn{a, b};

  mesh = CartesianMesh(n_mesh);
  float h = 2.0 / n_mesh;

  // sample the function over the mesh in its original configuration
  std::vector < float > mesh_values(mesh.vertices.size());
  for (int i = 0; i < mesh.vertices.size(); i++) {
    mesh_values[i] = fn(mesh.vertices[i]);
  }

  // apply a transformation to the mesh
  float c = cos(mesh_rotation);
  float s = sin(mesh_rotation);
  mat2f R {{{c, s}, {-s, c}}};
  for (auto & v : mesh.vertices) {
    v = dot(R, v) * mesh_scale;
  }

  background_mesh.clear();
  background_mesh.color = rgbcolor{0, 200, 0, 255};

  std::vector< AABB<3>> bounding_boxes(mesh.quads.size());
  for (int i = 0; i < mesh.quads.size(); i++) {
    auto quad = mesh.quads[i];
    auto box = bounding_box<4>({mesh.vertices[quad[0]],
                                mesh.vertices[quad[1]],
                                mesh.vertices[quad[2]],
                                mesh.vertices[quad[3]]});
    bounding_boxes[i] = box;

    // this is just for drawing the background mesh
    draw_quad(background_mesh, mesh.vertices[quad[0]],
                               mesh.vertices[quad[1]],
                               mesh.vertices[quad[2]],
                               mesh.vertices[quad[3]]);
  }
  bvh = BVH(bounding_boxes);

  std::vector< AABB<3> > edge_bounding_boxes(mesh.boundary_edges.size());
  for (int i = 0; i < mesh.boundary_edges.size(); i++) {
    auto edge = mesh.boundary_edges[i];
    edge_bounding_boxes[i] = bounding_box<2>({mesh.vertices[edge[0]], 
                                              mesh.vertices[edge[1]]});
  }
  bdr_bvh = BVH(edge_bounding_boxes);

  AABB<3>bounds{{-1.0, -1.0, 0.0}, {1.0, 1.0, 0.0}};

  float cell_size = 2.0 / n_um;
  float dx = 1.5 * cell_size;

  std::function<float(vec2f)> f = [&](vec2f x) -> float {

    AABB<3>point{
      {x[0], x[1], 0.0f}, 
      {x[0], x[1], 0.0f}
    };

    // in case nothing is hit, large number => outside
    float value = 10.0f;

    bool inside = false;

    // the callback function is invoked for each time `box`
    // overlaps with something in the BVH, where `i` is the
    // index of the object hit (in this case, element index)
    bvh.query(point, [&](int i) {

      // we need to check if this element contains the point `x`
      std::array<uint32_t, 4> v_ids = mesh.quads[i];

      // load the vertices for this quad
      auto vT = transpose(mat4x2f{{
        mesh.vertices[v_ids[0]],
        mesh.vertices[v_ids[1]],
        mesh.vertices[v_ids[2]],
        mesh.vertices[v_ids[3]]
      }});

      vec2f xi = find_isoparametric_coords(vT, x);

      // check if the point really does belong to element `i`
      if (norm(x - dot(vT, N(xi))) < 1.0e-5 * h) {
                        
        // then load the nodal values for this element
        vec4f local_values = {
          mesh_values[v_ids[0]],
          mesh_values[v_ids[1]],
          mesh_values[v_ids[2]],
          mesh_values[v_ids[3]]
        };

        // and interpolate the level set value
        value = dot(local_values, N(xi));

        inside = true;

      } 

    });

    AABB<3>box{
      {x[0] - dx, x[1] - dx, 0.0f}, 
      {x[0] + dx, x[1] + dx, 0.0f}
    };

    float nearest = 2.0f * cell_size;
    bdr_bvh.query(box, [&](int i) {

      // we want to get the distance to each edge nearby
      std::array<uint32_t, 2> v_ids = mesh.boundary_edges[i];

      // load the vertices for this edge
      auto vT = transpose(mat2f{{mesh.vertices[v_ids[0]], mesh.vertices[v_ids[1]]}});

      // find isoparametric coords of the closest point
      float xi = find_isoparametric_coords(vT, x);

      // keep track of how close we got to any edge
      nearest = std::min(nearest, norm(x - dot(vT, N(xi))));

    });

    float sdf = (inside) ? -nearest : nearest;
    float field = (inside) ? value - threshold : 1.5 * cell_size; 

    return std::max(-field, sdf);

  };

  scene.clear();

  auto mesh = universal_mesh(f, cell_size, bounds);

  vec3f red = {1.0f, 0.0f, 0.0f};
  vec3f gray = {0.25f, 0.25f, 0.25f};

  for (const auto& tri_ids : mesh.elements) {
    Triangle tri = {mesh.vertices[tri_ids[0]],
                    mesh.vertices[tri_ids[1]],
                    mesh.vertices[tri_ids[2]]};
    float q = quality(tri);

    float t = powf(std::max(q, 0.0f), 0.3f);
    vec3f rgb = red * (1 - t) + gray * t;
    scene.color = rgbcolor{uint8_t(255 * rgb[0]), uint8_t(255 * rgb[1]),
                           uint8_t(255 * rgb[2]), 255};
    scene.push_back(tri);
  }

  scene.color = rgbcolor{255, 0, 0, 255};
  scene.push_back(Line{vec3f{-1.0f, -1.0f, 0.0f}, vec3f{+1.0f, -1.0f, 0.0f}});
  scene.push_back(Line{vec3f{+1.0f, -1.0f, 0.0f}, vec3f{+1.0f, +1.0f, 0.0f}});
  scene.push_back(Line{vec3f{-1.0f, +1.0f, 0.0f}, vec3f{+1.0f, +1.0f, 0.0f}});
  scene.push_back(Line{vec3f{-1.0f, +1.0f, 0.0f}, vec3f{-1.0f, -1.0f, 0.0f}});

}

void UniversalMesher::loop() {

  update_camera_position();

  glClearColor(0.169f, 0.314f, 0.475f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // feed inputs to dear imgui, start new frame
  // these go before we render our stuff
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // render our stuff
  camera.set_aspect(getWindowRatio());
  scene.draw(camera.matrix());
  scene.draw_wireframe(camera.matrix());

  background_mesh.draw(camera.matrix());

  // render UI stuff
  ImGui::Begin("Meshing Parameters");

  bool should_remesh = false; 

  should_remesh |= ImGui::DragInt("original mesh divisions", &n_mesh, 0.5f, 8, 256);
  should_remesh |= ImGui::DragFloat("mesh rotation", &mesh_rotation, 0.01f, -1.0f, 1.0f);
  should_remesh |= ImGui::DragFloat("mesh scale", &mesh_scale, 0.01f, 0.2f, 1.0f);

  should_remesh |= ImGui::DragInt("universal_mesh() divisions", &n_um, 0.5f, 8, 256);
  should_remesh |= ImGui::DragFloat("rosenbrock a", &a, 0.05f, -2.0f, 2.0f);
  should_remesh |= ImGui::DragFloat("rosenbrock b", &b, 0.05f, 0.5f, 5.0f);
  should_remesh |= ImGui::DragFloat("threshold", &threshold, 0.01f, 0.1f, 1.0f);

  if (should_remesh) { remesh(); };

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}
UniversalMesher::UniversalMesher() : Application(), scene(), keys_down{} {

  camera_speed = 0.015;

  camera.lookAt(glm::vec3(0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  camera.orthographic(2.0f, getWindowRatio(), 0.1f, 10.0f);

  mesh_scale = 1.0;
  mesh_rotation = 0.0;
  n_mesh = 10;
  n_um = 16;
  a = 0.0f;
  b = 1.0f;
  threshold = 0.5f;

  remesh();

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};

int main() {
  UniversalMesher app;
  app.run();
  return 0;
}