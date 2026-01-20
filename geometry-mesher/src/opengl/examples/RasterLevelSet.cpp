#include <random>

#include "binary_io.hpp"
#include "geometry/geometry.hpp"

#include "timer.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

struct CartesianLevelSetFunction {
  CartesianLevelSetFunction(vec2f min, vec2f max, float cell_size) {
    min_ = min;
    max_ = max;
    int ex = roundf((max[0] - min[0]) / cell_size);
    int ey = roundf((max[1] - min[1]) / cell_size);
    cell_size_[0] = (max[0] - min[0]) / ex; 
    cell_size_[1] = (max[1] - min[1]) / ey;
    nx = ex + 1;
    ny = ey + 1;
    values.resize(nx * ny, 100.0);
  }

  float value(int i, int j) const {
    return values[i * nx + j];
  }

  float operator()(vec2f x) const {
    vec2f g = (x - min_) / cell_size_;
    int j = floor(g[0]);
    int i = floor(g[1]);
    float xi = g[0] - j;
    float eta = g[1] - i;
    return value(i  , j  ) * (1.0 - xi) * (1.0 - eta) +  
           value(i  , j+1) * (      xi) * (1.0 - eta) +  
           value(i+1, j  ) * (1.0 - xi) * (      eta) +  
           value(i+1, j+1) * (      xi) * (      eta);
  }

  vec2f vertex(int i, int j) const {
    return min_ + cell_size_ * vec2f{float(i), float(j)};
  }

  template < typename T >
  void boolean_union(T primitive) {
    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        vec2f x = min_ + cell_size_ * vec2f{float(i), float(j)};
        values[i * nx + j] = std::min(values[i * nx + j], primitive.SDF(xyz(x)));
      }
    }
  }

  template < typename T >
  void boolean_difference(T primitive) {
    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        vec2f x = min_ + cell_size_ * vec2f{float(i), float(j)};
        values[i * nx + j] = std::max(values[i * nx + j], -primitive.SDF(xyz(x)));
      }
    }
  }

  void draw(Scene & scene) const {

    scene.clear();

    auto back = [](vec2f x) {
      return vec3f{x[0], x[1], -1.0};
    };

    auto grayscale = [](float vf) {
      uint8_t vi = 255 * (0.5f + clamp(vf, -0.5f, 0.5f)); 
      return rgbcolor{vi, vi, vi, 255};
    };

    int ex = nx - 1;
    int ey = ny - 1;
    for (int i = 0; i < ey; i++) {
      for (int j = 0; j < ex; j++) {
        int v_id = i * nx + j; 
        float quad_values[4] = {values[v_id], values[v_id + 1], values[v_id + nx], values[v_id + nx + 1]};
        vec2f vertices[4] = {vertex(i,j), vertex(i,j+1), vertex(i+1,j), vertex(i+1,j+1)};

        scene.push_back(TriangleWithColors{{
          {back(vertices[0]), grayscale(quad_values[0])},
          {back(vertices[1]), grayscale(quad_values[1])},
          {back(vertices[3]), grayscale(quad_values[3])}
        }});
        scene.push_back(TriangleWithColors{{
          {back(vertices[3]), grayscale(quad_values[3])},
          {back(vertices[2]), grayscale(quad_values[2])},
          {back(vertices[0]), grayscale(quad_values[0])}
        }});
      }
    }

  }

  int nx, ny;
  vec2f min_, max_, cell_size_; 
  std::vector< float > values;
};

struct UniversalMeshing : public Application {

  UniversalMeshing();

  void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
  void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
  void mouse_motion_callback(GLFWwindow* window, double xpos, double ypos);
  void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

  void loop();
  void initialize(int d);

 private: 
  bool first;
  Camera camera;

  float camera_speed;
  bool keys_down[256];
  std::array<double,2> mouse_pos;

  bool lmb_down = false;
  bool mmb_down = false;
  bool rmb_down = false;

  // parameters for the meshing example
  float bg_cell_size;
  float fg_cell_size;
  float disk_radius;

  bool draw_bg;
  bool draw_fg;
  bool modifying_bg;
  Scene fg;
  Scene bg;
  Scene circle;
  SimplexMesh<2> mesh;

  CartesianLevelSetFunction levelset;

  void remesh(int ndvr = 10);
  void update_camera_position();
  void initialize_grid_values();

  vec2f projected_mousepos(std::array<double,2> pos);

};

rgbcolor red{255, 40, 40, 255};
rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (UniversalMeshing*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (UniversalMeshing*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (UniversalMeshing*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (UniversalMeshing*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

vec2f UniversalMeshing::projected_mousepos(std::array<double,2> pos) {

#ifdef __APPLE__
  // glfw reports the wrong mouse coordinates for "retina" displays
  pos[0] *= 2;
  pos[1] *= 2;
#endif 

  auto dir = camera.ray_cast({getWidth(), getHeight()}, pos[0], pos[1]);

  float t = - camera.m_pos[2] / dir[2];

  return vec2f{camera.m_pos[0] + t * dir[0], camera.m_pos[1] + t * dir[1]};
}

void UniversalMeshing::key_callback(GLFWwindow* window,
                          int key,
                          int scancode,
                          int action,
                          int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);

  // clang-format off
  if (key == GLFW_KEY_MINUS){ keys_down[uint8_t('-')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_EQUAL){ keys_down[uint8_t('=')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_D){ keys_down[uint8_t('d')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_W){ keys_down[uint8_t('w')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_A){ keys_down[uint8_t('a')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_S){ keys_down[uint8_t('s')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_D){ keys_down[uint8_t('d')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_SPACE){ keys_down[uint8_t(' ')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  // clang-format on
};

void UniversalMeshing::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void UniversalMeshing::mouse_motion_callback(GLFWwindow* window,
                                   double xpos,
                                   double ypos) {

  mouse_pos = {xpos, ypos};
  if (lmb_down && !mmb_down && !rmb_down) {
    if (ImGui::GetIO().WantCaptureMouse) {
      // if the mouse is interacting with ImGui
    } else {
      vec2f center = projected_mousepos(mouse_pos);
      Ball ball{xyz(center), disk_radius};
      if (keys_down['-']) {
        levelset.boolean_difference(ball);
        levelset.draw(bg);
        remesh(0);
      }

      if (keys_down['=']) {
        levelset.boolean_union(ball);
        levelset.draw(bg);
        remesh(0);
      }
    }
  }

  if (!lmb_down && !mmb_down && rmb_down) {
    // right click
  }
}

void UniversalMeshing::mouse_button_callback(GLFWwindow* window,
                                   int button,
                                   int action,
                                   int mods) {
  if (button == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS) {
    lmb_down = true;
    glfwGetCursorPos(window, &mouse_pos[0], &mouse_pos[1]);
    vec2f center = projected_mousepos(mouse_pos);
    Ball ball{xyz(center), disk_radius};
    if (keys_down['-']) {
      levelset.boolean_difference(ball);
      levelset.draw(bg);
      remesh(0);
    }

    if (keys_down['=']) {
      levelset.boolean_union(ball);
      levelset.draw(bg);
      remesh(0);
    }
  }

  if (button == GLFW_MOUSE_BUTTON_2 && action == GLFW_PRESS) {
    rmb_down = true;
    glfwGetCursorPos(window, &mouse_pos[0], &mouse_pos[1]);
  }

  if (button == GLFW_MOUSE_BUTTON_1 && action == GLFW_RELEASE) {
    lmb_down = false;
    remesh();
  }
  if (button == GLFW_MOUSE_BUTTON_2 && action == GLFW_RELEASE) {
    rmb_down = false;
  }
}

void UniversalMeshing::update_camera_position() {
  // clang-format off
  float scale = 1.0f;
  if (keys_down[uint8_t(' ')]) { scale = 0.1f; }
  if (keys_down[uint8_t('w')]) { camera.move_up(scale * camera_speed); }
  if (keys_down[uint8_t('a')]) { camera.move_left(scale * camera_speed); }
  if (keys_down[uint8_t('s')]) { camera.move_down(scale * camera_speed); }
  if (keys_down[uint8_t('d')]) { camera.move_right(scale * camera_speed); }
  // clang-format on
}

void UniversalMeshing::initialize_grid_values() {
  levelset.values = std::vector<float>(levelset.values.size(), 100.0f);
  levelset.boolean_union(Capsule{{-1.1, 1.0, 0.0}, {1.1, 1.0, 0.0}, 0.1, 0.1});
  levelset.boolean_union(Capsule{{-1.1, -1.0, 0.0}, {1.1, -1.0, 0.0}, 0.1, 0.1});
  levelset.boolean_union(Capsule{{0.3, -1.0, 0.0}, {-0.6, 1.0, 0.0}, 0.3, 0.3});
}

void UniversalMeshing::remesh(int ndvr) {

  AABB<3> bounds{{-1.1,-1.1,0}, {1.1,1.1,0}};

  AABB<2> box{{-1, -1}, {1, 1}};

  mesh = universal_mesh(std::function<float(vec2f)>([&](vec2f x){
    return std::max(box.SDF(x), levelset({x[1], x[0]}));
  }), fg_cell_size, bounds, 0.5f, 0.05f, ndvr);

  auto & v = mesh.vertices;

  std::cout << "generated mesh with:" << std::endl;
  std::cout << "  " << mesh.vertices.size() << " vertices" << std::endl;
  std::cout << "  " << mesh.elements.size() << " elements" << std::endl;
  std::cout << "  " << mesh.boundary_elements.size() << " boundary_elements" << std::endl;

  fg.clear();

  rgbcolor light_green{128, 177, 102, 255};
  fg.color = light_green;
  for (const auto& tri_ids : mesh.elements) {
    Triangle tri{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]};
    fg.push_back(tri);
  }

  rgbcolor darker_green{88, 137, 68, 255};
  fg.color = darker_green;
  for (const auto& tri_ids : mesh.elements) {
    fg.push_back(Line{v[tri_ids[0]], v[tri_ids[1]]});
    fg.push_back(Line{v[tri_ids[1]], v[tri_ids[2]]});
    fg.push_back(Line{v[tri_ids[2]], v[tri_ids[0]]});
  }

  rgbcolor black{20, 20, 20, 255};
  fg.color = black;
  for (const auto& edge_ids : mesh.boundary_elements) {
    fg.push_back(Line{v[edge_ids[0]], v[edge_ids[1]]});
  }

}

void UniversalMeshing::loop() {

  update_camera_position();

  glClearColor(0.9, 0.9, 0.9, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // feed inputs to dear imgui, start new frame
  // these go before we render our stuff
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // render our stuff
  camera.set_aspect(getWindowRatio());

  if (draw_bg) { bg.draw(camera.matrix()); }

  if (draw_fg && !modifying_bg) { fg.draw(camera.matrix()); }

  // render UI stuff
  ImGui::Begin("Meshing Parameters");

  ImGui::Checkbox("draw levelset fn", &draw_bg);
  ImGui::Checkbox("draw mesh", &draw_fg);

  bool should_remesh = false; 
  should_remesh = ImGui::DragFloat("foreground cell size", &fg_cell_size, 0.001f, 0.01f, 0.1f);

  //if (ImGui::DragFloat("background cell size", &bg_cell_size, 0.005f, 0.01f, 0.1f)) { 
  //  levelset = CartesianLevelSetFunction({-1.1, -1.1}, {1.1, 1.1}, bg_cell_size);
  //  initialize_grid_values(); 
  //  levelset.draw(bg);
  //  should_remesh = true;
  //};

  if (ImGui::Button("reset mesh")) {
    initialize_grid_values(); 
    levelset.draw(bg);
    should_remesh = true;
  };

  if (should_remesh) {
    remesh(); 
  }

  std::string output_filename;
  static char buf[128] = "output.msh";
  ImGui::InputText("", buf, 128); ImGui::SameLine();
  if (ImGui::Button("export mesh")) {
    export_mesh(mesh, buf);
  }

  ImGui::DragFloat("disk radius", &disk_radius, 0.005f, 0.01f, 0.2f);

  if (keys_down['-'] || keys_down['=']) {
    int nsegments = 128;
    circle.clear();
    circle.color = red;
    vec2f center = projected_mousepos(mouse_pos);
    float theta = 2 * M_PI / nsegments;
    float c = cosf(theta);
    float s = sinf(theta);
    mat2f R = {{{c, -s}, {s, c}}};
    vec2f v = {disk_radius, 0.0};
    for (int i = 0; i < 128; i++) {
      vec2f vnext = dot(R, v);
      circle.push_back(Line{xyz(center + v), xyz(center + vnext)});
      v = vnext;
    }
    circle.draw(camera.matrix());
  }

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

UniversalMeshing::UniversalMeshing() : Application(), keys_down{}, 
  bg_cell_size{0.01},
  fg_cell_size{0.02},
  levelset({-1.1, -1.1}, {1.1, 1.1}, bg_cell_size) {

  draw_bg = true;
  draw_fg = false;
  modifying_bg = false;

  disk_radius = 0.1;

  initialize_grid_values();
  levelset.draw(bg);

  camera_speed = 0.015;
  camera.lookAt(glm::vec3( 0.0, 0, 1), glm::vec3(0.0, 0, 0), glm::vec3(0, 1, 0));
  camera.orthographic(2.4f, getWindowRatio(), 0.1f, 10.0f);

  remesh();

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);
};

int main() {
  UniversalMeshing app;
  app.run();
  return 0;
}