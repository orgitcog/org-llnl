#include <random>

#include "BVH.hpp"
#include "geometry/geometry.hpp"
#include "geometry/parse_dat.hpp"

#include "timer.hpp"
#include "binary_io.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

struct MovingHalfSpace {
  vec3f n;
  float c;
  float speed;
  float SDF(vec3f p) const { return dot(p, n) - c; }
};

struct ConvexPolyhedron {
  std::vector< MovingHalfSpace > half_spaces;
  float time_to_reach(vec3f p) const {
    float t = -1.0e+50; 
    for (const auto & h : half_spaces) {
      t = std::max(h.SDF(p) / h.speed, t);
    }
    return t;
  }
};

float v_100 = 1.0;
float v_110 = 1.0;
float v_111 = 1.0;

struct FaceInfo {
  vec3f n;
  float scale;
  int type;
};

ConvexPolyhedron Rhombicuboctahedron(vec3f center, float r0, vec3f theta) {

  constexpr std::array< FaceInfo, 26 > rhombicuboctahedron_faces {{
    {{0., 0., -1.}, 0.828427, 0}, 
    {{0., 0., 1.}, 0.828427, 0}, 
    {{1., 0., 0.}, 0.828427, 0},
    {{-1., 0., 0.}, 0.828427, 0},
    {{0., -1., 0.}, 0.828427, 0},
    {{0., 1., 0.}, 0.828427, 0},
    {{0.707107, -0.707107, 0.}, 0.828427, 1}, 
    {{0.707107, 0.707107, 0.}, 0.828427, 1}, 
    {{-0.707107, -0.707107, 0.}, 0.828427, 1}, 
    {{-0.707107, 0.707107, 0.}, 0.828427, 1}, 
    {{0., -0.707107, 0.707107}, 0.828427, 1}, 
    {{0., 0.707107, 0.707107}, 0.828427, 1}, 
    {{0.707107, 0., 0.707107}, 0.828427, 1}, 
    {{-0.707107, 0., 0.707107}, 0.828427, 1}, 
    {{0., -0.707107, -0.707107}, 0.828427, 1}, 
    {{0., 0.707107, -0.707107}, 0.828427, 1}, 
    {{0.707107, 0., -0.707107}, 0.828427, 1}, 
    {{-0.707107, 0., -0.707107}, 0.828427, 1}, 
    {{0.57735, -0.57735, 0.57735}, 0.784761, 2}, 
    {{0.57735, 0.57735, 0.57735}, 0.784761, 2}, 
    {{-0.57735, -0.57735, 0.57735}, 0.784761, 2}, 
    {{-0.57735, 0.57735, 0.57735}, 0.784761, 2}, 
    {{0.57735, -0.57735, -0.57735}, 0.784761, 2}, 
    {{0.57735, 0.57735, -0.57735}, 0.784761, 2}, 
    {{-0.57735, -0.57735, -0.57735}, 0.784761, 2}, 
    {{-0.57735, 0.57735, -0.57735}, 0.784761, 2}
  }};


  const mat3f R = RotationMatrix(theta);

  float speeds[3] = {v_100, v_110, v_111};

  ConvexPolyhedron output;
  for (const auto & face : rhombicuboctahedron_faces) {
    vec3f n = dot(R, face.n);
    output.half_spaces.push_back({n, dot(n, center) + r0 / face.scale, speeds[face.type]});
  }
  return output;
}

struct CrystalGrowth : public Application {

  CrystalGrowth();

  void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
  void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
  void mouse_motion_callback(GLFWwindow* window, double xpos, double ypos);
  void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

  void loop();
  void initialize(int d);

 private: 
  Scene scene;
  Camera camera;

  float camera_speed;
  bool keys_down[256];
  double mouse_x, mouse_y;

  bool lmb_down = false;
  bool mmb_down = false;
  bool rmb_down = false;

  float time;
  float cell_size;

  void remesh();
  void update_camera_position();

};

rgbcolor red{255, 40, 40, 255};
rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (CrystalGrowth*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (CrystalGrowth*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (CrystalGrowth*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (CrystalGrowth*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

rgbcolor color_by_normal(vec3f n) {
  return rgbcolor{
    uint8_t(127 * (1 + n[0])), 
    uint8_t(127 * (1 + n[1])), 
    uint8_t(127 * (1 + n[2])), 
    255
  };
}

void CrystalGrowth::key_callback(GLFWwindow* window,
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

void CrystalGrowth::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void CrystalGrowth::mouse_motion_callback(GLFWwindow* window,
                                   double xpos,
                                   double ypos) {
  if (lmb_down && !mmb_down && !rmb_down) {
    float altitude = (ypos - mouse_y) * 0.01f;
    float azimuth = (xpos - mouse_x) * 0.01f;

    if (ImGui::GetIO().WantCaptureMouse) {
      // if the mouse is interacting with ImGui
    } else {
      camera.rotate(altitude, -azimuth);
    }

    mouse_x = xpos;
    mouse_y = ypos;
  }

  if (!lmb_down && !mmb_down && rmb_down) {
    // right click
  }
}

void CrystalGrowth::mouse_button_callback(GLFWwindow* window,
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

void CrystalGrowth::update_camera_position() {
  // clang-format off
  float scale = 1.0f;
  if (keys_down[uint8_t(' ')]) { scale = 0.1f; }
  if (keys_down[uint8_t('w')]) { camera.move_forward(scale * camera_speed); }
  if (keys_down[uint8_t('s')]) { camera.move_forward(-scale * camera_speed); }
  if (keys_down[uint8_t('a')]) { camera.move_left(scale * camera_speed); }
  if (keys_down[uint8_t('d')]) { camera.move_right(scale * camera_speed); }
  if (keys_down[uint8_t('q')]) { camera.move_down(scale * camera_speed); }
  if (keys_down[uint8_t('e')]) { camera.move_up(scale * camera_speed); }
  // clang-format on
}

void CrystalGrowth::remesh() {

  std::vector< ConvexPolyhedron > seed_crystals;
  seed_crystals.push_back(Rhombicuboctahedron({ 0.2,  0.4, -0.8}, 0.2, {0.0, 0.0, 0.0}));
  seed_crystals.push_back(Rhombicuboctahedron({-0.3,  0.6, -0.8}, 0.2, {0.1, 1.0, 0.7}));
  seed_crystals.push_back(Rhombicuboctahedron({ 0.5,  0.0, -0.8}, 0.2, {0.0, 0.2, 0.0}));
  seed_crystals.push_back(Rhombicuboctahedron({ 0.0,  0.0, -0.8}, 0.2, {0.4, 0.5, 0.3}));
  seed_crystals.push_back(Rhombicuboctahedron({-0.6,  0.0, -0.8}, 0.2, {0.4, 0.5, 0.3}));
  seed_crystals.push_back(Rhombicuboctahedron({-0.3, -0.7, -0.8}, 0.2, {0.0, 0.6, 0.1}));
  seed_crystals.push_back(Rhombicuboctahedron({ 0.4, -0.3, -0.8}, 0.2, {0.1, 0.6, 2.0}));

  scene.clear();

  AABB<3> bounds = {{-1.1, -1.1, -1.1}, {1.1, 1.1, 1.1}};

  AABB<3> clip = {{-1, -1, -1}, {1, 1, 1}};

  std::function<float(vec3f)> f = [&](vec3f x) -> float {
    float t_min = 1.0e+10;
    for (const auto & crystal : seed_crystals) {
      t_min = std::min(crystal.time_to_reach(x), t_min);
    }
    return std::max(t_min - time, clip.SDF(x));
  };

  auto mesh = universal_mesh(f, cell_size, bounds, 0.5, 0.05, 3);

  std::cout << mesh.vertices.size() << " " << mesh.elements.size() << " " << mesh.boundary_elements.size() << std::endl;

  auto & v = mesh.vertices;

  scene.color = rgbcolor{150, 150, 150, 255};
  scene.push_back(Triangle{vec3f{-1.5, -1.5, -1.01}, vec3f{+1.5, -1.5, -1.01}, vec3f{+1.5, +1.5, -1.01}});
  scene.push_back(Triangle{vec3f{-1.5, -1.5, -1.01}, vec3f{+1.5, +1.5, -1.01}, vec3f{-1.5, +1.5, -1.01}});

  scene.color = gray;
  for (const auto& tri_ids : mesh.boundary_elements) {
    Triangle tri{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]};
    vec3f n = normalize(cross(tri.vertices[1] - tri.vertices[0], tri.vertices[2] - tri.vertices[0]));
    scene.color = color_by_normal(n);
    scene.push_back(tri);
  }

}

void CrystalGrowth::loop() {

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

  // render UI stuff
  ImGui::Begin("Meshing Parameters");

  bool should_remesh = false; 
  should_remesh |= ImGui::DragFloat("v_100", &v_100, 0.005f, 0.05f, 1.5f, "%.5f");
  should_remesh |= ImGui::DragFloat("v_110", &v_110, 0.005f, 0.05f, 1.5f, "%.5f");
  should_remesh |= ImGui::DragFloat("v_111", &v_111, 0.005f, 0.05f, 1.5f, "%.5f");
  should_remesh |= ImGui::DragFloat("cell size", &cell_size, 0.0005f, 0.02f, 0.1f, "%.5f");
  should_remesh |= ImGui::DragFloat("time", &time, 0.005f, 0.0f, 3.0f, "%.5f");
  if (should_remesh) { remesh(); };

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

CrystalGrowth::CrystalGrowth() : Application(), scene(), keys_down{} {

  time = 0.0;
  cell_size = 0.05;

  remesh();

  scene.color = gray;

  camera_speed = 0.015;
  camera.lookAt(glm::vec3(1.5, 1.5, 1.5), glm::vec3(0.0f, 0.0f, 0.0f));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 0.01f, 100.0f);

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};

int main() {
  CrystalGrowth app;
  app.run();
  return 0;
}