
#include "BVH.hpp"
#include "geometry/geometry.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

struct Ball2D {
  vec2f c;
  float r;
  float SDF(vec2f p) {
    return norm(p - c) - r;
  }
};

struct QuadraticBezier {
  vec2f c[3];
  float r;

  // adapted from https://www.shadertoy.com/view/lsdBDS
  float SDF(vec2f p) {
    static constexpr float SQRT_3 = 1.7320508075688772935274463415059f;

    vec2f ny = normalize((c[0] + c[2]) * 0.5f - c[1]);
    vec2f nx = vec2f{ny[1], -ny[0]};
    float slope_a = (dot(c[1] - c[0], ny) / dot(c[1] - c[0], nx)) / 2.0;
    float slope_c = (dot(c[1] - c[2], ny) / dot(c[1] - c[2], nx)) / 2.0;
    float scale = (slope_c - slope_a) / dot(c[2] - c[0], nx);
    vec2f origin = c[0] - nx * (slope_a / scale) - ny * (slope_a * slope_a / scale);
    
    float px = dot(p - origin, nx) * scale;
    float py = dot(p - origin, ny) * scale;

    float min_x = std::min(slope_a, slope_c);
    float max_x = std::max(slope_a, slope_c);

    float e = ((1.5 - py) * py - 0.75) * py + 0.125;
    float f = 0.0625 * px * px + e / 27.0;
    if (f >= 0.0) { 
      float g = sqrt(f);
      float x = cbrt(0.25 * px + g) + cbrt(0.25 * px - g);
      float cx = clamp(x, min_x, max_x);
      return (norm(vec2f{cx - px, cx * cx - py}) / scale) - r;
    }
    float v = acos(sqrt(-27.0 / e) * px * 0.25) / 3.0;
    float m = cos(v);
    float n = sin(v) * SQRT_3;
    float o = sqrt((py - 0.5) / 3.0);

    float cx1 = clamp( (m + m) * o, min_x, max_x);
    float cx2 = clamp(-(m + n) * o, min_x, max_x);
    vec2f d1 = vec2f{cx1 - px, cx1 * cx1 - py};
    vec2f d2 = vec2f{cx2 - px, cx2 * cx2 - py};
    return (sqrt(std::min(dot(d1, d1), dot(d2, d2))) / scale) - r;
  }
};

struct HalfSpace {
  vec2f n; // unit normal
  float offset;
  float SDF(vec2f p) {
    return dot(p, n) - offset;
  }
};

struct Box {
  vec2f center, halfwidths;
  float SDF(vec2f p) {
    vec2f q = abs(p - center) - halfwidths;
    return norm(max(q,0.0f)) + std::min(std::max(q[0],q[1]),0.0f);
  }
};

struct ParametrizedScene : public Application {

  ParametrizedScene();

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

  std::vector < float > quality_bins;

  Ball2D ball;
  HalfSpace halfspace;
  QuadraticBezier bezier;
  Box rectangle;

  bool lmb_down = false;
  bool mmb_down = false;
  bool rmb_down = false;

  // parametrs for the meshing example
  int n;
  int ndvr;
  float snap_threshold;
  float dvr_step;

  void remesh();
  void update_camera_position();

};

int num_bins = 16;

rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (ParametrizedScene*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (ParametrizedScene*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (ParametrizedScene*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (ParametrizedScene*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

void ParametrizedScene::key_callback(GLFWwindow* window,
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

void ParametrizedScene::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void ParametrizedScene::mouse_motion_callback(GLFWwindow* window,
                                   double xpos,
                                   double ypos) {
  if (lmb_down && !mmb_down && !rmb_down) {

    if (ImGui::GetIO().WantCaptureMouse) {
      // if the mouse is interacting with ImGui
    } else {

    }

    mouse_x = xpos;
    mouse_y = ypos;
  }

  if (!lmb_down && !mmb_down && rmb_down) {
    // right click
  }
}

void ParametrizedScene::mouse_button_callback(GLFWwindow* window,
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

void ParametrizedScene::update_camera_position() {
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

void ParametrizedScene::remesh() {

  AABB<3>bounds{{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};

  std::function<float(vec2f)> f = [=](vec2f p) -> float {
    float v = ball.SDF(p);
    v = std::min(v, halfspace.SDF(p));
    v = std::min(v, bezier.SDF(p));
    v = std::min(v, rectangle.SDF(p));
    return v;
  };

  scene.clear();

  auto mesh = universal_mesh(f, 1.0 / n, bounds, snap_threshold, dvr_step, ndvr);

  vec3f red = {1.0f, 0.0f, 0.0f};
  vec3f gray = {0.25f, 0.25f, 0.25f};

  quality_bins = std::vector<float>(num_bins, 0);
  for (const auto& tri_ids : mesh.elements) {
    Triangle tri = {mesh.vertices[tri_ids[0]],
                    mesh.vertices[tri_ids[1]],
                    mesh.vertices[tri_ids[2]]};
    float q = quality(tri);
    int id = std::max(0.0f, std::min(q * num_bins, num_bins - 1.0f));
    quality_bins[id]++;

    float t = powf(std::max(q, 0.0f), 0.3f);
    vec3f rgb = red * (1 - t) + gray * t;
    scene.color = rgbcolor{uint8_t(255 * rgb[0]), uint8_t(255 * rgb[1]),
                           uint8_t(255 * rgb[2]), 255};
    scene.push_back(tri);
  }

  for (const auto& edge_ids : mesh.boundary_elements) {
    Line line = {mesh.vertices[edge_ids[0]], mesh.vertices[edge_ids[1]]};
    scene.color = rgbcolor{255, 255, 0, 255};
    scene.push_back(line);
  }
}

void ParametrizedScene::loop() {

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

  should_remesh |= ImGui::DragInt("n", &n, 0.5f, 15, 128);
  should_remesh |= ImGui::DragFloat("snap threshold", &snap_threshold, 0.001f, 0.0f, 0.5f, "%.4f");
  should_remesh |= ImGui::DragInt("dvr iterations", &ndvr, 0.1f, 0, 10);
  should_remesh |= ImGui::DragFloat("dvr step", &dvr_step, 0.001f, 0.0f, 0.05f, "%.4f");

  should_remesh |= ImGui::DragFloat("circle center x", &ball.c[0], 0.001f, 0.0f, 1.0f, "%.4f");
  should_remesh |= ImGui::DragFloat("circle center y", &ball.c[1], 0.001f, 0.0f, 1.0f, "%.4f");
  should_remesh |= ImGui::DragFloat("circle radius", &ball.r, 0.001f, 0.0f, 0.5f, "%.4f");

  should_remesh |= ImGui::DragFloat("half_space y", &halfspace.offset, 0.001f, 0.0f, 0.25f, "%.4f");

  should_remesh |= ImGui::DragFloat("rectangle center x", &rectangle.center[0], 0.001f, 0.0f, 1.0f, "%.4f");
  should_remesh |= ImGui::DragFloat("rectangle center y", &rectangle.center[1], 0.001f, 0.0f, 1.0f, "%.4f");
  should_remesh |= ImGui::DragFloat("rectangle half width", &rectangle.halfwidths[0], 0.001f, 0.0f, 0.25f, "%.4f");
  should_remesh |= ImGui::DragFloat("rectangle half height", &rectangle.halfwidths[1], 0.001f, 0.0f, 0.25f, "%.4f");

  should_remesh |= ImGui::DragFloat("bezier radius", &bezier.r, 0.001f, 0.0f, 0.25f, "%.4f");

  if (should_remesh) { remesh(); };

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

ParametrizedScene::ParametrizedScene() : Application(), scene(), keys_down{} {

  camera_speed = 0.015;

  camera.lookAt(glm::vec3(0.5, 0.5, 1), glm::vec3(0.5, 0.5, 0), glm::vec3(0, 1, 0));
  camera.orthographic(1.0f, getWindowRatio(), 0.1f, 10.0f);

  n = 32;
  snap_threshold = 0.5;

  ball = Ball2D{{0.0f, 0.0f}, 0.25f};
  halfspace = HalfSpace{{0.0, 1.0f}, 0.03f};
  bezier = {{{0.25, 0.0f}, {0.5f, 0.5f}, {1.0f, 0.5f}}, 0.05f};
  rectangle = Box{{1.0f, 0.5f}, {0.1f, 0.2f}};

  ndvr = 3;
  dvr_step = 0.05;

  scene.color = gray;

  remesh();

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};

int main() {
  ParametrizedScene app;
  app.run();
  return 0;
}