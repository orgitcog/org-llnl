#include <random>

#include "BVH.hpp"
#include "geometry/geometry.hpp"

#include "timer.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "geometry/region.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

using namespace geometry;

struct CSG : public Application {

  CSG();

  void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
  void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
  void mouse_motion_callback(GLFWwindow* window, double xpos, double ypos);
  void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);


  void loop();
  void initialize(int d);

 private: 
  Scene scene;
  Scene slice;
  Camera camera;

  float camera_speed;
  bool keys_down[256];
  double mouse_x, mouse_y;

  bool lmb_down = false;
  bool mmb_down = false;
  bool rmb_down = false;

  // parameters for the meshing example
  int n = 32;
  float w = 1.0;
  float r1 = 1.0;
  float r2 = 0.1;

  int num_iters;
  float stepsize;

  int num_dvr;
  float dvr_stepsize;

  bool color_by_error;

  void remesh();
  void update_camera_position();

};

rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (CSG*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (CSG*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (CSG*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (CSG*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

void CSG::key_callback(GLFWwindow* window,
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

void CSG::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void CSG::mouse_motion_callback(GLFWwindow* window,
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

void CSG::mouse_button_callback(GLFWwindow* window,
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

void CSG::update_camera_position() {
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

float sdf_union(float a, float b) { return std::min(a, b); }
float sdf_intersection(float a, float b) { return std::max(a, b); }
float sdf_difference(float a, float b) { return std::max(a, -b); }

void CSG::remesh() {

  float cell_size = 2.0 / n;
  AABB<3>bounds{{-1.1, -1.1, -1.1}, {1.1, 1.1, 1.1}};

  float w2 = 0.5 * w;

#if 1
  std::function<float(vec3f)> f = [=](vec3f p) { 
    float sdf_ball = Ball{{0,0,0}, r2}.SDF(p);
    float sdf_box = AABB<3>{{-w2,-w2,-w2}, {w2, w2, w2}}.SDF(p);
    float sdf_cylinders = std::min(
      Capsule{{-w2,0.0f,0.0f}, {w2, 0.0f, 0.0f}, r1, r1}.SDF(p),
      std::min(
        Capsule{{0.0f,-w2,0.0f}, {0.0f, w2, 0.0f}, r1, r1}.SDF(p),
        Capsule{{0.0f, 0.0f, -w2}, {0.0f,0.0f,w2}, r1, r1}.SDF(p)
      )
    );

    return sdf_difference(sdf_intersection(sdf_box, sdf_ball), sdf_cylinders);
  };
#else
  PrimitiveRegion cylinders({
    Capsule{{-w2,0.0f,0.0f}, {w2,0.0f,0.0f}, r1, r1},
    Capsule{{0.0f,-w2,0.0f}, {0.0f,w2,0.0f}, r1, r1},
    Capsule{{0.0f,0.0f,-w2}, {0.0f,0.0f,w2}, r1, r1}
  });

  PrimitiveRegion ball({Ball{{0,0,0}, r2}});
  
  PrimitiveRegion box({AABB<3>{{-w2,-w2,-w2}, {w2, w2, w2}}});

  //BooleanRegion combined = (box * ball) - cylinders;
  BooleanRegion combined = (box * ball);

  float cell_size = 2.0 / n;
  std::function<float(vec3f)> f = [&](vec3f p) { 
    return combined.SDF(p, cell_size);
  };
#endif

  scene.clear();

  auto mesh = universal_mesh(f, cell_size, bounds);

  improve_boundary(f, mesh, stepsize, num_iters, 1);

  dvr(mesh, 9.0f, dvr_stepsize * cell_size * cell_size, num_dvr, 12);

  auto & v = mesh.vertices;

  for (const auto& tri_ids : mesh.boundary_elements) {
    vec3f center = (v[tri_ids[0]] + v[tri_ids[1]] + v[tri_ids[2]]) / 3.0;

    if (color_by_error) {
      float sdf = f(center);

      uint8_t s = 255 * clamp(1.0f - (2.0f * abs(sdf) / cell_size), 0.0f, 1.0f);
      if (sdf < 0) {
        scene.color = rgbcolor{255, s, s, 255};
      } else {
        scene.color = rgbcolor{s, s, 255, 255};
      }
      scene.push_back(Triangle{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]});

      scene.color = gray;
      scene.push_back(Line{v[tri_ids[0]], v[tri_ids[1]]});
      scene.push_back(Line{v[tri_ids[1]], v[tri_ids[2]]});
      scene.push_back(Line{v[tri_ids[2]], v[tri_ids[0]]});
    } else {
      scene.push_back(Triangle{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]});
    }

  }

}

void CSG::loop() {

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
  if (!color_by_error) {
    scene.draw_wireframe(camera.matrix());
  }

  // render UI stuff
  ImGui::Begin("Meshing Parameters");

  bool should_remesh = false; 
  should_remesh |= ImGui::DragInt("n", &n, 0.5f, 8, 128);
  should_remesh |= ImGui::DragFloat("rod radius", &r1, 0.001f, 0.1f, 0.5f, "%.5f");
  should_remesh |= ImGui::DragFloat("sphere radius", &r2, 0.001f, 0.5f, 2.0f, "%.5f");
  should_remesh |= ImGui::DragFloat("box width", &w, 0.001f, 0.5f, 2.0f, "%.5f");

  should_remesh |= ImGui::DragInt("num_bdr_iterations", &num_iters, 0.5f, 0, 64);
  should_remesh |= ImGui::DragFloat("step size", &stepsize, 5.0f, 0.0f, 10000.0f, "%.5f");

  should_remesh |= ImGui::DragInt("num_dvr_iterations", &num_dvr, 0.5f, 0, 32);
  should_remesh |= ImGui::DragFloat("dvr step size", &dvr_stepsize, 0.001f, 0.0f, 0.5f, "%.5f");

  should_remesh |= ImGui::Checkbox("color by error", &color_by_error);

  if (should_remesh) { remesh(); };

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

CSG::CSG() : Application(), scene(), keys_down{} {

  camera_speed = 0.015;
  camera.lookAt(glm::vec3(1.5, 1.5, 1.5), glm::vec3(0.0f, 0.0f, 0.0f));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 0.01f, 100.0f);

  n = 32;
  r1 = 0.25f;
  r2 = 0.75f;
  w = 1.0f;

  num_iters = 0;
  stepsize = 0.0f;

  num_dvr = 0;
  dvr_stepsize = 0.0f;

  color_by_error = false;

  scene.color = gray;

  remesh();

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};


int main() {
  CSG app;
  app.run();
  return 0;
}