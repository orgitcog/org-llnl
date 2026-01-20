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

enum TPMS {
  GYROID,
  SCHWARZ_P,
  SCHWARZ_D,
  NEOVIUS,
  SCHOEN_IWP,
  FISCHER_KOCH_S,
  FISCHER_KOCH_Y,
  FISCHER_KOCH_CP
};

float (* functions[8])(vec3f x) = {
  tpms::gyroid,
  tpms::schwarz_p,
  tpms::schwarz_d,
  tpms::neovius,
  tpms::schoen_iwp,
  tpms::fischer_koch_s,
  tpms::fischer_koch_y,
  tpms::fischer_koch_cp
};

struct PeriodicStructures : public Application {

  PeriodicStructures();

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

  int which;
  int samples;
  float thickness;

  void remesh();
  void update_camera_position();

};

rgbcolor red{255, 40, 40, 255};
rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (PeriodicStructures*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (PeriodicStructures*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (PeriodicStructures*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (PeriodicStructures*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on



void PeriodicStructures::key_callback(GLFWwindow* window,
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

void PeriodicStructures::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void PeriodicStructures::mouse_motion_callback(GLFWwindow* window,
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

void PeriodicStructures::mouse_button_callback(GLFWwindow* window,
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

void PeriodicStructures::update_camera_position() {
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

void PeriodicStructures::remesh() {

  scene.clear();

  float cell_size = 2.0 * M_PI / samples;

  AABB<3> bounds = {{-1.1*M_PI, -1.1*M_PI, -1.1*M_PI}, {1.1*M_PI, 1.1*M_PI, 1.1*M_PI}};

  std::function<float(vec3f)> f;
  if (thickness > 0) {
    AABB<3> clip = {{-M_PI, -M_PI, -M_PI}, {M_PI, M_PI, M_PI}};
    f = [=](vec3f x){ 
      return std::max(clip.SDF(x), fabsf(functions[which](x)) - thickness); 
    };
  } else {
    f = [=](vec3f x){ return functions[which](x); };
  }

  SimplexMesh<3> mesh = (thickness > 0) ?
    universal_mesh(f, cell_size, bounds, 0.5, 0.05, 3, -1) :
    universal_boundary_mesh(f, cell_size, bounds, 0.5, 0.05, 3, -1);

  std::cout << mesh.vertices.size() << " " << mesh.elements.size() << " " << mesh.boundary_elements.size() << std::endl;

  scene.color = gray;

  auto & v = mesh.vertices;
  for (const auto& tri_ids : mesh.boundary_elements) {
    Triangle tri{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]};
    scene.push_back(tri);
  }

}

void PeriodicStructures::loop() {

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

  const char* listbox_items[] = { 
    "GYROID",
    "SCHWARZ_P",
    "SCHWARZ_D",
    "NEOVIUS",
    "SCHOEN_IWP",
    "FISCHER_KOCH_S",
    "FISCHER_KOCH_Y",
    "FISCHER_KOCH_CP"
  };

  bool should_remesh = false; 
  should_remesh |= ImGui::DragInt("level of detail", &samples, 0.5f, 3, 128, "%.5f");
  should_remesh |= ImGui::DragFloat("thickness", &thickness, 0.005f, 0.0f, 1.3f, "%.5f");
  should_remesh |= ImGui::ListBox("Function", &which, listbox_items, IM_ARRAYSIZE(listbox_items), 8);
  if (should_remesh) { remesh(); };

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

PeriodicStructures::PeriodicStructures() : Application(), scene(), keys_down{} {

  samples = 2;
  thickness = 0.0;
  which = TPMS::GYROID;

  remesh();

  scene.color = gray;

  vec3f center = {0, 0, 0};
  vec3f pov = {5, 5, 5};
  camera.lookAt(glm::vec3(pov[0], pov[1], pov[2]), glm::vec3(center[0], center[1], center[2]));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 0.05f, 100.0f);

  camera_speed = 0.05;

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};

int main() {
  PeriodicStructures app;
  app.run();
  return 0;
}