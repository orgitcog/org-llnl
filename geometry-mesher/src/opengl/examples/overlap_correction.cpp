#include <random>

#include "BVH.hpp"
#include "geometry/geometry.hpp"

#include "timer.hpp"
#include "binary_io.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

struct UniversalMeshing : public Application {

  UniversalMeshing();

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
  bool overlap_correction;
  int n;
  int highlight;
  int ndvr;
  float r;
  float blend_radius;
  float x, y;
  std::vector< float > qualities;
  std::vector< Capsule > capsules;

  void remesh();
  void update_camera_position();

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



void UniversalMeshing::key_callback(GLFWwindow* window,
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

void UniversalMeshing::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void UniversalMeshing::mouse_motion_callback(GLFWwindow* window,
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

void UniversalMeshing::mouse_button_callback(GLFWwindow* window,
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

void UniversalMeshing::update_camera_position() {
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

void UniversalMeshing::remesh() {

  float cell_size = 2.0 / n;
  AABB<3> bounds{{-1.3, -1.3, -1.3}, {1.3, 1.3, 1.3}};

  capsules = {
    {{-1, 0, 0}, { 1, 0, 0}, 0.1, 0.1},

    {{-1, 1, 0}, {-1, -1, 0}, 0.1, 0.1},

    {{0, 1, 0}, {0, 0, 0}, 0.1, 0.1},
    {{0, 0, 0}, {0, -1, 0}, 0.1, 0.1},

    {{1, 1, 0}, {1, 0, 0}, 0.1, 0.1},
    {{1, 0, 0}, {1, -1, 0}, 0.1, 0.1}
  };

  std::vector< Ball > spheres = {
    {{-1, 0, 0}, 0.1},

    {{0, 0, 0}, 0.1},
    {{0, 0, 0}, 0.1},

    {{1, 0, 0}, 0.1},
    {{1, 0, 0}, 0.1}
  };

  constexpr int dim = 3;
  std::vector< AABB<dim> > bounding_boxes(capsules.size(), AABB<dim>{});
  for (uint32_t i = 0; i < capsules.size(); i++) {
    AABB<3> box = bounding_box(capsules[i]);
    for (int j = 0; j < dim; j++) {
      bounding_boxes[i].min[j] = box.min[j];
      bounding_boxes[i].max[j] = box.max[j];
    }
  }

  BVH<dim> bvh(bounding_boxes);

  float dx = 1.5 * cell_size + 5.0 * blend_radius;

  std::function<float(vec3f)> f = [=](vec3f x) { 
  
    AABB<dim> box{};
    for (int j = 0; j < dim; j++) {
      box.min[j] = x[j] - dx;
      box.max[j] = x[j] + dx;
    };

    if (blend_radius == 0) {
      float value = 2 * dx;
      bvh.query(box, [&](int i) {
        value = std::min(value, capsules[i].SDF(x));
      });
      return value;
    } else {
      double value = 0.0;
      bvh.query(box, [&](int i) {
        value += exp(-capsules[i].SDF(x) / blend_radius);
      });

      if (overlap_correction) {
        for (auto sphere : spheres) {
          value -= exp(-sphere.SDF(x) / blend_radius);
        }
      }
      return -blend_radius * logf(value);
    }
  };

  int num_threads = 1;
  auto mesh = universal_mesh(f, cell_size, bounds, 0.5, 0.05f, ndvr, num_threads);

  auto & v = mesh.vertices;

  scene.clear();
  scene.color = gray;
  for (const auto& tri_ids : mesh.boundary_elements) {
    Triangle tri{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]};
    scene.push_back(tri);
  }

}

void UniversalMeshing::loop() {

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

  ImGui::DragInt("n", &n, 0.5f, 8, 128);
  ImGui::DragFloat("blend_radius", &blend_radius, 0.001f, 0, 0.25, "%.5f");
  ImGui::Checkbox("overlap correction", &overlap_correction);

  if (ImGui::Button("remesh")) {
    remesh();
  }

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

UniversalMeshing::UniversalMeshing() : Application(), scene(), keys_down{} {

  auto edges = read_binary<std::array<int,2>>(GEOMETRY_DATA_DIR"icosahedron_edges.bin");
  auto vertices = read_binary<vec3f>(GEOMETRY_DATA_DIR"icosahedron_vertices.bin");

  r = 0.15;
  for (auto edge : edges) {
    capsules.push_back({vertices[edge[0]], vertices[edge[1]], r, r});
  }

  camera_speed = 0.015;
  camera.lookAt(glm::vec3(0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 0.01f, 100.0f);

  overlap_correction = false;

  n = 32;
  ndvr = 3;
  r = 0.25f;
  x = 0.0f;
  y = 0.0f;
  blend_radius = 0;

  scene.color = gray;

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