#include <random>

#include "BVH.hpp"
#include "geometry/geometry.hpp"
#include "geometry/parse_dat.hpp"
#include "fm/operations/print.hpp"

#include "timer.hpp"
#include "binary_io.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

struct Logpile : public Application {

  Logpile();

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

  int n;
  float r;
  float t;

  bool use_quadratic_elements;
  int logs_per_layer;
  float length;
  float dz;
  std::vector< float > qualities;

  BVH<3> bvh;
  std::vector< Capsule > capsules;

  bool sdf_from_simplex_mesh;
  SimplexMesh<3> mesh;

  void remesh();
  void update_camera_position();

};

rgbcolor red{255, 40, 40, 255};
rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (Logpile*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (Logpile*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (Logpile*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (Logpile*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on



void Logpile::key_callback(GLFWwindow* window,
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

void Logpile::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void Logpile::mouse_motion_callback(GLFWwindow* window,
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

void Logpile::mouse_button_callback(GLFWwindow* window,
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

void Logpile::update_camera_position() {
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

void Logpile::remesh() {

  scene.clear();

  float blend_distance = 0.2 * r;
  vec3f widths = bvh.global.max - bvh.global.min;
  float cell_size = std::max(std::max(widths[0], widths[1]), widths[2]) / n;

  auto bounds = bvh.global;

  bounds.max += 0.05f * widths;
  bounds.min -= 0.05f * widths;

  bounds.max[2] = bounds.min[2] + 1.1 * widths[0];

  std::cout << bounds.min << ", " << bounds.max << std::endl;

  float dx = 1.5 * cell_size;
  if (!sdf_from_simplex_mesh) {
    dx += 2 * blend_distance;
  }

  AABB<3>cut = {
    {-length * 0.5f, -length * 0.5f, 0.0f},
    {+length * 0.5f, +length * 0.5f, length}
  };

  std::function<float(vec3f)> f = (sdf_from_simplex_mesh) ? SDF(mesh, dx) : [&](vec3f x) -> float {
    AABB<3>box{
      {x[0] - dx, x[1] - dx, x[2] - dx}, 
      {x[0] + dx, x[1] + dx, x[2] + dx}
    };

    double value = 0.0;
    bvh.query(box, [&](int i) {
      value += exp(-capsules[i].SDF(x) / blend_distance);
    });
    return std::max(cut.SDF(x), float(-blend_distance * log(value)));
  };

  mesh = universal_mesh(f, cell_size, bounds, 0.5, 0.05, 3, -1);

  //if (use_quadratic_elements) {
  //  promote_to_quadratic(mesh, f, cell_size);
  //}

  std::cout << mesh.vertices.size() << " " << mesh.elements.size() << " " << mesh.boundary_elements.size() << std::endl;

  auto & v = mesh.vertices;

  scene.color = gray;
  for (const auto& tri_ids : mesh.boundary_elements) {
    Triangle tri{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]};
    scene.push_back(tri);
  }

}

void Logpile::loop() {

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
  ImGui::DragFloat("radius", &r, 0.005f, 0.5f, 2.5f, "%.5f");
  ImGui::DragFloat("t", &t, 0.0005f, 0.0f, 1.0f, "%.5f");
  ImGui::Checkbox("from simplex mesh", &sdf_from_simplex_mesh);
  if (ImGui::Button("Remesh")) {
    remesh();
  }

  ImGui::PlotLines("qualities", qualities.data(), std::min(qualities.size(), std::size_t(250)), 0, 0, 0.0f, 1.0f, ImVec2(0, 250.0f));

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

struct DataFormat {
  vec3f p;
  int material;
};

Logpile::Logpile() : Application(), scene(), keys_down{} {

  sdf_from_simplex_mesh = false;
  use_quadratic_elements = false;

  n = 32;
  r = 1.0f;

  logs_per_layer = 6;
  length = 4 * logs_per_layer;
  dz = 2 * 0.9f * r;

  mat3f R = {{
    {0, -1, 0},
    {1,  0, 0},
    {0,  0, 1}
  }};

  for (int layer = 0; layer < 8; layer++) {
    for (int i = 0; i < logs_per_layer; i++) {
      float t = 0.8 * ((float(i) / (logs_per_layer - 1)) - 0.5f);
      vec3f p[2] = {
        {-0.5f * length, length * t, layer * dz + 1.05f * r},
        {+0.5f * length, length * t, layer * dz + 1.05f * r}
      };

      if (layer % 2 == 1) {
        p[0] = dot(R, p[0]);
        p[1] = dot(R, p[1]);
      }

      capsules.push_back(Capsule{p[0], p[1], r, r});
    }
  }

  std::vector< AABB<3>> bounding_boxes(capsules.size());
  for (uint32_t i = 0; i < capsules.size(); i++) {
    bounding_boxes[i] = bounding_box(capsules[i]);
  }
  bvh = BVH(bounding_boxes);

  scene.color = gray;

  auto [min, max] = bvh.global;
  auto width = max - min;
  auto center = (min + max) * 0.5f;
  auto pov = center + 0.6 * width;
  camera.lookAt(glm::vec3(pov[0], pov[1], pov[2]), glm::vec3(center[0], center[1], center[2]));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 1.0f, 1000.0f);

  camera_speed = 0.01 * norm(max - min);

  remesh();

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};

int main() {
  Logpile app;
  app.run();
  return 0;
}