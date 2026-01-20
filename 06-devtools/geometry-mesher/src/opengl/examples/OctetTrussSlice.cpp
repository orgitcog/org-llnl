#include <random>

#include "BVH.hpp"
#include "geometry/geometry.hpp"

#include "timer.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

auto octet_truss_SDF(float r, float blend = 0.01f) {
  std::array < Capsule, 24 > cylinders{
    Capsule{{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, r, r},
    Capsule{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, r, r},
    Capsule{{0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, r, r},
    Capsule{{1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, r, r},
    Capsule{{0.0, 0.0, 0.0}, {0.0, 1.0, 1.0}, r, r},
    Capsule{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, r, r},
    Capsule{{1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, r, r},
    Capsule{{1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, r, r},
    Capsule{{0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, r, r},
    Capsule{{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, r, r},
    Capsule{{0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, r, r},
    Capsule{{1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, r, r},
    Capsule{{0.0, 0.5, 0.5}, {0.5, 0.5, 1.0}, r, r},
    Capsule{{1.0, 0.5, 0.5}, {0.5, 0.5, 1.0}, r, r},
    Capsule{{0.5, 0.0, 0.5}, {0.5, 0.5, 1.0}, r, r},
    Capsule{{0.5, 1.0, 0.5}, {0.5, 0.5, 1.0}, r, r},
    Capsule{{0.5, 0.5, 0.0}, {0.0, 0.5, 0.5}, r, r},
    Capsule{{0.5, 0.5, 0.0}, {1.0, 0.5, 0.5}, r, r},
    Capsule{{0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, r, r},
    Capsule{{0.5, 0.5, 0.0}, {0.5, 1.0, 0.5}, r, r},
    Capsule{{0.5, 0.0, 0.5}, {1.0, 0.5, 0.5}, r, r},
    Capsule{{1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, r, r},
    Capsule{{0.5, 1.0, 0.5}, {0.0, 0.5, 0.5}, r, r},
    Capsule{{0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, r, r}
  };

  return [=](vec3f p) {
    float v = 0.0;
    for (auto& cylinder : cylinders) {
      v += exp(-cylinder.SDF(p) / blend);
    }
    return -blend * log(fmax(v, 1.0e-6));
  };
}

auto octet_truss_SDF_sym(vec3f p, float r) {

  static constexpr vec3f q[4] = {
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},
    {1.0f, 1.0f, 1.0f}
  };

  vec3f x = {
    std::abs(2.0f * p[0] - 1.0f),  
    std::abs(2.0f * p[1] - 1.0f),  
    std::abs(2.0f * p[2] - 1.0f),
  };

  return std::min(Capsule{q[0], q[1], 2 * r, 2 * r}.SDF(x),
         std::min(Capsule{q[1], q[2], 2 * r, 2 * r}.SDF(x),
         std::min(Capsule{q[2], q[0], 2 * r, 2 * r}.SDF(x),
         std::min(Capsule{q[0], q[3], 2 * r, 2 * r}.SDF(x),
         std::min(Capsule{q[1], q[3], 2 * r, 2 * r}.SDF(x),
                  Capsule{q[2], q[3], 2 * r, 2 * r}.SDF(x)))))) / 2.0f;

}

struct UnitCells : public Application {

  UnitCells();

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
  float rod_radius = 0.1;
  float z = 0.25f;
  int n;
  int ndvr;
  float dvr_step;
  float blend_distance;

  std::vector< Capsule > capsules; 

  void remesh();
  void reslice();
  void update_camera_position();

};

rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (UnitCells*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (UnitCells*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (UnitCells*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (UnitCells*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

void UnitCells::key_callback(GLFWwindow* window,
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

void UnitCells::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void UnitCells::mouse_motion_callback(GLFWwindow* window,
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

void UnitCells::mouse_button_callback(GLFWwindow* window,
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

void UnitCells::update_camera_position() {
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

void UnitCells::remesh() {

  AABB<3>bounds{{-0.3, -0.3, -0.3}, {1.3, 1.3, 1.3}};

  std::function<float(vec3f)> f = [=](vec3f p) { return octet_truss_SDF_sym(p, rod_radius); };

  scene.clear();

  auto mesh = universal_mesh(f, 2.0 / n, bounds);

  auto & v = mesh.vertices;

  for (const auto& tri_ids : mesh.boundary_elements) {
    scene.push_back(Triangle{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]});
  }

}

void UnitCells::reslice() {

  AABB<3>bounds{{0.0, 0.0, z}, {1.0, 1.0, z}};

  std::function<float(vec2f)> f = [=](vec2f p) { return octet_truss_SDF_sym({p[0], p[1], z}, rod_radius); };

  slice.clear();

  auto mesh = universal_mesh(f, 0.01, bounds, 0.5f, dvr_step, ndvr);

  slice.color = rgbcolor{255, 0, 0, 255};
  for (auto& tri_ids : mesh.elements) {
    Triangle tri{mesh.vertices[tri_ids[0]], 
                 mesh.vertices[tri_ids[1]], 
                 mesh.vertices[tri_ids[2]]}; 
    for (auto & v : tri.vertices) { v[2] = z; }
    slice.push_back(tri);
  }

}

void UnitCells::loop() {

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
  //scene.draw(camera.matrix());
  scene.draw_wireframe(camera.matrix());

  //slice.draw(camera.matrix());
  slice.draw_wireframe(camera.matrix());

  // render UI stuff
  ImGui::Begin("Meshing Parameters");

  bool should_remesh = false; 
  should_remesh |= ImGui::DragFloat("rod radius", &rod_radius, 0.0001f, 0.0f, 0.15f, "%.5f");
  if (should_remesh) { remesh(); };

  bool should_reslice = ImGui::DragFloat("z height", &z, 0.002f, 0.0f, 1.0f, "%.5f");
  if (should_remesh || should_reslice) { reslice(); };


  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

UnitCells::UnitCells() : Application(), scene(), keys_down{} {

  camera_speed = 0.015;
  camera.lookAt(glm::vec3(1.5, 1.5, 1.5), glm::vec3(0.5f, 0.5f, 0.5f));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 0.01f, 100.0f);

  n = 64;
  blend_distance = 0.003;
  rod_radius = 0.1f;

  ndvr = 3;
  dvr_step = 0.05;

  scene.color = gray;

  remesh();
  reslice();

//  std::vector< AABB<3>> bounding_boxes(capsules.size());
//  for (int i = 0; i < capsules.size(); i++) {
//    bounding_boxes[i] = bounding_box(capsules[i]);
//  }
//  bvh = BVH(bounding_boxes);
//  initialize(2);

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};


int main() {
  #if 1
  UnitCells app;
  app.run();
  return 0;
  #else

  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  int n = 1024;

  double error = 0.0f;
  std::vector< std::array< float, 4 > > data(n);
  for (int i = 0; i < n; i++) {
    data[i] = { dist(gen), dist(gen), dist(gen), dist(gen) * 0.14f };

    auto r = data[i][3];
    auto x = vec3f{data[i][0], data[i][1], data[i][2]};
    auto sdf1 = octet_truss_SDF(r)(x);
    auto sdf2 = octet_truss_SDF_sym(x, r);

    if (i < 32) {
      std::cout << "{" << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << "}: ";
      std::cout << sdf1 << ", ";
      std::cout << sdf2 << std::endl;
    }

    error += (sdf1 - sdf2) * (sdf1 - sdf2);
  }
  std::cout << "norm(error) / n: " << sqrt(error) / n << std::endl;

  timer stopwatch; 

  stopwatch.start();
  float total = 0.0f;
  for (auto & x : data) {
    total += octet_truss_SDF_sym({x[0], x[1], x[2]}, x[3]);
  }
  stopwatch.stop();
  std::cout << stopwatch.elapsed() << ": " << total << std::endl;

  stopwatch.start();
  total = 0.0f;
  for (auto & x : data) {
    auto sdf = octet_truss_SDF(x[3], 0.01f);
    total += sdf({x[0], x[1], x[2]});
  }
  stopwatch.stop();
  std::cout << stopwatch.elapsed() << ": " << total << std::endl;

  #endif
}