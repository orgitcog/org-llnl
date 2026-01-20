#include <random>

#include "BVH.hpp"
#include "binary_io.hpp"
#include "geometry/geometry.hpp"

#include "timer.hpp"

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
  bool first;
  Scene scene;
  Camera camera;

  float camera_speed;
  bool keys_down[256];
  double mouse_x, mouse_y;

  bool lmb_down = false;
  bool mmb_down = false;
  bool rmb_down = false;

  // parameters for the meshing example
  int n;
  int ndvr;
  float cell_size;
  float blend_radius;
  float soc;
  std::vector < float > qualities;

  float v[5];

  std::vector< Capsule > capsules;

  BVH<2> bvh;
  BackgroundGrid<2> grid;

  void remesh();
  void update_camera_position();
  void initialize_grid_values();

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
    if (ImGui::GetIO().WantCaptureMouse) {
      // if the mouse is interacting with ImGui
    } else {
      //camera.rotate(altitude, -azimuth);
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
  if (keys_down[uint8_t('w')]) { camera.move_up(scale * camera_speed); }
  if (keys_down[uint8_t('a')]) { camera.move_left(scale * camera_speed); }
  if (keys_down[uint8_t('s')]) { camera.move_down(scale * camera_speed); }
  if (keys_down[uint8_t('d')]) { camera.move_right(scale * camera_speed); }
  // clang-format on
}

void UniversalMeshing::initialize_grid_values() {

  std::function<float(vec2f)> sdf = [&](vec2f x) -> float {

    float d = soc * (1.0 + 0.4 * x[0]);

    float dx = 1.5 * grid.cell_size() + 2.0 * blend_radius + d;

    AABB<2>box{
      {x[0] - dx, x[1] - dx}, 
      {x[0] + dx, x[1] + dx}
    };

    if (blend_radius == 0) {
      float value = 2 * dx;
      bvh.query(box, [&](int i) {
        value = std::min(value, capsules[i].SDF(vec3f{x[0], x[1], 0.0f}));
      });
      return value - d;
    } else {
      double r = blend_radius;
      double value = 0.0;
      bvh.query(box, [&](int i) {
        value += exp(-capsules[i].SDF(vec3f{x[0], x[1], 0.0f}) / r);
      });
      return -r * log(value) - d;
    }

  };

  for (int j = 0; j < 2 * grid.n[1] + 1; j++) {
    for (int i = 0; i < 2 * grid.n[0] + 1; i++) {
      grid({i,j}) = sdf(grid.vertex({i,j}));
    }
  }

}

void UniversalMeshing::remesh() {

  auto mesh = universal_mesh(grid, 0.5f, 0.05f, ndvr);

  auto & v = mesh.vertices;

  std::cout << "generated mesh with:" << std::endl;
  std::cout << "  " << mesh.vertices.size() << " vertices" << std::endl;
  std::cout << "  " << mesh.elements.size() << " elements" << std::endl;
  std::cout << "  " << mesh.boundary_elements.size() << " boundary_elements" << std::endl;

  qualities.reserve(mesh.elements.size());
  qualities.clear();
  scene.clear();

  rgbcolor light_green{128, 177, 102, 255};
  scene.color = light_green;
  for (const auto& tri_ids : mesh.elements) {
    Triangle tri{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]};
    scene.push_back(tri);
    qualities.push_back(quality(tri));
  }

  rgbcolor darker_green{88, 137, 68, 255};
  scene.color = darker_green;
  for (const auto& tri_ids : mesh.elements) {
    scene.push_back(Line{v[tri_ids[0]], v[tri_ids[1]]});
    scene.push_back(Line{v[tri_ids[1]], v[tri_ids[2]]});
    scene.push_back(Line{v[tri_ids[2]], v[tri_ids[0]]});
  }

  rgbcolor black{20, 20, 20, 255};
  scene.color = black;
  for (const auto& edge_ids : mesh.boundary_elements) {
    scene.push_back(Line{v[edge_ids[0]], v[edge_ids[1]]});
  }

  std::sort(qualities.begin(), qualities.end());

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
  scene.draw(camera.matrix());

  // render UI stuff
  ImGui::Begin("Meshing Parameters");


  bool should_remesh = false; 
  should_remesh |= ImGui::DragFloat("cell_size", &cell_size, 0.005f, 0.01f, 0.1f);
  should_remesh |= ImGui::DragInt("ndvr", &ndvr, 0.5f, 0, 5);
  should_remesh |= ImGui::DragFloat("blend_radius", &blend_radius, 0.0002f, 0, 0.01f);
  should_remesh |= ImGui::DragFloat("soc", &soc, 0.0005f, 0.0f, 0.3);
  if (should_remesh) { 
    grid = BackgroundGrid<2>({{-1, -1}, {1, 1}}, cell_size);
    initialize_grid_values(); 
    remesh(); 
  };

  ImGui::PlotLines("qualities", qualities.data(), qualities.size(), 0, 0, 0.0f, 1.0f, ImVec2(0, 250.0f));

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

UniversalMeshing::UniversalMeshing() : Application(), scene(), keys_down{},
  grid({{-1, -1}, {1, 1}}, 0.02f) {

  ndvr = 0;
  cell_size = grid.cell_size();

  soc = 0.0f;
  blend_radius = 0.01;

  auto vedges = read_binary<mat2f>(GEOMETRY_DATA_DIR"voronoi_edges.bin");

  float r = 0.05f;
  for (auto edge : vedges) {
    vec3f p = {edge[0][0], edge[0][1], 0.0f};
    vec3f q = {edge[1][0], edge[1][1], 0.0f};
    capsules.push_back(Capsule{p, q, r, r});
  }

  std::vector< AABB<2> > bounding_boxes(capsules.size());
  for (uint32_t i = 0; i < capsules.size(); i++) {
    AABB<3> box = bounding_box(capsules[i]);
    bounding_boxes[i] = AABB<2>{{box.min[0], box.min[1]}, {box.max[0], box.max[1]}};
  }
  bvh = BVH<2>(bounding_boxes);

  initialize_grid_values();

  camera_speed = 0.015;
  camera.lookAt(glm::vec3(-0.7, 0, 1), glm::vec3(-0.7, 0, 0), glm::vec3(0, 1, 0));
  camera.orthographic(2.4f, getWindowRatio(), 0.1f, 10.0f);

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