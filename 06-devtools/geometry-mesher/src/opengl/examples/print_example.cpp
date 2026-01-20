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

std::string datafile;
std::string output_prefix = "mesh_";
int num_materials = 0;

struct DataFormat {
  vec3f p;
  int material;
};

struct PrintExample : public Application {

  PrintExample();

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

  std::vector< DataFormat > raw_data;

  int n;
  int sdf_type; // 0 = min, 1 = smoothmin
  float blend_ratio;
  float r;
  float t;
  std::vector< float > qualities;

  std::vector< BVH<3> > bvh;
  std::vector< std::vector< Capsule > > capsules;
  std::vector< std::vector< uint64_t > > capsule_ids;

  void update_bvh();
  void remesh();
  void update_camera_position();

};

rgbcolor red{255, 40, 40, 255};
rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (PrintExample*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (PrintExample*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (PrintExample*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (PrintExample*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

void PrintExample::key_callback(GLFWwindow* window,
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

void PrintExample::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void PrintExample::mouse_motion_callback(GLFWwindow* window,
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

void PrintExample::mouse_button_callback(GLFWwindow* window,
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

void PrintExample::update_camera_position() {
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

void PrintExample::update_bvh() {

  for (auto & arr : capsules) {
    for (auto & capsule : arr) { capsule.r1 = capsule.r2 = r; }
  }

  for (int k = 0; k < num_materials; k++) {
    std::vector< AABB<3>> bounding_boxes(capsules[k].size());
    for (uint32_t i = 0; i < capsules[k].size(); i++) {
      bounding_boxes[i] = bounding_box(capsules[k][i]);
    }
    bvh[k] = BVH(bounding_boxes);
  }

}

void PrintExample::remesh() {

  scene.clear();

  float blend_distance = blend_ratio * r;
  vec3f widths = bvh[0].global.max - bvh[0].global.min;
  float cell_size = std::max(std::max(widths[0], widths[1]), widths[2]) / n;

  auto bounds = bvh[0].global;

  bounds.max += 0.15f * widths;
  bounds.min -= 0.15f * widths;

  float dx = 1.5 * cell_size + 2 * blend_distance;

  for (int k = 0; k < num_materials; k++) {

    std::function<float(vec3f)> f = [&](vec3f x) -> float {
      AABB<3>box{
        {x[0] - dx, x[1] - dx, x[2] - dx}, 
        {x[0] + dx, x[1] + dx, x[2] + dx}
      };

      if (sdf_type == 0) {
        float value = dx;
        bvh[k].query(box, [&](int i) {
          value = std::min(value, capsules[k][i].SDF(x));
        });
        return value;
      } else {
        double value = 0.0;
        bvh[k].query(box, [&](int i) {
          value += exp(-capsules[k][i].SDF(x) / blend_distance);
        });
        return -blend_distance * log(value);
      }
    };

    auto mesh = universal_mesh(f, cell_size, bounds);

    std::cout << mesh.vertices.size() << " " << mesh.elements.size() << " " << mesh.boundary_elements.size() << std::endl;

    auto & v = mesh.vertices;

    scene.color = gray;
    for (const auto& tri_ids : mesh.boundary_elements) {
      Triangle tri{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]};
      scene.push_back(tri);
    }

    export_stl(mesh, output_prefix + std::to_string(k) + ".stl");

  }

}

void PrintExample::loop() {

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

  ImGui::RadioButton("min", &sdf_type, 0); ImGui::SameLine();
  ImGui::RadioButton("smoothmin", &sdf_type, 1);

  static char buf[128] = "mesh_";
  if (ImGui::InputText("output stl file prefix", buf, 128)) {
    output_prefix = buf;
  }

  if (sdf_type == 1) {
    ImGui::DragFloat("blend ratio", &blend_ratio, 0.005f, 0.1f, 2.0f, "%.5f");
  }

  if (ImGui::Button("Remesh")) {
    update_bvh();
    remesh();
  }

  //ImGui::PlotLines("qualities", qualities.data(), std::min(qualities.size(), std::size_t(250)), 0, 0, 0.0f, 1.0f, ImVec2(0, 250.0f));

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}


PrintExample::PrintExample() : Application(), scene(), keys_down{} {

  n = 64;
  r = 1.05f;
  sdf_type = 0;
  blend_ratio = 0.25f;

  raw_data = read_binary< DataFormat >(datafile);

  num_materials = 0;
  std::size_t skip = 16;
  for (std::size_t i = skip; i < raw_data.size(); i += skip) {
    num_materials = std::max(num_materials, raw_data[i].material);
  }
  num_materials++;

  std::cout << "found " << num_materials << " materials" << std::endl;

  capsules.resize(num_materials);
  capsule_ids.resize(num_materials);
  for (int i = 0; i < num_materials; i++) {
    capsules[i].reserve(raw_data.size() / skip);
    capsule_ids[i].reserve(raw_data.size() / skip);
  }

  for (std::size_t i = skip; i < raw_data.size(); i += skip) {
    auto prev = raw_data[i - skip];
    auto curr = raw_data[i];

    if (prev.material == curr.material && norm(prev.p - curr.p) < 3.0f) {
      capsules[prev.material].push_back(Capsule{prev.p, curr.p, r, r});
    }

    for (int k = 0; k < num_materials; k++) {
      capsule_ids[k].push_back(capsules[k].size());
    }
  }

  bvh.resize(num_materials);
  for (int k = 0; k < num_materials; k++) {
    std::cout << "capsules[" << k << "]: " << capsules[k].size() << std::endl;
    if (capsules[k].size() == 0) {
      std::cout << "error: material " << k << " has zero entries, invalid file format" << std::endl;
      std::exit(1);
    }

    std::vector< AABB<3>> bounding_boxes(capsules[k].size());
    for (uint32_t i = 0; i < capsules[k].size(); i++) {
      bounding_boxes[i] = bounding_box(capsules[k][i]);
    }
    bvh[k] = BVH(bounding_boxes);
  }

  scene.color = gray;

  auto [min, max] = bvh[0].global;
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

int main(int argc, char *argv[]) {

  if (argc == 1) {
    std::cout << "error: must provide an input file as the first argument" << std::endl;
    std::cout << "The input file format is a list of points: {x(float32), y(float32), z(float32), m(int32)}" << std::endl;
    std::cout << "where {x,y,z} are the coordinates of a point on the trajectory, and m is the material label (starting with 0)" << std::endl;
    exit(1);
  }

  if (argc == 2) {
    datafile = std::string(argv[1]);
  }

  if (argc > 2) { 
    std::cout << "too many input arguments" << std::endl;
    exit(1);
  }

  PrintExample app;
  app.run();
  return 0;
}