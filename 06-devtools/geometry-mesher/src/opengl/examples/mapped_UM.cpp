#include <random>
#include <iomanip>

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

std::string input_file;
std::string output_prefix = "mesh";

// map from the quasi-2D plate domain to the 3D mandrel domain
vec3f f(vec3f x, float R) {
  float theta_max = 1.57079632679;
  float phi = atan2(x[1], x[0]);
  float theta = theta_max * norm(vec2f{x[0], x[1]});
  vec3f p = {sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta)};
  vec3f n = p; // only true for spherical symmetry
  return R * (p + x[2] * n);
}

std::vector< vec3f > trajectory(float r, float R, int number_of_layers) {
  std::vector< vec3f > points;
  mat3f Q = Identity<3,float>();
  float theta = 3.14159265359 / number_of_layers;
  float dx = 3 * r / R;
  for (int i = 0; i < number_of_layers; i++) {
    // slightly less than 2*r per layer, so that there is some overlap
    float scale = 1.0f - 1.75f * i * (r / R);
    for (float x = -1.0f; x <= 1.0f; x += dx) {
      float Y = sqrt(1.0 - x * x);
      for (float y = -Y; y <= Y; y += dx) {
        points.push_back(scale * dot(Q, f({x, y, 0.0f}, R)));
      }
    }
    Q = dot(Q, RotationMatrix(vec3f{0.0, 0.0, theta}));
  }
  return points;
}

struct MappedUM : public Application {

  MappedUM();

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
  int sdf_type; // 0 = min, 1 = smoothmin
  float blend_ratio;
  float r_min, r_max, r_filament;
  std::vector< float > qualities;

  SimplexMesh<3> mesh;

  BVH<3> bvh;
  std::vector< Capsule > capsules;

  void update_bvh();
  void remesh();
  void update_camera_position();

};

rgbcolor red{255, 40, 40, 255};
rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (MappedUM*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (MappedUM*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (MappedUM*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (MappedUM*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

void MappedUM::key_callback(GLFWwindow* window,
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

void MappedUM::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void MappedUM::mouse_motion_callback(GLFWwindow* window,
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

void MappedUM::mouse_button_callback(GLFWwindow* window,
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

void MappedUM::update_camera_position() {
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

void MappedUM::update_bvh() {

  for (auto & capsule : capsules) { capsule.r1 = capsule.r2 = r_filament; }

  std::vector< AABB<3>> bounding_boxes(capsules.size());
  for (uint32_t i = 0; i < capsules.size(); i++) {
    bounding_boxes[i] = bounding_box(capsules[i]);
  }
  bvh = BVH(bounding_boxes);

}

void MappedUM::remesh() {

  scene.clear();

  float relative_thickness = (r_max - r_min) / r_min;

  float cell_size = 2.0 / n;
  AABB<3> bounds{
    {-1.1f, -1.1f, -1.1f*relative_thickness}, 
    {+1.1f, +1.1f, +0.1f*relative_thickness}
  };

  float dy = 1.5 * r_max * cell_size;

  std::function<float(vec3f)> sdf = [&](vec3f x) -> float {
    vec3f y = f(x, r_max);

    AABB<3>box{
      {y[0] - dy, y[1] - dy, y[2] - dy}, 
      {y[0] + dy, y[1] + dy, y[2] + dy}
    };

    float value = dy;
    bvh.query(box, [&](int i) {
      value = std::min(value, capsules[i].SDF(y));
    });
    return value;
  };

  mesh = universal_mesh(sdf, cell_size, bounds);

  std::cout << "V: " << mesh.vertices.size() << " T: " << mesh.elements.size() << " F: " << mesh.boundary_elements.size() << std::endl;
  
  for (auto & x : mesh.vertices) { x = f(x, r_max); }

  auto & v = mesh.vertices;

  scene.color = gray;
  for (const auto& tri_ids : mesh.boundary_elements) {
    Triangle tri{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]};
    scene.push_back(tri);
  }

}

void MappedUM::loop() {

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

  ImGui::DragInt("n", &n, 0.5f, 8, 1024);

  if (ImGui::Button("Remesh")) {
    update_bvh();
    remesh();
  }

  static char buf[128] = "mesh_";
  if (ImGui::InputText("output stl file prefix", buf, 128)) {
    output_prefix = buf;
  }

  if (ImGui::Button("export STL")) {
    export_stl(mesh, output_prefix + ".stl");
  }

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

MappedUM::MappedUM() : Application(), scene(), keys_down{} {

  n = 512;
  sdf_type = 0;
  blend_ratio = 0.25f;
  r_filament = 0.125f;

  std::vector< vec3f > points;
  if (input_file.empty()) {
    float outer_radius = 20.0;
    float number_of_layers = 4;
    points = trajectory(r_filament, outer_radius, number_of_layers);
  } else {
    points = read_binary<vec3f>(input_file);
  }

  r_min = +1.0e+10f;
  r_max = -1.0e+10f;
  for (const auto &p : points) {
    float r = norm(p);
    r_min = std::min(r_min, r);
    r_max = std::max(r_max, r);
  }
  r_min -= r_filament;
  r_max += r_filament;

  std::cout << r_min << " " << r_max << std::endl;

  capsules.reserve(points.size());

  for (std::size_t i = 1; i < points.size(); i++) {
    auto prev = points[i-1];
    auto curr = points[i];
    if (norm(prev - curr) < 0.2f * r_max) { // DEBUG
      capsules.push_back(Capsule{prev, curr, r_filament, r_filament});
    }
  }

  std::cout << capsules.size() << std::endl;

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
  float scale = norm(width);
  camera.lookAt(glm::vec3(pov[0], pov[1], pov[2]), glm::vec3(center[0], center[1], center[2]));
  camera.perspective(1.0f /* fov */, getWindowRatio(), scale * 0.001f, scale * 100.0f);
  camera_speed = 0.01 * scale;

  remesh();

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};

int main(int argc, char *argv[]) {

  if (argc == 2) {
    input_file = std::string(argv[1]);
  }

  if (argc > 2) { 
    std::cout << "too many input arguments" << std::endl;
    exit(1);
  }

  MappedUM app;
  app.run();
  return 0;
}
