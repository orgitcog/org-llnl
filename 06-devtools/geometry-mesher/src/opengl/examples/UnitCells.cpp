#include <random>

#include "BVH.hpp"
#include "geometry/geometry.hpp"
#include "geometry/unit_cells.hpp"

#include "timer.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

vec3f fractional_part(vec3f p) {
  return vec3f{ p[0] - floorf(p[0]), p[1] - floorf(p[1]), p[2] - floorf(p[2]) };
}

std::vector< Line > to_lines(AABB<3>box) {
  auto [min, max] = box;

  return {
    Line{{{min[0], min[1], min[2]}, {max[0], min[1], min[2]}}},
    Line{{{max[0], min[1], min[2]}, {max[0], max[1], min[2]}}},
    Line{{{max[0], max[1], min[2]}, {min[0], max[1], min[2]}}},
    Line{{{min[0], max[1], min[2]}, {min[0], min[1], min[2]}}},

    Line{{{min[0], min[1], min[2]}, {min[0], min[1], max[2]}}},
    Line{{{max[0], min[1], min[2]}, {max[0], min[1], max[2]}}},
    Line{{{min[0], max[1], min[2]}, {min[0], max[1], max[2]}}},
    Line{{{max[0], max[1], min[2]}, {max[0], max[1], max[2]}}},

    Line{{{min[0], min[1], max[2]}, {max[0], min[1], max[2]}}},
    Line{{{max[0], min[1], max[2]}, {max[0], max[1], max[2]}}},
    Line{{{max[0], max[1], max[2]}, {min[0], max[1], max[2]}}},
    Line{{{min[0], max[1], max[2]}, {min[0], min[1], max[2]}}}
  };
}

inline mat3f euler_to_rotation(const vec3f & pyr) {
  float CP = cos(pyr[0]);
  float SP = sin(pyr[0]);
  float CY = cos(pyr[1]);
  float SY = sin(pyr[1]);
  float CR = cos(pyr[2]);
  float SR = sin(pyr[2]);

  mat3f theta;

  // front direction
  theta(0, 0) = CP * CY;
  theta(1, 0) = CP * SY;
  theta(2, 0) = SP;

  // left direction
  theta(0, 1) = CY * SP * SR - CR * SY;
  theta(1, 1) = SY * SP * SR + CR * CY;
  theta(2, 1) = -CP * SR;

  // up direction
  theta(0, 2) = -CR * CY * SP - SR * SY;
  theta(1, 2) = -CR * SY * SP + SR * CY;
  theta(2, 2) = CP * CR;

  return theta;
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
  int blocks;
  int n;
  int ndvr;
  float dvr_step;
  float blend_distance;

  float inner_radius = 0.1;
  float outer_radius = 0.1;
  UnitCell::Type truss_type;

  float scaling;
  vec3f widths;
  vec3f offset;
  vec3f euler_angles;

  SimplexMesh<3> mesh;

  void remesh();
  void reslice();
  void update_camera_position();

};

rgbcolor red{255, 40, 40, 255};
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

  float dx = 1.0f / n;

  vec3f center = {0.5f, 0.5f, 0.5f};
  AABB<3>clip_box{center - widths * 0.5f, center + widths * 0.5f};
  AABB<3>sampling_bounds{clip_box.min - vec3f{dx, dx, dx}, clip_box.max + vec3f{dx, dx, dx}};

  mat3f R = euler_to_rotation(euler_angles);

  UnitCell u(truss_type, outer_radius, inner_radius);

  std::function<float(vec3f)> f = [=](vec3f p) { 
    float sdf = u.SDF(fractional_part((dot(R, p + offset - center) * scaling + center) * float(blocks)));
    return std::max(clip_box.SDF(p), sdf);
  };

  mesh = universal_mesh(f, dx, sampling_bounds, 0.5f, dvr_step, ndvr);

  auto & v = mesh.vertices;

  scene.clear();

  scene.color = gray;
  for (const auto& tri_ids : mesh.boundary_elements) {
    scene.push_back(Triangle{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]});
  }

  float total_volume = 0.0f;
  for (const auto& tet : mesh.elements) {
    total_volume += volume(Tetrahedron{v[tet[0]], v[tet[1]], v[tet[2]], v[tet[3]]});
    //scene.push_back(Triangle{v[tet[0]], v[tet[2]], v[tet[1]]});
    //scene.push_back(Triangle{v[tet[1]], v[tet[2]], v[tet[3]]});
    //scene.push_back(Triangle{v[tet[2]], v[tet[0]], v[tet[3]]});
    //scene.push_back(Triangle{v[tet[0]], v[tet[1]], v[tet[3]]});
  }

  std::cout << "volume fraction: " << total_volume / volume(sampling_bounds) << std::endl;

  //scene.color = red;
  //scene.push_back(to_lines(sampling_bounds));
  //scene.push_back(to_lines(unit_cell_bounds));

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
  scene.draw(camera.matrix());
  scene.draw_wireframe(camera.matrix());

  // render UI stuff
  ImGui::Begin("Settings");

  ImGui::DragInt("mesh resolution", &n, 0.1f, 8, 256);
  ImGui::DragInt("unit cells", &blocks, 0.1f, 1, 4);

  ImGui::DragFloat("inner radius", &inner_radius, 0.001f, 0.0f, 0.09f, "%.5f");
  ImGui::DragFloat("outer radius", &outer_radius, 0.001f, 0.1f, 0.1f, "%.5f");

  ImGui::RadioButton("SC", (int*)&truss_type, 0); ImGui::SameLine();
  ImGui::RadioButton("Iso", (int*)&truss_type, 1); ImGui::SameLine();
  ImGui::RadioButton("Octet", (int*)&truss_type, 2); ImGui::SameLine();
  ImGui::RadioButton("ORC", (int*)&truss_type, 3); ImGui::SameLine();
  ImGui::RadioButton("RD", (int*)&truss_type, 4); ImGui::SameLine();
  ImGui::RadioButton("TO", (int*)&truss_type, 5);

  ImGui::DragFloat("pitch", &euler_angles[0], 0.001f, -3.14f, 3.14f, "%.5f");
  ImGui::DragFloat("yaw", &euler_angles[1], 0.001f, -3.14f, 3.14f, "%.5f");
  ImGui::DragFloat("roll", &euler_angles[2], 0.001f, -3.14f, 3.14f, "%.5f");
  ImGui::DragFloat("x offset", &offset[0], 0.001f, 0.0f, 1.0f, "%.5f");
  ImGui::DragFloat("y offset", &offset[1], 0.001f, 0.0f, 1.0f, "%.5f");
  ImGui::DragFloat("z offset", &offset[2], 0.001f, 0.0f, 1.0f, "%.5f");

  ImGui::DragFloat("scale", &scaling, 0.001f, 0.25f, 1.50f, "%.5f");

  ImGui::DragFloat("x clip", &widths[0], 0.001f, 0.5f, 1.5f, "%.5f");
  ImGui::DragFloat("y clip", &widths[1], 0.001f, 0.5f, 1.5f, "%.5f");
  ImGui::DragFloat("z clip", &widths[2], 0.001f, 0.5f, 1.5f, "%.5f");

  if (ImGui::Button("Remesh")) {
    remesh();
  }

  ImGui::Dummy(ImVec2(0.0f, 20.0f));

  static char filename[128] = "output.stl";
  ImGui::InputText("STL filename", filename, 128);
  if (ImGui::Button("Export STL")) {
    export_stl(mesh, filename);
  }

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

UnitCells::UnitCells() : Application(), scene(), keys_down{} {

  camera_speed = 0.015;
  camera.lookAt(glm::vec3(1.5, 1.5, 1.5), glm::vec3(0.5f, 0.5f, 0.5f));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 0.01f, 100.0f);

  ndvr = 3;
  dvr_step = 0.05;

  n = 64;
  blocks = 1;
  blend_distance = 0.003;
  inner_radius = 0.025f;
  outer_radius = 0.050f;
  truss_type = UnitCell::OCTET;

  offset = {};
  scaling = 1.0f;
  widths = {1.0f, 1.0f, 1.0f};
  euler_angles = {};


  scene.color = gray;

  remesh();

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};

int main() {
  UnitCells app;
  app.run();
  return 0;
}