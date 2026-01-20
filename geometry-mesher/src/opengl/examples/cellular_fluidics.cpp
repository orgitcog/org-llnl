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

int show_hexes = 0;
int show_capsules = 1;
int show_fluid = 2;

template < int m, int n >
std::vector< std::string > to_strings(const bool (&mask)[m][n]) {
  std::vector< std::string > output(m, std::string(n, ' '));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      output[i][j] = mask[i][j] ? 'X' : ' ';
    }
    std::cout << output[i] << std::endl;
  }

  return output;
}

struct CellularFluidics : public Application {

  CellularFluidics();

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

  bool show_wireframe = false;

  int display_mode;

  static constexpr int nrows = 5;
  static constexpr int ncols = 11;
  bool mask[nrows][ncols] = {
    {0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0}
  };

  float cell_size;
  float r1;
  float r2;
  HexLattice lattice;

  void remesh();
  void update_camera_position();

};

rgbcolor red{255, 40, 40, 255};
rgbcolor gray{40, 40, 40, 255};
rgbcolor light_gray{140, 140, 140, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (CellularFluidics*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (CellularFluidics*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (CellularFluidics*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (CellularFluidics*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

void CellularFluidics::key_callback(GLFWwindow* window,
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

void CellularFluidics::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void CellularFluidics::mouse_motion_callback(GLFWwindow* window,
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

void CellularFluidics::mouse_button_callback(GLFWwindow* window,
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

void CellularFluidics::update_camera_position() {

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

void CellularFluidics::remesh() {

  lattice = HexLattice(to_strings(mask));

  scene.clear();

  if (display_mode == show_hexes) {

    const auto & v = lattice.vertices;
    for (uint32_t quad_id : lattice.boundary_faces) {
      auto [i,j,k,l] = lattice.faces[quad_id];
      scene.color = gray;
      scene.push_back(Triangle{v[i], v[j], v[k]});
      scene.push_back(Triangle{v[k], v[l], v[i]});

      scene.color = light_gray;
      scene.push_back(Line{v[i], v[j]});
      scene.push_back(Line{v[j], v[k]});
      scene.push_back(Line{v[k], v[l]});
      scene.push_back(Line{v[l], v[i]});
    }

    show_wireframe = false;

  } 
  if (display_mode == show_fluid || display_mode == show_capsules) {

    std::vector<float> radii(lattice.vertices.size(), 0.0);

    float ymin = lattice.bounds.min[1];
    float ymax = lattice.bounds.max[1];
    for (uint32_t i = 0; i < lattice.vertices.size(); i++) {
      auto y = lattice.vertices[i][1];
      float t = (y - ymin) / (ymax - ymin);
      radii[i] = r1 * (1.0f - t) + r2 * t;
    }

    SimplexMesh<3> mesh;
    if (display_mode == show_capsules) {
      mesh = lattice.capsule_mesh(radii, cell_size);
    } else {
      mesh = lattice.fluid_mesh(radii, cell_size);
    }
    std::cout << mesh.vertices.size() << " " << mesh.elements.size() << " " << mesh.boundary_elements.size() << std::endl;
    auto & v = mesh.vertices;

    scene.color = gray;
    for (const auto& tri_ids : mesh.boundary_elements) {
      Triangle tri{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]};
      scene.push_back(tri);
    }

    show_wireframe = true;

  }

}

void CellularFluidics::loop() {

  update_camera_position();

  glClearColor(0.169f, 0.314f, 0.475f, 1.0f);
  //glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // feed inputs to dear imgui, start new frame
  // these go before we render our stuff
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // render our stuff
  camera.set_aspect(getWindowRatio());
  scene.draw(camera.matrix());
  if (show_wireframe) {
    scene.draw_wireframe(camera.matrix());
  }

  // render UI stuff
  ImGui::Begin("Meshing Parameters");

  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      ImGui::PushID(i * ncols + j); 
      ImGui::Checkbox("", &(mask[i][j])); 
      ImGui::PopID();
      if (j != ncols - 1) {
        ImGui::SameLine();
      }
    }
  }

  ImGui::Dummy(ImVec2(0.0f, 20.0f));

  ImGui::DragFloat("cell size", &cell_size, 0.001f, 0.03f, 0.5f, "%.5f");
  ImGui::DragFloat("r1", &r1, 0.005f, 0.1f, 0.5f, "%.5f");
  ImGui::DragFloat("r2", &r2, 0.005f, 0.05f, 0.35f, "%.5f");

  ImGui::Dummy(ImVec2(0.0f, 20.0f));

  ImGui::RadioButton("hexes", &display_mode, 0); ImGui::SameLine();
  ImGui::RadioButton("capsules", &display_mode, 1); ImGui::SameLine();
  ImGui::RadioButton("fluid", &display_mode, 2);

  ImGui::Dummy(ImVec2(0.0f, 20.0f));

  if (ImGui::Button("Remesh")) { remesh(); }

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

CellularFluidics::CellularFluidics() : Application(), scene(), keys_down{} {

  r1 = 0.2f;
  r2 = 0.2f;
  cell_size = 0.2f;

  lattice = HexLattice(to_strings(mask));

  remesh();

  scene.color = gray;

  auto [min, max] = lattice.bounds;
  auto width = max - min;
  auto center = (min + max) * 0.5f;
  auto pov = center + 0.5f * norm(width) * vec3f{1,1,1};
  camera.lookAt(glm::vec3(pov[0], pov[1], pov[2]), glm::vec3(center[0], center[1], center[2]));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 0.05f, 100.0f);

  camera_speed = 0.01 * norm(max - min);

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};

int main() {
  CellularFluidics app;
  app.run();
  return 0;
}