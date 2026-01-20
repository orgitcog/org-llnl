#include <mutex>
#include <random>
#include <fstream>
#include <iostream>

#include "BVH.hpp"
#include "parallel_for.hpp"
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

using namespace geometry;

constexpr int num_bins = 32;

struct MeshOpt : public Application {

  MeshOpt();

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

  float cell_size;
  float quality_threshold;
  int num_cells;
  std::vector < float > quality_bins;
  SimplexMesh<3> mesh;
  std::string last_filename;

  bool hide;

  void import_mesh(std::string filename);
  void optimize();
  void redraw();
  void update_camera_position();

};

rgbcolor red{255, 40, 40, 255};
rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (MeshOpt*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (MeshOpt*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (MeshOpt*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (MeshOpt*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

void MeshOpt::key_callback(GLFWwindow* window,
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
  if (key == GLFW_KEY_H){ if (action & GLFW_PRESS) { mesh = {}; redraw(); } }
  // clang-format on
};

void MeshOpt::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void MeshOpt::mouse_motion_callback(GLFWwindow* window,
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

void MeshOpt::mouse_button_callback(GLFWwindow* window,
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

void MeshOpt::update_camera_position() {
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

void MeshOpt::import_mesh(std::string filename) {

  std::ifstream infile(filename);
  if (infile) {
    infile.close();
    mesh = import_gmsh22(filename);
  }

  vec3f min = {+1.0e10, +1.0e10, +1.0e10};
  vec3f max = {-1.0e10, -1.0e10, -1.0e10};
  for (auto x : mesh.vertices) {
    for (int i = 0; i < 3; i++) {
      min[i] = std::min(min[i], x[i]);
      max[i] = std::max(max[i], x[i]);
    }
  }

  float L = 0.5 * norm(max - min);
  vec3f focus = 0.5 * (min + max);
  vec3f pov = focus + 2 * vec3f{L, L, L};

  if (last_filename != filename) {
    camera_speed = 0.01 * L;
    camera.lookAt(glm::vec3{pov[0], pov[1], pov[2]}, glm::vec3{focus[0], focus[1], focus[2]});
    camera.set_far_plane(10 * L);
    camera.set_near_plane(0.01 * L);
    last_filename = filename;
  }

  redraw();

}

void MeshOpt::redraw() {

  vec3f red = {1.0f, 0.0f, 0.0f};
  vec3f gray = {0.25f, 0.25f, 0.25f};

  scene.clear();
  for (const auto& [i,j,k] : mesh.boundary_elements) {
    scene.color = rgbcolor{128, 128, 128, 255};
    if (i < j) { scene.push_back(Line{mesh.vertices[i], mesh.vertices[j]}); }
    if (j < k) { scene.push_back(Line{mesh.vertices[j], mesh.vertices[k]}); }
    if (k < i) { scene.push_back(Line{mesh.vertices[k], mesh.vertices[i]}); }
  }

  cell_size = 0.0f; 
  num_cells = mesh.elements.size();

  quality_bins = std::vector<float>(num_bins, 0);
  for (const auto& tet_ids : mesh.elements) {
    Tetrahedron tet = {mesh.vertices[tet_ids[0]],
                       mesh.vertices[tet_ids[1]],
                       mesh.vertices[tet_ids[2]],
                       mesh.vertices[tet_ids[3]]};

    cell_size += volume(tet);
    float q = quality(tet);
    int id = std::max(0.0f, std::min(q * num_bins, num_bins - 1.0f));
    quality_bins[id]++;

    float t = powf(std::max(q, 0.0f), 0.3f);
    vec3f rgb = red * (1 - t) + gray * t;
    if (q < quality_threshold) {
      scene.color = rgbcolor{uint8_t(255 * rgb[0]), uint8_t(255 * rgb[1]), uint8_t(255 * rgb[2]), 255};
      scene.push_back(Triangle{tet.vertices[2], tet.vertices[1], tet.vertices[0]});
      scene.push_back(Triangle{tet.vertices[3], tet.vertices[0], tet.vertices[1]});
      scene.push_back(Triangle{tet.vertices[3], tet.vertices[1], tet.vertices[2]});
      scene.push_back(Triangle{tet.vertices[3], tet.vertices[2], tet.vertices[0]});
    }
  }

  cell_size = cbrt(cell_size / mesh.elements.size());

}

static vec3f compute_unit_normal(const std::vector< vec3f > & v, const std::array< uint64_t, 3 > & tri) {
  return normalize(cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]]));
}

struct BasisR3 {
  mat3f A;
  int count;

  void gram_schmidt(vec3f & n) {
    for (int i = 0; i < count; i++) {
      n -= dot(A[i], n) * A[i];
    }
  }

  void insert(vec3f n /* unit vector */) {
    if (count < 3) {
      gram_schmidt(n);
      if (norm(n) > 1.0e-6) {
        A[count++] = normalize(n);
      }
    }
  }

  vec3f project_out(vec3f n) {
    return n - dot(dot(A, n), A);
  }
};

void MeshOpt::optimize() {

  timer stopwatch;

  stopwatch.start();

  float alpha = 30.0f;
  float step = 0.03 * cell_size * cell_size;
  constexpr int dim = 3;
  constexpr int ndvr = 10;

  constexpr int nmutex = 128;
  std::mutex mtx[nmutex];
  
  //std::vector< char > marker(mesh.vertices.size(), 0);
  std::vector< char > marker(mesh.vertices.size(), 1);
  std::vector< vec3f > normals(mesh.vertices.size(), vec3f{});

  std::vector< BasisR3 > constraint_basis(mesh.vertices.size(), BasisR3{});

  // compute surface normals of each boundary element
  for (uint64_t i = 0; i < mesh.boundary_elements.size(); i++) {
    auto bdr_elem = mesh.boundary_elements[i];
    vec3f n = compute_unit_normal(mesh.vertices, bdr_elem);
    constraint_basis[bdr_elem[0]].insert(n);
    constraint_basis[bdr_elem[1]].insert(n);
    constraint_basis[bdr_elem[2]].insert(n);
  }

  threadpool pool(8);
  std::vector< std::thread > threads;

  for (int k = 0; k < ndvr; k++) {

    std::vector< float > scale(mesh.vertices.size(), 0.0);
    std::vector< vec3f > grad(mesh.vertices.size(), vec3f{});

    pool.parallel_for(mesh.elements.size(), [&](uint32_t i, uint32_t /*tid*/) {
      auto elem_ids = mesh.elements[i];

      Simplex<dim> elem;
      for (int j = 0; j < (dim + 1); j++) {
        elem.vertices[j] = mesh.vertices[elem_ids[j]];
      }

      auto [Q, dQdX] = quality_and_gradient(elem);

      float expQ = expf(-alpha * Q);

      dQdX *= expQ;
      for (int j = 0; j < (dim + 1); j++) {
        vec3f g = vec3f{dQdX[j][0], dQdX[j][1], (dim == 2) ? 0.0f : dQdX[j][2]};

        int which = elem_ids[j] % nmutex;
        mtx[which].lock();
        grad[elem_ids[j]] += g;
        scale[elem_ids[j]] += expQ;
        mtx[which].unlock();
      }
    });

    // does this benefit from multiple threads (?)
    for (uint64_t i = 0; i < mesh.vertices.size(); i++) {
      if (scale[i] != 0.0) {
        vec3f g = grad[i];
        vec3f u = constraint_basis[i].project_out(g);

        mesh.vertices[i] += step * u / scale[i];
      }
    }

  }

  stopwatch.stop();

  std::cout << "dvr iterations: " << stopwatch.elapsed() * 1000 << "ms" << std::endl;

  redraw();

}

void MeshOpt::loop() {

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
  if (!hide) {
    scene.draw(camera.matrix());
    scene.draw_wireframe(camera.matrix());
  }

  // render UI stuff
  ImGui::Begin("Settings");

  static char input_filename[128] = "/Users/sam/stress3D.msh";
  ImGui::InputText("input filename", input_filename, 128);
  if (ImGui::Button("Load Mesh (gmsh22 format)")) {
    import_mesh(input_filename);
  }

  if (ImGui::Button("optimize") && mesh.elements.size() > 0) {
    for (int i = 0; i < 10; i++) {
      optimize();
    }
  }

  if (ImGui::DragFloat("quality threshold", &quality_threshold, 0.005f, 0.05f, 1.0f, "%.3f")) {
    redraw();
  };

  auto label = std::string("Element Quality\n") + std::to_string(num_cells) + std::string(" cells");
  ImGui::PlotHistogram("", quality_bins.data(), num_bins, 0, label.c_str(), 0.0, num_cells, ImVec2(300, 300));

  static char output_filename[128] = "output.stl";
  ImGui::InputText("output filename", output_filename, 128);
  if (ImGui::Button("Save Mesh")) {
    export_stl(mesh, output_filename);
  }

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

MeshOpt::MeshOpt() : Application(), scene(), keys_down{} {

  camera_speed = 0.015;
  camera.lookAt(glm::vec3(1.5, 1.5, 1.5), glm::vec3(0.5f, 0.5f, 0.5f));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 0.01f, 100.0f);

  hide = false;
  quality_threshold = 1.0f;
  quality_bins.resize(num_bins);

  scene.color = gray;

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};

int main() {
  MeshOpt app;
  app.run();
  return 0;
}