#include <random>

#include "BVH.hpp"
#include "geometry/geometry.hpp"
#include "geometry/parse_dat.hpp"
#include "geometry/image.hpp"

#include "timer.hpp"
#include "binary_io.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

struct CTScan : public Application {

  CTScan();

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

  float threshold;
  float snap_threshold;
  int ndvr;
  float dvr_step;

  float camera_speed;
  bool keys_down[256];
  double mouse_x, mouse_y;

  bool lmb_down = false;
  bool mmb_down = false;
  bool rmb_down = false;

  std::vector< float > qualities;
  std::vector< std::string > filenames;

  void remesh();
  void update_camera_position();

};

rgbcolor red{255, 40, 40, 255};
rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (CTScan*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (CTScan*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (CTScan*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (CTScan*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on



void CTScan::key_callback(GLFWwindow* window,
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

void CTScan::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void CTScan::mouse_motion_callback(GLFWwindow* window,
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

void CTScan::mouse_button_callback(GLFWwindow* window,
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

void CTScan::update_camera_position() {
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

void CTScan::remesh() {

  scene.clear();

  auto mesh = universal_mesh(filenames, threshold, snap_threshold, dvr_step, ndvr);

  std::cout << mesh.vertices.size() << " " << mesh.elements.size() << " " << mesh.boundary_elements.size() << std::endl;


  scene.color = gray;

  //auto & v = mesh.vertices;
  //for (const auto& tri_ids : mesh.boundary_elements) {
  //  Triangle tri{v[tri_ids[0]], v[tri_ids[1]], v[tri_ids[2]]};
  //  scene.push_back(tri);
  //}

  qualities.clear();

  for (const auto& [i,j,k] : mesh.boundary_elements) {
    scene.color = rgbcolor{128, 128, 128, 255};
    if (i < j) { scene.push_back(Line{mesh.vertices[i], mesh.vertices[j]}); }
    if (j < k) { scene.push_back(Line{mesh.vertices[j], mesh.vertices[k]}); }
    if (k < i) { scene.push_back(Line{mesh.vertices[k], mesh.vertices[i]}); }
  }

  for (const auto& tet_ids : mesh.elements) {
    Tetrahedron tet = {mesh.vertices[tet_ids[0]],
                       mesh.vertices[tet_ids[1]],
                       mesh.vertices[tet_ids[2]],
                       mesh.vertices[tet_ids[3]]};

    float q = quality(tet);
    qualities.push_back(q);

    vec3f red = {1.0f, 0.0f, 0.0f};
    vec3f gray = {0.25f, 0.25f, 0.25f};

    if (q < 0.2f) {
      float t = powf(std::max(q, 0.0f), 0.3f);
      vec3f rgb = red * (1 - t) + gray * t;
      scene.color = rgbcolor{uint8_t(255 * rgb[0]), uint8_t(255 * rgb[1]), uint8_t(255 * rgb[2]), 255};
      scene.push_back(Triangle{tet.vertices[2], tet.vertices[1], tet.vertices[0]});
      scene.push_back(Triangle{tet.vertices[3], tet.vertices[0], tet.vertices[1]});
      scene.push_back(Triangle{tet.vertices[3], tet.vertices[1], tet.vertices[2]});
      scene.push_back(Triangle{tet.vertices[3], tet.vertices[2], tet.vertices[0]});
    }

    //scene.push_back(Triangle{tet.vertices[2], tet.vertices[1], tet.vertices[0]});
    //scene.push_back(Triangle{tet.vertices[3], tet.vertices[0], tet.vertices[1]});
    //scene.push_back(Triangle{tet.vertices[3], tet.vertices[1], tet.vertices[2]});
    //scene.push_back(Triangle{tet.vertices[3], tet.vertices[2], tet.vertices[0]});
  }

  std::sort(qualities.begin(), qualities.end());

  //export_stl(mesh, "mesh_" + std::to_string(k) + "_" + std::to_string(i_max) + ".stl");

}

void CTScan::loop() {

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

  ImGui::DragFloat("snap threshold", &snap_threshold, 0.001f, 0.0f, 0.5f, "%.4f");
  ImGui::DragInt("dvr iterations", &ndvr, 0.1f, 0, 8);
  ImGui::DragFloat("dvr step", &dvr_step, 0.001f, 0.0f, 0.08f, "%.4f");

  if (ImGui::Button("remesh")) {
    remesh();
  }

  ImGui::PlotLines("qualities", qualities.data(), std::min(qualities.size(), std::size_t(250)), 0, 0, 0.0f, 1.0f, ImVec2(0, 250.0f));

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

std::string pad_left(std::string str, size_t width) {
  return std::string(width - std::min(width, str.length()), '0') + str; 
}

CTScan::CTScan() : Application(), scene(), keys_down{} {

  ndvr = 0;
  dvr_step = 0.0f;
  snap_threshold = 0.0f;

  scene.color = gray;

  //std::string prefix = "/Users/mish2/data/digital_twins/ct_scans/TSN2_bottom_XY_slices_";
  std::string prefix = "/Users/mish2/data/digital_twins/cropped_ct_scans/TSN2_bottom_XY_slices_";
  //std::string prefix = "/Users/mish2/data/digital_twins/artificial_ct_scans/sphere/image_";
  //std::string prefix = "/Users/mish2/data/digital_twins/artificial_ct_scans/full/image_";
  std::string suffix = ".tif";

  float min_value = 1.0f;
  float max_value = 0.0f;

  timer stopwatch;
  std::cout << "calculating min/max values" << std::endl;
  stopwatch.start();
  for (int i = 0; i < 104; i++) {
  //for (int i = 0; i < 13; i++) {
    filenames.push_back(prefix + pad_left(std::to_string(i), 3) + suffix);

    Image im = import_tiff(filenames.back());
    for (uint32_t r = 0; r < im.height; r++) {
      for (uint32_t c = 0; c < im.width; c++) {
        float value = im(r, c);
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
      }
    }
  }
  stopwatch.stop();
  std::cout << " done after " << 1000.0 * stopwatch.elapsed() << " ms" << std::endl;
  std::cout << "min: " << min_value << std::endl;
  std::cout << "max: " << max_value << std::endl;
  threshold = 0.5f * (min_value + max_value);

  vec3f center = {800.0f, 800.0f, 15.0f};
  vec3f pov = {1200.0f, 1200.0f, 100.0f};
  camera.lookAt(glm::vec3(pov[0], pov[1], pov[2]), glm::vec3(center[0], center[1], center[2]));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 1.0f, 3000.0f);

  camera_speed = 10.0f;

  remesh();

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};


int main() {
  CTScan app;
  app.run();
  return 0;
}