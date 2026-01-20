#include <random>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "json.hpp"

#include "BVH.hpp"
#include "geometry/geometry.hpp"

#include "timer.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

using namespace geometry;

struct Annulus : public Application {

  Annulus();

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
  float z;
  float cell_size;
  int ndvr;
  float dvr_step;
  float blend_distance;

  std::vector< Capsule > rods;
  BVH<3> bvh;

  void reslice();
  void update_camera_position();

};

rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (Annulus*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (Annulus*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (Annulus*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (Annulus*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

void Annulus::key_callback(GLFWwindow* window,
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

void Annulus::mouse_scroll_callback(GLFWwindow* window,
                                   double xoffset,
                                   double yoffset) {
  camera.zoom(1.0 + 0.10 * yoffset);
}

void Annulus::mouse_motion_callback(GLFWwindow* window,
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

void Annulus::mouse_button_callback(GLFWwindow* window,
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

void Annulus::update_camera_position() {
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

void Annulus::reslice() {

  AABB<3>bounds = bvh.global;

  float dx = 1.5 * cell_size;

  std::function<float(vec2f)> f = [&](vec2f x) -> float {
    AABB<3>box{
      {x[0] - dx, x[1] - dx, z}, 
      {x[0] + dx, x[1] + dx, z}
    };

    float value = 1.0e10;
    bvh.query(box, [&](int i) {
      value = std::min(rods[i].SDF({x[0], x[1], z}), value);
    });
    return value;
  };

  slice.clear();

  auto mesh = universal_mesh(f, cell_size, bounds, 0.5f, dvr_step, ndvr);

  slice.color = rgbcolor{255, 0, 0, 255};
  for (auto& tri_ids : mesh.elements) {
    Triangle tri{mesh.vertices[tri_ids[0]], 
                 mesh.vertices[tri_ids[1]], 
                 mesh.vertices[tri_ids[2]]}; 
    for (auto & v : tri.vertices) { v[2] = z; }
    slice.push_back(tri);
  }

}

void Annulus::loop() {

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

  float zmin = bvh.global.min[2];
  float zmax = bvh.global.max[2];

  bool should_reslice = false; 
  should_reslice |= ImGui::DragFloat("cell size", &cell_size, 0.0003f, 0.003f, 0.02f, "%.5f");
  should_reslice |= ImGui::DragFloat("z height", &z, 0.002f, zmin, zmax, "%.5f");
  if (should_reslice) { 
    reslice(); 
  };

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

namespace fm {
  template < typename T, uint32_t n >
  void from_json(const nlohmann::json& j, vec<n, T> & v) {
    for (int i = 0; i < n; i++) { v[i] = j[i]; }
  }
}

Annulus::Annulus() : Application(), scene(), keys_down{} {

  camera_speed = 0.015;
  camera.lookAt(glm::vec3(1.5, 1.5, 1.5), glm::vec3(0.5f, 0.5f, 0.5f));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 0.01f, 100.0f);

  // read the contents of the entire file into str
  std::string filename = "../data/annulus.json";
  std::ifstream infile(filename);
  std::string str;
  if (infile) {
    infile.seekg(0, std::ios::end);   
    str.reserve(infile.tellg());
    infile.seekg(0, std::ios::beg);
    str.assign((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
  } else {
    std::cout << "file: " << filename << " not found. exiting ... " << std::endl;
    std::exit(1);
  }

  // parse the string as json
  auto j = nlohmann::json::parse(str);

  auto elements = j["elements"].get<std::vector< std::array<int, 8> > >();
  auto vertices = j["nodes"].get<std::vector< vec<3,float> > >();
  auto diameters = j["diameters"].get< std::vector<float> >();

  static constexpr int faces[6][4] = {{0, 1, 2, 3}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};

  timer stopwatch;

  stopwatch.start();
  for (auto & element : elements) { 
    float face_diams[6];
    vec3f face_nodes[6];
    std::array< vec3f, 8 > v;
    std::array< float, 8 > d;

    auto corner_to_corner = [&](int i, int j) {
      rods.push_back(Capsule{v[i], v[j], 0.5f * d[i], 0.5f * d[j]});
    };

    auto face_to_face = [&](int i, int j) {
      rods.push_back(Capsule{face_nodes[i], face_nodes[j], 0.5f * face_diams[i], 0.5f * face_diams[j]});
    };

    float element_size = 0.3f; // TODO

    for (int i = 0; i < 8; i++) {
      v[i] = vertices[element[i]];
      d[i] = element_size * diameters[element[i]];
    }

    for (int i = 0; i < 6; i++) {
      face_diams[i] = 0.25f * (d[faces[i][0]] + d[faces[i][1]] + d[faces[i][2]] + d[faces[i][3]]);
      face_nodes[i] = 0.25f * (v[faces[i][0]] + v[faces[i][1]] + v[faces[i][2]] + v[faces[i][3]]);

      corner_to_corner(faces[i][0], faces[i][2]);
      corner_to_corner(faces[i][1], faces[i][3]);
    }

    face_to_face(0, 1); face_to_face(0, 2); face_to_face(0, 3); face_to_face(0, 4);
    face_to_face(1, 2); face_to_face(2, 3); face_to_face(3, 4); face_to_face(4, 1);
    face_to_face(1, 5); face_to_face(2, 5); face_to_face(3, 5); face_to_face(4, 5);
  }
  stopwatch.stop();
  std::cout << "creating rods: " << 1000.0 * stopwatch.elapsed() << " ms" << std::endl;

  cell_size = 0.02f;

  ndvr = 3;
  dvr_step = 0.05;

  scene.color = gray;

  auto palette = [](float r) {
    float bounds[2] = {0.07 * 0.3f, 0.15 * 0.5f};
    vec3f red{1.0f, 0.0f, 0.0f};
    vec3f gray{0.4f, 0.4f, 0.4f};
    vec3f mix = red * (r - bounds[0]) / (bounds[1] - bounds[0]) + 
                gray * (bounds[1] - r) / (bounds[1] - bounds[0]);

    return rgbcolor{uint8_t(255 * mix[0]), uint8_t(255 * mix[1]), uint8_t(255 * mix[2]), 255};
  };

  stopwatch.start();
  std::vector< AABB<3>> bounding_boxes(rods.size());
  for (int i = 0; i < rods.size(); i++) {
    scene.push_back(LineWithColors{{{rods[i].p1, palette(rods[i].r1)}, {rods[i].p2, palette(rods[i].r2)}}});
    bounding_boxes[i] = bounding_box(rods[i]);
  }
  bvh = BVH(bounding_boxes);
  stopwatch.stop();
  std::cout << "creating BVH: " << 1000.0 * stopwatch.elapsed() << " ms" << std::endl;
  //bvh.print();
  //std::exit(1);

  reslice();

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};


int main() {
  Annulus app;
  app.run();
  return 0;
}

// 2527936 intersections in 10.05 ms ~ 250M intersections/sec