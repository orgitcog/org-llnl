
#include "BVH.hpp"
#include "geometry/geometry.hpp"
#include "geometry/parse_dat.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "timer.hpp"

struct UniversalMesher : public Application {

  UniversalMesher();

  void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
  void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
  void mouse_motion_callback(GLFWwindow* window, double xpos, double ypos);
  void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

  void loop();
  void initialize(int d);

 private: 
  Scene scene;
  Camera camera;

  float camera_speed;
  bool keys_down[256];
  double mouse_x, mouse_y;

  std::vector < float > quality_bins;

  bool lmb_down = false;
  bool mmb_down = false;
  bool rmb_down = false;

  // parameters for the meshing example
  int dim;
  int n;
  int ndvr;
  float r1;
  float r2;
  float separation;
  float blend_distance;
  float snap_threshold;
  float dvr_step;

  // for BROCXB example in 3D
  BVH<3> bvh;
  std::vector< Capsule > capsules; 

  std::string example;

  void remesh(std::string filename = "");
  void update_camera_position();

};

int num_bins = 16;

rgbcolor gray{40, 40, 40, 255};

// clang-format off
void key_callback_helper(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto mesher = (UniversalMesher*)glfwGetWindowUserPointer(window);
  mesher->key_callback(window, key, scancode, action, mods);
}

void mouse_scroll_callback_helper(GLFWwindow* window, double xoffset, double yoffset) {
  auto mesher = (UniversalMesher*)glfwGetWindowUserPointer(window);
  mesher->mouse_scroll_callback(window, xoffset, yoffset);
}

void mouse_motion_callback_helper(GLFWwindow* window, double xpos, double ypos) {
  auto mesher = (UniversalMesher*)glfwGetWindowUserPointer(window);
  mesher->mouse_motion_callback(window, xpos, ypos);
}

void mouse_button_callback_helper(GLFWwindow* window, int button, int action, int mods) {
  auto mesher = (UniversalMesher*)glfwGetWindowUserPointer(window);
  mesher->mouse_button_callback(window, button, action, mods);
}
// clang-format on

void UniversalMesher::key_callback(GLFWwindow* window,
                          int key,
                          int /*scancode*/,
                          int action,
                          int /*mods*/) {
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

  if (key == GLFW_KEY_2 && action & GLFW_PRESS){ example = "balls"; initialize(2); }
  if (key == GLFW_KEY_3 && action & GLFW_PRESS){ example = "balls"; initialize(3); }
  // clang-format on
};

void UniversalMesher::mouse_scroll_callback([[maybe_unused]] GLFWwindow* window,
                                            [[maybe_unused]] double xoffset,
                                            double yoffset) {
  if (dim == 2) {
    camera.zoom(1.0 + 0.10 * yoffset);
  } else {
    camera.zoom(1.0 + 0.01 * yoffset);
  }
}

void UniversalMesher::mouse_motion_callback([[maybe_unused]] GLFWwindow* window,
                                   double xpos,
                                   double ypos) {
  if (lmb_down && !mmb_down && !rmb_down) {
    float altitude = (ypos - mouse_y) * 0.01f;
    float azimuth = (xpos - mouse_x) * 0.01f;

    if (ImGui::GetIO().WantCaptureMouse) {
      // if the mouse is interacting with ImGui
    } else {
      if (dim == 3) {
        camera.rotate(altitude, -azimuth);
      }
    }

    mouse_x = xpos;
    mouse_y = ypos;
  }

  if (!lmb_down && !mmb_down && rmb_down) {
    // right click
  }
}

void UniversalMesher::mouse_button_callback(GLFWwindow* window,
                                   int button,
                                   int action,
                                   [[maybe_unused]] int mods) {
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

void UniversalMesher::update_camera_position() {
  // clang-format off
  float scale = 1.0f;
  if (keys_down[uint8_t(' ')]) { scale = 0.1f; }
  if (keys_down[uint8_t('w')]) { 
    if (dim == 2) camera.move_up(scale * camera_speed);
    if (dim == 3) camera.move_forward(scale * camera_speed);
  }
  if (keys_down[uint8_t('s')]) {
    if (dim == 2) camera.move_up(-scale * camera_speed);
    if (dim == 3) camera.move_forward(-scale * camera_speed);
  }
  if (keys_down[uint8_t('a')]) { camera.move_left(scale * camera_speed); }
  if (keys_down[uint8_t('d')]) { camera.move_right(scale * camera_speed); }
  if (keys_down[uint8_t('q')]) { camera.move_down(scale * camera_speed); }
  if (keys_down[uint8_t('e')]) { camera.move_up(scale * camera_speed); }
  // clang-format on
}

void UniversalMesher::initialize(int d) {

  dim = d;
  camera_speed = 0.015;

  n = 32;
  r1 = 0.25;
  r2 = 0.5;
  separation = 1.0;
  blend_distance = 0.003;
  snap_threshold = 0.0;

  ndvr = 0;
  dvr_step = 0.0;

  if (dim == 3) {
    if (example == "balls") {
      camera.lookAt(glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
      camera.perspective(1.0f /* fov */, getWindowRatio(), 0.01f, 100.0f);
    }

    if (example == "BROCXB") {
      auto [min, max] = bvh.global;
      auto center = (min + max) * 0.5f;
      camera.lookAt(glm::vec3(max[0], max[1], max[2]), glm::vec3(center[0], center[1], center[2]));
      camera.perspective(1.0f /* fov */, getWindowRatio(), 0.01f, 1000.0f);
      //camera.orthographic(100.0f, getWindowRatio(), 0.01f, 1000.0f);

      n = 128;
      blend_distance = 0.003;
      snap_threshold = 0.5;

      ndvr = 3;
      dvr_step = 0.05;
    }
  } else {
    camera.lookAt(glm::vec3(0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    camera.orthographic(2.0f, getWindowRatio(), 0.1f, 10.0f);
  }

  scene.color = gray;

  remesh();

}

void UniversalMesher::remesh(std::string output_filename) {

  timer stopwatch;

  if (dim == 2) {
    AABB<3>bounds{{-1.0, -1.0, 0.0}, {1.0, 1.0, 0.0}};

    std::vector<Ball> primitives = {
        Ball{{-0.5f * separation, 0.0, 0.0}, r1},
        Ball{{0.5f * separation, 0.0, 0.0}, r2},
    };

    std::function<float(vec2f)> f = [=](vec2f p) -> float {
      float v = 0.0;
      for (auto& primitive : primitives) {
        v += exp(-primitive.SDF(vec3f{p[0], p[1], 0.0f}) / blend_distance);
      }
      return -blend_distance * log(fmax(v, 1.0e-6));
    };

    scene.clear();

    stopwatch.start();
    auto mesh = universal_mesh(f, 2.0 / n, bounds, snap_threshold, dvr_step, ndvr);
    stopwatch.stop();
    std::cout << "meshed " << mesh.elements.size() << " triangles and ";
    std::cout << mesh.boundary_elements.size() << " edges in " << stopwatch.elapsed() * 1000.0f << "ms " << std::endl;


    //std::cout << "verifying ... " << (verify(mesh) ? " pass " : " fail ") << std::endl;

    vec3f red = {1.0f, 0.0f, 0.0f};
    vec3f gray = {0.25f, 0.25f, 0.25f};

    quality_bins = std::vector<float>(num_bins, 0);
    for (const auto& tri_ids : mesh.elements) {
      Triangle tri = {mesh.vertices[tri_ids[0]],
                      mesh.vertices[tri_ids[1]],
                      mesh.vertices[tri_ids[2]]};
      float q = quality(tri);
      int id = std::max(0.0f, std::min(q * num_bins, num_bins - 1.0f));
      quality_bins[id]++;

      float t = powf(std::max(q, 0.0f), 0.3f);
      vec3f rgb = red * (1 - t) + gray * t;
      scene.color = rgbcolor{uint8_t(255 * rgb[0]), uint8_t(255 * rgb[1]),
                             uint8_t(255 * rgb[2]), 255};
      scene.push_back(tri);
    }
  }

  if (dim == 3) {

    if (example == "balls") {

      AABB<3>bounds{{-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

      std::vector<Ball> primitives = {
          Ball{{-0.5f * separation, 0.0, 0.0}, r1},
          Ball{{0.5f * separation, 0.0, 0.0}, r2},
      };

      std::function<float(vec3f)> f = [=](vec3f p) -> float {
        float v = 0.0;
        for (auto& primitive : primitives) {
          v += exp(-primitive.SDF(p) / blend_distance);
        }
        return -blend_distance * log(fmax(v, 1.0e-6));
      };

      scene.clear();

      stopwatch.start();
      auto mesh =
          universal_mesh(f, 2.0 / n, bounds, snap_threshold, dvr_step, ndvr);
      stopwatch.stop();
      std::cout << "meshed " << mesh.elements.size() << " tets and ";
      std::cout << mesh.boundary_elements.size() << " triangles in " << stopwatch.elapsed() * 1000.0f << "ms " << std::endl;

      if (!output_filename.empty()) {
        std::cout << "writing out file to " << output_filename << std::endl;
        export_mesh(mesh, output_filename);
      }

      //std::cout << "verifying ... " << (verify(mesh) ? " pass " : " fail ") << std::endl;

      vec3f red = {1.0f, 0.0f, 0.0f};
      vec3f gray = {0.25f, 0.25f, 0.25f};

      quality_bins = std::vector<float>(num_bins, 0);

      for (const auto& [i,j,k] : mesh.boundary_elements) {
        scene.color = rgbcolor{128, 128, 128, 255};
        if (i < j) { scene.push_back(Line{mesh.vertices[i], mesh.vertices[j]}); }
        if (j < k) { scene.push_back(Line{mesh.vertices[j], mesh.vertices[k]}); }
        if (k < i) { scene.push_back(Line{mesh.vertices[k], mesh.vertices[i]}); }

        scene.color = rgbcolor{32, 32, 32, 255};
        scene.push_back(Triangle{mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]});
      }

      for (const auto& tet_ids : mesh.elements) {
        Tetrahedron tet = {mesh.vertices[tet_ids[0]],
                           mesh.vertices[tet_ids[1]],
                           mesh.vertices[tet_ids[2]],
                           mesh.vertices[tet_ids[3]]};

        float q = quality(tet);
        int id = std::max(0.0f, std::min(q * num_bins, num_bins - 1.0f));
        quality_bins[id]++;

        if (q < 0.1f) {
          float t = powf(std::max(q, 0.0f), 0.3f);
          vec3f rgb = red * (1 - t) + gray * t;
          scene.color = rgbcolor{uint8_t(255 * rgb[0]), uint8_t(255 * rgb[1]), uint8_t(255 * rgb[2]), 255};
          scene.push_back(Triangle{tet.vertices[2], tet.vertices[1], tet.vertices[0]});
          scene.push_back(Triangle{tet.vertices[3], tet.vertices[0], tet.vertices[1]});
          scene.push_back(Triangle{tet.vertices[3], tet.vertices[1], tet.vertices[2]});
          scene.push_back(Triangle{tet.vertices[3], tet.vertices[2], tet.vertices[0]});
        } 
      }

    } else {

      vec3f widths = bvh.global.max - bvh.global.min;
      float cell_size = std::max(std::max(widths[0], widths[1]), widths[2]) / n;

      float dx = 1.5 * cell_size + 2 * blend_distance;

      std::function<float(vec3f)> f = [&](vec3f x) -> float {
        AABB<3>box{
          {x[0] - dx, x[1] - dx, x[2] - dx}, 
          {x[0] + dx, x[1] + dx, x[2] + dx}
        };

        double value = 0.0;
        bvh.query(box, [&](int i) {
          value += exp(-capsules[i].SDF(x) / blend_distance);
        });
        return -blend_distance * log(value);
      };

      scene.clear();

      stopwatch.start();
      auto mesh = universal_mesh(f, cell_size, bvh.global, snap_threshold, dvr_step, ndvr);
      stopwatch.stop();
      std::cout << "meshed " << mesh.elements.size() << " tets and ";
      std::cout << mesh.boundary_elements.size() << " triangles in " << stopwatch.elapsed() * 1000.0f << "ms " << std::endl;

      //std::cout << "verifying ... " << (verify(mesh) ? " pass " : " fail ") << std::endl;

      vec3f red = {1.0f, 0.0f, 0.0f};
      vec3f gray = {0.25f, 0.25f, 0.25f};

      for (const auto& [i,j,k] : mesh.boundary_elements) {
        scene.color = rgbcolor{128, 128, 128, 255};
        if (i < j) { scene.push_back(Line{mesh.vertices[i], mesh.vertices[j]}); }
        if (j < k) { scene.push_back(Line{mesh.vertices[j], mesh.vertices[k]}); }
        if (k < i) { scene.push_back(Line{mesh.vertices[k], mesh.vertices[i]}); }

        scene.color = rgbcolor{64, 64, 64, 255};
        scene.push_back(Triangle{mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]});
      }

      int bad = 0;
      quality_bins = std::vector<float>(num_bins, 0);
      for (const auto& tet_ids : mesh.elements) {
        Tetrahedron tet = {mesh.vertices[tet_ids[0]],
                           mesh.vertices[tet_ids[1]],
                           mesh.vertices[tet_ids[2]],
                           mesh.vertices[tet_ids[3]]};
        float q = quality(tet);
        int id = std::max(0.0f, std::min(q * num_bins, num_bins - 1.0f));
        quality_bins[id]++;

        if (q < 0.1f) {
          float t = powf(std::max(q, 0.0f), 0.3f);
          vec3f rgb = red * (1 - t) + gray * t;
          scene.color = rgbcolor{uint8_t(255 * rgb[0]), uint8_t(255 * rgb[1]), uint8_t(255 * rgb[2]), 255};
          scene.push_back(Triangle{tet.vertices[2], tet.vertices[1], tet.vertices[0]});
          scene.push_back(Triangle{tet.vertices[3], tet.vertices[0], tet.vertices[1]});
          scene.push_back(Triangle{tet.vertices[3], tet.vertices[1], tet.vertices[2]});
          scene.push_back(Triangle{tet.vertices[3], tet.vertices[2], tet.vertices[0]});
          bad++;
        }

      }

      if (bad) {
        std::cout << bad << " bad quality tets out of " << mesh.elements.size() << std::endl;
      }

    }

  }
}

void UniversalMesher::loop() {

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

  bool should_remesh = false; 

  if (example == "balls") {
    should_remesh |= ImGui::DragInt("n", &n, 0.5f, 8, 128);
  } else {
    should_remesh |= ImGui::DragInt("n", &n, 0.5f, 8, 512);
  }
  if (example == "balls") {
    should_remesh |= ImGui::DragFloat("r1", &r1, 0.01f, -0.25f, 2.0f, "%.4f");
    should_remesh |= ImGui::DragFloat("r2", &r2, 0.01f, -0.25f, 2.0f, "%.4f");
    should_remesh |= ImGui::DragFloat("separation", &separation, 0.01f, -2.0f, 2.0f, "%.4f");
  }

  should_remesh |= ImGui::DragFloat("blending distance", &blend_distance, 0.0003f, 0.003f, 0.5f, "%.5f");
  should_remesh |= ImGui::DragFloat("snap threshold", &snap_threshold, 0.001f, 0.0f, 0.5f, "%.4f");
  should_remesh |= ImGui::DragInt("dvr iterations", &ndvr, 0.1f, 0, 8);
  should_remesh |= ImGui::DragFloat("dvr step", &dvr_step, 0.001f, 0.0f, 0.08f, "%.4f");

  //if (dim == 2) {
  //  auto num_tris = scene.tris.data.size();
  //  auto label = std::string("Element Quality\n") + std::to_string(num_tris) + std::string(" tris");
  //  ImGui::PlotHistogram("", quality_bins.data(), num_bins, 0, label.c_str(), 0.0, num_tris, ImVec2(400, 300));
  //}
  //if (dim == 3) {
  //  auto num_tets = scene.tris.data.size() / 4;
  //  auto label = std::string("Element Quality\n") + std::to_string(num_tets) + std::string(" tets");
  //  ImGui::PlotHistogram("", quality_bins.data(), num_bins, 0, label.c_str(), 0.0, num_tets, ImVec2(400, 300));
  //}

  std::string output_filename;
  static char buf[128] = "mesh_";
  if (ImGui::InputText("output filename", buf, 128)) {

  }

  if (ImGui::Button("export mesh")) {
    output_filename = buf;
    should_remesh = true;
  }

  if (should_remesh) { remesh(output_filename); };

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

UniversalMesher::UniversalMesher() : Application(), scene(), keys_down{} {
  std::string filenames[] = {
    "/home/sam/data/digital_twins/BROCXB/Haleakala-na-20_25x25_SC_velocity_OFF_speed_29.75_Zlayerheight_0.475-BROCXB_1__174648__.dat",
    "/home/sam/data/digital_twins/BROCXB/Haleakala-na-20_25x25_SC_velocity_OFF_speed_29.75_Zlayerheight_0.475-BROCXB_2__223315__.dat",
    "/home/sam/data/digital_twins/BROCXB/Haleakala-na-20_25x25_SC_velocity_OFF_speed_29.75_Zlayerheight_0.475-BROCXB_3__032036__.dat",
    "/home/sam/data/digital_twins/BROCXB/Haleakala-na-20_25x25_SC_velocity_OFF_speed_29.75_Zlayerheight_0.475-BROCXB_4__080658__.dat",
    "/home/sam/data/digital_twins/BROCXB/Haleakala-na-20_25x25_SC_velocity_OFF_speed_29.75_Zlayerheight_0.475-BROCXB_5__125424__.dat"
  }; 

  //float filament_radius = 0.25f;
  //std::array<int,3> position_columns = {1, 9, 17};
  //capsules = combine<Capsule>({
  //  parse_datfile("../data/BROCXB1.dat", position_columns, filament_radius),
  //  parse_datfile("../data/BROCXB2.dat", position_columns, filament_radius),
  //  parse_datfile("../data/BROCXB3.dat", position_columns, filament_radius),
  //  parse_datfile("../data/BROCXB4.dat", position_columns, filament_radius),
  //  parse_datfile("../data/BROCXB5.dat", position_columns, filament_radius)
  //});

  //std::vector< AABB<3>> bounding_boxes(capsules.size());
  //for (uint32_t i = 0; i < capsules.size(); i++) {
  //  bounding_boxes[i] = bounding_box(capsules[i]);
  //}
  //bvh = BVH(bounding_boxes);

  example = "balls";

  // start with the 2D demo
  initialize(2);

  glfwSetWindowUserPointer(window, (void*)this);
  glfwSetKeyCallback(window, key_callback_helper);
  glfwSetScrollCallback(window, mouse_scroll_callback_helper);
  glfwSetCursorPosCallback(window, mouse_motion_callback_helper);
  glfwSetMouseButtonCallback(window, mouse_button_callback_helper);

};

int main() {
  UniversalMesher app;
  app.run();
  return 0;
}