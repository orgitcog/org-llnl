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

struct UniversalMeshing : public Application {

  UniversalMeshing();

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
  int n;
  float r;
  bool discrete;

  vec3f omega;
  vec3f phi;

  vec3f lissajous(float t) {
    return vec3f {
      sinf(6.2831853071 * (omega[0] * t + phi[0])),
      sinf(6.2831853071 * (omega[1] * t + phi[1])),
      sinf(6.2831853071 * (omega[2] * t + phi[2]))
    };
  }

  void remesh();
  void update_camera_position();

};

rgbcolor red{255, 40, 40, 255};
rgbcolor blue{40, 40, 255, 255};
rgbcolor gray{40, 40, 40, 255};

rgbcolor continuous_palette(float t) {
  uint8_t r = 255 * t;
  uint8_t g = 0;
  uint8_t b = 255 * (1.0 - t);
  uint8_t a = 255;
  return rgbcolor{r, g, b, a};
}

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
  if (key == GLFW_KEY_Q){ keys_down[uint8_t('q')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_E){ keys_down[uint8_t('e')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
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
  if (keys_down[uint8_t('w')]) { camera.move_forward(scale * camera_speed); }
  if (keys_down[uint8_t('s')]) { camera.move_forward(-scale * camera_speed); }
  if (keys_down[uint8_t('a')]) { camera.move_left(scale * camera_speed); }
  if (keys_down[uint8_t('d')]) { camera.move_right(scale * camera_speed); }
  if (keys_down[uint8_t('q')]) { camera.move_down(scale * camera_speed); }
  if (keys_down[uint8_t('e')]) { camera.move_up(scale * camera_speed); }
  // clang-format on
}

void UniversalMeshing::remesh() {

  AABB<3>bounds{{-1.2, -1.2, -1.2}, {1.2, 1.2, 1.2}};
  float cell_size = 2.0 / n;

  int num_capsules = 1000;
  float dt = 1.0 / num_capsules;
  std::vector< Capsule > capsules(num_capsules);
  std::vector< int > capsule_attributes(num_capsules);
  std::vector< vec2f > capsule_values(num_capsules);

  vec3f p = lissajous(0.0);
  for (int i = 0; i < num_capsules; i++) {
    vec3f q = lissajous((i+1) * dt);
    capsules[i] = {p, q, r, r};
    capsule_attributes[i] = ((i / 64) % 2) == 0;
    capsule_values[i] = {i*dt, (i+1)*dt};
    p = q;
  }

  std::vector< AABB<3> > bounding_boxes(capsules.size());
  for (uint32_t i = 0; i < capsules.size(); i++) {
    bounding_boxes[i] = bounding_box(capsules[i]);
  }
  BVH<3> bvh(bounding_boxes);

  float dx = 1.5 * cell_size;
  std::function<float(vec3f)> sdf = [&](vec3f x) -> float {

    AABB<3>box{
      {x[0] - dx, x[1] - dx, x[2] - dx}, 
      {x[0] + dx, x[1] + dx, x[2] + dx}
    };

    float value = 2 * dx;
    bvh.query(box, [&](int i) {
      value = std::min(value, capsules[i].SDF(x));
    });
    return value;
  };

  auto mesh = universal_mesh(sdf, cell_size, bounds, 0.5, 0.05f, 5);

  auto & v = mesh.vertices;

  scene.clear();
  
  scene.color = gray;

  if (discrete) {
    auto cell_attributes = cell_values(mesh, capsules, capsule_attributes, cell_size);
    for (int i = 0; i < mesh.elements.size(); i++) {
      auto tet_ids = mesh.elements[i];
      Tetrahedron tet{v[tet_ids[0]], v[tet_ids[1]], v[tet_ids[2]], v[tet_ids[3]]};
      scene.color = (cell_attributes[i] == 0) ? red : blue;
      scene.push_back(Triangle{tet.vertices[2], tet.vertices[1], tet.vertices[0]});
      scene.push_back(Triangle{tet.vertices[3], tet.vertices[0], tet.vertices[1]});
      scene.push_back(Triangle{tet.vertices[3], tet.vertices[1], tet.vertices[2]});
      scene.push_back(Triangle{tet.vertices[3], tet.vertices[2], tet.vertices[0]});
    }
  } else {
    auto values = vertex_values(mesh, capsules, capsule_values, cell_size);
    for (int i = 0; i < mesh.boundary_elements.size(); i++) {
      auto tri_ids = mesh.boundary_elements[i];

      TriangleWithColors tri{{
        {v[tri_ids[0]], continuous_palette(values[tri_ids[0]])}, 
        {v[tri_ids[1]], continuous_palette(values[tri_ids[1]])}, 
        {v[tri_ids[2]], continuous_palette(values[tri_ids[2]])}
      }};

      scene.push_back(tri);
    }
  }

}

void UniversalMeshing::loop() {

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

  should_remesh |= ImGui::Checkbox("discrete", &discrete);

  should_remesh |= ImGui::DragInt("n", &n, 0.5f, 8, 128);
  should_remesh |= ImGui::DragFloat("radius", &r, 0.005f, 0.02f, 0.2f, "%.5f");
  should_remesh |= ImGui::DragFloat("omega_x", &omega[0], 0.05f, 0.50f, 7.0f, "%.5f");
  should_remesh |= ImGui::DragFloat("omega_y", &omega[1], 0.05f, 0.50f, 7.0f, "%.5f");
  should_remesh |= ImGui::DragFloat("omega_z", &omega[2], 0.05f, 0.50f, 7.0f, "%.5f");

  should_remesh |= ImGui::DragFloat("phi_x", &phi[0], 0.005f, 0.02f, 1.0f, "%.5f");
  should_remesh |= ImGui::DragFloat("phi_y", &phi[1], 0.005f, 0.02f, 1.0f, "%.5f");
  should_remesh |= ImGui::DragFloat("phi_z", &phi[2], 0.005f, 0.02f, 1.0f, "%.5f");
  if (should_remesh) { remesh(); };

  ImGui::End();

  // Render dear imgui into screen
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

UniversalMeshing::UniversalMeshing() : Application(), scene(), keys_down{} {

  discrete = false;

  n = 40;
  r = 0.1;
  omega = {2, 3, 5};
  phi = {0.1, 0.5, 0.8};

  camera_speed = 0.015;
  camera.lookAt(glm::vec3(1, 1, 1), glm::vec3(0, 0, 0), glm::vec3(0, 0, 1));
  camera.perspective(1.0f /* fov */, getWindowRatio(), 0.01f, 100.0f);

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