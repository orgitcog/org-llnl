#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <utility>

#include <unistd.h>

#include "binary_io.hpp"

#include "opengl/Scene.hpp"
#include "opengl/Shader.hpp"
#include "opengl/Camera.hpp"
#include "opengl/Application.hpp"

#define PIXELS_PER_UNIT 600.0f

glm::ivec2 window_size(900, 900);

Camera gCamera;

GLFWwindow * window;

std::unique_ptr < Scene > scene;

float speed = 0.003;

bool keys_down[256];
bool wireframe = false;

bool lmb_down = false;
bool mmb_down = false;
bool rmb_down = false;

double mouse_x, mouse_y;

glm::vec3 grayscale[] = {
  glm::vec3(240.0f / 255.0f, 240.0 / 255.0f, 240.0f / 255.0f),
  glm::vec3(160.0f / 255.0f, 160.0 / 255.0f, 160.0f / 255.0f),
  glm::vec3( 80.0f / 255.0f,  80.0 / 255.0f,  80.0f / 255.0f),
  glm::vec3(  0.0f / 255.0f,   0.0 / 255.0f,   0.0f / 255.0f)
};

glm::vec3 light_gray[] = {
  glm::vec3(120.0f / 255.0f, 120.0 / 255.0f, 120.0f / 255.0f),
  glm::vec3(120.0f / 255.0f, 120.0 / 255.0f, 120.0f / 255.0f),
  glm::vec3(120.0f / 255.0f, 120.0 / 255.0f, 120.0f / 255.0f),
  glm::vec3(120.0f / 255.0f, 120.0 / 255.0f, 120.0f / 255.0f)
};

glm::vec3 dark_gray[] = {
  glm::vec3( 40.0f / 255.0f,  40.0 / 255.0f,  40.0f / 255.0f),
  glm::vec3( 40.0f / 255.0f,  40.0 / 255.0f,  40.0f / 255.0f),
  glm::vec3( 40.0f / 255.0f,  40.0 / 255.0f,  40.0f / 255.0f),
  glm::vec3( 40.0f / 255.0f,  40.0 / 255.0f,  40.0f / 255.0f)
};

void render();

static void key_callback(GLFWwindow* window,
                         int key,
                         int scancode,
                         int action,
                         int mods){

  if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);

  // clang-format off
  if (key == GLFW_KEY_W){ keys_down[uint8_t('w')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_A){ keys_down[uint8_t('a')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_S){ keys_down[uint8_t('s')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_D){ keys_down[uint8_t('d')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_Q){ keys_down[uint8_t('q')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  if (key == GLFW_KEY_E){ keys_down[uint8_t('e')] = (action & (GLFW_PRESS | GLFW_REPEAT)); }
  // clang-format on

}

void update_position(){
  // clang-format off
  if (keys_down[uint8_t('w')]) { gCamera.move_forward(speed); }
  if (keys_down[uint8_t('s')]) { gCamera.move_forward(-speed); }
  if (keys_down[uint8_t('a')]) { gCamera.move_left(speed); }
  if (keys_down[uint8_t('d')]) { gCamera.move_right(speed); }
  if (keys_down[uint8_t('q')]) { gCamera.move_down(speed); }
  if (keys_down[uint8_t('e')]) { gCamera.move_up(speed); }
  // clang-format on
}

static void mouse_button_callback(GLFWwindow* window,
                                  int button,
                                  int action,
                                  int mods){

  if(button == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS){
    lmb_down = true;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);
  }

  if(button == GLFW_MOUSE_BUTTON_2 && action == GLFW_PRESS){
    rmb_down = true;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);
  }

  if(button == GLFW_MOUSE_BUTTON_1 && action == GLFW_RELEASE){
    lmb_down = false;
  }

  if(button == GLFW_MOUSE_BUTTON_2 && action == GLFW_RELEASE){
    rmb_down = false;
  }

}

static void mouse_motion_callback(GLFWwindow* window, double xpos, double ypos){

  if(lmb_down && !mmb_down && !rmb_down){

    float altitude = (ypos - mouse_y) * 0.01f;
    float azimuth = (xpos - mouse_x) * 0.01f;

    gCamera.rotate(altitude, -azimuth);

    mouse_x = xpos;
    mouse_y = ypos;

  }

  if(!lmb_down && !mmb_down && rmb_down){

  }

}

static void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
  gCamera.zoom(1.0 + 0.01 * yoffset);
}

void init(){

  // set up window
  if (!glfwInit())
    exit(EXIT_FAILURE);

  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  window = glfwCreateWindow(window_size.x, window_size.y, "mesh viewer", NULL, NULL);

  if (!window){
    std::cout << "no window!" << std::endl;
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, key_callback);
  glfwSetScrollCallback(window, mouse_scroll_callback);
  glfwSetCursorPosCallback(window, mouse_motion_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);

  // initialize glew
  glewExperimental = GL_TRUE;
  glewInit();

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS); 
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  //glEnable(GL_CULL_FACE);
  //glCullFace(GL_BACK);

  gCamera.lookAt(glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
  gCamera.set_aspect(float(window_size.x) / float(window_size.y));
  gCamera.set_near_plane(0.01);

}

void render(){

  glClearColor(0.169f, 0.314f, 0.475f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  scene->draw(gCamera.matrix());
  
  glfwSwapBuffers(window);
  
}

int main(int argc, char* argv[]){

  if (argc != 2) {
    std::cout << "Need to specify an input file" << std::endl;
    exit(1);
  }

  init();

  scene = std::make_unique< Scene >();
  scene->push_back(read_binary<Triangle>(argv[1]));

  while (!glfwWindowShouldClose(window)){

    glfwGetFramebufferSize(window, &window_size.x, &window_size.y);
    glViewport(0, 0, window_size.x, window_size.y);

    render();
    update_position();

    glfwPollEvents();

  }

  glfwDestroyWindow(window);
  glfwTerminate();

}
