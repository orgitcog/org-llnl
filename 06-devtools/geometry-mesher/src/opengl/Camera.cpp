#include "opengl/Camera.hpp"

#include <iostream>
#include <algorithm>

const glm::vec3 Camera::x = glm::vec3(1,0,0);
const glm::vec3 Camera::y = glm::vec3(0,1,0);
const glm::vec3 Camera::z = glm::vec3(0,0,1);

Camera::Camera():
  m_pos(1.0f, 1.0f, 1.0f),
  m_focus(0.0f, 0.0f, 0.0f),
  m_up(0.0f, 0.0f, 1.0f),
  m_fov(1.0f),
  m_near(0.01f),
  m_far(10000.0f),
  m_aspect(3.0f/2.0f),
  m_ortho_height(2.0f) {
  m_translation = glm::translate(glm::mat4(), -m_pos);
  m_orientation = glm::lookAt(m_pos, m_focus, m_up);
  perspective(m_fov, m_aspect, m_near, m_far);
}

Camera::Camera(glm::vec3 starting_pos, glm::vec3 starting_focus):
  m_pos(starting_pos),
  m_focus(starting_focus),
  m_up(0.0f, 0.0f, 1.0f),
  m_fov(1.0f),
  m_near(0.01f),
  m_far(100.0f),
  m_aspect(3.0f/2.0f),
  m_ortho_height(2.0f) {
  m_translation = glm::translate(glm::mat4(), -m_pos);
  m_orientation = glm::lookAt(m_pos, m_focus, m_up);
  perspective(m_fov, m_aspect, m_near, m_far);
}

const glm::vec3 & Camera::pos() const {
  return m_pos;
}

const glm::vec3 & Camera::focus() const {
  return m_focus;
}

const glm::vec3 & Camera::up() const {
  return m_up;
}

void Camera::set_up(const glm::vec3 & next_up) {
  m_up = next_up;
}

void Camera::set_pos(const glm::vec3 & next_pos){
  m_pos = next_pos;
}

void Camera::offset_pos(const glm::vec3 & offset) {
  m_pos += offset;
}

void Camera::offset_focus(const glm::vec3 & offset) {
  m_focus += offset;
}

void Camera::offset_both(const glm::vec3 & offset) {
  m_pos += offset;
  m_focus += offset;
}

float Camera::fov() const {
  return m_fov;
}

void Camera::set_fov(float fov){
  m_fov = std::max(std::min(fov, 2.8f), 0.3f);
}

float Camera::aspect() const {
  return m_aspect;
}

void Camera::set_aspect(float aspect){
  m_aspect = aspect;
}

void Camera::perspective(float fov, float aspect, float near, float far){
  m_fov = fov;
  m_aspect = aspect;
  m_near = near;
  m_far = far;
  m_projection_type = ProjectionType::PERSPECTIVE;
}

void Camera::orthographic(float height, float aspect, float near, float far){
  m_ortho_height = height;
  m_aspect = aspect;
  m_near = near;
  m_far = far;
  m_projection_type = ProjectionType::ORTHOGRAPHIC;
}

float Camera::near() const {
  return m_near;
}

void Camera::set_near_plane(float near){
  m_near = near;
}

float Camera::far() const {
  return m_far;
}

void Camera::set_far_plane(float far){
  m_far = far;
}

void Camera::lookAt(const glm::vec3 & next_pos, const glm::vec3 & next_focus, const glm::vec3 & next_up){
  m_pos = next_pos;
  m_focus = next_focus;
  m_up = next_up;
}

void Camera::orbit(glm::vec3 point, float altitude, float azimuth){
  glm::vec3 direction = m_pos - point;

  glm::vec3 altitude_axis = glm::cross(m_up, direction); glm::vec3 azimuth_axis = m_up; 
  direction = glm::rotate(direction, altitude, altitude_axis);
  direction = glm::rotate(direction, azimuth, azimuth_axis);

  m_pos = point + direction;
}

void Camera::orbit_focus(float altitude, float azimuth){
  orbit(m_focus, altitude, azimuth);
}

void Camera::rotate(float altitude, float azimuth){
  glm::vec3 direction = m_focus - m_pos;

  glm::vec3 altitude_axis = glm::cross(m_up, direction);
  glm::vec3 azimuth_axis = m_up;

  direction = glm::rotate(direction, altitude, altitude_axis);
  direction = glm::rotate(direction, azimuth, azimuth_axis);

  m_focus = m_pos + direction;
}

void Camera::move_forward(float distance){

  glm::vec3 direction = glm::normalize(m_focus - m_pos);

  m_focus += direction * distance; 
  m_pos += direction * distance; 

}

void Camera::move_up(float distance){

  m_focus += m_up * distance; 
  m_pos += m_up * distance; 

}

void Camera::move_down(float distance){

  m_focus -= m_up * distance;
  m_pos -= m_up * distance;

}

void Camera::move_left(float distance){

  glm::vec3 left = glm::normalize(glm::cross(m_up, m_focus - m_pos));
  m_focus += left * distance;
  m_pos += left * distance;

}

void Camera::move_right(float distance){

  glm::vec3 right = glm::normalize(glm::cross(m_up, m_pos - m_focus));
  m_focus += right * distance;
  m_pos += right * distance;

}

void Camera::zoom(float ratio){
  if (m_projection_type == ProjectionType::PERSPECTIVE) {
    set_fov(m_fov * ratio);
  } else {
    m_ortho_height /= ratio;
  }
}

glm::mat4 Camera::matrix(){
  m_orientation = glm::lookAt(m_pos, m_focus, m_up);
  m_projection = projection();
  return m_projection * m_orientation;
}

glm::mat4 Camera::projection(){
  if (m_projection_type == ProjectionType::PERSPECTIVE) {
    return glm::perspective(m_fov, m_aspect, m_near, m_far);
  } else {
    return glm::ortho(-0.5f * m_ortho_height * m_aspect, 
                       0.5f * m_ortho_height * m_aspect,
                      -0.5f * m_ortho_height, 
                       0.5f * m_ortho_height,
                       m_near, m_far);
  }
}

glm::mat4 Camera::view(){
  return glm::lookAt(m_pos, m_focus, m_up);
}

glm::vec3 Camera::ray_cast(glm::ivec2 window_size, double window_x, double window_y){

  float x = (2.0f * window_x) / window_size.x - 1.0f;
  float y = 1.0f - (2.0f * window_y) / window_size.y;

  glm::vec4 ray_clip = glm::vec4(x, y, -1.0, 1.0);
  glm::vec4 ray_eye = glm::inverse(projection()) * ray_clip;

  ray_eye = glm::vec4(glm::vec2(ray_eye), -1.0, 0.0);

  return glm::normalize(glm::vec3(glm::inverse(view()) * ray_eye));

}
