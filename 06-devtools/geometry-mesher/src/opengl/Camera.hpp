#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include <algorithm>

class Camera {

  enum class ProjectionType{PERSPECTIVE, ORTHOGRAPHIC};

  public:

    static const glm::vec3 x, y, z;

    Camera();
    Camera(glm::vec3 starting_pos, glm::vec3 starting_focus);

    void perspective(float fov, float aspect, float near, float far);
    void orthographic(float width, float aspect, float near, float far);

    float fov() const;
    float aspect() const;
    float near() const;
    float far() const;

    void set_fov(float fov);
    void set_aspect(float aspect);
    void set_near_plane(float near);
    void set_far_plane(float far);
    void set_near_and_far_plane(float near, float far);

    void lookAt(const glm::vec3 & next_pos, const glm::vec3 & next_focus, const glm::vec3 & next_up = {0, 0, 1});

    const glm::vec3 & pos() const;
    const glm::vec3 & focus() const;
    const glm::vec3 & up() const;

    void set_up(const glm::vec3 & next_up);
    void set_pos(const glm::vec3 & next_pos);
    void set_focus(const glm::vec3 & next_focus);

    void offset_pos(const glm::vec3 & offset);
    void offset_focus(const glm::vec3 & offset);
    void offset_both(const glm::vec3 & offset);

    void orbit_focus(float altitude, float azimuth);
    void orbit(glm::vec3 point, float altitude, float azimuth);

    void rotate(float altitude, float azimuth);
    void move_forward(float distance);
    void move_up(float distance);
    void move_down(float distance);
    void move_left(float distance);
    void move_right(float distance);

    void zoom(float ratio);

    glm::mat4 matrix();
    glm::mat4 projection();
    glm::mat4 view();

    glm::vec3 ray_cast(glm::ivec2 window_size, double window_x, double window_y);

    glm::vec3 m_pos, m_focus, m_up;

  private:

    glm::mat4 m_projection, m_orientation, m_translation;

    float m_fov, m_near, m_far, m_aspect, m_ortho_height;

    ProjectionType m_projection_type;

};
