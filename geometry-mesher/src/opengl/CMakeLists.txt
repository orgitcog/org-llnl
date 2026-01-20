add_library(opengl_tools STATIC
  Camera.hpp
  Camera.cpp
  Application.hpp
  Application.cpp
  glError.hpp
  glError.cpp
  Shader.hpp
  Shader.cpp
  Scene.hpp
  Scene.cpp
)

target_include_directories(opengl_tools PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(opengl_tools PUBLIC ${PROJECT_SOURCE_DIR}/src/opengl/imgui)

add_definitions(-DGLEW_STATIC)

FetchContent_Declare(
  TPL_glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 0.9.9.8
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
message("resolving dependencies: glm")
FetchContent_MakeAvailable(TPL_glm)

FetchContent_Declare(
  TPL_GLEW
  GIT_REPOSITORY https://github.com/Perlmint/glew-cmake.git
  GIT_TAG glew-cmake-2.2.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
message("resolving dependencies: glew")
FetchContent_MakeAvailable(TPL_GLEW)

FetchContent_Declare(
  TPL_glfw
  GIT_REPOSITORY https://github.com/glfw/glfw
  GIT_TAG 3.3.8
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
message("resolving dependencies: glfw")
FetchContent_MakeAvailable(TPL_glfw)

add_library(imgui STATIC 
  imgui/imconfig.h
  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_internal.h
  imgui/imgui_widgets.cpp
  imgui/imgui.h
  imgui/imgui.cpp
  imgui/imstb_rectpack.h
  imgui/imstb_textedit.h
  imgui/imstb_truetype.h
  imgui/imgui_impl_glfw.h
  imgui/imgui_impl_glfw.cpp
  imgui/imgui_impl_opengl3.h
  imgui/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC .)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)
target_link_libraries(imgui PUBLIC glfw libglew_static)

target_link_libraries(opengl_tools PUBLIC glfw glm libglew_static imgui geometry)

if (BUILD_EXAMPLES)

file(GLOB cpp_tests ${PROJECT_SOURCE_DIR}/src/opengl/examples/*.cpp)

if (NOT TIFF_FOUND)
  message("not using libtiff, skipping ct_scan example")
  list(REMOVE_ITEM cpp_tests ${PROJECT_SOURCE_DIR}/src/opengl/examples/ct_scan.cpp)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}) 

message("${testname}")

foreach(filename ${cpp_tests})		

  get_filename_component(testname ${filename} NAME_WE)

  message("${testname}")
  add_executable(${testname} ${filename})
  target_link_libraries(${testname} PUBLIC opengl_tools geometry)
  target_compile_definitions(${testname} PUBLIC "-DGEOMETRY_DATA_DIR=\"${PROJECT_SOURCE_DIR}/data/\"")

endforeach(filename ${cpp_tests})

endif()
