# Projects Settings
cmake_minimum_required (VERSION 3.15)
project (geometry_tools)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# if we're making a shared library, then
# we'll need the depenendencies it links to
# to have the position-independent code flag
if (GEOM_ENABLE_MATHEMATICA_BINDINGS OR GEOM_ENABLE_PYTHON_BINDINGS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(geometry_sources 
  ${PROJECT_SOURCE_DIR}/src/geometry/geometry.hpp
  ${PROJECT_SOURCE_DIR}/src/geometry/geometry.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/hex_lattice.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/unit_cells.hpp
  ${PROJECT_SOURCE_DIR}/src/geometry/unit_cells.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/universal_meshing.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/mesh_import.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/mesh_export.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/improve_boundary.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/laplace_smoothing.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/region.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/parse_dat.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/postprocessing.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/quality_summary.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/verification.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/periodic_functions.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/promote_to_quadratic.cpp
  ${PROJECT_SOURCE_DIR}/src/geometry/simplex_mesh.cpp
)

add_library(geometry STATIC ${geometry_sources})

if(GEOM_PEDANTIC) 
  set_property(TARGET geometry PROPERTY COMPILE_WARNING_AS_ERROR ON)
endif()

target_include_directories(geometry PUBLIC ${PROJECT_SOURCE_DIR}/src)

include(FetchContent)

include(cmake/mesh_stuff.cmake)
include(cmake/lbvh.cmake)
target_link_libraries(geometry PUBLIC mesh_stuff libLBVH)

find_package(TIFF)
if (TIFF_FOUND)
  message("Found libTIFF, enabling TIFF features")
  target_compile_definitions(geometry PUBLIC -DUM_TIFF_SUPPORT)
  target_link_libraries(geometry PUBLIC ${TIFF_LIBRARIES})
  target_include_directories(geometry PUBLIC ${TIFF_INCLUDE_DIRS})
  target_sources(geometry PRIVATE src/geometry/image.cpp)
endif()

if (ENABLE_OPENGL)
  add_subdirectory(src/opengl)
endif()

if (BUILD_TESTS)
  enable_testing()
  include(cmake/gtest.cmake)

  file(GLOB cpp_tests ${PROJECT_SOURCE_DIR}/src/tests/*.cpp)
  foreach(filename ${cpp_tests})		
    get_filename_component(testname ${filename} NAME_WE)
    add_executable(${testname} ${filename})
    target_link_libraries(${testname} PUBLIC geometry GTest::gtest_main)
    add_test(${testname} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testname})
    target_compile_definitions(${testname} PUBLIC "-DGEOMETRY_DATA_DIR=\"${PROJECT_SOURCE_DIR}/data/\"")
  endforeach(filename ${cpp_tests})
endif()

#target_compile_options(mylib PUBLIC -save-temps)
#target_compile_options(geometry PUBLIC -fsanitize=thread)
#target_link_options(geometry PUBLIC -fsanitize=thread)

if (ENABLE_ASAN) 
  target_compile_options(geometry PUBLIC -fsanitize=address)
  target_link_options(geometry PUBLIC -fsanitize=address)
endif()

if (GEOM_ENABLE_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()

if (GEOM_ENABLE_MATHEMATICA_BINDINGS)
  add_subdirectory(mathematica)
endif()
