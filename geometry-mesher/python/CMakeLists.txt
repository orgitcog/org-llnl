include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
)
FetchContent_MakeAvailable(nanobind)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
nanobind_add_module(
    
  # name, must match module name in python_bindings.cpp
  geometry_bindings 

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # sources
  python_bindings.cpp

)

target_link_libraries(geometry_bindings PUBLIC geometry)
set_target_properties(geometry_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
set_target_properties(geometry_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/python)

install(TARGETS geometry_bindings LIBRARY DESTINATION .)