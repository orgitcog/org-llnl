include(FetchContent)
FetchContent_Declare(
  wll_interface
  GIT_REPOSITORY https://github.com/samuelpmish/wll_interface.git
  GIT_TAG main
)
FetchContent_MakeAvailable(wll_interface)

#file(GLOB mathematica_headers ${PROJECT_SOURCE_DIR}/mathematica/*.h)
file(GLOB mathematica_bindings ${PROJECT_SOURCE_DIR}/mathematica/mathematica_bindings.cpp)

add_library(UM_wll SHARED ${mathematica_headers} ${mathematica_bindings})
target_include_directories(UM_wll PUBLIC ${PROJECT_SOURCE_DIR}/mathematica/src)
target_link_libraries(UM_wll PUBLIC geometry wll_interface)

set(UM_WLL_COPY_DIR ${PROJECT_SOURCE_DIR}/mathematica)
message("copying UM_wll to ${UM_WLL_COPY_DIR}")
add_custom_command(TARGET UM_wll
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:UM_wll> "${UM_WLL_COPY_DIR}"
  COMMENT "Copy dll file to ${UM_WLL_COPY_DIR} directory" VERBATIM
)
