if (NOT TARGET libLBVH)
  FetchContent_Declare(
    LBVH
    GIT_REPOSITORY https://github.com/samuelpmish/LBVH.git
    GIT_TAG 160ea948dcb610c9c1d75840f9bf3e3b669826c6 # as of January 15, 2025
  )

message("resolving dependencies: LBVH")
  set(LBVH_ENABLE_PYTHON_BINDINGS OFF CACHE INTERNAL "")
  set(LBVH_ENABLE_MATHEMATICA_BINDINGS OFF CACHE INTERNAL "")
  set(LBVH_ENABLE_TESTING OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(LBVH)
endif()
