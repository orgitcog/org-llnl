if (NOT TARGET mesh_stuff)
  FetchContent_Declare(                                                            
    mesh_stuff                                                                        
    GIT_REPOSITORY https://github.com/samuelpmish/mesh_stuff.git
    GIT_TAG 13cc1b1fd9b01dc397b880fd395dabf03e0a2061 # as of January 15, 2025
  )

  message("resolving dependencies: mesh_stuff")
  FetchContent_MakeAvailable(mesh_stuff)
endif()