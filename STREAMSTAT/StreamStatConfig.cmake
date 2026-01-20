# - Config file for the STREAMSTAT package
# It defines the following variables
#  STREAMSTAT_INCLUDE_DIRS - include directories for DataFlow
#  STREAMSTAT_LIBRARIES    - libraries to link against
 
if(NOT STREAMSTAT_FOUND)

  # Compute paths
  get_filename_component(STREAMSTAT_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
 
  # include the main exported targets
  include("${STREAMSTAT_CMAKE_DIR}/StreamStat.cmake")

  set(STREAMSTAT_FOUND TRUE)

endif()
