cmake_minimum_required(VERSION 3.0)

project(StreamingStatistics)

# too many warnings under windows
if(WIN32)
  set(CMAKE_C_WARNING_LEVEL   0)
  set(CMAKE_CXX_WARNING_LEVEL 0)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /W0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
endif()

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(utilities)

INSTALL(EXPORT StreamStat
        NAMESPACE StreamStat::
        DESTINATION lib/cmake
)

INSTALL(FILES StreamStatConfig.cmake
        DESTINATION lib/cmake
)
