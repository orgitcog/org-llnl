INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../src
)


SET(LINK_LIBRARIES
    StreamingStatistics
)


ADD_EXECUTABLE(statistics_unit_test  statistics_unit_test.cpp)   
TARGET_LINK_LIBRARIES(statistics_unit_test ${LINK_LIBRARIES})


INSTALL(TARGETS   statistics_unit_test
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
)

