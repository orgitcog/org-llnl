# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# h5_cpp_io Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.14)
project(
    saltatlas_h5_io
    VERSION 0.1
    LANGUAGES C CXX
)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(SALTATLAS_H5_IO_MAIN_PROJECT ON)
endif ()

#
# HDF5
#
find_package(
    HDF5 1.10.7
    COMPONENTS C CXX
    REQUIRED
)

#
# MPI
#
find_package(MPI)

add_library(saltatlas_h5_io INTERFACE)
add_library(saltatlas_h5_io::saltatlas_h5_io ALIAS saltatlas_h5_io)
target_compile_features(saltatlas_h5_io INTERFACE cxx_std_17)
target_include_directories(
    saltatlas_h5_io
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
              $<INSTALL_INTERFACE:include>
)
target_compile_definitions(saltatlas_h5_io INTERFACE ${HDF5_DEFINITIONS})
target_link_libraries(saltatlas_h5_io INTERFACE ${HDF5_LIBRARIES} MPI::MPI_CXX)
target_include_directories(saltatlas_h5_io INTERFACE ${HDF5_INCLUDE_DIRS})

option(SALTATLAS_H5_IO_INSTALL "Generate the install target"
       ${SALTATLAS_H5_IO_MAIN_PROJECT}
)
if (SALTATLAS_H5_IO_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(SALTATLAS_H5_IO_EXPORT_TARGETS saltatlas_h5_io)

    install(
        TARGETS ${SALTATLAS_H5_IO_EXPORT_TARGETS}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # create version file
    write_basic_package_version_file(
        "${PROJECT_NAME}ConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY ExactVersion
    )

    # create config file
    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

    install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

endif ()

option(JUST_INSTALL_SALTATLAS_H5_IO "Skip executables" OFF)
if (JUST_INSTALL_SALTATLAS_H5_IO)
    return()
endif ()

#
# Executables
#
add_subdirectory(examples)
add_subdirectory(test)
