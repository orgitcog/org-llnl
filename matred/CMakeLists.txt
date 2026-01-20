# BHEADER ####################################################################
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# LLNL-CODE-2008147. All Rights reserved. See file COPYRIGHT for details.
#
# This file is part of matred. For more information and source code
# availability, see https://www.github.com/LLNL/matred.
#
# matred is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license.
#
#################################################################### EHEADER #

###
# Setup the basics
###
cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
set(PROJECT_NAME matred)
project(${PROJECT_NAME})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Enforce C++11
if ((NOT CMAKE_CXX_STANDARD) OR (CMAKE_CXX_STANDARD LESS 17))
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Using CXX standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fstack-protector-all")
set(CMAKE_CXX_FLAGS_OPTIMIZED "${CMAKE_CXX_FLAGS_OPTIMIZED} -O3")

###
# Find third-party libraries
###
set(TPL_LIBRARIES "")

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
list(APPEND TPL_LIBRARIES ${MPI_LIBRARIES})
if(MPI_CXX_COMPILE_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
endif()

# BLAS/LAPACK
find_package(BLAS REQUIRED)
list(APPEND TPL_LIBRARIES ${BLAS_LIBRARIES})
find_package(LAPACK REQUIRED)
list(APPEND TPL_LIBRARIES ${LAPACK_LIBRARIES})

# Gfortran
list(APPEND TPL_LIBRARIES "gfortran")

# HYPRE
find_package(HYPRE REQUIRED)
include_directories(${HYPRE_INCLUDE_DIRS})
list(APPEND TPL_LIBRARIES ${HYPRE_LIBRARIES})

# Determine Hypre version
try_run(HYPRE_VERSION_RUN_RESULT HYPRE_VERSION_COMPILE_RESULT
        ${CMAKE_CURRENT_BINARY_DIR}/config
        ${CMAKE_CURRENT_SOURCE_DIR}/config/get_hypre_version.cpp
        CMAKE_FLAGS -DINCLUDE_DIRECTORIES:STRING=${HYPRE_INCLUDE_DIRS}
        RUN_OUTPUT_VARIABLE HYPRE_VERSION_OUTPUT)
if ((HYPRE_VERSION_RUN_RESULT EQUAL 0) AND HYPRE_VERSION_OUTPUT)
  string(STRIP "${HYPRE_VERSION_OUTPUT}" HYPRE_VERSION)
  set(${PROJECT_NAME}_HYPRE_VERSION ${HYPRE_VERSION} CACHE STRING "HYPRE version in MATRED." FORCE)
  message(STATUS "Found HYPRE version ${${PROJECT_NAME}_HYPRE_VERSION} in MATRED dependency")
else()
  message(FATAL_ERROR "Unable to determine HYPRE version.")
endif()

list(REMOVE_DUPLICATES TPL_LIBRARIES)

#####
# export configuration of optional packages
#####
configure_file(
  "${PROJECT_SOURCE_DIR}/config/${PROJECT_NAME}_config.h.in"
  "${PROJECT_BINARY_DIR}/config/${PROJECT_NAME}_config.h"
  )
include_directories(${PROJECT_BINARY_DIR}) # for _config.h

###
# Setup internal directories
###
enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})

add_subdirectory(${PROJECT_SOURCE_DIR}/src)
