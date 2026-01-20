# BHEADER ####################################################################
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# LLNL-CODE-XXXXXX. All Rights reserved. See file COPYRIGHT for details.
#
# This file is part of matred. For more information and source code
# availability, see https://www.github.com/LLNL/matred.
#
# matred is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
#################################################################### EHEADER #

set(SOURCES "")
set(HEADERS "")

FILE(GLOB SRCS *.cpp)
FILE(GLOB HDRS *.hpp)

set(SOURCES ${SOURCES} ${SRCS})
set(HEADERS ${HEADERS} ${HDRS})

add_library(${PROJECT_NAME} ${SOURCES})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include)
