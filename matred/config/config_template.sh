#!/bin/bash
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


MATRED_BASE_DIR=/path/to/matred/main/directory
MATRED_BUILD_DIR=${MATRED_BASE_DIR}/build
MATRED_INSTALL_DIR=/path/to/where/matred/is/intended/to/be/installed

mkdir -p $MATRED_BUILD_DIR
cd $MATRED_BUILD_DIR

# Force a reconfigure
rm CMakeCache.txt
rm -rf CMakeFiles

# DEBUG or OPTIMIZED
cmake \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DHYPRE_DIR=/path/to/hypre \
    -DBLAS_LIBRARIES=blas \
    -DLAPACK_LIBRARIES=lapack \
    -DCMAKE_INSTALL_PREFIX=${MATRED_INSTALL_DIR} \
    ${MATRED_BASE_DIR}
