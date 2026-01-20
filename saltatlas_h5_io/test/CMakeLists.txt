# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# h5_cpp_io Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

function (add_saltatlas_h5_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PUBLIC saltatlas_h5_io)
endfunction ()

if (HDF5_FOUND AND MPI_CXX_FOUND)
    add_saltatlas_h5_test(test)
endif ()
