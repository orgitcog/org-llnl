# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# h5_cpp_io Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

function (add_saltatlas_h5_example ex_name)
    add_executable(${ex_name} ${ex_name}.cpp)
    target_link_libraries(${ex_name} PUBLIC saltatlas_h5_io)
endfunction ()

if (HDF5_FOUND AND MPI_CXX_FOUND)
    add_saltatlas_h5_example(reader_example)
    add_saltatlas_h5_example(writer_example)
endif ()
