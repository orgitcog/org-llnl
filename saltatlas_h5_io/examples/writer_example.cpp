// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas_h5_io Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>

#include "saltatlas_h5_io/h5_writer.hpp"

int main() {
  std::string              hdf_file_path{"./out.hdf"};
  std::vector<std::string> column_names = {"col000", "col001"};
  std::cout << "Create a HDF5 file: " << hdf_file_path << std::endl;
  saltatlas::h5_io::h5_writer writer(hdf_file_path, column_names);

  if (!writer.is_open()) {
    std::cerr << "Failed to create" << std::endl;
    return EXIT_FAILURE;
  }

  // First write
  {
    saltatlas::h5_io::h5_writer::matrix_type<double> matrix;
    matrix.resize(2);
    for (auto &array : matrix) {
      // Column size must be equal to column_names.size()
      array.resize(column_names.size());
    }
    matrix[0][0] = 0;
    matrix[0][1] = 1;
    matrix[1][0] = 2;
    matrix[1][1] = 3;

    std::cout << "Write data" << std::endl;
    if (!writer.write(matrix)) {
      std::cerr << "Failed to write" << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Second write
  {
    saltatlas::h5_io::h5_writer::matrix_type<double> matrix;
    matrix.resize(3);  // Can be an arbitrary size
    for (auto &array : matrix) {
      // Column size must be equal to column_names.size()
      array.resize(column_names.size());
    }
    matrix[0][0] = 4;
    matrix[0][1] = 5;
    matrix[1][0] = 6;
    matrix[1][1] = 7;
    matrix[2][0] = 8;
    matrix[2][1] = 9;

    std::cout << "Write data" << std::endl;
    if (!writer.write(matrix)) {
      std::cerr << "Failed to write" << std::endl;
      return EXIT_FAILURE;
    }
  }

  return 0;
}
