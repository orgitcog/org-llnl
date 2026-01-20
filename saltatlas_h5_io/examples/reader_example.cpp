// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas_h5_io Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>

#include "saltatlas_h5_io/h5_reader.hpp"

int main() {
  // std::string
  // hdf_file_path{"/p/lustre1/salta/generators/dpockets/dp_1e9_k010_d008_first_million.hdf"};
  std::string hdf_file_path{"./out.hdf"};

  std::cout << "Open a HDF5 file: " << hdf_file_path << std::endl;
  saltatlas::h5_io::h5_reader reader(hdf_file_path);

  if (!reader.is_open()) {
    std::cerr << "Failed to open" << std::endl;
    return EXIT_FAILURE;
  }

  {
    std::vector<std::string> column_names = {"col000", "col001"};
    auto rows = reader.read_columns_row_wise<double>(column_names);
    std::cout << "\nRead two columns" << std::endl;
    if (rows.empty()) {
      std::cerr << "Failed to read" << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "Print Data (up to 10 x 10)" << std::endl;
    for (std::size_t r = 0; r < std::min(rows.size(), (std::size_t)10); ++r) {
      for (std::size_t c = 0; c < std::min(rows[r].size(), (std::size_t)10);
           ++c) {
        std::cout << rows[r][c] << " ";
      }
      std::cout << std::endl;
    }
  }

  {
    auto col = reader.read_column<double>("col000");
    std::cout << "\nRead a column" << std::endl;
    if (col.empty()) {
      std::cerr << "Failed to read" << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "Print Data (up to 10)" << std::endl;
    for (std::size_t c = 0; c < std::min(col.size(), (std::size_t)10); ++c) {
      std::cout << col[c] << std::endl;
    }
  }

  {
    auto rows = reader.read_columns_row_wise<double>();
    std::cout << "\nRead all columns" << std::endl;
    if (rows.empty()) {
      std::cerr << "Failed to read" << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "Print Data (upto 10 x 10)" << std::endl;
    for (std::size_t r = 0; r < std::min(rows.size(), (std::size_t)10); ++r) {
      for (std::size_t c = 0; c < std::min(rows[r].size(), (std::size_t)10);
           ++c) {
        std::cout << rows[r][c] << " ";
      }
      std::cout << std::endl;
    }
  }

  return 0;
}
