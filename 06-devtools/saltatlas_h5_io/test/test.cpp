// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas_h5_io Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>

#include "saltatlas_h5_io/h5_reader.hpp"
#include "saltatlas_h5_io/h5_writer.hpp"

void test_data_type() {
  if (saltatlas::h5_io::data_type(char(10)) != H5::PredType::NATIVE_CHAR) {
    std::abort();
  }
  if (saltatlas::h5_io::data_type(int32_t(10)) != H5::PredType::NATIVE_INT32) {
    std::abort();
  }
  if (saltatlas::h5_io::data_type(uint32_t(10)) !=
      H5::PredType::NATIVE_UINT32) {
    std::abort();
  }
  if (saltatlas::h5_io::data_type(int64_t(10)) != H5::PredType::NATIVE_INT64) {
    std::abort();
  }
  if (saltatlas::h5_io::data_type(uint64_t(10)) !=
      H5::PredType::NATIVE_UINT64) {
    std::abort();
  }
  if (saltatlas::h5_io::data_type(float(10)) != H5::PredType::NATIVE_FLOAT) {
    std::abort();
  }
  if (saltatlas::h5_io::data_type(double(10)) != H5::PredType::NATIVE_DOUBLE) {
    std::abort();
  }

  // Should throw an std::invalid_argument exception when non-supported type is
  // given
  try {
    saltatlas::h5_io::data_type(std::byte());
  } catch (const std::invalid_argument &) {
  } catch (...) {
    std::abort();
  }
}

void test_write(const std::string              &hdf_file_path,
                const std::vector<std::string> &column_names) {
  saltatlas::h5_io::h5_writer writer(hdf_file_path, column_names);
  if (!writer.is_open()) {
    std::abort();
  }

  // First write
  {
    saltatlas::h5_io::h5_writer::matrix_type<double> matrix;
    matrix.resize(2);
    for (auto &array : matrix) {
      array.resize(
          column_names
              .size());  // Column size must be equal to column_names.size()
    }
    matrix[0][0] = 0.0;
    matrix[0][1] = 0.1;
    matrix[0][2] = 0.2;
    matrix[1][0] = 1.0;
    matrix[1][1] = 1.1;
    matrix[1][2] = 1.2;
    if (!writer.write(matrix)) {
      std::abort();
    }
  }

  // Second write
  {
    saltatlas::h5_io::h5_writer::matrix_type<double> matrix;
    matrix.resize(1);
    for (auto &array : matrix) {
      array.resize(
          column_names
              .size());  // Column size must be equal to column_names.size()
    }
    matrix[0][0] = 2.0;
    matrix[0][1] = 2.1;
    matrix[0][2] = 2.2;
    if (!writer.write(matrix)) {
      std::abort();
    }
  }
}

void test_read(const std::string              &hdf_file_path,
               const std::vector<std::string> &column_names) {
  saltatlas::h5_io::h5_reader reader(hdf_file_path);
  if (!reader.is_open()) {
    std::abort();
  }

  const auto col = reader.read_column<double>(column_names[0]);
  if (col[0] != 0.0 || col[1] != 1.0 || col[2] != 2.0) {
    std::abort();
  }

  std::vector<std::string> sub_col_names{column_names[1], column_names[2]};
  const auto cols = reader.read_columns_row_wise<double>(sub_col_names);
  if (cols[0][0] != 0.1 || cols[0][1] != 0.2 || cols[1][0] != 1.1 ||
      cols[1][1] != 1.2 || cols[2][0] != 2.1 || cols[2][1] != 2.2) {
    std::abort();
  }

  const auto matrix = reader.read_columns_row_wise<double>(column_names);
  if (matrix[0][0] != 0.0 || matrix[0][1] != 0.1 || matrix[0][2] != 0.2 ||
      matrix[1][0] != 1.0 || matrix[1][1] != 1.1 || matrix[1][2] != 1.2 ||
      matrix[2][0] != 2.0 || matrix[2][1] != 2.1 || matrix[2][2] != 2.2) {
    std::abort();
  }
}

int main() {
  test_data_type();

  std::string              hdf_file_path{"./test.hdf"};
  std::vector<std::string> column_names = {"col0", "col1", "col2"};

  test_write(hdf_file_path, column_names);
  test_read(hdf_file_path, column_names);

  std::cout << "Succeeded!" << std::endl;

  return 0;
}
