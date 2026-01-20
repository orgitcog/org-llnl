// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas_h5_io Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#ifndef H5_CPP_READER_H5_READER_V2_HPP
#define H5_CPP_READER_H5_READER_V2_HPP

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "H5Cpp.h"

#include "h5_utility.hpp"

namespace saltatlas {
namespace h5_io {

/// \brief Reads a column-wise 2-D array data in a HDF5 file.
/// Each column is an independent HDF5 'dataset'.
/// Example input file:
/// /p/lustre1/salta/generators/dpockets/dp_1e9_k010_d008_first_million.hdf
///
/// \example
/// Original data
///   vector0 = {0, 1, 2}
///   vector1 = {3, 4, 5}
///
/// HDF5 file (column-wise matrix):
///   Dataset-0
///     dataset name : "column0"
///     data : [0, 3]
///   Dataset-1
///     dataset name : "column1"
///     data : [1, 4]
///   Dataset-2
///     dataset name : "column2"
///     data : [2, 5]
///
/// Row-major matrix:
///   value_type matrix[2][3] = {{0, 1, 2},
///                              {3, 4, 5}}
class h5_reader {
 public:
  explicit h5_reader(const std::string &hdf5_file_name) {
    priv_open_file(hdf5_file_name);
  }

  /// \brief Returns true if the HDF5 file is open;
  /// otherwise, returns false.
  bool is_open() const { return (!!m_ptr_h5file); }

  template <typename T>
  std::vector<std::vector<T>> read_columns_row_wise() {
    return read_columns_row_wise<T>(priv_get_column_names());
  }

  template <typename T>
  std::vector<std::vector<T>> read_columns_row_wise(
      const std::vector<std::string> &column_names) {
    if (!is_open()) return std::vector<std::vector<T>>{};

    return priv_read_matrix<T>(column_names);
  }

  template <typename T>
  std::vector<T> read_column(const std::string &column_name) {
    if (!is_open()) return std::vector<T>{};

    std::vector<T> column;
    if (!priv_read_array_dataset(column_name, &column)) {
      column.clear();
    }

    return column;
  }

  auto column_names() { return priv_get_column_names(); }

 private:
  /// \brief Opens a HDF5 file
  /// Returns true on success; otherwise, false.
  bool priv_open_file(const std::string &file_name) {
    const bool success = utility::try_and_catch([this, &file_name]() -> bool {
      // MEMO: do not pass std::string to HDF5 functions.
      m_ptr_h5file =
          std::make_unique<H5::H5File>(file_name.c_str(), H5F_ACC_RDONLY);
      return true;
    });
    if (!success) {
      m_ptr_h5file.reset(nullptr);
    }

    return success;
  }

  /// \brief Gets column (dataset in HDF5's terminology) names.
  /// Read names are sorted.
  /// Returns true on success; otherwise, false.
  std::vector<std::string> priv_get_column_names() {
    std::vector<std::string> column_names;

    auto op_func = [](hid_t loc_id, const char *name, const H5O_info_t *info,
                      void *out_data) -> herr_t {
      if (info->type == H5O_TYPE_DATASET) {
        auto *out = reinterpret_cast<std::vector<std::string> *>(out_data);
        out->emplace_back(std::string(name));
      }
      return 0;
    };

    const auto status = H5Ovisit(m_ptr_h5file->getId(), H5_INDEX_NAME,
                                 H5_ITER_NATIVE, op_func, &column_names);
    if (status != 0) {
      std::cerr << __FILE__ << " : " << __LINE__ << "Failed to get column names"
                << std::endl;
      column_names.clear();
      return column_names;
    }

    std::sort(column_names.begin(), column_names.end());

    return column_names;
  }

  /// \brief Reads column-wise data from a HDF5 file and store it into a
  /// row-wise matrix.
  template <typename T>
  auto priv_read_matrix(const std::vector<std::string> &column_names) {
    std::vector<std::vector<T>> row_wise_matrix;
    for (std::size_t col_no = 0; col_no < column_names.size(); ++col_no) {
      std::vector<T> column;
      if (!priv_read_array_dataset(column_names[col_no], &column)) {
        row_wise_matrix.clear();
        return row_wise_matrix;  // Returns an empty one
      }
      if (col_no == 0) {
        // Allocate a matrix, assuming that all columns have the same sizes.
        row_wise_matrix.resize(column.size());
        for (auto &row : row_wise_matrix) {
          row.resize(column_names.size());
        }
      }
      // Put the read data into the row-major matrix
      for (std::size_t j = 0; j < column.size(); ++j) {
        row_wise_matrix[j][col_no] = column[j];
      }
    }
    return row_wise_matrix;
  }

  /// \brief Reads a 'dataset' whose format is an array.
  /// Returns true on success; otherwise, false.
  template <typename T>
  bool priv_read_array_dataset(const std::string &dataset_name,
                               std::vector<T>    *buf) {
    if (!m_ptr_h5file) return false;

    const bool ret = utility::try_and_catch([this, &dataset_name, buf]() {
      H5::DataSet dataset = m_ptr_h5file->openDataSet(dataset_name.c_str());

      hsize_t length = 0;
      dataset.getSpace().getSimpleExtentDims(&length);

      buf->resize(length);
      dataset.read(buf->data(), data_type(T{}));

      return true;
    });

    return ret;
  }

  std::unique_ptr<H5::H5File> m_ptr_h5file{nullptr};
};

}  // namespace h5_io
}  // namespace saltatlas

#endif  // H5_CPP_READER_H5_READER_V2_HPP
