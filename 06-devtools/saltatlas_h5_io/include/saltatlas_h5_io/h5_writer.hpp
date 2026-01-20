// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas_h5_io Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Memo:
// https://support.hdfgroup.org/HDF5/doc/cpplus_RM/extend_ds_8cpp-example.html

#ifndef H5_CPP_READER_H5_WRITER_HPP
#define H5_CPP_READER_H5_WRITER_HPP

#include <cassert>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "H5Cpp.h"
#include "h5_utility.hpp"

namespace saltatlas {
namespace h5_io {

/// \brief Writes a row-wise 2D matrix data into a HDF5 file.
/// The HDF5 file uses the column-wise format.
/// Each column is an independent HDF5 'dataset', there is a single 1D array per
/// HDF5 'dataset'.
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
///   uint64_t matrix[2][3] = {{0, 1, 2},
///                              {3, 4, 5}}
class h5_writer {
 public:
  template <typename T>
  using array_type = std::vector<T>;
  template <typename T>
  using matrix_type = std::vector<array_type<T>>;

  h5_writer(const std::string              &hdf5_file_name,
            const std::vector<std::string> &column_names)
      : m_column_names(column_names) {
    priv_create_file(hdf5_file_name);
  }

  /// \brief Destructor.
  /// Calls close().
  ~h5_writer() noexcept { close(); }

  /// \brief Returns true if the HDF5 file is open;
  /// otherwise, returns false.
  bool is_open() const { return (!!m_ptr_h5file); }

  /// \brief Writes data to the already opened HDF5 file.
  /// This function keeps adding new matrix at the bottom.
  /// An arbitrary number of rows in row_wise_matrix is acceptable.
  /// On the other hand, its column size must be equal to the number of column
  /// names passed at the constructor.
  template <typename T>
  bool write(const matrix_type<T> &row_wise_matrix) {
    if (!is_open()) return false;

    if (!priv_write_matrix<T>(row_wise_matrix)) {
      return false;
    }

    return true;
  }

  /// \brief Close the HDF5 file and release used buffers.
  /// Returns true on success; otherwise, false.
  bool close() noexcept { return priv_close_file(); }

 private:
  /// \brief Creates a HDF5 file.
  /// Returns true on success; otherwise, false.
  bool priv_create_file(const std::string &file_name) {
    const bool success = utility::try_and_catch([this, &file_name]() -> bool {
      // MEMO: do not pass std::string to HDF5 functions.
      m_ptr_h5file =
          std::make_unique<H5::H5File>(file_name.c_str(), H5F_ACC_TRUNC);
      return true;
    });
    if (!success) {
      m_ptr_h5file.reset(nullptr);
    }
    return true;
  }

  /// \brief Close the HDF5 file and release used buffers.
  /// Returns true on success; otherwise, false.
  bool priv_close_file() {
    const bool ret = utility::try_and_catch([this]() {
      m_ptr_h5datasets.clear();
      m_ptr_h5file.reset(nullptr);
      m_column_names.clear();
      m_num_rows = 0;
      return true;
    });
    return ret;
  }

  /// \brief Writes a row-wise matrix into a HDF5 file with the column-wise
  /// style. The output HDF5 contains multiple 'datasets'. One column per
  /// dataset. Returns true on success; otherwise, false.
  template <typename T>
  bool priv_write_matrix(const matrix_type<T> &row_wise_matrix) {
    for (std::size_t col_no = 0; col_no < m_column_names.size(); ++col_no) {
      array_type<T> column;
      // Extract a column
      for (const auto &row : row_wise_matrix) {
        column.push_back(row[col_no]);
      }

      // Create dataset before writing the first matrix
      if (m_num_rows == 0) {
        if (!priv_create_1d_datasets<T>(m_column_names[col_no],
                                        column.size())) {
          return false;
        }
      }

      // Write a column as a 1D array (one array per dataset)
      if (!priv_write_array_dataset(m_column_names[col_no], column)) {
        return false;
      }
    }

    m_num_rows += row_wise_matrix.size();

    return true;
  }

  /// \brief Create a 1D dataset.
  /// Returns true on success; otherwise, false.
  template <typename T>
  bool priv_create_1d_datasets(const std::string &dataset_name,
                               const hsize_t      size) {
    if (!m_ptr_h5file) return false;

    // Create a new dataset
    const bool ret = utility::try_and_catch([this, &dataset_name, size]() {
      hsize_t       max_size = H5S_UNLIMITED;
      H5::DataSpace dataspace(1, &size, &max_size);

      // Modify dataset creation properties, i.e. enable chunking.
      H5::DSetCreatPropList cparms;
      hsize_t               chunk_dim = {1024};
      cparms.setChunk(1, &chunk_dim);

      m_ptr_h5datasets[dataset_name] =
          std::make_unique<H5::DataSet>(m_ptr_h5file->createDataSet(
              dataset_name.c_str(), data_type(T{}), dataspace, cparms));
      return true;
    });

    return ret;
  }

  /// \brief Writes a 'dataset' whose format is an array.
  /// Returns true on success; otherwise, false.
  template <typename T>
  bool priv_write_array_dataset(const std::string   &dataset_name,
                                const array_type<T> &buf) {
    if (!m_ptr_h5file) return false;

    const bool ret = utility::try_and_catch([this, &dataset_name, buf]() {
      hsize_t write_size = buf.size();

      // Extend the dataset.
      hsize_t new_size = m_num_rows + write_size;
      m_ptr_h5datasets.at(dataset_name)->extend(&new_size);

      // Select a hyperslab.
      H5::DataSpace space_to_write =
          m_ptr_h5datasets.at(dataset_name)->getSpace();
      hsize_t offset = m_num_rows;
      space_to_write.selectHyperslab(H5S_SELECT_SET, &write_size, &offset);

      H5::DataSpace dataspace(1, &write_size);

      m_ptr_h5datasets.at(dataset_name)
          ->write(buf.data(), data_type(T{}), dataspace,
                  space_to_write);
      return true;
    });

    return ret;
  }

  std::unique_ptr<H5::H5File> m_ptr_h5file{nullptr};
  std::unordered_map<std::string, std::unique_ptr<H5::DataSet>>
                           m_ptr_h5datasets;
  std::vector<std::string> m_column_names;
  std::size_t              m_num_rows{0};
};

}  // namespace h5_io
}  // namespace saltatlas

#endif  // H5_CPP_READER_H5_WRITER_HPP
