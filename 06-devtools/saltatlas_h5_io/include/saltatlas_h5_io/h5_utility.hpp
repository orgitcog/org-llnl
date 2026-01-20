// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas_h5_io Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#ifndef H5_CPP_IO_H5_UTILITY_HPP
#define H5_CPP_IO_H5_UTILITY_HPP

#include <functional>

#include "H5Cpp.h"

namespace saltatlas {
namespace h5_io {

template <typename T>
inline auto data_type(const T) {
  throw std::invalid_argument("Unsupported HDF5 data type");
  return H5::PredType::NATIVE_CHAR;
}

template <>
inline auto data_type<char>(const char) {
  return H5::PredType::NATIVE_CHAR;
}

template <>
inline auto data_type<>(const int32_t) {
  return H5::PredType::NATIVE_INT32;
}

template <>
inline auto data_type<>(const uint32_t) {
  return H5::PredType::NATIVE_UINT32;
}

template <>
inline auto data_type<>(const int64_t) {
  return H5::PredType::NATIVE_INT64;
}

template <>
inline auto data_type<>(const uint64_t) {
  return H5::PredType::NATIVE_UINT64;
}

template <>
inline auto data_type<>(const float) {
  return H5::PredType::NATIVE_FLOAT;
}

template <>
inline auto data_type<>(const double) {
  return H5::PredType::NATIVE_DOUBLE;
}

}  // namespace h5_io
}  // namespace saltatlas

namespace saltatlas {
namespace h5_io {
namespace utility {

/// \brief Utility function.
/// Executes HDF5 operations and catches an exception if it is thrown.
inline bool try_and_catch(const std::function<bool()> &command,
                          const bool                   verbose = true) {
  if (!command) return true;
  try {
    if (!verbose) {
      H5::Exception::dontPrint();
    }
    return command();
  } catch (const H5::FileIException &error) {
    if (verbose) error.printErrorStack();
    return false;
  } catch (const H5::DataSetIException &error) {
    if (verbose) error.printErrorStack();
    return false;
  } catch (const H5::DataSpaceIException &error) {
    if (verbose) error.printErrorStack();
    return false;
  } catch (const H5::DataTypeIException &error) {
    if (verbose) error.printErrorStack();
    return false;
  } catch (...) {
    if (verbose) std::cerr << "An exception was thrown" << std::endl;
    return false;
  }
  assert(false);
  return false;
}

}  // namespace utility
}  // namespace h5_io
}  // namespace saltatlas

#endif  // H5_CPP_IO_H5_UTILITY_HPP
