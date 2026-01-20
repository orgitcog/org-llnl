/*BHEADER**********************************************************************
 *
 * Copyright (c) 2025, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-2008147. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of matred. For more information and source code
 * availability, see https://www.github.com/LLNL/matred.
 *
 * matred is free software; you can redistribute it and/or modify it under the
 * terms of the BSD-3 license.
 *
 ***********************************************************************EHEADER*/

/**
   @file linalg.hpp
   @brief Wrapper classes for hypre_ParCSRMatrix and hypre_ParVector that
          provide basic linear algebra operations like matrix-vector
          multiplications, matrix-matrix multiplications, transpose, etc.
*/

#ifndef __MATRED_LINALG_HPP
#define __MATRED_LINALG_HPP

#include <vector>
#include <set>

// hypre header files
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "temp_multivector.h"

using namespace std;

namespace matred
{

/**
 * @class ParMatrix
 * @brief Parallel matrix wrapper class for `hypre_ParCSRMatrix`.
 */
class ParMatrix
{
   hypre_ParCSRMatrix* A_;
   vector<HYPRE_Int> row_starts_;
   vector<HYPRE_Int> col_starts_;
   bool hypre_free_;
public:
   /// Constructor for an empty matrix
   ParMatrix()
      : A_(nullptr), row_starts_(0), col_starts_(0), hypre_free_(false) { }

   /// Constructor for parallel matrices containing both diag and offd blocks
   ParMatrix(MPI_Comm comm,
             HYPRE_Int global_num_rows, HYPRE_Int global_num_cols,
             vector<HYPRE_Int> row_starts, vector<HYPRE_Int> col_starts,
             HYPRE_Int *diag_i, HYPRE_Int *diag_j, double *diag_data,
             HYPRE_Int *offd_i, HYPRE_Int *offd_j, double *offd_data,
             HYPRE_Int offd_num_cols, HYPRE_Int *offd_col_map);

   /// Constructor for parallel matrices containing only the diag block
   ParMatrix(MPI_Comm comm,
             HYPRE_Int global_num_rows, HYPRE_Int global_num_cols,
             vector<HYPRE_Int> row_starts, vector<HYPRE_Int> col_starts,
             HYPRE_Int *diag_i, HYPRE_Int *diag_j, double *diag_data)
      : ParMatrix(comm, global_num_rows, global_num_cols, move(row_starts),
                  move(col_starts), diag_i, diag_j, diag_data,
                  NULL, NULL, NULL, 0, NULL) { }

   /// Constructor for wrapping hypre_ParCSRMatrix
   ParMatrix(hypre_ParCSRMatrix* A, bool hypre_free = true);

   /// Move constructor
   ParMatrix(ParMatrix&& other) noexcept;

   /// Destructor
   ~ParMatrix() { if (hypre_free_) { hypre_ParCSRMatrixDestroy(A_); } }

   /// Move assignment operator
   ParMatrix& operator=(ParMatrix&& other) noexcept;

   /// Set all nonzero entries to a constant value
   ParMatrix& operator=(double value);

   /// Set ownership of the hypre_ParCSRMatrix pointer A_
   /// If true, the destructor will call hypre_ParCSRMatrixDestroy(A_)
   void SetOwnerShip(bool ownA) { hypre_free_ = ownA; }

   /// Typecasting to hypre's hypre_ParCSRMatrix*
   operator hypre_ParCSRMatrix*() const { return A_; }
};

/// Matrix-matrix multiplication A * B
ParMatrix Mult(const ParMatrix& A, const ParMatrix& B);

/// Transpose of A
ParMatrix Transpose(const ParMatrix& A);

/// Used for constructing a ParVector from ParMatrix
enum Side { Row, Column };

/**
 * @class ParVector
 * @brief Parallel vector wrapper class for `hypre_ParVector`.
 */
class ParVector
{
   hypre_ParVector* vec_;
   vector<HYPRE_Int> starts_;
   bool hypre_free_;
public:

   /// Constructor based on given parallel offsets
   ParVector(MPI_Comm comm, HYPRE_Int global_size, vector<HYPRE_Int> starts);

   /// Constructor that takes parallel offsets from row or column side of a parallel matrix
   ParVector(const ParMatrix& A_ref, Side side);

   /// Constructor for wrapping hypre_ParVector
   ParVector(hypre_ParVector* vec, bool hypre_free);

   /// Move constructor
   ParVector(ParVector&& other) noexcept;

   /// Destructor
   ~ParVector() { if (hypre_free_) { hypre_ParVectorDestroy(vec_); } }

   /// Move assignment operator
   ParVector& operator=(ParVector&& other) noexcept;

   /// Casting ParVector as hypre_ParVector
   operator hypre_ParVector*() const { return vec_; }

   /// *this += a * vec
   ParVector& Add(double a, const ParVector& vec);

   /// l2 norm of the parallel vector
   double Norml2() const;
};

/// Matrix-vector multiplication A * v
ParVector Mult(const ParMatrix& A, const ParVector& v);

/// Matrix-vector multiplication A^T * v
ParVector MultTranspose(const ParMatrix& A, const ParVector& v);

/// Find the set of non-zero columns of a given hypre_CSRMatrix
set<unsigned> FindNonZeroColumns(const hypre_CSRMatrix& mat);

/// Set all entries of A to the given value
void SetConstantValue(hypre_CSRMatrix* A, double value);

/// Set all entries of A to the given value
void SetConstantValue(ParMatrix& A, double value);

/// Check if A_ is an identity matrix
bool IsIdentity(const ParMatrix& A_);

/// Check if A_ is a permutation
bool IsPermutation(const ParMatrix& A_);

/// Return the pointer to the data of local vector
double* Data(const ParVector& vec);

} // namespace matred

#endif /* __MATRED_LINALG_HPP */
