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
   @file linalg.cpp
   @brief Wrapper classes for hypre_ParCSRMatrix and hypre_ParVector that
          provide basic linear algebra operations like matrix-vector
          multiplications, matrix-matrix multiplications, transpose, etc.
*/

#include <algorithm>

#include "linalg.hpp"
#include "utilities.hpp"
#include "config/matred_config.h"

using namespace std;

namespace matred
{

ParMatrix::ParMatrix(MPI_Comm comm,
                     HYPRE_Int global_num_rows, HYPRE_Int global_num_cols,
                     vector<HYPRE_Int> row_starts, vector<HYPRE_Int> col_starts,
                     HYPRE_Int *diag_i, HYPRE_Int *diag_j, double *diag_data,
                     HYPRE_Int *offd_i, HYPRE_Int *offd_j, double *offd_data,
                     HYPRE_Int offd_num_cols, HYPRE_Int *offd_col_map)
   : row_starts_(std::move(row_starts)), col_starts_(std::move(col_starts)), hypre_free_(true)
{
   A_ = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                 row_starts_.data(), col_starts_.data(),
                                 offd_num_cols, 0, 0);

   auto SetCSR=[](HYPRE_Int *i, HYPRE_Int *j, double *data, hypre_CSRMatrix *mat)
   {
      mat->i = i ? i : HypreCTAlloc<HYPRE_Int>(mat->num_rows+1);
      mat->j = j;
      mat->data = data;
      mat->num_nonzeros = mat->i[mat->num_rows];
      hypre_CSRMatrixSetRownnz(mat);
   };
   SetCSR(diag_i, diag_j, diag_data, A_->diag);
   SetCSR(offd_i, offd_j, offd_data, A_->offd);

   hypre_ParCSRMatrixColMapOffd(A_) = offd_col_map;
#if MATRED_HYPRE_VERSION <= 22200
   hypre_ParCSRMatrixSetRowStartsOwner(A_, false);
   hypre_ParCSRMatrixSetColStartsOwner(A_, false);
#endif
   hypre_ParCSRMatrixSetNumNonzeros(A_);
   hypre_MatvecCommPkgCreate(A_);
}

ParMatrix::ParMatrix(hypre_ParCSRMatrix* A, bool hypre_free)
   : A_(A), row_starts_(2), col_starts_(2), hypre_free_(hypre_free)
{
#if MATRED_HYPRE_VERSION <= 22200
   if (A->owns_row_starts == false)
   {
      std::copy_n(A->row_starts, 2, row_starts_.begin());
      hypre_ParCSRMatrixRowStarts(A_) = row_starts_.data();
   }
   if (A->owns_col_starts == false)
   {
      std::copy_n(A->col_starts, 2, col_starts_.begin());
      hypre_ParCSRMatrixColStarts(A_) = col_starts_.data();
   }
#endif

   hypre_ParCSRMatrixSetNumNonzeros(A_);
   if (!A_->comm_pkg) { hypre_MatvecCommPkgCreate(A_); }
}

ParMatrix::ParMatrix(ParMatrix&& other) noexcept
   : A_(other.A_), row_starts_(std::move(other.row_starts_)),
     col_starts_(std::move(other.col_starts_)), hypre_free_(other.hypre_free_)
{
   other.A_ = nullptr;
   other.hypre_free_ = false;
}

ParMatrix& ParMatrix::operator=(ParMatrix&& other) noexcept
{
   A_ = other.A_;
   other.A_ = nullptr;
   row_starts_ = std::move(other.row_starts_);
   col_starts_ = std::move(other.col_starts_);
   hypre_free_ = other.hypre_free_;
   other.hypre_free_ = false;
   return *this;
}

ParMatrix& ParMatrix::operator=(double value)
{
   SetConstantValue(*this, value);
   return *this;
}

ParMatrix Mult(const ParMatrix& A, const ParMatrix& B)
{
   return ParMatrix(hypre_ParMatmul(A, B));
}

ParMatrix Transpose(const ParMatrix& A)
{
   hypre_ParCSRMatrix * At;
   hypre_ParCSRMatrixTranspose(A, &At, 1);
   return ParMatrix(At);
}

ParVector::ParVector(MPI_Comm comm, HYPRE_Int global_size, vector<HYPRE_Int> starts)
   : starts_(std::move(starts)), hypre_free_(true)
{
   vec_ = hypre_ParVectorCreate(comm, global_size, starts_.data());
   hypre_ParVectorInitialize(vec_);
#if MATRED_HYPRE_VERSION <= 22200
   hypre_ParVectorSetPartitioningOwner(vec_, false);
#endif
}

ParVector::ParVector(const ParMatrix& A_ref, Side side)
   : hypre_free_(true)
{
   auto A = (hypre_ParCSRMatrix*)A_ref;
   HYPRE_Int glob_size = side == Row ? A->global_num_rows : A->global_num_cols;
   HYPRE_Int* starts = side == Row ? A->row_starts : A->col_starts;
   vec_ = hypre_ParVectorCreate(A->comm, glob_size, starts);
   hypre_ParVectorInitialize(vec_);
#if MATRED_HYPRE_VERSION <= 22200
   hypre_ParVectorSetPartitioningOwner(vec_, false);
#endif
}

ParVector::ParVector(hypre_ParVector* vec, bool hypre_free)
   : vec_(vec), starts_(2), hypre_free_(hypre_free)
{
#if MATRED_HYPRE_VERSION <= 22200
   if (vec_->owns_partitioning == false)
   {
      std::copy_n(vec->partitioning, 2, starts_.begin());
      hypre_ParVectorPartitioning(vec_) = starts_.data();
   }
#endif
}


ParVector::ParVector(ParVector&& other) noexcept
   : vec_(other.vec_), starts_(std::move(other.starts_)), hypre_free_(other.hypre_free_)
{
   other.vec_ = nullptr;
   other.hypre_free_ = false;
}

ParVector& ParVector::operator=(ParVector&& other) noexcept
{
   vec_ = other.vec_;
   other.vec_ = nullptr;
   starts_ = std::move(other.starts_);
   hypre_free_ = other.hypre_free_;
   other.hypre_free_ = false;
   return *this;
}

ParVector& ParVector::Add(double a, const ParVector& vec)
{
   auto v = (hypre_ParVector*)vec;
   for (int i = 0; i < vec_->actual_local_size; ++i)
   {
      vec_->local_vector->data[i] += (a * v->local_vector->data[i]);
   }
   return *this;
}

double ParVector::Norml2() const
{
   double loc_norm_sq = 0.0;
   for (int i = 0; i < vec_->actual_local_size; i++)
   {
      double entry = vec_->local_vector->data[i];
      loc_norm_sq += entry * entry;
   }
   double norm_sq = AllReduce(loc_norm_sq, MPI_SUM, vec_->comm);
   return sqrt(norm_sq);
}

ParVector Mult(const ParMatrix& A, const ParVector& v)
{
   ParVector out(A, Side::Row);
   hypre_ParCSRMatrixMatvec(1, A, v, 0, out);
   return out;
}

ParVector MultTranspose(const ParMatrix& A, const ParVector& v)
{
   ParVector out(A, Side::Column);
   hypre_ParCSRMatrixMatvecT(1, A, v, 0, out);
   return out;
}

set<unsigned> FindNonZeroColumns(const hypre_CSRMatrix& mat)
{
   std::set<unsigned> cols;
   const int* mat_j = mat.j;
   const int* end = mat_j + mat.num_nonzeros;
   for (; mat_j != end; mat_j++) { cols.insert(*mat_j); }
   return cols;
}

void SetConstantValue(hypre_CSRMatrix* A, double value)
{
   std::fill_n(A->data, A->num_nonzeros, value);
}

void SetConstantValue(ParMatrix& A, double value)
{
   SetConstantValue(((hypre_ParCSRMatrix*)A)->diag, value);
   SetConstantValue(((hypre_ParCSRMatrix*)A)->offd, value);
}

bool IsIdentity(const ParMatrix& A_)
{
   hypre_ParCSRMatrix* A = A_;
   hypre_CSRMatrix* diag = A->diag;
   bool diag_is_identity = true;
   for (int i = 0; i < diag->num_rows; ++i)
   {
      if ((diag->i[i+1]-diag->i[i]) != 1 || diag->j[i] != i || diag->data[i] != 1.0)
      {
         diag_is_identity = false;
         break;
      }
   }

   bool owned_is_identity = diag_is_identity && (A->offd->num_nonzeros == 0);
   return AllReduce(owned_is_identity, MPI_LAND, A->comm);
}

bool IsPermutation(const ParMatrix& A_)
{
   auto AT = Transpose(A_);
   auto ATA = Mult(AT, A_);
   auto AAT = Mult(A_, AT);

   hypre_ParCSRMatrix* A = A_;
   const bool A_is_square = (A->global_num_rows == A->global_num_cols);

   return IsIdentity(ATA) && IsIdentity(AAT) && A_is_square;
}

double* Data(const ParVector& vec)
{
   return ((hypre_ParVector *)vec)->local_vector->data;
}

} // namespace matred
