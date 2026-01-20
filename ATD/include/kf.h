#ifndef KF_H_
#define KF_H_

#include <functional>
#include <Eigen/Dense>
#include "common.h"

// Kalman Filter
//  This unit implements various Kalman filters (or model-based processors) for
//  sequential state estimation.  It is not fully generic, since it is adapted
//  to some domain-specific needs, most notably the block-diagonal structure of
//  the model and the need to adaptively update the model for potentially
//  varying measurement intervals.
//
// Copyright (C) 2021, Lawrence Livermore National Security, LLC.
//  All rights reserved. LLNL-CODE-837067
//
// The Department of Homeland Security sponsored the production of this
//  material under DOE Contract Number DE-AC52-07N427344 for the management
//  and operation of Lawrence Livermore National Laboratory. Contract no.
//  DE-AC52-07NA27344 is between the U.S. Department of Energy (DOE) and
//  Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
//  See license for disclaimers, notice of U.S. Government Rights and license
//  terms and conditions.

namespace tmon
{

namespace KF
{
using Scalar = double;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

struct State
{
    Vector x; // state vector (N x 1)
    Matrix P; // state error covariance matrix (N x N)

    // Check validity of the state (e.g., conformability of dimensions)
    bool valid() const;
};

// M = number of measurements, N = number of state variables
struct Model
{
    // Initial Conditions
    Vector x0; // initial state vector (N x 1)
    Matrix P0; // initial state error covariance matrix (N x N)

    // Deterministic Inputs
    Vector u; // control input (here, mean a.k.a. mu) (N x 1)

    // State-Space Matrices
    Matrix A; // process or state transition matrix (N x N)
    Matrix B; // control matrix (N x N)
    Matrix C; // measurement matrix (M x N)
    // no D matrix needed

    // Covariance Matrices
    Matrix Q; // process noise matrix (a.k.a. Rww) (N x N)
    Matrix R; // measurement noise matrix (a.k.a. Rvv) (M x M)

    // This resize is destructive; matrices are zero-initialized
    void resize_zeroed(int num_meas, int num_states);
    // This resize conserves current values and zeroes out new entries
    void resize(int num_meas, int num_states);
    // Check validity of the model (e.g., conformability of dimensions)
    bool valid() const;
};

struct ModelBlockRef
{
    explicit ModelBlockRef(Model& m);
    explicit ModelBlockRef(Model& m, int N_single, int block_idx);

    // Initial Conditions
    Eigen::Ref<Vector> x0; // initial state vector
    Eigen::Ref<Matrix> P0; // initial state error covariance matrix

    // Deterministic Inputs
    Eigen::Ref<Vector> u; // control input (here, mean a.k.a. mu)

    // State-Space Matrices
    Eigen::Ref<Matrix> A; // process or state transition matrix
    Eigen::Ref<Matrix> B; // control matrix
    // Omitting C since it does not have block structure:
    // Eigen::Ref<Matrix> C; // measurement matrix
    // no D matrix needed

    // Covariance Matrices
    Eigen::Ref<Matrix> Q; // process noise matrix (a.k.a. Rww)
    // Omitting R since it lacks dependence on N:
    // Eigen::Ref<Matrix> R; // measurement noise matrix (a.k.a. Rvv) (M x M)
};

struct IdentifiedModel : public Model
{
    ClockId clock_id;
};

struct ModelDims
{
    int M; // number of measurements
    int N; // number of state variables
};

struct Prediction : public State
{
    Vector y;       // Predicted measurements (M x 1)
};

struct Innovations
{
    Vector e;       // innovations vector (M x 1)
    Matrix e_cov;   // innovations covariance matrix (M x M)
    Matrix K;       // Kalman gain matrix (N x M)
};

struct Correction : public State
{
};

class Filter
{
  public:
    explicit Filter(Model model);
    virtual ~Filter() = default;
    Filter(const Filter&) = delete;
    Filter& operator=(const Filter&) = delete;
    Filter(Filter&&) = default;
    Filter& operator=(Filter&&) = default;

    ModelDims get_model_dims() const;
    const Model& model() const;
    void reset();
    void set_meas_model(Matrix C, const Vector& R_diag);
    void set_model(Model m);
    const State& state() const;
    Innovations update(const Vector& meas);
    Innovations update(Scalar meas, int meas_row);
    bool valid() const;

  protected:
    Model& model();

  private:
    Model model_;
    State state_;
};

class BlockFilter : public Filter
{
  public:
    BlockFilter();
    BlockFilter(std::vector<IdentifiedModel> models,
        const Eigen::MatrixX2i& meas_pairs);
    ~BlockFilter() override = default;
    BlockFilter(BlockFilter&&) = default;
    BlockFilter& operator=(BlockFilter&&) = default;

    bool add_model(IdentifiedModel model);
    int add_new_meas_pair(ClockId id1, ClockId id2, Scalar new_R_entry);
    int get_meas_pair_row(ClockId id1, ClockId id2) const noexcept;
    ModelDims get_single_model_dims() const;
    bool has_model_for(ClockId id) const noexcept;
    bool remove_model(ClockId clk_id);
    // Updates all components of a single model (one block) identified by clk_id
    //  that have a dependence on tau, the measurement interval; these are the
    //  state transition matrix (A), control matrix (B), and process noise
    //  matrix (Q)
    template <typename DerivedA, typename DerivedB, typename DerivedQ>
    void update_single_model_for_tau(ClockId clk_id,
        Eigen::MatrixBase<DerivedA>&& A, Eigen::MatrixBase<DerivedB>&& B,
        Eigen::MatrixBase<DerivedQ>&& Q)
    {
        auto blk_ref = this->get_model_block_ref(clk_id);
        blk_ref.A = std::move(A);
        blk_ref.B = std::move(B);
        blk_ref.Q = std::move(Q);
    }

  private:
    std::vector<IdentifiedModel> models_;

    int get_idx_for(ClockId id) const noexcept;
    ModelBlockRef get_model_block_ref(ClockId clk_id);
    auto get_model_from_id(ClockId id) const ->
        optional<std::reference_wrapper<const IdentifiedModel>>;
    void set_meas_model_from_pairs(const Eigen::MatrixX2i& meas_pairs,
        const Vector& R_diag);
};

Prediction predict(const Model& model, const State& state);
Prediction seq_blk_predict(const Model& model, const State& state,
    int meas_row);
Innovations compute_innovations(const Model& model, const Prediction& pred,
    const Vector& meas);
Innovations compute_innovations(const Model& model, const Prediction& pred,
    Scalar meas, int meas_row);
Correction correct(const Model& model, const Prediction& pred,
    const Innovations& innov); 

} // end namespace KF
} // end namespace tmon

// Copyright (C) 2021, Lawrence Livermore National Security, LLC.
//  All rights reserved. LLNL-CODE-837067
//
// The Department of Homeland Security sponsored the production of this
//  material under DOE Contract Number DE-AC52-07N427344 for the management
//  and operation of Lawrence Livermore National Laboratory. Contract no.
//  DE-AC52-07NA27344 is between the U.S. Department of Energy (DOE) and
//  Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
//  See license for disclaimers, notice of U.S. Government Rights and license
//  terms and conditions.

#endif // KF_H_

