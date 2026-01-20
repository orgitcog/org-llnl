#include "kf.h"
#include <algorithm>
#include <fstream>
#include <Eigen/SVD>

// Kalman Filter
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

namespace
{
    template <typename Derived>
    bool is_psd(const Eigen::MatrixBase<Derived>& m)
    {
        if (m.rows() == 0)
            return true; // vacuously true
        auto min_sv = m.jacobiSvd().singularValues().minCoeff();
        bool nonneg_svs = (min_sv >= 0);
        double sym_tol = 1e-20;
        bool is_sym = (m.rows() == m.cols()) &&
            ((m - m.transpose()).norm() < sym_tol);
        return (is_sym && nonneg_svs);
    }

    template <typename Derived>
    bool has_nan(const Eigen::MatrixBase<Derived>& x)
    {
        return x.unaryExpr(
            [](const auto& elt){ return std::isnan(elt); }).any();
    }
} // end local namespace

namespace tmon
{
namespace KF
{

bool State::valid() const
{
    return (P.rows() == P.cols()) && (x.rows() == P.rows());
}

bool Model::valid() const
{
    auto M = C.rows();
    auto N = x0.rows();

    bool conformable_dims = (P0.rows() == N) && (P0.cols() == N) &&
        (A.rows() == N) && (A.cols() == N) && (C.cols() == N) &&
        (Q.rows() == N) && (Q.cols() == N) &&
        (R.rows() == M) && (R.cols() == M);
    bool valid_P0 = (P0.size() == 0) || is_psd(P0);
    return (conformable_dims && valid_P0);
}

Filter::Filter(Model model)
    : model_{std::move(model)}, state_{model_.x0, model_.P0}
{
    assert(valid() && "KF initial model/state invalid");
}

Model& Filter::model()
{
    return model_;
}

const Model& Filter::model() const
{
    return model_;
}

ModelDims Filter::get_model_dims() const
{
    ModelDims ret;
    ret.M = model_.C.rows();
    ret.N = model_.A.rows();
    return ret;
}

void Filter::reset()
{
    state_ = State{model_.x0, model_.P0};
}

void Filter::set_meas_model(Matrix C, const Vector& R_diag)
{
    assert(C.rows() == R_diag.rows());
    model_.C = std::move(C);
    model_.R = R_diag.asDiagonal();
    assert(valid() && "Invalid KF model when setting meas. model");
}

void Filter::set_model(Model m)
{
    model_ = std::move(m);
    // Must reset state to ensure conformability
    reset();
    assert(valid() && "Invalid KF model set");
}

const State& Filter::state() const
{
    return state_;
}

bool Filter::valid() const
{
    bool conformable = (state_.x.rows() == model_.x0.rows());
    return model_.valid() && state_.valid() && conformable;
}

Innovations Filter::update(const Vector& meas)
{
    BOOST_LOG_FUNCTION();
    assert((meas.rows() == model_.C.rows()) &&
        "Measurement not dimensionally-compatible with KF model");

    Prediction pred = predict(model_, state_);

    Innovations innov = compute_innovations(model_, pred, meas);

    Correction corr = correct(model_, pred, innov);

    // Commit corrections to state
    state_.x = corr.x;
    state_.P = corr.P;

    return innov;
}

void store_summary(std::ofstream& ofs, const Prediction& pred,
    const Innovations& innov, const Correction& corr, int meas_row)
{
    ofs << pred.P.diagonal().transpose() << " | "
        << pred.x.transpose() << " | "
        << corr.P.diagonal().transpose() << " | "
        << corr.x.transpose() << " | "
        << innov.K.col(meas_row).transpose() << " | "
        << innov.e.transpose() << " | "
        << innov.e_cov(0, 0) << "\n";
}

Innovations Filter::update(Scalar meas, int meas_row)
{
    BOOST_LOG_FUNCTION();
    assert((meas_row >= 0) && (meas_row <= model_.C.rows()));

    Prediction pred = seq_blk_predict(model_, state_, meas_row);

    Innovations innov = compute_innovations(model_, pred, meas, meas_row);

    Correction corr = correct(model_, pred, innov);

    // To debug the filter, this checkpoint logging can be enabled:
    //static std::ofstream ofs("kf-debug.dump");
    //store_summary(ofs, pred, innov, corr, meas_row);

    // Commit corrections to state
    state_.x = corr.x;
    state_.P = corr.P;

    return innov;
}

Prediction predict(const Model& model, const State& state)
{
    BOOST_LOG_FUNCTION();
    Prediction pred;

    // Compute state and measurement predictions
    pred.x = model.A * state.x + model.B * model.u;
    pred.P = model.A * state.P * model.A.transpose() + model.Q;
    pred.y = model.C * pred.x;

    return pred;
}

// FIXME: Assumes block-diagonal A, B and fixed block sizes
Prediction seq_blk_predict(const Model& model, const State& state, int meas_row)
{
    BOOST_LOG_FUNCTION();
    Prediction pred;

    pred.x = state.x;
    pred.P = state.P;

    int meas_idx1 = -1;
    int meas_idx2 = -1;
    int N = model.C.cols();
    assert((model.C.row(meas_row).array().abs() > 0).count() == 2);
    model.C.row(meas_row).cwiseAbs().maxCoeff(&meas_idx1);
    assert((meas_idx1 >= 0) && (meas_idx1 < N - 1));
    model.C.row(meas_row).tail(N - 1 - meas_idx1).cwiseAbs().maxCoeff(
        &meas_idx2);
    meas_idx2 += meas_idx1 + 1;
    assert((meas_idx2 > meas_idx1) && (meas_idx2 < N));

    // Compute state and measurement predictions (only updating the components
    //  that are related to the current measurement)
    int N_single = 3; // Fixed size of single state block
    assert(N % N_single == 0);
    pred.x.segment(meas_idx1, N_single) =
        model.A.middleRows(meas_idx1, N_single) * state.x +
        model.B.middleRows(meas_idx1, N_single) * model.u;
    pred.x.segment(meas_idx2, N_single) =
        model.A.middleRows(meas_idx2, N_single) * state.x +
        model.B.middleRows(meas_idx2, N_single) * model.u;

    auto update_P_blk = [&](int i, int j){
        pred.P.block(i, j, N_single, N_single) =
            model.A.block(i, j, N_single, N_single) *
            state.P.block(i, j, N_single, N_single) *
            model.A.block(i, j, N_single, N_single).transpose() +
            model.Q.block(i, j, N_single, N_single); };
    update_P_blk(meas_idx1, meas_idx1);
    update_P_blk(meas_idx1, meas_idx2);
    update_P_blk(meas_idx2, meas_idx1);
    update_P_blk(meas_idx2, meas_idx2);
    pred.y = model.C * pred.x;

    return pred;
}

Innovations compute_innovations(const Model& model, const Prediction& pred,
    const Vector& meas)
{
    BOOST_LOG_FUNCTION();
    assert(meas.rows() == pred.y.rows());

    Innovations innov;

    // Compute innovations
    innov.e = meas - pred.y;
    innov.e_cov = model.C * pred.P * model.C.transpose() + model.R;

    // Compute Kalman gain
    innov.K = pred.P * innov.e_cov.ldlt().solve(model.C).transpose();

    return innov;
}

Innovations compute_innovations(const Model& model, const Prediction& pred,
    Scalar meas, int meas_row)
{
    BOOST_LOG_FUNCTION();
    assert((meas_row >= 0) && (meas_row <= pred.y.rows()));
    assert(meas_row <= model.C.rows());
    assert(meas_row <= model.R.rows());

    // TODO: Make another struct like Innovations but with sparse matrices
    Innovations innov;
    int M = model.C.rows();
    int N = model.C.cols();

    // Compute innovations
    innov.e.setZero(M);
    innov.e[meas_row] = meas - pred.y[meas_row];
    // Innovations covariance is a scalar in this case
    innov.e_cov.setZero(1, 1);
    innov.e_cov << model.C.row(meas_row) * pred.P *
        model.C.row(meas_row).transpose() + model.R(meas_row, meas_row);

    // Compute Kalman gain
    innov.K.setZero(N, M);
    // assert(::is_psd(pred.P));
    innov.K.col(meas_row) = pred.P * model.C.row(meas_row).transpose() /
        innov.e_cov(0, 0);

    return innov;
}

Correction correct(const Model& model, const Prediction& pred,
    const Innovations& innov)
{
    BOOST_LOG_FUNCTION();
    Correction corr;

    assert(!has_nan(innov.K));
    assert(!has_nan(innov.e));

    // Compute corrections
    corr.x = pred.x + innov.K * innov.e;
    auto N = innov.K.rows();
    Matrix I_nn = Matrix::Identity(N, N);
    // Joseph stabilized form of the corrected covariance used for numerical
    //  stability (R. Bucy and P. Joseph, _Filtering for Stochastic Processes
    //  with Applications to Guidance_, Wiley & Sons, 1968); ensures that 
    //  P_corr is symmetric positive definite (provided that it is at the
    //  previous timestep)
    corr.P = (I_nn - innov.K * model.C) * pred.P *
        (I_nn - innov.K * model.C).transpose() +
        innov.K * model.R * innov.K.transpose();

    bool P_has_nan = has_nan(corr.P);
    if (P_has_nan)
    {
        std::ofstream ofs("corr-p-nan.dump");
        ofs << "pred.P: " << pred.P << "\n"
            << "corr.P: " << corr.P << "\n"
            << "model.A: " << model.A << "\n"
            << "model.B: " << model.B << "\n"
            << "model.C: " << model.C << "\n"
            << "model.Q: " << model.Q << "\n"
            << "model.R: " << model.R << "\n"
            << "pred.x: " << pred.x << "\n"
            << "corr.x: " << corr.x << "\n"
            << "innov.e: " << innov.e << "\n"
            << "innov.e_cov: " << innov.e_cov << "\n"
            << "innov.K: " << innov.K << "\n";
    }
    assert(!P_has_nan);

    return corr;
}

ModelBlockRef::ModelBlockRef(Model& m)
    : x0{m.x0}, P0{m.P0}, u{m.u}, A{m.A}, B{m.B}, Q{m.Q}
{
}

ModelBlockRef::ModelBlockRef(Model& m, int N_single, int block_idx)
    : x0{m.x0.middleRows(N_single * block_idx, N_single)},
      P0{m.P0.block(N_single * block_idx, N_single * block_idx, N_single,
          N_single)},
      u{m.u.middleRows(N_single * block_idx, N_single)},
      A{m.A.block(N_single * block_idx, N_single * block_idx, N_single,
          N_single)},
      B{m.B.block(N_single * block_idx, N_single * block_idx, N_single,
          N_single)},
      Q{m.Q.block(N_single * block_idx, N_single * block_idx, N_single,
          N_single)}
{
}

Model make_block_model(const std::vector<IdentifiedModel>& models_with_id,
    const Eigen::MatrixX2i& meas_pairs)
{
    using std::begin;
    using std::end;
    assert(!models_with_id.empty());
    const IdentifiedModel& first_model = models_with_id[0];
    assert(std::all_of(begin(models_with_id), end(models_with_id),
        [&first_model](const auto& x) {
            return (x.A.rows() == first_model.A.rows()) &&
                (x.A.cols() == first_model.A.cols()); }));
    assert(meas_pairs.cols() == 2);
    int num_models = models_with_id.size();
    assert((meas_pairs.minCoeff() >= 0) &&
        (meas_pairs.maxCoeff() < num_models));

    Model ret;
    int M = meas_pairs.rows();
    int N_single = first_model.A.rows();
    int N = N_single * num_models;
    ret.resize_zeroed(M, N);

    for (int i = 0; i < num_models; ++i)
    {
        int Ni = N_single * i;
        const Model& model = models_with_id[i];
        ret.x0.middleRows(Ni, N_single) = model.x0;
        ret.P0.block(Ni, Ni, N_single, N_single) = model.P0;
        ret.u.middleRows(Ni, N_single) = model.u;
        ret.A.block(Ni, Ni, N_single, N_single) = model.A;
        ret.B.block(Ni, Ni, N_single, N_single) = model.B;
        ret.Q.block(Ni, Ni, N_single, N_single) = model.Q;
    }

    // Initialize matrices with measurement-space component
    for (int i = 0; i < M; ++i)
    {
        ret.C(i, N_single * meas_pairs(i, 0)) = -1;
        ret.C(i, N_single * meas_pairs(i, 1)) = 1;
        assert(ret.C.row(i).sum() == 0);
    }
    ret.R = first_model.R;
    return ret;
}

// This resize conserves current values and zeroes out new entries
void Model::resize(int num_meas, int num_states)
{
    int M = num_meas;
    int N = num_states;
    x0.conservativeResizeLike(Vector::Zero(N, 1));
    P0.conservativeResizeLike(Matrix::Zero(N, N));
    u.conservativeResizeLike(Vector::Zero(N, 1));
    A.conservativeResizeLike(Matrix::Zero(N, N));
    B.conservativeResizeLike(Matrix::Zero(N, N));
    C.conservativeResizeLike(Matrix::Zero(M, N));
    Q.conservativeResizeLike(Matrix::Zero(N, N));
    R.conservativeResizeLike(Matrix::Zero(M, M));
}

// This resize is destructive; matrices are zero-initialized upon return
void Model::resize_zeroed(int num_meas, int num_states)
{
    int M = num_meas;
    int N = num_states;
    x0.setZero(N, 1);
    P0.setZero(N, N);
    u.setZero(N, 1);
    A.setZero(N, N);
    B.setZero(N, N);
    C.setZero(M, N);
    Q.setZero(N, N);
    R.setZero(M, M);
}

BlockFilter::BlockFilter()
    : Filter{Model{}}, models_{}
{
}

BlockFilter::BlockFilter(std::vector<IdentifiedModel> models,
        const Eigen::MatrixX2i& meas_pairs)
   : Filter{make_block_model(models, meas_pairs)}, models_{std::move(models)}
{
}

bool BlockFilter::add_model(IdentifiedModel model_with_id)
{
    using std::begin;
    using std::end;

    if (has_model_for(model_with_id.clock_id))
    {
        // Fail if asked to add a model that is already registered
        return false;
    }

    auto N_single = model_with_id.A.rows();
    assert(N_single == model_with_id.A.cols());
    if (!models_.empty())
    {
        // Must be conformable with models already registered, if any
        assert(models_[0].A.rows() == N_single);
    }

    Model new_model = model();
    auto M = new_model.C.rows();
    auto N_old = new_model.A.rows();
    auto N = N_old + N_single;
    new_model.resize(M, N);

    new_model.x0.middleRows(N_old, N_single) = model_with_id.x0;
    new_model.P0.block(N_old, N_old, N_single, N_single) = model_with_id.P0;
    new_model.u.middleRows(N_old, N_single) = model_with_id.u;
    new_model.A.block(N_old, N_old, N_single, N_single) = model_with_id.A;
    new_model.B.block(N_old, N_old, N_single, N_single) = model_with_id.B;
    new_model.Q.block(N_old, N_old, N_single, N_single) = model_with_id.Q;

    // Note: No updates to measurement-space (C or R) matrices here

    // Update list of individual models as well as the overall block model
    models_.push_back(model_with_id);
    // Setting the model will automatically reset the state
    set_model(new_model);
    return true;
}

ModelBlockRef BlockFilter::get_model_block_ref(ClockId clk_id)
{
    int clk_idx = get_idx_for(clk_id);
    assert(clk_idx >= 0);
    assert(!models_.empty());
    int N_single = models_[0].A.rows();
    return ModelBlockRef{model(), N_single, clk_idx};
}

bool BlockFilter::remove_model(ClockId clk_id)
{
    if (!has_model_for(clk_id))
    {
        // Fail if asked to remove a model that is not already registered
        return false;
    }

    // TODO: Consider any additional steps needed here
    // For now, just invalidate the clock ID, so it won't show up as being
    //  in the model, but won't invalidate the row positions in the joint model
    using std::begin;
    using std::end;
    auto match_result = std::find_if(begin(models_), end(models_),
        [clk_id](const auto& x) { return (x.clock_id == clk_id); });
    assert(match_result != end(models_));
    match_result->clock_id = -1; // mark as invalid

    return true;
}

// Updates measurement-space matrices (C and R) to include a new measurement
//  pair (assumed, and here verified, to not already be in the model)
int BlockFilter::add_new_meas_pair(ClockId id1, ClockId id2, Scalar new_R_entry)
{
    int id1_idx = get_idx_for(id1);
    int id2_idx = get_idx_for(id2);
    assert(((id1_idx >= 0) && (id2_idx >= 0)) &&
        "Adding new meas. pair for clocks that are not both registered");
    assert((get_meas_pair_row(id1, id2) < 0) && "Meas pair already exists");
    int new_M = model().C.rows() + 1;
    int N = model().C.cols();
    Matrix C{new_M, N};
    assert(!models_.empty());
    int N_single = models_[0].A.rows();
    int id1_nz_pos = N_single * id1_idx;
    int id2_nz_pos = N_single * id2_idx;
    Vector new_C_row;
    new_C_row.setZero(N);
    new_C_row[id1_nz_pos] = -1;
    new_C_row[id2_nz_pos] = 1;
    C << model().C, new_C_row.transpose();
    Vector R_diag{new_M};
    R_diag << model().R.diagonal(), new_R_entry;

    set_meas_model(std::move(C), R_diag);
    return new_M - 1;
}

ModelDims BlockFilter::get_single_model_dims() const
{
    assert(!models_.empty());
    ModelDims ret;
    ret.M = models_.front().C.rows();
    ret.N = models_.front().A.rows();
    return ret;
}

int BlockFilter::get_meas_pair_row(ClockId id1, ClockId id2) const noexcept
{
    // Find measurement model indices corresponding to passed IDs in meas. pair
    int id1_idx = get_idx_for(id1);
    int id2_idx = get_idx_for(id2);
    if ((id1_idx < 0) || (id2_idx < 0))
        return -1;
    assert(!models_.empty());
    int N_single = models_[0].A.rows();
    int id1_nz_pos = N_single * id1_idx;
    int id2_nz_pos = N_single * id2_idx;
    auto& C = model().C;
    assert((id1_nz_pos < C.cols()) && (id2_nz_pos < C.cols()));
    for (int r = 0; r < C.rows(); ++r)
    {
        if ((C(r, id1_nz_pos) == -1) && (C(r, id2_nz_pos) == 1))
            return r;
    }
    return -1;
}

int BlockFilter::get_idx_for(ClockId id) const noexcept
{
    using std::begin;
    using std::end;
    auto match_result = std::find_if(begin(models_), end(models_),
        [id](const auto& x) { return (x.clock_id == id); });
    if (match_result == end(models_))
        return -1;
    return std::distance(begin(models_), match_result);
}

bool BlockFilter::has_model_for(ClockId id) const noexcept
{
    using std::begin;
    using std::end;
    auto match_result = std::find_if(begin(models_), end(models_),
        [id](const auto& x) { return (x.clock_id == id); });
    return (match_result != end(models_));
}

auto BlockFilter::get_model_from_id(ClockId id) const ->
    optional<std::reference_wrapper<const IdentifiedModel>>
{
    using std::begin;
    using std::end;
    auto match_result = std::find_if(begin(models_), end(models_),
        [id](const auto& x) { return (x.clock_id == id); });
    if (match_result == end(models_))
        return {};
    return *match_result;
}

void BlockFilter::set_meas_model_from_pairs(const Eigen::MatrixX2i& meas_pairs,
    const Vector& R_diag)
{
    assert(meas_pairs.unaryExpr([this](const auto& id){
        return this->has_model_for(id); }).all());
    int num_models = models_.size();
    int M = meas_pairs.rows();
    if (models_.empty())
        return; // nop without any clock models
    int N_single = models_[0].A.rows();
    int N = N_single * models_.size();
    Matrix new_C{M, N};
    new_C.fill(0);

    for (int i = 0; i < M; ++i)
    {
        int meas_model_idx1 = -1;
        int meas_model_idx2 = -1;
        for (std::size_t j = 0; j < models_.size(); ++j)
        {
            if (models_[j].clock_id == meas_pairs(i, 0))
                meas_model_idx1 = j;
            if (models_[j].clock_id == meas_pairs(i, 1))
                meas_model_idx2 = j;
        }
        assert((meas_model_idx1 >= 0) && (meas_model_idx1 < num_models));
        assert((meas_model_idx2 >= 0) && (meas_model_idx2 < num_models));
        new_C(i, N_single * meas_model_idx1) = -1;
        new_C(i, N_single * meas_model_idx2) = 1;
        assert(new_C.row(i).sum() == 0);
    }
    set_meas_model(new_C, R_diag);
}

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

