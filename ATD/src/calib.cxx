#include "calib.h"
#include <algorithm>
#include <type_traits>
#include <boost/math/constants/constants.hpp>
#include <Eigen/unsupported/Eigen/NonLinearOptimization>
#include "param.h"

// Calibrator
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
// Returns the log of the argument if positive, and -inf otherwise
template <typename T,
    typename = std::enable_if_t<std::numeric_limits<T>::has_infinity, void>>
T logpos(T x)
{
    static constexpr T neg_inf = -std::numeric_limits<T>::infinity();
    return (x <= 0) ? neg_inf : std::log(x);
}

template <typename Derived>
auto logpos(const Eigen::MatrixBase<Derived>& v)
{
    return v.unaryExpr([](const auto& x){ return logpos(x); }).eval();
}

template <typename Derived>
bool has_nan(const Eigen::MatrixBase<Derived>& x)
{
    return x.unaryExpr(
        [](const auto& elt){ return std::isnan(elt); }).any();
}

template <typename Fn, typename ScalarType,
    int NumInputs = Eigen::Dynamic, int NumFvals = Eigen::Dynamic>
struct LMFunctor
{
    using Index = Eigen::Index;
    using Scalar = ScalarType;
    using InputType = Eigen::Matrix<Scalar, NumInputs, 1>;
    using ValueType = Eigen::Matrix<Scalar, NumFvals, 1>;
    using JacobianType = Eigen::Matrix<Scalar, NumFvals, NumInputs>;
    static constexpr int InputsAtCompileTime = NumInputs;
    static constexpr int ValuesAtCompileTime = NumFvals;
    LMFunctor(Fn&& f)
        : fn_(std::move(f)), num_vals_{NumFvals}
    {
    }
    LMFunctor(Fn&& f, Index /*num_inputs*/, Index num_vals)
        : fn_(std::move(f)), num_vals_{num_vals}
    {
    }
    Index values() const noexcept { return num_vals_; }
    int operator()(const InputType& x, ValueType& fv) const
    {
        fv = fn_(x);
        return 0; // <0 aborts early
    }

  private:
    Fn fn_;
    Index num_vals_;
};

// Minimize a nonlinear scalar-valued function of a vector (f : R^n -> R)
//  via the Levenberg-Marquardt algorithm as implemented in the Eigen library
template <typename Functor, typename Scalar, int Rows>
auto minimize_nonlin(Functor&& obj_fn, Eigen::Matrix<Scalar, Rows, 1>& x) ->
    Eigen::LevenbergMarquardtSpace::Status
{
    Functor local_obj_fn = std::move(obj_fn);
    Eigen::NumericalDiff<Functor> numdiff_obj_fn{local_obj_fn};
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<Functor>> lm{numdiff_obj_fn};
    // Customize tolerances and termination parameters (e.g., # function evals)
    static constexpr double tol = 1e-30;
    assert(tol > 0);
    lm.parameters.maxfev = 100000;
    lm.parameters.ftol = tol;
    lm.parameters.xtol = tol;
    auto lm_status = lm.minimize(x);
    return lm_status;
}

struct MinimizeNonnegStatus
{
    Eigen::LevenbergMarquardtSpace::Status status;

    explicit operator bool() const noexcept
    {
        bool normal_termination =
            (status != Eigen::LevenbergMarquardtSpace::NotStarted) &&
            (status != Eigen::LevenbergMarquardtSpace::Running) &&
            (status !=
                Eigen::LevenbergMarquardtSpace::ImproperInputParameters) &&
            (status !=
                Eigen::LevenbergMarquardtSpace::TooManyFunctionEvaluation) &&
            (status != Eigen::LevenbergMarquardtSpace::UserAsked);
        return normal_termination;
    }
};

std::string describe(const MinimizeNonnegStatus& x)
{
    using namespace Eigen::LevenbergMarquardtSpace;
    switch (x.status)
    {
        case Status::NotStarted: return "not started";
        case Status::Running: return "running";
        case Status::ImproperInputParameters: return "bad input";
        case Status::RelativeReductionTooSmall: return "rel. red. small";
        case Status::RelativeErrorTooSmall: return "rel. err. small";
        case Status::RelativeErrorAndReductionTooSmall: return "too small";
        case Status::CosinusTooSmall: return "cos too small";
        case Status::TooManyFunctionEvaluation: return "too many feval";
        case Status::FtolTooSmall: return "ftol small";
        case Status::XtolTooSmall: return "xtol small";
        case Status::GtolTooSmall: return "gtol small";
        case Status::UserAsked: return "user req.";
    }
    return "<UNKNOWN>";
}

// Minimize ||A*x - b|| subject to A*x > 0 and x > 0 (componentwise)
//  using the Levenberg-Marquardt optimizer
template <typename Derived1, typename Derived2, typename Scalar, int Rows>
[[nodiscard]] auto minimize_nonneg(const Eigen::MatrixBase<Derived1>& A,
    const Eigen::MatrixBase<Derived2>& b, Eigen::Matrix<Scalar, Rows, 1>& x)
    -> MinimizeNonnegStatus
{
    assert(A.rows() == b.rows());
    assert(b.cols() == 1);
    assert(A.cols() == x.rows());
    auto obj_fn = [&A, &b](const auto& x){ return (logpos(A * x.cwiseAbs()) -
        logpos(b)).cwiseAbs(); };
    LMFunctor<decltype(obj_fn), Scalar> functor{std::move(obj_fn), -1,
        A.rows()};
    auto status = minimize_nonlin(std::move(functor), x);
    x = x.cwiseAbs();
    return {status};
}

template <typename RandomAccessContainer>
auto median(RandomAccessContainer cont)
    -> typename RandomAccessContainer::value_type
{
    using std::begin;
    using std::end;
    auto middle_iter = std::next(begin(cont), cont.size() / 2);
    std::nth_element(begin(cont), middle_iter, end(cont));
    return *middle_iter;
}

} // end local namespace

namespace tmon
{

Calibration::Calibration(const GaussMarkovModel& gm_model)
    : GaussMarkovModel{gm_model}
{
}

Calibrator::Calibrator(const ProgState& prog_state, Calibration default_cal)
    : prog_state_{prog_state}, phase_dev_est_{}, sample_interval_{},
        avar_est_{},
        tau_mults_{prog_state.get_opt_as<VectorParam<int>>("cal.tau_mults")},
        tau0_{-1}, default_cal_{std::move(default_cal)}, reference_clk_id_{}
{
}

// Retrieves the fundamental measurement interval (derived as the median of the
//  sample intervals observed)
double Calibrator::get_tau0() const
{
    assert(!sample_interval_.empty());
    auto median_sample_interval = ::median(sample_interval_);
    using std::cbegin;
    using std::cend;
    auto min_sample_iter = std::min_element(cbegin(sample_interval_),
        cend(sample_interval_));
    auto min_vs_median_tol = 0.1 * median_sample_interval;
    if (median_sample_interval - *min_sample_iter > min_vs_median_tol)
    {
        TM_LOG(warning) << "When calibrating, median sample interval assumed "
            "to represent fundamental sampling interval exceeds tolerance vs. "
            "minimum";
    }
    return median_sample_interval;
}

Calibration Calibrator::get_cal(const Calibration& prior) const
{
    assert(has_cal());
    Calibration ret{default_cal_};

    // Estimate linear trend (syntonization error)
    int N = phase_dev_est_.size();
    assert(N > 1);
    auto phase_dev_est_map = Eigen::VectorXd::Map(phase_dev_est_.data(), N);
    Eigen::VectorXd phase_diff = phase_dev_est_map.tail(N - 1) -
        phase_dev_est_map.head(N - 1);
    double synton_err_est = phase_diff.mean() / tau0_;
    ret.mu[0] = synton_err_est;

    // Estimate diffusion coefficients (WFM and RWFM process noise components)
    const Eigen::Index M{tau_mults_.size()}; // # of integration periods (tau)
    const Eigen::Index K{2}; // number of powers of tau to regress against
    Eigen::MatrixXd A{M, K};
    Eigen::VectorXd b = avar_est_;
    Eigen::VectorXd h{K};
    Eigen::VectorXi tau_pows{K};
    tau_pows << 1, -1; // model includes tau (RWFM) and tau^-1 (WFM)
    const double scaleA{4 * boost::math::double_constants::pi_sqr_div_six};
    const double scaleC{0.5};
    for (int i = 0; i < K; ++i)
        A.col(i) = (tau0_ * tau_mults_.cast<double>()).array().pow(tau_pows[i]);

    bool fix_wfm{false};
    const double min_tau_wfm{prog_state_.get().get_opt_as<double>(
        "cal.min_tau_wfm")};
    if (tau0_ >= min_tau_wfm)
    {
        // Unable to accurately estimate WFM diffusion coefficient without
        //  fast enough sampling
        fix_wfm = true;
        TM_LOG(info) << "Skipping estimation of WFM during calibration (tau0 "
            << tau0_ << " >= min. " << min_tau_wfm << ')';
        b -= A.col(K - 1) * (prior.q(0) / scaleC);
        A.conservativeResize(M, K - 1);
        h.resize(K - 1);
    }
    assert(A.rows() == b.size());
    MinimizeNonnegStatus min_status = ::minimize_nonneg(A, b, h);
    if (!min_status)
    {
        throw CalibrationError("minimization failed; error: " +
                describe(min_status));
    }
    if (fix_wfm)
    {
        h.conservativeResize(K);
        h[K - 1] = prior.q(0) / scaleC;
    }
    ret.q[0] = h[1] * scaleC;     // WFM diffusion coefficient
    ret.q[1] = 3 * h[0] * scaleA; // RWFM diffusion coefficient
    TM_LOG(debug) << "Calibration conducted optimization with A=" << A << ", b="
        << b.transpose() << " (cf. AVAR=" << avar_est_.transpose() << "), tau0="
        << tau0_ << " with result h=" << h.transpose();
    if (has_nan(ret.q))
        throw CalibrationError("minimization led to NaN value");
    assert(ret.q.minCoeff() >= 0);

    return ret;
}

optional<ClockId> Calibrator::get_reference_clock_id() const
{
    return reference_clk_id_;
}

bool Calibrator::has_cal() const noexcept
{
    return has_cal_;
}

void Calibrator::update(ClockId reporting_clk_id, picoseconds elapsed,
    picoseconds phase_dev_sample)
{
    assert(!reference_clk_id_ || (*reference_clk_id_ == reporting_clk_id));
    if (!reference_clk_id_)
        reference_clk_id_ = reporting_clk_id;
    sample_interval_.push_back(std::chrono::duration<double>{elapsed}.count());
    phase_dev_est_.push_back(
        std::chrono::duration<double>{phase_dev_sample}.count());
    int N = phase_dev_est_.size();
    auto phase_dev_est_map = Eigen::VectorXd::Map(phase_dev_est_.data(), N);
    int min_len = 2 * tau_mults_.maxCoeff() + 1;
    if (N < min_len)
    {
        // can't process yet
        has_cal_ = false;
    }
    else if (N == min_len)
    {
        // Estimate the fundamental measurement interval
        tau0_ = get_tau0();

        // Bootstrap via batch method for subsequent stepwise 
        avar_est_ = avar_overlapping(phase_dev_est_map, tau_mults_, tau0_);
        has_cal_ = true;
    }
    else
    {
        avar_est_ = update_avar_overlapping(avar_est_, phase_dev_est_map,
            tau_mults_, tau0_);
    }
}

} // end namespace tmon

////////////////////////////////////////////////////////////////////////////
// Unit Tests

#ifdef UNIT_TEST
#include <iostream>
#include <Eigen/QR>
#include "utility.h"

// Tests the implementation of the overlapping Allan variance computation
//  vis-a-vis the 1000 data point set in Riley, _Handbook of Frequency Stability
//  Analysis_, NIST, produced by a linear congruential generator
void test_avar_overlap_riley()
{
    static constexpr int riley_dataset_len = 1000;
    std::uint64_t riley_dataset[riley_dataset_len];
    riley_dataset[0] = 1'234'567'890;
    static constexpr std::uint64_t modulus = 2'147'483'647;
    for (int i = 1; i < riley_dataset_len; ++i)
        riley_dataset[i] = (16807 * riley_dataset[i - 1]) % modulus;
    Eigen::VectorXd riley_freq_dev{
        Eigen::Matrix<std::uint64_t, Eigen::Dynamic, 1>::Map(
        riley_dataset, riley_dataset_len, 1).cast<double>()};
    riley_freq_dev /= modulus; // normalize to [0, 1)
    static constexpr double riley_freq_dev_mean = 0.4897745;
    double riley_mean_diff_mag = std::abs(riley_freq_dev.mean() -
        riley_freq_dev_mean);
    std::cerr << "Diff. versus expected mean (1000-pt Riley): "
        << riley_mean_diff_mag << std::endl;
    assert(riley_mean_diff_mag < 1e-6);

    // Convert frequency data to phase data (with assumed fundamental
    //  measurement interval tau0 = 1 second)
    double tau0{1};
    Eigen::VectorXd riley_phase_dev{riley_dataset_len + 1};
    riley_phase_dev[0] = 0;
    for (int i = 1; i < riley_dataset_len + 1; ++i)
        riley_phase_dev[i] = riley_freq_dev[i - 1] + riley_phase_dev[i - 1];
    static constexpr double riley_phase_dev_mean = 244.3468632;
    double riley_phase_mean_diff_mag = std::abs(riley_phase_dev.mean() -
        riley_phase_dev_mean);
    std::cerr << "Diff. versus expected phase mean (1000-pt Riley): "
        << riley_phase_mean_diff_mag << std::endl;
    assert(riley_phase_mean_diff_mag < 1e-6);

    Eigen::VectorXi tau_mults{3};
    tau_mults << 1, 10, 100;

    Eigen::VectorXd avar_over_expected{3};
    // "Overlap Allan Dev" from Table 31 of Riley:
    avar_over_expected << 2.922319e-1, 9.159953e-2, 3.241343e-2;
    avar_over_expected = avar_over_expected.cwiseAbs2();

    auto avar_over_at_once = avar_overlapping(riley_phase_dev, tau_mults, tau0);
    std::cerr << "Riley overlapping AVAR: " << avar_over_at_once
        << "\n vs. expected: " << avar_over_expected << std::endl;
    auto avar_diff_norm = (avar_over_at_once - avar_over_expected).norm();
    double diff_norm_pct = (avar_diff_norm / avar_over_expected.norm() * 100);
    std::cerr << "||Riley(expected) - AVAR(over)|| = " << avar_diff_norm << " ("
        << diff_norm_pct << "%)" << std::endl;
    assert(diff_norm_pct < 1e-4);

    // Repeat the test for the stepwise variant, bootstrapped with an initial
    //  estimate from the batch variant
    Eigen::VectorXd avar_over_stepwise =
        Eigen::VectorXd::Zero(tau_mults.size());
    auto min_len_for_stepwise = 2 * tau_mults.maxCoeff() + 1;
    avar_over_stepwise = avar_overlapping(
        riley_phase_dev.topRows(min_len_for_stepwise), tau_mults, tau0);
    for (int i = min_len_for_stepwise + 1; i <= riley_phase_dev.size(); ++i)
    {
        avar_over_stepwise = update_avar_overlapping(avar_over_stepwise,
            riley_phase_dev.topRows(i), tau_mults, tau0);
    }
    std::cerr << "Riley overlapping (stepwise) AVAR: " << avar_over_stepwise
        << std::endl;
    auto avar_diff_norm_step = (avar_over_stepwise - avar_over_expected).norm();
    double diff_norm_pct_step = (avar_diff_norm_step /
        avar_over_expected.norm() * 100);
    std::cerr << "||Riley(expected) - AVAR(over-stepwise)|| = "
        << avar_diff_norm_step << " ("
        << diff_norm_pct_step << "%)" << std::endl;
    assert(diff_norm_pct_step < 1e-4);
}

void test_avar_overlap_update()
{
    Stopwatch sw_total{"Total", std::cerr};
    Eigen::VectorXd rand_phase_dev = Eigen::VectorXd::Random(86400);
    Eigen::VectorXd tau_mults{5};
    tau_mults << 1, 2, 3, 4, 10;
    double tau0{0.1};

    Stopwatch sw_at_once{"At-once", std::cerr};
    auto avar_over_at_once = avar_overlapping(rand_phase_dev, tau_mults, tau0);
    sw_at_once.report();
    std::cerr << "all at once: " << avar_over_at_once << std::endl;

    Eigen::VectorXd avar_over_stepwise =
        Eigen::VectorXd::Zero(tau_mults.size());
    auto min_len_for_stepwise = 2 * tau_mults.maxCoeff() + 1;
    Stopwatch sw_stepwise{"Stepwise", std::cerr};
    avar_over_stepwise = avar_overlapping(
        rand_phase_dev.topRows(min_len_for_stepwise), tau_mults, tau0);
    //std::cerr << "stepwise bootstrap: " << avar_over_stepwise << std::endl;
    for (int i = min_len_for_stepwise + 1; i <= rand_phase_dev.size(); ++i)
    {
        avar_over_stepwise = update_avar_overlapping(avar_over_stepwise,
            rand_phase_dev.topRows(i), tau_mults, tau0);
    }
    sw_stepwise.report();
    std::cerr << "stepwise: " << avar_over_stepwise << std::endl;
    double diff_norm = (avar_over_at_once - avar_over_stepwise).norm();
    double diff_norm_pct = (diff_norm / avar_over_at_once.norm() * 100);
    std::cerr << "||At-once - stepwise|| = " << diff_norm << " ("
        << diff_norm_pct << "%)" << std::endl;
    assert(diff_norm_pct < 1e-8);
}

void test_min_nonneg()
{
    Eigen::MatrixXd A{14, 5};
    Eigen::VectorXd b{14};

    A << 0.1000000000004, 1, 9.999999999964, 99.99999999927, 999.99999998909,
        0.2000000000007, 1, 4.999999999982, 24.99999999982, 124.9999999986,
        0.3000000000011, 1, 3.333333333321, 11.11111111103, 37.03703703663,
        0.4000000000015, 1, 2.499999999991, 6.249999999955, 15.62499999983,
        0.5000000000018, 1, 1.999999999993, 3.999999999971, 7.999999999913,
        0.6000000000022, 1, 1.666666666661, 2.777777777758, 4.629629629579,
        0.7000000000025, 1, 1.428571428566, 2.040816326516, 2.915451895012,
        0.8000000000029, 1, 1.249999999995, 1.562499999989, 1.953124999979,
        0.9000000000033, 1, 1.111111111107, 1.234567901226, 1.371742112468,
        1.0000000000036, 1, 0.9999999999964, 0.9999999999927, 0.9999999999891,
        3.0000000000109, 1, 0.3333333333321, 0.1111111111103, 0.03703703703663,
        5.0000000000182, 1, 0.1999999999993, 0.03999999999971, 0.00799999999991,
        7.0000000000255, 1, 0.1428571428566, 0.02040816326516, 0.00291545189501,
        9.0000000000327, 1, 0.1111111111107, 0.01234567901226, 0.00137174211247;

    b << 9.67955891161431e-21,
        1.67901124143775e-20,
        2.46520343851164e-20,
        3.26348584011273e-20,
        4.06678777337762e-20,
        4.87100118965361e-20,
        5.67380115590790e-20,
        6.47442399551423e-20,
        7.27297017051795e-20,
        8.06999374045679e-20,
        2.36373067928037e-19,
        3.81310557226640e-19,
        5.23921305633203e-19,
        6.66061403463188e-19;

    Eigen::VectorXd x0{5};
    x0 = A.colPivHouseholderQr().solve(b);
    x0 = x0.cwiseAbs();
    std::cerr << "||Ax - b||_2 initial min: " << x0 << std::endl;

    Eigen::VectorXd x{5};
    x = x0;
    bool ok = ::minimize_nonneg(A, b, x);
    std::cerr << "||Ax - b|| constrained min OK: " << ok << std::endl;
    std::cerr << "argmin_x ||Ax - b||: " << x << std::endl;
    double inf_norm = (A * x - b).lpNorm<Eigen::Infinity>();
    std::cerr << "||Ax - b||_inf = " << inf_norm << std::endl;
    assert(inf_norm < 1e-15);

    std::cerr << "Repeating minimization with columns 1 & 3 only" << std::endl;
    Eigen::MatrixXd A2{14, 2};
    Eigen::VectorXd x2{2};
    A2.col(0) = A.col(0);
    A2.col(1) = A.col(2);
    x2 << x0(0), x0(2);
    ok = ::minimize_nonneg(A2, b, x2);
    std::cerr << "||Ax - b|| constrained min OK: " << ok << std::endl;
    std::cerr << "argmin_x ||Ax - b||: " << x2 << std::endl;
    inf_norm = (A2 * x2 - b).lpNorm<Eigen::Infinity>();
    std::cerr << "||Ax - b||_inf = " << inf_norm << std::endl;
    assert(inf_norm < 1e-15);
    Eigen::Vector2d x2_expected{2};
    x2_expected << 7.8937e-20, 1.8614e-22;
    double x2_norm_vs_expected = (x2 - x2_expected).norm();
    std::cerr << "||x - expected||: " << x2_norm_vs_expected << std::endl;
    assert(x2_norm_vs_expected < 1e-20);
}

int main()
{
    test_avar_overlap_riley();
    test_avar_overlap_update();
    test_min_nonneg();
}
#endif // UNIT_TEST

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

