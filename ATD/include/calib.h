#ifndef CALIB_H_
#define CALIB_H_

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>
#include <Eigen/Dense>
#include "common.h"
#include "gm_model.h"
#include "prog_state.h"

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

namespace tmon
{

struct Calibration : public GaussMarkovModel
{
    using GaussMarkovModel::GaussMarkovModel;
    Calibration(const GaussMarkovModel& gm_model);
    Calibration(const Calibration&) = default;
    Calibration(Calibration&&) = default;
    Calibration& operator=(Calibration&&) = default;
};

struct CalibrationError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class Calibrator
{
  public:
    Calibrator(const ProgState& prog_state, Calibration default_cal);
    // Disabling copy and assignment operators; class could be copyable, but
    //  any copies are likely unintentional and unnecessary
    Calibrator(const Calibrator&) = delete;
    Calibrator& operator=(const Calibrator&) = delete;
    Calibrator(Calibrator&&) = default;
    Calibrator& operator=(Calibrator&&) = default;

    Calibration get_cal(const Calibration& prior) const;
    optional<ClockId> get_reference_clock_id() const;
    bool has_cal() const noexcept;
    // Update the data for the calibration for this clock based upon a sample
    //  of its phase-time deviation over an interval (the elapsed time reported
    //  by its measurement system)
    void update(ClockId reporting_clk_id, picoseconds elapsed,
        picoseconds phase_dev_sample);

  private:
    std::reference_wrapper<const ProgState> prog_state_;
    // Estimate of the phase-time deviations of the associated time source
    //  in seconds (consistent measurement interval, tau0, assumed)
    std::vector<double> phase_dev_est_;
    // Duration of the sampling interval (in seconds) corresponding to the
    //  entry at the same index in phase_dev_est_
    std::vector<double> sample_interval_;
    // Estimate of the (overlapping) Allan variance
    Eigen::VectorXd avar_est_;
    // Integration periods (tau) as multiples of the fundamental measurement
    //  interval (tau0)
    Eigen::VectorXi tau_mults_;
    // Fundamental measurement interval (estimated by get_tau0 once sufficient
    //  samples have been obtained)
    double tau0_;
    // Default calibration (required as it may provide components that are
    //  not easily observable/calculable during the calibration process
    //  performed by this class)
    Calibration default_cal_;
    // Identifier of the reference clock (if any) used to calibrate the clock
    //  of interest for this calibrator
    optional<ClockId> reference_clk_id_;
    // True if sufficient data has been collected to perform a calibration
    //  (may not already have been computed and memoized)
    bool has_cal_ = false;

    LoggerType& get_logger() const { return prog_state_.get().get_logger(); }
    // Retrieves the fundamental measurement interval (derived as the median
    //  of the sample intervals observed)
    double get_tau0() const;
};

// Computes the overlapping Allan variance
template <typename Derived, typename TauMultCont, typename TauScalar>
auto avar_overlapping(const Eigen::MatrixBase<Derived>& phase_dev,
    const TauMultCont& tau_mults, TauScalar tau0)
{
    using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    Vector avar_ret = Vector::Zero(tau_mults.size());
    auto N = phase_dev.size();
    assert(tau0 > 0);
    for (int t = 0; t < tau_mults.size(); ++t)
    {
        auto m = tau_mults[t];
        auto tau = m * tau0;
        Scalar sum{0};
        assert(N > 2 * m);
        for (int i = 0; i < N - 2 * m; ++i)
        {
            sum += std::pow(phase_dev[i + 2 * m] - 2 * phase_dev[i + m] +
                phase_dev[i], 2);
        }
        avar_ret[t] = sum / (2 * (N - 2 * m) * std::pow(tau, 2));
    }
    return avar_ret;
}

// Updates the overlapping Allan variance based on the preceding timestep
template <typename Derived, typename Derived2, typename TauMultCont, typename TauScalar>
auto update_avar_overlapping(const Eigen::MatrixBase<Derived>& avar_prev,
    const Eigen::MatrixBase<Derived2>& phase_dev, const TauMultCont& tau_mults, TauScalar tau0)
{
    assert(tau0 > 0);
    auto avar_ret = avar_prev.eval();
    auto N = phase_dev.size();
    assert(N > 0);
    auto N_prev = N - 1;
    for (int t = 0; t < tau_mults.size(); ++t)
    {
        auto m = tau_mults[t];
        auto tau = m * tau0;
        assert(N >= 2 * m + 1);
        avar_ret[t] = avar_prev[t] * (N_prev - 2 * m) / (N - 2 * m) +
            std::pow(phase_dev[N - 1] - 2 * phase_dev[N - m - 1] +
                phase_dev[N - 2 * m - 1], 2) /
                (2 * (N - 2 * m) * std::pow(tau, 2));
    }
    return avar_ret;
}

#if 0
// Computes the modified Allan variance
avar_mod(auto phase_dev, int m_min, int m_max)
{
    auto avar_ret;
    auto N = phase_dev.size();
    for (int i = 0; i < i_max; ++i)
    {
        auto m = m_vec[i];
        auto max_idx{N - 3 * m + 1};
        for (int j = 0; j < max_idx; ++j)
        {
            for (int k = j; k <= j + m + 1; ++k)
            {
                sum += phase_dev[k + 2 * m] - 2 * phase_dev[k + m] +
                    phase_dev[k];
            }
            avar_ret[i] = std::pow(sum, 2) /
                (2 * std::pow(m, 2) * std::pow(tau, 2) * max_idx);
        }
    }
}
#endif

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

#endif // CALIB_H_

