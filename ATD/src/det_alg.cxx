#include "det_alg.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <boost/filesystem.hpp>
#include <date/date.h>
#include <Eigen/Dense>
#include "detector.h"
#include "param.h"
#include "utility.h" // for to_string(duration)

// Detection Algorithms
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
        auto min_sv = m.jacobiSvd().singularValues().minCoeff();
        bool nonneg_svs = (min_sv >= 0);
        double sym_tol = 1e-20;
        bool is_sym = (m.rows() == m.cols()) &&
            ((m - m.transpose()).norm() < sym_tol);
        return (is_sym && nonneg_svs);
    }

    template <typename Derived>
    void ensure_psd(const Eigen::MatrixBase<Derived>& m)
    {
        // TODO: Consider regularizing the matrix here rather than asserting
        assert(is_psd(m));
    }

    struct TauDeps
    {
        Eigen::Matrix3d A; // State transition matrix, A (a.k.a. Phi)
        Eigen::Matrix3d B; // Control matrix, B
        Eigen::Matrix3d Q; // Process noise matrix, Q
    };

    // Gets state transition matrix (A), control matrix (B), process noise
    //  matrix (Q) based on 3-state Gauss-Markov diffusion coefficients (q)
    TauDeps get_zt_tau_dependencies(double tau, const Eigen::Vector3d& q)
    {
        using std::pow;
        TauDeps ret;

        // State transition matrix, A (a.k.a. Phi)
        ret.A << 1,     tau,    pow(tau, 2) / 2.0,
                 0,     1,      tau,
                 0,     0,      1;

        // Control matrix, B
        ret.B << tau,   pow(tau, 2) / 2.0,  pow(tau, 3) / 6.0,
                 0,     tau,                pow(tau, 2) / 2.0,
                 0,     0,                  tau;

        // Process noise matrix, Q
        ret.Q <<
            q[0] * tau + q[1] * pow(tau, 3) / 3.0 + q[2] * pow(tau, 5) / 20.0,
                q[1] * pow(tau, 2) / 2.0 + q[2] * pow(tau, 4) / 8.0,
                q[2] * pow(tau, 3) / 6.0,
            q[1] * pow(tau, 2) / 2.0 + q[2] * pow(tau, 4) / 8.0,
                q[1] * tau + q[2] * pow(tau, 3) / 3.0,
                q[2] * pow(tau, 2) / 2.0,
            q[2] * pow(tau, 3) / 6.0, q[2] * pow(tau, 2) / 2.0, q[2] * tau;

        return ret;
    }
} // end local namespace

namespace tmon
{

DetectorAlg::DetectorAlg(const ProgState& prog_state)
    : prog_state_{prog_state}
{
}

void DetectorAlg::process_cal(const MsgCont& c, Detector& d)
{
    process_cal_hook(c, d);
}

void DetectorAlg::process_cal_hook(const MsgCont&, Detector&)
{
}

void DetectorAlg::process_det(const MsgCont& c, Detector& d)
{
    process_det_hook(c, d);
}

LoggerType& DetectorAlg::get_logger() const
{
    return prog_state_.get_logger();
}

template <typename T>
const T& DetectorAlg::get_opt_as(const std::string& opt_name) const
    /* throws std::out_of_range */
{
    return prog_state_.get_opt_as<T>(opt_name);
}

template <typename T>
const T& DetectorAlg::get_opt_as(const std::string& opt_name,
    std::size_t idx) const
    /* throws std::out_of_range */
{
    return prog_state_.get_opt_as<T>(opt_name, idx);
}

bool DetectorAlg::register_clock(ClockId clk_id, const ClockDesc& clk_desc)
{
    return register_clock_hook(clk_id, clk_desc);
}

bool DetectorAlg::unregister_clock(ClockId clk_id)
{
    return unregister_clock_hook(clk_id);
}

FileWriterPseudoAlg::FileWriterPseudoAlg(const ProgState& prog_state,
    const std::string& filename)
        : DetectorAlg{prog_state}, sent_header_{false}, file_{filename.c_str()}
{
    if (!file_)
        throw std::runtime_error{"Unable to open file " + filename};
}

void FileWriterPseudoAlg::process_det_hook(const MsgCont& c, Detector& det)
{
    if (!sent_header_)
    {
        file_ << det.get_header();
        sent_header_ = true;
    }
    for (const TimeMsg& x : c)
        file_ << to_string(x);
}

bool FileWriterPseudoAlg::register_clock_hook(ClockId, const ClockDesc&)
{
    return true; // nop succeeded
}

bool FileWriterPseudoAlg::unregister_clock_hook(ClockId)
{
    return true; // nop succeeded
}

KFDetectorAlg::KFDetectorAlg(const ProgState& prog_state,
    std::size_t window_len, double thresh_sigma, double lower_thresh_sigma,
    double meas_noise)
        : DetectorAlg{prog_state}, prog_state_{prog_state},
            window_len_{window_len}, innov_history_{window_len}, filter_{},
            last_msg_map_{}, gm_model_map_{}, calibrator_map_{},
            det_start_time_{}, thresh_sigma_{thresh_sigma},
            lower_thresh_sigma_{lower_thresh_sigma},
            meas_noise_{meas_noise}, stat_log_stream_{}
{
    auto stat_logfile = get_opt_as<std::string>("det.stat_logfile");
    if (!stat_logfile.empty() && (stat_logfile != "disable"))
    {
        if (boost::filesystem::exists(stat_logfile))
        {
            std::string logfile_bak = stat_logfile + date::format(
                ".%F-%H%M%S.bak", date::floor<std::chrono::seconds>(
                    std::chrono::system_clock::now()));
            boost::filesystem::copy_file(stat_logfile, logfile_bak);
        }
        stat_log_stream_.open(stat_logfile);
    }
}

void KFDetectorAlg::get_meas_info_from_msgs(const MsgCont& c)
{
    assert(meas_info_vec_.empty());
    KF::Scalar R_diag_default{meas_noise_};
    for (const auto& reporting_msg : c)
    {
        auto tau_for_reporting = get_tau(reporting_msg, reporting_msg.clock_id);
        bool first_comp_in_msg = true;
        for (const auto& comp : reporting_msg.comparisons)
        {
            int meas_row = filter_.get_meas_pair_row(reporting_msg.clock_id,
                comp.other_clock_id);
            if (meas_row < 0)
            {
                meas_row = filter_.add_new_meas_pair(reporting_msg.clock_id,
                    comp.other_clock_id, R_diag_default);
            }
            assert(meas_row >= 0);
            MeasInfo meas_info;
            meas_info.meas = comp.time_versus_other;
            meas_info.meas_row = meas_row;
            meas_info.reporting_tau = tau_for_reporting;
            meas_info.comp_tau = get_tau(reporting_msg, comp.other_clock_id);
            meas_info.reporting_id = reporting_msg.clock_id;
            meas_info.comp_id = comp.other_clock_id;
            meas_info.time_since_last_orig = reporting_msg.time_since_last_orig;
            // Maintain a flag of the first comparison in each time message,
            //  which allows for a single tau update for the reporting clock
            meas_info.first_comp_in_msg = first_comp_in_msg;
            first_comp_in_msg = false;
            meas_info_vec_.push_back(std::move(meas_info));

            last_msg_map_[comp.other_clock_id] = reporting_msg;
        }
        last_msg_map_[reporting_msg.clock_id] = reporting_msg;
    }
}

CalibrationMsg KFDetectorAlg::apply_cal()
{
    using std::end;
    CalibrationMsg ret_msg{CalibrationMsg::EventType::success};
    for (const auto& c : calibrator_map_)
    {
        ClockId clk_id = c.first;
        const Calibrator& cal = c.second;
        if (!cal.has_cal())
            continue;
        Calibration cal_result;
        auto gm_model_iter = gm_model_map_.find(clk_id);
        assert(gm_model_iter != end(gm_model_map_));
        const GaussMarkovModel& orig_model = gm_model_iter->second;
        try
        {
            cal_result = cal.get_cal(orig_model);
        }
        catch (const CalibrationError& e)
        {
            TM_LOG(error) << "Failed calibration attempt for clock #"
                << clk_id << " vs. reference clock #"
                << *cal.get_reference_clock_id() << "; error: " << e.what();
            ret_msg.evt_type = CalibrationMsg::EventType::failure;
            continue; // Calibration failed; reject and consider next clock
        }
        GaussMarkovModel new_cal{cal_result};
        double pos_constrain_pct =
            get_opt_as<double>("cal.pos_constrain_thresh_pct");
        double pos_reject_pct = get_opt_as<double>("cal.pos_reject_thresh_pct");
        double neg_constrain_pct =
            get_opt_as<double>("cal.neg_constrain_thresh_pct");
        double neg_reject_pct = get_opt_as<double>("cal.neg_reject_thresh_pct");
        assert(pos_constrain_pct >= 0);
        assert(pos_reject_pct >= 0);
        assert((neg_constrain_pct <= 0) && (neg_constrain_pct >= -100));
        assert((neg_reject_pct <= 0) && (neg_reject_pct >= -100));
        // Note that only the diffusion coefficients are currently compared,
        //  not the deterministic means
        GaussMarkovModel::Vector q_diff = cal_result.q - orig_model.q;
        GaussMarkovModel::Vector diff_pct = 100 * q_diff.array() /
            orig_model.q.array();
        if (((diff_pct.array() < neg_reject_pct) ||
             (diff_pct.array() > pos_reject_pct)).any())
        {
            TM_LOG(warning) << "Rejecting calibration attempt for clock #"
                << clk_id << " vs. reference clock #"
                << *cal.get_reference_clock_id()
                << "; {q: " << new_cal.q.transpose() << ", mu: "
                << new_cal.mu.transpose() << "} vs. prior {q: "
                << orig_model.q.transpose() << ", mu: "
                << orig_model.mu.transpose() << "}; exceeded ["
                << neg_reject_pct << "%, " << pos_reject_pct << "%] limit";
            ret_msg.evt_type = CalibrationMsg::EventType::failure;
            continue; // Calibration failed; reject and consider next clock
        }
        if (((diff_pct.array() < neg_constrain_pct) ||
             (diff_pct.array() > pos_constrain_pct)).any())
        {
            TM_LOG(warning) << "Constraining calibration attempt for clock #"
                << clk_id << "; exceeded [" << neg_constrain_pct
                << "%, " << pos_constrain_pct << "%] limit";
            q_diff = (diff_pct.array() > pos_constrain_pct).select(
                (q_diff * pos_constrain_pct).cwiseQuotient(diff_pct.matrix()),
                q_diff);
            q_diff = (diff_pct.array() < neg_constrain_pct).select(
                (q_diff * neg_constrain_pct).cwiseQuotient(diff_pct.matrix()),
                q_diff);
            new_cal.q = orig_model.q + q_diff;
            // FIXME: calibration msg per clock? at least don't override failure
            //  with constrained success
            ret_msg.evt_type = CalibrationMsg::EventType::constrained_success;
        }
        TM_LOG(info) << "Applying calibrated model to clock #" << clk_id
            << " vs. reference clock #" << *cal.get_reference_clock_id()
            << "; {q: " << new_cal.q.transpose() << ", mu: "
            << new_cal.mu.transpose() << "} vs. prior {q: "
            << orig_model.q.transpose() << ", mu: " << orig_model.mu.transpose()
            << '}';
        gm_model_iter->second = std::move(new_cal);
    }
    calibrated_ = true;
    return ret_msg;
}

void KFDetectorAlg::process_cal_hook(const MsgCont& c, Detector&)
{
    using std::begin;
    using std::end;
    meas_info_vec_.clear();
    get_meas_info_from_msgs(c);

    for (const auto& meas_info : meas_info_vec_)
    {
        if (!meas_info.time_since_last_orig)
            continue;
        Calibrator& cal = calibrator_map_.at(meas_info.comp_id);
        if (!cal.get_reference_clock_id())
        {
            const GaussMarkovModel& reporting_gm_model =
                gm_model_map_.at(meas_info.reporting_id);
            const GaussMarkovModel& comp_gm_model =
                gm_model_map_.at(meas_info.comp_id);

            // Check if the reporting clock is (comparatively) quiet enough to
            //  consider as a reference for the clock being calibrated
            if (!is_much_quieter_than(reporting_gm_model, comp_gm_model))
                continue;

            // TODO: Consider the case where the same clock may be paired with
            // multiple reporting clocks -- should only one be allowed to
            // calibrate it?  How is that one determined?  It could be the
            // quietest such reporting clock, but should the previous
            // calibration info be reset if a new, quieter clock starts
            // reporting?
        }
        picoseconds phase_dev{meas_info.meas};
        cal.update(meas_info.reporting_id, *meas_info.time_since_last_orig,
            phase_dev);
    }
}

void KFDetectorAlg::process_det_hook(const MsgCont& c, Detector& det)
{
    using std::begin;
    using std::end;
    using std::cbegin;
    using std::cend;
    int total_num_comp = std::accumulate(begin(c), end(c), 0,
        [](int sum, const auto& x) { return sum + x.comparisons.size(); });
    if (total_num_comp <= 0)
        return; // nop if no new timing comparisons

    if (!calibrated_ && get_opt_as<bool>("cal.enabled"))
    {
        CalibrationMsg cal_msg = apply_cal();
        last_cal_time_ = std::chrono::steady_clock::now();
        det.handle_cal_msg(cal_msg);
    }

    // Note the time at which the detection algorithm started, if not yet done
    if (det_start_time_ == std::chrono::steady_clock::time_point{})
        det_start_time_ = std::chrono::steady_clock::now();

    meas_info_vec_.clear();
    meas_info_vec_.reserve(total_num_comp);
    get_meas_info_from_msgs(c);

    Stopwatch sw([this](auto x){
        auto get_logger = [this]() -> decltype(auto){
            return this->get_logger(); };
        TM_LOG(debug) << "DEBUG: process all meas time: " << to_string(x);
        });
    auto meas_info_begin = begin(meas_info_vec_);
    while (meas_info_begin != end(meas_info_vec_))
    {
        meas_info_begin = process_unique_meas(meas_info_begin, det);
    }
    auto proc_fn = [](const auto& x){ return x.processed; };
    std::size_t num_procd = std::count_if(cbegin(meas_info_vec_),
        cend(meas_info_vec_), proc_fn);
    TM_LOG(trace) << "KF Det. processed " << num_procd << " of "
        << meas_info_vec_.size() << " measurement comparisons";

    assert((num_procd == meas_info_vec_.size()) &&
        "Not all measurement comparisons processed");
}

void KFDetectorAlg::handle_anomaly(const AnomalyTestResults& results,
    Detector& det) const
{
    auto anomaly_alert_lvl = get_opt_as<Alert::Level>(
        "alert.level.phase_anomaly");
    std::string reason_extra = "WSSR threshold exceeded (" +
        std::to_string(results.metric) + " > " +
        std::to_string(results.threshold) + ")";
    Alert anomaly_alert{anomaly_alert_lvl,
        Alert::ReasonCode::phase_anomaly, std::move(reason_extra)};
    auto now = std::chrono::steady_clock::now();
    auto since_start = now - det_start_time_;
    auto since_last_cal = now - last_cal_time_;
    std::chrono::seconds alert_ignore_duration{
        get_opt_as<int>("det.alert_ignore_duration")};
    std::chrono::seconds alert_ignore_duration_postcal{
        get_opt_as<int>("det.alert_ignore_duration_postcal")};

    if (calibrated_ && (since_last_cal <= alert_ignore_duration_postcal))
    {
        TM_LOG(warning) << "Ignoring alert: " << describe(anomaly_alert)
            << " (within ignore duration "
            << to_string(alert_ignore_duration_postcal)
            << ", " << to_string(since_last_cal) << " since last cal.)";
        return;
    }

    if (since_start <= alert_ignore_duration)
    {
        TM_LOG(warning) << "Ignoring alert: " << describe(anomaly_alert)
            << " (within ignore duration " << to_string(alert_ignore_duration)
            << ", " << to_string(since_start) << " since start)";
        return;
    }
    det.raise_alert(std::move(anomaly_alert));
}

void KFDetectorAlg::handle_lower_anomaly(const AnomalyTestResults& results,
    Detector& det) const
{
    auto anomaly_alert_lvl = get_opt_as<Alert::Level>(
        "alert.level.lower_phase_anomaly");
    std::string reason_extra = "Lower WSSR threshold crossed (" +
        std::to_string(results.metric) + " < " +
        std::to_string(results.threshold) + ")";
    Alert anomaly_alert{anomaly_alert_lvl,
        Alert::ReasonCode::phase_anomaly, std::move(reason_extra)};
    auto now = std::chrono::steady_clock::now();
    auto since_start = now - det_start_time_;
    auto since_last_cal = now - last_cal_time_;
    std::chrono::seconds alert_ignore_duration{
        get_opt_as<int>("det.alert_ignore_duration")};
    std::chrono::seconds alert_ignore_duration_postcal{
        get_opt_as<int>("det.alert_ignore_duration_postcal")};

    if (calibrated_ && (since_last_cal <= alert_ignore_duration_postcal))
    {
        TM_LOG(warning) << "Ignoring alert: " << describe(anomaly_alert)
            << " (within ignore duration "
            << to_string(alert_ignore_duration_postcal)
            << ", " << to_string(since_last_cal) << " since last cal.)";
        return;
    }

    if (since_start <= alert_ignore_duration)
    {
        TM_LOG(warning) << "Ignoring alert: " << describe(anomaly_alert)
            << " (within ignore duration " << to_string(alert_ignore_duration)
            << ", " << to_string(since_start) << " since start)";
        return;
    }
    det.raise_alert(std::move(anomaly_alert));
}

// Returns iterator to first not-yet-processed MeasInfo entry (or end if none)
std::vector<KFDetectorAlg::MeasInfo>::iterator
KFDetectorAlg::process_unique_meas(
    std::vector<MeasInfo>::iterator meas_info_begin, Detector& det)
{
    using std::begin;
    using std::end;
    KF::Innovations innov;
    KF::ModelDims model_dims = filter_.get_model_dims();
    int M = model_dims.M;
    int N = model_dims.N;
    innov.e.setZero(M);
    innov.e_cov.setZero(M, M);
    innov.K.setZero(N, M);
    bool processed_any = false;
    auto begin_iter_for_next_pass = end(meas_info_vec_);

    for (auto i = meas_info_begin; i != end(meas_info_vec_); ++i)
    {
        auto& meas_info = *i;
        if (meas_info.processed)
            continue;
        int curr_meas_row = meas_info.meas_row;
        if (innov.e[curr_meas_row] != 0)
        {
            if (begin_iter_for_next_pass == end(meas_info_vec_))
                begin_iter_for_next_pass = i;
            continue; // Already processed that meas. row; save for next pass
        }
        // Only update the state for the reporting clock once per time message
        //  (here, chosen to be only for the first reported comparison)
        auto tau_for_reporting = meas_info.first_comp_in_msg ?
            meas_info.reporting_tau : TimeMsg::DurationType::zero();
        update_single_model_for_tau(meas_info.reporting_id,
            tau_for_reporting);
        update_single_model_for_tau(meas_info.comp_id,
            meas_info.comp_tau);
        // Filter assumes times are stated in units of seconds, so convert
        //  accordingly (tau above is converted likewise to seconds)
        double meas_s = std::chrono::duration<double>(meas_info.meas).count();
        KF::Innovations curr_innov = filter_.update(meas_s, curr_meas_row);
        meas_info.processed = true;
        processed_any = true;
        innov.e[curr_meas_row] = curr_innov.e[curr_meas_row];
        // e_cov is a scalar not otherwise needed in the sequential KF variant,
        //  so it is accessed as the first and only element below
        innov.e_cov(curr_meas_row, curr_meas_row) = curr_innov.e_cov(0, 0);
        innov.K.col(curr_meas_row).swap(curr_innov.K.col(curr_meas_row));
        if (innov.e.all())
        {
            if (begin_iter_for_next_pass == end(meas_info_vec_))
                begin_iter_for_next_pass = ++i;
            break; // Done processing all possible unique measurements
        }
    }

    if (!processed_any)
        return end(meas_info_vec_);

    bool innov_history_conformable = innov_history_.empty() ||
        ((innov_history_.front().K.rows() == N) &&
            (innov_history_.front().K.cols() == M));
    if (!innov_history_conformable)
    {
        // Innovations from this time-step are not conformable with the history;
        //  need to reset the history
        TM_LOG(debug) << "Cleared innovations history; "
            "innov. cov. dims changed from " << innov_history_.front().K.rows()
            << "x" << innov_history_.front().K.cols() << "->" << N << "x" << M;
        innov_history_.clear();
    }
    innov_history_.push_front(std::move(innov));

    //KF::Matrix W = KF::Matrix::Identity(M, M);
    KF::Vector W = KF::Vector::Ones(M);

    if (innov_history_.size() >= window_len_)
    {
        double wssr = calc_wssr(W);
        // Test if upper bound exceeded
        AnomalyTestResults results = anomaly_test(wssr);
        if (results.is_anomaly)
            handle_anomaly(results, det);
        // Test if lower bound crossed
        AnomalyTestResults lower_results = lower_anomaly_test(wssr);
        if (lower_results.is_anomaly)
            handle_lower_anomaly(lower_results, det);
        DetMetricMsg det_msg{};
        det_msg.det_metric = wssr;
        det_msg.det_threshold = results.threshold;
        det.handle_det_metric_msg(std::move(det_msg));
    }
    return begin_iter_for_next_pass;
}

double KFDetectorAlg::calc_wssr(const KF::Vector& W) const
{
    using std::cbegin;
    using std::cend;
    double wssr = std::accumulate(cbegin(innov_history_), cend(innov_history_),
        0.0,
        [&W](auto sum, const auto& innov) {
            KF::Vector e_cov_inv = innov.e_cov.diagonal().unaryExpr(
                [](const auto& x){
                    KF::Scalar min_cov{1e-50};
                    return (x < min_cov) ? (1.0 / min_cov) : (1.0 / x); });
            KF::Scalar summand{innov.e.transpose() * e_cov_inv.asDiagonal() *
                W.asDiagonal() * innov.e};
            return sum + summand;
    });
    return wssr;
}

// Returns a struct with is_anomaly true if WSSR is anomalously-high at the
//  specified significance level; false otherwise
auto KFDetectorAlg::anomaly_test(double wssr) const -> AnomalyTestResults
{
    assert(!innov_history_.empty());
    int M = innov_history_[0].e.size();
    double K = thresh_sigma_;
    assert(K > 0);
    double thresh = M * window_len_ + K * std::sqrt(2 * M * window_len_);
    TM_LOG(debug) << "WSSR test: " << wssr << " vs. " << thresh;

    if (stat_log_stream_.is_open())
    {
        stat_log_stream_ << wssr << ", " << thresh << '\n';
    }

    return {(wssr > thresh), wssr, thresh};
}

// Returns a struct with is_anomaly true if WSSR is anomalously-low;
//  false otherwise
//  (cf. anomaly_test, which seeks an anomaly in the opposite direction)
auto KFDetectorAlg::lower_anomaly_test(double wssr) const -> AnomalyTestResults
{
    assert(!innov_history_.empty());
    int M = innov_history_[0].e.size();
    double K = lower_thresh_sigma_;
    assert(K > 0);
    double thresh = M * window_len_ + K * std::sqrt(2 * M * window_len_);
    TM_LOG(debug) << "Lower WSSR test: " << wssr << " vs. " << thresh;

    return {(wssr < thresh), wssr, thresh};
}

void KFDetectorAlg::update_single_model_for_tau(ClockId clk_id,
    TimeMsg::DurationType tau)
{
    double tau_s = std::chrono::duration<double>(tau).count();
    auto gm_model_iter = gm_model_map_.find(clk_id);
    assert(gm_model_iter != gm_model_map_.end());
    const auto& gm_model = gm_model_iter->second;
    TauDeps tau_deps = get_zt_tau_dependencies(tau_s, gm_model.q);
    filter_.update_single_model_for_tau(clk_id, std::move(tau_deps.A),
        std::move(tau_deps.B), std::move(tau_deps.Q));
}

// Returns true if the first time source (specified by its Gauss-Markov
//  model, a) is much quieter (lower phase noise over two different
//  integration periods) than the second time source (b)
bool KFDetectorAlg::is_much_quieter_than(const GaussMarkovModel& a,
    const GaussMarkovModel& b) const
{
    // To assess if a time source is "much quieter" than another (with
    //  respect to phase noise), we compare the variance of the Wiener
    //  process noise component affecting the phase state variable
    Eigen::VectorXd reference_tau_ms_list = get_opt_as<VectorParam<double>>(
        "cal.reference_tau_ms");
    assert((reference_tau_ms_list.size() == 2) &&
        "cal.reference_tau_ms should have length 2");
    std::chrono::milliseconds short_tau{
        static_cast<int>(reference_tau_ms_list[0])};
    std::chrono::milliseconds long_tau{
        static_cast<int>(reference_tau_ms_list[1])};
    auto get_wiener_phase_var = [](std::chrono::duration<double> tau,
            const auto& q){
        return get_zt_tau_dependencies(tau.count(), q).Q(0, 0); };
    auto wiener_phase_var_a_short = get_wiener_phase_var(short_tau, a.q);
    auto wiener_phase_var_b_short = get_wiener_phase_var(short_tau, b.q);
    auto wiener_phase_var_a_long = get_wiener_phase_var(long_tau, a.q);
    auto wiener_phase_var_b_long = get_wiener_phase_var(long_tau, b.q);

    Eigen::VectorXd reference_quieter_mults = get_opt_as<VectorParam<double>>(
        "cal.reference_quieter_mults");
    assert((reference_quieter_mults.size() == 2) &&
        "cal.reference_quieter_mults should have length 2");
    auto is_quieter = [&reference_quieter_mults](const auto& x, const auto& y) {
        return (reference_quieter_mults[0] * x) < y; };
    auto is_much_quieter = [&reference_quieter_mults](const auto& x,
            const auto& y) {
        return (reference_quieter_mults[1] * x) < y; };
    // Require that clock 'a' is quieter over both integration times and
    //  *much* quieter over one or both
    return is_quieter(wiener_phase_var_a_short, wiener_phase_var_b_short) &&
        is_quieter(wiener_phase_var_a_long, wiener_phase_var_b_long) &&
        (is_much_quieter(wiener_phase_var_a_long, wiener_phase_var_b_long)
            || is_much_quieter(wiener_phase_var_a_short,
                wiener_phase_var_b_short));
}

// Sets state transition matrix (A), control matrix (B), process noise
//  matrix (Q), and deterministic control input (u) based on 3-state
//  Gauss-Markov diffusion coefficients (q) and means (mu)
void update_single_zt_model(KF::Model& m, double tau,
    const Eigen::Vector3d& q, const Eigen::Vector3d& mu)
{
    TauDeps tau_deps = get_zt_tau_dependencies(tau, q);

    m.A = std::move(tau_deps.A);
    m.B = std::move(tau_deps.B);
    ensure_psd(tau_deps.Q);
    m.Q = std::move(tau_deps.Q);

    // Deterministic control input (mean), u
    m.u = mu;
}

bool KFDetectorAlg::register_clock_hook(ClockId clk_id,
    const ClockDesc& clk_desc)
{
    // Register the Gauss-Markov model for this clock
    const auto& gm_model = clk_desc.gm_model;
    assert((gm_model_map_.find(clk_id) == gm_model_map_.end()) &&
        "Clock double-registered in KFDetectorAlg");
    gm_model_map_[clk_id] = gm_model;
    auto N = gm_model.mu.rows();

    // Add a calibrator for this clock
    assert((calibrator_map_.find(clk_id) == calibrator_map_.end()) &&
        "Clock calibrator double-registered");
    Calibration default_cal{gm_model};
    Calibrator new_cal{prog_state_, std::move(default_cal)};
    calibrator_map_.emplace(clk_id, std::move(new_cal));

    // Determine Kalman filter model parameters based on specified Gauss-Markov
    //  model parameters and add the model to the block filter
    double initial_cov = get_opt_as<double>("det.initial_cov");
    KF::Matrix default_P0 = KF::Matrix::Identity(N, N) * initial_cov;
    // tau (the measurement interval in sampled-data representation)
    //  is here defaulted to 1 s, without parameterization, since the sequential
    //  processing workflow will update the measurement interval at each
    //  time-step
    double default_tau{1.0};

    KF::IdentifiedModel kf_model;
    kf_model.clock_id = clk_id;
    kf_model.x0.setZero(N, 1);
    kf_model.P0 = default_P0;
    // Note: Measurement-space matrics (C & R) not used by add_model, so not set
    update_single_zt_model(kf_model, default_tau, gm_model.q, gm_model.mu);
    bool added = filter_.add_model(kf_model);
    return added;
}

bool KFDetectorAlg::unregister_clock_hook(ClockId clk_id)
{
    if (!filter_.has_model_for(clk_id))
        return false;
    last_msg_map_.erase(clk_id);
    gm_model_map_.erase(clk_id);
    calibrator_map_.erase(clk_id);
    return filter_.remove_model(clk_id);
}

TimeMsg::DurationType KFDetectorAlg::get_tau(TimeMsg new_msg,
    ClockId clk_id) const
{
    using date::operator<<;
    assert(filter_.has_model_for(clk_id));
    // Check if there is a previous time message that reports on this clock (and
    //  may have therefore updated its state variables) -- if not, this current
    //  message provides all needed information about the time interval
    auto last_msg_iter = last_msg_map_.find(clk_id);
    TimeMsg::DurationType new_time_since_last_orig{0};
    if (new_msg.time_since_last_orig)
        new_time_since_last_orig = *new_msg.time_since_last_orig;
    if (last_msg_iter == last_msg_map_.end())
        return new_time_since_last_orig;
    TimeMsg last_msg_for_id = last_msg_iter->second;
    if (last_msg_for_id.clock_id == new_msg.clock_id)
        return new_time_since_last_orig;

    // Find an estimate of the phase difference between the clock reporting
    //  in the new time message and the clock reporting in the previous time
    //  message through the measured phase differences with the common-view
    //  clock in question (guaranteed, since the function returns above if the
    //  previous report is from the same clock as the new report)
    //
    // Note that this assumes the phase relationship between the two clocks
    //  (with the third clock in common view) is at least somewhat stable; i.e.,
    //  they cannot have diverged too dramatically during a fraction of a
    //  reporting interval
    TimeMsg::DurationType delta1{0}, delta2{0};
    auto cmp_clk_id_match = [clk_id](const auto& x){
        return x.other_clock_id == clk_id; };
    if (last_msg_for_id.clock_id != clk_id)
    {
        auto matching_cmp = std::find_if(begin(last_msg_for_id.comparisons),
            end(last_msg_for_id.comparisons), cmp_clk_id_match);
        assert(matching_cmp != end(last_msg_for_id.comparisons));
        delta1 = matching_cmp->time_versus_other;
    }
    if (new_msg.clock_id != clk_id)
    {
        auto matching_cmp = std::find_if(begin(new_msg.comparisons),
            end(new_msg.comparisons), cmp_clk_id_match);
        assert(matching_cmp != end(new_msg.comparisons));
        delta2 = matching_cmp->time_versus_other;
    }
    auto phase_diff_ps = delta2 - delta1;

    // The following might overflow due to the range of duration in ps:
    //   auto last_time_in_new_msg_frame = last_msg_for_id.orig_timestamp -
    //      phase_diff_ps;
    //   auto tau = new_msg.orig_timestamp - last_time_in_new_msg_frame;
    // So, we instead calculate:
    auto delta_orig = new_msg.orig_timestamp - last_msg_for_id.orig_timestamp;
    auto tau = delta_orig + phase_diff_ps;
    if (tau < tmon::picoseconds{0})
    {
        std::ostringstream oss;
        oss << tau;
        TM_LOG(warning) << "Inconsistent tau obtained (" << oss.str()
            << " from prev. msg. by clk #" << last_msg_for_id.clock_id
            << " for clk #" << clk_id << ")";
        tau = tmon::picoseconds{0};
    }
    // While not impossible (e.g., there could be a missed report), it is
    //  unlikely that another time message is more recent than the last report
    //  from this clock, and yet reporting over a longer time interval than
    //  this clock's reporting duration; issue a warning and use the latter tau
    if (new_msg.time_since_last_orig && (tau > new_time_since_last_orig))
    {
        std::ostringstream oss;
        oss << tau << " > " << new_time_since_last_orig;
        TM_LOG(warning) << "Inconsistent tau obtained (" << oss.str()
            << " expected max. with prev. msg. by clk #"
            << last_msg_for_id.clock_id << " for clk #" << clk_id << ")";
        tau = new_time_since_last_orig;
    }
    return tau;
}

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

