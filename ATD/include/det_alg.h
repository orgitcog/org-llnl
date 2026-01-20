#ifndef DET_ALG_H_
#define DET_ALG_H_

#include <fstream>
#include <vector>
#include <boost/circular_buffer.hpp>
#include "calib.h"
#include "common.h"
#include "det_msg.h"
#include "kf.h"
#include "prog_state.h"
#include "time_msg.h"

// Detection Algorithms
//  This unit defines a base class (DetectorAlg) for all detection algorithms
//  that can be registered with the Detector.  Since detection algorithms may
//  vary in terms of their underlying models and associated model parameters,
//  there is a mechanism for each algorithm to receive data during the
//  calibration period (the process_cal method and associated hook).  Once the
//  detection processing begins (when the process_det method and associated hook
//  are first called), the algorithms are expected to apply any calibration
//  adjustments necessary.  A pseudo-algorithm is also provided that archives
//  timing messages for subsequent replay.
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

class Detector;

class DetectorAlg
{
  public:
    using MsgCont = std::vector<TimeMsg>;

    DetectorAlg(const ProgState& prog_state);
    virtual ~DetectorAlg() = default;
    void process_cal(const MsgCont& c, Detector& d);
    void process_det(const MsgCont& c, Detector& d);
    bool register_clock(ClockId clk_id, const ClockDesc& clk_desc);
    bool unregister_clock(ClockId clk_id);

  protected:
    LoggerType& get_logger() const;
    template <typename T>
    const T& get_opt_as(const std::string& opt_name) const;
        /* throws std::out_of_range */
    template <typename T>
    const T& get_opt_as(const std::string& opt_name, std::size_t idx) const;
        /* throws std::out_of_range */
    virtual void process_cal_hook(const MsgCont& c, Detector& d);
    virtual void process_det_hook(const MsgCont& c, Detector& d) = 0;
    virtual bool register_clock_hook(ClockId clk_id,
        const ClockDesc& clk_desc) = 0;
    virtual bool unregister_clock_hook(ClockId clk_id) = 0;

  private:
    const ProgState& prog_state_;
};

class FileWriterPseudoAlg : public DetectorAlg
{
  public:
    FileWriterPseudoAlg(const ProgState& prog_state,
        const std::string& filename);
    ~FileWriterPseudoAlg() override = default;

  protected:
    void process_det_hook(const MsgCont& c, Detector& d) override;
    bool register_clock_hook(ClockId clk_id, const ClockDesc& clk_desc)
        override;
    bool unregister_clock_hook(ClockId clk_id) override;

  private:
    bool sent_header_;
    std::ofstream file_;
};

class KFDetectorAlg : public DetectorAlg
{
  public:
    KFDetectorAlg(const ProgState& prog_state, std::size_t window_len,
        double thresh_sigma, double lower_thresh_sigma, double meas_noise);
    ~KFDetectorAlg() override = default;

    struct AnomalyTestResults
    {
        bool is_anomaly;
        double metric;
        double threshold;
    };
    double calc_wssr(const KF::Vector& W) const;
    AnomalyTestResults anomaly_test(double wssr) const;
    AnomalyTestResults lower_anomaly_test(double wssr) const;

  protected:
    void process_cal_hook(const MsgCont& c, Detector& d) override;
    void process_det_hook(const MsgCont& c, Detector& d) override;
    bool register_clock_hook(ClockId clk_id, const ClockDesc& clk_desc)
        override;
    bool unregister_clock_hook(ClockId clk_id) override;

  private:
    using CalibratorMap = std::map<ClockId, Calibrator>;
    const ProgState& prog_state_;
    std::size_t window_len_;
    boost::circular_buffer<KF::Innovations> innov_history_;
    KF::BlockFilter filter_;
    std::map<ClockId, TimeMsg> last_msg_map_;
    std::map<ClockId, GaussMarkovModel> gm_model_map_;
    CalibratorMap calibrator_map_;
    bool calibrated_ = false;
    std::chrono::steady_clock::time_point last_cal_time_;
    std::chrono::steady_clock::time_point det_start_time_;
    double thresh_sigma_;
    double lower_thresh_sigma_;
    double meas_noise_;
    mutable std::ofstream stat_log_stream_;

    struct MeasInfo
    {
        TimeMsg::DurationType meas;
        int meas_row;
        TimeMsg::DurationType reporting_tau; // tau for reporting clock
        TimeMsg::DurationType comp_tau;      // tau for compared clock
        ClockId reporting_id;
        ClockId comp_id;
        optional<TimeMsg::DurationType> time_since_last_orig;
        bool processed = false;
        bool first_comp_in_msg = false;
    };
    std::vector<MeasInfo> meas_info_vec_;

    CalibrationMsg apply_cal();
    void get_meas_info_from_msgs(const MsgCont& c);
    TimeMsg::DurationType get_tau(TimeMsg new_msg, ClockId clk_id) const;
    void handle_anomaly(const AnomalyTestResults& results, Detector& det) const;
    void handle_lower_anomaly(const AnomalyTestResults& results,
        Detector& det) const;
    bool is_much_quieter_than(const GaussMarkovModel& a,
        const GaussMarkovModel& b) const;
    std::vector<MeasInfo>::iterator process_unique_meas(
        std::vector<MeasInfo>::iterator meas_info_begin, Detector& det);
    void update_single_model_for_tau(ClockId clk_id, TimeMsg::DurationType tau);
};

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

#endif // DET_ALG_H_

