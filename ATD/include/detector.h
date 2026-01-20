#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <functional>
#include <map>
#include "common.h"
#include "alert_msg.h"
#include "alerter.h"
#include "clock.h"
#include "det_alg.h"
#include "det_msg.h"
#include "spmc.h"
#include "task.h"
#include "time_msg.h"

// Detector
//  This module is responsible for maintaining a registry of associated
//  clocks, processing the time messages that they generate as they
//  make timing measurements, containing detection algorithms and providing
//  them with time message batches for calibration and detection.
//  An associated Alerter is notified when a detection algorithm detects
//  an anomaly or an associated clock raises an alert.  Finally, the module is
//  able to report when all associated clocks are finished generating time
//  messages as well as when all processing has been completed.
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

class Detector : public Task
{
  public:
    using MsgQueue = SpmcQueue<variant<Alert, CalibrationMsg, DetMetricMsg>>;

    Detector(string_view name, const ProgState& prog_state, Alerter& alerter);
    ~Detector() override = default;
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    enum class Mode {
        Calibration, Detection
    };

    bool all_clocks_done() const;
    std::string describe_clock(tmon::ClockId clk_id) const;
    GaussMarkovModel get_gauss_markov_clock_model(tmon::ClockId clk_id) const;
    TimeMsgHeader get_header() const;
    Mode get_mode() const noexcept;
    // Locks registry_mutex_:
    bool is_clock_registered(const tmon::ClockDesc& desc) const;
    bool is_processing() const;
    tmon::ClockId register_clock(const tmon::ClockDesc& names);
    bool unregister_clock(tmon::ClockId clk_id);

    void handle_cal_msg(CalibrationMsg msg);
    void handle_clock_done(tmon::ClockId clk_id);
    void handle_det_metric_msg(DetMetricMsg msg);
    void handle_time_msg(TimeMsg msg);
    void raise_alert(tmon::Alert alert);
    MsgQueue::Listener register_msg_queue_listener();

  protected:
    void run() override;
    void stop_hook() override;

  private:
    using TimeMsgCont = std::vector<TimeMsg>;
    using AlgCont = std::vector<std::unique_ptr<DetectorAlg>>;
    using RegistryMap = std::map<tmon::ClockId, tmon::ClockDesc>;
    using ClockDoneMap = std::map<tmon::ClockId, bool>;
    using AlertListener = Alerter::Listener;

    RegistryMap clock_registry_;
    ClockDoneMap clock_done_flags_;
    condition_variable cond_;
    mutable mutex msg_mutex_;
    mutable mutex registry_mutex_;
    std::size_t min_to_process_;
    // Container into which time messages are added by reporting clocks:
    TimeMsgCont msg_cont_;
    // Container into which time messages are swapped for processing by the
    //  detection algorithms:
    TimeMsgCont processing_msg_cont_;
    AlgCont alg_cont_;
    Alerter& alerter_;
    AlertListener alert_listener_;
    optional<Alert::Level> prev_alert_msg_level_;
    MsgQueue msg_queue_;
    Mode mode_;
    // Starting time of calibration period
    std::chrono::steady_clock::time_point cal_start_;

    void check_if_cal_done();
    // Call under lock on registry_mutex_ (passed as second parameter)
    bool is_clock_registered(const tmon::ClockDesc& desc,
        const lock_guard<mutex>&) const;
    void process_alert_msgs();
    void process_time_msgs(const TimeMsgCont& msgs);
    bool ready_to_process() const noexcept;
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

#endif // DETECTOR_H_
