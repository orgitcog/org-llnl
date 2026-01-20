#ifndef ALERT_MSG_H_
#define ALERT_MSG_H_

#include "common.h"

// Alert Message
//  This class describes an Alert raised by a clock or detection algorithm and
//  communicated to the Alerter module and any associated listeners.  It
//  contains both a numeric ReasonCode that identifies the basis for the alert
//  as well as a numeric Level.  Additionally, a string providing additional
//  explanation for the alert and its basis may be populated.  Streaming
//  operators are provided.
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

struct Alert
{
    enum class ReasonCode 
    {
        test,
        fatal_error,
        lost_fix,
        gnss_inconsistency,
        position_mismatch,
        gnss_misc,
        // Low C/N0 (carrier-to-noise ratio)
        low_c_vs_noise,
        lost_clock,
        degraded_clock,
        // Environmental condition (e.g., detected high temperature, vibration)
        environment,
        phase_anomaly,
        freq_anomaly,
        expiry,
        other
    };

    enum class Level
    {
        green,
        yellow,
        red
    };

    // Alert level:
    //  Green - the system is operating normally and has inputs of sufficient
    //      quality to declare a "normal" (non-anomalous) operating condition
    //  Yellow - system has detected a fault that may signal an anomalous
    //      condition or interfere in the detection of one
    //  Red - system has inputs of sufficient quality and based upon them can
    //      declare detection of an anomalous condition
    Level level;
    // Reason code: numeric representation of the reason for the alert
    ReasonCode reason_code;
    // Reason (extra): more specific description of the basis for the alert
    std::string reason_extra;
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    TimePoint timestamp = Clock::now();
};

std::string describe(const Alert& alert);
std::string describe(Alert::ReasonCode r);
Alert::Level from_string(string_view sv);
std::string to_string(Alert::Level level);

std::istream& operator>>(std::istream& is, Alert::Level& level);
std::ostream& operator<<(std::ostream& os, const Alert::Level& level);

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

#endif // ALERT_MSG_H_
