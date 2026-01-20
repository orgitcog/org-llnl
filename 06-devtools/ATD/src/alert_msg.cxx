#include "alert_msg.h"
#include <algorithm>
#include "utility.h"

// Alert_msg
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

Alert::Level from_string(string_view sv)
    /* throws std::out_of_range */
{
    auto ci_char_equals = [](char a, char b){
        return std::tolower(a) == std::tolower(b); };
    auto ci_equal = [ci_char_equals](string_view a, string_view b) {
        return std::equal(std::begin(a), std::end(a),
            std::begin(b), std::end(b), ci_char_equals); };

    if (ci_equal(sv, "green"))
    {
        return Alert::Level::green;
    }
    else if (ci_equal(sv, "yellow"))
    {
        return Alert::Level::yellow;
    }
    else if (ci_equal(sv, "red"))
    {
        return Alert::Level::red;
    }
    else
    {
        throw std::out_of_range("Unknown alert level");
    }
}

std::string to_string(Alert::Level level)
{
    switch (level)
    {
        case Alert::Level::green:
            return "green";
        case Alert::Level::yellow:
            return "yellow";
        case Alert::Level::red:
            return "red";
    }
    return "<UNKNOWN>";
}

std::string describe(const Alert& alert)
{
    std::string reason_extra =
        (alert.reason_extra.empty() ? "" : " (" + alert.reason_extra + ")");
    return to_upper(to_string(alert.level)) + " ALERT: "
        + describe(alert.reason_code) + reason_extra;
}

std::string describe(Alert::ReasonCode r)
{
    switch (r)
    {
        case Alert::ReasonCode::test:
            return "test alert";
        case Alert::ReasonCode::fatal_error:
            return "fatal application error";
        case Alert::ReasonCode::lost_fix:
            return "lost GNSS satellite fix";
        case Alert::ReasonCode::gnss_inconsistency:
            return "GNSS data inconsistency";
        case Alert::ReasonCode::position_mismatch:
            return "GNSS position mismatch";
        case Alert::ReasonCode::gnss_misc:
            return "unspecified GNSS anomaly";
        case Alert::ReasonCode::low_c_vs_noise:
            return "low C/N0 (carrier-to-noise ratio)";
        case Alert::ReasonCode::lost_clock:
            return "lost clock connection";
        case Alert::ReasonCode::degraded_clock:
            return "degraded clock condition";
        case Alert::ReasonCode::environment:
            return "environmental condition";
        case Alert::ReasonCode::phase_anomaly:
            return "phase anomaly";
        case Alert::ReasonCode::freq_anomaly:
            return "frequency anomaly";
        case Alert::ReasonCode::expiry:
            return "alert expired";
        case Alert::ReasonCode::other:
            return "unspecified alert condition";
    }
    return "<UNKNOWN>";
}

std::istream& operator>>(std::istream& is, Alert::Level& level)
{
    try
    {
        std::string s;
        is >> s;
        level = from_string(s);
    }
    catch(std::out_of_range& e)
    {
        is.setstate(std::ios::failbit);
    }
    return is;
}

std::ostream& operator<<(std::ostream& os, const Alert::Level& level)
{
    return (os << to_string(level));
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

