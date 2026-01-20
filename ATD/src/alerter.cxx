#include "alerter.h"

// Alerter
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

Alerter::Alerter(string_view name, const ProgState& prog_state)
    : Task(name, prog_state), alert_level_{Alert::Level::green}, alert_time_{},
        queue_{}, timer_{}
{
}

// Returns the time since the alert level was last raised
// Precondition: alert level is above green
Alerter::Clock::duration Alerter::time_since_alert() const
{
    return Clock::now() - alert_time_;
}

void Alerter::run()
{
    BOOST_LOG_FUNCTION();
    using namespace std::chrono_literals;
    std::chrono::milliseconds yellow_expiry_duration{
        get_opt_as<int>("alert.expiry.yellow")};
    std::chrono::milliseconds red_expiry_duration{
        get_opt_as<int>("alert.expiry.red")};
    while (!should_quit())
    {
        auto ms_since_alert =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                time_since_alert());
        std::chrono::milliseconds expiry_limit_ms{};
        switch (alert_level_)
        {
            case Alert::Level::green:
                break; // expiry is N/A for green alert state
            case Alert::Level::yellow:
                expiry_limit_ms = yellow_expiry_duration;
                break;
            case Alert::Level::red:
                expiry_limit_ms = red_expiry_duration;
                break;
        }
        if ((alert_level_ > Alert::Level::green) &&
            (ms_since_alert > expiry_limit_ms))
        {
            TM_LOG(debug) << "Clearing alert";
            bool cleared = clear_alert();
            TM_LOG(debug) << (cleared ? "Alert cleared" :
                "Alert clear overridden");
        }
        // Compute the desired sleep time until we next check for alert expiry
        // This will need to take into account:
        //  (1) The minimum sleep time permissible (so the loop doesn't spin
        //      too fast)
        //  (2) The time until any current alert will expire
        //  (3) The shortest new expiry time for any new alert that may be
        //      raised
        //  (4) The maximum sleep time permissible (if thread interruption is
        //      not used, this needs to be small so the application will exit
        //      cleanly)
        std::chrono::milliseconds min_sleep_duration{1};
        // This maximum duration can be long (currently 30 s) since the blocking
        //  wait below is interruptible (and canceled on shutdown)
        std::chrono::milliseconds max_sleep_duration{30000};
        auto curr_expiry_ms = (expiry_limit_ms - ms_since_alert);
        if ((expiry_limit_ms == 0ms) || (curr_expiry_ms < 0ms))
            curr_expiry_ms = 0ms;
        auto min_new_expiry_ms = std::min(yellow_expiry_duration,
            red_expiry_duration);
        std::chrono::milliseconds sleep_duration = std::max(min_sleep_duration,
            std::min({curr_expiry_ms, min_new_expiry_ms, max_sleep_duration}));

        timer_.wait_for(sleep_duration);
    }
}

void Alerter::stop_hook()
{
    timer_.cancel();
}

// Returns: true if alert successfully cleared (or not needed to be cleared);
//  false otherwise (e.g., by intervening higher alert level)
bool Alerter::clear_alert()
{
    static_assert(Alert::Level::green == Alert::Level{},
        "Alert should initialize to level green");
    Alert::Level orig_level = alert_level_;
    if (orig_level == Alert::Level::green)
    {
        // Nothing to do; alert not currently raised
        // Return true for trivial success
        return true;
    }
    Alert::Level expected_level = orig_level;
    // Don't interfere with an elevation to a higher alert level by
    //  someone else; if no higher alert has come in, clear the alert
    //  state by dropping down to green
    while (expected_level <= orig_level &&
        !alert_level_.compare_exchange_weak(expected_level,
            Alert::Level::green))
    {
        // nop; spin-loop
    }

    if (expected_level > orig_level)
        return false;
    assert(alert_level_ == Alert::Level::green);
    Alert green_alert{Alert::Level::green, Alert::ReasonCode::expiry, ""};
    queue_.push_back(green_alert);
    return true;
}

Alert::Level Alerter::get_alert_level() const
{
    return alert_level_;
}

void Alerter::raise_alert(Alert alert)
{
    // Only change the current alert level if the new level is greater than the
    //  current one
    static_assert(Alert::Level::green < Alert::Level::yellow && 
        Alert::Level::yellow < Alert::Level::red,
        "Unexpected alert level ordering");
    Alert::Level orig_level = alert_level_;
    Alert::Level expected_level = orig_level;
    // Need to satisfy one of these conditions to exit the CAS loop:
    //  (1) the current alert level (status-quo-ante state or one imposed by
    //      another thread) exceeds the "new" level to set here, or
    //  (2) the "new" level is successfully stored
    while (alert.level > expected_level && 
        !alert_level_.compare_exchange_weak(expected_level, alert.level))
    {
        // nop; spin-loop
    }
    assert(alert_level_ >= orig_level);

    if (alert_level_ > orig_level)
    {
        TM_LOG(debug) << "Raised warning level from " << to_string(orig_level)
            << " to " << to_string(alert_level_);
        // FIXME: Need to enforce alert time atomicity?
        alert_time_ = Clock::now();
    }
    // FIXME: Need to reconsider when to update alert time; don't want to
    //  shorten alert duration if alert level is not raised, but probably want
    //  to extend alert duration if new grounds are in effect

    queue_.push_back(alert);
}

Alerter::Listener Alerter::register_listener()
{
    return queue_.get_new_listener();
}

} // end namespace tmon

#ifdef UNIT_TEST
#include <atomic>
#include <iostream>
int main(int argc, char* argv[])
try
{
    volatile std::atomic<bool> fake_quit_flag{false};
    ProgState prog_state{fake_quit_flag};
    prog_state.parse_config(argc, argv);
    Alerter test_alerter{"Alerter", prog_state};

    Alert test_green_alert{Alert::Level::green, Alert::ReasonCode::test};
    Alert test_yellow_alert{Alert::Level::yellow, Alert::ReasonCode::test};
    Alert test_red_alert{Alert::Level::red, Alert::ReasonCode::test};

    // Initial condition should be green alert state
    assert(test_alerter.get_alert_level() == Alert::Level::green);
    test_alerter.raise_alert(test_green_alert);
    assert(test_alerter.get_alert_level() == Alert::Level::green);
    test_alerter.raise_alert(test_yellow_alert);
    assert(test_alerter.get_alert_level() == Alert::Level::yellow);
    test_alerter.raise_alert(test_green_alert);
    assert(test_alerter.get_alert_level() == Alert::Level::yellow &&
        "Alert level should not be lowered by raise_alert");
    test_alerter.raise_alert(test_red_alert);
    assert(test_alerter.get_alert_level() == Alert::Level::red);
    return EXIT_SUCCESS;
}
catch (std::runtime_error& e)
{
    std::cerr << "FATAL ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
#endif

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

