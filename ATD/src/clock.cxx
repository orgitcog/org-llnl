#include "clock.h"
#include "detector.h"

// Clock
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

Clock::Clock(Detector& det, ClockDesc names)
    : detector_{det}, names_{names}, id_{det.register_clock(names_)},
        done_{false}, registered_{true}
{
}

Clock::~Clock()
{
    if (registered_)
        unregister();
}

bool Clock::done() const
{
    return done_;
}

void Clock::set_done_flag()
{
    done_ = true;
    detector_.handle_clock_done(id_);
}

void Clock::unregister()
{
    assert(registered_);
    bool was_registered = detector_.unregister_clock(id_);
    assert(was_registered);
    registered_ = false;
}

std::string Clock::describe() const
{
    using std::to_string;
    return to_string(id_) + ":" + names_.describe();
}

// Send the message to the Detector after first computing difference times
//  versus the last message sent from this clock
void Clock::send_message(TimeMsg msg)
{
    assert(!done());
    if (last_msg_)
    {
        msg.time_since_last_msg = msg.msg_creation - last_msg_->msg_creation;
        msg.time_since_last_orig = msg.orig_timestamp -
            last_msg_->orig_timestamp;
    }
    last_msg_ = msg;
    detector_.handle_time_msg(std::move(msg));
}

// Send the message to the Detector without computing difference times versus
//  the last message sent from this clock
// (Note: Usually, send_message should be used instead)
void Clock::send_message_verbatim(TimeMsg msg)
{
    assert(!done());
    last_msg_ = msg;
    detector_.handle_time_msg(std::move(msg));
}

const ClockDesc& Clock::desc() const
{
    return names_;
}

ClockId Clock::get_id() const
{
    return id_;
}

const GaussMarkovModel& Clock::gauss_markov_model() const
{
    return names_.gm_model;
}

} // namespace tmon

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

