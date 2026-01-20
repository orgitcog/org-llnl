#ifndef CLOCK_H_
#define CLOCK_H_

#include <vector>
#include "common.h"
#include "clock_desc.h"
#include "gm_model.h"
#include "time_msg.h"

// Clock
//  This unit provides a base class for clocks that will be registered with the
//  Detector.  Associated with the clock is a Gauss-Markov model description,
//  a set of unique names, a clock ID (obtained upon registration), and a stored
//  version of the preceding time message (used to measure the inter-message
//  duration).  The clock is able to produce a time message (reported to
//  Detector).  The clock will automatically unregister itself upon destruction
//  (if not previously unregistered), but derived classes should take care to
//  call set_done_flag when no more time messages will be generated, as this
//  facilitates clean shutdown.
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

class Clock
{
  public:
    Clock(Detector& det, ClockDesc names);
    virtual ~Clock();
    Clock(const Clock&) = delete;
    Clock& operator=(const Clock&) = delete;

    const ClockDesc& desc() const;
    bool done() const;
    std::string describe() const;
    tmon::ClockId get_id() const;
    const GaussMarkovModel& gauss_markov_model() const;
    // Send a TimeMsg to the Detector for processing
    //  Note: send_message is almost always the correct function to call for
    //  this, as it will automatically compute the difference times versus the
    //  previous message; in the case where these should not be computed, use
    //  send_message_verbatim instead
    void send_message(TimeMsg msg);
    void send_message_verbatim(TimeMsg msg);
    void set_done_flag();

  private:
    Detector& detector_;
    ClockDesc names_;
    ClockId id_;
    bool done_;
    bool registered_;

    optional<TimeMsg> last_msg_;

    // Unregisters the clock from the Detector clock registry
    //  (precondition: currently registered)
    void unregister();
};

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

#endif // CLOCK_H_
