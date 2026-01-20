#ifndef ALERTER_H_
#define ALERTER_H_

#include <array>
#include <atomic>
#include "alert_msg.h"
#include "common.h"
#include "spmc.h"
#include "task.h"
#include "utility.h"

// Alerter
//  This unit maintains the alert level for the application, as a single atomic
//  variable that allows reporting from multiple threads.  An alert may be
//  raised (which will either maintain the alert level or increase it), either
//  by a clock or detection algorithm.  Registered listeners, each of which
//  refers to a single-producer/multiple-consumer queue, receive these
//  alerts when generated.  The alert level only drops when the specified expiry
//  interval has elapsed.
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

class Alerter : public Task
{
  public:
    using Clock = Alert::Clock;
    using Queue = SpmcQueue<Alert>;

    Alerter(string_view name, const ProgState& prog_state);
    ~Alerter() override = default;
    Alerter(const Alerter&) = delete;
    Alerter& operator=(const Alerter&) = delete;

    Alert::Level get_alert_level() const;
    void raise_alert(Alert alert);
    Clock::duration time_since_alert() const;

    using Listener = Queue::Listener;
    Listener register_listener();

  protected:
    void run() override;
    void stop_hook() override;

  private:
    using TimePoint = Alert::TimePoint;

    std::atomic<Alert::Level> alert_level_;
    TimePoint alert_time_;
    Queue queue_;
    InterruptibleTimer timer_;

    bool clear_alert();
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

#endif // ALERTER_H_
