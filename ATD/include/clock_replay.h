#ifndef CLOCK_REPLAY_H_
#define CLOCK_REPLAY_H_

#include <vector>
#include <boost/lockfree/spsc_queue.hpp>
#include "common.h"
#include "task.h"
#include "task_container.h"
#include "time_msg.h"
#include "utility.h"

// Clock Replay
//  This unit allows for the controlled storage and replay of timing messages
//  from an associated clock.  Ownership of a StreamClock is assumed, and two
//  subordinate tasks are launched: the first task reads from the clock and
//  stores time messages in a queue, and the second task replays these messages
//  by transmitting them to the Detector.  The delay between messages is
//  configurable: it can either be set to a fixed-duration delay or it can
//  simulate the inter-arrival delay captured in the message timestamps.
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

// Forward declarations
class Detector;
class StreamClock;

class ClockReplay : public TaskContainer
{
  public:
    static constexpr size_t queue_capacity = 65536;
    // Although TimeMsg is default-constructible, using optional wrapper since
    //  spsc_queue is not move-enabled without it
    using Queue = boost::lockfree::spsc_queue<optional<TimeMsg>,
        boost::lockfree::capacity<queue_capacity>>;

    ClockReplay(string_view name, const ProgState& prog_state, Detector& det,
        std::unique_ptr<StreamClock> clock);
    ~ClockReplay() override = default;
    ClockReplay(const ClockReplay&) = delete;
    ClockReplay& operator=(const ClockReplay&) = delete;

  protected:
    void stop_hook() override;

  private:
    Queue buffer_;
    std::unique_ptr<StreamClock> clock_;
};

namespace detail
{
class ClockReplayReader : public Task
{
  public:
    ClockReplayReader(string_view name, const ProgState& prog_state,
        ClockReplay::Queue& buffer, StreamClock& clock);
    ~ClockReplayReader() override = default;
    ClockReplayReader(const ClockReplayReader&) = delete;
    ClockReplayReader& operator=(const ClockReplayReader&) = delete;

  protected:
    void run() override;
    void stop_hook() override;

  private:
    ClockReplay::Queue& buffer_;
    StreamClock& clock_;
};

class ClockReplayPlayer : public Task
{
  public:
    ClockReplayPlayer(string_view name, const ProgState& prog_state,
        Detector& det, ClockReplay::Queue& buffer, StreamClock& clock);
    ~ClockReplayPlayer() override;
    ClockReplayPlayer(const ClockReplayPlayer&) = delete;
    ClockReplayPlayer& operator=(const ClockReplayPlayer&) = delete;

  protected:
    void run() override;
    void stop_hook() override;

  private:
    ClockReplay::Queue& buffer_;
    StreamClock& clock_;
    Detector& detector_;
    InterruptibleTimer timer_;

    bool could_read() const;
    void wait_after_msg(const TimeMsg& msg);
    void wait_for_next();
};
} // end namespace detail

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

#endif // CLOCK_REPLAY_H_
