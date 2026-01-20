#include "clock_replay.h"
#include "detector.h"
#include "stream_clock.h"
#include "utility.h"

// Clock Replay
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

using namespace std::chrono_literals;

namespace tmon
{

namespace detail
{
ClockReplayPlayer::ClockReplayPlayer(string_view name,
    const ProgState& prog_state, Detector& det, ClockReplay::Queue& buffer,
    StreamClock& clock)
        : Task{name, prog_state}, buffer_{buffer}, clock_{clock},
            detector_{det}, timer_{}
{
}

ClockReplayReader::ClockReplayReader(string_view name,
    const ProgState& prog_state, ClockReplay::Queue& buffer, StreamClock& clock)
        : Task{name, prog_state}, buffer_{buffer}, clock_{clock}
{
}

void ClockReplayReader::run()
{
    // Read clock registry header before parsing time messages that follow
    //  (this will register any previously-unknown clocks in the global clock
    //  registry as well as enroll them in a translator that maps the clock ID
    //  on the stream to the ID in the global registry)
    if (!clock_.stream_done())
        clock_.read_header();

    while (!should_quit() && !clock_.stream_done())
    {
        try
        {
            TimeMsg new_msg = clock_.read_message();
            if (!buffer_.write_available())
            {
                TM_LOG(warning) << "Out of space in ClockReplay queue; waiting";
                std::chrono::milliseconds full_buffer_delay{get_opt_as<int>(
                    "test.replay.full_buffer_delay")};
                boost::this_thread::sleep_for(boost_duration_cast(
                    full_buffer_delay));
                if (!buffer_.write_available())
                {
                    TM_LOG(warning) <<
                        "Out of space in ClockReplay queue; message lost";
                }
            }
            buffer_.push(std::move(new_msg));
        }
        catch(const std::exception& e)
        {
            // Swallow exceptions when quitting (likely just interrupted read)
            // FIXME: May be able to avoid a throw here on EOF
            if (!should_quit() && !clock_.stream_done())
                throw;
        }
    }
}

void ClockReplayReader::stop_hook()
{
}

ClockReplayPlayer::~ClockReplayPlayer()
{
}

void ClockReplayPlayer::run()
{
    while (!should_quit() && could_read())
    {
        if (buffer_.read_available())
        {
            optional<TimeMsg> msg = buffer_.front();
            assert(msg);

            // Send the message through the clock to the Detector, but make
            //  it use the difference times verbatim from the message itself
            clock_.send_message_verbatim(*msg);
            buffer_.pop();

            wait_after_msg(*msg);
        }
        else
        {
            wait_for_next();
        }
    }

    // Signal that the clock is done, since it won't send any more time messages
    //  now that the task is complete; this may allow for early shutdown
    clock_.set_done_flag();
    clock_.propagate_done_flag(); // propagate flag to subordinate clocks
}

void ClockReplayPlayer::stop_hook()
{
    timer_.cancel();
}

// Returns true if an additional read is possible, which in turn means that
//  either the underlying clock could report additional messages or there are
//  remaining messages in the buffer (note that this is not *can*-read -- it
//  does not mean that there is necessarily a message in the buffer immediately
//  available)
bool ClockReplayPlayer::could_read() const
{
    return !clock_.stream_done() || buffer_.read_available();
}

void ClockReplayPlayer::wait_after_msg(const TimeMsg& curr_msg)
{
    std::chrono::microseconds fixed_delay{
        get_opt_as<int>("test.replay.fixed_delay")};
    bool use_fixed_dur = get_opt_as<bool>("test.replay.use_fixed_delay");
    std::chrono::microseconds min_delay{
        get_opt_as<int>("test.replay.min_delay")};

    if (should_quit() || !could_read())
        return;

    InterruptibleTimer::Duration wait_dur = min_delay;

    if (use_fixed_dur)
    {
        // Use a fixed delay between messages, regardless of the original
        //  interval between message timestamps
        wait_dur = fixed_delay;
    }
    else
    {
        // Emulate the interval between the original message creation timestamps
        //  (which requires first making sure the following message is available
        //  in the buffer, then waiting for the inter-message interval)
        wait_for_next();
        if (should_quit() || !could_read())
            return;
        assert(buffer_.read_available());
        optional<TimeMsg> next_msg = buffer_.front();
        assert(next_msg);
        // Prefer the duration between successive timestamps over the
        //  inter-message duration reported directly within the message, since
        //  that duration is specific to a given clock, and a different clock
        //  may be reporting sooner
        auto delta_dur = next_msg->msg_creation - curr_msg.msg_creation;

        if (delta_dur > min_delay)
            wait_dur = delta_dur;
    }
    TM_LOG(trace) << "Replay waiting " 
        << std::chrono::duration_cast<std::chrono::microseconds>(
            wait_dur).count() << "us between msgs";
    timer_.wait_for(wait_dur);
}

void ClockReplayPlayer::wait_for_next()
{
    auto wait_delay = 1ms;
    while (!should_quit() && !clock_.stream_done() && !buffer_.read_available())
        timer_.wait_for(wait_delay);
}

} // end namespace detail

ClockReplay::ClockReplay(string_view name, const ProgState& prog_state,
    Detector& det, std::unique_ptr<StreamClock> clock)
        : TaskContainer{name, prog_state}, buffer_{}, clock_{std::move(clock)}
{
    assert(clock_);
    auto reader_task = std::make_unique<detail::ClockReplayReader>(
        std::string{name} + ".Reader", prog_state, buffer_, *clock_);
    auto player_task = std::make_unique<detail::ClockReplayPlayer>(
        std::string{name} + ".Player", prog_state, det, buffer_, *clock_);
    add_task(std::move(reader_task));
    add_task(std::move(player_task));
}

void ClockReplay::stop_hook()
{
    if (clock_)
        clock_->close();
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

