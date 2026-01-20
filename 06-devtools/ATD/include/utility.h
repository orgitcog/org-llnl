#ifndef UTILITY_H_
#define UTILITY_H_

#include <functional>
#include <boost/asio/completion_condition.hpp>
#include "common.h"
#include "date/date.h"

// Utility
//  This unit contains various utility functions such as a timer that provides
//  an interruptible wait method as well as various measurement and streaming
//  functions.
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

class InterruptibleTimer
{
  public:
    InterruptibleTimer() = default;
    ~InterruptibleTimer();
    InterruptibleTimer(const InterruptibleTimer&) = delete;
    InterruptibleTimer& operator=(const InterruptibleTimer&) = delete;
    
    using Clock = std::chrono::steady_clock;
    using Duration = Clock::duration;

    void cancel();
    void wait_for(const Duration& dur);

  private:
    mutex mutex_;
    condition_variable cond_;
};

class interruptible_xfr_at_least
{
  public:
    using Pred = std::function<bool()>;
    explicit interruptible_xfr_at_least(Pred pred, std::size_t minimum);

    template <typename Error>
    std::size_t operator()(const Error& err, std::size_t bytes_transferred)
    {
        if (pred_())
            return 0;
        return boost::asio::transfer_at_least(minimum_)(err, bytes_transferred);
    }

private:
    const Pred pred_;
    std::size_t minimum_;
};

template <class Rep, class Period>
std::string to_string(const std::chrono::duration<Rep, Period>& dur)
{
    using date::operator<<;
    std::ostringstream oss;
    oss << dur;
    return oss.str();
}

template <class Clock, class Duration>
auto to_string(const std::chrono::time_point<Clock, Duration>& tp) ->
    std::enable_if_t<std::is_same<Clock, std::chrono::system_clock>::value,
        std::string>
{
    return date::format("%F %T", tp);
}

template <class Clock, class Duration>
auto to_string(const std::chrono::time_point<Clock, Duration>& tp) ->
     std::enable_if_t<!std::is_same<Clock, std::chrono::system_clock>::value,
        std::string>
{
    // To convert between clocks (as in clock_cast expected for C++20),
    //  ensure that the original duration is maintained as adjusted for the
    //  (potentially) different epoch of the system clock
    //
    // (Note that this epoch offset is only computed once and memoized; this
    //  assumes that the system clock is effectively monotonic.
    //  TODO: evaluate this tradeoff versus performance hit of recomputing)
    static auto epoch_diff = Clock::now().time_since_epoch() -
        std::chrono::system_clock::now().time_since_epoch();
    auto sys_tp = std::chrono::time_point<std::chrono::system_clock, Duration>{
        tp.time_since_epoch() - epoch_diff};
    return to_string(sys_tp);
}

std::string to_upper(std::string s);

class Stopwatch
{
  public:
    using Clock = std::chrono::steady_clock;
    using Duration = Clock::duration;
    using ReportFn = std::function<void(Duration)>;
    // Generic version: calls a reporting function on-demand or from dtor with
    //  the elapsed time (if not previously reported)
    Stopwatch(ReportFn report_fn);
    // Convenience version: streams out the name then elapsed time to specified
    //  stream
    template <typename OStream>
    Stopwatch(std::string sw_name, OStream& os)
        : Stopwatch([sw_name = std::move(sw_name), &os](const auto& elapsed){
            os << sw_name << ": " << to_string(elapsed) << "\n"; })
    {
    }
    ~Stopwatch();
    Stopwatch(const Stopwatch&) = delete;
    Stopwatch& operator=(const Stopwatch&) = delete;

    Duration elapsed() const;
    void report();

  private:
    Clock::time_point start_;
    ReportFn report_fn_;
    bool reported_;
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

#endif // UTILITY_H_
