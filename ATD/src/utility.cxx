#include "utility.h"
#include <algorithm>
#include <cctype>
#include <string>

// Utility
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

InterruptibleTimer::~InterruptibleTimer()
{
    cancel();
}

void InterruptibleTimer::cancel()
{
    cond_.notify_all();
}

void InterruptibleTimer::wait_for(const Duration& dur)
{
    if (dur <= Duration::zero())
        return;

    // If the wait duration is short enough, use a spin-loop instead of
    //  acquiring a lock, context switching, etc.
    using namespace std::chrono_literals;
    constexpr Duration max_spin_dur = 1ms;
    if (dur <= max_spin_dur)
    {
        static_assert(Clock::is_steady, "Steady clock required");
        auto done_time = Clock::now() + dur;
        while (Clock::now() < done_time)
        {
            // nop; spin-loop
        }
        return;
    }

    // For durations too long to use the spin-loop, perform an interruptible
    //  wait on the condition variable
    unique_lock<mutex> lock{mutex_};
    cond_.wait_for(lock, boost_duration_cast(dur));
}

interruptible_xfr_at_least::interruptible_xfr_at_least(Pred pred,
        std::size_t minimum)
    : pred_{std::move(pred)}, minimum_{minimum}
{
}

std::string to_upper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
        [](char c){ return std::toupper(c); });
    return s;
}

Stopwatch::Stopwatch(ReportFn report_fn)
    : start_{Clock::now()}, report_fn_{report_fn}, reported_{false}
{
}

Stopwatch::~Stopwatch()
{
    if (!reported_)
        report();
}

Stopwatch::Duration Stopwatch::elapsed() const
{
    return (Clock::now() - start_);
}

void Stopwatch::report()
{
    report_fn_(elapsed());
    reported_ = true;
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

