#include "ubx_clock.h"
#include <cmath>
#include "date/date.h"
#include "parse_ubx.h"
#include "prog_state.h"

// UBX_Clock
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

namespace
{
    ClockDesc make_sub_desc(const ProgState& prog_state,
        const ClockDesc& base_desc, const std::string& sub_clk_name,
        const std::string& opt_name, std::size_t opt_idx)
            /* throws: std::range_error */
    {
        ClockDesc new_desc;
        new_desc.task_name = base_desc.task_name;
        new_desc.clock_name = base_desc.clock_name + "[" + sub_clk_name + "]";
        auto class_str = prog_state.get_opt_as<std::string>(opt_name, opt_idx);
        GaussMarkovModel new_gm_model{class_str};
        new_desc.gm_model = std::move(new_gm_model);
        return new_desc;
    }
} // end local namespace

namespace tmon
{

UbxClock::UbxClock(Detector& det, const ProgState& prog_state, ClockDesc names,
        std::size_t gnss_idx)
    : Clock{det, names}, 
        internal_clk_{det, make_sub_desc(prog_state, names, "Internal",
            "gnss.internal_class", gnss_idx)},
        gnss_clk_{det, make_sub_desc(prog_state, names, "GNSS",
            "gnss.gnss_class", gnss_idx)},
        extint0_clk_{det, make_sub_desc(prog_state, names, "EXTINT0",
            "gnss.extint0_class", gnss_idx)},
        extint1_clk_{det, make_sub_desc(prog_state, names, "EXTINT1",
            "gnss.extint1_class", gnss_idx)}
{
}

TimeMsg UbxClock::generate_message(const UbxPayloadTimSmeas& smeas) const
{
    TimeMsg msg{get_id()};

    // TODO: Make sure GNSS week number matches that of the local system.
    //  Since this is not transmitted in every message, assuming for now that
    //  the local week number is correct.

    // TODO: Consider using the gps_time feature of tz.h within the Date library
    //  to handle leap seconds

     date::sys_days msg_creation_day_flr{date::floor<date::days>(
        msg.msg_creation)};
     date::year_month_weekday msg_creation_ymw{msg_creation_day_flr};
     date::days day_into_week{static_cast<unsigned>(
        msg_creation_ymw.weekday())};
     auto msg_creation_wk_start_days = date::sys_days{msg_creation_ymw} -
        day_into_week;
     auto msg_creation_wk_start_time = 
        msg_creation_wk_start_days.time_since_epoch();
     msg.orig_timestamp = TimeMsg::OrigTimePointType{
        msg_creation_wk_start_time + std::chrono::milliseconds{smeas.itow}};

    // Include clock comparisons from the TIM-SMEAS payload
    auto get_smeas_phase_offset = [](auto& x){ 
        using picoseconds = std::chrono::duration<std::int64_t, std::pico>;
        picoseconds offset_ps{(x.phase_offset * 1000) + 
            ((1000 * x.phase_offset_frac) / 256)};
        return offset_ps;
        };

    for (int i = 0; i < smeas.num_meas; ++i)
    {
        assert(smeas.num_meas <= smeas.meas_array.size());
        if (!smeas.meas_array[i].is_phase_valid())
            continue;
        auto push_comp_for_clk = [&](const auto& clk, int i){
            const auto offset = get_smeas_phase_offset(smeas.meas_array[i]);
            const auto id = clk.get_id();
            const auto mu = clk.gauss_markov_model().mu;
            if (mu.array().isNaN().all())
                return; // ignore measurement if mu vector is all NaN
            msg.comparisons.push_back({id, offset});
            auto find_ret = std::find_if(cbegin(synced_subclock_list),
                cend(synced_subclock_list),
                [id](const auto& x){ return x.first == id; });
            if (find_ret != cend(synced_subclock_list))
            {
                // Mirror this measurement for synced clock
                ClockId synced_clk_id = find_ret->second;
                msg.comparisons.push_back({synced_clk_id, offset});
            }
        };
        switch (smeas.meas_array[i].src_id)
        {
            case 0:
                push_comp_for_clk(internal_clk_, i);
                break;
            case 1:
                push_comp_for_clk(gnss_clk_, i);
                break;
            case 2:
                push_comp_for_clk(extint0_clk_, i);
                break;
            case 3:
                push_comp_for_clk(extint1_clk_, i);
                break;
            default:
                // nop; ignored
                break;
        }
    }

    return msg;
}

void UbxClock::generate_and_send_message(const UbxPayloadTimSmeas& smeas)
{
    TimeMsg new_msg = generate_message(smeas);
    send_message(new_msg);
}

void UbxClock::handle_smeas(const UbxPayloadTimSmeas& smeas)
{
    generate_and_send_message(smeas);
}

void UbxClock::propagate_done_flag()
{
    internal_clk_.set_done_flag();
    gnss_clk_.set_done_flag();
    extint0_clk_.set_done_flag();
    extint1_clk_.set_done_flag();
}

// Synchronizes a subclock under this UbxClock with the subclock of another;
//  this is effected by injecting phase-offset measurements that mirror those of
//  the synchronized clock (as though it was part of the same measurement
//  system) when time messages are generated thereafter (as a result, it is only
//  necessary to call this function on one instance of UbxClock for each pair)
//
// Precondition: subclocks not already synchronized in this UbxClock
bool UbxClock::sync_clocks(const UbxClock& other_ubx_clk,
        SubclockPair subclocks)
{
    ClockId own_subclk_id, other_subclk_id;
    switch (subclocks.own_subclk)
    {
        case SubclockEnum::internal:
            own_subclk_id = internal_clk_.get_id();
            break;
        case SubclockEnum::gnss:
            own_subclk_id = gnss_clk_.get_id();
            break;
        case SubclockEnum::extint0:
            own_subclk_id = extint0_clk_.get_id();
            break;
        case SubclockEnum::extint1:
            own_subclk_id = extint1_clk_.get_id();
            break;
    }
    switch (subclocks.other_subclk)
    {
        case SubclockEnum::internal:
            other_subclk_id = other_ubx_clk.internal_clk_.get_id();
            break;
        case SubclockEnum::gnss:
            other_subclk_id = other_ubx_clk.gnss_clk_.get_id();
            break;
        case SubclockEnum::extint0:
            other_subclk_id = other_ubx_clk.extint0_clk_.get_id();
            break;
        case SubclockEnum::extint1:
            other_subclk_id = other_ubx_clk.extint1_clk_.get_id();
            break;
    }
    assert(own_subclk_id != other_subclk_id);
    auto new_subclk_pair = std::make_pair(own_subclk_id, other_subclk_id);
    if (std::find(begin(synced_subclock_list), end(synced_subclock_list),
            new_subclk_pair) != end(synced_subclock_list))
    {
        return false;
    }
    synced_subclock_list.push_back(std::move(new_subclk_pair));
    return true;
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

