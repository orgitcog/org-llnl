#ifndef UBX_CLOCK_H_
#define UBX_CLOCK_H_

#include "common.h"
#include "clock.h"
#include "time_msg.h"

// UBX Clock
//  This unit represents a composite clock that reports through the UBX M8
//  protocol (cf. parse_ubx.h).  It is typically contained in a task derived
//  from UbxBase.  It contains several subordinate clocks, such as the clock
//  representing the internal timeframe, the clock synchronized to the GNSS
//  timeframe, and clocks connected to external inputs.
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

class UbxPayloadTimSmeas;

namespace tmon
{

class ProgState;
class UbxClock : public Clock
{
  public:
    UbxClock(Detector& det, const ProgState& prog_state, ClockDesc names,
        std::size_t gnss_idx);
    ~UbxClock() override = default;

    void handle_smeas(const UbxPayloadTimSmeas& smeas);
    void propagate_done_flag();

    enum class SubclockEnum { internal, gnss, extint0, extint1 };
    struct SubclockPair
    {
        SubclockEnum own_subclk;
        SubclockEnum other_subclk;
    };
    bool sync_clocks(const UbxClock& other_ubx_clk, SubclockPair subclocks);

  private:
    Clock internal_clk_;
    Clock gnss_clk_;
    Clock extint0_clk_;
    Clock extint1_clk_;
    // Could add host clocks here (source IDs 4 and 5), but not yet used

    std::vector<std::pair<ClockId, ClockId>> synced_subclock_list;

    TimeMsg generate_message(const UbxPayloadTimSmeas& smeas) const;
    void generate_and_send_message(const UbxPayloadTimSmeas& smeas);
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

#endif // UBX_CLOCK_H_
