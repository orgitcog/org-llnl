#include "ubx_base.h"
#include <boost/filesystem.hpp>
#include "alert_msg.h"
#include "detector.h"

// UBX_Base
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

using namespace std::string_literals;

namespace
{
    auto get_clock_class(const ProgState& prog_state,
        const std::string& opt_name) /* throws: std::range_error */
    {
        auto class_name = prog_state.get_opt_as<std::string>(opt_name);
        auto class_opt = tmon::GaussMarkovModel::get_class_by_name(class_name);
        if (!class_opt)
        {
            throw std::range_error{"Invalid clock class specified: " +
                class_name};
        }
        return *class_opt;
    }
} // end local namespace

namespace tmon
{

UbxBase::UbxBase(string_view name, const ProgState& prog_state, Detector& det,
        std::size_t gnss_idx)
    : Task{name, prog_state},
        clock_{det, prog_state, {std::string{name}, "Clock"s,
            get_clock_class(prog_state, "gnss.overall_class")}, gnss_idx},
        buffer_{}, detector_{det}
{
}

// Packet handler will dispatch to payload handlers with the appropriate
//  dynamic type
void UbxBase::handle(const UbxPkt& pkt)
{
    pkt.apply([this](const auto& payload){ this->handle(payload); });
}

void UbxBase::handle(const UbxPayload&)
{
    // nop; generic handler for payloads not specifically handled below
}

void UbxBase::handle(const UbxPayloadNavSol& navsol)
{
    long ecef_x_err{};
    long ecef_y_err{};
    long ecef_z_err{};
    if (has_opt("gnss.ecef_x"))
    {
        long ground_truth_x = get_opt_as<long>("gnss.ecef_x");
        ecef_x_err = std::abs(navsol.ecef_x - ground_truth_x);
    }
    if (has_opt("gnss.ecef_y"))
    {
        long ground_truth_y = get_opt_as<long>("gnss.ecef_y");
        ecef_y_err = std::abs(navsol.ecef_y - ground_truth_y);
    }
    if (has_opt("gnss.ecef_z"))
    {
        long ground_truth_z = get_opt_as<long>("gnss.ecef_z");
        ecef_z_err = std::abs(navsol.ecef_z - ground_truth_z);
    }

    bool bad_x = ecef_x_err > get_opt_as<long>("gnss.ecef_x_tol");
    bool bad_y = ecef_y_err > get_opt_as<long>("gnss.ecef_y_tol");
    bool bad_z = ecef_z_err > get_opt_as<long>("gnss.ecef_z_tol");

    auto pos_error_level = get_opt_as<Alert::Level>("alert.level.pos_error");
    if ((pos_error_level != Alert::Level::green) && (bad_x || bad_y || bad_z))
    {
        std::string reason_extra = "x-coordinate error (" +
            std::to_string(ecef_x_err) + " cm); y-coordinate error (" +
            std::to_string(ecef_y_err) + " cm); z-coordinate error (" +
            std::to_string(ecef_z_err) + " cm)";
        Alert alert{pos_error_level, Alert::ReasonCode::position_mismatch,
            std::move(reason_extra)};
        detector_.raise_alert(alert);
    }

    if (get_opt_as<bool>("gnss.ecef_log"))
    {
        TM_LOG(info) << get_name() << " ECEF coords x: " << navsol.ecef_x
            << ", y: " << navsol.ecef_y << ", z: " << navsol.ecef_z
            << "; dx: " << ecef_x_err << ", dy: " << ecef_y_err
            << ", dz: " << ecef_z_err;
    }
}

#if defined(ENABLE_TEST_INJECT)
void UbxBase::inject_test(UbxPayloadTimSmeas& smeas) const noexcept
{
    std::string task_name = get_name();
    int inject_idx = get_opt_as<int>("tmp.inject_gnss_idx");
    std::string inject_idx_str = "[" + std::to_string(inject_idx) + "]";
    if (task_name.compare(task_name.size() - inject_idx_str.length(),
                inject_idx_str.length(), inject_idx_str) != 0)
    {
        return; // no injection unless task index matches
    }

    std::string inject_file = get_opt_as<std::string>("tmp.inject_file");
    std::ifstream inject_stream{inject_file};
    int offset;
    if (!inject_stream || !(inject_stream >> offset))
    {
        TM_LOG(warning) << "Inject test file exists, but cannot be read";
        return;
    }
    inject_stream.close();

    std::chrono::nanoseconds offset_ns{offset};
    for (int i = 0; i < smeas.num_meas; ++i)
    {
        if (smeas.meas_array[i].src_id == 2)
        {
            // Inject test deviation into EXTINT0 measurement
            auto& curr_phase_offset = smeas.meas_array[i].phase_offset;
            TM_LOG(warning) << "Injecting test time deviation of "
                << offset_ns.count() << "ns into EXTINT0 ("
                << curr_phase_offset << " -> "
                << (curr_phase_offset + offset_ns.count())
                << ") of " << task_name;
            curr_phase_offset = curr_phase_offset + offset_ns.count();
        }
    }
}
#endif // defined(ENABLE_TEST_INJECT)

// Generate and send timing message to the detector based on UBX TIM-SMEAS
//  payload, which contains multiple phase and frequency comparisons
void UbxBase::handle(const UbxPayloadTimSmeas& smeas)
{
  #if defined(ENABLE_TEST_INJECT)
    std::string inject_file = get_opt_as<std::string>("tmp.inject_file");
    if (boost::filesystem::exists(inject_file))
    {
        UbxPayloadTimSmeas smeas_mod{smeas};
        inject_test(smeas_mod);
        clock_.handle_smeas(smeas_mod);
        return;
    }
  #endif // defined(ENABLE_TEST_INJECT)

    clock_.handle_smeas(smeas);
}

void UbxBase::handle(const UbxPayloadMonHw& hw)
{
    auto interference_level =
        get_opt_as<Alert::Level>("alert.level.interference");
    auto jam_state = hw.get_jam_state();
    bool bad_jam_state = (jam_state == UbxPayloadMonHw::JamState::warning) ||
        (jam_state == UbxPayloadMonHw::JamState::critical);
    if ((interference_level != Alert::Level::green) && bad_jam_state)
    {
        Alert alert{interference_level, Alert::ReasonCode::low_c_vs_noise,
            "GNSS interference monitor alert"};
        if (jam_state == UbxPayloadMonHw::JamState::critical)
            alert.reason_extra += " (critical - no fix)";
        detector_.raise_alert(alert);
    }

    // TODO Could also add comparison of the AGC level, CW jamming level, etc.
}

void UbxBase::handle(const UbxPayloadNavStatus& status)
{
    auto lost_fix_level =
        get_opt_as<Alert::Level>("alert.level.lost_fix");
    if ((lost_fix_level != Alert::Level::green) && !status.is_fix_ok())
    {
        Alert alert{lost_fix_level, Alert::ReasonCode::lost_fix, ""};
        detector_.raise_alert(alert);
    }

    auto invalid_tow_level =
        get_opt_as<Alert::Level>("alert.level.invalid_tow");
    if ((invalid_tow_level != Alert::Level::green) && !status.is_tow_set())
    {
        Alert alert{invalid_tow_level, Alert::ReasonCode::gnss_misc, ""};
        alert.reason_extra = "GNSS time-of-week not yet valid";
        detector_.raise_alert(alert);
    }

    auto spoof_det_level =
        get_opt_as<Alert::Level>("alert.level.external_spoof_det");
    auto spoof_det = status.get_spoof_det_state();
    bool bad_spoof_det =
        (spoof_det == UbxPayloadNavStatus::SpoofDetState::indicated) ||
        (spoof_det == UbxPayloadNavStatus::SpoofDetState::multi_indicated);
    if ((spoof_det_level != Alert::Level::green) && bad_spoof_det)
    {
        Alert alert{spoof_det_level, Alert::ReasonCode::gnss_inconsistency,
            "GNSS data inconsistency detected"};
        if (spoof_det == UbxPayloadNavStatus::SpoofDetState::multi_indicated)
            alert.reason_extra += " (multiple indications)";
        detector_.raise_alert(alert);
    }

    // TODO Could also check validity of week and/or GPS fix quality (and not
    //  just validity) here
}

void UbxBase::set_done_flag()
{
    clock_.set_done_flag();
    // TODO: Revisit how to consider composite clocks such as UbxClock -- this
    //  may mean a virtual hook within set_done_flag or a clock container class;
    //  for now, there is a separate method to propagate the done flag to
    //  subordinate clocks
    clock_.propagate_done_flag();
}

void UbxBase::start_hook()
{
    // Nop by default; derived classes may need to do init here
    //  (e.g., send init file to the receiver)
}

bool UbxBase::sync_clocks(const UbxBase& other_ubx,
        UbxClock::SubclockPair subclocks)
{
    TM_LOG(info) << "Synchronizing subclocks of " << get_name() << " and "
        << other_ubx.get_name();
    return clock_.sync_clocks(other_ubx.clock_, subclocks);
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

