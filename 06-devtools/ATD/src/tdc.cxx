#include "tdc.h"
#include <array>
#include <cstdio>
#include <regex>

// TDC (Time-to-Digital Converter, TI TDC7201) Interface
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

    std::array<unsigned char, 3> print_hex_octet(std::uint8_t x)
    {
        std::array<unsigned char, 3> ret;
        std::snprintf(reinterpret_cast<char*>(ret.data()), 3, "%02X", x);
        return ret;
    }
    std::array<unsigned char, 5> print_hex_octets(std::uint8_t x1,
        std::uint8_t x2)
    {
        std::array<unsigned char, 5> ret;
        std::snprintf(reinterpret_cast<char*>(ret.data()), 3, "%02X", x1);
        std::snprintf(reinterpret_cast<char*>(&ret[2]), 3, "%02X", x2);
        return ret;
    }
} // end local namespace

namespace tmon
{

Tdc::Tdc(const ProgState& prog_state, string_view port_spec, double ref_clk_hz)
    : ref_clk_hz_{ref_clk_hz}, serial_dev_{prog_state, port_spec}
{
    assert(ref_clk_hz_ > 0);

    serial_dev_.open();

    TdcConfig cfg{};
    cfg.meas_mode = TdcMeasMode::mode2;
    cfg.avg_cycles = 128;
    cfg.stop_mask_cycles = (1 << 16) - 3; // 2^16 - 2 cycles is max possible
    setup(cfg);
}

picoseconds Tdc::get_avg_meas_mode2()
{
    auto meas1 = get_unit_meas_mode2(TdcUnit::tdc1);
    auto meas2 = get_unit_meas_mode2(TdcUnit::tdc2);
    // Transforming average to reduce the risk of overflow:
    // (meas1 + meas2) / 2 <==> meas1 + (meas2 - meas1) / 2
    return meas1 + (meas2 - meas1) / 2;
}

picoseconds Tdc::get_unit_meas_mode2(TdcUnit unit)
{
    set_unit(unit);

    auto cal1 = get_register(TdcRegister::TDCx_CALIBRATION1);
    auto cal2 = get_register(TdcRegister::TDCx_CALIBRATION2);
    auto cal2_periods = get_register(TdcRegister::TDCx_CONFIG2);

    auto time1 = get_register(TdcRegister::TDCx_TIME1);
    auto time2 = get_register(TdcRegister::TDCx_TIME2);

    auto clk_count1 = get_register(TdcRegister::TDCx_CLOCK_COUNT1);

    double cal_count = (cal2 - cal1) / (cal2_periods - 1.0);
    double norm_lsb = 1.0 / (ref_clk_hz_ * cal_count);
    double meas_sec = norm_lsb * (time1 - time2) + clk_count1 / ref_clk_hz_;
    picoseconds::rep meas_ps = meas_sec * 1.0e12;
    return picoseconds{meas_ps};
}

picoseconds Tdc::get_meas_from_stream()
{
    static const std::basic_regex<unsigned char> timing_regex{
        reinterpret_cast<const unsigned char*>(
            R"(Start_to_Stop\[(\d+)\]:\s+([0-9.eE-]+|nan|NaN))")};

    ustring_view line_view;
    using std::cbegin;
    using std::cend;
    while (!std::regex_search(cbegin(line_view), cend(line_view), timing_regex))
    {
        serial_dev_.consume(line_view.size());
        line_view = serial_dev_.read_until("\r\n");
    }

    constexpr std::size_t max_stops{5};
    std::array<optional<double>, max_stops> read_ns;
    std::size_t n{0};
    using uregex_iterator = std::regex_iterator<ustring_view::const_iterator>;
    for (uregex_iterator i{cbegin(line_view), cend(line_view), timing_regex};
         (i != uregex_iterator{}) && (n < read_ns.size());
         ++i, ++n)
    {
        assert(i->size() >= 2);
        std::string time_str{reinterpret_cast<const char*>(i->str(2).c_str())};
        read_ns[n] = std::stod(time_str);
    }
    serial_dev_.consume(line_view.size());
    if (!read_ns[0])
        throw std::runtime_error("Failed to parse TDC meas");
    picoseconds::rep read_ps = *read_ns[0] * 1e3;
    picoseconds ret{read_ps};
    return ret;
}

TdcResponse Tdc::send_command(TdcUsbCmd cmd, std::uint8_t cmd_payload)
{
    std::array<unsigned char, 3> cmd_payload_chars =
        ::print_hex_octet(cmd_payload);
    return send_command(cmd, ustring_view{cmd_payload_chars.data(), 2});
}

TdcResponse Tdc::send_command(TdcUsbCmd cmd, std::uint8_t cmd_payload1,
    std::uint8_t cmd_payload2)
{
    std::array<unsigned char, 5> cmd_payload_chars =
        ::print_hex_octets(cmd_payload1, cmd_payload2);
    return send_command(cmd, ustring_view{cmd_payload_chars.data(), 4});
}

TdcResponse Tdc::send_command(TdcUsbCmd cmd, ustring_view cmd_payload)
{
    std::uint8_t cmd_octet{static_cast<std::uint8_t>(cmd)};
    std::array<unsigned char, 3> cmd_chars = ::print_hex_octet(cmd_octet);
    serial_dev_.write(boost::asio::buffer(cmd_chars, 2));
    serial_dev_.write(boost::asio::buffer(cmd_payload));

    // Read response
    // First, command is echoed back in first 8 bytes, then the response follows
    //  in up to 25 additional bytes
    TdcResponse response{};
    const std::size_t echo_len{8};
    serial_dev_.read_bytes(echo_len);
    serial_dev_.consume(echo_len);
    ustring_view response_view = serial_dev_.read_bytes(response.size());
    serial_dev_.consume(response.size());
    using std::begin;
    using std::cbegin;
    using std::cend;
    std::copy(cbegin(response_view), cend(response_view), begin(response));
    return response;
}

void Tdc::set_unit(TdcUnit unit)
{
    if (selected_unit_ && (*selected_unit_ == unit))
        return; // nop; unit is already selected
    // See Command_Set_Current_TDC_For_Access
    std::uint8_t unit_num = (unit == TdcUnit::tdc1 ? 0 : 1);
    send_command(TdcUsbCmd::Set_Current_TDC_For_Access, unit_num);
    selected_unit_ = unit;    
}

void Tdc::setup_unit(TdcUnit unit, const TdcConfig& cfg)
{
    set_unit(unit);

    assert(cfg.valid(ref_clk_hz_) && "Invalid cfg for TDC");

    unsigned char cfg1{};
    if (cfg.meas_mode == TdcMeasMode::mode2)
        cfg1 |= (1 << 1);
    if (cfg.start_edge == TdcEdge::falling)
        cfg1 |= (1 << 3);
    if (cfg.stop_edge == TdcEdge::falling)
        cfg1 |= (1 << 4);
    if (cfg.trig_edge == TdcEdge::falling)
        cfg1 |= (1 << 5);
    // no parity currently, bit 6 = 0
    if (cfg.force_cal == TdcCal::always)
        cfg1 |= (1 << 7);

    unsigned char cfg2{};
    cfg2 |= (cfg.num_stops - 1);
    switch (cfg.avg_cycles)
    {
        case 1:
            break; // nop
        case 2:
            cfg2 |= (1 << 3);
            break;
        case 4:
            cfg2 |= (2 << 3);
            break;
        case 8:
            cfg2 |= (3 << 3);
            break;
        case 16:
            cfg2 |= (4 << 3);
            break;
        case 32:
            cfg2 |= (5 << 3);
            break;
        case 64:
            cfg2 |= (6 << 3);
            break;
        case 128:
            cfg2 |= (7 << 3);
            break;
        default:
            assert(!"Invalid avg cycles");
    }
    switch (cfg.cal2_periods)
    {
        case 2:
            break; // nop
        case 10:
            cfg2 |= (1 << 6);
            break;
        case 20:
            cfg2 |= (2 << 6);
            break;
        case 40:
            cfg2 |= (3 << 6);
            break;
        default:
            assert(!"Invalid cal periods");
    }

    unsigned char stop_mask_low = (cfg.stop_mask_cycles & 0xFF);
    unsigned char stop_mask_high = ((cfg.stop_mask_cycles >> 8) & 0xFF);

    set_register(TdcRegister::TDCx_CONFIG1, cfg1);
    set_register(TdcRegister::TDCx_CONFIG2, cfg2);
    set_register(TdcRegister::TDCx_CLOCK_CNTR_STOP_MASK_L, stop_mask_low);
    set_register(TdcRegister::TDCx_CLOCK_CNTR_STOP_MASK_H, stop_mask_high);
}

void Tdc::setup(const TdcConfig& cfg)
{
    assert(cfg.valid(ref_clk_hz_) && "Invalid cfg for TDC");

    std::uint8_t tdc7201_idx{1}; // 1 = TDC7201
    send_command(TdcUsbCmd::Set_Device, tdc7201_idx);
    send_command(TdcUsbCmd::Set_Fast_Trig, {});
    std::uint8_t use_both{1}; // 0 = current TDC only, 1 = both TDCs
    send_command(TdcUsbCmd::Set_TDC_Graph_Select, use_both);

    setup_unit(TdcUnit::tdc1, cfg);
    setup_unit(TdcUnit::tdc2, cfg);
}

std::uint32_t Tdc::get_register(TdcRegister reg)
{
    // See Command_TDC720x_SPI_Byte_Read and Command_TDC720x_SPI_Word_Read

    std::uint8_t reg_addr{static_cast<std::uint8_t>(reg)};
    if (reg <= TdcRegister::TDCx_CLOCK_CNTR_STOP_MASK_L)
    {
        TdcResponse response = send_command(TdcUsbCmd::SPI_Byte_Read, reg_addr);
        return response[0];
    }
    else
    {
        TdcResponse response = send_command(TdcUsbCmd::SPI_Word_Read, reg_addr);
        std::uint32_t ret = response[0] + (response[1] << 8) +
            (response[2] << 16);
        assert((ret >> 24) == 0);
        return ret;
    }
}

void Tdc::set_register(TdcRegister reg, std::uint32_t val)
{
    if (reg <= TdcRegister::TDCx_CLOCK_CNTR_STOP_MASK_L)
    {
        // Command_TDC720x_SPI_Byte_Write
        assert((val >> 8) == 0);
        std::uint8_t reg_addr{static_cast<std::uint8_t>(reg)};
        std::uint8_t byte_to_write = val & 0x0F;
        send_command(TdcUsbCmd::SPI_Byte_Write, reg_addr, byte_to_write);
    }
    else
    {
        assert((val >> 24) == 0);
        // TODO
        // See Command_TDC720x_SPI_Word_Write
        assert(!"SPI word writes not yet implemented");
    }
}

void Tdc::start_meas_stream()
{
    // See Command_Enable_UART_Stream
    send_command(TdcUsbCmd::Enable_UART_Stream, {});
}

#if 0
void Tdc::start_new_meas(TdcUnit unit)
{
    set_unit(unit);
    assert(!"Not impl");
}

picoseconds Tdc::get_meas()
{
    start_new_meas(TdcUnit::tdc1);
    start_new_meas(TdcUnit::tdc2);

    assert(!"Not impl");

    auto avg_meas = get_avg_meas_mode2();
    return avg_meas;
}
#endif // TODO

bool TdcConfig::valid(double ref_clk_hz) const
{
    bool avg_cycles_ok = false;
    bool cal2_periods_ok = (cal2_periods == 2) || (cal2_periods == 10) ||
        (cal2_periods == 20) || (cal2_periods == 40);
    bool num_stops_ok = (num_stops >= 1) && (num_stops <= 5);
    bool stop_mask_ok = false;
    switch (avg_cycles)
    {
        case 1:
        case 2:
        case 4:
        case 8:
        case 16:
        case 32:
        case 64:
        case 128:
            avg_cycles_ok = true;
            break;
        default:
            avg_cycles_ok = false;
            break;
    }

    if (meas_mode == TdcMeasMode::mode1)
    {
        // Meas. Mode 1: Maximum time between start and stop: 2000 ns
        stop_mask_ok = (stop_mask_cycles >= 0) &&
            (stop_mask_cycles <= (ref_clk_hz * 2000e-9));
    }
    else
    {
        // Meas. Mode 2: Maximum time between start and stop: (2^16 - 2) cycles
        stop_mask_ok = (stop_mask_cycles >= 0) &&
            (stop_mask_cycles <= ((1 << 16) - 2));
    }
    return avg_cycles_ok && cal2_periods_ok && num_stops_ok && stop_mask_ok;
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

