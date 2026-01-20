#include "ubx_serial.h"
#include <regex>
#include <sstream>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#include <boost/asio/serial_port_base.hpp>
#include <boost/asio/write.hpp>
#pragma GCC diagnostic pop
#include "parse_ubx.h"

// UBX_Serial
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
std::regex port_spec_regex{
    R"(([^|]+)(\|(\d+),(\d),(N|O|E),(1|2),(none|sw|hw))?)"};
} // end local namespace

UbxSerial::UbxSerial(string_view name, const ProgState& prog_state,
    Detector& det, string_view port_spec, std::size_t gnss_idx)
    : UbxBase{name, prog_state, det, gnss_idx}, context_{}, port_{context_},
        port_spec_{port_spec}
{
    if (!std::regex_match(port_spec_, port_spec_regex))
    {
        throw std::out_of_range{"Invalid serial port spec '" +
            std::string{port_spec_} + "'"};
    }
}

void UbxSerial::configure()
{
    // TODO: Evaluate configuration for parameterization opportunities
    TM_LOG(debug) << "Configuring UbxSerial";
    // CFG-PRT: UBX + NMEA input, UBX output protocols for USB
    std::array<unsigned char, 24> cfg_prt_usb = {0x06, 0x00, 20, 0x00,
        0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00};
    // CFG-PRT: UBX + NMEA input, UBX output protocols for UART
    std::array<unsigned char, 24> cfg_prt_uart = {0x06, 0x00, 20, 0x00,
        0x01, 0x00, 0x00, 0x00,
        0xC0, 0x04, 0x00, 0x00, 0x80, 0x25, 0x00, 0x00,
        0x03, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00};

    // CFG-MSG: TIM-SMEAS (0x0D 0x13) enable
    std::array<unsigned char, 7> cfg_msg = {0x06, 0x01, 3, 0x00,
        0x0D, 0x13, 0x01};

    send_ubx_pkt(ustring_view{cfg_prt_usb.data(), cfg_prt_usb.size()});
    send_ubx_pkt(ustring_view{cfg_prt_uart.data(), cfg_prt_uart.size()});
    send_ubx_pkt(ustring_view{cfg_msg.data(), cfg_msg.size()});
    TM_LOG(debug) << "UbxSerial config complete";
}

void UbxSerial::read_and_handle() /* throws system_error */
try
{
    BOOST_LOG_FUNCTION();
    if (!port_.is_open())
        return;
    read_until_sync(port_);
    if (should_quit())
        return;
    UbxPkt new_pkt = read_pkt(port_);
    if (should_quit())
        return;
    handle(new_pkt);
    TM_LOG(trace) << "UbxSerial new pkt: " << new_pkt.describe();
}
catch (const boost::system::system_error& err)
{
    // EOF errors can be silently ignored; others are rethrown
    if (err.code() != boost::asio::error::eof)
        throw;
}

void UbxSerial::run()
{
    while (!should_quit() && port_.is_open())
    {
        read_and_handle();
    }
}

void UbxSerial::start_hook()
{
    std::smatch match_res;
    bool match_ok = std::regex_match(port_spec_, match_res, port_spec_regex);
    assert(match_ok); // already checked in ctor
    assert(match_res.size() == 8);
    const auto& port_name = match_res[1];

    try
    {
        port_.open(port_name);
    }
    catch(std::exception& e)
    {
        TM_LOG(error) << "Unable to open serial port '" << port_name
            << "' from spec '" << port_spec_ << "'";
        throw;
    }

    try
    {
        if (match_res[2].matched)
        {
            set_properties(match_res[3], match_res[4], match_res[5],
                match_res[6], match_res[7]);
        }
    }
    catch(std::exception& e)
    {
        TM_LOG(error) << "Unable to set serial properties from spec '"
            << port_spec_ << "'";
        throw;
    }

    configure();
}

void UbxSerial::stop_hook()
{
    if (port_.is_open())
        port_.close();
    context_.stop();
}

// Prepends header, appends checksum
void UbxSerial::send_ubx_pkt(ustring_view pkt_view)
{
    unsigned char hdr[2] = {UbxPkt::sync_char1, UbxPkt::sync_char2};
    std::uint16_t cksum = Ubx::checksum(pkt_view);
    // Checksum bytes are little-endian
    unsigned char cksum_char1 = cksum & 0xFF;
    unsigned char cksum_char2 = cksum >> 8;
    unsigned char cksum_buf[2] = {cksum_char1, cksum_char2};

    write(boost::asio::buffer(hdr));
    write(boost::asio::buffer(pkt_view));
    write(boost::asio::buffer(cksum_buf));
}

void UbxSerial::set_properties(const std::string& baud_rate,
    const std::string& char_size, const std::string& parity,
    const std::string& stop_bits, const std::string& flow_ctrl)
    /* throws out_of_range */
{
    using boost::asio::serial_port_base;

    port_.set_option(serial_port_base::baud_rate(std::stoi(baud_rate)));

    switch (std::stoi(char_size))
    {
        case 7:
            port_.set_option(serial_port_base::character_size(7));
            break;
        case 8:
            port_.set_option(serial_port_base::character_size(8));
            break;
        default:
            throw std::out_of_range("Bad char size '" + char_size + "'");
    }

    assert(!parity.empty());
    switch (parity[0])
    {
        case 'N':
            port_.set_option(serial_port_base::parity(
                serial_port_base::parity::none));
            break;
        case 'O':
            port_.set_option(serial_port_base::parity(
                serial_port_base::parity::odd));
            break;
        case 'E':
            port_.set_option(serial_port_base::parity(
                serial_port_base::parity::even));
            break;
        default:
            throw std::out_of_range("Bad parity '" + parity + "'");
    }

    switch (std::stoi(stop_bits))
    {
        case 1:
            port_.set_option(serial_port_base::stop_bits(
                serial_port_base::stop_bits::one));
            break;
        case 2:
            port_.set_option(serial_port_base::stop_bits(
                serial_port_base::stop_bits::two));
            break;
        default:
            throw std::out_of_range("Bad stop bits '" + stop_bits + "'");
    }

    if (flow_ctrl == "none")
    {
        port_.set_option(serial_port_base::flow_control(
            serial_port_base::flow_control::none));
    }
    else if (flow_ctrl == "sw")
    {
        port_.set_option(serial_port_base::flow_control(
            serial_port_base::flow_control::software));
    }
    else if (flow_ctrl == "hw")
    {
        port_.set_option(serial_port_base::flow_control(
            serial_port_base::flow_control::hardware));
    }
    else
    {
        throw std::out_of_range("Bad flow ctrl '" + flow_ctrl + "'");
    }
}

template <typename Buffer>
void UbxSerial::write(Buffer&& b)
{
    if (!port_.is_open() || should_quit())
        return;
    auto num_written = boost::asio::write(port_, std::forward<Buffer>(b));
    if (num_written != boost::asio::buffer_size(b))
    {
        TM_LOG(warning) << "UbxSerial partial write of " << num_written
            << " (expected " << boost::asio::buffer_size(b) << ")";
    }
}

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

