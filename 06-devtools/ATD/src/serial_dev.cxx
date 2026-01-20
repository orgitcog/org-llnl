#include "serial_dev.h"
#include <boost/asio/read.hpp>
#include <boost/asio/read_until.hpp>
#include <regex>
#include <sstream>
#include "utility.h"

// Serial_Dev
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

namespace tmon
{

SerialDev::SerialDev(const ProgState& prog_state, string_view port_spec)
    : prog_state_{prog_state}, context_{}, port_{context_},
        port_spec_{port_spec}, buffer_{}
{
    if (!std::regex_match(port_spec_, port_spec_regex))
    {
        throw std::out_of_range{"Invalid serial port spec '" +
            std::string{port_spec_} + "'"};
    }
}

SerialDev::~SerialDev()
{
    close();
}

bool SerialDev::is_open() const
{
    return port_.is_open();
}

void SerialDev::open()
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
    catch(const std::exception& e)
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
    catch(const std::exception& e)
    {
        TM_LOG(error) << "Unable to set serial properties from spec '"
            << port_spec_ << "'";
        throw;
    }
}

void SerialDev::close()
{
    if (is_open())
        port_.close();
    context_.stop();
}

void SerialDev::set_properties(const std::string& baud_rate,
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

ustring_view SerialDev::read_bytes(std::size_t num_bytes)
    /* throws system_error */
{
    // Read more data if needed
    boost::system::error_code err;
    if (buffer_.size() < num_bytes)
    {
        std::size_t num_read = boost::asio::read(port_, 
            boost::asio::dynamic_buffer(buffer_), 
            interruptible_xfr_at_least{
                [this]{ return prog_state_.should_quit(); }, num_bytes},
                err);
        if (should_quit())
            return {}; // bail out ignoring any errors if quitting
        if (num_read == 0)
        {
            // Read failed
            throw boost::system::system_error(err);
        }
    }
    assert(buffer_.size() >= num_bytes);
    return ustring_view{buffer_.data(), num_bytes};
}

ustring_view SerialDev::read_until(string_view delim)
    /* throws boost::system::system_error */
{
    boost::system::error_code err;
    std::size_t bytes_to_delim_end = boost::asio::read_until(port_,
        boost::asio::dynamic_buffer(buffer_), delim, err);
    if (should_quit())
        return {}; // bail out ignoring any errors if quitting
    if (bytes_to_delim_end == 0)
    {
        if (err == boost::asio::error::eof)
            return {};
        // Read failed for a reason other than end-of-file; throw error
        throw boost::system::system_error(err);
    }
    assert(buffer_.size() >= bytes_to_delim_end);
    return ustring_view{buffer_.data(), bytes_to_delim_end};
}

void SerialDev::consume(std::size_t num_bytes)
{
    // If the number of bytes to consume exceeds the buffer size, the entire
    //  buffer is safely consumed
    boost::asio::dynamic_buffer(buffer_).consume(num_bytes);
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

