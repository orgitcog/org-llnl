#ifndef SERIAL_DEV_H_
#define SERIAL_DEV_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#include <boost/asio/serial_port.hpp>
#include <boost/asio/write.hpp>
#pragma GCC diagnostic pop
#include "common.h"
#include "prog_state.h"

// Serial Device
//  This unit provides an abstraction for a serial port via the ASIO library.
//  It also includes a buffer for data read from the raw device.
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

class SerialDev
{
  public:
    SerialDev(const ProgState& prog_state, string_view port_spec);
    SerialDev(const SerialDev&) = delete;
    SerialDev& operator=(const SerialDev&) = delete;
    SerialDev(SerialDev&&) = default;
    virtual ~SerialDev();

    void close(); /* throws */
    bool is_open() const;
    void open(); /* throws */

    // Reads should be followed by a call to consume with the number of bytes
    //  read
    void consume(std::size_t num_bytes);
    ustring_view read_bytes(std::size_t num_bytes); /* throws system_error */
    ustring_view read_until(string_view delim);
        /* throws boost::system::system_error */

    template <typename Buffer>
    void write(Buffer&& b) /* throws std::runtime_error */
    {
        if (!is_open() || should_quit())
            return;
        std::size_t bytes_to_write = b.size();
        std::size_t num_written = boost::asio::write(
            port_, std::forward<Buffer>(b));
        if (num_written != bytes_to_write)
            throw std::runtime_error{"Serial write failed"};
    }

  private:
    const ProgState& prog_state_;
    boost::asio::io_context context_;
    boost::asio::serial_port port_;
    std::string port_spec_;
    std::vector<unsigned char> buffer_;

    LoggerType& get_logger() const { return prog_state_.get_logger(); }
    void set_properties(const std::string& baud_rate,
        const std::string& char_size, const std::string& parity,
        const std::string& stop_bits, const std::string& flow_ctrl);
        /* throws out_of_range */
    bool should_quit() const { return prog_state_.should_quit(); }
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

#endif // SERIAL_DEV_H_
