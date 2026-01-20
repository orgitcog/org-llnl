#ifndef UBX_SERIAL_H_
#define UBX_SERIAL_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#include <boost/asio/serial_port.hpp>
#pragma GCC diagnostic pop
#include "common.h"
#include "ubx_base.h"
#include "task.h"

// UBX Serial
//  This unit supports processing the UBX M8 binary protocol (cf. parse_ubx.h)
//  via a serial port interface.  This is generally a thin wrapper over UbxBase,
//  although it does add some configuration options for the serial port as well
//  as some initialization for the device.
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
class UbxSerial : public UbxBase
{
  public:
    UbxSerial(string_view name, const ProgState& prog_state, Detector& det,
        string_view port_spec, std::size_t gnss_idx);
    ~UbxSerial() override = default;
    UbxSerial(const UbxSerial&) = delete;
    UbxSerial& operator=(const UbxSerial&) = delete;

  protected:
    void start_hook() override;
    void run() override;
    void stop_hook() override;

  private:
    boost::asio::io_context context_;
    boost::asio::serial_port port_;
    std::string port_spec_;

    void configure();
    void read_and_handle();
    std::string read_nmea();
    void set_properties(const std::string& baud_rate,
        const std::string& char_size, const std::string& parity,
        const std::string& stop_bits, const std::string& flow_ctrl);
        /* throws out_of_range */
    void send_ubx_pkt(ustring_view pkt_view);
    template <typename Buffer>
    void write(Buffer&& b);
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

#endif // UBX_SERIAL_H_

