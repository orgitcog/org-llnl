#ifndef TCP_ALERT_HANDLER_H_
#define TCP_ALERT_HANDLER_H_

#include <boost/asio/ip/tcp.hpp>
#include "alerter.h"
#include "common.h"
#include "prog_state.h"
#include "task.h"
#include "utility.h"

// TCP Alert Handler
//  This module allows for the dissemination of alerts through a network socket
//  connection governed by the Transmission Control Protocol (TCP).  Alerts are
//  provided through a queue associated with the provided AlertListener; this
//  task waits until an alert becomes available, then transmits a serialized
//  summary of the alert to a connected TCP client (if any).
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

class TCPAlertHandler : public Task
{
  public:
    TCPAlertHandler(string_view name, const ProgState& prog_state,
            Alerter::Listener&& listener, unsigned short port_num);
    virtual ~TCPAlertHandler() = default;
    TCPAlertHandler(const TCPAlertHandler&) = delete;
    TCPAlertHandler& operator=(const TCPAlertHandler&) = delete;
    TCPAlertHandler(TCPAlertHandler&&) = default;

  protected:
    void run() override;
    void stop_hook() override;

  private:
    Alerter::Listener listener_;
    boost::asio::io_context io_context_;
    boost::asio::ip::tcp::socket socket_;
    mutable InterruptibleTimer timer_;
    unsigned short port_num_;

    bool accept(unsigned short port_num);
    void handle_new_cxn(const boost::asio::ip::tcp::endpoint& peer_endpoint,
        unsigned short port_num, const boost::system::error_code& err);
    bool wait_for_alert() const;
    bool process_single_alert();
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

#endif // TCP_ALERT_HANDLER_H_

