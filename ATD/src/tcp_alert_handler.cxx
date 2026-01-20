#include "tcp_alert_handler.h"
#include <boost/asio/write.hpp>

// TCP Alert Handler
//  This module allows for the dissemination of alerts through a network socket
//  connection governed by the Transmission Control Protocol (TCP).  Alerts are
//  provided through a queue associated with the provided AlertListener; this
//  task waits until an alert becomes available, then transmits a serialized
//  summary of the alert to a connected TCP client (if any).
//
// TODO: This module has not yet been tested
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
TCPAlertHandler::TCPAlertHandler(string_view name, const ProgState& prog_state,
        Alerter::Listener&& listener, unsigned short port_num)
    : Task{name, prog_state}, listener_{std::move(listener)}, io_context_{},
        socket_{io_context_}, timer_{}, port_num_{port_num}
{
}

void TCPAlertHandler::run()
{
    bool accept_ok = accept(port_num_);
    if (!accept_ok || !socket_.is_open())
    {
        if (!should_quit())
        {
            TM_LOG(error) << "Unable to accept socket cxn for alert handler";
        }
        return;
    }

    while (wait_for_alert())
        process_single_alert();
}

void TCPAlertHandler::stop_hook()
{
    TM_LOG(debug) << "TCP alert handler stopping; interrupting blocking calls";
    timer_.cancel();
    if (socket_.is_open())
    {
        socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
        socket_.close();
    }
    io_context_.stop();
    listener_.cancel();
}

void TCPAlertHandler::handle_new_cxn(
    const boost::asio::ip::tcp::endpoint& peer_endpoint,
    unsigned short port_num, const boost::system::error_code& err)
{
    if (!err)
    {
        TM_LOG(info) << "Accepted alert handler socket connection "
            << "from " << peer_endpoint << " to port " << port_num;
    }
    else
    {
        TM_LOG(error) << "Failed to accept alert handler socket "
            << "connection from " << peer_endpoint << " to port "
            << port_num << " (err " << err << ")";
    }
}

bool TCPAlertHandler::accept(unsigned short port_num)
{
    using boost::asio::ip::tcp;
    tcp::endpoint endpoint{tcp::v4(), port_num};
    tcp::acceptor acceptor{io_context_, endpoint};
    tcp::endpoint peer_endpoint;
    acceptor.set_option(tcp::socket::reuse_address{true});
    auto new_cxn_handler =
        [peer_endpoint, port_num, this](const boost::system::error_code& err){
            this->handle_new_cxn(peer_endpoint, port_num, err); };
    acceptor.async_accept(socket_, peer_endpoint, new_cxn_handler);
    //acceptor.wait(tcp::acceptor::wait_read);
    while (!socket_.is_open() && !should_quit())
    {
        // TODO: Improve this wait loop
        io_context_.run_for(std::chrono::milliseconds(50));
    }
    return socket_.is_open();
}

bool TCPAlertHandler::wait_for_alert() const
{
    return listener_.wait([this]{ return this->should_quit(); });
}

bool TCPAlertHandler::process_single_alert()
{
    assert(!listener_.empty());
    if (!socket_.is_open())
        return false;
    optional<Alert> front_alert = listener_.pop_front();
    assert(front_alert);
    std::string alert_str = describe(*front_alert) + '\n';
    std::size_t num_written = boost::asio::write(socket_,
        boost::asio::buffer(alert_str));
    if (num_written != alert_str.length())
    {
        TM_LOG(warning) << "Unable to write alert to TCP socket (wrote "
            << num_written << " vs. " << alert_str.length() << " expected)";
        return false;
    }
    TM_LOG(info) << "Sent alert to TCP socket (local port " << port_num_ << ")";
    return true;
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

