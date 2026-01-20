#ifndef WS_SERVER_H_
#define WS_SERVER_H_

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/websocket.hpp>
#include "alert_msg.h"
#include "common.h"
#include "detector.h"
#include "det_msg.h"
#include "prog_state.h"
#include "spmc.h"
#include "task.h"
#include "task_container.h"

// Websocket Server
//  This unit provides a Websocket interface for a status display, in
//      particular, hosting a server that can be connected to via JavaScript
//      and display status information regarding calibration, detection metrics,
//      and similar data within a (possibly remote) browser session
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

class WsServer : public TaskContainer
{
  public:
    using WsStream =
        boost::beast::websocket::stream<boost::asio::ip::tcp::socket>;

    WsServer(string_view name, const ProgState& prog_state, Detector& det,
        unsigned short port_num);
    WsServer(const WsServer&) = delete;
    WsServer& operator=(const WsServer&) = delete;
    WsServer(WsServer&&) = default;
    WsServer& operator=(WsServer&&) = default;

  protected:
    void stop_hook() override;

  private:
    unsigned short port_num_;

    using MsgListener = Detector::MsgQueue::Listener;
    MsgListener msg_listener_;

    std::vector<WsStream> listeners_;
    mutable mutex listener_mutex_;

    class AcceptorTask;
    class PublisherTask;

    void add_listener(WsStream&& s);
    void prune_closed_listeners();
    template <typename MsgType>
    bool publish_single_msg(const MsgType& x);
    bool publish_to_single_listener(WsStream& listener, string_view buf_str);
    void stop_all_listeners();
};

class WsServer::AcceptorTask : public Task
{
  public:
    AcceptorTask(string_view name, const ProgState& prog_state,
        WsServer& server, unsigned short port_num);
    ~AcceptorTask() override;

  protected:
    void start_hook() override;
    void run() override;
    void stop_hook() override;

  private:
    unsigned short port_num_;
    WsServer& server_;
    boost::asio::io_context io_context_;
    boost::asio::ip::tcp::acceptor acceptor_;

    bool accept();
    void handle_new_cxn(const boost::asio::ip::tcp::endpoint& peer_endpoint,
        boost::asio::ip::tcp::socket&& new_cxn_socket,
        const boost::system::error_code& err) const;
};

class WsServer::PublisherTask : public Task
{
  public:
    PublisherTask(string_view name, const ProgState& prog_state,
        WsServer& server);
    ~PublisherTask() override;

  protected:
    void run() override;
    void stop_hook() override;

  private:
    using MsgType = WsServer::MsgListener::MsgType;

    WsServer& server_;

    bool wait_for_msg() const;
    void publish_all() const;
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

#endif // WS_SERVER_H_

