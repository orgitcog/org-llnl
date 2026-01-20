#include "ws_server.h"
#include <chrono>
#include <boost/asio/write.hpp>
#include <date/date.h>
#include "detector.h"
#include "utility.h"

// WebSocket Server
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
template <typename TimePoint>
std::string to_iso8601_string(const TimePoint& tp)
{
    // Transform a string like "2019-10-24 11:04:09.203985" to
    //  "2019-10-24T11:04:09.203Z" in accordance with standard ISO 8601
    //  used by JSON (note that the milliseconds are truncated)
    std::string datetime = to_string(tp);
    datetime.replace(10, 1, "T"); // space -> T (for Time)
    datetime.erase(23);           // truncate fractional ms
    datetime.push_back('Z');      // Z = Zulu = GMT
    return datetime;
}

std::string to_json_message(const Alert& alert)
{
    return R"({ "msg_type": "alert", "timestamp": ")" +
            to_iso8601_string(alert.timestamp) +
        R"(", "level": ")" + to_string(alert.level) +
        R"(", "reason_code": )" +
            std::to_string(static_cast<int>(alert.reason_code)) +
        R"(, "reason_desc": ")" + describe(alert.reason_code) +
        R"(", "reason_extra": ")" + alert.reason_extra + R"(" })";
}

std::string to_json_message(const CalibrationMsg& cal_msg)
{
    return R"({ "msg_type": "status", "timestamp": ")" +
            to_iso8601_string(cal_msg.msg_creation) +
        R"(", "component": "cal", "event": ")" + to_string(cal_msg.evt_type) +
        R"(" })";
}

std::string to_json_message(const DetMetricMsg& det_met_msg)
{
    return R"({ "msg_type": "metric", "timestamp": ")" +
            to_iso8601_string(det_met_msg.msg_creation) +
        R"(", "det_metric": )" + std::to_string(det_met_msg.det_metric) +
        R"(, "threshold": )" + std::to_string(det_met_msg.det_threshold) +
        R"( })";
}

class WebSocketSyncStreamAdapter
{
  public:
    WebSocketSyncStreamAdapter(WsServer::WsStream& wss)
        : wss_{wss}
    {
    }
    template <typename ConstBufSeq, typename ErrorCode>
    std::size_t write_some(ConstBufSeq&& cbs, ErrorCode&& ec)
    {
        bool fin{true};
        return wss_.write_some(fin, std::forward<ConstBufSeq>(cbs),
                std::forward<ErrorCode>(ec));
    }
  private:
    WsServer::WsStream& wss_;
};

} // end anonymous namespace

using tcp = boost::asio::ip::tcp;

namespace tmon
{

WsServer::WsServer(string_view name, const ProgState& prog_state,
        Detector& det, unsigned short port_num)
    : TaskContainer{name, prog_state}, port_num_{port_num},
        msg_listener_{det.register_msg_queue_listener()}
{
    add_task(std::make_unique<WsServer::AcceptorTask>("WebSocket-Acceptor",
        prog_state, *this, port_num_));
    add_task(std::make_unique<WsServer::PublisherTask>("WebSocket-Publisher",
        prog_state, *this));
}

void WsServer::stop_all_listeners()
{
    unique_lock<mutex> guard{listener_mutex_};
    for (auto& listener : listeners_)
    {
        namespace websocket = boost::beast::websocket;
        websocket::close_reason reason{websocket::close_code::normal};
        listener.close(reason);
    }
}

void WsServer::stop_hook()
{
    TM_LOG(debug) << "WebSocket server stopping; interrupting blocking calls";
    stop_all_listeners();
}

void WsServer::add_listener(WsStream&& s)
{
    unique_lock<mutex> guard{listener_mutex_};
    listeners_.push_back(std::move(s));
}

bool WsServer::publish_to_single_listener(WsStream& listener,
        string_view buf_str)
{
    boost::beast::error_code ec;
    ::WebSocketSyncStreamAdapter adapted_listener{listener};
    std::size_t num_written = boost::asio::write(
            adapted_listener,
            boost::asio::buffer(buf_str),
            interruptible_xfr_at_least{
                [this]{ return this->should_quit(); }, buf_str.size()},
            ec);
    if (num_written != buf_str.length())
    {
        TM_LOG(warning) << "Unable to write to WebSocket (wrote "
            << num_written << " vs. " << buf_str.length()
            << " expected); error: " << ec;
        return false;
    }
    return true;
}

void WsServer::prune_closed_listeners()
{
    unique_lock<mutex> guard{listener_mutex_};
    listeners_.erase(std::remove_if(begin(listeners_), end(listeners_),
                [](const auto& x) { return !x.is_open(); }), end(listeners_));
}

template <typename MsgType>
bool WsServer::publish_single_msg(const MsgType& x)
{
    std::string buf_str = to_json_message(x);

    // Send string to each listener
    bool ok{true};
    {
        unique_lock<mutex> guard{listener_mutex_};
        for (auto& listener : listeners_)
        {
            if (listener.is_open())
                ok = publish_to_single_listener(listener, buf_str) && ok;
        }
    }
    return ok;
}

WsServer::AcceptorTask::AcceptorTask(string_view name,
        const ProgState& prog_state, WsServer& server, unsigned short port_num)
    : Task{name, prog_state}, port_num_{port_num}, server_{server},
        io_context_{}, acceptor_{io_context_}
{
}

WsServer::AcceptorTask::~AcceptorTask()
{
}

void WsServer::AcceptorTask::start_hook()
{
    tcp::endpoint endpoint{tcp::v4(), port_num_};
    boost::system::error_code err;
    acceptor_.open(endpoint.protocol(), err);
    acceptor_.set_option(tcp::socket::reuse_address{true});
    if (err)
    {
        TM_LOG(error) << "Unable to accept WebSocket connections (err: "
            << err << ")";
        return;
    }
    acceptor_.bind(endpoint, err);
    if (err)
    {
        TM_LOG(error) << "Unable to bind WebSocket port (err: " << err << ")";
        return;
    }
    acceptor_.listen(boost::asio::socket_base::max_listen_connections, err);
    if (err)
    {
        TM_LOG(error) << "Unable to listen on WebSocket port (err: "
            << err << ")";
        return;
    }
    TM_LOG(info) << "Opened WebSocket server accepting on port " << port_num_;
}

void WsServer::AcceptorTask::stop_hook()
{
    acceptor_.close();
    io_context_.stop();
}

void WsServer::AcceptorTask::run()
{
    while (!should_quit())
    {
        bool accept_ok = accept();
        if (!accept_ok && !should_quit())
        {
            TM_LOG(error) << "Unable to accept socket cxn for WebSocket server";
            boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
        }
    }
}

bool WsServer::AcceptorTask::accept()
{
    using boost::asio::ip::tcp;
    if (!acceptor_.is_open())
        return false;
    tcp::endpoint peer_endpoint;
    tcp::socket new_cxn_socket{io_context_};
    std::atomic<bool> handled_new_cxn{false};
    auto new_cxn_handler = [&peer_endpoint, &handled_new_cxn,
            &new_cxn_socket, this](const boost::system::error_code& err) {
            this->handle_new_cxn(peer_endpoint, std::move(new_cxn_socket), err);
            handled_new_cxn = true;
        };
    acceptor_.async_accept(new_cxn_socket, peer_endpoint, new_cxn_handler);
    while (!handled_new_cxn && !should_quit())
    {
        // TODO: Improve this wait loop
        io_context_.run_for(std::chrono::milliseconds(200));
    }
    return handled_new_cxn;
}

void WsServer::AcceptorTask::handle_new_cxn(
    const boost::asio::ip::tcp::endpoint& peer_endpoint,
    boost::asio::ip::tcp::socket&& new_cxn_socket,
    const boost::system::error_code& err) const
{
    namespace websocket = boost::beast::websocket;
    if (err)
    {
        TM_LOG(error) << "Failed to accept WebSocket server socket "
            << "connection from " << peer_endpoint << " to port "
            << port_num_ << " (err " << err << ")";
        return;
    }
    TM_LOG(info) << "Accepted WebSocket server socket connection "
        << "from " << peer_endpoint << " to port " << port_num_;

    // Create a WebSocket stream and perform the WebSocket handshake
    try
    {
        websocket::stream<tcp::socket> new_ws_stream{std::move(new_cxn_socket)};
        new_ws_stream.accept();
        server_.add_listener(std::move(new_ws_stream));
    }
    catch(std::runtime_error& e)
    {
        TM_LOG(error) << "Failed to negotiate WebSocket server socket "
            << "connection from " << peer_endpoint << " to port "
            << port_num_ << " (err " << e.what() << ")";
        return;
    }
}

WsServer::PublisherTask::PublisherTask(string_view name,
        const ProgState& prog_state, WsServer& server)
    : Task{name, prog_state}, server_{server}
{
}

WsServer::PublisherTask::~PublisherTask()
{
}

void WsServer::PublisherTask::run()
{
    // FIXME: Adding a temporary pause here to allow clients to receive the
    //  first messages published, since the acceptor task will have started
    //  first.  Consider alternatives.
    boost::this_thread::sleep_for(boost::chrono::seconds(3));

    while (!should_quit())
    {
        if (wait_for_msg())
            publish_all();
    }
}

void WsServer::PublisherTask::stop_hook()
{
    server_.msg_listener_.cancel();
}

bool WsServer::PublisherTask::wait_for_msg() const
{
    return server_.msg_listener_.wait([this]{ return this->should_quit(); });
}

void WsServer::PublisherTask::publish_all() const
{
    auto visitor = [this](const auto& typed_msg){
        this->server_.publish_single_msg(typed_msg); };
    while (optional<MsgType> msg = server_.msg_listener_.pop_front())
        visit(visitor, *msg);
}

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


