#include "stream_clock.h"
#include "detector.h"

// Stream_Clock
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

StreamClock::StreamClock(Detector& det, ClockDesc names, std::istream& is)
    : Clock{det, names}, detector_{det}, stream_{is}, clock_id_xlator_{}
{
}

StreamClock::~StreamClock()
{
    close();
}

void StreamClock::close()
{
    close_hook();
}

void StreamClock::close_hook()
{
}

void StreamClock::propagate_done_flag() const
{
    // TODO: Consider if this should be a virtual hook in Clock::set_done_flag
    for (const auto& x : clock_id_xlator_)
        detector_.handle_clock_done(x.second);
}

bool StreamClock::stream_done() const
{
    return stream_.eof();
}

TimeMsgHeader StreamClock::read_header()
{
    TimeMsgHeader ret = TimeMsgHeader::parse(stream_);

    // Check that protocol version is compatible
    if (ret.protocol_version.find("1.") == std::string::npos)
        throw ParseError("Unsupported protocol version");

    // Register clock IDs from provided registry and enroll in the ID
    //  translator as well
    int unique_clocks{0};
    for (const auto& x : ret.clock_registry)
    {
        if (clock_id_xlator_.find(x.first) == clock_id_xlator_.end())
        {
            ++unique_clocks;
            ClockDesc new_desc{x.second};
            new_desc.task_name = desc().task_name;
            // Append a [#] suffix (with the number being the index among the
            //  unique clocks in the registry) if already registered
            if (detector_.is_clock_registered(new_desc))
            {
                new_desc.clock_name += "[" + std::to_string(unique_clocks) +
                    "]";
                assert(!detector_.is_clock_registered(new_desc));
            }
            ClockId new_id = detector_.register_clock(new_desc);
            clock_id_xlator_[x.first] = new_id;
            auto get_logger = [this]() -> auto& {
                return detector_.get_logger(); };
            TM_LOG(debug) << "Mapped registry ID " << x.first << "->" << new_id;
        }
    }
    return ret;
}

TimeMsg StreamClock::read_message() const
{
    TimeMsg ret = parse_time_msg(stream_);

    // Translate clock IDs, then return
    ret.clock_id = clock_id_xlator_.at(ret.clock_id);
    for (auto& x : ret.comparisons)
        x.other_clock_id = clock_id_xlator_.at(x.other_clock_id);
    return ret;
}

void StreamClock::read_and_send_message()
{
    TimeMsg new_msg = read_message();
    send_message(new_msg);
}

FileClock::FileClock(Detector& det, ClockDesc names,
        const std::string& filename)
    : StreamClock{det, names, file_}, file_{filename.c_str()}
{
    if (!file_)
        throw std::runtime_error("Unable to open file " + filename);
}

FileClock::~FileClock()
{
}

void FileClock::close_hook()
{
    file_.close();
}

SocketClock::SocketClock(Detector& det, ClockDesc names, string_view host,
    string_view service)
        : StreamClock{det, names, socket_}
{
    socket_.connect(host, service);
    if (!socket_)
        throw std::runtime_error(socket_.error().message());
}

SocketClock::~SocketClock()
{
}

void SocketClock::close_hook()
{
    socket_.close();
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

