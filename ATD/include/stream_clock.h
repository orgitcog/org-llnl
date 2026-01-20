#ifndef STREAM_CLOCK_H_
#define STREAM_CLOCK_H_

#include <fstream>
#include <istream>
#include <vector>
#include <boost/asio/ip/tcp.hpp>
#include "clock.h"
#include "common.h"
#include "time_msg.h"

// Stream Clock
//  This unit specifies a base class for clocks that read the time message
//  exchange protocol (which includes an archive header that specifies the
//  registry of clock identifiers and model parameters, followed by time
//  measurement messages) from a generic input stream.  Derived classes further
//  specifying that stream as either a file-based stream or a network
//  socket-based stream are also provided here.  Finally, note that a
//  translation table is maintained that maps the clock IDs reported over the
//  stream to those registered in the Detector module within the application.
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

class Detector;

class StreamClock : public Clock
{
  public:
    StreamClock(Detector& det, ClockDesc names, std::istream& is);
    ~StreamClock() override;
    StreamClock(const StreamClock&) = delete;
    StreamClock& operator=(const StreamClock&) = delete;

    void close();
    void propagate_done_flag() const;
    bool stream_done() const;

    TimeMsgHeader read_header();
    TimeMsg read_message() const;
    void read_and_send_message();

  protected:
    virtual void close_hook();

  private:
    Detector& detector_;
    std::istream& stream_;
    // Map of stream-field clock identifiers to ClockId values as registered
    //  dynamically within the current process
    std::map<int, ClockId> clock_id_xlator_;
};

class FileClock : public StreamClock
{
  public:
    FileClock(Detector& det, ClockDesc names, const std::string& filename);
    ~FileClock() override;
    FileClock(const FileClock&) = delete;
    FileClock& operator=(const FileClock&) = delete;

  protected:
    void close_hook() override;

  private:
    std::ifstream file_;
};

class SocketClock : public StreamClock
{
  public:
    SocketClock(Detector& det, ClockDesc names, string_view host,
        string_view service);
    ~SocketClock() override;
    SocketClock(const SocketClock&) = delete;
    SocketClock& operator=(const SocketClock&) = delete;

  protected:
    void close_hook() override;

  private:
    boost::asio::ip::tcp::iostream socket_;
};

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

#endif // STREAM_CLOCK_H_
