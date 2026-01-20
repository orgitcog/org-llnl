#include "ubx_file_reader.h"
#include "parse_ubx.h"

// UBX_File_Reader
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
class AdaptIstreamToSyncReadStream
{
  public:
    AdaptIstreamToSyncReadStream(std::istream& is) : is_{is}
    {
    }
    template <typename MutableBufferSequence>
    std::size_t read_some(MutableBufferSequence&& mb)
    {
        auto buf_size = boost::asio::buffer_size(mb);
        if (buf_size <= 0)
        {
            return 0;
        }
        auto num_read = is_.readsome(mb.size(), mb.max_size());
        if (num_read < 0)
            return 0;
        return num_read;
    }
    template <typename MutableBufferSequence, typename ErrorCode>
    std::size_t read_some(MutableBufferSequence&& mb, ErrorCode& ec)
    {
        auto buf_size = boost::asio::buffer_size(mb);
        if (buf_size <= 0)
        {
            ec = boost::asio::error::no_buffer_space;
            return 0;
        }
        auto num_read = is_.readsome(static_cast<char*>(mb.data()), buf_size);
        // Since readsome may not set eofbit on some implementations, the call
        //  to peek below is intended to do so when necessary:
        is_.peek();
        if (num_read <= 0)
        {
            if (is_.eof())
                ec = boost::asio::error::eof;
            else if (num_read < 0)
                ec = boost::asio::error::not_found;
            return 0;
        }
        return num_read;
    }
  private:
    std::istream& is_;
};
} // end namespace

UbxFileReader::UbxFileReader(string_view name, const ProgState& prog_state,
    Detector& det, string_view filename, std::size_t gnss_idx)
    : UbxBase{name, prog_state, det, gnss_idx},
        reader_{std::string{filename}, std::ios_base::binary}
{
}

void UbxFileReader::read_and_handle() /* throws system_error */
try
{
    BOOST_LOG_FUNCTION();
    AdaptIstreamToSyncReadStream adapted_reader{reader_};
    bool sync_ok = read_until_sync(adapted_reader);
    if (should_quit())
        return;
    if (!sync_ok)
    {
        set_done_flag(); // EOF
        return;
    }
    UbxPkt new_pkt = read_pkt(adapted_reader);
    if (should_quit())
        return;
    handle(new_pkt);
    TM_LOG(trace) << "UbxFile new pkt: " << new_pkt.describe();
}
catch (const boost::system::system_error& err)
{
    // EOF errors can be silently ignored; others are rethrown
    if (err.code() != boost::asio::error::eof)
        throw;
    else
        set_done_flag();
}

void UbxFileReader::run()
{
    while (!should_quit())
    {
        read_and_handle();
    }
    // Set flag to indicate clock will not send any further messages
    set_done_flag();
}

void UbxFileReader::stop_hook()
{
    reader_.close();
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

