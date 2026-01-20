#include <boost/asio/read.hpp>
#include <boost/asio/read_until.hpp>
#include "parse_ubx.h"
#include "utility.h"

// UBX_Base
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

template <class SyncReadStream>
ustring_view UbxBase::read_bytes(SyncReadStream& s, std::size_t num_bytes)
    /* throws system_error */
{
    // Read more data if needed
    boost::system::error_code err;
    if (buffer_.size() < num_bytes)
    {
        std::size_t num_read = boost::asio::read(s, 
            boost::asio::dynamic_buffer(buffer_), 
            interruptible_xfr_at_least{[this]{ return this->should_quit(); },
                num_bytes},
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

template <class SyncReadStream>
UbxPkt UbxBase::read_pkt(SyncReadStream& s)
    /* throws boost::system::system_error, runtime_error */
{
    ustring_view hdr_view = read_bytes(s, Ubx::Hdr::hdr_len);
    std::pair<Ubx::Hdr, UbxPktReader> header_and_reader = 
        UbxPkt::parse_header(hdr_view);
    auto& reader = std::get<UbxPktReader>(header_and_reader);
    auto& header = std::get<Ubx::Hdr>(header_and_reader);
    boost::asio::dynamic_buffer(buffer_).consume(hdr_view.size());
    // Read payload + 2-byte checksum
    ustring_view pkt_remainder_view = read_bytes(s, header.len + 2);
    if (pkt_remainder_view.empty())
    {
        if (!should_quit())
            throw std::runtime_error{"Empty read for payload"};
        return {};
    }
    reader.set_view(pkt_remainder_view);

    UbxPkt pkt;
    try
    {
        pkt = UbxPkt::finish_parse(reader, header);
        TM_LOG(trace) << "Parsed pkt of length " << header.len <<
            " (Class: x" << std::hex << static_cast<int>(header.class_id)
            << ", Msg: x" << static_cast<int>(header.msg_id) << ")" << std::dec;
    }
    catch(const Ubx::RecoverableParseError& e)
    {
        TM_LOG(warning) << "Ignored parse error for pkt of length "
            << header.len <<
            " (Class: x" << std::hex << static_cast<int>(header.class_id)
            << ", Msg: x" << static_cast<int>(header.msg_id) << ")" << std::dec
            << "; parse error follows: " << e.what();
        // Ensure packet header is in place; payload will be empty
        pkt.header = header;
    }
    boost::asio::dynamic_buffer(buffer_).consume(pkt_remainder_view.size());
    return pkt;
}

// Reads until the two-byte Ubx sync code is matched, and discards all bytes
//  up to and including the sync
// Returns: True if read succeeded, false if EOF or stream closed
// Throws: On other read errors
template <class SyncReadStream>
bool UbxBase::read_until_sync(SyncReadStream& s)
    /* throws boost::system::system_error */
{
    std::string sync_str = {static_cast<char>(UbxPkt::sync_char1),
        static_cast<char>(UbxPkt::sync_char2)};
    boost::system::error_code err;
    std::size_t bytes_to_sync_end = boost::asio::read_until(s,
        boost::asio::dynamic_buffer(buffer_), sync_str, err);
    if (should_quit())
        return false; // bail out ignoring any errors if quitting
    if (bytes_to_sync_end == 0)
    {
        if (err == boost::asio::error::eof)
            return false;
        // Read failed for a reason other than end-of-file; throw error
        throw boost::system::system_error(err);
    }
    boost::asio::dynamic_buffer(buffer_).consume(bytes_to_sync_end);
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

