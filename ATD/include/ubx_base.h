#ifndef UBX_BASE_H_
#define UBX_BASE_H_

#include <boost/asio/buffer.hpp>
#include "common.h"
#include "ubx_clock.h"
#include "task.h"

// UBX Base
//  Base class for tasks that interface to time sources using the UBX M8 binary
//  protocol (cf. parse_ubx.h).  This class owns the clock object and a buffer
//  for protocol bytes yet to be parsed.  It can read and buffer bytes from a
//  provided stream (satisfying the SyncReadStream concept from ASIO), look for
//  the synchronization marker in the stream, and parse a packet.  Finally, it
//  can handle a variety of different payload types that may be parsed, such as
//  validating the positioning information in UbxPayloadNavSol or generating a
//  time message from UbxPayloadTimSmeas.
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

// Forward declarations
class UbxPayload;
class UbxPayloadMonHw;
class UbxPayloadNavSol;
class UbxPayloadNavStatus;
class UbxPayloadTimSmeas;
class UbxPkt;

namespace tmon
{
class Detector;

class UbxBase : public Task
{
  public:
    UbxBase(string_view name, const ProgState& prog_state, Detector& det,
        std::size_t gnss_idx);
    ~UbxBase() override = default;
    UbxBase(const UbxBase&) = delete;
    UbxBase& operator=(const UbxBase&) = delete;

    bool sync_clocks(const UbxBase& other, UbxClock::SubclockPair subclocks);

  protected:
    void start_hook() override;
    void run() override = 0;
    void stop_hook() override = 0;

    // Packet handler will dispatch to payload handlers below
    void handle(const UbxPkt& pkt);

    void handle(const UbxPayload& generic);
    void handle(const UbxPayloadMonHw& hw);
    void handle(const UbxPayloadNavSol& navsol);
    void handle(const UbxPayloadNavStatus& status);
    void handle(const UbxPayloadTimSmeas& smeas);

    template <class SyncReadStream>
    UbxPkt read_pkt(SyncReadStream& s);
    template <class SyncReadStream>
    bool read_until_sync(SyncReadStream& s);

    // Raw read function; prefer using higher-level read functions (read_pkt,
    //  read_until_sync) above
    template <class SyncReadStream>
    ustring_view read_bytes(SyncReadStream& s, std::size_t num_bytes);

    void set_done_flag();

  private:
    UbxClock clock_;
    std::vector<unsigned char> buffer_;
    Detector& detector_;

  #if defined(ENABLE_TEST_INJECT)
    void inject_test(UbxPayloadTimSmeas& smeas) const noexcept;
  #endif // ENABLE_TEST_INJECT
};

} // end namespace tmon

#include "../src/ubx_base.txx"

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

#endif // UBX_BASE_H_
