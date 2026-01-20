#ifndef PARSE_UBX_H_
#define PARSE_UBX_H_
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "common.h"

// parse_ubx
//  This unit parses the UBX M8 binary protocol (see "u-blox 8 / u-blox M8
//  Receiver Description Including Protocol Specification," u-blox Document
//  UBX-13003221 - R13, July 2017 for more information about this protocol)
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

namespace Ubx
{
    using ClassIdType = std::uint8_t;
    using MsgIdType = std::uint8_t;

    using ChecksumPair = std::pair<std::uint8_t, std::uint8_t>;

    // Enumerations
    enum class Field 
    { 
        U1, RU1_3, I1, X1, U2, I2, X2, U4, I4, X4, R4, R8, CH
    };

    enum class ClassId : ClassIdType
    {
        NAV = 0x01, RXM = 0x02, INF = 0x04, ACK = 0x05, CFG = 0x06, UPD = 0x09,
            MON = 0x0A, AID = 0x0B, TIM = 0x0D, ESF = 0X10, MGA = 0x13, 
            LOG = 0x21, SEC = 0x27, HNR = 0x28,
        // Undocumented debug output classes added here:
        UNK1 = 0x03, UNK2 = 0x0C
    };

    struct Hdr
    {
        // Header length (after sync removal)
        static constexpr size_t hdr_len = 4; // Class(1B) + MsgID(1B) + Len(2B)
        // Header fields
        Ubx::ClassIdType class_id;
        Ubx::MsgIdType msg_id;
        std::uint16_t len;
    };

    // Utility functions
    constexpr std::uint16_t checksum(ustring_view s);

    // Exceptions
    class ParseError : public std::runtime_error
    {
      public:
        ParseError(const std::string& err) : std::runtime_error(err)
        {
        }
        virtual ~ParseError() = default;
    };

    class RecoverableParseError : public std::runtime_error
    {
      public:
        RecoverableParseError(const std::string& err) : std::runtime_error(err)
        {
        }
        virtual ~RecoverableParseError() = default;
    };
}

class UbxPayload;
class UbxPktReader;

struct UbxPkt
{
    static const unsigned char sync_char1 = '\xB5';
    static const unsigned char sync_char2 = '\x62';

    static UbxPkt parse(ustring_view s);
    static std::pair<Ubx::Hdr, UbxPktReader> parse_header(ustring_view s);
    static UbxPkt finish_parse(UbxPktReader& reader, Ubx::Hdr header);

    std::string describe() const;
    void parse_payload(UbxPktReader& reader, Ubx::ClassId cls_id, 
        Ubx::MsgIdType msg_id);

    template <typename T>
    const T* get_payload_checked() const
    {
        if ((static_cast<Ubx::ClassId>(header.class_id) != T::class_id) || 
            (header.msg_id != T::msg_id))
        {
            return nullptr;
        }
        return static_cast<const T*>(payload.get());
    }

    template <typename Fn>
    void apply(Fn fn) const;

    Ubx::Hdr header;
    std::unique_ptr<UbxPayload> payload;
};

// Template implementation file defines field types and sizes as well as a 
//  field wrapper (UbxPktMember) that captures this information
#include "../src/parse_ubx.txx"

class UbxPktReader
{
  public:
    explicit UbxPktReader(ustring_view s);

    template <Ubx::Field F>
    ubx_field_type<F> read();

    template <Ubx::Field F, ubx_field_type<F> range_min,
        ubx_field_type<F> range_max, template <class T, T, T> class Policy>
    void read_into(UbxPktMember<F, range_min, range_max, Policy>& pkt_member);

    ustring_view::const_reference operator[](size_t n);
    std::uint16_t checksum() const;
    bool done() const;
    size_t remaining() const;
    void set_view(ustring_view new_view);
    void skip_payload();

  private:
    ustring_view pkt_view_;
    Ubx::ChecksumPair checksum_;
};

struct UbxPayload
{
    static std::unique_ptr<UbxPayload> parse(UbxPktReader& reader);
    std::string describe() const;
};
struct UbxPayloadAckAck : public UbxPayload
{
    static constexpr Ubx::ClassId class_id = Ubx::ClassId::ACK;
    static constexpr Ubx::MsgIdType msg_id = 0x01; 
    static std::unique_ptr<UbxPayload> parse(UbxPktReader& reader);
    std::string describe() const;

    UbxPktMember<Ubx::Field::U1> ack_class_id;
    UbxPktMember<Ubx::Field::U1> ack_msg_id;
};
struct UbxPayloadAckNak : public UbxPayload
{
    static constexpr Ubx::ClassId class_id = Ubx::ClassId::ACK;
    static constexpr Ubx::MsgIdType msg_id = 0x00; 
    static std::unique_ptr<UbxPayload> parse(UbxPktReader& reader);
    std::string describe() const;

    UbxPktMember<Ubx::Field::U1> nak_class_id;
    UbxPktMember<Ubx::Field::U1> nak_msg_id;
};
struct UbxPayloadMonHw : public UbxPayload
{
    static constexpr Ubx::ClassId class_id = Ubx::ClassId::MON;
    static constexpr Ubx::MsgIdType msg_id = 0x09; 
    static std::unique_ptr<UbxPayload> parse(UbxPktReader& reader);
    std::string describe() const;

    enum class JamState { unknown = 0, ok = 1, warning = 2, critical = 3 };
    JamState get_jam_state() const
        { return static_cast<JamState>((flags >> 2) & 0x3); }

    UbxPktMember<Ubx::Field::X4> pin_sel;
    UbxPktMember<Ubx::Field::X4> pin_bank;
    UbxPktMember<Ubx::Field::X4> pin_dir;
    UbxPktMember<Ubx::Field::X4> pin_val;
    UbxPktMember<Ubx::Field::U2> noise_level;
    UbxPktMember<Ubx::Field::U2, 0, 8191> agc_count;
    UbxPktMember<Ubx::Field::U1, 0, 4> antenna_status;
    UbxPktMember<Ubx::Field::U1, 0, 2> antenna_power;
    UbxPktMember<Ubx::Field::X1> flags;
    UbxPktMember<Ubx::Field::U1> rsvd;
    UbxPktMember<Ubx::Field::X4> pins_used;
    UbxPktMember<Ubx::Field::U4> vp1;
    UbxPktMember<Ubx::Field::U4> vp2;
    UbxPktMember<Ubx::Field::U4> vp3;
    UbxPktMember<Ubx::Field::U4> vp4;
    UbxPktMember<Ubx::Field::U1> vp5;
    UbxPktMember<Ubx::Field::U1> cw_jam_ind;
    UbxPktMember<Ubx::Field::U2> rsvd2;
    UbxPktMember<Ubx::Field::X4> pin_irq;
    UbxPktMember<Ubx::Field::X4> pin_pull_hi;
    UbxPktMember<Ubx::Field::X4> pin_pull_lo;
};
struct UbxPayloadNavSbas : public UbxPayload
{
    static constexpr Ubx::ClassId class_id = Ubx::ClassId::NAV;
    static constexpr Ubx::MsgIdType msg_id = 0x32; 
    static std::unique_ptr<UbxPayload> parse(UbxPktReader& reader);
    std::string describe() const;

    UbxPktMember<Ubx::Field::U4> itow;
    UbxPktMember<Ubx::Field::U1> geo;
    UbxPktMember<Ubx::Field::U1, 0, 3> mode;
    UbxPktMember<Ubx::Field::I1, -1, 16> sys;
    UbxPktMember<Ubx::Field::X1> service;
    UbxPktMember<Ubx::Field::U1> cnt;
    UbxPktMember<Ubx::Field::U1> rsvd1a;
    UbxPktMember<Ubx::Field::U1> rsvd1b;
    UbxPktMember<Ubx::Field::U1> rsvd1c;
    struct SbasSv
    {
        UbxPktMember<Ubx::Field::U1> svid;
        UbxPktMember<Ubx::Field::U1> flags;
        UbxPktMember<Ubx::Field::U1> udre;
        UbxPktMember<Ubx::Field::I1, -1, 16> sv_sys; // U1 in protocol spec
        UbxPktMember<Ubx::Field::U1> sv_service;
        UbxPktMember<Ubx::Field::U1> rsvd2;
        UbxPktMember<Ubx::Field::I2> prc;
        UbxPktMember<Ubx::Field::U2> rsvd3;
        UbxPktMember<Ubx::Field::I2> ic;
    };
    std::vector<SbasSv> sv_vec;
};
struct UbxPayloadNavSol : public UbxPayload
{
    static constexpr Ubx::ClassId class_id = Ubx::ClassId::NAV;
    static constexpr Ubx::MsgIdType msg_id = 0x06; 
    static std::unique_ptr<UbxPayload> parse(UbxPktReader& reader);
    std::string describe() const;

    UbxPktMember<Ubx::Field::U4> itow;
    UbxPktMember<Ubx::Field::I4, -500000, 500000> ftow;
    UbxPktMember<Ubx::Field::I2> week;
    UbxPktMember<Ubx::Field::U1, 0, 5> gps_fix;
    UbxPktMember<Ubx::Field::X1> flags;
    UbxPktMember<Ubx::Field::I4> ecef_x;
    UbxPktMember<Ubx::Field::I4> ecef_y;
    UbxPktMember<Ubx::Field::I4> ecef_z;
    UbxPktMember<Ubx::Field::U4> p_acc;
    UbxPktMember<Ubx::Field::I4> ecef_x_vel;
    UbxPktMember<Ubx::Field::I4> ecef_y_vel;
    UbxPktMember<Ubx::Field::I4> ecef_z_vel;
    UbxPktMember<Ubx::Field::U4> s_acc;
    UbxPktMember<Ubx::Field::U2> p_dop;
    UbxPktMember<Ubx::Field::U1> rsvd1;
    UbxPktMember<Ubx::Field::U1> num_sv;
    UbxPktMember<Ubx::Field::U4> rsvd2;
};
struct UbxPayloadNavStatus : public UbxPayload
{
    static constexpr Ubx::ClassId class_id = Ubx::ClassId::NAV;
    static constexpr Ubx::MsgIdType msg_id = 0x03; 
    static std::unique_ptr<UbxPayload> parse(UbxPktReader& reader);
    std::string describe() const;

    // Check if position and velocity are valid, within DOP & ACC masks
    bool is_fix_ok() const      { return flags & 0x1; }
    bool is_week_set() const    { return flags & 0x4; }
    bool is_tow_set() const     { return flags & 0x8; }
    enum class SpoofDetState
        { unknown = 0, not_indicated = 1, indicated = 2, multi_indicated = 3 };
    SpoofDetState get_spoof_det_state() const
        { return static_cast<SpoofDetState>((flags2 >> 3) & 0x3); }

    UbxPktMember<Ubx::Field::U4> itow;
    UbxPktMember<Ubx::Field::U1, 0, 5> gps_fix;
    UbxPktMember<Ubx::Field::X1> flags;
    UbxPktMember<Ubx::Field::X1> fix_status;
    UbxPktMember<Ubx::Field::X1> flags2;
    UbxPktMember<Ubx::Field::U4> ttff;
    UbxPktMember<Ubx::Field::U4> ms_since_start;
};
struct UbxPayloadNavTimeGps : public UbxPayload
{
    static constexpr Ubx::ClassId class_id = Ubx::ClassId::NAV;
    static constexpr Ubx::MsgIdType msg_id = 0x20; 
    static std::unique_ptr<UbxPayload> parse(UbxPktReader& reader);
    std::string describe() const;

    UbxPktMember<Ubx::Field::U4> itow;
    UbxPktMember<Ubx::Field::I4, -500000, 500000> ftow;
    UbxPktMember<Ubx::Field::I2> week;
    UbxPktMember<Ubx::Field::I1> leap_s;
    UbxPktMember<Ubx::Field::X1> valid;
    UbxPktMember<Ubx::Field::U4> t_acc;
};
struct UbxPayloadTimSmeas : public UbxPayload
{
    static constexpr Ubx::ClassId class_id = Ubx::ClassId::TIM;
    static constexpr Ubx::MsgIdType msg_id = 0x13; 
    static std::unique_ptr<UbxPayload> parse(UbxPktReader& reader);
    std::string describe() const;

    UbxPayloadTimSmeas() = default;
    UbxPktMember<Ubx::Field::U1, 0, 0> version;
    UbxPktMember<Ubx::Field::U1, 0, 6> num_meas;
    UbxPktMember<Ubx::Field::U4> itow;
    struct Smeas
    {
        bool is_freq_valid() const  { return flags & 0x1; }
        bool is_phase_valid() const { return flags & 0x2; }

        UbxPktMember<Ubx::Field::U1, 0, 5> src_id;
        UbxPktMember<Ubx::Field::X1> flags;
        // TODO: Handle scaled fields
        UbxPktMember<Ubx::Field::I1> phase_offset_frac;
        UbxPktMember<Ubx::Field::U1> phase_unc_frac;
        UbxPktMember<Ubx::Field::I4> phase_offset;
        UbxPktMember<Ubx::Field::U4> phase_unc;
        UbxPktMember<Ubx::Field::U4> rsvd3;
        UbxPktMember<Ubx::Field::I4> freq_offset;
        UbxPktMember<Ubx::Field::U4> freq_unc;
    };
    std::array<Smeas, 6> meas_array;
};

using UbxPayloadTypes = std::tuple<
    UbxPayloadAckAck,
    UbxPayloadAckNak,
    UbxPayloadMonHw,
    UbxPayloadNavSbas,
    UbxPayloadNavSol,
    UbxPayloadNavStatus,
    UbxPayloadNavTimeGps,
    UbxPayloadTimSmeas>;

template <typename Fn>
void UbxPkt::apply(Fn fn) const
{
    // List all packet payload types that should be used for apply here:
    using ApplyPayloadTypes = UbxPayloadTypes;

    assert(Ubx::is_valid_class(header.class_id));
    auto cls_id = static_cast<Ubx::ClassId>(header.class_id);

    Ubx::for_matching_payload<ApplyPayloadTypes>(cls_id, header.msg_id,
        Ubx::PayloadApplyFunctor<Fn>{*this, fn});
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

#endif // PARSE_UBX_H_
