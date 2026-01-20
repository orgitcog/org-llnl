#include "parse_ubx.h"
#include <algorithm>
#include <cstring> // for memcpy
#include <iomanip>
#include <sstream>
#include <tuple>
#include <type_traits>

// parse_ubx
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
std::string describe_class(ClassId id)
{
    switch (id)
    {
        case ClassId::NAV:
            return "NAV";
        case ClassId::RXM:
            return "RXM";
        case ClassId::INF:
            return "INF";
        case ClassId::ACK:
            return "ACK";
        case ClassId::CFG:
            return "CFG";
        case ClassId::UPD:
            return "UPD";
        case ClassId::MON:
            return "MON";
        case ClassId::AID:
            return "AID";
        case ClassId::TIM:
            return "TIM";
        case ClassId::ESF:
            return "ESF";
        case ClassId::MGA:
            return "MGA";
        case ClassId::LOG:
            return "LOG";
        case ClassId::SEC:
            return "SEC";
        case ClassId::HNR:
            return "HNR";
        default:
            return "<UNKNOWN>";
    }
}

class PayloadParseFunctor 
{
  public:
    explicit PayloadParseFunctor(UbxPktReader& reader) 
        : reader_{reader}, result_{}
    {
    }
    PayloadParseFunctor(PayloadParseFunctor&&) = default;
    
    template <typename T>
    void operator()()
    {
        result_ = T::parse(reader_);
    }

    std::unique_ptr<UbxPayload> get_payload()
    {
        return std::move(result_);
    }

  private:
    UbxPktReader& reader_;
    std::unique_ptr<UbxPayload> result_;
};

class PayloadDescribeFunctor
{
  public:
    PayloadDescribeFunctor(const UbxPkt& pkt) : pkt_{pkt}, result_{"<UNKNOWN>"}
    {
    }

    template <typename T>
    void operator()()
    {
        const T* payload = pkt_.get_payload_checked<T>();
        assert(payload);
        result_ = payload->describe();
    }

    std::string get() const
    {
        return result_;
    }

  private:
    const UbxPkt& pkt_;
    std::string result_;
};

// Especially for logging purposes, provide a stream insertion operator for
//  UbxPktMember that doesn't show single bytes as an ASCII character, but
//  instead as the underlying value
template <Ubx::Field F, ubx_field_type<F> range_min,
    ubx_field_type<F> range_max, template <class T, T, T> class Policy,
    typename = std::enable_if_t<ubx_field_size<F> == 1>>
std::ostream& operator<<(std::ostream& os,
    const UbxPktMember<F, range_min, range_max, Policy>& pkt_member)
{
    // Note: to_string both performs the desired type promotion here and
    //  prevents infinite recursion
    return os << std::to_string(pkt_member);
}
} // end namespace Ubx


template<Ubx::Field T>
constexpr ubx_field_type<T> parse_ubx_impl(ustring_view s) 
{ 
    static_assert(T != Ubx::Field::RU1_3, 
        "UBX RU1_3 field handling not yet impl");
    static_assert(std::is_trivially_copyable<ubx_field_type<T>>::value,
        "UBX field types must be trivially copyable");
    // Perform type punning via memcpy to avoid invoking any undefined behavior
    //  (the memcpy is likely to be optimized out)
    ubx_field_type<T> ret;
    std::memcpy(&ret, s.data(), sizeof(ret));
    return ret;
}

template <Ubx::Field T>
constexpr ubx_field_type<T> parse_ubx_field(ustring_view s)
{
    assert((s.size() == ubx_field_size<T>) &&
        "Size mismatch for ubx parse");
    static_assert(sizeof(ubx_field_type<T>) == ubx_field_size<T>,
        "Type size does not match specified field length");
    return parse_ubx_impl<T>(s);
}

UbxPktReader::UbxPktReader(ustring_view s) : pkt_view_{s}, checksum_{}
{
}

ustring_view::const_reference UbxPktReader::operator[](size_t n)
{
    return pkt_view_[n];
}

std::uint16_t UbxPktReader::checksum() const
{
    // Return rolling checksum in little-endian format
    return (checksum_.second << 8) + checksum_.first;
}

bool UbxPktReader::done() const
{
    return pkt_view_.empty();
}

size_t UbxPktReader::remaining() const
{
    return pkt_view_.size();
}

void UbxPktReader::set_view(ustring_view new_view)
{
    pkt_view_ = std::move(new_view);
}

void UbxPktReader::skip_payload()
{
    // Discard the rest of the buffer, except checksum
    // First, compute the checksum of the payload
    Ubx::update_checksum(pkt_view_.substr(0, pkt_view_.length() - 2),
        checksum_);
    pkt_view_.remove_prefix(pkt_view_.length() - 2);
}

// Read the specified field from the front of the buffer and then remove it
template <Ubx::Field F>
ubx_field_type<F> UbxPktReader::read()
{
    auto field_len = ubx_field_size<F>;
    if (pkt_view_.length() < field_len)
        throw Ubx::ParseError("Not enough remaining data for field");
    auto prefix_view = pkt_view_.substr(0, field_len);
    auto ret = parse_ubx_field<F>(prefix_view);
    Ubx::update_checksum(prefix_view, checksum_);
    pkt_view_.remove_prefix(field_len);
    return ret;
}

template <Ubx::Field F, ubx_field_type<F> range_min, 
    ubx_field_type<F> range_max, template <class T, T, T> class Policy>
void UbxPktReader::read_into(UbxPktMember<F, range_min, range_max, Policy>& 
    pkt_member)
{
    pkt_member = read<UbxPktMember<F>::field>();
}


std::string UbxPayload::describe() const
{
    return {};
}
std::string UbxPayloadAckAck::describe() const
{
    return UbxPayload::describe() + "(ACK)";
}
std::string UbxPayloadAckNak::describe() const
{
    return UbxPayload::describe() + "(NAK)";
}
std::string UbxPayloadMonHw::describe() const
{
    std::ostringstream oss;
    oss << UbxPayload::describe() + "(HW)\n";
    oss << "\tNOISE=" << noise_level << ", AGC=" << agc_count
        << ", FLAGS=" << flags << ", JAMIND=" << cw_jam_ind;
    return oss.str();
}
std::string UbxPayloadNavSbas::describe() const
{
    std::ostringstream oss;
    oss << UbxPayload::describe() + "(SBAS)\n";
    oss << "\tITOW=" << itow << ", GEO=" << geo << ", SYS=" << sys;
    return oss.str();
}
std::string UbxPayloadNavSol::describe() const
{
    std::ostringstream oss;
    oss << UbxPayload::describe() + "(SOL)\n";
    oss << "\tITOW=" << itow << "ms, FIX=" << gps_fix << ", X=" << ecef_x
        << ", Y=" << ecef_y << ", Z=" << ecef_z << ", #SV=" << num_sv;
    return oss.str();
}
std::string UbxPayloadNavStatus::describe() const
{
    std::ostringstream oss;
    oss << UbxPayload::describe() + "(STATUS)\n";
    oss << "\tITOW=" << itow << "ms, FIX=" << gps_fix << ", TTFF=" << ttff
        << ", FLAGS=" << flags << ", FLAGS2=" << flags2;
    return oss.str();
}
std::string UbxPayloadNavTimeGps::describe() const
{
    std::ostringstream oss;
    oss << UbxPayload::describe() + "(TIMEGPS)\n";
    oss << "\tITOW=" << itow << "ms, FTOW=" << ftow << "ns, VALID: " << valid
        << ", TACC=" << t_acc << "ns";
    return oss.str();
}
std::string UbxPayloadTimSmeas::describe() const
{
    std::ostringstream oss;
    oss << UbxPayload::describe() + "(SMEAS)\n";
    oss << "\tITOW=" << itow << "ms, #MEAS=" << num_meas;
    return oss.str();
}

std::unique_ptr<UbxPayload> UbxPayloadAckAck::parse(UbxPktReader& reader)
{
    auto ret = std::make_unique<UbxPayloadAckAck>();
    reader.read_into(ret->ack_class_id);
    reader.read_into(ret->ack_msg_id);
    return ret;
}
std::unique_ptr<UbxPayload> UbxPayloadAckNak::parse(UbxPktReader& reader)
{
    auto ret = std::make_unique<UbxPayloadAckNak>();
    reader.read_into(ret->nak_class_id);
    reader.read_into(ret->nak_msg_id);
    return ret;
}
std::unique_ptr<UbxPayload> UbxPayloadMonHw::parse(UbxPktReader& reader)
{
    auto ret = std::make_unique<UbxPayloadMonHw>();
    reader.read_into(ret->pin_sel);
    reader.read_into(ret->pin_bank);
    reader.read_into(ret->pin_dir);
    reader.read_into(ret->pin_val);
    reader.read_into(ret->noise_level);
    reader.read_into(ret->agc_count);
    reader.read_into(ret->antenna_status);
    reader.read_into(ret->antenna_power);
    reader.read_into(ret->flags);
    reader.read_into(ret->rsvd);
    reader.read_into(ret->pins_used);
    reader.read_into(ret->vp1);
    reader.read_into(ret->vp2);
    reader.read_into(ret->vp3);
    reader.read_into(ret->vp4);
    reader.read_into(ret->vp5);
    reader.read_into(ret->cw_jam_ind);
    reader.read_into(ret->rsvd2);
    reader.read_into(ret->pin_irq);
    reader.read_into(ret->pin_pull_hi);
    reader.read_into(ret->pin_pull_lo);
    return ret;
}
std::unique_ptr<UbxPayload> UbxPayloadNavSbas::parse(UbxPktReader& reader)
{
    auto ret = std::make_unique<UbxPayloadNavSbas>();
    reader.read_into(ret->itow);
    reader.read_into(ret->geo);
    reader.read_into(ret->mode);
    reader.read_into(ret->sys);
    reader.read_into(ret->service);
    reader.read_into(ret->cnt);
    reader.read_into(ret->rsvd1a);
    reader.read_into(ret->rsvd1b);
    reader.read_into(ret->rsvd1c);
    ret->sv_vec.resize(ret->cnt);
    for (size_t i = 0; i < ret->cnt; ++i)
    {
        SbasSv& curr_sv = ret->sv_vec[i];
        reader.read_into(curr_sv.svid);
        reader.read_into(curr_sv.flags);
        reader.read_into(curr_sv.udre);
        reader.read_into(curr_sv.sv_sys);
        reader.read_into(curr_sv.sv_service);
        reader.read_into(curr_sv.rsvd2);
        reader.read_into(curr_sv.prc);
        reader.read_into(curr_sv.rsvd3);
        reader.read_into(curr_sv.ic);
    }
    return ret;
}
std::unique_ptr<UbxPayload> UbxPayloadNavSol::parse(UbxPktReader& reader)
{
    auto ret = std::make_unique<UbxPayloadNavSol>();
    reader.read_into(ret->itow);
    reader.read_into(ret->ftow);
    reader.read_into(ret->week);
    reader.read_into(ret->gps_fix);
    reader.read_into(ret->flags);
    reader.read_into(ret->ecef_x);
    reader.read_into(ret->ecef_y);
    reader.read_into(ret->ecef_z);
    reader.read_into(ret->p_acc);
    reader.read_into(ret->ecef_x_vel);
    reader.read_into(ret->ecef_y_vel);
    reader.read_into(ret->ecef_z_vel);
    reader.read_into(ret->s_acc);
    reader.read_into(ret->p_dop);
    reader.read_into(ret->rsvd1);
    reader.read_into(ret->num_sv);
    reader.read_into(ret->rsvd2);
    return ret;
}
std::unique_ptr<UbxPayload> UbxPayloadNavStatus::parse(UbxPktReader& reader)
{
    auto ret = std::make_unique<UbxPayloadNavStatus>();
    reader.read_into(ret->itow);
    reader.read_into(ret->gps_fix);
    reader.read_into(ret->flags);
    reader.read_into(ret->fix_status);
    reader.read_into(ret->flags2);
    reader.read_into(ret->ttff);
    reader.read_into(ret->ms_since_start);
    return ret;
}
std::unique_ptr<UbxPayload> UbxPayloadNavTimeGps::parse(UbxPktReader& reader)
{
    auto ret = std::make_unique<UbxPayloadNavTimeGps>();
    reader.read_into(ret->itow);
    reader.read_into(ret->ftow);
    reader.read_into(ret->week);
    reader.read_into(ret->leap_s);
    reader.read_into(ret->valid);
    reader.read_into(ret->t_acc);
    return ret;
}
std::unique_ptr<UbxPayload> UbxPayloadTimSmeas::parse(
    UbxPktReader& reader)
{
    auto ret = std::make_unique<UbxPayloadTimSmeas>();
    reader.read_into(ret->version);
    reader.read_into(ret->num_meas);
    // Sanity check: the protocol limits num_meas to 6
    if (ret->num_meas > 6)
        throw Ubx::ParseError("Too many meas in TIM-SMEAS decoding");
    reader.read<Ubx::Field::U2>();
    reader.read_into(ret->itow);
    reader.read<Ubx::Field::U4>();
    for (int i = 0; i < ret->num_meas; ++i)
    {
        auto& curr_meas = ret->meas_array[i];
        reader.read_into(curr_meas.src_id);
        reader.read_into(curr_meas.flags);
        reader.read_into(curr_meas.phase_offset_frac);
        reader.read_into(curr_meas.phase_unc_frac);
        reader.read_into(curr_meas.phase_offset);
        reader.read_into(curr_meas.phase_unc);
        reader.read_into(curr_meas.rsvd3);
        reader.read_into(curr_meas.freq_offset);
        reader.read_into(curr_meas.freq_unc);
    }
    return ret;
}

std::string UbxPkt::describe() const
{
    // List all packet payload types that should be described here:
    using DescribePayloadTypes = UbxPayloadTypes;

    assert(Ubx::is_valid_class(header.class_id));
    auto cls_id = static_cast<Ubx::ClassId>(header.class_id);
    std::ostringstream header_oss;
    header_oss << Ubx::describe_class(cls_id) << "-0x" << std::hex
        << static_cast<int>(header.msg_id);

    Ubx::PayloadDescribeFunctor desc_functor =
        Ubx::for_matching_payload<DescribePayloadTypes>(
            static_cast<Ubx::ClassId>(header.class_id), header.msg_id,
            Ubx::PayloadDescribeFunctor{*this});
    std::string payload_desc = desc_functor.get();

    return header_oss.str() + payload_desc;
}

void UbxPkt::parse_payload(UbxPktReader& reader, Ubx::ClassId cls_id, 
    Ubx::MsgIdType msg_id)
{
    // List all packet payload types that should be parsed here:
    using ParsePayloadTypes = UbxPayloadTypes;

    Ubx::PayloadParseFunctor parse_functor =
        Ubx::for_matching_payload<ParsePayloadTypes>(cls_id, msg_id, 
        Ubx::PayloadParseFunctor{reader});
    payload = parse_functor.get_payload();
    if (!payload)
    {
        // Default behavior if no match: skip unhandled message types
        reader.skip_payload();
    }
}

// Arg: a view of the packet after sync char prefix has been stripped
// Throws: Ubx::ParseError
std::pair<Ubx::Hdr, UbxPktReader> UbxPkt::parse_header(ustring_view s)
{
    if (s.size() < Ubx::Hdr::hdr_len)
    {
        // Must have at least class(1B) + ID(1) + length(2)
        throw Ubx::ParseError("Packet too short to parse header");
    }

    // Check validity of represented packet length (excludes sync chars, class,
    //  packet ID, length and checksum fields)
    UbxPktReader reader(s);
    Ubx::Hdr header;
    header.class_id = reader.read<Ubx::Field::U1>();
    header.msg_id = reader.read<Ubx::Field::U1>();
    header.len = reader.read<Ubx::Field::U2>();
    return std::make_pair(header, reader);
}

// Arg: a view of the packet after sync char prefix has been stripped, and
//  header has already been parsed (see parse_header)
// Throws: Ubx::ParseError
UbxPkt UbxPkt::finish_parse(UbxPktReader& reader, Ubx::Hdr header)
{
    auto num_left = reader.remaining();
    if (num_left < 2)
    {
        // Must have at least checksum (2 bytes) remaining
        throw Ubx::ParseError("Packet too short");
    }

    // Check validity of packet length
    std::uint16_t expected_payload_len = num_left - 2; // no checksum
    if (header.len != expected_payload_len)
    {
        throw Ubx::ParseError("Unexpected packet length");
    }

    UbxPkt pkt;
    pkt.header = std::move(header);

    if (!Ubx::is_valid_class(pkt.header.class_id))
        throw Ubx::ParseError("Invalid class ID");
    auto class_id = static_cast<Ubx::ClassId>(pkt.header.class_id);
    pkt.parse_payload(reader, class_id, pkt.header.msg_id);

    // Skip undocumented debug classes, which may not present a valid checksum
    if ((class_id == Ubx::ClassId::UNK1) || (class_id == Ubx::ClassId::UNK2) ||
            ((class_id == Ubx::ClassId::MON) && (pkt.header.msg_id == 0x11)))
    {
        reader.read<Ubx::Field::U2>(); // read checksum
        if (!reader.done())
            throw Ubx::RecoverableParseError("Incomplete parse (debug msg)");
        return pkt;
    }

    // Check validity of packet checksum
    // Compute checksum of header + payload (but not checksum or sync bytes)
    std::uint16_t calc_pkt_checksum = reader.checksum();
    // Note: This next read needs to happen *after* grabbing the rolling 
    //  checksum in the line above
    std::uint16_t pkt_checksum = reader.read<Ubx::Field::U2>();
    if (pkt_checksum != calc_pkt_checksum)
    {
        std::ostringstream oss;
        oss << "Invalid pkt checksum (class: x" << std::uppercase << std::hex
            << static_cast<int>(pkt.header.class_id) << ", msg: x"
            << static_cast<int>(pkt.header.msg_id) << "); read x"
            << pkt_checksum << " but expected x" << calc_pkt_checksum;
        oss << "[DEBUG INFO: has payload: " << (pkt.payload.get() != nullptr)
            << ", reader done: " << reader.done() << "]";

    #if defined(PERMISSIVE_CHECKSUM_HANDLING)
        // Checksum error tolerated even if payload is parseable
        // Nonetheless, clear any payload so potentially invalid data is not
        //   used
        bool has_payload = false;
        pkt.payload.reset();
    #else
        bool has_payload = (pkt.payload != nullptr);
    #endif // PERMISSIVE_CHECKSUM_HANDLING

        if (has_payload || !reader.done())
        {
            throw Ubx::ParseError(oss.str());
        }
        else
        {
            oss << "; ignored since payload not parsed";
            throw Ubx::RecoverableParseError(oss.str());
        }
    }

    // Check that the reader's view is exhausted by the parse; if not, this
    //  indicates a parsing error
    if (!reader.done())
        throw Ubx::ParseError("Incomplete parse");

    return pkt;
}

// Arg: a view of the packet after sync char prefix has been stripped
// Throws: Ubx::ParseError
UbxPkt UbxPkt::parse(ustring_view s)
{
    std::pair<Ubx::Hdr, UbxPktReader> header_and_reader = parse_header(s);
    auto& reader = std::get<UbxPktReader>(header_and_reader);
    auto& header = std::get<Ubx::Hdr>(header_and_reader);
    return finish_parse(reader, header);
}

#if defined(UNIT_TEST)
#include <fstream>
#include <iostream>
#include <type_traits>
#include <vector>

namespace
{
    // Adds a checksum and returns a string view to the packet (minus sync)
    auto add_ubx_checksum(std::vector<unsigned char>& v)
    {
        ustring_view sv_cs{v.data(), v.size()};
        sv_cs.remove_prefix(2); // ignore sync bytes
        std::uint16_t test_checksum = Ubx::checksum(sv_cs);
        // Include checksum bytes (little-endian)
        v.push_back(test_checksum & 0xFF);
        v.push_back(test_checksum >> 8);
        ustring_view sv{v.data(), v.size()};
        sv.remove_prefix(2); // ignore sync bytes
        return sv;
    }

    bool discard_until_sync(ustring_view& buf)
    {
        std::vector<unsigned char> sync = 
            {UbxPkt::sync_char1, UbxPkt::sync_char2};
        auto sync_start_pos = buf.find(ustring_view{sync.data(), sync.size()});
        if (sync_start_pos == ustring_view::npos)
            return false;
        // Matched sync
        buf.remove_prefix(sync_start_pos);
        if (sync_start_pos > 0)
            std::cout << "Skipped " << sync_start_pos << " bytes\n";
        return true;
    }

    template <Ubx::Field F>
    void add_payload_member_bytes(UbxPktMember<F>& member, 
        std::vector<unsigned char>& v)
    {
        // Push bytes in little-endian ordering
        for (size_t i = 0; i < ubx_field_size<F>; ++i)
            v.push_back((member >> (8 * i)) & 0xFF);
    }

    template <class UbxPayloadType, Ubx::Field... Ts>
    void test_ubx_for_payload(const UbxPayloadType& test_payload, 
        UbxPktMember<Ts>&... payload_members)
    {
        std::vector<unsigned char> test_pkt_vec = {
            UbxPkt::sync_char1, UbxPkt::sync_char2,
            static_cast<unsigned char>(UbxPayloadType::class_id),
            UbxPayloadType::msg_id};
        std::vector<unsigned char> test_payload_vec;
        // This unusual construct performs a sequenced call of the function
        //  with variadic pack expansion
        int dummy[] = {0, ((void)add_payload_member_bytes(payload_members, 
            test_payload_vec), 0)...};
        (void)dummy; // just to nix unused variable warning
        test_pkt_vec.push_back(test_payload_vec.size() & 0xFF);
        test_pkt_vec.push_back(test_payload_vec.size() >> 8);
        test_pkt_vec.insert(test_pkt_vec.end(), test_payload_vec.begin(), 
            test_payload_vec.end());

        ustring_view test_sv = add_ubx_checksum(test_pkt_vec);
        UbxPkt test_parsed = UbxPkt::parse(test_sv);
        static_assert(std::is_standard_layout<UbxPayloadType>::value,
            "Using memcmp to compare payloads, assumed to be std layout types");
        assert(test_parsed.payload && "Null payload");
        assert((std::memcmp(test_parsed.payload.get(), &test_payload, 
            sizeof(UbxPayloadType)) == 0) && "Parsed payload failed to match");
        // If not standard layout, define operator== for each type, then:
        // assert(*static_cast<const UbxPayloadType*>(
        //    test_parsed.payload.get()) == test_payload);
    }
}

int main()
{
    try
    {
        // Test: Parse generated packets
        // (ACK-ACK is first demonstrated to show the full process; to test 
        //  additional simple message types, the automated process below is 
        //  instead recommended)
        std::vector<unsigned char> test_ack = {
            UbxPkt::sync_char1, UbxPkt::sync_char2,
            0x05,  // ACK class
            0x01,  // ACK msg. ID
            0x02, 0x00, // Length
            static_cast<int>(Ubx::ClassId::CFG),  // Ack'd pkt class (say, CFG)
            0x13,  // Acknowledged msg. ID (say, 0x13 for CFG-ANT)
            };
        ustring_view test_ack_sv_cs{test_ack.data(), test_ack.size()};
        test_ack_sv_cs.remove_prefix(2); // ignore sync bytes
        std::uint16_t test_ack_checksum = Ubx::checksum(test_ack_sv_cs);
        // Include checksum bytes
        test_ack.push_back(test_ack_checksum & 0xFF);
        test_ack.push_back(test_ack_checksum >> 8);
        ustring_view test_ack_sv{test_ack.data(), test_ack.size()};
        test_ack_sv.remove_prefix(2); // ignore sync bytes
        UbxPkt test_ack_parsed = UbxPkt::parse(test_ack_sv);
        UbxPayloadAckAck* test_ack_payload = 
            static_cast<UbxPayloadAckAck*>(test_ack_parsed.payload.get());
        assert(test_ack_payload != 0);
        assert(test_ack_payload->ack_class_id == 
            static_cast<int>(Ubx::ClassId::CFG));
        assert(test_ack_payload->ack_msg_id == 0x13);

        UbxPayloadAckNak test_ack_nak;
        test_ubx_for_payload(test_ack_nak, test_ack_nak.nak_class_id, 
            test_ack_nak.nak_msg_id);

        // Test: Parse packets from binary capture
        std::ifstream ifs(
            "../include/thirdparty/gpsd/ublox-8-time.log", 
            std::ios_base::binary);
        ifs.unsetf(std::ios::skipws);
        ifs.exceptions(std::ios_base::badbit);
        if (!ifs)
            throw std::runtime_error("Unable to open UBX data file");
        size_t buf_len{15801}; // size of ublox-8-time.log test file
        std::vector<unsigned char> buf;
        buf.reserve(buf_len);
        size_t num_read = 0;
        while (ifs)
        {
            const size_t small_buf_len{100};
            unsigned char small_buf[small_buf_len];
            char* small_buf_ptr = reinterpret_cast<char*>(small_buf);
            int read_ret = ifs.readsome(small_buf_ptr, 100);
            if (read_ret <= 0)
            {
                if ((read_ret < 0) && !ifs.eof())
                    throw std::runtime_error("Unable to read packet");
                if (read_ret == 0)
                    break;
            }
            else
            {
                buf.insert(buf.end(), small_buf, small_buf + read_ret);
                num_read += read_ret;
            }
        }
        assert(num_read == buf.size());
        std::cout << "Read " << num_read << " bytes from binary file\n";
        auto pkt_view = ustring_view{buf.data(), num_read};
        // Capture ends with a partial packet; remove for testing here
        pkt_view.remove_suffix(19);
        while (discard_until_sync(pkt_view))
        {
            // Found sync; remove and parse the packet
            pkt_view.remove_prefix(2); // remove sync bytes
            std::pair<Ubx::Hdr, UbxPktReader> header_and_reader = 
                UbxPkt::parse_header(pkt_view);
            auto& reader = std::get<UbxPktReader>(header_and_reader);
            auto& header = std::get<Ubx::Hdr>(header_and_reader);
            ustring_view pkt_remainder_view{pkt_view};
            pkt_remainder_view.remove_prefix(Ubx::Hdr::hdr_len);
            pkt_remainder_view.remove_suffix(pkt_remainder_view.length() - 
                header.len - 2);
            reader.set_view(pkt_remainder_view);
            UbxPkt pkt = UbxPkt::finish_parse(reader, header);
            std::cout << "Parsed pkt of length " << header.len << 
                " (Class: x" << std::hex << static_cast<int>(header.class_id) 
                << ", Msg: x" << static_cast<int>(header.msg_id) 
                << ")" << std::dec << std::endl;
            if (pkt.get_payload_checked<UbxPayloadNavSbas>() != 0)
            {
                auto sbas_payload = 
                    pkt.get_payload_checked<UbxPayloadNavSbas>();
                std::cout << "\tNAV-SBAS: itow: " << sbas_payload->itow 
                    << ", geo: " << static_cast<int>(sbas_payload->geo)
                    << ", num SVs: " << static_cast<int>(sbas_payload->cnt) 
                    << std::endl;
            }
            if (pkt.get_payload_checked<UbxPayloadNavSol>() != 0)
            {
                auto sol_payload = pkt.get_payload_checked<UbxPayloadNavSol>();
                std::cout << "\tNAV-SOL: itow: " << sol_payload->itow 
                    << ", ftow: " << sol_payload->ftow << ", fix: " 
                    << static_cast<int>(sol_payload->gps_fix) 
                    << ", x: " << sol_payload->ecef_x
                    << ", y: " << sol_payload->ecef_y << std::endl;
            }
            if (pkt.get_payload_checked<UbxPayloadNavTimeGps>() != 0)
            {
                auto time_payload = 
                    pkt.get_payload_checked<UbxPayloadNavTimeGps>();
                std::cout << "\tNAV-TIMEGPS: itow: " << time_payload->itow 
                    << ", ftow: " << time_payload->ftow << std::endl;
            }
            pkt_view.remove_prefix(Ubx::Hdr::hdr_len);
            pkt_view.remove_prefix(header.len + 2);
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "UNSPECIFIED FATAL ERROR" << std::endl;
    }
}

#endif

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

