#include <limits>
#include <type_traits>
#include <utility>
#include <boost/endian/arithmetic.hpp>
#include <boost/mp11.hpp>
#include "parse_ubx.h"

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

// The UBX protocol is stated in terms of little-endian fields, which the
// architecture is assumed to match; check this assumption first:
#if defined(BOOST_BIG_ENDIAN)
    #error "Decoding assumes a little-endian architecture"
#endif

// Similarly, the UBX protocol (and thus this parser) assumes IEEE 754 standard
//  floating point types
static_assert(std::numeric_limits<float>::is_iec559, 
    "IEC 559 / IEEE 754 standard floats are assumed");


template <Ubx::Field T>
struct ubx_field_impl
{
    using type = void;
    static constexpr size_t size = 0;
};

template <>
struct ubx_field_impl<Ubx::Field::U1>
{
    using type = std::uint8_t;
    static constexpr size_t size = 1;
};
template <>
struct ubx_field_impl<Ubx::Field::RU1_3>
{
    using type = std::uint8_t;
    static constexpr size_t size = 1;
};
template <>
struct ubx_field_impl<Ubx::Field::I1>
{
    using type = std::int8_t;
    static constexpr size_t size = 1;
};
template <>
struct ubx_field_impl<Ubx::Field::X1>
{
    using type = std::uint8_t;
    static constexpr size_t size = 1;
};
template <>
struct ubx_field_impl<Ubx::Field::U2>
{
    using type = std::uint16_t;
    static constexpr size_t size = 2;
};
template <>
struct ubx_field_impl<Ubx::Field::I2>
{
    using type = std::int16_t;
    static constexpr size_t size = 2;
};
template <>
struct ubx_field_impl<Ubx::Field::X2>
{
    using type = std::uint16_t;
    static constexpr size_t size = 2;
};
template <>
struct ubx_field_impl<Ubx::Field::U4>
{
    using type = std::uint32_t;
    static constexpr size_t size = 4;
};
template <>
struct ubx_field_impl<Ubx::Field::I4>
{
    using type = std::int32_t;
    static constexpr size_t size = 4;
};
template <>
struct ubx_field_impl<Ubx::Field::X4>
{
    using type = std::uint32_t;
    static constexpr size_t size = 4;
};
template <>
struct ubx_field_impl<Ubx::Field::R4>
{
    using type = float;
    static constexpr size_t size = 4;
};
template <>
struct ubx_field_impl<Ubx::Field::R8>
{
    using type = double;
    static constexpr size_t size = 8;
};
template <>
struct ubx_field_impl<Ubx::Field::CH>
{
    using type = char;
    static constexpr size_t size = 1;
};

template <Ubx::Field T>
using ubx_field_type = typename ubx_field_impl<T>::type;
template <Ubx::Field T>
constexpr size_t ubx_field_size = ubx_field_impl<T>::size;

namespace Ubx
{
// Checksum calculated via the 8-bit Flecther algorithm of RFC 1145, Section
//  31.4 of the UBX M8 Protocol Specification
constexpr void update_checksum(ustring_view s,
    ChecksumPair& ck_pair)
{
    for (auto x : s)
    {
        ck_pair.first  += x;
        ck_pair.second += ck_pair.first;
    }
}

constexpr std::uint16_t checksum(ustring_view s)
{
    // Return checksum in little-endian format
    ChecksumPair ck{};
    update_checksum(s, ck);
    return (ck.second << 8) + ck.first;
}

template <typename T>
struct MemberPolicy
{
    static constexpr void range_check(T)
    {
        // default: all values permitted
    }
};

template <typename T,
    T range_min = std::numeric_limits<T>::min(),
    T range_max = std::numeric_limits<T>::max()>
struct CheckedMemberPolicy
{
    static constexpr void range_check(T x)
    {
        if ((x < range_min) || (x > range_max))
            throw Ubx::ParseError("UBX field member out of range");
    }
};

template <typename PayloadTypes, typename Functor>
Functor for_matching_payload(Ubx::ClassId cls_id, Ubx::MsgIdType msg_id,
    Functor fn)
{
    using Idxs = boost::mp11::mp_iota<boost::mp11::mp_size<PayloadTypes>>;
    boost::mp11::mp_for_each<Idxs>([&](auto N){
        using CurrPayloadType = boost::mp11::mp_at_c<PayloadTypes, N>;
        if ((CurrPayloadType::class_id == cls_id) && 
            (CurrPayloadType::msg_id == msg_id))
        {
            fn.template operator()<CurrPayloadType>();
        }
    });
    return fn;
}

template <typename Fn>
class PayloadApplyFunctor
{
  public:
    PayloadApplyFunctor(const UbxPkt& pkt, Fn f) : pkt_{pkt}, fn_{f}
    {
    }

    template <typename T>
    void operator()()
    {
        const T* payload = pkt_.get_payload_checked<T>();
        if (payload)
            fn_(*payload);
        else
            assert(!pkt_.payload); // type check should succeed unless null
    }

  private:
    const UbxPkt& pkt_;
    Fn fn_;
};

constexpr bool is_valid_class(ClassIdType val)
{
    switch (val)
    {
        case static_cast<ClassIdType>(ClassId::NAV):
        case static_cast<ClassIdType>(ClassId::RXM):
        case static_cast<ClassIdType>(ClassId::INF):
        case static_cast<ClassIdType>(ClassId::ACK):
        case static_cast<ClassIdType>(ClassId::CFG):
        case static_cast<ClassIdType>(ClassId::UPD):
        case static_cast<ClassIdType>(ClassId::MON):
        case static_cast<ClassIdType>(ClassId::AID):
        case static_cast<ClassIdType>(ClassId::TIM):
        case static_cast<ClassIdType>(ClassId::ESF):
        case static_cast<ClassIdType>(ClassId::MGA):
        case static_cast<ClassIdType>(ClassId::LOG):
        case static_cast<ClassIdType>(ClassId::SEC):
        case static_cast<ClassIdType>(ClassId::HNR):
        case static_cast<ClassIdType>(ClassId::UNK1):
        case static_cast<ClassIdType>(ClassId::UNK2):
            return true;
        default:
            return false;
    }
}

} // namespace Ubx

template <Ubx::Field F,
    ubx_field_type<F> range_min = std::numeric_limits<ubx_field_type<F>>::min(),
    ubx_field_type<F> range_max = std::numeric_limits<ubx_field_type<F>>::max(),
    template <class T, T, T> class Policy = Ubx::CheckedMemberPolicy>
struct UbxPktMember
{
    static const Ubx::Field field = F;
    using type = ubx_field_type<F>;
    using CheckingPolicy = Policy<type, range_min, range_max>;
    operator type() { return t_; }
    operator const type() const { return t_; }
    UbxPktMember() = default;
    explicit UbxPktMember(type t) : t_(std::move(t))
    {
        CheckingPolicy::range_check(t_);
    }
    ~UbxPktMember() = default;
    UbxPktMember(UbxPktMember&& other) = default;
    UbxPktMember& operator=(UbxPktMember&& other) = default;
    UbxPktMember(const UbxPktMember& other) = default;
    UbxPktMember& operator=(const UbxPktMember& other) = default;
    UbxPktMember& operator=(type t)
    {
        t_ = std::move(t);
        CheckingPolicy::range_check(t_);
        return *this;
    }

  private:
    type t_;
};

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

