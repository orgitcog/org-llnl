#ifndef COMMON_H_
#define COMMON_H_

#include <boost/chrono.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/trivial.hpp>
#define BOOST_THREAD_PROVIDES_FUTURE
#include <boost/thread/future.hpp>
#include <boost/variant.hpp>
#include <cassert>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <experimental/optional>
#include <experimental/string_view>
#include <utility>

// common - Common includes for all files
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

// Severity levels are trace < debug < info < warning < error < fatal
using SeverityType = boost::log::trivial::severity_level; 
static constexpr SeverityType min_static_severity{boost::log::trivial::trace};
//#define TM_LOG(lvl) BOOST_LOG_SEV(get_logger(), boost::log::trivial::lvl)
#define TM_LOG(lvl) if (boost::log::trivial::lvl >= min_static_severity) \
    BOOST_LOG_SEV(get_logger(), boost::log::trivial::lvl)
using LoggerType = boost::log::sources::severity_logger_mt<SeverityType>;

namespace tmon
{
template <typename T>
using basic_string_view = std::experimental::basic_string_view<T>;
using string_view = std::experimental::string_view;
using ustring_view = basic_string_view<unsigned char>;

template <typename T>
using optional = std::experimental::optional<T>;

template <typename... Ts>
using variant = boost::variant<Ts...>;
template <typename Visitor, typename Variant>
inline decltype(auto) visit(const Visitor& visitor, Variant&& variant)
{
    return boost::apply_visitor(visitor, variant);
}

// Synchronization primitives
using boost::condition_variable;
template <typename T>
using future = boost::future<T>;
template <typename T>
using lock_guard = boost::lock_guard<T>;
using boost::mutex;
template <typename T>
using unique_lock = boost::unique_lock<T>;

using ClockId = int;

// Time-related
template <class Rep, class Period>
auto boost_duration_cast(std::chrono::duration<Rep, Period> d)
{
    using BoostRatio = boost::ratio<Period::num, Period::den>;
    return boost::chrono::duration<Rep, BoostRatio>(d.count());
}
using picoseconds = std::chrono::duration<std::int64_t, std::pico>;


} // namespace tmon

using namespace tmon;

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

#endif // COMMON_H_
