#include "time_msg.h"
#include <regex>
#include <sstream>
#include "date/date.h"
#include "utility.h"

// time_msg
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
template <typename Clock, typename Duration>
std::chrono::time_point<Clock, Duration> from_sys_time_point(
    const date::sys_time<Duration>& tp)
{
    // FIXME Double-check; likely needs to adjust for epoch differences
    return std::chrono::time_point<Clock, Duration>{tp.time_since_epoch()};
}
} // end namespace

namespace tmon
{

TimeMsgHeader TimeMsgHeader::parse(std::istream& is) /* throws ParseError */
{
    TimeMsgHeader ret{};

    std::string line;
    static const std::regex start_regex{R"(^CLOCK_REGISTRY\s+VERSION=(\S+)$)"};
    static const std::regex registry_item_regex{
        R"#(^\s*ID=(\d+)\s+NAME="(.+)"\s+MODEL_CLASS="(.+)"$)#"};
    static const std::regex registry_item_regex2{
        R"#(^\s*ID=(\d+)\s+NAME="(.+)"\s+MODEL_DIFF="((?:[0-9.,\-\[\]eE]+|nan|NaN|\s)+)"\s+MODEL_MEAN="((?:[0-9.,\-\[\]eE]+|nan|NaN|\s)+)"$)#"};
    static const std::regex end_regex{R"(^END_CLOCK_REGISTRY$)"};

    auto parse_dbl_vector = [](const std::string& s){
        static const std::regex vector_elt_regex{R"([0-9.eE-]+|nan|NaN)"};
        std::vector<double> ret;
        for (std::sregex_iterator i{s.begin(), s.end(), vector_elt_regex};
             i != std::sregex_iterator{};
             ++i)
        {
            ret.push_back(std::stod(i->str()));
        }
        return ret;
        };

    while (std::getline(is, line))
    {
        std::smatch match_res;
        if (std::regex_match(line, match_res, start_regex))
        {
            assert(match_res.size() >= 2); // whole string + 1 sub-match
            ret.protocol_version = match_res[1].str();
        }
        else if (std::regex_match(line, match_res, registry_item_regex))
        {
            assert(match_res.size() >= 4); // whole string + 3 sub-matches
            ClockId clock_id{std::stoi(match_res[1].str())};
            std::string task_name{"<unspecified-task>"};
            std::string clock_name{match_res[2].str()};
            optional<GaussMarkovModel::ClockClass> clock_class =
                GaussMarkovModel::get_class_by_name(match_res[3].str());
            if (!clock_class)
                throw ParseError{"Invalid clock class name in TimeMsgHeader"};
            GaussMarkovModel clock_model{*clock_class};
            ClockDesc clock_desc{task_name, clock_name, std::move(clock_model)};
            auto insert_result = ret.clock_registry.insert(std::make_pair(
                clock_id, std::move(clock_desc)));
            if (insert_result.second != true)
                throw ParseError{"Duplicate clock id in TimeMsgHeader"};
        }
        else if (std::regex_match(line, match_res, registry_item_regex2))
        {
            assert(match_res.size() >= 5); // whole string + 4 sub-matches
            ClockId clock_id{std::stoi(match_res[1].str())};
            std::string task_name{"<unspecified-task>"};
            std::string clock_name{match_res[2].str()};
            GaussMarkovModel clock_model{};
            auto q_vec = parse_dbl_vector(match_res[3].str());
            auto mu_vec = parse_dbl_vector(match_res[4].str());
            int q_len = q_vec.size();
            int mu_len = mu_vec.size();
            clock_model.q = Eigen::Map<Eigen::VectorXd>{q_vec.data(), q_len};
            clock_model.mu = Eigen::Map<Eigen::VectorXd>{mu_vec.data(), mu_len};
            ClockDesc clock_desc{task_name, clock_name, std::move(clock_model)};
            auto insert_result = ret.clock_registry.insert(std::make_pair(
                clock_id, std::move(clock_desc)));
            if (insert_result.second != true)
                throw ParseError{"Duplicate clock id in TimeMsgHeader"};
        }
        else if (std::regex_match(line, match_res, end_regex))
        {
            return ret;
        }
        else
        {
            throw ParseError{"Unexpected line for TimeMsgHeader"};
        }
    }

    throw ParseError{"Early EOF for TimeMsgHeader"};
}

std::ostream& operator<<(std::ostream& os, const TimeMsgHeader& header)
{
    static const std::string protocol_version{"1.0"};
    assert(header.protocol_version == protocol_version);
    os << "CLOCK_REGISTRY VERSION=" << protocol_version << '\n';
    for (const auto& x : header.clock_registry)
    {
        const GaussMarkovModel& gm_model = x.second.gm_model;
        os << "\tID=" << x.first << " NAME=\"" << x.second.describe()
            << "\" MODEL_DIFF=\"" << gm_model.q.transpose()
            << "\" MODEL_MEAN=\"" << gm_model.mu.transpose() << "\"\n";
    }
    os << "END_CLOCK_REGISTRY\n";
    return os;
}

std::string to_string(const TimeMsgHeader& header)
{
    std::ostringstream oss;
    oss << header;
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const TimeMsg& msg)
{
    using date::operator<<;
    os << "ID=" << msg.clock_id << " MSG_TIME="
        << to_string(msg.msg_creation)
        << " ORIG_TIME="
        << to_string(msg.orig_timestamp) << '\n';
    if (msg.time_since_last_msg)
        os << "TIME_VS_LAST_MSG=" << *msg.time_since_last_msg << '\n';
    if (msg.time_since_last_orig)
        os << "TIME_VS_LAST_ORIG=" << *msg.time_since_last_orig << '\n';
    for (const auto& x : msg.comparisons)
    {
        os << "TIME_VS ID=" << x.other_clock_id << " DELTA="
            << x.time_versus_other << '\n';
    }
    os << "END_TIME_MSG\n";
    return os;
}

std::string to_string(const TimeMsg& msg)
{
    std::ostringstream oss;
    oss << msg;
    return oss.str();
}

TimeMsg parse_time_msg(std::istream& is) /* throws ParseError */
{
    using date::operator>>;
    TimeMsg ret{};
    std::string line;
    static const std::regex id_regex{
        R"(^ID=(\d+)\s+MSG_TIME=(\S+\s\S+)\s+ORIG_TIME=(\S+\s\S+)$)"};
    static const std::regex since_last_msg_regex{
        R"(^\s*TIME_VS_LAST_MSG=(\d+)ps$)"};
    static const std::regex since_last_orig_regex{
        R"(^\s*TIME_VS_LAST_ORIG=(\d+)ps$)"};
    static const std::regex compare_regex{
        R"(^\s*TIME_VS ID=(\d+)\s+DELTA=(-?\d+)ps$)"};
    static const std::regex end_regex{R"(^END_TIME_MSG$)"};
    while (std::getline(is, line))
    {
        std::smatch match_res;
        if (std::regex_match(line, match_res, id_regex))
        {
            if (match_res.size() < 4) // whole string + 3 sub-matches
                throw ParseError{"Bad ID line for TimeMsg"};
            std::istringstream iss{match_res[1].str()};
            iss >> ret.clock_id;
            if (!iss)
                throw ParseError{"Failed parse of clock ID"};
            iss.clear();
            iss.str(match_res[2].str());
            if (!date::from_stream(iss, "%F %T", ret.msg_creation))
                throw ParseError{"Failed parse of msg timestamp"};
            iss.clear();
            iss.str(match_res[3].str());
            date::sys_time<TimeMsg::OrigTimePointType::duration> orig_time_st;
            if (!date::from_stream(iss, "%F %T", orig_time_st))
                throw ParseError{"Failed parse of orig timestamp"};
            TimeMsg::OrigTimePointType orig_time =
                from_sys_time_point<TimeMsg::OrigClockType>(orig_time_st);
            ret.orig_timestamp = orig_time;
        }
        else if (std::regex_match(line, match_res, since_last_msg_regex))
        {
            if (match_res.size() < 2) // whole string + 1 sub-match
                throw ParseError{"Bad since-last-msg line for TimeMsg"};
            TimeMsg::DurationType::rep read_ps = std::stoll(match_res[1].str());
            ret.time_since_last_msg = TimeMsg::DurationType{read_ps};
        }
        else if (std::regex_match(line, match_res, since_last_orig_regex))
        {
            if (match_res.size() < 2) // whole string + 1 sub-match
                throw ParseError{"Bad since-last-orig line for TimeMsg"};
            TimeMsg::DurationType::rep read_ps = std::stoll(match_res[1].str());
            ret.time_since_last_orig = TimeMsg::DurationType{read_ps};
        }
        else if (std::regex_match(line, match_res, compare_regex))
        {
            if (match_res.size() < 3) // whole string + 2 sub-matches
                throw ParseError{"Bad comparison line for TimeMsg"};
            TimeMsg::Comparison new_comp;
            std::istringstream iss{match_res[1].str()};
            iss >> new_comp.other_clock_id;
            TimeMsg::DurationType::rep read_ps = std::stoll(match_res[2].str());
            new_comp.time_versus_other = TimeMsg::DurationType{read_ps};
            ret.comparisons.push_back(new_comp);
        }
        else if (std::regex_match(line, match_res, end_regex))
        {
            return ret;
        }
        else
        {
            throw ParseError{"Unexpected line for TimeMsg"};
        }
    }
    throw ParseError{"Early EOF for TimeMsg"};
}

} // end namespace

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

