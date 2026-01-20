#include "gm_model.h"
#include <limits>
#include <regex>

// Gauss-Markov Model
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

GaussMarkovModel::GaussMarkovModel(ClockClass clk_class)
{
    switch (clk_class)
    {
        case ClockClass::cesium:
            q  << 2.50e-23, 4.44e-37, 5.0e-53;
            mu << 1.0e-14, 0, 0;
            break;
        case ClockClass::rubidium:
            q  << 1.0e-24, 1.1e-35, 2.8e-46;
            mu << 5.0e-11, 1.0e-10 / 365, 0;
            break;
        case ClockClass::vctcxo:
            q  << 5.0e-22, 1.97e-19, 1.0e-30;
            mu << 1.0e-6, 5.0e-7 / 365, 0;
            break;
        case ClockClass::ocxo_poor:
            q  << 1.26e-26, 4.95e-21, 1.0e-32;
            mu << 1.0e-8, 1.0e-9, 0;
            break;
        case ClockClass::ocxo_mid:
            q  << 1.0e-26, 1.0e-22, 1.0e-32;
            mu << 1.0e-9, 5.0e-10, 0;
            break;
        case ClockClass::ocxo_good:
            q  << 1.0e-27, 1.0e-23, 1.0e-32;
            mu << 1.0e-10, 3.0e-10, 0;
            break;
        case ClockClass::ocxo_fs740:
            q  << 1.8e-27, 1.4e-24, 1.0e-32;
            mu << 1.0e-10, 5.0e-10, 0;
            break;
        case ClockClass::ocxo_aocjy6:
            q  << 7.3e-25, 2.9e-22, 1.0e-34;
            mu << 1.0e-10, 1.0e-10, 0;
            break;
        case ClockClass::gpsdo_best:
            q  << 1.0e-22, 1.0e-30, 0;
            mu << 2.0e-13, 0, 0;
            break;
        case ClockClass::gpsdo_indoor:
            q  << 1.0e-18, 1.0e-30, 0;
            mu << 2.0e-13, 0, 0;
            break;
        case ClockClass::perfect:
            // Perfect clock; no process noise, no deterministic drift
            q.setZero();
            mu.setZero();
            break;
        case ClockClass::pseudo:
            q.setConstant(std::numeric_limits<double>::quiet_NaN());
            mu.setConstant(std::numeric_limits<double>::quiet_NaN());
            break;
        default:
            assert(!"Unknown clock class");
    }
}

// Construct the model from a string that either names a generic clock
//  class (cf. get_class_by_name) or provides a custom clock specification
//  in the form {q: [q1, q2, q3], mu: [mu1, mu2, mu3]}, where the components
//  of these vectors are floating-point coefficients
GaussMarkovModel::GaussMarkovModel(string_view class_str_view)
    /* throws std::range_error */
{
    // First, try to match a named generic clock class
    std::string class_str{class_str_view};
    auto class_opt = get_class_by_name(class_str);
    if (class_opt)
    {
        *this = GaussMarkovModel{*class_opt};
        return;
    }

    // Next, try to match a custom clock class specification in terms of
    // diffusion coefficients (q) and deterministic drift parameters (mu)
    //  Example: {q: [1e-20, 2.0e-10, NaN], mu: [1e-10, 2e-10, 3e-10]}
    static const std::regex custom_class_regex{R"(\s*\{q:\s*\[((?:(?:[0-9.\-eE]+|nan|NaN),?\s*){3})\],\s*mu:\s*\[((?:(?:[0-9.\-eE]+|nan|NaN),?\s*){3})\]\}\s*)"};
    std::smatch custom_class_match_res;
    if (!std::regex_match(class_str, custom_class_match_res,
            custom_class_regex))
    {
        // Neither class names nor custom class regex match; throw error
        throw std::range_error{"Invalid clock class specified: " +
            class_str};
    }
    // Custom class regex matches; parse parameter vectors
    assert(custom_class_match_res.size() >= 3); // whole string + 2 matches
    static const std::regex vector_elt_regex{R"([0-9.eE-]+|nan|NaN)"};
    std::smatch q_match, mu_match;
    std::string q_str = custom_class_match_res[1].str();
    std::string mu_str = custom_class_match_res[2].str();
    std::sregex_iterator qi{begin(q_str), end(q_str), vector_elt_regex};
    std::sregex_iterator mi{begin(mu_str), end(mu_str), vector_elt_regex};

    // Components are first extracted, then the vectors are initialized, since
    //  this ensures proper sequencing
    double q1{std::stod(qi->str())}, q2{std::stod((++qi)->str())},
        q3{std::stod((++qi)->str())};
    q << q1, q2, q3;
    double mu1{std::stod(mi->str())}, mu2{std::stod((++mi)->str())},
        mu3{std::stod((++mi)->str())};
    mu << mu1, mu2, mu3;
}

auto GaussMarkovModel::get_class_by_name(string_view name) ->
    optional<GaussMarkovModel::ClockClass>
{
    if (name == "cesium")
    {
        return ClockClass::cesium;
    }
    else if (name == "rubidium")
    {
        return ClockClass::rubidium;
    }
    else if (name == "vctcxo")
    {
        return ClockClass::vctcxo;
    }
    else if (name == "ocxo_poor")
    {
        return ClockClass::ocxo_poor;
    }
    else if (name == "ocxo_mid")
    {
        return ClockClass::ocxo_mid;
    }
    else if (name == "ocxo_good")
    {
        return ClockClass::ocxo_good;
    }
    else if (name == "ocxo_fs740")
    {
        return ClockClass::ocxo_fs740;
    }
    else if (name == "ocxo_aocjy6")
    {
        return ClockClass::ocxo_aocjy6;
    }
    else if (name == "gpsdo_best")
    {
        return ClockClass::gpsdo_best;
    }
    else if (name == "gpsdo_indoor")
    {
        return ClockClass::gpsdo_indoor;
    }
    else if (name == "perfect")
    {
        return ClockClass::perfect;
    }
    else if (name == "pseudo")
    {
        return ClockClass::pseudo;
    }
    else
    {
        return {};
    }
}

} // end namespace tmon

#ifdef UNIT_TEST
#include <iostream>
int main()
{
    GaussMarkovModel gm1{
        "{q: [1e-20, 2.0e-10, NaN], mu: [1e-10, 2e-10, 3e-10]}"};
    GaussMarkovModel::Vector gm1_q_goal{1e-20, 2e-10,
            std::numeric_limits<double>::quiet_NaN()};
    GaussMarkovModel::Vector gm1_mu_goal{1e-10, 2e-10, 3e-10};
    auto print_with_goal = [](const auto& x, const auto& q_goal,
            const auto& mu_goal)
        { std::cerr << x.q.transpose() << "\t | " << q_goal.transpose()
            << std::endl << x.mu.transpose() << "\t | "
            << mu_goal.transpose() << std::endl; };
    print_with_goal(gm1, gm1_q_goal, gm1_mu_goal);
    // Need a comparison lambda that tolerates NaNs for testing equality
    //  (so long as both vectors have a NaN in the same location)
    auto check_if_same = [](const auto& a, const auto& b) -> bool
        { return ((a.array() == b.array()) ||
                (a.array().isNaN() && b.array().isNaN())).all(); };
    assert(check_if_same(gm1.q, gm1_q_goal));
    assert(check_if_same(gm1.mu, gm1_mu_goal));

    GaussMarkovModel::Vector gm2_q_goal{2.50e-23, 4.44e-37, 5.0e-53};
    GaussMarkovModel::Vector gm2_mu_goal{1.0e-14, 0, 0};
    GaussMarkovModel gm2a{"cesium"};
    print_with_goal(gm2a, gm2_q_goal, gm2_mu_goal);
    assert(check_if_same(gm2a.q, gm2_q_goal));
    assert(check_if_same(gm2a.mu, gm2_mu_goal));
    GaussMarkovModel gm2b{GaussMarkovModel::ClockClass::cesium};
    print_with_goal(gm2b, gm2_q_goal, gm2_mu_goal);
    assert(check_if_same(gm2b.q, gm2_q_goal));
    assert(check_if_same(gm2b.mu, gm2_mu_goal));

    auto ensure_throws = [](const auto& x){
        bool threw{false};
        try
        {
            x();
        }
        catch(...)
        {
            threw = true;
        }
        assert(threw);
        std::cerr << "Caught exception as expected" << std::endl;
    };
    ensure_throws([]{GaussMarkovModel{
            "{q: [1e-20, 2.0e-10], mu: [1e-10, 2e-10, 3e-10]}"};});
    ensure_throws([]{GaussMarkovModel{
            "{q: [1e-20, 2.0e-10, 1], mu: [1e-10, 2e-10]}"};});
    ensure_throws([]{GaussMarkovModel{
            "{q: [1e-20, 2.0e-10, 1], mu: [1e-10, 2e-10, zebra]}"};});
    ensure_throws([]{GaussMarkovModel{
            "{q: [1e-20, 2.0e-10, 1], mu: [1e-10, 2e-10, 3e-10], junk}"};});
    ensure_throws([]{GaussMarkovModel{
            "{mu: [1e-10, 2e-10, 3e-10], q: [1e-20, 2e-10, 1]}"};});

    std::cerr << "Test succceeded." << std::endl;
}

#endif // UNIT_TEST

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

