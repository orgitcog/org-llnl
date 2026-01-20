#ifndef GM_MODEL_H_
#define GM_MODEL_H_

#include <Eigen/Core>
#include "common.h"

// Gauss-Markov Model
//  Represents a three-state Gauss-Markov model capable of representing a broad
//  array of time sources (including quartz oscillators).  The process noise
//  diffusion coefficients capture the levels of white FM, random-walk FM and
//  random-run FM noise.  The deterministic mean coefficients capture the levels
//  of syntonization error, aging, and linear coefficient of the frequency
//  drift.  This class allows for the specification of these coefficients
//  directly, or it can be initialized with a set of defaults characteristic of
//  a class of time sources.
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

struct GaussMarkovModel
{
    enum class ClockClass
    {
        cesium, rubidium, vctcxo, ocxo_poor, ocxo_mid, ocxo_good, ocxo_fs740,
            ocxo_aocjy6, gpsdo_best, gpsdo_indoor, perfect, pseudo
    };
    using Vector = Eigen::Vector3d;
    Vector q;   // process noise diffusion coefficients
    Vector mu;  // deterministic mean (control input)
    GaussMarkovModel() = default;
    // Construct the model from a generic clock class
    GaussMarkovModel(ClockClass clk_class);
    // Construct the model from a string that either names a generic clock
    //  class (cf. get_class_by_name) or provides a custom clock specification
    //  in the form {q: [q1, q2, q3], mu: [mu1, mu2, mu3]}, where the components
    //  of these vectors are floating-point coefficients
    GaussMarkovModel(string_view class_str);
    GaussMarkovModel(const GaussMarkovModel&) = default;
    GaussMarkovModel& operator=(const GaussMarkovModel&) = default;
    GaussMarkovModel(GaussMarkovModel&&) = default;
    GaussMarkovModel& operator=(GaussMarkovModel&&) = default;
    friend bool operator==(const GaussMarkovModel& a, const GaussMarkovModel& b)
    {
        return (a.q == b.q) && (a.mu == b.mu);
    }
    static optional<ClockClass> get_class_by_name(string_view name);

    // Ensure operator new returns an aligned pointer since this structure
    //  contains fixed-size Eigen vectors that may be vectorizable:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // or assert that it's not necessary:
    static_assert(sizeof(Vector) % 16 != 0, "GaussMarkovModel::Vector is "
        "vectorizable; needs operator new overload");
};

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

#endif // GM_MODEL_H_
