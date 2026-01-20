#ifndef CLOCK_DESC_H_
#define CLOCK_DESC_H_

#include "common.h"
#include "gm_model.h"

// ClockDesc - Description of a clock (name, task name, Gauss-Markov model)
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

struct ClockDesc
{
    std::string task_name;
    std::string clock_name;
    GaussMarkovModel gm_model;
    friend bool operator==(const ClockDesc& a, const ClockDesc& b)
    {
        return (a.task_name == b.task_name) && 
            (a.clock_name == b.clock_name) &&
            (a.gm_model == b.gm_model);
    }
    std::string describe() const;
};

} // namespace tmon

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

#endif // CLOCK_DESC_H_
