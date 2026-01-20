#ifndef PARAM_H_
#define PARAM_H_

#include <initializer_list>
#include <iostream>
#include <Eigen/Core>
#include "common.h"

// Parameter Types (cf. prog_state.h)
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

template <typename T>
class VectorParam
{
  public:
    VectorParam() = default;
    VectorParam(std::initializer_list<T> i)
        : cont_{i}
    {
    }
    ~VectorParam() = default;
    VectorParam(const VectorParam&) = default;
    VectorParam& operator=(const VectorParam&) = default;
    VectorParam& operator=(VectorParam&&) = default;
    VectorParam(VectorParam&&) = default;

    using EigenVectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    operator EigenVectorX() const
    {
        return EigenVectorX::Map(cont_.data(), cont_.size());
    }

    friend std::ostream& operator<<(std::ostream& os, const VectorParam& p)
    {
        os << '[';
        for (std::size_t i = 0; i < p.cont_.size(); ++i)
        {
            os << p.cont_[i];
            if (i != p.cont_.size() - 1)
                os << ", ";
        }
        os << ']';
        return os;
    }

    friend std::istream& operator>>(std::istream& is, VectorParam& p)
    {
        is >> std::ws;
        if (is.peek() == '[')
            is.get();

        while (is)
        {
            T t;
            if (is >> t >> std::ws)
                p.cont_.push_back(std::move(t));
            if (is.peek() == ',')
            {
                is.get();
            }
            else if (is.peek() == ']')
            {
                is.get();
                break;
            }
        }
        return is;
    }

  private:
    std::vector<T> cont_;
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

#endif // PARAM_H_

