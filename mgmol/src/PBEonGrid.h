// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_PBEONGRID_H
#define MGMOL_PBEONGRID_H

#include "PBEFunctional.h"
#include "Rho.h"
#include "XConGrid.h"

class Potentials;

template <class T>
class PBEonGrid : public XConGrid
{
    int np_;
    PBEFunctional* pbe_;
    Rho<T>& rho_;

    Potentials& pot_;

public:
    PBEonGrid(Rho<T>& rho, Potentials& pot) : rho_(rho), pot_(pot)
    {
        np_  = rho.rho_[0].size();
        pbe_ = new PBEFunctional(rho.rho_);
    }

    ~PBEonGrid() override { delete pbe_; }

    void update() override;

    double getExc() const override;
};

#endif
