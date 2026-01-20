// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_LDAONGRIDSPIN_H
#define MGMOL_LDAONGRIDSPIN_H

#include "LDAFunctional.h"
#include "MGmol_MPI.h"
#include "Mesh.h"
#include "Rho.h"
#include "XConGrid.h"

#include <vector>

class Potentials;

template <class T>
class LDAonGridSpin : public XConGrid
{
    int np_;
    int myspin_;

    LDAFunctional* lda_;
    Rho<T>& rho_;

    Potentials& pot_;

public:
    LDAonGridSpin(Rho<T>& rho, Potentials& pot)
        : np_(rho.rho_[0].size()), rho_(rho), pot_(pot)
    {
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        myspin_         = mmpi.myspin();
        lda_            = new LDAFunctional(rho.rho_);
    }

    ~LDAonGridSpin() override { delete lda_; }

    void update() override;

    double getExc() const override; // in [Ha]
};

#endif
