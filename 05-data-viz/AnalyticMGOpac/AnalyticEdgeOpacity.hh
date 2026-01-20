// Copyright 2024 Lawrence Livermore National Security, LLC.
// See the top-level LICENCE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef ANALYTICEDGEOPACITY_HH__
#define ANALYTICEDGEOPACITY_HH__

#include <vector>

namespace AnalyticMGOpac
{

// A simplified frequency-dependent analytic opacity with edge- and line-like features.
// The form and background of this opacity can be found in
//    T. A. Brunner, "A Family of Multi-Dimensional Thermal Radiative Transfer Test Problems",
//    2023, LLNL-TR-858450
// This class stores the parameters and evaluates the frequency dependent opacity
// in Eq. 24 of that document.
class AnalyticEdgeOpacity
{
  public:
   // epsilonMin: below this photon energy, the opacity is a constant
   // epsilonEdge: the photon energy of the shell edge
   // C0: The overall strength of the opacity
   // C1: The strength of the edge
   // C2: The strength of the lines
   // lineWidth: if numLines>0, the width of line-like features in photon energy
   // lineSep: if numLines>0, the separation line-like features in photon energy
   // numLines: the number of line-like features.
   AnalyticEdgeOpacity(double epsilonMin_,
                       double epsilonEdge_,
                       double C0_,
                       double C1_,
                       double C2_,
                       double lineWidth_,
                       double lineSep_,
                       int numLines_)
      : epsilonMin{epsilonMin_},
        epsilonEdge{epsilonEdge_},
        C0{C0_},
        C1{C1_},
        C2{C2_},
        lineWidth{lineWidth_},
        lineSep{lineSep_},
        numLines{numLines_}
   {
   }

   // Computes the frequency dependent specific opacity with units of area per mass,
   // $\kappa =\sigma/\rho$, where $\sigma$ is from Eq. 24.
   // epsilon: photon energy $\epsilon = h\nu$ in the same units as T
   // T: material temperature, really $k T$, in energy units.
   // rho: material density, mass per volume
   [[nodiscard]] double computeKappa(const double epsilon, const double T, const double rho) const;

   // Computes the frequency dependent opacity with units of inverse length in Eq. 24
   // Same inputs as on computeKappa
   [[nodiscard]] double computeSigma(const double epsilon, const double T, const double rho) const
   {
      return rho * computeKappa(epsilon, T, rho);
   }

   // Compute locations to break numerical integrations into subregions near features
   [[nodiscard]] std::vector<double> computeBreaks() const;

  private:
   // The lowest photon energy used to compute an opacity,
   // mimicking a plasma frequency cut-off, $\epsilon_\text{min}$
   double epsilonMin;
   // The photon energy of the shell edge feature, $\epsilon_\text{edge}$
   double epsilonEdge;
   // A scaling factor to fit the opacity, $C_0$
   double C0;
   // The strength of the edge in the opacity, $C_1$
   double C1;
   // The strength of the lines in the opacity, $C_2$
   double C2;
   // The width of the line-like features, in photon energy, $\delta_w$
   double lineWidth;
   // The distance between the line-like features, in photon energy, $\delta_s$
   double lineSep;
   // The number of line-like features, $N_l$.  Can be zero.
   int numLines;
};

} // namespace AnalyticMGOpac
#endif // ANALYTICEDGEOPACITY_HH__
