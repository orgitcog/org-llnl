// Copyright 2024 Lawrence Livermore National Security, LLC.
// See the top-level LICENCE file for details.
//
// SPDX-License-Identifier: MIT

#include "AnalyticEdgeOpacity.hh"
#include <algorithm>
#include <cmath>

namespace AnalyticMGOpac
{

//-----------------------------------------------------------------------------

// Compute locations to break numerical integrations into subregions near features
std::vector<double> AnalyticEdgeOpacity::computeBreaks() const
{
   // These parameters specify how detailed to integrate the line-like features.
   // These were chosen to get accurate results, not speed.
   constexpr int numSigmas = 5;
   constexpr int breaksPerSigma = 3;

   std::vector<double> breakPoints;
   breakPoints.reserve(3 + 2 * numSigmas * breaksPerSigma * numLines);

   // Add the discontinuities in the opacity formula
   breakPoints.push_back(epsilonMin);
   breakPoints.push_back(epsilonEdge);

   // Capture the general shape of each Gaussian line
   for (int l = 0; l < numLines; ++l)
   {
      const double lineCenter = epsilonEdge - (l + 1) * lineSep;
      for (int s = -numSigmas * breaksPerSigma; s <= numSigmas * breaksPerSigma; ++s)
      {
         breakPoints.push_back(lineCenter + s * lineWidth / breaksPerSigma);
      }
   }

   std::sort(breakPoints.begin(), breakPoints.end());

   return breakPoints;
}

//-----------------------------------------------------------------------------

// Compute the opacity for a given frequency, temperature, and density
double AnalyticEdgeOpacity::computeKappa(const double epsilon, const double T, const double rho) const
{
   const double epsilonHat = std::max(epsilon, epsilonMin);

   const double term1 = C0 * rho / (std::sqrt(T) * epsilonHat * epsilonHat * epsilonHat);
   const double term2 = -std::expm1(-epsilonHat / T);
   const double term3 = 1.0 + (epsilonHat > epsilonEdge ? C1 : 0.0);

   double kappa = term1 * term2 * term3;

   for (int l = 0; l < numLines; ++l)
   {
      const double lineCenter = epsilonEdge - (l + 1) * lineSep;
      const double normPos = (epsilonHat - lineCenter) / lineWidth;
      kappa += term1 * term2 * C2 * std::exp(-0.5 * normPos * normPos) / (numLines - l);
   }

   return kappa;
}

//-----------------------------------------------------------------------------

} // namespace AnalyticMGOpac
