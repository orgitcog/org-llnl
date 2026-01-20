// Copyright 2024 Lawrence Livermore National Security, LLC.
// See the top-level LICENCE file for details.
//
// SPDX-License-Identifier: MIT

#include "AnalyticEdgeOpacity.hh"
#include "MultiGroupIntegrator.hh"

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>
#define FMT_HEADER_ONLY 1
#include "fmt/format.h"

#if defined(__linux__)
#include <fenv.h>
#endif

void printRange(std::FILE *file, const std::vector<double> &xs, std::string name)
{
   fmt::print(file, "{} = [", name);
   for (auto x : xs)
   {
      fmt::print(file, "{},\n", x);
   }
   fmt::print(file, "]\n");
}

template <typename Func> void printRange(std::FILE *file, std::vector<double> &xs, std::string name, Func f)
{
   fmt::print(file, "{} = [", name);
   for (auto x : xs)
   {
      fmt::print(file, "{},\n", f(x));
   }
   fmt::print(file, "]\n");
}

int main()
{
#if defined(__linux__)
   // This is in here for supporting Linux's floating point exceptions.
   feenableexcept(FE_DIVBYZERO);
   feenableexcept(FE_INVALID);
   feenableexcept(FE_OVERFLOW);
#endif

   //-------------------------------------------------------------------
   // Open an output file for testing.
   std::FILE *planckFile = std::fopen("data.py", "w");

   //-------------------------------------------------------------------
   // Now compute some group averages for output

   std::vector groupBounds{1.000000000000000e-04,
                           3.000000000000001e-03,
                           1.095445115010333e-02,
                           4.000000000000001e-02,
                           5.000000000000000e-02,
                           7.825422900366437e-02,
                           1.224744871391589e-01,
                           1.916829312738817e-01,
                           2.999999999999999e-01,
                           6.708203932499368e-01,
                           1.500000000000000e+00,
                           3.240370349203930e+00,
                           7.000000000000000e+00,
                           1.114619555925213e+01,
                           1.774823934929885e+01,
                           2.826076380281411e+01,
                           4.500000000000000e+01};

   std::vector temps{1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0};
   std::vector materials{std::make_tuple(/*name =*/"iron",
                                         /*epsilonMin =*/0.05,
                                         /*epsilonEdge =*/7.0,
                                         /*C0 =*/20.1,
                                         /*C1 =*/1.2e3,
                                         /*C2 =*/1.2e3,
                                         /*numLines =*/5,
                                         /*lineWidth =*/0.01,
                                         /*lineSep =*/
                                         0.2
                                         //,std::vector{6.0,8.0}
                                         ,
                                         std::vector{6.0}),
                         std::make_tuple(/*name =*/"carbon",
                                         /*epsilonMin =*/0.04,
                                         /*epsilonEdge =*/1.5,
                                         /*C0 =*/0.77,
                                         /*C1 =*/1.2e3,
                                         /*C2 =*/30.0,
                                         /*numLines =*/1,
                                         /*lineWidth =*/0.01,
                                         /*lineSep =*/1.2,
                                         std::vector{2.0}),
                         std::make_tuple(/*name =*/"foam",
                                         /*epsilonMin =*/0.04,
                                         /*epsilonEdge =*/0.3,
                                         /*C0 =*/2.00,
                                         /*C1 =*/4.0e2,
                                         /*C2 =*/0.0,
                                         /*numLines =*/0,
                                         /*lineWidth =*/0.0,
                                         /*lineSep =*/0.0,
                                         std::vector{0.2})};

   fmt::print(planckFile, "detailedKeys=[]\n");
   fmt::print(planckFile, "epsilon={{}}\n");
   fmt::print(planckFile, "b={{}}\n");
   fmt::print(planckFile, "r={{}}\n");
   fmt::print(planckFile, "kappa={{}}\n");
   fmt::print(planckFile, "kappaP_g={{}}\n");
   fmt::print(planckFile, "kappaR_g={{}}\n");
   fmt::print(planckFile, "b_g={{}}\n");
   fmt::print(planckFile, "dbdT_g={{}}\n");

   fmt::print(planckFile, "greyKeys=[]\n");
   fmt::print(planckFile, "kappaP={{}}\n");
   fmt::print(planckFile, "kappaR={{}}\n");

   printRange(planckFile, groupBounds, "groupBounds");
   std::vector<double> greyTemps;
   const double dx = 1.15;
   for (double x = 0.001; x <= 10.0; x *= dx)
   {
      greyTemps.push_back(x);
   }
   printRange(planckFile, greyTemps, "greyTemps");

   for (const auto &[name, epsilonMin, epsilonEdge, C0, C1, C2, numLines, lineWidth, lineSep, densities] : materials)
   {
      const AnalyticMGOpac::AnalyticEdgeOpacity opac(epsilonMin, epsilonEdge, C0, C1, C2, lineWidth, lineSep, numLines);
      AnalyticMGOpac::MultiGroupIntegrator opacInt(opac, groupBounds);

      for (auto rho : densities)
      {
         for (auto T : temps)
         {
            std::vector<double> epsilons = opacInt.computeAllSubRanges(T);
            // Add a bunch of points to stress exponentials
            epsilons.push_back(1.0e-30 * T);
            epsilons.push_back(1.0e-15 * T);
            epsilons.push_back(1.0e-8 * T);
            epsilons.push_back(1.0e-6 * T);
            epsilons.push_back(1.0e-5 * T);
            epsilons.push_back(1.0e-4 * T);
            epsilons.push_back(1.0e4 * T);
            epsilons.push_back(1.0e6 * T);

            // Add a bunch more for plotting at our temperature
            const double dx = 0.01 * T;
            for (double x = dx; x <= AnalyticMGOpac::cumulative_planck_max() * T; x += dx)
            {
               epsilons.push_back(x);
            }
            // Add a bunch more for plotting at the max temperature
            if (T < 1.0)
            {
               const double dx = 0.01 * 1.0;
               for (double x = dx; x <= AnalyticMGOpac::cumulative_planck_max() * 1.0; x += dx)
               {
                  epsilons.push_back(x);
               }
            }
            std::sort(epsilons.begin(), epsilons.end());
            auto last = std::unique(epsilons.begin(), epsilons.end());
            epsilons.erase(last, epsilons.end());

            std::string key = fmt::format("('{}',{},{})", name, rho, T);
            fmt::print(planckFile, "detailedKeys.append({})\n", key);
            printRange(planckFile, epsilons, fmt::format("epsilon[{}]", key));
            printRange(planckFile, epsilons, fmt::format("b[{}]", key), [=](double x) {
               return AnalyticMGOpac::safePlanck(x / T);
            });
            printRange(planckFile, epsilons, fmt::format("r[{}]", key), [=](double x) {
               return AnalyticMGOpac::safeRoss(x / T);
            });

            std::vector<double> planckAverage;
            std::vector<double> rosselandAverage;
            std::vector<double> b_g;
            std::vector<double> dbdT_g;
            double planckMean{0.0};
            double rosselandMean{0.0};

            printRange(planckFile, epsilons, fmt::format("kappa[{}]", key), [=](double x) {
               return opac.computeKappa(x, T, rho);
            });
            opacInt
               .computeGroupAverages(T, rho, planckAverage, rosselandAverage, b_g, dbdT_g, planckMean, rosselandMean);

            printRange(planckFile, planckAverage, fmt::format("kappaP_g[{}]", key));
            printRange(planckFile, rosselandAverage, fmt::format("kappaR_g[{}]", key));
            printRange(planckFile, b_g, fmt::format("b_g[{}]", key));
            printRange(planckFile, dbdT_g, fmt::format("dbdT_g[{}]", key));
         }

         std::vector<double> kappaP;
         std::vector<double> kappaR;
         std::string greyKey = fmt::format("('{}',{})", name, rho);
         fmt::print(planckFile, "greyKeys.append({})\n", greyKey);
         for (auto T : greyTemps)
         {
            std::vector<double> epsilons = opacInt.computeAllSubRanges(T);

            std::vector<double> planckAverage;
            std::vector<double> rosselandAverage;
            std::vector<double> b_g;
            std::vector<double> dbdT_g;
            double planckMean{0.0};
            double rosselandMean{0.0};

            opacInt
               .computeGroupAverages(T, rho, planckAverage, rosselandAverage, b_g, dbdT_g, planckMean, rosselandMean);

            kappaP.push_back(planckMean);
            kappaR.push_back(rosselandMean);
         }
         printRange(planckFile, kappaP, fmt::format("kappaP[{}]", greyKey));
         printRange(planckFile, kappaR, fmt::format("kappaR[{}]", greyKey));
      }
   }

   std::fclose(planckFile);

   return 0;
}
