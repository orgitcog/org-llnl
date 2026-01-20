#include "Tests.hpp"
#include "MultiMatTest.hpp"
#include "genmalloc.h"
#include "timer.h"
#include <stdio.h>

void single_material(const int ncells, const bool memory_verbose,
                     const int itermax, const double nmatconst) {

  printf("=======================================\n");
  printf("Starting Single Material Data Structure\n");
  printf("=======================================\n\n");

  double *Density = (double *)genvector("Density", ncells, sizeof(double));
  double *Temperature =
      (double *)genvector("Temperature", ncells, sizeof(double));
  double *Pressure = (double *)genvector("Pressure", ncells, sizeof(double));
  double *Vol = (double *)genvector("Volume", ncells, sizeof(double));
  double *ReduceArray =
      (double *)genvector("reduce_array", ncells, sizeof(double));

  double VolTotal = 0.0;
  for (int ic = 0; ic < ncells; ic++) {
    Density[ic] = 2.0;
    Temperature[ic] = 0.5;
    Vol[ic] = 1.0;
    VolTotal += Vol[ic];
  }

  if (memory_verbose) {
    genmalloc_MB_memory_report();
  }
  genmalloc_MB_memory_total();
  printf("\n");

  struct timeval tstart_cpu;

  //  Average density with cell densities (pure cells)
  double time_sum = 0.0;
  double density_ave = 0.0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    double density_ave = 0.0;
    for (int ic = 0; ic < ncells; ic++) {
      density_ave += Density[ic] * Vol[ic];
    }

    density_ave /= VolTotal;

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  float act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of pure cells    %lf, compute time is %lf msecs\n",
         density_ave, act_perf);

  int64_t memops = 2 * ncells;
  int64_t flops = 2 * ncells;
  float penalty_msecs = 0.0;
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  //    Calculate pressure using ideal gas law
  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      Pressure[ic] = (nmatconst * Density[ic] * Temperature[ic]) / Vol[ic];
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf(
      "Pressure Calculation for cell,             compute time is %lf msecs\n",
      act_perf);

  memops = 4 * ncells;
  flops = 3 * ncells;
  penalty_msecs = 0.0;
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  genvectorfree(Vol);
  genvectorfree(Density);
  genvectorfree(Temperature);
  genvectorfree(Pressure);
}

void cell_dominant_full_matrix(const int ncells, const bool memory_verbose,
                               const int itermax, const double nmatconst,
                               const int method, const int nmats,
                               const float L_f, const int nnbrs_ave,
                               const double *nmatconsts, const double *cen_x,
                               const double *cen_y, const int *nnbrs,
                               int **nbrs) {

  //    Starting Cell-Dominant Full Matrix Data Structure
  printf("\n");
  printf("=================================================\n");
  printf("Starting Cell-Dominant Full Matrix Data Structure\n");
  printf("=================================================\n\n");

  double *Vol, *Density, *Temperature, *Pressure;
  double **Densityfrac, **Temperaturefrac, **Pressurefrac, **Volfrac;

  float filled_percentage;
  setup_cell_dominant_data_structure(
      method, Vol, Density, Temperature, Pressure, Volfrac, Densityfrac,
      Temperaturefrac, Pressurefrac, filled_percentage);

  float filled_fraction = filled_percentage / 100.0;

  if (memory_verbose) {
    genmalloc_MB_memory_report();
  }
  genmalloc_MB_memory_total();
  printf("\n");

  //    Average density with fractional densities
  double *Density_average =
      (double *)genvector("Density_average", ncells, sizeof(double));
  double *ReduceArray =
      (double *)genvector("ReduceArray", ncells, sizeof(double));

  struct timeval tstart_cpu;
  double time_sum = 0.0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      double density_ave = 0.0;
      for (int m = 0; m < nmats; m++) {
        density_ave += Densityfrac[ic][m] * Volfrac[ic][m];
      }
      Density_average[ic] = density_ave / Vol[ic];
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  float act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of mixed material cells    compute time is %lf "
         "msecs\n",
         act_perf);

  genvectorfree(Density_average);

  int64_t memops = 2 * ncells * nmats; // line 4 loads
  memops += 2 * ncells;                // line 6 stores
  int64_t flops = 2 * ncells * nmats;  // line 4 flops
  float penalty_msecs = 0.0;
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  //    Average density with fractional densities with if test
  Density_average =
      (double *)genvector("Density_average", ncells, sizeof(double));

  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      double density_ave = 0.0;
      for (int m = 0; m < nmats; m++) {
        if (Volfrac[ic][m] > 0.0) {
          density_ave += Densityfrac[ic][m] * Volfrac[ic][m];
        }
      }
      Density_average[ic] = density_ave / Vol[ic];
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of frac with if            compute time is %lf "
         "msecs\n",
         act_perf);

  genvectorfree(Density_average);

  float cache_miss_freq = method ? 0.7 : 1.0;
  memops = ncells * nmats; // line 4 loads
  memops +=
      (int64_t)(filled_fraction * (float)(ncells * nmats)); // line 5 loads
  memops += 2 * ncells; // line 8 stores and loads
  flops =
      (int64_t)(filled_fraction * (float)(2 * ncells * nmats)); // line 5 flops
  flops += ncells;                                              // line 8 flops
  float branch_wait = 1.0 / CLOCK_RATE *
                      16; // Estimate a 16 cycle wait for branch misprediction
  // for a 2.7 GHz processor
  float cache_wait =
      1.0 / CLOCK_RATE * 7 *
      16; // Estimate a 7*16 or 112 cycle wait for missing prefetch
  // for a 2.7 GHz processor
  penalty_msecs = 1000.0 * cache_miss_freq * (branch_wait + cache_wait) *
                  (filled_fraction * (float)(ncells * nmats)); // line 4 if
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  //   Calculate pressure using ideal gas law
  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      for (int m = 0; m < nmats; m++) {
        if (Volfrac[ic][m] > 0.) {
          Pressurefrac[ic][m] =
              (nmatconsts[m] * Densityfrac[ic][m] * Temperaturefrac[ic][m]) /
              (Volfrac[ic][m]);
        } else {
          Pressurefrac[ic][m] = 0.0;
        }
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Pressure Calculation of mixed material cells with if compute time "
         "is %lf msecs\n",
         act_perf);

  float sparsity_fraction = 1.0 - filled_fraction;
  memops = ncells * nmats; // line 3 loads
  memops +=
      (int64_t)(filled_fraction * (float)(ncells * nmats)); // line 5 stores
  memops +=
      (int64_t)(filled_fraction * (float)(3 * ncells * nmats)); // line 6 loads
  memops +=
      (int64_t)(sparsity_fraction * (float)(ncells * nmats)); // line 8 stores
  flops =
      (int64_t)(filled_fraction * (float)(3 * ncells * nmats)); // line 6 flops
  branch_wait = 1.0 / CLOCK_RATE *
                16; // Estimate a 16 cycle wait for branch misprediction
  // for a 2.7 GHz processor
  cache_wait = 1.0 / CLOCK_RATE * 7 *
               16; // Estimate a 7*16 or 112 cycle wait for missing prefetch
  // for a 2.7 GHz processor
  penalty_msecs = 1000.0 * cache_miss_freq * (branch_wait + cache_wait) *
                  (filled_fraction * (float)(ncells * nmats)); // line 3 if
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  //    Average material density over neighborhood of each cell
  double **MatDensity_average =
      (double **)genmatrix("MatDensity_average", ncells, nmats, sizeof(double));

  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {

    for (int ic = 0; ic < ncells; ic++) {
      double xc[2];
      xc[0] = cen_x[ic];
      xc[1] = cen_y[ic];
      int nn = nnbrs[ic];
      int cnbrs[8];
      double dsqr[8];

      for (int n = 0; n < nn; n++)
        cnbrs[n] = nbrs[ic][n];

      for (int n = 0; n < nn; n++) {
        dsqr[n] = 0.0;
        // TODO: Fairly sure this was meant to iterate over both dimensions??
        double ddist = (xc[0] - cen_x[cnbrs[n]]);
        dsqr[n] += ddist * ddist;
      }

      for (int m = 0; m < nmats; m++) {
        if (Volfrac[ic][m] > 0.0) {
          int nnm = 0; // number of nbrs with this material
          for (int n = 0; n < nn; n++) {
            int jc = cnbrs[n];
            if (Volfrac[jc][m] > 0.0) {
              MatDensity_average[ic][m] += Densityfrac[ic][m] / dsqr[n];
              nnm++;
            }
          }
          MatDensity_average[ic][m] /= nnm;
        } else {
          MatDensity_average[ic][m] = 0.0;
        }
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Average Material Density            compute time is %lf msecs\n",
         act_perf);

  genmatrixfree((void **)MatDensity_average);

  memops = (2 + 2 * nmats + (0.5 + 16) * nnbrs_ave) * ncells;
  //        // Formula differs from paper because it is 2D here
  memops +=
      (int64_t)(filled_fraction * 8 * (1 + L_f) * ncells * nmats * nnbrs_ave);
  flops = 6 * ncells * nnbrs_ave;
  flops += (int64_t)(filled_fraction * 3 * ncells * nmats * nnbrs_ave * L_f);
  branch_wait = 1.0 / CLOCK_RATE *
                16; // Estimate a 16 cycle wait for branch misprediction
  // for a 2.7 GHz processor
  cache_wait = 1.0 / CLOCK_RATE * 7 *
               16; // Estimate a 7*16 or 112 cycle wait for missing prefetch
  // for a 2.7 GHz processor
  penalty_msecs = 1000.0 * cache_miss_freq * (branch_wait + cache_wait) *
                  (filled_fraction * (float)(ncells * nmats));
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  genvectorfree((void *)Vol);
  genvectorfree((void *)Density);
  genvectorfree((void *)Temperature);
  genvectorfree((void *)Pressure);
  genmatrixfree((void **)Volfrac);
  genmatrixfree((void **)Densityfrac);
  genmatrixfree((void **)Temperaturefrac);
  genmatrixfree((void **)Pressurefrac);
}

void material_dominant_matrix(const int ncells, const bool memory_verbose,
                              const int itermax, const double nmatconst,
                              const int method, const int nmats,
                              const float L_f, const int nnbrs_ave,
                              const double *nmatconsts, const double *cen_x,
                              const double *cen_y, const int *nnbrs,
                              int **nbrs) {

  //    Starting Material-Dominant Full Matrix Data Structure
  printf("\n");
  printf("===================================================\n");
  printf("Starting Material-Dominant Full Matrix Data Structure\n");
  printf("===================================================\n");

  double *Vol, *Density, *Temperature, *Pressure;
  double **Volfrac, **Densityfrac, **Temperaturefrac, **Pressurefrac;

  float filled_percentage;
  setup_material_dominant_data_structure(
      method, Vol, Density, Temperature, Pressure, Volfrac, Densityfrac,
      Temperaturefrac, Pressurefrac, filled_percentage);

  float filled_fraction = filled_percentage / 100.0;

  if (memory_verbose) {
    genmalloc_MB_memory_report();
  }
  genmalloc_MB_memory_total();
  printf("\n");

  //    Average density with fractional densities
  double *Density_average =
      (double *)genvector("Density_average", ncells, sizeof(double));

  double time_sum = 0;
  struct timeval tstart_cpu;

  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      Density_average[ic] = 0.0;
    }
    for (int m = 0; m < nmats; m++) {
      for (int ic = 0; ic < ncells; ic++) {
        Density_average[ic] += Densityfrac[m][ic] * Volfrac[m][ic];
      }
    }
    for (int ic = 0; ic < ncells; ic++) {
      Density_average[ic] /= Vol[ic];
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  double act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of mixed material cells    compute time is %lf "
         "msecs\n",
         act_perf);

  genvectorfree(Density_average);

  int64_t memops = ncells;            // line 3 loads
  memops += ncells * nmats;           // line 6 stores
  memops += 2 * ncells * nmats;       // line 7 loads
  int64_t flops = 2 * ncells * nmats; // line 7 flops
  memops += 2 * ncells;               // line 11 loads/stores
  flops += ncells;                    // line 11 flops
  float penalty_msecs = 0.0;
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  //    Average density with fractional densities with if test
  Density_average =
      (double *)genvector("Density_average", ncells, sizeof(double));

  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    // NOTE: Fairly sure this is mean to be Density_average, it was previously
    // Density.
    for (int ic = 0; ic < ncells; ic++) {
      Density[ic] = 0.0;
    }
    for (int m = 0; m < nmats; m++) {
      for (int ic = 0; ic < ncells; ic++) {
        if (Volfrac[m][ic] > 0.0) {
          Density_average[ic] += Densityfrac[m][ic] * Volfrac[m][ic];
        }
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of mixed material cells with if compute time is "
         "%lf msecs\n",
         act_perf);

  genvectorfree(Density_average);

  float cache_miss_freq = method ? 0.2 : 1.0;
  memops = ncells;          // line 2 loads
  memops += ncells * nmats; // line 6 loads
  memops +=
      (int64_t)(filled_fraction * (float)(ncells * nmats)); // line 7 stores
  memops +=
      (int64_t)(filled_fraction * (float)(ncells * nmats)); // line 8 loads
  flops =
      (int64_t)(filled_fraction * (float)(2 * ncells * nmats)); // line 8 flops
  memops += 2 * ncells; // line 11 stores and loads
  flops += ncells;      // line 11 flops
  float branch_wait = 1.0 / CLOCK_RATE *
                      16; // Estimate a 16 cycle wait for branch misprediction
  // for a 2.7 GHz processor
  float cache_wait =
      1.0 / CLOCK_RATE * 7 *
      16; // Estimate a 7*16 or 112 cycle wait for missing prefetch
  // for a 2.7 GHz processor
  penalty_msecs = 1000.0 * cache_miss_freq * (branch_wait + cache_wait) *
                  (filled_fraction * (float)(ncells * nmats)); // line 6 if
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  //    Calculate pressure using ideal gas law
  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int m = 0; m < nmats; m++) {
      for (int ic = 0; ic < ncells; ic++) {
        if (Volfrac[m][ic] > 0.0) {
          Pressurefrac[m][ic] =
              (nmatconsts[m] * Densityfrac[m][ic] * Temperaturefrac[m][ic]) /
              Volfrac[m][ic];
        } else {
          Pressurefrac[m][ic] = 0.0;
        }
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Pressure Calculation of frac with if       compute time is %lf "
         "msecs\n",
         act_perf);

  float sparsity_fraction = 1.0 - filled_fraction;
  memops = nmats;           // line 2 loads
  memops += ncells * nmats; // line 4 loads
  memops +=
      (int64_t)(filled_fraction * (float)(ncells * nmats)); // line 5 stores
  memops +=
      (int64_t)(filled_fraction * (float)(2 * ncells * nmats)); // line 6 loads
  flops =
      (int64_t)(filled_fraction * (float)(3 * ncells * nmats)); // line 6 flops
  memops +=
      (int64_t)(sparsity_fraction * (float)(ncells * nmats)); // line 8 stores
  branch_wait = 1.0 / CLOCK_RATE *
                16; // Estimate a 16 cycle wait for branch misprediction
  // for a 2.7 GHz processor
  cache_wait = 1.0 / CLOCK_RATE * 7 *
               16; // Estimate a 7*16 or 112 cycle wait for missing prefetch
  // for a 2.7 GHz processor
  penalty_msecs = 1000.0 * cache_miss_freq * (branch_wait + cache_wait) *
                  (filled_fraction * (float)(ncells * nmats)); // line 6 if
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  //    Average material density over neighborhood of each cell
  double **MatDensity_average =
      (double **)genmatrix("MatDensity_average", nmats, ncells, sizeof(double));

  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int m = 0; m < nmats; m++) {
      for (int ic = 0; ic < ncells; ic++) {
        if (Volfrac[m][ic] > 0.0) {
          double xc[2];
          xc[0] = cen_x[ic];
          xc[1] = cen_y[ic];
          int nn = nnbrs[ic];
          int cnbrs[8];
          double dsqr[8];
          for (int n = 0; n < nn; n++)
            cnbrs[n] = nbrs[ic][n];
          for (int n = 0; n < nn; n++) {
            dsqr[n] = 0.0;
            // TODO: Fairly sure this was meant to iterate over both
            // dimensions??
            double ddist = (xc[0] - cen_x[cnbrs[n]]);
            dsqr[n] += ddist * ddist;
          }

          int nnm = 0; // number of nbrs with this material
          for (int n = 0; n < nn; n++) {
            int jc = cnbrs[n];
            if (Volfrac[m][jc] > 0.0) {
              MatDensity_average[m][ic] += Densityfrac[m][ic] / dsqr[n];
              nnm++;
            }
          }
          MatDensity_average[m][ic] /= nnm;
        } else {
          MatDensity_average[m][ic] = 0.0;
        }
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Average Material Density            compute time is %lf msecs\n",
         act_perf);

  genmatrixfree((void **)MatDensity_average);

  memops = 2 * ncells * nmats;
  //        // Formula differs from paper because it is 2D here
  memops += (int64_t)(2 * filled_fraction * ncells * nmats);
  memops += (int64_t)(8.5 * filled_fraction * ncells * nmats * nnbrs_ave);
  memops += (int64_t)(24 * filled_fraction * L_f * ncells * nmats * nnbrs_ave);
  flops = (int64_t)(filled_fraction * ncells * nmats);
  flops += (int64_t)(9 * filled_fraction * ncells * nmats * nnbrs_ave * L_f);
  branch_wait = 1.0 / CLOCK_RATE *
                16; // Estimate a 16 cycle wait for branch misprediction
  // for a 2.7 GHz processor
  cache_wait = 1.0 / CLOCK_RATE * 7 *
               16; // Estimate a 7*16 or 112 cycle wait for missing prefetch
  // for a 2.7 GHz processor
  penalty_msecs = 1000.0 * cache_miss_freq * (branch_wait + cache_wait) *
                  (filled_fraction * ncells * nmats);
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  genvectorfree((void *)Vol);
  genvectorfree((void *)Density);
  genvectorfree((void *)Temperature);
  genvectorfree((void *)Pressure);
  genmatrixfree((void **)Volfrac);
  genmatrixfree((void **)Densityfrac);
  genmatrixfree((void **)Temperaturefrac);
  genmatrixfree((void **)Pressurefrac);
}

void cell_dominant_compact(const int ncells, const bool memory_verbose,
                           const int itermax, const double nmatconst,
                           const int method, const int nmats, const float L_f,
                           const int nnbrs_ave, const double *nmatconsts,
                           const double *cen_x, const double *cen_y,
                           const int *nnbrs, int **nbrs) {
  //    Starting Cell-Dominant Compact Data Structure
  printf("\n");
  printf("===================================================\n");
  printf("Starting Cell-Dominant Compact Data Structure\n");
  printf("===================================================\n");

  int *imaterial, *nmaterials, *imaterialfrac, *nextfrac, *frac2cell;
  double *Vol, *Density, *Temperature, *Pressure, *Volfrac, *Densityfrac,
      *Temperaturefrac, *Pressurefrac;

  float filled_percentage;
  setup_cell_dominant_compact_data_structure(
      method, imaterial, nmaterials, Vol, Density, Temperature, Pressure,
      imaterialfrac, nextfrac, frac2cell, Volfrac, Densityfrac, Temperaturefrac,
      Pressurefrac, filled_percentage);

  float filled_fraction = filled_percentage / 100.0;

  int nmixlength = 0;
  int pure_cell_count = 0;
  int mixed_cell_count = 0;

  for (int ic = 0; ic < ncells; ic++) {
    int ix = imaterial[ic];
    if (ix <= 0) {
      for (ix = -ix; ix >= 0; ix = nextfrac[ix])
        nmixlength++;
      mixed_cell_count++;
    } else {
      pure_cell_count++;
    }
  }
  float mixed_cell_fraction = mixed_cell_count / ncells;
  float pure_cell_fraction = pure_cell_count / ncells;
  int nmats_ave = (pure_cell_count + nmixlength) / ncells;

  if (memory_verbose) {
    genmalloc_MB_memory_report();
  }
  genmalloc_MB_memory_total();
  printf("\n");

  //    Average density with fractional densities
  double time_sum = 0;
  struct timeval tstart_cpu;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      double density_ave = 0.0;
      int ix = imaterial[ic];
      if (ix <= 0) { // material numbers for clean cells start at 1
        for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
          density_ave += Densityfrac[ix] * Volfrac[ix];
        }
        Density[ic] = density_ave / Vol[ic];
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  float act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of mixed material cells    compute time is %lf "
         "msecs\n",
         act_perf);

  float cache_miss_freq = method ? 0.1 : 1.0;
  int64_t memops4byte = ncells;                // line 3 loads
  memops4byte += nmixlength;                   // line 5 loads
  int64_t memops8byte = 2 * nmixlength;        // line 6 loads
  int64_t flops = 2 * nmixlength;              // line 6 flops
  memops8byte += mixed_cell_fraction * ncells; // line 8 stores
  memops8byte += mixed_cell_fraction * ncells; // line 8 loads
  flops += mixed_cell_fraction * ncells;       // line 8 flops
  float loop_overhead =
      1.0 / CLOCK_RATE * 20; // Estimate a 20 cycle loop exit overhead
  float penalty_msecs = 1000.0 * cache_miss_freq * loop_overhead *
                        mixed_cell_fraction * (float)ncells; // line 5 for
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);

  //    Average density with fractional densities using nmats
  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      double density_ave = 0.0;
      int mstart = imaterial[ic];
      if (mstart <= 0) { // material numbers for clean cells start at 1
        mstart = -mstart;
        for (int ix = 0; ix < nmaterials[ic]; ix++) {
          density_ave += Densityfrac[mstart + ix] * Volfrac[mstart + ix];
        }
        Density[ic] = density_ave / Vol[ic];
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of mixed material cells with nmats  compute time "
         "is %lf msecs\n",
         act_perf);

  memops4byte = ncells;                        // line 3 loads
  memops4byte += mixed_cell_fraction * ncells; // line 5 loads
  memops8byte = 2 * nmixlength;                // line 6 loads
  flops = 2 * nmixlength;                      // line 6 flops
  memops8byte += mixed_cell_fraction * ncells; // line 8 stores
  memops8byte += mixed_cell_fraction * ncells; // line 8 loads
  flops += mixed_cell_fraction * ncells;       // line 8 flops
  loop_overhead =
      1.0 / CLOCK_RATE * 20; // Estimate a 20 cycle loop exit overhead
  penalty_msecs = 1000.0 * cache_miss_freq * loop_overhead *
                  mixed_cell_fraction * (float)ncells; // line 5 for
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);

  //    Average density with fractional densities
  double *Density_average =
      (double *)genvector("Density_average", ncells, sizeof(double));

  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      double density_ave = 0.0;
      int ix = imaterial[ic];
      if (ix <= 0) { // material numbers for clean cells start at 1
        for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
          density_ave += Densityfrac[ix] * Volfrac[ix];
        }
        Density_average[ic] = density_ave / Vol[ic];
      } else {
        Density_average[ic] = Density[ic];
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of mixed material cells    compute time is %lf "
         "msecs\n",
         act_perf);

  genvectorfree(Density_average);

  memops4byte = ncells;                        // line 3 loads
  memops4byte += nmixlength;                   // line 5 loads
  memops8byte = 2 * nmixlength;                // line 6 loads
  flops = 2 * nmixlength;                      // line 6 flops
  memops8byte += mixed_cell_fraction * ncells; // line 8 stores
  memops8byte += mixed_cell_fraction * ncells; // line 8 loads
  flops += mixed_cell_fraction * ncells;       // line 8 flops
  memops8byte += pure_cell_fraction * ncells;  // line 10 stores
  memops8byte += pure_cell_fraction * ncells;  // line 10 loads
  loop_overhead =
      1.0 / CLOCK_RATE * 20; // Estimate a 20 cycle loop exit overhead
  penalty_msecs = 1000.0 * cache_miss_freq * loop_overhead *
                  mixed_cell_fraction * (float)ncells; // line 5 for
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);

  //    Average density with fractional densities with pure calculation filler
  Density_average =
      (double *)genvector("Density_average", ncells, sizeof(double));

  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      int ix = imaterial[ic];
      double density_ave = 0.0;
      if (ix <= 0) { // material numbers for clean cells start at 1
        for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
          density_ave += Densityfrac[ix] * Volfrac[ix];
        }
      } else { // Pure cell
        density_ave = Density[ic] * Vol[ic];
      }
      Density_average[ic] = density_ave / Vol[ic];
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of mixed materials cells pure filler  compute time "
         "is %lf msecs\n",
         act_perf);

  genvectorfree(Density_average);

  memops4byte = ncells;                                    // line 3 loads
  memops4byte += nmixlength;                               // line 5 loads
  memops8byte = 2 * nmixlength;                            // line 6 loads
  flops = 2 * nmixlength;                                  // line 6 flops
  memops8byte += (int64_t)mixed_cell_fraction * ncells;    // line 8 stores
  memops8byte += (int64_t)mixed_cell_fraction * ncells;    // line 8 loads
  flops += (int64_t)mixed_cell_fraction * ncells;          // line 8 flops
  memops8byte += (int64_t)pure_cell_fraction * ncells;     // line 10 stores
  memops8byte += (int64_t)2 * pure_cell_fraction * ncells; // line 10 loads
  flops += (int64_t)pure_cell_fraction * ncells;           // line 8 flops
  loop_overhead =
      1.0 / CLOCK_RATE * 20; // Estimate a 20 cycle loop exit overhead
  penalty_msecs = 1000.0 * cache_miss_freq * loop_overhead *
                  mixed_cell_fraction * (float)ncells; // line 5 for
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);

  //   Calculate pressure using ideal gas law
  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      int ix = imaterial[ic];
      if (ix <= 0) { // material numbers for clean cells start at 1
        for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
          int m = imaterialfrac[ix];
          Pressurefrac[ix] =
              (nmatconsts[m] * Densityfrac[ix] * Temperaturefrac[ix]) /
              Volfrac[ix];
        }
      } else {
        Pressure[ic] = nmatconsts[ix] * Density[ic] * Temperature[ic] / Vol[ic];
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Pressure Calculation of mixed material cells compute time is %lf "
         "msecs\n",
         act_perf);

  memops4byte = ncells;                                    // line 2 loads
  memops4byte += nmixlength;                               // line 4 loads
  memops4byte += nmixlength;                               // line 5 loads
  memops8byte = 3 * nmixlength;                            // line 6 loads
  memops8byte += nmixlength;                               // line 6 stores
  flops = 3 * nmixlength;                                  // line 6 flops
  memops8byte += (int64_t)pure_cell_fraction * ncells;     // line 9 stores
  memops8byte += (int64_t)4 * pure_cell_fraction * ncells; // line 9 loads
  flops += (int64_t)3 * pure_cell_fraction * ncells;       // line 9 flops
  loop_overhead =
      1.0 / CLOCK_RATE * 20; // Estimate a 20 cycle loop exit overhead
  penalty_msecs = 1000.0 * cache_miss_freq * loop_overhead *
                  mixed_cell_fraction * (float)ncells; // line 5 for
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);

  //    Average material density over neighborhood of each cell
  double **MatDensity_average =
      (double **)genmatrix("MatDensity_average", ncells, nmats, sizeof(double));

  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int ic = 0; ic < ncells; ic++) {
      for (int m = 0; m < nmats; m++)
        MatDensity_average[ic][m] = 0.0;

      double xc[2];
      xc[0] = cen_x[ic];
      xc[1] = cen_y[ic];
      int nn = nnbrs[ic];
      int cnbrs[8];
      double dsqr[8];
      for (int n = 0; n < nn; n++)
        cnbrs[n] = nbrs[ic][n];
      for (int n = 0; n < nn; n++) {
        dsqr[n] = 0.0;
        double ddist = (xc[0] - cen_x[cnbrs[n]]);
        dsqr[n] += ddist * ddist;
      }

      int ix = imaterial[ic];
      if (ix <= 0) {
        for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
          int m = imaterialfrac[ix];

          int nnm = 0; // number of nbrs with this material
          for (int n = 0; n < nn; n++) {
            int jc = cnbrs[n];

            int jx = imaterial[jc];
            if (jx <= 0) {
              for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
                if (imaterialfrac[jx] == m) {
                  MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
                  nnm++;
                  break;
                }
              }
            } else {
              if (imaterialfrac[jx] == m) {
                MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
                nnm++;
              }
            }
          }
          MatDensity_average[ic][m] /= nnm;
        }
      } else {
        int m = imaterialfrac[ix];

        int nnm = 0; // number of nbrs with this material
        for (int n = 0; n < nn; n++) {
          int jc = cnbrs[n];

          int jx = imaterial[jc];
          if (jx <= 0) {
            for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
              if (imaterialfrac[jx] == m) {
                MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
                nnm++;
                break;
              }
            }
          } else {
            if (imaterialfrac[jx] == m) {
              MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
              nnm++;
            }
          }
        }
        MatDensity_average[ic][m] /= nnm;
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Average Material Density            compute time is %lf msecs\n",
         act_perf);

  genmatrixfree((void **)MatDensity_average);

  filled_fraction = filled_percentage / 100.0;
  // Formula differs a bit from paper because it is 2D here
  int64_t memops = (int64_t)(2.5 * ncells * (1 + nnbrs_ave) + 0.5 * nmixlength);
  memops += (int64_t)(ncells * nmats * (1 + 1.5 * filled_fraction));
  memops += (int64_t)(4 * filled_fraction * ncells * nmats * nnbrs_ave);
  memops +=
      (int64_t)(8 * filled_fraction * ncells * nmats * nnbrs_ave * nmats_ave);
  memops += (int64_t)(8 * filled_fraction * ncells * nmats * nnbrs_ave * L_f);
  flops = 6 * ncells * nnbrs_ave;
  flops += (int64_t)(3 * filled_fraction * ncells * nmats * nnbrs_ave * L_f);
  flops += (int64_t)(filled_fraction * ncells * nmats);
  float branch_wait = 1.0 / CLOCK_RATE *
                      16; // Estimate a 16 cycle wait for branch misprediction
  // for a 2.7 GHz processor
  float cache_wait =
      1.0 / CLOCK_RATE * 7 *
      16; // Estimate a 7*16 or 112 cycle wait for missing prefetch
  // for a 2.7 GHz processor
  penalty_msecs = 1000.0 * cache_miss_freq * (branch_wait + cache_wait) *
                  mixed_cell_fraction * ncells;
  print_performance_estimates(act_perf, memops, 0, flops, penalty_msecs);

  genvectorfree(imaterial);
  genvectorfree(nmaterials);
  genvectorfree(Vol);
  genvectorfree(Density);
  genvectorfree(Temperature);
  genvectorfree(Pressure);
  genvectorfree(imaterialfrac);
  genvectorfree(nextfrac);
  genvectorfree(frac2cell);
  genvectorfree(Volfrac);
  genvectorfree(Densityfrac);
  genvectorfree(Temperaturefrac);
  genvectorfree(Pressurefrac);
}

void material_centric_compact(const int ncells, const bool memory_verbose,
                              const int itermax, const double nmatconst,
                              const int method, const int nmats,
                              const float L_f, const int nnbrs_ave,
                              const double *nmatconsts, const double *cen_x,
                              const double *cen_y, const int *nnbrs,
                              int **nbrs) {

  //    Starting Material-Centric Compact Data Structure
  printf("\n");
  printf("===================================================\n");
  printf("Starting Material-Centric Compact Data Structure\n");
  printf("===================================================\n");

  int *nmatscell, *matids, *ncellsmat;
  int **subset2mesh, **mesh2subset;
  double *Vol, *Density;
  double **Volfrac, **Densityfrac, **Temperaturefrac, **Pressurefrac;
  float filled_percentage;
  float cache_miss_freq = method ? 0.2 : 1.0;

  setup_mat_dominant_compact_data_structure(
      method, subset2mesh, mesh2subset, nmatscell, matids, ncellsmat, Vol,
      Density, Volfrac, Densityfrac, Temperaturefrac, Pressurefrac,
      filled_percentage);

  float filled_fraction = filled_percentage / 100.0;

  if (memory_verbose) {
    genmalloc_MB_memory_report();
  }
  genmalloc_MB_memory_total();
  printf("\n");

  //    Average density with fractional densities - MAT-DOMINANT LOOP
  double time_sum = 0;
  struct timeval tstart_cpu;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int C = 0; C < ncells; C++)
      Density[C] = 0.0;

    for (int m = 0; m < nmats; m++) {
      for (int c = 0; c < ncellsmat[m]; c++) { // Note that this is c not C
        int C = subset2mesh[m][c];
        Density[C] += Densityfrac[m][c] * Volfrac[m][c];
      }
    }

    for (int C = 0; C < ncells; C++)
      Density[C] /= Vol[C];

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  float act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of cells - Mat Dominant loop  -  compute time is "
         "%lf msecs\n",
         act_perf);

  float ninner = 0.0; // number of times inner loop executed
  for (int m = 0; m < nmats; m++)
    for (int c = 0; c < ncellsmat[m]; c++)
      ninner++;
  int64_t memops8byte = ncells; // Initialization of Density
  memops8byte += (int64_t)(8 * cache_miss_freq + (1 - cache_miss_freq)) *
                 ninner;        // load Density (cache miss, reload 8 doubles)
  memops8byte += 2 * ninner;    // load Densityfrac, Volfrac
  memops8byte += ninner;        // store Density
  memops8byte += ncells;        // load cell volume, Vol
  memops8byte += 2 * ncells;    // Load and Store Density
  int64_t memops4byte = ninner; // load subset2mesh
  int64_t flops = 2 * ninner;   // multiply and add
  flops += ncells;              // divide cell density by cell volume
  float penalty_msecs =
      0.0; // Only if we account for removed materials with an if-check
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);

  //    Average density with fractional densities - CELL-DOMINANT LOOP
  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int C = 0; C < ncells; C++) {
      double density_ave = 0.0;
      for (int im = 0; im < nmatscell[C]; im++) {
        int m = matids[4 * C + im];
        int c = mesh2subset[m][C];
        density_ave += Densityfrac[m][c] * Volfrac[m][c];
      }
      Density[C] = density_ave / Vol[C];
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Average Density of cells - Cell Dominant loop  -  compute time is "
         "%lf msecs\n",
         act_perf);

  ninner = 0;
  for (int C = 0; C < ncells; C++)
    for (int im = 0; im < nmatscell[C]; im++)
      ninner++;

  memops4byte = ncells;  // load nmatscells
  memops4byte += ninner; // load matids
  memops4byte += (int64_t)(16 * cache_miss_freq + (1 - cache_miss_freq)) *
                 ninner; // load mesh2subset (cache miss, reload 16 integers)
  memops8byte =
      (int64_t)(8 * cache_miss_freq + (1 - cache_miss_freq)) * 2 *
      ninner; // load Densityfrac, Volfrac (cache miss, reload 8 doubles)
  memops8byte += ncells; // load of cell volume Vol
  memops8byte += ncells; // Store of Density
  flops = 2 * ninner;    // multiply and add
  flops += ncells;       // divide density_ave by Vol
  penalty_msecs =
      0.0; // Only if we account for removed materials with an if-check
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);

  //   Calculate pressure using ideal gas law - MAT-CENTRIC COMPACT STRUCTURE
  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int m = 0; m < nmats; m++) {
      for (int c = 0; c < ncellsmat[m]; c++) {
        Pressurefrac[m][c] =
            (nmatconsts[m] * Densityfrac[m][c] * Temperaturefrac[m][c]) /
            Volfrac[m][c];
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Pressure Calculation of cells - Mat-Centric -  compute time is %lf "
         "msecs\n",
         act_perf);

  ninner = 0;
  for (int m = 0; m < nmats; m++)
    for (int c = 0; c < ncellsmat[m]; c++)
      ninner++;
  memops4byte = 0;
  memops8byte = nmats;       // load of nmatsconsts
  memops8byte += 3 * ninner; // load DensityFrac, TemperatureFrac, VolFrac
  memops8byte += ninner;     // store PressureFrac
  flops = 3 * ninner;
  penalty_msecs =
      0.0; // Only if we account for removed materials with an if-check
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);

  //    Average material density over neighborhood of each cell
  double **MatDensity_average =
      (double **)genmatrix("MatDensity_average", nmats, ncells, sizeof(double));

  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    cpu_timer_start(&tstart_cpu);

    for (int m = 0; m < nmats; m++) {
      for (int C = 0; C < ncells; C++)
        MatDensity_average[m][C] = 0.0;

      for (int c = 0; c < ncellsmat[m]; c++) { // Note that this is c not C
        int C = subset2mesh[m][c];
        double xc[2];
        xc[0] = cen_x[C];
        xc[1] = cen_y[C];
        int nn = nnbrs[C];
        int cnbrs[9];
        double dsqr[8];
        for (int n = 0; n < nn; n++)
          cnbrs[n] = nbrs[C][n];
        for (int n = 0; n < nn; n++) {
          dsqr[n] = 0.0;
          double ddist = (xc[0] - cen_x[cnbrs[n]]);
          dsqr[n] += ddist * ddist;
        }

        int nnm = 0; // number of nbrs with this material
        for (int n = 0; n < nn; n++) {
          int C_j = cnbrs[n];
          int c_j = mesh2subset[m][C_j];
          if (c_j >= 0) {
            MatDensity_average[m][C] += Densityfrac[m][c_j] / dsqr[n];
            nnm++;
          }
        }
        MatDensity_average[m][C] /= nnm;
      }
    }

    time_sum += cpu_timer_stop(tstart_cpu);
  }
  act_perf = time_sum * 1000.0 / itermax;
  printf("Average Material Density  -  compute time is %lf msecs\n", act_perf);

  memops8byte = ncells * nmats;
  memops8byte += (int64_t)24.5 * filled_fraction * ncells * nmats;
  memops8byte += (int64_t)8 * filled_fraction * ncells * nmats * nnbrs_ave;
  memops8byte +=
      (int64_t)17 * filled_fraction * ncells * nmats * nnbrs_ave * L_f;
  flops = (int64_t)8 * filled_fraction * ncells * nmats * nnbrs_ave * L_f;
  flops += (int64_t)filled_fraction * ncells * nmats;
  penalty_msecs = 0.0;
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);
//   Convert from MATERIAL-CENTRIC COMPACT DATA STRUCTURE to CELL_CENTRIC
//   COMPACT DATA STRUCTURE
//#define CONVERSION_CHECK 1
#ifdef CONVERSION_CHECK
  int *CCimaterial, *CCnmaterials, *CCimaterialfrac, *CCnextfrac, *CCfrac2cell;
  double *CCVol, *CCDensity, *CCTemperature, *CCPressure, *CCVolfrac,
      *CCDensityfrac, *CCTemperaturefrac, *CCPressurefrac;

  setup_cell_dominant_compact_data_structure(
      method, CCimaterial, CCnmaterials, CCVol, CCDensity, CCTemperature,
      CCPressure, CCimaterialfrac, CCnextfrac, CCfrac2cell, CCVolfrac,
      CCDensityfrac, CCTemperaturefrac, CCPressurefrac, filled_percentage);
#endif

  time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    int *Cimaterial;
    int *Cnmaterials;
    int *Cimaterialfrac;
    int *Cnextfrac;
    int *Cfrac2cell;
    double *CVolfrac;
    double *CDensityfrac;
    double *CTemperaturefrac;
    double *CPressurefrac;

    cpu_timer_start(&tstart_cpu);
    convert_compact_material_2_compact_cell(
        ncells, subset2mesh, mesh2subset, nmatscell, matids, ncellsmat, Volfrac,
        Densityfrac, Temperaturefrac, Pressurefrac, Cimaterial, Cnmaterials,
        Cimaterialfrac, Cnextfrac, Cfrac2cell, CVolfrac, CDensityfrac,
        CTemperaturefrac, CPressurefrac);
    time_sum += cpu_timer_stop(tstart_cpu);

#ifdef CONVERSION_CHECK
    int ix = 0;
    for (int ic = 0; ic < ncells; ic++) {
      int CNmats = nmatscell[ic];
      if (CNmats == 1) {
        if (CCimaterial[ic] != Cimaterial[ic] && ic < 100) {
          printf("DEBUG line %d ic %d CNmats %d CCimaterial %d Cimaterial %d\n",
                 __LINE__, ic, CNmats, CCimaterial[ic], Cimaterial[ic]);
        }
        if (matids[ic * 4] + 1 != Cimaterial[ic]) {
          printf("DEBUG line %d ic %d CNmats %d matids %d Cimaterial %d\n",
                 __LINE__, ic, CNmats, matids[ic * 4], Cimaterial[ic]);
          exit(1);
        }
      } else {
        ix = abs(Cimaterial[ic]);
        int m = 0;
        while (ix > 0) {
          if (matids[ic * 4 + m] + 1 != Cimaterialfrac[ix]) {
            printf(
                "DEBUG CNmats %d ix %d mixed material %d Cimaterialfrac %d\n",
                CNmats, ix, matids[ic * 4 + m] + 1, Cimaterialfrac[ix]);
          }
          if (CCimaterialfrac[ix] != Cimaterialfrac[ix]) {
            printf("DEBUG CNmats %d ix %d CCimaterialfrac %d Cimaterialfrac "
                   "%d\n",
                   CNmats, ix, CCimaterialfrac[ix], Cimaterialfrac[ix]);
          }
          ix = Cnextfrac[ix];
          m++;
        }
      }
    }
    exit(0);
#endif

    genvectorfree(Cimaterial);
    genvectorfree(Cimaterialfrac);
    genvectorfree(Cnextfrac);
    genvectorfree(Cfrac2cell);
    genvectorfree(CVolfrac);
    genvectorfree(CDensityfrac);
    genvectorfree(CTemperaturefrac);
    genvectorfree(CPressurefrac);
  }

#ifdef CONVERSION_CHECK
  genvectorfree(CCimaterial);
  genvectorfree(CCimaterialfrac);
  genvectorfree(CCnextfrac);
  genvectorfree(CCfrac2cell);
  genvectorfree(CCVolfrac);
  genvectorfree(CCDensityfrac);
  genvectorfree(CCTemperaturefrac);
  genvectorfree(CCPressurefrac);
#endif

  printf("Conversion from compact material data structure to compact cell "
         "data structure\n");
  act_perf = time_sum * 1000.0 / itermax;
  // 4 arrays read from and stored to. Add 8x penalty for non-contiguous reads
  memops8byte = (4 * 8 + 4) * 5.0e5;
  // reads with 8x penalty for non-contiguous reads
  memops4byte = 3 * 8 * ncells + 2 * 8 * 5.0e5 + 3 * 4 * (ncells + 5.0e5);
  flops = 0.1;
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);

  genvectorfree((void *)nmatscell);
  genvectorfree((void *)matids);
  genvectorfree((void *)Vol);
  genvectorfree((void *)Density);

  for (int m = 0; m < nmats; m++) {
    genvectorfree((void *)Volfrac[m]);
    genvectorfree((void *)Densityfrac[m]);
    genvectorfree((void *)Temperaturefrac[m]);
    genvectorfree((void *)Pressurefrac[m]);
    genvectorfree((void *)subset2mesh[m]);
    genvectorfree((void *)mesh2subset[m]);
  }
  genvectorfree((void *)Volfrac);
  genvectorfree((void *)Densityfrac);
  genvectorfree((void *)Temperaturefrac);
  genvectorfree((void *)Pressurefrac);
  genvectorfree((void *)subset2mesh);
  genmatrixfree((void **)mesh2subset);
}
