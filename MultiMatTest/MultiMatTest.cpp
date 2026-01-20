/*
 * Copyright (c) 2017, Los Alamos National Security, LLC.
 * All rights Reserved.
 *
 * This is the code released under LANL Copyright Disclosure C17041/LA-CC-17-041
 * Copyright 2017.  Los Alamos National Security, LLC. This material was
 * produced
 * under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National
 * Laboratory (LANL), which is operated by Los Alamos National Security, LLC
 * for the U.S. Department of Energy. See LICENSE file for details.
 *
 * Released under the New BSD License
 *
 * Bob Robey brobey@lanl.gov and Rao Garimella rao@lanl.gov
 */

#include "MultiMatTest.hpp"
#include "Tests.hpp"
#include "genmalloc.h"
#include "input.h"
#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MININT(i, j) (((i) < (j)) ? (i) : (j))

bool verbose = false;
bool memory_verbose = false;
int itermax = 100;
int ncells = 1000000;
int nmats = 50;
float model_error;

int main(int argc, char **argv) {
  struct timeval tstart_cpu;
  double nmatconst = 5.0;
  double *nmatconsts = (double *)genvector("nmatconsts", nmats, sizeof(double));
  for (int i = 0; i < nmats; i++) {
    nmatconsts[i] = 5.0;
  }
  int64_t flops;
  int method = 0; // VF initialization: 0 - random, 1 - read volfrac.dat

  if (argc > 1)
    sscanf(argv[1], "%d", &method);

  printf("Run stream benchmark for your system\n");
  printf("L3 Cache on Macbook Pro is 6MB so problem size is just bigger at "
         "16MB min\n");
  printf(
      "First test should give Stream Benchmark or problem size is too small\n");
  printf("Second problem should give about twice the first\n");

  // Some globals

  float L_f = method ? 0.5 : 1.0; // ave frac of nbrs containing material
  int nnbrs_ave = 8; // nearly so; 4000 boundary cells in 1 million cells
  //                  // in 3D, nnbrs_ave would be 26

  // Build up list of neighbors for each cell
  // Assuming a 2D structured mesh, each cell will have a maximum of 8 nbrs

  int *nnbrs = (int *)genvector("Num_Neighbors", ncells, sizeof(int));
  int **nbrs = (int **)genmatrix("Neighbors", ncells, nmats, sizeof(int));

  get_neighbors(ncells, nnbrs, nbrs);

  // Compute centroids of cells
  double *cen_x = (double *)genvector("Centroids_x", ncells, sizeof(double));
  double *cen_y = (double *)genvector("Centroids_y", ncells, sizeof(double));
  get_centroids(cen_x, cen_y);

  single_material(ncells, memory_verbose, itermax, nmatconst);

  cell_dominant_full_matrix(ncells, memory_verbose, itermax, nmatconst, method,
                            nmats, L_f, nnbrs_ave, nmatconsts, cen_x, cen_y,
                            nnbrs, nbrs);

  material_dominant_matrix(ncells, memory_verbose, itermax, nmatconst, method,
                           nmats, L_f, nnbrs_ave, nmatconsts, cen_x, cen_y,
                           nnbrs, nbrs);

  cell_dominant_compact(ncells, memory_verbose, itermax, nmatconst, method,
                        nmats, L_f, nnbrs_ave, nmatconsts, cen_x, cen_y, nnbrs,
                        nbrs);

  material_centric_compact(ncells, memory_verbose, itermax, nmatconst, method,
                           nmats, L_f, nnbrs_ave, nmatconsts, cen_x, cen_y,
                           nnbrs, nbrs);

  int *Cimaterial, *Cnmaterials, *Cimaterialfrac, *Cnextfrac, *Cfrac2cell;
  double *CVol, *CDensity, *CTemperature, *CPressure, *CVolfrac, *CDensityfrac,
      *CTemperaturefrac, *CPressurefrac;

  float filled_percentage = 0.0;
  setup_cell_dominant_compact_data_structure(
      method, Cimaterial, Cnmaterials, CVol, CDensity, CTemperature, CPressure,
      Cimaterialfrac, Cnextfrac, Cfrac2cell, CVolfrac, CDensityfrac,
      CTemperaturefrac, CPressurefrac, filled_percentage);
  for (int ic = 0; ic < ncells; ic++) {
    if (Cnmaterials[ic] < 1 || Cnmaterials[ic] > nmats) {
      printf("DEBUG -- ic %d Cnmaterials %d\n", ic, Cnmaterials[ic]);
    }
  }

  double time_sum = 0;
  for (int iter = 0; iter < itermax; iter++) {
    int **subset2mesh;
    int **mesh2subset;
    int *nmatscell;
    int *matids;
    int *ncellsmat;
    double **Volfrac;
    double **Densityfrac;
    double **Temperaturefrac;
    double **Pressurefrac;

    cpu_timer_start(&tstart_cpu);

    convert_compact_cell_2_compact_material(
        ncells, nmats, Cimaterial, Cnmaterials, Cimaterialfrac, CVol, CDensity,
        CTemperature, CVolfrac, CDensityfrac, CTemperaturefrac, subset2mesh,
        mesh2subset, nmatscell, matids, ncellsmat, Volfrac, Densityfrac,
        Temperaturefrac, Pressurefrac);

    time_sum += cpu_timer_stop(tstart_cpu);

    for (int m = 0; m < nmats; m++) {
      genvectorfree((void *)subset2mesh[m]);
      genvectorfree((void *)Volfrac[m]);
      genvectorfree((void *)Densityfrac[m]);
      genvectorfree((void *)Temperaturefrac[m]);
      genvectorfree((void *)Pressurefrac[m]);
    }
    genvectorfree((void **)subset2mesh);
    genvectorfree((void **)Volfrac);
    genvectorfree((void **)Densityfrac);
    genvectorfree((void **)Temperaturefrac);
    genvectorfree((void **)Pressurefrac);

    genvectorfree((void *)matids);
    genvectorfree((void *)ncellsmat);

    genmatrixfree((void **)mesh2subset);
  }

  printf("Conversion from compact cell data structure to compact material data "
         "structure\n");
  float act_perf = time_sum * 1000.0 / itermax;
  // 4 arrays read from and stored to. Add 8x penalty for non-contiguous reads
  int64_t memops8byte = (3 * 8 + 3) * (ncells + 5.0e5);
  // reads with 8x penalty for non-contiguous reads
  int64_t memops4byte =
      3 * 8 * ncells + 2 * 8 * 5.0e5 + 3 * 4 * (ncells + 5.0e5);
  memops4byte += 1 * (ncells * nmats);
  flops = 0.1;
  float penalty_msecs = 0.0;
  print_performance_estimates(act_perf, memops8byte, memops4byte, flops,
                              penalty_msecs);

  genvectorfree(Cimaterial);
  genvectorfree(Cimaterialfrac);
  genvectorfree(Cnextfrac);
  genvectorfree(Cfrac2cell);
  genvectorfree(CVolfrac);
  genvectorfree(CDensityfrac);
  genvectorfree(CTemperaturefrac);
  genvectorfree(CPressurefrac);
  genvectorfree(nmatconsts);
}

void setup_cell_dominant_data_structure(int method, double *&Vol,
                                        double *&Density, double *&Temperature,
                                        double *&Pressure, double **&Volfrac,
                                        double **&Densityfrac,
                                        double **&Temperaturefrac,
                                        double **&Pressurefrac,
                                        float &filled_percentage) {

  Vol = (double *)genvector("Volume", ncells, sizeof(double));
  Density = (double *)genvector("Density", ncells, sizeof(double));
  Temperature = (double *)genvector("Temperature", ncells, sizeof(double));
  Pressure = (double *)genvector("Pressure", ncells, sizeof(double));
  Densityfrac =
      (double **)genmatrix("DensityFrac", ncells, nmats, sizeof(double));
  Temperaturefrac =
      (double **)genmatrix("TemperatureFrac", ncells, nmats, sizeof(double));
  Pressurefrac =
      (double **)genmatrix("PressureFrac", ncells, nmats, sizeof(double));

  get_vol_frac_matrix(method, Volfrac, filled_percentage);

  for (int ic = 0; ic < ncells; ic++) {
    Vol[ic] = 0.0;
    for (int m = 0; m < nmats; m++) {
      if (Volfrac[ic][m] > 0.0) {
        Densityfrac[ic][m] = 2.0;
        Temperaturefrac[ic][m] = 0.5;
        Vol[ic] += Volfrac[ic][m];
      } else {
        Densityfrac[ic][m] = 0.0;
        Temperaturefrac[ic][m] = 0.0;
      }
      Pressurefrac[ic][m] = 0.0;
    }
  }
}

void setup_material_dominant_data_structure(
    int method, double *&Vol, double *&Density, double *&Temperature,
    double *&Pressure, double **&Volfrac, double **&Densityfrac,
    double **&Temperaturefrac, double **&Pressurefrac,
    float &filled_percentage) {

  Vol = (double *)genvector("Volume", ncells, sizeof(double));
  Density = (double *)genvector("Density", ncells, sizeof(double));
  Temperature = (double *)genvector("Temperature", ncells, sizeof(double));
  Pressure = (double *)genvector("Pressure", ncells, sizeof(double));
  Volfrac = (double **)genmatrix("VolumeFrac", nmats, ncells, sizeof(double));
  Densityfrac =
      (double **)genmatrix("DensityFrac", nmats, ncells, sizeof(double));
  Temperaturefrac =
      (double **)genmatrix("TemperatureFrac", nmats, ncells, sizeof(double));
  Pressurefrac =
      (double **)genmatrix("PressureFrac", nmats, ncells, sizeof(double));

  double **Volfrac_fullcc; // cell centric full matrix of fractional volumes
  get_vol_frac_matrix(method, Volfrac_fullcc, filled_percentage);

  for (int m = 0; m < nmats; m++) {
    for (int ic = 0; ic < ncells; ic++) {
      if (Volfrac_fullcc[ic][m] > 0.0) {
        Volfrac[m][ic] = Volfrac_fullcc[ic][m];
        Densityfrac[m][ic] = 2.0;
        Temperaturefrac[m][ic] = 0.5;
      } else {
        Volfrac[m][ic] = Volfrac_fullcc[ic][m];
        Densityfrac[m][ic] = 0.0;
        Temperaturefrac[m][ic] = 0.0;
      }
      Pressurefrac[m][ic] = 0.0;
    }
  }

  // Now free the full data structures
  genmatrixfree((void **)Volfrac_fullcc);
}

void setup_cell_dominant_compact_data_structure(
    int method, int *&imaterial, int *&nmaterials, double *&Vol,
    double *&Density, double *&Temperature, double *&Pressure,
    int *&imaterialfrac, int *&nextfrac, int *&frac2cell, double *&Volfrac,
    double *&Densityfrac, double *&Temperaturefrac, double *&Pressurefrac,
    float &filled_percentage) {

  imaterial = (int *)genvector("imaterial", ncells, sizeof(int));
  nmaterials = (int *)genvector("nmaterials", ncells, sizeof(int));
  Vol = (double *)genvector("Vol", ncells, sizeof(double));
  Density = (double *)genvector("Density", ncells, sizeof(double));
  Temperature = (double *)genvector("Temperature", ncells, sizeof(double));
  Pressure = (double *)genvector("Pressure", ncells, sizeof(double));

  double **Volfrac_fullcc; // full cell-centric matrix of fractional volumes
  get_vol_frac_matrix(method, Volfrac_fullcc, filled_percentage);

  int ix = 0;
  for (int ic = 0; ic < ncells; ic++) {
    int nnz = 0;
    for (int im = 0; im < nmats; im++) {
      if (Volfrac_fullcc[ic][im] > 0.0)
        nnz++;
    }
    if (nnz > 1)
      ix += nnz;
  }
  int mxsize = ix;

  imaterialfrac = (int *)genvector("imaterialfrac", mxsize, sizeof(int));
  nextfrac = (int *)genvector("nextfrac", mxsize, sizeof(int));
  frac2cell = (int *)genvector("frac2cell", mxsize, sizeof(int));
  Volfrac = (double *)genvector("Volfrac", mxsize, sizeof(double));
  Densityfrac = (double *)genvector("Densityfrac", mxsize, sizeof(double));
  Temperaturefrac =
      (double *)genvector("Temperaturefrac", mxsize, sizeof(double));
  Pressurefrac = (double *)genvector("Pressurefrac", mxsize, sizeof(double));

  ix = 0;
  for (int ic = 0; ic < ncells; ic++) {
    int m1, m2, m3, m4;
    int nnz = 0;
    for (int im = 0; im < nmats; im++) {
      if (Volfrac_fullcc[ic][im] > 0.0) {
        nnz++;
        if (nnz == 1)
          m1 = im;
        else if (nnz == 2)
          m2 = im;
        else if (nnz == 3)
          m3 = im;
        else if (nnz == 4)
          m4 = im;
      }
    }
    if (nnz == 4) {
      imaterial[ic] = -ix;
      nmaterials[ic] = 4;
      imaterialfrac[ix] = (m1 + 1);
      imaterialfrac[ix + 1] = (m2 + 1);
      imaterialfrac[ix + 2] = (m3 + 1);
      imaterialfrac[ix + 3] = (m4 + 1);
      Volfrac[ix] = 0.4;
      Volfrac[ix + 1] = 0.3;
      Volfrac[ix + 2] = 0.2;
      Volfrac[ix + 3] = 0.1;
      Densityfrac[ix] = 2.0;
      Densityfrac[ix + 1] = 2.0;
      Densityfrac[ix + 2] = 2.0;
      Densityfrac[ix + 3] = 2.0;
      Temperaturefrac[ix] = 0.5;
      Temperaturefrac[ix + 1] = 0.5;
      Temperaturefrac[ix + 2] = 0.5;
      Temperaturefrac[ix + 3] = 0.5;
      nextfrac[ix] = ix + 1;
      nextfrac[ix + 1] = ix + 2;
      nextfrac[ix + 2] = ix + 3;
      nextfrac[ix + 3] = -1;
      frac2cell[ix] = ic;
      frac2cell[ix + 1] = ic;
      frac2cell[ix + 2] = ic;
      frac2cell[ix + 3] = ic;
      ix += 4;
    } else if (nnz == 3) {
      imaterial[ic] = -ix;
      nmaterials[ic] = 3;
      imaterialfrac[ix] = (m1 + 1);
      imaterialfrac[ix + 1] = (m2 + 1);
      imaterialfrac[ix + 2] = (m3 + 1);
      Volfrac[ix] = 0.5;
      Volfrac[ix + 1] = 0.3;
      Volfrac[ix + 2] = 0.2;
      Densityfrac[ix] = 2.0;
      Densityfrac[ix + 1] = 2.0;
      Densityfrac[ix + 2] = 2.0;
      Temperaturefrac[ix] = 0.5;
      Temperaturefrac[ix + 1] = 0.5;
      Temperaturefrac[ix + 2] = 0.5;
      nextfrac[ix] = ix + 1;
      nextfrac[ix + 1] = ix + 2;
      nextfrac[ix + 2] = -1;
      frac2cell[ix] = ic;
      frac2cell[ix + 1] = ic;
      frac2cell[ix + 2] = ic;
      ix += 3;
    } else if (nnz == 2) {
      imaterial[ic] = -ix;
      nmaterials[ic] = 2;
      imaterialfrac[ix] = (m1 + 1);
      imaterialfrac[ix + 1] = (m2 + 1);
      Volfrac[ix] = 0.5;
      Volfrac[ix + 1] = 0.5;
      Densityfrac[ix] = 2.0;
      Densityfrac[ix + 1] = 2.0;
      Temperaturefrac[ix] = 0.5;
      Temperaturefrac[ix + 1] = 0.5;
      nextfrac[ix] = ix + 1;
      nextfrac[ix + 1] = -1;
      frac2cell[ix] = ic;
      frac2cell[ix + 1] = ic;
      ix += 2;
    } else {
      imaterial[ic] = (m1 + 1);
      nmaterials[ic] = 1;
    }
  }

  int filled_count = 0;
  int pure_cell_count = 0;
  int mixed_cell_count = 0;
  int mixed_frac_count = 0;
  for (int ic = 0; ic < ncells; ic++) {
    int ix = imaterial[ic];
    if (ix > 0) { // material numbers for clean cells start at 1
                  // clean cells
      Vol[ic] = 1.0;
      Density[ic] = 2.0;
      Temperature[ic] = 0.5;
      pure_cell_count++;
      filled_count++;
    } else {
      // multimaterial cells
      mixed_cell_count++;
      for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
        Vol[ic] += Volfrac[ix];
        filled_count++;
        mixed_frac_count++;
      }
    }
  }

#ifdef XXX
  for (int ic = 0; ic < ncells; ic++) {
    int ix = imaterial[ic];
    // clean cells
    printf("DEBUG cell is %3d Density %8.2lf material is %d\n", ic, Density[ic],
           ix);
  }

  for (int ix = 0; ix < mxsize; ix++) {
    printf("DEBUG mx is %d Dfrac %lf irf %d nxtf %d ixf %d\n", ix,
           Densityfrac[ix], imaterialfrac[ix], nextfrac[ix], frac2cell[ix]);
  }

  for (int ic = 0; ic < ncells; ic++) {
    int ix = imaterial[ic];
    if (ix > 0) { // material numbers for clean cells start at 1
      // clean cells
      printf("DEBUG cell is %3d Density %8.2lf material is %d\n", ic,
             Density[ic], ix);
    } else {
      // multimaterial cells
      for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
        printf("DEBUG cell is %3d mx is %3d Dfrac %8.2lf irf %d nxtf %4d ixf "
               "%3d\n",
               ic, ix, Densityfrac[ix], imaterialfrac[ix], nextfrac[ix],
               frac2cell[ix]);
        if (frac2cell[ix] != ic) {
          printf("DEBUG -- error!!! mix item %d points to wrong cell %d, "
                 "should be ic %d\n",
                 ic, frac2cell[ix], ic);
          break;
        }
      }
    }
  }
#endif

  // Now free the full data structures
  genmatrixfree((void **)Volfrac_fullcc);
}

void setup_mat_dominant_compact_data_structure(
    int method, int **&subset2mesh, int **&mesh2subset, int *&nmatscell,
    int *&matids, int *&ncellsmat, double *&Vol, double *&Density,
    double **&Volfrac, double **&Densityfrac, double **&Temperaturefrac,
    double **&Pressurefrac, float &filled_percentage) {

  Vol = (double *)genvector("Volume", ncells, sizeof(double));
  Density = (double *)genvector("Density", ncells, sizeof(double));

  mesh2subset = (int **)genmatrix("mesh2subset", nmats, ncells, sizeof(int));
  for (int m = 0; m < nmats; m++)
    for (int C = 0; C < ncells; C++)
      mesh2subset[m][C] = -1;
  nmatscell = (int *)genvector("nmatscell", ncells, sizeof(int));
  for (int C = 0; C < ncells; C++)
    nmatscell[C] = 0;
  matids = (int *)genvector("matids", 4 * ncells, sizeof(int));
  for (int C = 0; C < 4 * ncells; C++)
    matids[C] = -1;
  ncellsmat = (int *)genvector("ncellsmat", nmats, sizeof(int));
  for (int m = 0; m < nmats; m++)
    ncellsmat[m] = 0;

  subset2mesh = (int **)genvector("subset2mesh", nmats, sizeof(int *));
  Volfrac = (double **)genvector("VolumeFrac", nmats, sizeof(double *));
  Densityfrac = (double **)genvector("DensityFrac", nmats, sizeof(double *));
  Temperaturefrac =
      (double **)genvector("TemperatureFrac", nmats, sizeof(double *));
  Pressurefrac = (double **)genvector("PressureFrac", nmats, sizeof(double *));

  double **Volfrac_fullcc;
  get_vol_frac_matrix(method, Volfrac_fullcc, filled_percentage);

  for (int ic = 0; ic < ncells; ic++) {
    nmatscell[ic] = 0;
    for (int m = 0; m < nmats; m++) {
      if (Volfrac_fullcc[ic][m] > 0.0) {
        matids[4 * ic + nmatscell[ic]] = m;
        nmatscell[ic]++;
        ncellsmat[m]++;
      }
    }
  }

  // Allocate compact data structures

  for (int m = 0; m < nmats; m++) {
    subset2mesh[m] =
        (int *)genvector("subset2mesh_m", ncellsmat[m], sizeof(int));
    Volfrac[m] = (double *)genvector("VolFrac_m", ncellsmat[m], sizeof(double));
    Densityfrac[m] =
        (double *)genvector("DensityFrac_m", ncellsmat[m], sizeof(double));
    Temperaturefrac[m] =
        (double *)genvector("TemperatureFrac_m", ncellsmat[m], sizeof(double));
    Pressurefrac[m] =
        (double *)genvector("PressureFrac_m", ncellsmat[m], sizeof(double));
  }

  // Now populate the compact data structures
  for (int m = 0; m < nmats; m++)
    ncellsmat[m] = 0;
  for (int C = 0; C < ncells; C++) {
    for (int im = 0; im < nmatscell[C]; im++) {
      int m = matids[4 * C + im];
      int c = ncellsmat[m];
      subset2mesh[m][c] = C;
      mesh2subset[m][C] = c;
      Volfrac[m][c] = Volfrac_fullcc[C][m];
      Densityfrac[m][c] = 2.0;
      Temperaturefrac[m][c] = 0.5;
      (ncellsmat[m])++;
    }
  }

  // Now free the full data structures
  genmatrixfree((void **)Volfrac_fullcc);
}

void convert_compact_material_2_compact_cell(
    int ncells, int **subset2mesh, int **mesh2subset, int *nmatscell,
    int *matids, int *ncellsmat, double **Volfrac, double **Densityfrac,
    double **Temperaturefrac, double **Pressurefrac, int *&Cimaterial,
    int *&Cnmaterials, int *&Cimaterialfrac, int *&Cnextfrac, int *&Cfrac2cell,
    double *&CVolfrac, double *&CDensityfrac, double *&CTemperaturefrac,
    double *&CPressurefrac) {
  Cimaterial = (int *)genvector("Cimaterial", ncells, sizeof(int));
  Cnmaterials = nmatscell; // This is the same in both, but we assign pointer
                           // to maintain the naming convention

  // First we count up the number of mixed cells for the mix cell data
  // structure. We
  // skip single material cells since they do not get put into the compact
  // structure.
  int mxsize = 0;
  for (int ic = 0; ic < ncells; ic++) {
    if (nmatscell[ic] > 1)
      mxsize += nmatscell[ic];
  }

  Cimaterialfrac = (int *)genvector("Cimaterialfrac", mxsize, sizeof(int));
  Cnextfrac = (int *)genvector("Cnextfrac", mxsize, sizeof(int));
  Cfrac2cell = (int *)genvector("Cfrac2cell", mxsize, sizeof(int));
  CVolfrac = (double *)genvector("CVolfrac", mxsize, sizeof(double));
  CDensityfrac = (double *)genvector("CDensityfrac", mxsize, sizeof(double));
  CTemperaturefrac =
      (double *)genvector("CTemperaturefrac", mxsize, sizeof(double));
  CPressurefrac = (double *)genvector("CPressurefrac", mxsize, sizeof(double));

  // This assumes that Vol, Density, Temperature, Pressure hold the correct
  // values for single material values. If not, they should be copied
  // from CVolfrac, CDensityfrac, CTemperaturefrac, and CPressurefrac

  int ix = 0;
  for (int ic = 0; ic < ncells; ic++) {
    int nmats = nmatscell[ic];
    if (nmats == 1) {
      int m = matids[4 * ic];
      Cimaterial[ic] = m + 1;
    } else { // nmatscell > 1
      Cimaterial[ic] = -ix;
      for (int im = 0; im < nmats; im++) {
        int m = matids[4 * ic + im];
        int c = mesh2subset[m][ic];
        Cimaterialfrac[ix] = m + 1;
        Cnextfrac[ix] = ix + 1;
        Cfrac2cell[ix] = ic;
        CVolfrac[ix] = Volfrac[m][c];
        CDensityfrac[ix] = Densityfrac[m][c];
        CTemperaturefrac[ix] = Temperaturefrac[m][c];
        CPressurefrac[ix] = Pressurefrac[m][c];
        ix++;
      }
      Cnextfrac[ix - 1] = -1;
    } // nmatscell > 1
  }
}

void convert_compact_cell_2_compact_material(
    int ncells, int nmats, int *Cimaterial, int *Cnmaterials,
    int *Cimaterialfrac, double *CVol, double *CDensity, double *CTemperature,
    double *CVolfrac, double *CDensityfrac, double *CTemperaturefrac,
    int **&subset2mesh, int **&mesh2subset, int *&nmatscell, int *&matids,
    int *&ncellsmat, double **&Volfrac, double **&Densityfrac,
    double **&Temperaturefrac, double **&Pressurefrac) {

  // Already setup and just needs a name change
  nmatscell = Cnmaterials;

  mesh2subset = (int **)genmatrix("mesh2subset", nmats, ncells, sizeof(int));
  for (int m = 0; m < nmats; m++)
    for (int C = 0; C < ncells; C++)
      mesh2subset[m][C] = -1;

  matids = (int *)genvector("matids", 4 * ncells, sizeof(int));
  for (int C = 0; C < 4 * ncells; C++)
    matids[C] = -1;
  ncellsmat = (int *)genvector("ncellsmat", nmats, sizeof(int));
  for (int m = 0; m < nmats; m++)
    ncellsmat[m] = 0;

  // We need ncellsmat for each material
  int ix = 0;
  for (int ic = 0; ic < ncells; ic++) {
    int CNmats = Cnmaterials[ic];
    if (CNmats == 1) {
      int m = Cimaterial[ic] - 1;
      (ncellsmat[m])++;
    } else {
      for (int im = 0; im < CNmats; im++) {
        int m = Cimaterialfrac[ix] - 1;
        (ncellsmat[m])++;
        ix++;
      }
    }
  }

  subset2mesh = (int **)genvector("subset2mesh", nmats, sizeof(int *));
  Volfrac = (double **)genvector("VolumeFrac", nmats, sizeof(double *));
  Densityfrac = (double **)genvector("DensityFrac", nmats, sizeof(double *));
  Temperaturefrac =
      (double **)genvector("TemperatureFrac", nmats, sizeof(double *));
  Pressurefrac = (double **)genvector("PressureFrac", nmats, sizeof(double *));

  // Allocate compact data structures

  for (int m = 0; m < nmats; m++) {
    subset2mesh[m] =
        (int *)genvector("subset2mesh_m", ncellsmat[m], sizeof(int));
    Volfrac[m] = (double *)genvector("VolFrac_m", ncellsmat[m], sizeof(double));
    Densityfrac[m] =
        (double *)genvector("DensityFrac_m", ncellsmat[m], sizeof(double));
    Temperaturefrac[m] =
        (double *)genvector("TemperatureFrac_m", ncellsmat[m], sizeof(double));
    Pressurefrac[m] =
        (double *)genvector("PressureFrac_m", ncellsmat[m], sizeof(double));
  }

  // Now populate the compact data structures
  for (int m = 0; m < nmats; m++)
    ncellsmat[m] = 0;
  ix = 0;
  for (int ic = 0; ic < ncells; ic++) {
    int CNmats = Cnmaterials[ic];
    if (CNmats == 1) {
      int m = Cimaterial[ic] - 1;
      int c = ncellsmat[m];
      matids[4 * ic] = m;
      mesh2subset[m][ic] = c;
      subset2mesh[m][c] = ic;
      Volfrac[m][c] = CVol[ic];
      Densityfrac[m][c] = CDensity[ic];
      Temperaturefrac[m][c] = CTemperature[ic];
      (ncellsmat[m])++;
    } else {
      for (int im = 0; im < CNmats; im++) {
        int m = Cimaterialfrac[ix] - 1;
        int c = ncellsmat[m];
        matids[4 * ic + im] = m;
        mesh2subset[m][ic] = c;
        subset2mesh[m][c] = ic;
        Volfrac[m][c] = CVolfrac[ix];
        Densityfrac[m][c] = CDensityfrac[ix];
        Temperaturefrac[m][c] = CTemperaturefrac[ix];
        (ncellsmat[m])++;
        ix++;
      }
    }
  }
}

void print_performance_estimates(float est_perf, int64_t memops8byte,
                                 int64_t memops4byte, int64_t flops,
                                 float penalty_msecs) {

  float Megamemops, Megabytes, Megaflops;
  int STREAM = STREAM_RATE;
  float act_perf = est_perf;

  Megamemops = (float)(memops8byte + memops4byte) / 1000000.0;

  // First divide by 1000000 and then multiply by bytes to avoid overflow
  // Using floats to make sure this works on GPUs as well?
  Megabytes = 8 * (((float)memops8byte) / 1000000) +
              4 * (((float)memops4byte) / 1000000.);
  Megaflops = (float)flops / 1000000.;
  printf("Memory Operations  are %.1f M memops, %.1f Mbytes, %.1f Mflops, "
         "%0.2f:1 memops:flops\n",
         Megamemops, Megabytes, Megaflops,
         (float)Megamemops / (float)Megaflops);
  est_perf = (float)Megabytes / (float)STREAM * 1000.0 + penalty_msecs;
  model_error = (est_perf - act_perf) / act_perf * 100.0;
  printf("Estimated performance %.2f msec, actual %.2f msec, model error %f "
         "%%\n\n",
         est_perf, act_perf, model_error);
}

void get_vol_frac_matrix(int method, double **&Volfrac,
                         float &filled_percentage) {

  if (method == 0)
    get_vol_frac_matrix_rand(Volfrac, filled_percentage);
  else if (method == 1)
    get_vol_frac_matrix_file(Volfrac, filled_percentage);
}

void get_vol_frac_matrix_rand(double **&Volfrac, float &filled_percentage) {

  Volfrac = (double **)genmatrix("VolumeBase", ncells, nmats, sizeof(double));
  int *mf_rand = (int *)genvector("mf_rand", ncells, sizeof(int));

  srand(0);
  for (int ic = 0; ic < ncells; ic++) {
    mf_rand[ic] =
        (int)((float)rand() * 1000.0 / (float)((long long)RAND_MAX + 1));
  }

  for (int ic = 0; ic < ncells; ic++)
    for (int m = 0; m < nmats; m++)
      Volfrac[ic][m] = 0.0;

  double VolTotal = 0.0;
  int filled_count = 0;
  int mixed_cell_count = 0;
  int mixed_frac_count = 0;
  int pure_frac_count = 0;
  int pure_cell_count = 0;
  int onematcell = 0;
  int twomatcell = 0;
  int threematcell = 0;
  int fourmatcell = 0;
  int fiveplusmatcell = 0;
  for (int ic = 0; ic < ncells; ic++) {
    int m1 =
        (int)((float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1));
    m1 = MININT(m1, nmats - 1);
    Volfrac[ic][m1] = 1.0;
    int mf = mf_rand[ic];
    if (mf < 25) {
      int m2 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      while (m2 == m1) {
        m2 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      }
      int m3 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      while (m3 == m2 || m3 == m1) {
        m3 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      }
      int m4 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      while (m4 == m3 || m4 == m2 || m4 == m1) {
        m4 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      }
      m2 = MININT(m2, nmats - 1);
      m3 = MININT(m3, nmats - 1);
      m4 = MININT(m4, nmats - 1);
      Volfrac[ic][m1] = 0.4;
      Volfrac[ic][m2] = 0.3;
      Volfrac[ic][m3] = 0.2;
      Volfrac[ic][m4] = 0.1;
    } else if (mf < 75) {
      int m2 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      while (m2 == m1) {
        m2 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      }
      int m3 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      while (m3 == m2 || m3 == m1) {
        m3 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      }
      m2 = MININT(m2, nmats - 1);
      m3 = MININT(m3, nmats - 1);
      Volfrac[ic][m1] = 0.5;
      Volfrac[ic][m2] = 0.3;
      Volfrac[ic][m3] = 0.2;
    } else if (mf < 200) {
      int m2 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      while (m2 == m1) {
        m2 = (float)rand() * (float)nmats / (float)((long long)RAND_MAX + 1);
      }
      m2 = MININT(m2, nmats - 1);
      Volfrac[ic][m1] = 0.5;
      Volfrac[ic][m2] = 0.5;
    }
    int mat_count = 0;
    for (int m = 0; m < nmats; m++) {
      if (Volfrac[ic][m] > 0.0) {
        filled_count++;
        mat_count++;
      }
      VolTotal += Volfrac[ic][m];
    }
    if (mat_count >= 2) {
      mixed_cell_count++;
      mixed_frac_count += mat_count;
    } else {
      pure_frac_count++;
    }
    if (mat_count == 1)
      pure_cell_count++;
    if (mat_count == 1)
      onematcell++;
    if (mat_count == 2)
      twomatcell++;
    if (mat_count == 3)
      threematcell++;
    if (mat_count == 4)
      fourmatcell++;
    if (mat_count >= 5)
      fiveplusmatcell++;
  }

  genvectorfree(mf_rand);

  printf("Ratios to Full Data Structure\n");
  filled_percentage = (float)filled_count * 100.0 / (float)(ncells * nmats);
  float sparsity_percentage =
      (float)(ncells * nmats - filled_count) * 100.0 / (float)(ncells * nmats);
  printf("Sparsity %lf percent/Filled %lf percent\n\n", sparsity_percentage,
         filled_percentage);

  printf("Ratios to Number of Cells\n");
  float pure_cell_percentage = (float)pure_cell_count * 100.0 / (float)ncells;
  float mixed_cell_percentage = (float)mixed_cell_count * 100.0 / (float)ncells;
  printf("Pure cell %lf percentage/Mixed material %lf percentage\n\n",
         pure_cell_percentage, mixed_cell_percentage);

  printf("Ratios to Mixed Material Data Structure\n");
  float mixed_material_sparsity_percentage =
      (float)mixed_frac_count * 100.0 / (float)(mixed_cell_count * nmats);
  float mixed_material_filled_percentage =
      (float)(mixed_cell_count * nmats - mixed_frac_count) * 100.0 /
      (float)(mixed_cell_count * nmats);
  printf("Mixed material Sparsity %lf percent/Mixed material Filled %lf "
         "percent\n\n",
         mixed_material_sparsity_percentage, mixed_material_filled_percentage);

  // printf("Vol Total %lf\n",VolTotal);
  // printf("%f percent of the cells are
  // filled\n",(float)filled_count*100.0/(float)(ncells*nmats));
  // printf("%f percent of the cells are
  // mixed\n",(float)mixed_cell_count*100.0/(float)ncells);
  // printf("%f percent of the total are
  // mixed\n",(float)mixed_frac_count*100.0/(float)(ncells*nmats));
  // printf("%f percent of the frac are
  // mixed\n",(float)mixed_frac_count*100.0/(float)(mixed_cell_count*nmats));
  // printf("%f percent
  // sparsity\n",(float)(ncells*nmats-mixed_frac_count)*100.0/(float)(ncells*nmats));
  // printf("%f percent of the frac are
  // pure\n",(float)pure_frac_count*100.0/(float)ncells);
  printf("1 matcell %d 2 matcell %d 3 matcell %d 4 matcell %d 5 matcell %d\n\n",
         onematcell, twomatcell, threematcell, fourmatcell, fiveplusmatcell);
  // printf("Total cells %d\n\n",
  // onematcell+2*twomatcell+3*threematcell+4*fourmatcell+5*fiveplusmatcell);
}

void get_vol_frac_matrix_file(double **&Volfrac, float &filled_percentage) {
  int status;
  FILE *fp;
  fp = fopen("volfrac.dat", "r");
  if (!fp) {
    fprintf(stderr, "unable to read volume fractions from file \"%s\"\n",
            "volfrac.dat");
    exit(-1);
  }

  status = fscanf(fp, "%d", &nmats);
  if (status < 0) {
    printf("error in read at line %d\n", __LINE__);
    exit(1);
  }
  Volfrac = (double **)genmatrix("VolumeBase", ncells, nmats, sizeof(double));

  for (int ic = 0; ic < ncells; ic++)
    for (int m = 0; m < nmats; m++)
      Volfrac[ic][m] = 0.0;

  char matname[256];
  for (int m = 0; m < nmats; m++) {
    status = fscanf(fp, "%s", matname); // read and discard
    if (status < 0) {
      printf("error in read at line %d\n", __LINE__);
      exit(1);
    }
  }

  double VolTotal = 0.0;
  int filled_count = 0;
  int mixed_cell_count = 0;
  int mixed_frac_count = 0;
  int pure_frac_count = 0;
  int pure_cell_count = 0;
  int onematcell = 0;
  int twomatcell = 0;
  int threematcell = 0;
  int fourmatcell = 0;
  int fiveplusmatcell = 0;
  for (int ic = 0; ic < ncells; ic++) {
    int mat_count = 0;
    for (int m = 0; m < nmats; m++) {
      status = fscanf(fp, "%lf", &(Volfrac[ic][m]));
      if (status < 0) {
        printf("error in read at line %d\n", __LINE__);
        exit(1);
      }
      if (Volfrac[ic][m] > 0.0) {
        filled_count++;
        mat_count++;
      }
      VolTotal += Volfrac[ic][m];
    }
    if (mat_count >= 2) {
      mixed_cell_count++;
      mixed_frac_count += mat_count;
    } else {
      pure_frac_count++;
    }
    if (mat_count == 1)
      pure_cell_count++;
    if (mat_count == 1)
      onematcell++;
    if (mat_count == 2)
      twomatcell++;
    if (mat_count == 3)
      threematcell++;
    if (mat_count == 4)
      fourmatcell++;
    if (mat_count >= 5)
      fiveplusmatcell++;
  }
  fclose(fp);

  printf("Ratios to Full Data Structure\n");
  filled_percentage = (float)filled_count * 100.0 / (float)(ncells * nmats);
  float sparsity_percentage =
      (float)(ncells * nmats - filled_count) * 100.0 / (float)(ncells * nmats);
  printf("Sparsity %lf percent/Filled %lf percent\n\n", sparsity_percentage,
         filled_percentage);

  printf("Ratios to Number of Cells\n");
  float pure_cell_percentage = (float)pure_cell_count * 100.0 / (float)ncells;
  float mixed_cell_percentage = (float)mixed_cell_count * 100.0 / (float)ncells;
  printf("Pure cell %lf percentage/Mixed material %lf percentage\n\n",
         pure_cell_percentage, mixed_cell_percentage);

  printf("Ratios to Mixed Material Data Structure\n");
  float mixed_material_sparsity_percentage =
      (float)mixed_frac_count * 100.0 / (float)(mixed_cell_count * nmats);
  float mixed_material_filled_percentage =
      (float)(mixed_cell_count * nmats - mixed_frac_count) * 100.0 /
      (float)(mixed_cell_count * nmats);
  printf("Mixed material Sparsity %lf percent/Mixed material Filled %lf "
         "percent\n\n",
         mixed_material_sparsity_percentage, mixed_material_filled_percentage);

  printf("Vol Total %lf\n", VolTotal);
  printf("%f percent of the cells are filled\n",
         (float)filled_count * 100.0 / (float)(ncells * nmats));
  printf("%f percent of the cells are mixed\n",
         (float)mixed_cell_count * 100.0 / (float)ncells);
  printf("%f percent of the total are mixed\n",
         (float)mixed_frac_count * 100.0 / (float)(ncells * nmats));
  printf("%f percent of the frac are mixed\n",
         (float)mixed_frac_count * 100.0 / (float)(mixed_cell_count * nmats));
  printf("%f percent sparsity\n",
         (float)(ncells * nmats - mixed_frac_count) * 100.0 /
             (float)(ncells * nmats));
  printf("%f percent of the frac are pure\n",
         (float)pure_frac_count * 100.0 / (float)ncells);
  printf("1 matcell %d 2 matcell %d 3 matcell %d 4 matcell %d 5 matcell %d\n\n",
         onematcell, twomatcell, threematcell, fourmatcell, fiveplusmatcell);
  printf("Total cells %d\n\n",
         onematcell + 2 * twomatcell + 3 * threematcell + 4 * fourmatcell +
             5 * fiveplusmatcell);
}

void get_neighbors(int ncells, int *&num_nbrs, int **&nbrs) {
  int ncells1 = (int)sqrt(ncells); // assumes ncells is a perfect square
  if (ncells1 * ncells1 != ncells) {
    fprintf(stderr, "Number of cells in mesh is not a perfect square");
    exit(-1);
  }

  for (int i = 0; i < ncells1; i++) {
    for (int j = 0; j < ncells1; j++) {
      int c = i * ncells1 + j;
      int ilo = i == 0 ? i : i - 1;
      int jlo = j == 0 ? j : j - 1;
      int ihi = i == ncells1 - 1 ? i : i + 1;
      int jhi = j == ncells1 - 1 ? j : j + 1;
      int n = 0;
      for (int i1 = ilo; i1 <= ihi; i1++)
        for (int j1 = jlo; j1 <= jhi; j1++) {
          int c2 = i1 * ncells1 + j1;
          if (c2 != c) {
            nbrs[c][n] = i1 * ncells1 + j1;
            n++;
          }
        }
      num_nbrs[c] = n;
    }
  }
}

void get_centroids(double *cen_x, double *cen_y) {
  int ncells1 = (int)sqrt(ncells); // assumes ncells is a perfect square
  if (ncells1 * ncells1 != ncells) {
    fprintf(stderr, "Number of cells in mesh is not a perfect square");
    exit(-1);
  }

  // Assume domain is a unit square

  double XLO = 0.0, YLO = 0.0, XHI = 1.0, YHI = 1.0;
  double dx = (XHI - XLO) / ncells1, dy = (YHI - YLO) / ncells1;

  for (int i = 0; i < ncells1; i++) {
    for (int j = 0; j < ncells1; j++) {
      int c = i * ncells1 + j;
      cen_x[c] = XLO + i * dx;
      cen_y[c] = YLO + j * dy;
    }
  }
}
