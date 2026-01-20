#pragma once

#include <stdlib.h>

#define CLOCK_RATE 2.7e9   // GHz laptop
#define STREAM_RATE 13375; // MB/sec

void get_neighbors(int ncells, int *&num_nbrs, int **&nbrs);
void get_centroids(double *cen_x, double *cen_y);
void get_vol_frac_matrix_rand(double **&Volfrac, float &filled_percentage);
void get_vol_frac_matrix_file(double **&Volfrac, float &filled_percentage);
void get_vol_frac_matrix(int method, double **&Volfrac,
                         float &filled_percentage);
void setup_cell_dominant_data_structure(int method, double *&Vol,
                                        double *&Density, double *&Temperature,
                                        double *&Pressure, double **&Volfrac,
                                        double **&Densityfrac,
                                        double **&Temperaturefrac,
                                        double **&Pressurefrac,
                                        float &filled_percentage);
void setup_material_dominant_data_structure(
    int method, double *&Vol, double *&Density, double *&Temperature,
    double *&Pressure, double **&Volfrac, double **&Densityfrac,
    double **&Temperaturefrac, double **&Pressurefrac,
    float &filled_percentage);
void setup_cell_dominant_compact_data_structure(
    int method, int *&imaterial, int *&nmaterials, double *&Vol,
    double *&Density, double *&Temperature, double *&Pressure,
    int *&imaterialfrac, int *&nextfrac, int *&frac2cell, double *&Volfrac,
    double *&Densityfrac, double *&Temperaturefrac, double *&Pressurefrac,
    float &filled_percentage);
void setup_mat_dominant_compact_data_structure(
    int method, int **&subset2mesh, int **&mesh2subset, int *&nmatscell,
    int *&matids, int *&ncellsmat, double *&Vol, double *&Density,
    double **&Volfrac, double **&Densityfrac, double **&Temperaturefrac,
    double **&Pressurefrac, float &filled_percentage);

void convert_compact_material_2_compact_cell(
    int ncells, int **subset2mesh, int **mesh2subset, int *nmatscell,
    int *matids, int *ncellsmat, double **Volfrac, double **Densityfrac,
    double **Temperaturefrac, double **Pressurefrac, int *&Cimaterial,
    int *&Cnmaterials, int *&Cimaterialfrac, int *&Cnextfrac, int *&Cfrac2cell,
    double *&CVolfrac, double *&CDensityfrac, double *&CTemperaturefrac,
    double *&CPressurefrac);
void convert_compact_cell_2_compact_material(
    int ncells, int nmats, int *Cimaterial, int *Cnmaterials,
    int *Cimaterialfrac, double *CVol, double *CDensity, double *CTemperature,
    double *CVolfrac, double *CDensityfrac, double *CTemperaturefrac,
    int **&subset2mesh, int **&mesh2subset, int *&nmatscell, int *&matids,
    int *&ncellsmat, double **&Volfrac, double **&Densityfrac,
    double **&Temperaturefrac, double **&Pressurefrac);
void print_performance_estimates(float est_perf, int64_t memops8byte,
                                 int64_t memops4byte, int64_t flops,
                                 float penalty_msecs);
