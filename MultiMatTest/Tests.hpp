#pragma once

void single_material(const int ncells, const bool memory_verbose,
                     const int itermax, const double nmatconst);
void cell_dominant_full_matrix(const int ncells, const bool memory_verbose,
                               const int itermax, const double nmatconst,
                               const int method, const int nmats,
                               const float L_f, const int nnbrs_ave,
                               const double *nmatconsts, const double *cen_x,
                               const double *cen_y, const int *nnbrs,
                               int **nbrs);

void material_dominant_matrix(const int ncells, const bool memory_verbose,
                              const int itermax, const double nmatconst,
                              const int method, const int nmats,
                              const float L_f, const int nnbrs_ave,
                              const double *nmatconsts, const double *cen_x,
                              const double *cen_y, const int *nnbrs,
                              int **nbrs);

void cell_dominant_compact(const int ncells, const bool memory_verbose,
                           const int itermax, const double nmatconst,
                           const int method, const int nmats, const float L_f,
                           const int nnbrs_ave, const double *nmatconsts,
                           const double *cen_x, const double *cen_y,
                           const int *nnbrs, int **nbrs);

void material_centric_compact(const int ncells, const bool memory_verbose,
                              const int itermax, const double nmatconst,
                              const int method, const int nmats,
                              const float L_f, const int nnbrs_ave,
                              const double *nmatconsts, const double *cen_x,
                              const double *cen_y, const int *nnbrs,
                              int **nbrs);
