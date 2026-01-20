#include <iostream>
#include <vector>
#include <string>

#include "../common/linear_solvers.hpp"
#include "../common/blas.hpp"
#include "../common/io.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    // Check if the correct number of arguments is provided
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix file>" << std::endl;
        return 1;
    }

    string filename = "./matrix.csv";
    filename = argv[1];

    cout << "Loading matrix A:" << endl;
    vector<vector<double>> A = load_matrix_from_csv(filename);
    print_matrix(A);
    cout << "-----------------------------------------------------------" << endl;

    // Solve linear system Ax = 1
    vector<double> b(A.size(), 1.0);
    auto x = solve_system_with_LU(A, b);

    // Print solution
    cout << "Solution x:" << endl;
    for (size_t i = 0; i < x.size(); ++i)
        cout << "x[" << i << "]: " << x[i] << endl;
    cout << "-----------------------------------------------------------" << endl;

    // Residual
    auto a = multiply_matrix_vector(A, x);  // calculate Ax
    auto residual = subtract_vectors(a, b); // calculate Ax - b
    auto norm = l2_norm(residual);          // calculate ||Ax - b||
    cout << "Residual norm ||Ax - b||: " << norm << endl;

    /*
    cout << "-----------------------------------------------------------" << endl;
    cout << "Testing matrix loading from CSV file" << endl;
    cout << "-----------------------------------------------------------" << endl;
    // Load matrix from CSV file
    const std::string &filename = "matrix.csv";
    vector<vector<double>> A = load_matrix_from_csv(filename);
    print_matrix(A);

    // Solve inear system Ax = 1
    vector<double> b(A.size(), 1.0);
    // vector<double> x = solve_system_with_LU(A, b);
    auto [L, U, P] = lu_factorization_partial_pivot(A);
    cout << "Matrix L:" << endl;
    print_matrix(L);
    save_matrix_to_csv(L, "L_matrix.csv");
    cout << "Matrix U:" << endl;
    print_matrix(U);
    save_matrix_to_csv(U, "U_matrix.csv");
    cout << "Matrix P:" << endl;
    print_matrix(P);
    save_matrix_to_csv(P, "P_matrix.csv");

    vector<double> x = solve_lu_system_by_substitution(L, U, P, b);
    for (size_t i = 0; i < x.size(); ++i)
        cout << "x[" << i << "]: " << x[i] << endl;

    // Residual
    auto a = multiply_matrix_vector(A, x);
    auto residual = subtract_vectors(a, b);
    auto norm = l2_norm(residual);
    cout << "Residual norm: " << norm << endl;

    cout << "-----------------------------------------------------------" << endl;
    cout << "Loading Bad U matrix" << endl;
    vector<vector<double>> U_loaded = load_matrix_from_csv("U_bad_matrix.csv");
    cout << "Matrix U (bad) loaded from CSV file:" << endl;
    x = solve_lu_system_by_substitution(L, U_loaded, P, b);
    for (size_t i = 0; i < x.size(); ++i)
        cout << "x[" << i << "]: " << x[i] << endl;

    cout << "\nReconstruct A" << endl;
    auto P_T = transpose_matrix(P);
    auto L_P_T = matrix_multiply(P_T, L);
    auto A_reconstructed = matrix_multiply(L_P_T, U_loaded);
    cout << "Matrix A reconstructed:" << endl;
    print_matrix(A_reconstructed);

    vector<vector<double>> difference = subtract_matrices(A, A_reconstructed);
    norm = frobenius_norm(difference);
    cout << ">>> Norm of (A - A_bad): " << norm << endl;

    cout << "Solve system with bad A" << endl;
    x = solve_system_with_LU(A_reconstructed, b);
    for (size_t i = 0; i < x.size(); ++i)
        cout << "x[" << i << "]: " << x[i] << endl;
    // Residual
    auto a_reconstructed = multiply_matrix_vector(A_reconstructed, x);
    auto residual_reconstructed = subtract_vectors(a_reconstructed, b);
    auto norm_reconstructed = l2_norm(residual_reconstructed);
    cout << "Residual norm: " << norm_reconstructed << endl;
    save_matrix_to_csv(A_reconstructed, "bad_matrix.csv");

    cout << "-----------------------------------------------------------" << endl;
    */

    return 0;
}