import numpy as np
import scipy.sparse as sparse
import pyamg.krylov as solver
import matplotlib.pyplot as plt

from fd_matrix import conv_diff_rxn_matrix
import complex_block_jacobi as cbj
from complex_block_jacobi import ComplexPreconditioner

from aihwkit.simulator.presets import IdealizedPreset

def main():
    ## PARAMETER SETUP ##
    m = 50 # number of finite difference points in one dimension
    diag_shift = 0.0
    num_blocks = 5 # number of blocks to use in analog block Jacobi preconditioning
    tol = 1e-10
    maxit = 400

    # Load IdealizedPreset and match parameters to MATLAB simulator RPU_Analog_Basic.Get_Baseline_Settings
    analog_config = cbj.sisc_config(IdealizedPreset())
    digital_config = cbj.float_config()

    ## SET UP PROBLEM AND PRECONDITIONER ##
    A = conv_diff_rxn_matrix(m, 0.0, 0.0, diag_shift)
    n = A.shape[0]

    b = np.random.normal(size=(n,))
    b = b / np.linalg.norm(b) # random right-hand side vector with norm 1
    b_complex = b.astype("complex128")
    x0 = np.zeros(n)
    x0_complex = np.zeros(n).astype("complex128")

    # Set up preconditioners and anonymous functions for the SciPy linear operators
    DP_info0 = ComplexPreconditioner(A, num_blocks, digital_config, False, False, 0.0, 0.0)

    AP_info0 = ComplexPreconditioner(A, num_blocks, analog_config, True, False, 0.0, 0.0)
    AP_info1 = ComplexPreconditioner(A, num_blocks, analog_config, True, False, 0.0, 0.2)
    AP_info2 = ComplexPreconditioner(A, num_blocks, analog_config, True, False, 0.2 + 0.5j, 0.0)

    DP0 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: DP_info0.apply(u)), dtype="complex128")

    AP0 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: AP_info0.apply(u)), dtype="complex128")
    AP1 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: AP_info1.apply(u)), dtype="complex128")
    AP2 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: AP_info2.apply(u)), dtype="complex128")

    ## RUN FGMRES ##

    n_resvec0 = []
    d_resvec0 = []
    a_resvec0, a_resvec1, a_resvec2 = [], [], []

    n_x0, n_flag0 = solver.fgmres(A, b, x0, tol=tol, restart=None, maxiter=maxit, M=None, residuals=n_resvec0)

    d_x0, d_flag0 = solver.fgmres(A, b_complex, x0_complex, tol=tol, restart=None, maxiter=maxit, M=DP0, residuals=d_resvec0)

    a_x0, a_flag0 = solver.fgmres(A, b_complex, x0_complex, tol=tol, restart=None, maxiter=maxit, M=AP0, residuals=a_resvec0)
    a_x1, a_flag1 = solver.fgmres(A, b_complex, x0_complex, tol=tol, restart=None, maxiter=maxit, M=AP1, residuals=a_resvec1)
    a_x2, a_flag2 = solver.fgmres(A, b_complex, x0_complex, tol=tol, restart=None, maxiter=maxit, M=AP2, residuals=a_resvec2)

    ## VISUALIZATION ##

    plt.semilogy(n_resvec0, '-k', linewidth=1.2, label="FGMRES (no precond.)")

    plt.semilogy(d_resvec0, '--r', linewidth=1.2, label="FGMRES (digital)")

    plt.semilogy(a_resvec0, '-r', linewidth=1.2, label="FGMRES (analog, shift 1)")
    plt.semilogy(a_resvec1, '-g', linewidth=1.2, label="FGMRES (analog, shift 2)")
    plt.semilogy(a_resvec2, '-b', linewidth=1.2, label="FGMRES (analog, shift 3)")

    plt.xlabel("Iteration number")
    plt.ylabel("Relative residual norm")
    # plt.title("$A = -\\nabla^2 + c \\cdot \\nabla + \\alpha I + XX^T$, n = %i, d = %i, $\\alpha$ = %.2f" % (n, d, diag_shift))
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.savefig("fgmres_cbj_comparison.pdf", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    main()