import numpy as np
import scipy.sparse as sparse
import pyamg.krylov as solver
import matplotlib.pyplot as plt

from fd_matrix import conv_diff_rxn_matrix
import analog_block_jacobi as abj
from analog_block_jacobi import ABJPreconditioner
from block_jacobi import BlockJacobiPreconditioner

from aihwkit.simulator.presets import ReRamSBPreset
from aihwkit.simulator.presets import IdealizedPreset

def main():
    ## PARAMETER SETUP ##
    m = 32 # number of finite difference points in one dimension
    d = 10 # number of vectors for the low-rank update
    d_sparsity = 0.10 # sparsity of the low-rank update vectors
    low_rank_coeff = 1.0
    diag_shift = 0.0
    num_blocks = [1, 4, 16, 64, m**2] # number of blocks to use in analog block Jacobi preconditioning
    tol = 1e-12
    maxit = 200

    # Load IdealizedPreset and match parameters to MATLAB simulator RPU_Analog_Basic.Get_Baseline_Settings
    rpu_config = abj.matlab_config(IdealizedPreset())

    ## SET UP PROBLEM AND PRECONDITIONER ##
    A = conv_diff_rxn_matrix(m, 0.0, 0.0, diag_shift)
    n = A.shape[0]

    X = sparse.random(n, d, density=d_sparsity)
    X.data = np.random.randn(X.nnz)
    X = X.toarray()
    A = A + low_rank_coeff*np.dot(X, X.T)
    D = np.diag(np.diag(A)) # needed for digital Jacobi preconditioning

    b = np.random.normal(size=(n,))
    b = b / np.linalg.norm(b) # random right-hand side vector with norm 1

    # Set up preconditioners and anonymous functions for the SciPy linear operators
    G_info0 = BlockJacobiPreconditioner(A, num_blocks[0])
    G_info1 = BlockJacobiPreconditioner(A, num_blocks[1])
    G_info2 = BlockJacobiPreconditioner(A, num_blocks[2])
    G_info3 = BlockJacobiPreconditioner(A, num_blocks[3])
    G_info4 = BlockJacobiPreconditioner(A, num_blocks[4])

    F_info0 = ABJPreconditioner(A, num_blocks[0], rpu_config)
    F_info2 = ABJPreconditioner(A, num_blocks[2], rpu_config)
    F_info1 = ABJPreconditioner(A, num_blocks[1], rpu_config)
    F_info3 = ABJPreconditioner(A, num_blocks[3], rpu_config)
    F_info4 = ABJPreconditioner(A, num_blocks[4], rpu_config)

    G0 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: G_info0.apply(u)))
    G1 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: G_info1.apply(u)))
    G2 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: G_info2.apply(u)))
    G3 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: G_info3.apply(u)))
    G4 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: G_info4.apply(u)))

    F0 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: F_info0.apply(u)))
    F1 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: F_info1.apply(u)))
    F2 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: F_info2.apply(u)))
    F3 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: F_info3.apply(u)))
    F4 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: F_info4.apply(u)))

    ## RUN GMRES AND FGMRES ##

    n_resvec0, d_resvec0 = [], []
    g_resvec0, g_resvec1, g_resvec2, g_resvec3, g_resvec4 = [], [], [], [], []
    f_resvec0, f_resvec1, f_resvec2, f_resvec3, f_resvec4 = [], [], [], [], []

    n_x0, n_flag0 =  solver.gmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=None, residuals=n_resvec0)
    d_x0, d_flag0 =  solver.gmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=np.linalg.inv(D), residuals=d_resvec0)

    g_x0, g_flag0 =  solver.gmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=G0, residuals=g_resvec0)
    g_x1, g_flag1 =  solver.gmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=G1, residuals=g_resvec1)
    g_x2, g_flag2 =  solver.gmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=G2, residuals=g_resvec2)
    g_x3, g_flag3 =  solver.gmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=G3, residuals=g_resvec3)
    g_x4, g_flag4 =  solver.gmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=G4, residuals=g_resvec4)

    f_x0, f_flag0 = solver.fgmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=F0, residuals=f_resvec0)
    f_x1, f_flag1 = solver.fgmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=F1, residuals=f_resvec1)
    f_x2, f_flag2 = solver.fgmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=F2, residuals=f_resvec2)
    f_x3, f_flag3 = solver.fgmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=F3, residuals=f_resvec3)
    f_x4, f_flag4 = solver.fgmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=F4, residuals=f_resvec4)

    ## VISUALIZATION ##

    plt.semilogy(n_resvec0, '-k', linewidth=1.2, label="GMRES (no precond.)")
    # plt.semilogy(d_resvec0, '.k', linewidth=1.2, label="GMRES (Jacobi)")

    plt.semilogy(g_resvec0, '--r', linewidth=1.2, label="GMRES (BJ%i)" % (num_blocks[0]))
    plt.semilogy(g_resvec1, '--g', linewidth=1.2, label="GMRES (BJ%i)" % (num_blocks[1]))
    plt.semilogy(g_resvec2, '--c', linewidth=1.2, label="GMRES (BJ%i)" % (num_blocks[2]))
    plt.semilogy(g_resvec3, '--b', linewidth=1.2, label="GMRES (BJ%i)" % (num_blocks[3]))
    plt.semilogy(g_resvec4, '--m', linewidth=1.2, label="GMRES (BJ%i)" % (num_blocks[4]))

    plt.semilogy(f_resvec0, '-r', linewidth=1.2, label="FGMRES (ABJ%i)" % (num_blocks[0]))
    plt.semilogy(f_resvec1, '-g', linewidth=1.2, label="FGMRES (ABJ%i)" % (num_blocks[1]))
    plt.semilogy(f_resvec2, '-c', linewidth=1.2, label="FGMRES (ABJ%i)" % (num_blocks[2]))
    plt.semilogy(f_resvec3, '-b', linewidth=1.2, label="FGMRES (ABJ%i)" % (num_blocks[3]))
    plt.semilogy(f_resvec4, '-m', linewidth=1.2, label="FGMRES (ABJ%i)" % (num_blocks[4]))

    plt.xlabel("Iteration number")
    plt.ylabel("Relative residual norm")
    plt.title("$A = -\\nabla^2 + c \\cdot \\nabla + \\alpha I + XX^T$, n = %i, d = %i, $\\alpha$ = %.2f" % (n, d, diag_shift))
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.savefig("fgmres_abj_comparison.pdf", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    main()