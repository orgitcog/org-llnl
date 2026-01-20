"""
Interpolative Decomposition (ID) module for computing low-rank approximations of matrices.

This module implements the Interpolative Decomposition algorithm used for
neural network pruning in the PruningAMR system. ID decomposes a matrix A
into a product of a subset of columns and an interpolation matrix, enabling
efficient low-rank approximations.

Key functions:
- ID(): Main interpolative decomposition function
- torch_solve(): PyTorch-based linear system solver for ID computation

The ID algorithm is used in PruningAMR to identify which neurons can be
removed while maintaining approximation accuracy within a specified tolerance.
"""

from scipy.linalg import qr
import numpy as np
import torch


def torch_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve linear system Ax = b using PyTorch for thread safe computation.
    
    Uses PyTorch's linear algebra solver which is thread safe, unlike
	 NumPy's solver. Handles both vector and matrix right-hand sides.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n,) or matrix (n x m)
        
    Returns:
        np.ndarray: Solution vector or matrix
    """
    # ensure contiguous arrays; preserve dtype (fp32 or fp64)
    A = np.ascontiguousarray(A)
    b = np.ascontiguousarray(b)

    A_t = torch.from_numpy(A)
    b_t = torch.from_numpy(b)

    # allow vector or matrix RHS
    if b_t.ndim == 1:
        x_t = torch.linalg.solve(A_t, b_t.unsqueeze(-1)).squeeze(-1)
    else:
        x_t = torch.linalg.solve(A_t, b_t)

    return x_t.cpu().numpy()

def ID(A: torch.Tensor, 
       epsilon: float = 0.01, 
       debugging: bool = False) -> tuple:
    """
    Compute the interpolative decomposition (ID) of matrix A.
    
    Decomposes matrix A into A ≈ A[:,P1] * T where:
    - A[:,P1] contains a subset of columns from A (specified by indices P1)
    - T is an interpolation matrix that reconstructs the original columns
    
    The algorithm uses rank-revealing QR factorization to determine which
    columns are most important for maintaining approximation accuracy.
    
    Args:
        A: Input matrix to decompose (n x m)
        epsilon: Maximum relative error tolerance for decomposition
        debugging: If True, prints accuracy information
        
    Returns:
        tuple: (P1, T, k) where:
            - P1: Indices of selected columns
            - T: Interpolation matrix (k x m)
            - k: Rank of the approximation
    """

	n, m = A.shape

	# compute rank-revealing QR factorization (this is O(mnk) for A of size mxn)
	Q, R, P = qr(A.detach().numpy(), pivoting = True, mode = 'economic')
 
	# determine rank k from epsilon tolerance
	k = R.shape[0]
	while np.abs(R[k-1, k-1]/R[0,0]) < epsilon and k-1 > 0:
		k -= 1

	# extract submatrices from QR factorization
	R11 = R[:k, :k]  # upper triangular part
	R12 = R[:k, k:]  # remaining columns

	# select k most important columns based on QR pivoting
	P1 = P[:k] # this is the pivoting order that can get the correct columns of A

	# construct interpolation matrix T
	T        = np.zeros((k, m))
	T[:, :k] = np.identity(k)  # identity for selected columns

	# solve for interpolation coefficients of remaining columns
	T[:, k:] = torch_solve(R11, R12)
	
	# apply column permutation to finalize T
	P        = np.identity(m)[:, P]
	T        = np.matmul(T, np.transpose(P)) 

	if debugging:
		# verify that the ID decomposition meets the epsilon accuracy requirement
		A_approx = np.matmul(A.detach().numpy()[:, P1], T)
		error = np.linalg.norm(A.detach().numpy() - A_approx, ord = 2) / np.linalg.norm(A.detach().numpy(), ord = 2)
		print("\n******* ID Accuracy Check *******")
		print("Requested epsilon = {}".format(epsilon))
		print("Actual error      = {}".format(error))
		print("Accuracy achieved: {}".format("YES" if error <= epsilon else "NO"))

	return P1, torch.tensor(T, dtype = torch.float32), k
