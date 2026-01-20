import numpy as np

def conv_diff_rxn_matrix(m, cx, cy, r):
    # Consider the PDE: -Lap(u) + (cx, cy) \dot \grad(u) + ru = g(x, y), with h = dx = dy = 1
    # Use a five-point stencil for the 2D Laplacian: kron(L, I) + kron(I, L), where L is the 1D stencil for u''
    # Without using upwinding for convection: cx kron(I, C) + cy kron(C, I), where C is the 1D stencil for u'
    # Add a diagonal shift corresponding to reaction: r I

    C_onedim = np.diag(np.ones(m - 1), 1) + np.diag(-1 * np.ones(m - 1), -1)
    L_onedim = np.diag(2 * np.ones(m), 0) + np.diag(-1 * np.ones(m - 1), 1) + np.diag(-1 * np.ones(m - 1), -1)
    I_onedim = np.eye(m)

    C_xdim = cx * np.kron(I_onedim, C_onedim)
    C_ydim = cy * np.kron(C_onedim, I_onedim)
    L_twodim = np.kron(L_onedim, I_onedim) + np.kron(I_onedim, L_onedim)
    R_twodim = r * np.eye(m**2)

    return L_twodim + C_xdim + C_ydim + R_twodim