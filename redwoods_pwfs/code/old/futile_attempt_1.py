import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d

# Data
lam_eff = np.array([0.438, 0.545, 0.641, 0.798, 1.22, 1.63, 2.19])  # microns
lam_phi = np.array([1392.6, 995.5, 702.0, 452.0, 193.1, 93.3, 43.6])

with fits.open('data/J_PASP_110_863_a0v.dat.fits') as hdul:
    data = hdul[1].data
    lam = np.array(data['lambda']) * 1e-4  # Angstrom to micron
    nflam = np.array(data['nflam'])
    a0v = [lam, nflam, 800, 100] # manually set scaling

# Interpolate Pickles at filter wavelengths
interp_flux = interp1d(a0v[0], a0v[1]*a0v[2]+a0v[3], bounds_error=False, fill_value='extrapolate')
pickles_at_filters = interp_flux(lam_eff)

# Linear least squares fit: scale and offset
A = np.vstack([pickles_at_filters, np.ones_like(pickles_at_filters)]).T
solution, residuals, rank, s = np.linalg.lstsq(A, lam_phi, rcond=None)
scale, offset = solution

# Apply to full Pickles spectrum
scaled_pickles = a0v[1]*a0v[2] + a0v[3]
scaled_pickles = scale * scaled_pickles + offset

# Plot
plt.figure()
plt.grid(which='both', alpha=0.25)
plt.xlim(left=0.1, right=2)
plt.ylim(bottom=0, top=2000)
plt.title('Vega Flux Zeropoints')
plt.xlabel('λ$_{eff}$ (μm)' + '$\\frac{\\text{photons}}{\\text{cm}^2\\text{s }Å}$')
plt.ylabel('$\Phi_\lambda$ $\\left( \\frac{\\text{photons}}{\\text{cm}^2\\text{s }Å} \\right)$')
plt.plot(a0v[0], scaled_pickles, label='Pickles (scaled+offset)', c='saddlebrown',
         alpha=0.95, linestyle='solid', linewidth=0.95)
plt.scatter(lam_eff, lam_phi, label='zero points (Bessell et al. 1998)', c='saddlebrown',
         alpha=0.95, s=35, marker='o')
plt.scatter([0.36], [756.1], c='saddlebrown',
         alpha=0.95, s=35, marker='o')
plt.legend()
plt.minorticks_on()
plt.tight_layout()
plt.show()