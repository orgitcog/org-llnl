import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.constants import c, h, k
from scipy.optimize import curve_fit

def black_body(lam):
    """
    Equation obtained from https://www.oceanopticsbook.info/view/light-and-radiometry/level-2/blackbody-radiation

    inputs: 
        lam, wavelength in micron
        
    ouputs:
        photon irradiance in photons/ (s * m^2 * m)
    """
    T=9602 # temp of vega in K
    lam_m = lam * 1e-6 # convert to meter
    phi = ((2 * np.pi * c) / (lam_m**4)) * (1 / (np.exp((h * c) / (lam_m * k * T)) - 1))
    return phi 

def plot(vf, a0v, model):
    plt.figure()
    plt.scatter(vf[0], vf[1]*vf[2]+vf[3], label='zero points', c='forestgreen', s=10)
    plt.plot(a0v[0], a0v[1]*a0v[2]+a0v[3], label='speckle a0v data', c='dodgerblue')
    plt.plot(model[0], model[1]*model[2]+model[3], label='ideal blackbody model', c='sienna', linestyle='dashed')
    
    plt.title('vega spectral flux')
    plt.legend()
    plt.xlabel('λ (μm)')
    plt.ylabel('$\Phi_\lambda (normalized)$')
    plt.show()

# ohio data https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
lam_eff = np.array([0.36, 0.438, 0.545, 0.641, 0.798, 1.22, 1.63, 2.19])  # microns
lam_phi = np.array([756.1, 1392.6, 995.5, 702.0, 452.0, 193.1, 93.3, 43.6])
vf = [lam_eff, lam_phi/np.nanmax(lam_phi), 1, 0]

# a0v data https://cdsarc.u-strasbg.fr/cgi-bin/nph-Cat/html/max=1895?J/PASP/110/863/a0v.dat
with fits.open('data/J_PASP_110_863_a0v.dat.fits') as hdul:
    data = hdul[1].data
    lam = data['lambda'] * 1e-4  # Angstrom to micron
    nflam = np.array(data['nflam'])
    a0v = [lam, nflam/np.nanmax(nflam), 1, 0]

# model
model_lam = np.linspace(min(np.min(lam), np.min(lam_eff)), max(np.max(lam), np.max(lam_eff)), 1000)
model_phi = black_body(model_lam)
model = [model_lam, model_phi/np.nanmax(model_phi), 1, 0]

# scaling and offseting
vf[2] = 1
vf[3] = .05
a0v[2] = 1
a0v[3] = .15

# plot
plot(vf, a0v, model)
