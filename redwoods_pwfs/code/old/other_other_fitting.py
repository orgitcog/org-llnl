import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.constants import c, h, k
from scipy.optimize import curve_fit
import pandas as pd

def black_body(lam, T=9602):
    """
    Equation obtained from https://www.oceanopticsbook.info/view/light-and-radiometry/level-2/blackbody-radiation

    inputs: 
        lam, wavelength in micron
        
    ouputs:
        photon irradiance in photons/ (s * m^2 * m)
    """
    T = 14000
    lam_m = lam * 1e-6 # convert to meter
    phi = ((2 * np.pi * c) / (lam_m**4)) * (1 / (np.exp((h * c) / (lam_m * k * T)) - 1))
    return phi

def approx_blackbody(lam, T=9602):
    """
    Rayleigh-Jean Law to approximate blackbody
    https://en.wikipedia.org/wiki/Rayleigh–Jeans_law
    """
    T = 14000
    lam_m = lam * 1e-6 # convert to meter
    return (2 * c * k * T)/(lam_m**4)

def linear_fit(m, x, b):
    """
    linear model for shifting/scaling data to match zero points in ohio dataset
    """
    return m*x + b

def plot(vf, a0v_points, a0v, hi, hi_points, blackbody, rayleigh):
    plt.figure()
    plt.scatter(vf[0], vf[1], label='zero points', c='forestgreen', s=25, marker='x')
    plt.scatter(a0v_points[0], a0v_points[1]*a0v_points[2]+a0v_points[3], label='avg a0v data', s=10)
    plt.plot(a0v[0], a0v[1]*a0v[2]+a0v[3], label='speckle a0v data', c='dodgerblue', alpha=0.5)
    #plt.plot(blackbody[0], blackbody[1]*blackbody[2]+blackbody[3], label='ideal blackbody (T=14000K)', c='sienna', linestyle='dashed')
    #plt.plot(rayleigh[0], rayleigh[1]*rayleigh[2]+rayleigh[3], label = 'rayleigh approx', linestyle='dashed')
    #plt.plot(hi[0], hi[1]*hi[2]+hi[3])
    #plt.scatter(hi_points[0], hi_points[1]*hi_points[2]+hi_points[3])
    
    plt.title('vega spectral flux')
    plt.legend()
    plt.xlabel('λ (μm)')
    plt.ylabel('$\Phi_\lambda (normalized)$')
    plt.grid()
    plt.tight_layout()
    plt.show()

# ohio data https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
lam_eff = np.array([0.36, 0.438, 0.545, 0.641, 0.798, 1.22, 1.63, 2.19])  # microns
del_lam = np.array([0.06, 0.09, 0.085, 0.15, 0.15, 0.26, 0.29, 0.41])
lam_phi = np.array([756.1, 1392.6, 995.5, 702.0, 452.0, 193.1, 93.3, 43.6])
vf = [lam_eff, lam_phi, 1, 0]

# a0v data https://cdsarc.u-strasbg.fr/cgi-bin/nph-Cat/html/max=1895?J/PASP/110/863/a0v.dat
with fits.open('data/J_PASP_110_863_a0v.dat.fits') as hdul:
    data = hdul[1].data
    lam = np.array(data['lambda']) * 1e-4  # Angstrom to micron
    nflam = np.array(data['nflam'])
    a0v = [lam, nflam, 1, 0]
# splice a0v data into points to match ohio data
a0v_avg = np.zeros(len(lam_eff))
for i in range(len(lam_eff)):
    lower_bound = lam_eff[i]-del_lam[i]/2
    upper_bound = lam_eff[i]+del_lam[i]/2
    indices = np.where( (a0v[0] >= lower_bound) & (a0v[0]<=upper_bound) )
    window = np.nan_to_num(np.array([nflam[index] for index in indices]), nan=0)
    avg = np.nan_to_num(np.average(window), nan=0)
    a0v_avg[i] = avg
# filter out where there are zeros (nan)
non_zero_index = (a0v_avg != 0)
a0v_points = [lam_eff[non_zero_index], a0v_avg[non_zero_index], 1, 0]

# hawaii data https://irtfweb.ifa.hawaii.edu/~spex/IRTF_Extended_Spectral_Library/Data/Unreddened/HD074721.txt
df = pd.read_csv('./data/hawaii.txt', sep='\s+')
hi_lam = df.iloc[:, 0]
hi_flux = df.iloc[:, 1]
hi = [hi_lam, hi_flux, 1, 0]
# splice data into points to match ohio
hi_avg = np.zeros(len(lam_eff))
for i in range(len(lam_eff)):
    lower_bound = lam_eff[i]-del_lam[i]/2
    upper_bound = lam_eff[i]+del_lam[i]/2
    indices = np.where( (hi[0] >= lower_bound) & (hi[0]<=upper_bound) )
    window = np.nan_to_num(np.array([hi_flux[index] for index in indices]), nan=0)
    avg = np.nan_to_num(np.average(window), nan=0)
    hi_avg[i] = avg
# filter out where there are zeros (nan)
hi_points = [lam_eff, hi_avg, 1, 0]

# blackbody model 
blackbody_lam = np.linspace(min(np.min(lam), np.min(lam_eff)), max(np.max(lam), np.max(lam_eff)), 1000)
blackbody_phi = black_body(blackbody_lam)
blackbody_model = [blackbody_lam, blackbody_phi, 1, 0]

# Rayleigh-Jeans
rayleigh_lam = np.linspace(0.5, 2.0, 1000) # microns
rayleigh_phi = approx_blackbody(rayleigh_lam)
rayleigh_model = [rayleigh_lam, rayleigh_phi, 1, 0]

# scaling and offseting
# vf not adjusted since that is the target
a0v_points[2] = 650
a0v_points[3] = 240

a0v[2] = a0v_points[2]
a0v[3] = a0v_points[3]

hi[2] = 1e26
hi[3] = -450

hi_points[2] = hi[2]
hi_points[3] = hi[3]

blackbody_model[2] = 2.5e-31
blackbody_model[3] = 0

rayleigh_model[2] = 5e-13
rayleigh_model[3] = 100
# plot
plot(vf, a0v_points, a0v, hi, hi_points, blackbody_model, rayleigh_model)
