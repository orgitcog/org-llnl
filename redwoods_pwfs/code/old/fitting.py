import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.constants import c, h, k # standard constants 
from scipy.optimize import curve_fit


def plot(a0v, vf, model):
    # unload data 
    model_x = model[0]
    model_y = model[1]
    a0v_x = a0v[0]
    a0v_y = a0v[1]
    vf_x = vf[0]
    vf_y = vf[1]
    
    plt.figure()
    #plt.plot(model_x, model_y, label='ideal blackbody model', c='sienna', linestyle='dashed')
    plt.plot(a0v_x, a0v_y*a0v_scale+a0v_offset, label='speckle a0v data', c='dodgerblue')
    #plt.scatter(vf_x, vf_y*vf_scale+vf_offset, label='zero points', c='forestgreen', s=10)
    
    plt.title('vega spectral flux')
    plt.legend()
    plt.xlabel('λ (μm)')
    plt.ylabel('$\Phi_\lambda (normalized)$')
    plt.show()
    
def black_body(lam, scale=1, offset=0):
    
    T=9602 #temperature of vega
    phi = ((2 * np.pi * c)/(lam**4)) * (1 / (np.exp((h*c)/(lam*k*T))-1))
    nphi = phi/np.nanmax(phi)*1000 #set peak of curve to 1000
    return phi*scale+offset

# opens and read data from .fits data obtained from 
with fits.open('data/J_PASP_110_863_a0v.dat.fits') as hdul:
    data = hdul[1].data
    lam = data['lambda']*1e-4 # data from angstrom to micron
    nflam = np.array(data['nflam']) # normalized units
    sdflam = np.array(data['sdnflam']) # standard deviation
    a0v=[lam, nflam] # save data, units in microns and photons 

# ohio data
lam_eff = np.array([0.36, 0.438, 0.545, 0.641, 0.798, 1.22, 1.63, 2.19])
lam_phi = np.array([756.1, 1392.6, 995.5, 702.0, 452.0, 193.1, 93.3, 43.6]) #photons per (cm^2 * s * A)
vf=[lam_eff, lam_phi]

# blackbody equation model
min_lam = int(min(np.min(lam), np.min(lam_eff)))
max_lam = int(max(np.max(lam), np.max(lam_eff)))
model_lam = np.linspace(min_lam, max_lam, 10**3)
model_phi = np.array(black_body(model_lam*1e-6))
model = [model_lam, model_phi]

# fit data to ohio zero points
# initial guess
#scale_model = 1#1e-30
#offset_model = 0#-250

# optimizing
popt, _ = curve_fit(black_body, a0v[0], a0v[1])
a0v_scale = popt[0]
a0v_offset = popt[1]

popt, _ = curve_fit(black_body, vf[0], vf[1])
vf_scale= popt[0]
vf_offset = popt[1]

# plot
plot(a0v, vf, model)
