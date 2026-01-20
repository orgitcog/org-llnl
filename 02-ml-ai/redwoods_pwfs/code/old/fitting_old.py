import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u #convert to standard units
from scipy.constants import c, h, k #standard constants 
import os

def plot(a0v, vf):
    # unload data 
    a0v_x = a0v[0]
    a0v_y = a0v[1]
    vf_x = vf[0]
    vf_y = vf[1]

    fudge = 1
    plt.figure()
    plt.plot(a0v_x, fudge*a0v_y, label='a0v data', c='dodgerblue')
    plt.scatter(vf_x, vf_y, label='zero points', c='forestgreen')

    plt.title('vega spectral flux')
    plt.legend()
    plt.xlabel('λ (μm)')
    plt.ylabel('$\Phi_\lambda$ $\\left(\\frac{\\text{photons}}{\\text{cm}^2\\text{s Å}}\\right)$')
    plt.show()
    
def main():
    
    path = os.path.dirname(os.path.abspath(__file__)) 

    # opens and read data from .fits data obtained from https://cdsarc.u-strasbg.fr/cgi-bin/nph-Cat/html/max=1895?J/PASP/110/863/a0v.dat
    with fits.open(os.path.join(path,'data/J_PASP_110_863_a0v.dat.fits')) as hdul:
        data = hdul[1].data
        lam = data['lambda']*u.AA
        nflam = data['nflam']# erg per cm^2 s A
        #sdflam = data['sdnflam']# standard deviation

    # save data, units in microns and photons 
    a0v=[lam.to(u.micron), nflam]

    # ohio data https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
    lam_eff = np.array([0.36, 0.438, 0.545, 0.641, 0.798, 1.22, 1.63, 2.19])*u.micron
    lam_phi = np.array([756.1, 1392.6, 995.5, 702.0, 452.0, 193.1, 93.3, 43.6]) #photons per (cm^2 * s * A)
    vf=[lam_eff.to(u.micron), lam_phi]

    # plot data
    plot(a0v, vf)

if __name__ == '__main__':
    main()