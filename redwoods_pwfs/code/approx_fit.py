import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, h, k
from scipy.optimize import curve_fit
from astropy.io import fits

def black_body(lam, T=9550, scale=1 , offset=0):
    """
    Equation obtained from https://www.oceanopticsbook.info/view/light-and-radiometry/level-2/blackbody-radiation

    inputs: 
        lam, wavelength in micron
        
    ouputs:
        photon irradiance in photons/ (s * m^2 * m)
    """
    T = 9550 # force this value. Not sure why it makes a difference to declare this inside the function but it does
    lam_m = lam * 1e-6 # convert to meter
    phi = ((2 * np.pi * c) / (lam_m**4)) * (1 / (np.exp((h * c) / (lam_m * k * T)) - 1)) * scale + offset
    return phi

# ohio data https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
lam_eff = np.array([0.36, 0.438, 0.545, 0.641, 0.798, 1.22, 1.63, 2.19])  # microns
del_lam = np.array([0.06, 0.09, 0.085, 0.15, 0.15, 0.26, 0.29, 0.41])
lam_phi = np.array([756.1, 1392.6, 995.5, 702.0, 452.0, 193.1, 93.3, 43.6])
label = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
color = ['darkviolet', 'blue', 'yellowgreen', 'red', 'firebrick', 'silver', 'darkgrey', 'dimgray']
vf = [lam_eff, lam_phi, 1, 0]

# pickles data https://cdsarc.u-strasbg.fr/viz-bin/ftp-index?J/PASP/110/863
with fits.open('data/J_PASP_110_863_a0v.dat.fits') as hdul:
    data = hdul[1].data
    lam = np.array(data['lambda']) * 1e-4  # Angstrom to micron
    nflam = np.array(data['nflam'])
    a0v = [lam, nflam, 775, 75] # manually set scaling

# optimizing ideal blackbody to zero points
guess = [9550, 1e-30, 0]
model_popt, _ = curve_fit(black_body, lam_eff, lam_phi, p0 = guess)
t = model_popt[0]
model_m = model_popt[1]
model_b = model_popt[2]
print(model_b)

# blackbody model 
blackbody_lam = np.linspace(0.0, 2.25, 1000)
blackbody_phi = black_body(blackbody_lam, T = t, scale = model_m, offset = model_b)
blackbody_model = [blackbody_lam, blackbody_phi]

# find average value
lower = 0.95
upper = 1.60
indices = np.where((blackbody_lam >= 0.95) & (blackbody_lam <= 1.60))
avg = np.average(blackbody_phi[indices])

# caption info
text = f'$\phi_\lambda$ from $\\lambda$ =  {lower} to {upper} $\mu$m is {round(avg,1)}'

# setup plot
plt.figure()
plt.grid(which='both', alpha = 0.15, c='dimgrey', zorder=0)
plt.xlim(left=0.125, right=2.25)
plt.ylim(bottom= 0, top=2000)
plt.title('vega flux zeropoints')
plt.xlabel('λ$_{eff}$ (μm)'+f'\n\n{text} ' + '$\\frac{\\text{photons}}{\\text{cm}^2\\text{s }Å}$')
plt.ylabel('$\Phi_\lambda$ $\\left( \\frac{\\text{photons}}{\\text{cm}^2\\text{s }Å} \\right)$')

# horizontal bar
for x, y, w, txt, c in zip(lam_eff, lam_phi, del_lam, label, color):
    plt.hlines(y, x - (w/2), x + (w/2), color=c, linewidth=2.5, alpha=1, zorder=60)
    plt.text(x, y + 50, f'{txt}', ha='center', va='bottom', alpha=1,
          bbox=dict(facecolor=c, edgecolor='black', boxstyle='round, pad=0.25', alpha=0.90), zorder=60)
 
#plt.hlines(avg, blackbody_lam[indices[0][0]], blackbody_lam[indices[0][-1]])

plt.plot(a0v[0], a0v[1]*a0v[2]+a0v[3], label='a0v data (Pickles et al. 1998)$^2$', c='saddlebrown',
         alpha = 0.95, linestyle='solid', linewidth = 0.95, zorder=10)   
plt.plot(blackbody_model[0], blackbody_model[1], label=(f'ideal blackbody (T={int(t)}K)'),
         alpha = 0.95, linewidth=1, c='black', linestyle='dashed', zorder=20)
plt.scatter(vf[0], vf[1], label='zero points (Bessell et al. 1998)$^3$', c='saddlebrown',
         alpha = 0.95, s=35, marker='o', zorder = 70)
plt.fill_between(blackbody_model[0][indices], blackbody_model[1][indices],
         color = 'gold', alpha = 1, zorder = 0)
plt.vlines(blackbody_lam[indices[0][0]], ymin=0,  ymax=blackbody_phi[indices[0][0]], color='k', linewidth=0.5)
plt.vlines(blackbody_lam[indices[0][-1]], ymin=0,  ymax=blackbody_phi[indices[0][-1]], color='k', linewidth=0.5)

plt.legend(framealpha=0)
plt.minorticks_on()
plt.tight_layout()
plt.savefig('./plots/blackbody.png',transparent=True, dpi=600)