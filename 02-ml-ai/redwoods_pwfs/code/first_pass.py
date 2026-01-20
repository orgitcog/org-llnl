import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib
import numpy as np
import itertools
from numpy.fft import fftshift
from scipy.ndimage.interpolation import rotate
import itertools
import math
from scipy.io import savemat
from func import *
import datetime
from matplotlib import rc, cm
import matplotlib.animation as animation
import sys

read_noise_level=30
wav0=1.275e-6 # NOT SURE WHAT WAVELENGTH TO USE, DEFAULT WAS 0.95 WHICH IS AT THE BOTTOM OF BANDPASS
imagepix=128
pupilpix=29
beam_ratio=imagepix/pupilpix
grid=np.mgrid[0:imagepix,0:imagepix]
xcen,ycen=imagepix/2,imagepix/2
xy_dh=np.sqrt((grid[1]-imagepix/2.)**2.+(grid[0]-imagepix/2.)**2.)
aperture=np.zeros((imagepix,imagepix))
p_obs=0.3
aperture[np.where(np.logical_and(xy_dh>p_obs*pupilpix/2,xy_dh<pupilpix/2))]=1. #obscured aperture
indpup=np.where(aperture==1.)
rho,phi=polar_grid(imagepix,pupilpix)
rphi=-phi+np.pi
xtlt=(grid[1]-imagepix/2+0.5)/beam_ratio
ytlt=(grid[0]-imagepix/2+0.5)/beam_ratio

pwfs=np.zeros(aperture.shape)
ind1=np.where(np.logical_and(rphi>=0,rphi<2*np.pi/3))
ind2=np.where(np.logical_and(rphi>=2*np.pi/3,rphi<2*np.pi*2/3))
ind3=np.where(np.logical_and(rphi>=2*np.pi*2/3,rphi<2*np.pi))
fudge=3.
#fudge=4.5
ff=-0.6 #random fudge factor I need to get the right pupil centering...
g1,g2,g3=(xtlt-ytlt)*np.sqrt(2)-ff*ytlt,ytlt-ff*ytlt,(-xtlt-ytlt)*np.sqrt(2)-ff*ytlt
tt1,tt2,tt3=(g1*fudge),(g2*fudge),(g3*fudge)
#tt1,tt2,tt3=(xtlt*fudge1),((xtlt)*fudge2),((xtlt)*fudge3)
pwfs[ind1]=tt1[ind1]#-np.mean((tt1)[ind1])
pwfs[ind2]=tt2[ind2]#-np.mean((tt2)[ind2])
pwfs[ind3]=tt3[ind3]#-np.mean((tt3)[ind3])

wavefronterror=100e-9/wav0*2*np.pi #m rms wavefront error
im_tar_phase=make_noise_pl(wavefronterror,imagepix,pupilpix,-2,16)

Dtel=3
bandpass= 0.65/1.275 #0.26/1.22 #bandpass in fraction
percent_bandpass=0.5 #actual bandpass in fraction
f0=192.9*(wav0*(percent_bandpass/bandpass)*1e6)*10**8. #photons/m**2/s, mag 0 star, 50 % bandpass; https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html

m_object = 6

flux_object_ini=f0*10.**(-m_object/2.5)
tr_atm,th,qe=0.3,0.2,0.6 #assume transmission through the atmosphere, instrument throughput, quantum efficiency of CCD
flux_object=flux_object_ini*tr_atm*th*qe
t_int=1e-3
Nphot=flux_object*t_int*np.pi*(Dtel/2.)**2. 

def propagate(im_tar_phase, aperture, phn = False):
    amp=0.15
    pupil_wavefront_dm_unnorm=aperture*np.exp(1j*im_tar_phase*amp)
    norm_phot=np.sum(intensity(np.ones((imagepix,imagepix))[indpup]))
    pupil_wavefront_dm=complex_amplitude(np.sqrt(intensity(pupil_wavefront_dm_unnorm)/norm_phot*Nphot),phase(pupil_wavefront_dm_unnorm))

    fpm_wavefront_ini=np.fft.fftshift(pupil_to_image(pupil_wavefront_dm)) #ft to image plane
    fpm_wavefront=complex_amplitude(np.abs(fpm_wavefront_ini),np.angle(fpm_wavefront_ini)+pwfs) #add phase mask
    pup_wf=image_to_pupil(fpm_wavefront) #ift to pupil plane
    im=intensity(pup_wf)
    if (phn == True):
        noisy = np.random.poisson(im)
        return noisy - np.mean(noisy)
    else:
        return im

# unobsured aperature
#u_aperture=np.zeros((imagepix,imagepix)) 
#u_aperture[np.where(xy_dh<pupilpix/2)]=1

# no wavefront error
no_phase_offset=make_noise_pl(0,imagepix,pupilpix,-2,16)

# add read noise
read_noise= np.random.poisson(np.ones(aperture.shape)*read_noise_level**2)

# set up plots
im = propagate(im_tar_phase, aperture)
imn = read_noise -np.mean(read_noise)#- 1050
im_phn = propagate(im_tar_phase, aperture, phn=True)

diff = imn - im
diff_phn = im_phn - im
both = imn + im_phn
diff_both = both - im

plots = np.array([[im, imn, diff], [im, diff_phn, im_phn], [im, both, diff_both]])
headings = np.array([['no noise', 'read noise', 'difference'],
                    ['no noise', 'light noise', 'difference'],
                    ['no noise', 'both noise', 'difference']])

#plot
row, col, _ , _ = plots.shape
fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(10, 8))
for j in range(row):
    for i in range(col):
        im_i = ax[j][i].imshow(plots[j][i])
        fig.colorbar(im_i, shrink=1)
        ax[j][i].set_title(headings[j][i])
[axs.set_axis_off() for axs in ax.ravel()] # remove axes
fig.suptitle(f'M: {m_object}\nt_int: {t_int}s')
plt.tight_layout()
plt.savefig('./plots/first_pass.png', dpi = 600)
plt.show()
#plt.savefig('./plots/first_pass.png', dpi = 1200)

'''
Observations 
    No Photon Noise                     With Photon Noise            
        Read Noise = default                Read Noise = default
            1ms exposure, m = 6             1ms exposure, m = 13
            10ms exposure, m = 9                10ms exposure, m = 16
        Read Noise = 1                      Read Noise = 1
            1ms exposure, m = 10                1ms exposure, m = 13
            10ms exposure, m = 13               10ms exposure, m = 15
'''