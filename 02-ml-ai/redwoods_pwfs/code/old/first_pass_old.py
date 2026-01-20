import matplotlib.pyplot as plt
from matplotlib import cm
from func import *
import numpy as np

read_noise_level=30
wav0=1.275e-6 # center wave between .95 and 1.6
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

# unobsured aperature
u_aperture=np.zeros((imagepix,imagepix)) 
u_aperture[np.where(xy_dh<pupilpix/2)]=1

pwfs=np.zeros(aperture.shape)
ind1=np.where(np.logical_and(rphi>=0,rphi<2*np.pi/3))
ind2=np.where(np.logical_and(rphi>=2*np.pi/3,rphi<2*np.pi*2/3))
ind3=np.where(np.logical_and(rphi>=2*np.pi*2/3,rphi<2*np.pi))
fudge=3
ff=-0.6 #random fudge factor I need to get the right pupil centering...
g1,g2,g3=(xtlt-ytlt)*np.sqrt(2)-ff*ytlt, ytlt-ff*ytlt, (-xtlt-ytlt)*np.sqrt(2)-ff*ytlt
tt1,tt2,tt3=(g1*fudge),(g2*fudge),(g3*fudge)
#tt1,tt2,tt3=(xtlt*fudge1),((xtlt)*fudge2),((xtlt)*fudge3)
pwfs[ind1]=tt1[ind1]#-np.mean((tt1)[ind1])
pwfs[ind2]=tt2[ind2]#-np.mean((tt2)[ind2])
pwfs[ind3]=tt3[ind3]#-np.mean((tt3)[ind3])

wavefronterror = 0.1e-6/wav0*2*np.pi # um rms wavefront error
im_tar_phase = make_noise_pl(wavefronterror,imagepix,pupilpix,-2,16)

Dtel = 3
bandpass = (1.6 - 0.95) / wav0 #bandpass in fraction 
percent_bandpass = 0.5 #actual bandpass in fraction
f0=192.9*(wav0*(percent_bandpass/bandpass)*1e6)*1e8 #photons/m**2/s, mag 0 star, 50 % bandpass; https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html

m_object=3 # higher number means dimmer object

flux_object_ini=f0*10.**(-m_object/2.5)
tr_atm,th,qe=0.9,0.2,0.6 #assume transmission through the atmosphere, instrument throughput, quantum efficiency of CCD
flux_object=flux_object_ini*tr_atm*th*qe
t_int=1e-3 # exposure time
Nphot=flux_object*t_int*np.pi*(Dtel/2.)**2 

no_phase_offset=np.zeros((imagepix,imagepix)) #no noise

def propagate(ph, amp, phn = False):
    pupil_wavefront_dm_unnorm=amp*np.exp(1j*ph) 
    norm_phot=np.sum(intensity(np.ones((imagepix,imagepix))[indpup]))
    pupil_wavefront_dm=complex_amplitude(np.sqrt(intensity(pupil_wavefront_dm_unnorm)/norm_phot*Nphot),phase(pupil_wavefront_dm_unnorm))

    fpm_wavefront_ini=np.fft.fftshift(pupil_to_image(pupil_wavefront_dm)) #ft to image plane
    fpm_wavefront=complex_amplitude(np.abs(fpm_wavefront_ini),np.angle(fpm_wavefront_ini)+pwfs) #add phase mask
    pup_wf=image_to_pupil(fpm_wavefront) #ift to pupil plane
    im=intensity(pup_wf)
    if (phn == True):
        return np.random.poisson(im)
    else:
        return im


# figures to be plotted
plots = [propagate(no_phase_offset, u_aperture), 
         propagate(im_tar_phase, aperture, phn=True), 
         propagate(no_phase_offset, u_aperture)-propagate(im_tar_phase, aperture, phn=True)]

# add read noise
read_noise=np.random.poisson(np.ones(aperture.shape)*read_noise_level**2)
plots[1]= plots[1] + read_noise - np.mean(read_noise) #add bias from read noise for apples to apples comparison
#imn=np.random.poisson(im)+read_noise-np.mean(read_noise)

# plotting
headings = ['no noise', 'noise', 'diff']
fig, ax = plt.subplots(nrows=1, ncols=3)
[axs.set_axis_off() for axs in ax]
for i in range(3):
    im1 = ax[i].imshow(plots[i])#, cmap = cm.Greys_r, vmin = 0, vmax=1e-6)
    ax[i].set_title(headings[i])


