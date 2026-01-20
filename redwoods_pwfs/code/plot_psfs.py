'''
plot psfs for a variety of phase and amplitude aberrations

'''

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc, cm
import os

#utility functions to use throughout the simulation
import func as functions

def get_path():
	'''
	vscode was not opening the .fits file 
	added this to increase compatibility with other IDEs/OSs
	'''
	return os.path.dirname(os.path.abspath(__file__))

path = get_path()
wav0=1.65e-6 #assumed wav0 for sine amplitude input in meters

imagepix=2048
beam_ratio=4
pupilpix=imagepix/beam_ratio

grid=np.mgrid[0:imagepix,0:imagepix]
xcen,ycen=imagepix/2,imagepix/2
xy_dh=np.sqrt((grid[1]-imagepix/2.)**2.+(grid[0]-imagepix/2.)**2.)

aperture=np.zeros((imagepix,imagepix))
aperture[np.where(xy_dh<pupilpix/2)]=1. #unobscured aperture

no_phase_offset=np.zeros((imagepix,imagepix))

tmtpup=np.pad(fits.getdata(os.path.join(path, 'data/TMT-M1-Melco.fits')),512,mode='constant',constant_values=(0,))
tmtpup=tmtpup/np.max(tmtpup)

#obscured aperture
pobs=0.2
aperture2=np.zeros(aperture.shape)
aperture2[np.where(xy_dh<pupilpix/2)]=1.
aperture2[np.where(xy_dh<pobs*pupilpix/2)]=0.

rho,phi=functions.polar_grid(imagepix,pupilpix)

def propagate(pupil_phase_dm,amp):
	pupil_wavefront_dm=amp*np.exp(1j*(pupil_phase_dm)) #initial 
	norm=np.max(functions.intensity(np.fft.fftshift(functions.pupil_to_image(aperture*np.exp(1j*(no_phase_offset)))))) #different definition of 
	im=functions.intensity(np.fft.fftshift(functions.pupil_to_image(pupil_wavefront_dm)))/norm
	return im

amp=[aperture,aperture2,tmtpup,aperture,aperture,aperture,aperture*(1./3.*np.cos(2.*np.pi*3.*imagepix/pupilpix*grid[0]/grid[0][-1,0])+(1-1./3.)),aperture*functions.make_amp_err(0.01,imagepix,pupilpix),aperture]
ph=[no_phase_offset,no_phase_offset,no_phase_offset,3.*functions.zernike(2,-2,rho,phi),4.*functions.zernike(3,-1,rho,phi),np.cos(2.*np.pi*3.*imagepix/pupilpix*grid[0]/grid[0][-1,0]),no_phase_offset,functions.make_noise_pl(100e-9,imagepix,pupilpix,wav0,-2),functions.make_noise_pl(10e-6,imagepix,pupilpix,wav0,-11/3)]

size=20
font = {'family' : 'Times New Roman',
        'size'   : size}

mpl.rc('font', **font)
mpl.rcParams['image.interpolation'] = 'nearest'


fig,ax=plt.subplots(nrows=3,ncols=9,figsize=(16,6))
[axi.set_axis_off() for axi in ax.ravel()]
plt.subplots_adjust(wspace=0.1,hspace=0.05)
titles=['no\naberration','secondary\nobscuration','TMT\npupil','astigmatism','coma','phase\ngrating','amplitude\ngrating','static\naberration','atmospheric\naberration']
for i in range(len(ph)):
	im1=ax[0,i].imshow(amp[i][int(imagepix/2-pupilpix/2):int(imagepix/2+pupilpix/2),int(imagepix/2-pupilpix/2):int(imagepix/2+pupilpix/2)],vmin=0,vmax=1,cmap=cm.Greys_r)
	im2=ax[1,i].imshow((aperture*ph[i])[int(imagepix/2-pupilpix/2):int(imagepix/2+pupilpix/2),int(imagepix/2-pupilpix/2):int(imagepix/2+pupilpix/2)],vmin=-1,vmax=1,cmap=cm.Greys_r)
	im3=ax[2,i].imshow(np.log10(propagate(ph[i],amp[i]))[int(imagepix/2.-5*beam_ratio):int(imagepix/2.+5*beam_ratio),int(imagepix/2.-5*beam_ratio):int(imagepix/2.+5*beam_ratio)],vmin=-7,vmax=0.,cmap=cm.Greys_r)
	ax[0,i].set_title(titles[i],size=size)


labels=['$A$','$\\phi$','PSF']
def setl(j,label):
	ax[j,0].set_axis_on()
	ax[j,0].get_xaxis().set_visible(False)
	ax[j,0].get_yaxis().set_ticks([])
	ax[j,0].set_ylabel(label,rotation=0,y=0.4,labelpad=30,size=20)
	ax[j,0].spines['right'].set_visible(False)
	ax[j,0].spines['top'].set_visible(False)
	ax[j,0].spines['left'].set_visible(False)
	ax[j,0].spines['bottom'].set_visible(False)
[setl(j,labels[j]) for j in range(len(labels))]

cax1=fig.add_axes([0.91,0.65,0.01,0.2])
cb1=fig.colorbar(im1,cax=cax1,label='normalized\namplitude')
cb1.ax.tick_params(labelsize='x-small')
cax2=fig.add_axes([0.91,0.4,0.01,0.2])
cb2=fig.colorbar(im2,cax=cax2,label='radians')
cb2.ax.tick_params(labelsize='x-small')
cax3=fig.add_axes([0.91,0.14,0.01,0.2])
cb3=fig.colorbar(im3,cax=cax3,label='log(normalized\nintensity)')
cb3.ax.tick_params(labelsize='x-small')
plt.show()
fig.savefig(os.path.join(path,'plots/psfs.png'),bbox_inches='tight')