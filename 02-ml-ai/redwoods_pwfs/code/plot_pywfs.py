'''
plot psfs for a variety of phase and amplitude aberrations

'''

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc, cm

import func as functions

wav0=1.275e-6 #assumed wav0 for sine amplitude input in meters

# resolution of image?
imagepix=2048 # seems to fixed to data in 
beam_ratio=3 #can edit this, higher number -> more pixelated
pupilpix=imagepix/beam_ratio

grid=np.mgrid[0:imagepix,0:imagepix] # multidimensional grid
#xcen,ycen=imagepix/2,imagepix/2
xy_dh=np.sqrt((grid[1]-imagepix/2.)**2.+(grid[0]-imagepix/2.)**2.) # distance from center?

aperture=np.zeros((imagepix,imagepix)) # square opening size 2048 by 2048
aperture[np.where(xy_dh<pupilpix/2)]=1. #unobscured aperture

no_phase_offset=np.zeros((imagepix,imagepix)) # same as aperature

tmtpup=np.pad(fits.getdata(('./data/TMT-M1-Melco.fits')),512,mode='constant',constant_values=(0,)) # very large array full of very small numbers , max 5e-6
tmtpup=tmtpup/np.max(tmtpup) # normalize to max

#obscured aperture
pobs=0.4 # secondary obscuration
aperture2=np.zeros(aperture.shape)
aperture2[np.where(xy_dh<pupilpix/2)]=1.
aperture2[np.where(xy_dh<pobs*pupilpix/2)]=0. # affects radius of wfs image

# find parameters for polar coordinate transformation
rho,phi=functions.polar_grid(imagepix,pupilpix)

#pywfs mask
rphi=-phi + np.pi # causes overlapping in wfs image, 3pi results in smaller A image where wfs should be
xtlt=(grid[1]-imagepix/2+0.5)/beam_ratio
ytlt=(grid[0]-imagepix/2+0.5)/beam_ratio
pwfs=np.zeros(aperture.shape) # same as first aperature
ind1=np.where(np.logical_and(rphi>=0,rphi<np.pi/2)) # array of index values 
ind2=np.where(np.logical_and(rphi>=np.pi/2,rphi<np.pi))
ind3=np.where(np.logical_and(rphi>=np.pi,rphi<3*np.pi/2))
ind4=np.where(np.logical_and(rphi>=3*np.pi/2,rphi<2*np.pi))
fudge=5. # how far apart from center pywfs
tt1,tt2,tt3,tt4=(xtlt-ytlt)*fudge,(xtlt+ytlt)*fudge,(-xtlt+ytlt)*fudge,(-xtlt-ytlt)*fudge
pwfs[ind1]=tt1[ind1]
pwfs[ind2]=tt2[ind2]
pwfs[ind3]=tt3[ind3]
pwfs[ind4]=tt4[ind4]

def propagate_unnorm(pupil_phase_dm, amp):
	pupil_wavefront=amp*np.exp(1j*(pupil_phase_dm)) # complex number 
	focal_wavefront_b4pwfs=np.fft.fftshift(functions.pupil_to_image(pupil_wavefront)) # unitary fft and shift zero?
	focal_wavefront=functions.complex_amplitude(np.abs(focal_wavefront_b4pwfs),np.angle(focal_wavefront_b4pwfs)+pwfs) # numpy angle of complex number
	im=functions.intensity(functions.image_to_pupil(focal_wavefront))
	return im

def propagate(pupil_phase_dm,amp):
	return propagate_unnorm(pupil_phase_dm,amp)/np.sum(propagate_unnorm(no_phase_offset,amp))

amp=[aperture,aperture2,tmtpup,aperture,aperture,aperture,aperture*(1./3.*np.cos(2.*np.pi*3.*imagepix/pupilpix*grid[0]/grid[0][-1,0])+(1-1./3.)),
	aperture*functions.make_amp_err(0.01,imagepix,pupilpix),aperture]

ph=[no_phase_offset,no_phase_offset,no_phase_offset,3.*functions.zernike(2,-2,rho,phi),4.*functions.zernike(3,-1,rho,phi),
	np.cos(2.*np.pi*3.*imagepix/pupilpix*grid[0]/grid[0][-1,0]),no_phase_offset,functions.make_noise_pl(100e-9,imagepix,pupilpix,wav0,-2),
	functions.make_noise_pl(10e-6,imagepix,pupilpix,wav0,-11/3)]

size=20
font = {'family' : 'Times New Roman',
		'size'   : size}

mpl.rc('font', **font)
mpl.rcParams['image.interpolation'] = 'nearest'

fig,ax=plt.subplots(nrows=4,ncols=9,figsize=(16,8))
[axi.set_axis_off() for axi in ax.ravel()] #turns off auto axes for each image
plt.subplots_adjust(wspace=0.1,hspace=0.05)
titles=['no\naberration','secondary\nobscuration','TMT\npupil','astigmatism','coma','phase\ngrating','amplitude\ngrating','static\naberration','atmospheric\naberration']
subtract = [0, 0, 0, 3, 4, 5, 0, 7, 8]
for i in range(len(amp)):
	im1=ax[0,i].imshow(amp[i][int(imagepix/2-pupilpix/2):int(imagepix/2+pupilpix/2),int(imagepix/2-pupilpix/2):int(imagepix/2+pupilpix/2)],vmin=0,vmax=1,cmap=cm.Greys_r)
	im2=ax[1,i].imshow((aperture*ph[i])[int(imagepix/2-pupilpix/2):int(imagepix/2+pupilpix/2),int(imagepix/2-pupilpix/2):int(imagepix/2+pupilpix/2)],vmin=-1,vmax=1,cmap=cm.Greys_r)
	im3=ax[2,i].imshow(propagate(ph[i],amp[i])[int(imagepix/2.-1.5*pupilpix):int(imagepix/2.+1.5*pupilpix),int(imagepix/2.-1.5*pupilpix):int(imagepix/2.+1.5*pupilpix)],vmin=0,vmax=1e-5,cmap=cm.Greys_r)
	if i == subtract[i]:
		im4=ax[3,i].imshow(propagate(ph[i],amp[i])[int(imagepix/2.-1.5*pupilpix):int(imagepix/2.+1.5*pupilpix),int(imagepix/2.-1.5*pupilpix):int(imagepix/2.+1.5*pupilpix)]-
					propagate(ph[0],amp[0])[int(imagepix/2.-1.5*pupilpix):int(imagepix/2.+1.5*pupilpix),int(imagepix/2.-1.5*pupilpix):int(imagepix/2.+1.5*pupilpix)]
					,vmin=-7.5e-7,vmax=7.5e-7,cmap=cm.Greys_r)
	ax[0,i].set_title(titles[i],size=size)	

labels=['$A$','$\\phi$','PyWFS\nimage','Diff']
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

cax1=fig.add_axes([0.91,0.70,0.01,0.175])
cb1=fig.colorbar(im1,cax=cax1,label='normalized\namplitude')
cb1.ax.tick_params(labelsize='x-small')
cax2=fig.add_axes([0.91,0.5,0.01,0.175])
cb2=fig.colorbar(im2,cax=cax2,label='radians')
cb2.ax.tick_params(labelsize='x-small')
cax3=fig.add_axes([0.91,0.3,0.01,0.175])
cb3=fig.colorbar(im3,cax=cax3,label='normalized\nintensity')
cb3.ax.tick_params(labelsize='x-small')
cax4=fig.add_axes([0.91,0.10,0.01,0.175]) 
cb4=fig.colorbar(im4,cax=cax4,label='normalized\nintensity')
cb4.ax.tick_params(labelsize='x-small')
plt.show()
