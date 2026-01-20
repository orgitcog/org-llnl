import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from scipy.special import gamma, hyp2f1

def polar_grid(imagepix,pupilpix):
	'''
	make a polar image grid from a cartesian grid
	'''
	grid=np.mgrid[0:imagepix,0:imagepix]
	xy=np.sqrt((grid[0]-imagepix/2.)**2.+(grid[1]-imagepix/2.)**2.)
	xy[np.where(xy>pupilpix/2.)]=0.
	rad_norm=xy/np.max(xy)
	phi=np.arctan2(grid[1]-imagepix/2.,grid[0]-imagepix/2.)
	return rad_norm,phi

def intensity(wavefront):
     return (np.abs(wavefront))**2.

def complex_amplitude(mag,phase):
	'''
	complex amplitude in terms of magnitude and phase
	'''
	return mag*np.cos(phase)+1j*mag*np.sin(phase)

def phase(wavefront):
	return np.arctan2(np.imag(wavefront),np.real(wavefront))

def pupil_to_image(im):
	return np.fft.fft2(im,norm='ortho')

def image_to_pupil(im):
	return np.fft.ifft2(im,norm='ortho')

def im(c1, c2):
    '''
    interaction matrix, m x n where
        m = total number of pixels in square
        n = number of modes (default 2)
    '''
    col_1 = np.array(c1.reshape(-1, 1))
    col_2 = np.array(c2.reshape(-1, 1))
    vec = np.hstack([col_1, col_2])
   
    return np.matrix(vec)

def zernike(n,m,rho,phi):
	'''
    from psfs.py
	make a zernike polynomial of specified n,m given input polar coordinate maps of rho (normalized to one; pupil coordinates only) and phi (radians)
	'''

	rad=gamma(n+1)*hyp2f1(-1./2.*(m+n),1./2.*(m-n),-n,rho**(-2))/gamma(1./2.*(2+n-m))/gamma(1./2.*(2+n+m))*rho**n
	if m>=0:
		cos=np.cos(m*phi)
		out=rad*cos
	else:
		sin=np.sin(-1*m*phi)
		out=rad*sin
	out[np.where(np.isnan(out)==True)]=0.
	return out

def fourier(im_pix, pup_pix):
    '''
    high freq diagonal mode 
    '''
    rho, _ = polar_grid(im_pix, pup_pix)
    
    x = np.linspace(0 , np.pi, im_pix)
    y = np.linspace(0, np.pi, im_pix)
    X, Y = np.meshgrid(x, y)

    mode = np.sin(14*(X - Y))
    mode = np.where(rho>0, mode, rho)

    return mode  # need to make global amp variable

def firstpass(m_object, t_int, im_tar_phase, noise=False, wfe=True):
    read_noise_level=30
    wav0=1.275e-6 # middle of the bandpass
    imagepix=128
    pupilpix=29
    beam_ratio=imagepix/pupilpix
    grid=np.mgrid[0:imagepix,0:imagepix]
    xy_dh=np.sqrt((grid[1]-imagepix/2.)**2.+(grid[0]-imagepix/2.)**2.)
    aperture=np.zeros((imagepix,imagepix))
    p_obs=0.3
    aperture[np.where(np.logical_and(xy_dh>p_obs*pupilpix/2,xy_dh<pupilpix/2))]=1. #obscured aperture
    indpup=np.where(aperture==1.)
    _, phi=polar_grid(imagepix,pupilpix)
    rphi=-phi+np.pi
    xtlt=(grid[1]-imagepix/2+0.5)/beam_ratio
    ytlt=(grid[0]-imagepix/2+0.5)/beam_ratio

    pwfs=np.zeros(aperture.shape)
    ind1=np.where(np.logical_and(rphi>=0,rphi<2*np.pi/3))
    ind2=np.where(np.logical_and(rphi>=2*np.pi/3,rphi<2*np.pi*2/3))
    ind3=np.where(np.logical_and(rphi>=2*np.pi*2/3,rphi<2*np.pi))
    fudge=3.
    ff=-0.6 #random fudge factor I need to get the right pupil centering...
    g1,g2,g3=(xtlt-ytlt)*np.sqrt(2)-ff*ytlt,ytlt-ff*ytlt,(-xtlt-ytlt)*np.sqrt(2)-ff*ytlt
    tt1,tt2,tt3=(g1*fudge),(g2*fudge),(g3*fudge)
    pwfs[ind1]=tt1[ind1]#-np.mean((tt1)[ind1])
    pwfs[ind2]=tt2[ind2]#-np.mean((tt2)[ind2])
    pwfs[ind3]=tt3[ind3]#-np.mean((tt3)[ind3])

    Dtel=3
    bandpass= 0.65/1.275 #0.26/1.22 #bandpass in fraction
    percent_bandpass=0.5 #actual bandpass in fraction
    f0=192.9*(wav0*(percent_bandpass/bandpass)*1e6)*10**8. #photons/m**2/s, mag 0 star, 50 % bandpass; https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html

    flux_object_ini=f0*10.**(-m_object/2.5)
    tr_atm,th,qe=0.3,0.2,0.75 #assume transmission through the atmosphere, instrument throughput, quantum efficiency of CCD
    flux_object=flux_object_ini*tr_atm*th*qe
    Nphot=flux_object*t_int*np.pi*(Dtel/2.)**2. 

    # no wavefront error
    no_phase_offset=np.zeros((imagepix,imagepix)) 

    def propagate(im_tar_phase, aperture, noise = False):
        pupil_wavefront_dm_unnorm=1* aperture*np.exp(1j*im_tar_phase)
        norm_phot=np.sum(intensity(np.ones((imagepix,imagepix))[indpup]))
        pupil_wavefront_dm=complex_amplitude(np.sqrt(intensity(pupil_wavefront_dm_unnorm)/norm_phot*Nphot),phase(pupil_wavefront_dm_unnorm))

        fpm_wavefront_ini=np.fft.fftshift(pupil_to_image(pupil_wavefront_dm)) #ft to image plane
        fpm_wavefront=complex_amplitude(np.abs(fpm_wavefront_ini),np.angle(fpm_wavefront_ini)+pwfs) #add phase mask
        pup_wf=image_to_pupil(fpm_wavefront) #ift to pupil plane
        im=intensity(pup_wf)
        if noise:
            read_noise= np.random.poisson(np.ones(aperture.shape)*read_noise_level**2) # add read noise
            photon_noise = np.random.poisson(im) # add photon noise
            noisy = photon_noise + read_noise -np.mean(read_noise)
            return noisy # noisy
        else:
            return im
        
    # create image for either w/wo wfe
    if wfe:
        im = propagate(im_tar_phase, aperture, noise)
    else:
        im = propagate(no_phase_offset, aperture, noise)

    return im/np.sum(im) # normalize for apple-to-apple comparison accross magnitude range

def coeff(signal, signal_wfe):
    '''
    signal input to create interaction matrix pseudoinverse to command matrix,
    reshape signal input to T matrix. Matrix multiply CM w/ T to get coeffs
    '''
    # create interaction matrix
    
    IM = signal_wfe.reshape(-1,1)#im(signal, signal_wfe)
 
    # pseudoinverse matrix
    # CM = IM^-1
    CM = pinv(IM, rcond=1e-10)
    T = CM.reshape(-1,1)
    # reshape input matrix, T
    #T = np.array(signal.reshape(-1, 1))
    
    phi = np.matmul(CM, T)

    #PHI = CM * T

    return phi

def signal(amp, rho, phi, imagepix, magnitude, exposure_time, wfe_type, noise=False):
    '''
    creates zernike and fourier signals
    '''
    if wfe_type=='zernike':
        # zernike, focus n = 4, m = 4
        wfe = (amp/2) * zernike(4, 4, rho, phi)

    elif wfe_type=='fourier':
        # fourier 
        wfe = (amp/2) * fourier(imagepix, imagepix)

    # signal w/o wfe, no noise
    signal = firstpass(magnitude, exposure_time, wfe, noise=False, wfe=False)

    # signal with wfe, no noise 
    signal_wfe = firstpass(magnitude, exposure_time, wfe, noise=False, wfe=True)

    return wfe, signal, signal_wfe

def plot_wavefront(data, heading, row, col, magnitude, exposure_time):
    '''
    plot of panels depicting wfe modes
    '''
    # plot
    plt.figure()
    fig, axs = plt.subplots(nrows=row, ncols=col, figsize=(10, 4))
    for i in range(row):
        for j in range(col):
            im_i = axs[i][j].imshow(data[i][j]) # change to axs[i][j]..... for multiple rows
            axs[i][j].set_title(heading[i][j])
            axs[i][j].set_axis_off()
            fig.colorbar(im_i)
    fig.suptitle(f'M: {magnitude}\nt_int: {exposure_time}s')
    
    plt.tight_layout()
    plt.savefig('./plots/noisy_calc.png', dpi = 400)
    plt.show()

def main():
    '''
    Using first_pass_testing to construct image data to use for signal-to-noise computation
    '''
    # input data
    magnitude = 2
    exposure_time = 1e-3
    amp = 0.15

    # the following also need to be changed in first pass
    imagepix = 128 #pupilpix = 29
    rho, phi = polar_grid(imagepix, imagepix)
    
    # make True to plot specific case
    if True: 
        amp = 0.15

        # create data for plotting
        n_1, signal_1, signal_1wfe = signal(amp, rho, phi, imagepix, magnitude, exposure_time, wfe_type='zernike')
        n_2, signal_2, signal_2wfe = signal(amp, rho, phi, imagepix, magnitude, exposure_time, wfe_type='fourier')

        # package data to be plotted
        data = [[n_1, signal_1, signal_1wfe, signal_1 - signal_1wfe], 
                [n_2, signal_2, signal_2wfe, signal_2 - signal_2wfe]]
        heading = [['zernike', 'no wfe', 'sig + zernike', 'diff'], 
                ['fourier', 'no wfe', 'sig + fourier', 'diff']]
        row, col = 2, 4

        # plot 
        plot_wavefront(data, heading, row, col, magnitude, exposure_time)
    
    
"""    # doing this w/o noise
    amp = np.linspace(-10, 10, 1000) # radians
    co = np.zeros(len(amp))
    for i in range(len(amp)):
        amplitude = amp[i]
        mode, sig, sig_wfe = signal(amplitude, rho, phi, imagepix, magnitude, exposure_time, wfe_type='fourier')
        T = coeff(sig, sig_wfe)
        co[i] = T 

    plt.plot(amp, co*amp)
    plt.title(f'Fourier mode, M: {magnitude}, t_int: {exposure_time}')
    plt.grid()
    plt.show()"""
if __name__ == '__main__':
    main()
