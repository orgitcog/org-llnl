import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from scipy.special import gamma, hyp2f1
from scipy.ndimage import shift
import multiprocessing
import time as tm

def make_noise_pl(wavefronterror,imagepix,pupilpix,wavelength,pl):
	'''
	make noise with a user input power law:

	(1) take white noise in image plane
	(2) IFFT to pupil plane
	(3) weight complex pupil plane (spatial frequency) by -1 power law (power~amplitude**2~x**(-2), so amplitude goes as (x**(-2))**(1/2)=x(-1)
	(4) FFT back to image plane

	wavefronterror = rms WFE (nm)
	'''

	white_noise=np.random.random((imagepix,imagepix))*2.*np.pi #phase
	#noisefft=np.fft.fftshift(np.fft.fft2(white_noise))
	xy=xy_plane(imagepix)
	#grid=np.mgrid[0:imagepix,0:imagepix]
	#xy=np.sqrt((grid[0])**2.+(grid[1])**2.)
	amplitude=(xy+1)**(pl/2.) #amplitude central value in xy grid is one, so max val is one in power law, everything else lower

	#make all spatial frequencies below 1/Dpup/10 into a tophat function
	ind=np.where(np.abs(xy-(pupilpix/2.)/10.)==np.min(np.abs(xy-(pupilpix/2.)/10.)))
	amplitude=(xy+1)**(-1.)*xy[ind][0]
	amplitude[np.where(xy<(pupilpix/2.)/10.)]=1.
	amp_step_val=amplitude[np.where(np.abs(xy-(pupilpix/2.)/10.)==np.min(np.abs(xy-(pupilpix/2.)/10.)))][0]
	amplitude[np.where(xy<(pupilpix/2.)/10.)]=amp_step_val

	amplitude[int(imagepix/2),int(imagepix/2)]=0. #remove piston

	#remove alaising effects by cutting off power law just before the edge of the image
	amplitude[np.where(xy>imagepix/2.-1)]=0.

	amp=shift(amplitude,(-imagepix/2.,-imagepix/2.),mode='wrap')
	image_wavefront=complex_amplitude(amp,white_noise)
	noise_wavefront=np.real(np.fft.fft2(image_wavefront))
	norm_factor=(wavefronterror/wavelength*2.*np.pi)/np.std(noise_wavefront[np.where(xy<pupilpix/2.)]) #normalization factor for phase error over the pupil
	phase_out_ini=noise_wavefront*norm_factor

	phase_out=remove_tt(phase_out_ini,imagepix,pupilpix) #tip tilt removed phase screen

	return phase_out

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

def remove_tt(im_tar_phase,imagepix,pupilpix):
	'''
	remove tip and tilt

	input:
	im_tar_phase = normal input phase screen generated from some power law
	imagepix = width of focal plane image in pixels
	pupilpix = width of pupil in pixels

	output: phase screen with tip and tilt removed from the pupil
	'''

	xy=xy_plane(imagepix)
	phase_screen=im_tar_phase-np.mean(im_tar_phase[np.where(xy<pupilpix/2.)]) #remove piston in the pupil just by changing to zero mean

	rho,phi=polar_grid(imagepix,pupilpix)

	zern_nm=[]
	for n in range(1,2): #remove tip,tilt zernikes
		m=range(-1*n,n+2,2)
		for mm in m:
			zern_nm.append([n,mm])

	#reference array
	refarr=np.zeros((len(zern_nm),imagepix**2))
	for i in range(len(zern_nm)):
		z=zernike(zern_nm[i][0],zern_nm[i][1],rho,phi)
		refarr[i]=z.flatten()

	#covariance matrix:
	n=len(zern_nm)
	cov=np.zeros((n,n))
	for i in range(n):
		for j in range(i+1):
			if cov[j,i]==0.:
				cov[i,j]=np.sum(refarr[i,:]*refarr[j,:])
				cov[j,i]=cov[i,j]
			#print i*n+j,n**2-1
	covinv=np.linalg.pinv(cov,rcond=1e-7)

	#correlation image vector:
	tar=np.ndarray.flatten(phase_screen)
	cor=np.zeros((n,1))
	for i in range(n):
		cor[i]=np.sum(refarr[i]*tar)
		#print i, n-1

	coeffs=np.dot(covinv,cor)

	all_zern=np.dot(coeffs.T,refarr).reshape(imagepix,imagepix)
	out_phase=phase_screen-all_zern
	return out_phase
	
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

def xy_plane(dim):
	'''
	define xy plane to use for future functions
	'''
	grid=np.mgrid[0:dim,0:dim]
	xy=np.sqrt((grid[0]-dim/2.)**2.+(grid[1]-dim/2.)**2.)
	return xy

def image(mag, t_int, amp, wfe_phase, phn=False, readn= False):
	wavelength = 1.275e-6 # middle of bandpass
	imagepix = 128 # size of total image
	pupilpix = 29 # size of pupil image
	beam_ratio = imagepix/pupilpix
	grid = np.mgrid[0:imagepix, 0:imagepix] # two grids 128 x 128, all rows from 0 -> 127
	xy_dh = np.sqrt( (grid[1] - imagepix/2.)**2 + (grid[0] - imagepix/2.)**2 ) # grid of distances from center of grid
	aperature = np.zeros((imagepix, imagepix)) # grid of zeros, unobscured aperature
	
	p_obs = 0.3 # how much to obsure aperature
	aperature[np.where(np.logical_and( xy_dh > p_obs * pupilpix/2, xy_dh < pupilpix/2))] = 1. # creates ring with 1's bounded, zeros everywhere. OD~28 ID~8	
	indpup = np.where(aperature == 1.) # return boolean value where ring is filled with ones
	rho, phi = polar_grid(imagepix, pupilpix) # clockwise grid from pi to -pi
	rphi = -phi + np.pi # clockwise grid from 0 to 2pi

	xtlt = (grid[1] - imagepix/2 + 0.5)/beam_ratio # gradient along x on grid from -beam (left) to +beam (right)
	ytlt = (grid[0] - imagepix/2 + 0.5)/beam_ratio # gradiend along y on grid from -beam (top) to +beam (bottom)

	pwfs = np.zeros(aperature.shape)
	ind1 = np.where(np.logical_and(rphi >= 0, rphi < 2*np.pi/3))
	ind2 = np.where(np.logical_and(rphi >= 2*np.pi/3, rphi < 2*np.pi*2/3))
	ind3 = np.where(np.logical_and(rphi >= 2*np.pi*2/3, rphi < 2* np.pi))
	fudge = 3
	ff = -0.6 # for centering
	g1 = (xtlt - ytlt)*np.sqrt(2) - ff*ytlt # gradient along y=x line roughly
	g2 = ytlt - ff*ytlt # gradient neg (top) to pos (down) 
	g3 = (-xtlt - ytlt)*np.sqrt(2) - ff*ytlt # gradient along y=-x
	tt1, tt2, tt3 = g1*fudge, g2*fudge, g3*fudge

	# gradient zero at center in pyramid apex pattern
	pwfs[ind1] = tt1[ind1] # 1st 3rd cw
	pwfs[ind2] = tt2[ind2]	# 2nd 3rd cw
	pwfs[ind3] = tt3[ind3] # 3rd 3rd cw

	Dtel = 3 # diameter of aperature
	bandpass = 0.65/1.275 # bandpass fraction, range over mean
	percent_bandpass = 0.5
	f0 = 192.9 * (wavelength * (percent_bandpass/bandpass)*1e6)*10**8. # photons/m^2 s, mag 0 star, 50% bandpass

	flux_ini = f0 * 10.**(-mag/2.5)
	tr_atm = 0.3 # transmission through atmosphere
	th = 0.2 # instrument throughput
	qe = 0.6 # quantum efficiency of CCD
	flux = flux_ini * tr_atm * th * qe
	num_photon = flux * t_int * np.pi*(Dtel/2.)**2.
	
	def propagate(wfe_phase, aperature):
		pupil_wavefront_dm_unnorm =  aperature * np.exp(1j*wfe_phase * amp) 
		norm_phot = np.sum(intensity(np.ones((imagepix, imagepix))[indpup]))
		pupil_wavefront_dm = complex_amplitude(np.sqrt(intensity(pupil_wavefront_dm_unnorm)/norm_phot*num_photon),phase(pupil_wavefront_dm_unnorm))

		fpm_wavefront_ini = np.fft.fftshift(pupil_to_image(pupil_wavefront_dm)) # ft to image plane
		fpm_wavefront = complex_amplitude(np.abs(fpm_wavefront_ini), np.angle(fpm_wavefront_ini) + pwfs)
		pup_wf = image_to_pupil(fpm_wavefront) # ift to pupil plane
		im = intensity(pup_wf)
		
		if phn:
			noise = np.random.poisson(im) 
			return noise 
		else:
			return im
	
	# no wavefront error
	if wfe_phase == 'none':
		wfe_phase = np.zeros_like(aperature)

	# power law wavefront error
	elif wfe_phase == 'pl':
		wavefronterror=100e-9/wavelength*2*np.pi #m rms wavefront error
		wfe_phase = make_noise_pl(wavefronterror, imagepix, pupilpix, -2, 16)

	# fourier wavefront error
	elif wfe_phase == 'fourier':
		wfe_phase = fourier(imagepix, imagepix)

	# zernike wavefront error
	elif wfe_phase == 'zernike':
		wfe_phase =  zernike(4, 4, rho, phi)

	im = propagate(wfe_phase, aperature)

	# read noise
	n_noise = 0
	if readn:
		read_noise_level = 30
		read_noise = np.random.poisson(np.ones_like(im) * read_noise_level)
		n_noise = read_noise
	return (im + n_noise)/np.sum(im)

def CM(mag, t_int, i_amp):
	im_fourier = image(mag, t_int, i_amp, 'fourier') - image(mag, t_int, i_amp, 'none')
	im_zernike = image(mag, t_int, i_amp, 'zernike') - image(mag, t_int, i_amp, 'none')
	
	IM = np.hstack([im_fourier.ravel().reshape(-1, 1), im_zernike.ravel().reshape(-1,1)])
	cm = pinv(IM, rcond=1e-10)
	return cm

def signal_fz(mag, t_int, amp, cm):
	'''
	return coefficients for signal fourier wfe
	'''
	wfe = image(mag, t_int, amp, wfe_phase='fourier') + image(mag, t_int, amp, wfe_phase='zernike') - 2*image(mag, t_int, amp, 'none')
	T = wfe.ravel().reshape(-1, 1)

	return cm @ T

def signal_pl(mag, t_int, amp, cm):
	'''
	return coefficients for signal pl wfe
	'''
	pl_wfe = image(mag, t_int, amp, wfe_phase='pl') 
	T = pl_wfe.ravel().reshape(-1, 1)

	return cm @ T

def noise_coeff(mag, t_int, amp, cm):
	'''
	return coefficients for noise, fourier & zernike mode
	'''
	photon_noise = (image(mag, t_int, amp, wfe_phase='fourier', phn=True, readn=False) + image(mag, t_int, amp, wfe_phase='zernike', phn=True, readn=False))/2 # added read noise twice, one for each mode
	read_noise = image(mag, t_int, amp, wfe_phase='fourier', phn=False, readn=True) + image(mag, t_int, amp, wfe_phase='zernike', phn=False, readn=True) - (image(mag, t_int, amp, wfe_phase='fourier', phn=False, readn=False) + image(mag, t_int, amp, wfe_phase='zernike', phn=False, readn=False))
	noise = photon_noise + read_noise
	
	T = noise.ravel().reshape(-1, 1)


	return cm @ T

def noise_coeff_OLD(mag, t_int, amp, cm):
	'''
	return coefficients for noise, fourier & zernike mode
	'''
	photon_noise = image(mag, t_int, amp, wfe_phase='pl', phn=True, readn=False)
	read_noise = image(mag, t_int, amp, wfe_phase='pl', phn=False, readn=True) - image(mag, t_int, amp, wfe_phase='pl', phn=False, readn=False)
	noise = photon_noise + read_noise
	T = noise.ravel().reshape(-1, 1)

	return cm @ T

def sd_noise(mag, t_int, i_amp, cm):
	'''
	standard deviation of noise coeff after running n times
	'''
	# iterate n times and take SD for noise 
	n = np.arange(50, 251, 25) 

	# set up lists to save signal data
	fourier_sd = np.zeros(len(n)) 
	zernike_sd = np.zeros(len(n))
	time_start = tm.time()
	for i in range(len(n)):
		f_loop = np.zeros(n[i])
		z_loop = np.zeros(n[i])
		for j in range(n[i]):
			coeffs = noise_coeff(mag, t_int, i_amp, cm)
			f_loop[j] = coeffs[0].item()
			z_loop[j] = coeffs[1].item()
		fourier_sd[i] = np.std(f_loop, ddof=1)
		zernike_sd[i] = np.std(z_loop, ddof=1)
	time_end = tm.time()
	print(time_end-time_start)
	plt.figure()
	plt.scatter(n, (fourier_sd), label='fourier mode')
	plt.scatter(n, zernike_sd, label='zernike mode')
	plt.legend()
	plt.xlabel('n trials')
	plt.ylabel('standard deviation')
	plt.grid(which='both', alpha = 0.15, c='dimgrey', zorder=0)
	plt.minorticks_on()
	plt.savefig('./plots/sd_test.png', dpi=400)
	plt.show()

def sd_hist(mag, t_int, i_amp, cm):
	'''
	histogram of noise coeff after running n times
	'''

	def hist_subplot(n):
		# set up lists to save signal data
		f_loop = np.zeros(n) 
		z_loop = np.zeros(n)
		
		for j in range(n):	
			coeffs = noise_coeff(mag, t_int, i_amp, cm)
			f_loop[j] = coeffs[0].item()
			z_loop[j] = coeffs[1].item()
		return f_loop, z_loop
	
	f_50, z_50 = hist_subplot(50)
	f_250, z_250 = hist_subplot(500)
	f_500, z_500 = hist_subplot(5000)
	plots = [[f_50, f_250, f_500], [z_50, z_250, z_500]]
	label = ['n=50', 'n=500', 'n=5000']

	row, col = 2, 3
	fig, ax = plt.subplots(nrows=row, ncols=col)
	for j in range(col):
		ax[0][j].hist(plots[0][j], color=(14/255, 96/255, 39/255))
		ax[1][j].hist(plots[1][j], color=(0/255, 83/255, 154/255))
		ax[0][j].set_title(label[j])

	
	ax[0][0].set_ylabel('Fourier', size=14)
	ax[1][0].set_ylabel('Zernike', size=14)

	plt.savefig('./plots/sd_histograms.png', dpi=400, transparent=True)
	plt.show()

def mean_hist(mag, t_int, i_amp, cm, n):
	'''
	histogram of noise coeff after running n times
	'''

	# set up lists to save signal data
	f_loop = np.zeros(n) 
	z_loop = np.zeros(n)

	
	for j in range(n):
		coeffs = signal_pl(mag, t_int, i_amp, cm)
		f_loop[j] = coeffs[0].item()
		z_loop[j] = coeffs[1].item()
	
	
	plt.figure()
	plt.title(f'M: {mag}\nt_int:{t_int}s\nn = {n}')
	plt.hist(f_loop, label='fourier mode', color='blue', alpha=1)
	plt.savefig('./plots/four_mean_hist.png', dpi = 400)
	plt.legend()

	plt.figure()
	plt.title(f'M: {mag}\nt_int:{t_int}s\nn = {n}')
	plt.hist(z_loop, label='zernike mode', color='darkorange', alpha=1)
	plt.legend()
	plt.savefig('./plots/zern_mean_hist.png', dpi = 400)
	plt.show()
	
def mean_pl(mag, t_int, amp, cm):
	# iterate n times  
	n = np.arange(1, 21) 

	# set up lists to save signal data
	fourier_mean = np.zeros(len(n)) 
	zernike_mean = np.zeros(len(n))
	
	for i in range(len(n)):
		f_loop = np.zeros(n[i])
		z_loop = np.zeros(n[i])
		for j in range(n[i]):
			coeffs = signal_pl(mag, t_int, amp, cm)
			f_loop[j] = coeffs[0].item()
			z_loop[j] = coeffs[1].item()
		fourier_mean[i] = np.mean(f_loop)
		zernike_mean[i] = np.mean(z_loop)
	
	plt.figure()
	plt.title(f'M: {mag}\nt_int:{t_int}s')
	plt.scatter(n, (fourier_mean), label='fourier mode')
	plt.scatter(n, zernike_mean, label='zernike mode')
	plt.legend()
	plt.xlabel('n trials')
	plt.ylabel('mean')
	plt.grid(which='both', alpha = 0.15, c='dimgrey', zorder=0)
	plt.minorticks_on()
	plt.savefig('./plots/mean_test.png', dpi=400)
	plt.show()

def fraction(mag, t_int, i_amp, cm, n):
	# set up lists to save signal data
	f_loop_mean = np.zeros(n) 
	z_loop_mean = np.zeros(n)
	f_loop_sd = np.zeros(n) 
	z_loop_sd = np.zeros(n)

	for j in range(n):
		coeff_sig = signal_pl(mag, t_int, i_amp, cm)
		coeff_noise = noise_coeff(mag, t_int, i_amp, cm)
		f_loop_mean[j] = coeff_sig[0].item()
		z_loop_mean[j] = coeff_sig[1].item()
		f_loop_sd[j] = coeff_noise[0].item()
		z_loop_sd[j] = coeff_noise[1].item()
	fourier_mean = np.mean(f_loop_mean)
	zernike_mean = np.mean(z_loop_mean)
	fourier_sd = np.std(f_loop_sd, ddof=1)
	zernike_sd = np.std(z_loop_sd, ddof=1)

	print(fourier_mean/fourier_sd)
	print(zernike_mean/zernike_sd)

def ratio(mag, t_int, i_amp, cm, n):
	'''
	same idea as the fraction func above but calculating snr for each pair instead of using mean/std
	'''
	f_sig = np.zeros(n)
	z_sig = np.zeros(n)
	f_noise = np.zeros(n)
	z_noise = np.zeros(n)
	for i in range(n):
		if i>=5:
			sf_1 = np.zeros(i) # create temp lists to store coeffs for both modes
			nf_1 = np.zeros(i)
			sz_1 = np.zeros(i)
			nz_1 = np.zeros(i)
			for j in range(i):
				coeff_sig = signal_fz(mag, t_int, i_amp, cm)
				coeff_noise = noise_coeff(mag, t_int, i_amp, cm)
				sf_1[j] = coeff_sig[0].item()
				nf_1[j] = coeff_noise[0].item()
				sz_1[j] = coeff_sig[1].item()
				nz_1[j] = coeff_noise[1].item()
			f_sig[i] = np.mean(sf_1)
			f_noise[i] = np.std(nf_1, ddof=1)
			z_sig[i] = np.mean(sz_1)
			z_noise[i] = np.std(nz_1, ddof=1)
		else:
			pass
	f_snr = f_sig/f_noise
	z_snr = z_sig/z_noise
	plt.figure()
	plt.title(f'M: {mag}, t_int: {t_int}s')
	plt.plot(np.arange(n), f_snr, label='fourier mode snr')
	plt.plot(np.arange(n), z_snr, label='zernike mode snr')
	plt.legend()
	plt.show()

def im_prop(mag, t_int, i_amp):
	# define plots
	no_wfe = image(mag, t_int, i_amp, wfe_phase='none') 
	zf_wfe = image(mag, t_int, i_amp, wfe_phase='fourier') + image(mag, t_int, i_amp, wfe_phase='zernike') - 2*image(mag, t_int, i_amp, wfe_phase='none')
	noise = image(mag, t_int, i_amp, wfe_phase='fourier', phn=True, readn=True) + image(mag, t_int, i_amp, wfe_phase='zernike', phn=True, readn=True) - 2*image(mag, t_int, i_amp, 'none', phn=False, readn=False)
	phn = image(mag, t_int, i_amp, wfe_phase='fourier', phn=True, readn=False) + image(mag, t_int, i_amp, wfe_phase='zernike', phn=True, readn=False) - 2*image(mag, t_int, i_amp, 'none', phn=False, readn=False)

	# save plots/headers
	plots = [[no_wfe, zf_wfe], [noise, phn]]
	header = [['No WFE', 'Fourier + Zernike\nWFE'], ['Photon + Read Noise', 'Photon Noise']]

	row, col = 2, 2
	fig, axs = plt.subplots(nrows=row, ncols=col)
	for i in range(row):
		for j in range(col):
			im_i = axs[i][j].imshow(plots[i][j])
			fig.colorbar(im_i)
			axs[i][j].set_axis_off()
			axs[i][j].set_title(header[i][j])
		
	plt.tight_layout()
	plt.savefig('./plots/wfs_modes.png', transparent=True, dpi=400)
	plt.show()

def t_snr(t, mag, i_amp, n):
	'''
	was inside mag_snr but needed to make 'pickleable' to use multiprocessing
	'''
	f_snr = np.zeros(len(mag))
	z_snr = np.zeros(len(mag))

	for m in range(len(mag)):
		sf_1 = np.zeros(n) # create temp lists to store coeffs for both modes
		nf_1 = np.zeros(n)
		sz_1 = np.zeros(n)
		nz_1 = np.zeros(n)
		cm = CM(mag[m], t, i_amp)
		for i in range(n):
			coeff_sig = signal_pl(mag[m], t, i_amp, cm)
			coeff_noise = noise_coeff(mag[m], t, i_amp, cm)
			sf_1[i] = coeff_sig[0].item()
			nf_1[i] = coeff_noise[0].item()
			sz_1[i] = coeff_sig[1].item()
			nz_1[i] = coeff_noise[1].item()
		f_snr[m] = (np.mean(sf_1)) / (np.std(nf_1, ddof=1))
		z_snr[m] = (np.mean(sz_1) / (np.std(nz_1, ddof=1)))
	return f_snr, z_snr

def snr_mag(i_amp, n=500):
	'''
	similar to ratio func but for varying magnitudes from M1 to M10
	for a fixed n (default 250 iterations)
	'''
	num_cores = multiprocessing.cpu_count()
	# set up range of magnitudes to look at
	mag = np.arange(1, 12, 0.5)

	t_start = tm.time()
	args = [(t, mag, i_amp, n) for t in [0.001, 0.010]]
	with multiprocessing.Pool(processes=num_cores) as pool:
		results = pool.starmap(t_snr, args)
	(f_snr_1ms, z_snr_1ms), (f_snr_10ms, z_snr_10ms) = results
	t_end = tm.time()
	print(f'time: {t_end-t_start}s')

	plt.figure()
	plt.plot(mag, np.abs(f_snr_1ms), label='Fourier mode (1ms)', marker='x' , c=(14/255, 96/255, 39/255))
	plt.plot(mag, np.abs(z_snr_1ms), label='Zernike mode (1ms)', marker='x' , c=(0/255, 83/255, 154/255))
	plt.plot(mag, np.abs(f_snr_10ms), label='Fourier mode (10ms)', marker='o' ,  c=(14/255, 96/255, 39/255))
	plt.plot(mag, np.abs(z_snr_10ms), label='Zernike mode (10ms)', marker='o' , c=(0/255, 83/255, 154/255))
	plt.axhline(y=3, color='dimgrey', label='SNR = 3', linestyle='--')
	plt.legend(framealpha=0)
	plt.title(f'n: {n} iterations')
	plt.grid(which='both', alpha = 0.15, c='dimgrey', zorder=0)
	plt.minorticks_on()
	plt.xlabel('Magnitude')
	plt.ylabel('SNR')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig('./plots/snr_mag.png', transparent=True, dpi=400)
	plt.show()

def main():
	'''
	Determine signal to noise ratio for a zernike/fourier wavefront error mode
	Plot input vs output amplitude
	Plot SNR for increasing magnitude (M = 1 to M = 15) for 1ms & 10ms integration time
	'''
	# input data
	magnitude = 3
	exposure_time = 1e-3
	amplitude = 0.15

	# create command matrix
	cm = CM(magnitude, exposure_time, amplitude)

	# fourier coeff[0]
	# zernike coeff[1]

	#sig_strength = signal_pl(magnitude, exposure_time, amplitude, cm)
	#noise_strength = noise_coeff(magnitude, exposure_time, amplitude, cm)

	#sd_noise(magnitude, exposure_time, amplitude, cm)
	#sd_hist(magnitude, exposure_time, amplitude, cm)

	#mean_pl(magnitude, exposure_time, amplitude, cm)
	#mean_hist(magnitude, exposure_time, amplitude, cm, 250)
	#fraction(magnitude, exposure_time, amplitude, cm, 50)
	
	#noise_coeff(magnitude, exposure_time, amplitude, cm)

	################ USE THIS ONE #####################
	#ratio(magnitude, exposure_time, amplitude, cm, 50) 
	###################################################

	#snr_mag(amplitude)

	# propation w/o noise, w/ noise, just photon noise
	im_prop(magnitude, exposure_time, amplitude)

if __name__ == '__main__':
    main()