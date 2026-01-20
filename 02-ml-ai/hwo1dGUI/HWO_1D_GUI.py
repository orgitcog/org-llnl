# Copyright 2025 Lawrence Livermore National Security, LLC and other hwo1dGUI Project Developers. See the top-level LICENCE file for details.
#
# SPDX-License-Identifier: MIT

'''
Explore various parameters related to reaching (or not) sigma10 (10 pm rms over 10 minutes) for Habitable Worlds Observatory (HWO), building off of Douglas et al. 2019, Fig. 4.
'''
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.widgets import Slider, Button, RadioButtons

#AO transfer functions
Hwfs = lambda s, Ts: (1. - np.exp(-Ts*s))/(Ts*s)
Hzoh=Hwfs
Hlag = lambda s,tau: np.exp(-tau*s)
Hint = lambda s, Ts, g: g/(1. - np.exp(-Ts*s)) #integrator
Hlint=  lambda s, Ts, g, l: g/(1. - l*np.exp(-Ts*s)) #leaky integrator
Hcont = Hlint
Holsplane = lambda s, Ts, tau, g, l:  Hwfs(s, Ts)*Hlag(s,tau)*Hcont(s, Ts, g, l)*Hzoh(s,Ts)
s2f = lambda f: 1.j*2.*np.pi*f
Hol = lambda f, Ts, tau, g, l:  Holsplane(s2f(f),Ts,tau,g,l) #open loop transfer function
Hrej = lambda f, Ts, tau, g, l: 1./(1. + Hol(f, Ts, tau, g, l)) #rejection transfer function


Twfs=100e-3 #WFS frame rate in s
tau=1.5e-3 #delay in s
wfeol=100 #desired input WFE in pm rms between 1/10 m and 5 Hz (Nyquist limit at 10 Hz, but for faster frame rates it's not the full psd; this is done so the input WFE doesn't change with changing frame rate for an apples to apples comparison)
f0=2 #PSD turnover frequency in Hz
pl=-2 #PSD power law, unitless
mv=-3 #magnitude in v band, informing ZWFS noise level based on Douglas et al. 2019 Fig. 4
beta=0.5 #WFS sensitivity, from Douglas et al. fig. 4, originally from, Guyon et al. 2005; 0.5 is for a ZWFS and generally all other WFSs are lower. here specifically this means sensitivity to segment modes. 

def gendata(Twfs,tau,wfeol,f0,pl,mv,beta):
	nframes=1000
	freq=np.logspace(np.log10(1/(10*60)),np.log10(1/Twfs/2),nframes)
	linfreq=np.linspace(np.min(freq),np.max(freq),nframes)
	def getbw(gain,leak): #get bandwidth
		etf=np.abs(Hrej(freq, Twfs, tau, gain,leak))**2
		indsbw=np.where(np.diff(np.sign(etf-1)))[0] #find the zero crossings, including ones higher than the BW
		bw=np.min(freq[indsbw]) #the BW is the minimum frequency
		return bw
	def genpm(Hol): #phase margin
	        phase_margin_point=np.real(Hol)**2+np.imag(Hol)**2-1 #where the curve hits the Unit Circle. the -1 term allows a sign change to occur when it crosses the unit circle
	        phase_margin_inds=np.where(np.diff(np.sign(phase_margin_point)))[0]
	        if len(phase_margin_inds)==0: #if there are no unit circle crossings, it's definitely not an unstable system! in that case just retun dummy numbers to indicate it is stable
	        	return 0,0,180
	        else:
		        pms=180-np.abs(np.angle(Hol[phase_margin_inds])*180/np.pi)
		        indpms=np.where(pms==np.min(pms)) #use the unit circle crossing corresponding to the minimum phase margin
		        pm,phase_margin_ind=pms[indpms],phase_margin_inds[indpms]
		        pmreal,pmimag=np.real(Hol[phase_margin_ind]),np.imag(Hol[phase_margin_ind])
		        return pmreal,pmimag,pm
	def gengm(Hol): #gain margin
	        zero_crossings = np.where(np.diff(np.sign(np.imag(Hol))))[0]
	        zero_points=np.real(Hol[zero_crossings])
	        return np.min(zero_points)
	def genpmgm(Holp,Holn): #generate phase and gain margins
	        pmrealp,pmimagp,pmp=genpm(Holp)
	        pmrealn,pmimajn,pmn=genpm(Holn)
	        gmp,gmn=gengm(Holp),gengm(Holn)
	        return pmp,pmn,gmp,gmn
	def genNyquist(gain,leak): #generate data for Nyquist diagram
		Holp=Hol(freq, Twfs, tau, gain, leak)
		Holn=Hol(-1*freq, Twfs, tau, gain, leak)
		return Holp,Holn
	def pltNyquist(gain,leak): #plot a Nyquist diagram
		Holp,Holn=genNyquist(gain,leak)
		plt.figure()
		plt.plot(np.real(Holp),np.imag(Holp),color='blue')
		plt.plot(np.real(Holn),np.imag(Holn),color='blue')
		plt.ylabel('Imaginary{Hol}')
		plt.xlabel('Real{Hol}')

	def find_phase_and_gain_margins(gain,leak): #given this frame rate and delay, here's a function to find the phase and gain margins
		Holp,Holn=genNyquist(gain,leak)
		pmp,pmn,gmp,gmn=genpmgm(Holp,Holn)
		pm,gm=np.round(np.min([pmp,pmn]),1),np.round(np.min([gmp,gmn]),1)
		return pm,gm
	ngl=20
	garr=np.linspace(0.01,2,ngl)
	#larr=np.linspace(0.01,1,ngl)
	larr=np.flip(1-np.logspace(np.log10(1e-5),np.log10(0.99),ngl))
	glarr=[] #array of all the stable gain and leak pairs
	optwfeglarr=np.zeros((ngl,ngl)) #array to be populated further below with CL WFE as a function of gain and leak to determine optimal gain and leak; here just mark off what parameter space is unstable with nans
	for i in range(len(garr)):
		for j in range(len(larr)):
			pm,gm=find_phase_and_gain_margins(garr[i],larr[j])
			if pm<=45 or -1/gm<=2.5:
				optwfeglarr[i,j]=np.nan
				continue
			else:
				glarr.append([garr[i],larr[j]])

	indfreq5=np.where(freq<=5)
	wfe_unnorm=(freq**2+f0**2)**pl/2
	wfe=wfe_unnorm*wfeol**2/np.trapz(wfe_unnorm[indfreq5],x=freq[indfreq5]) #normalize the input WFE component to the desired input
	ncoeff = (0.5**2)/beta**2*8*10**((-3-mv)/-2.5) #convert from magnitudes to noise coefficient on the noise psd
	noisepsd=np.ones(nframes)*ncoeff
	psdm=wfe+noisepsd
	wfearrm=np.zeros(len(glarr))
	for i in range(len(glarr)):
		etf=np.abs(Hrej(freq, Twfs, tau, glarr[i][0],glarr[i][1])**2)
		wfearrm[i]=np.sqrt(np.trapz(etf*psdm,x=freq))
		optwfeglarr[np.where(garr==glarr[i][0])[0][0],np.where(larr==glarr[i][1])[0][0]]=wfearrm[i]
	optglm=np.array(glarr)[np.where(wfearrm==np.min(wfearrm))]
	if optglm.shape[0]>2: #if there are multiple gains and leaks that equally minimize the wavefront, just pick one
		optglm=optglm[0]
	gm,lm=optglm.flatten()
	etfm=np.abs(Hrej(freq, Twfs, tau, gm,lm)**2)
	bwm=getbw(gm,lm)
	pmargin,gmargin=find_phase_and_gain_margins(gm,lm)
	Holp,Holn=genNyquist(gm,lm)
	return wfearrm,freq,wfe,noisepsd,psdm,etfm,gm,lm,bwm,optwfeglarr,larr,garr,pmargin,gmargin,Holp,Holn

wfearrm,freq,wfe,noisepsd,psdm,etfm,gm,lm,bwm,optwfeglarr,larr,garr,pmargin,gmargin,Holp,Holn=gendata(Twfs,tau,wfeol,f0,pl,mv,beta)

size=15
font = {'family' : 'Times New Roman',
        'size'   : size}

mpl.rc('font', **font)
mpl.rcParams['image.interpolation'] = 'nearest'

colors=cm.viridis(np.linspace(0,0.75,3))
#axis_color = 'lightgoldenrodyellow'
fig,axs=plt.subplots(nrows=1,ncols=3,figsize=(15,10))
ax1,ax2,ax3=axs.ravel()
fig.suptitle('output WFE = '+str(int(round(np.min(wfearrm))))+'pm rms',size=size)

ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylabel('PSD [pm$^2$/Hz]')
ax1.set_xlabel('temporal frequency [Hz]')
[olpsd_line]=ax1.plot(freq,wfe,color=colors[0],ls=':')
[npsd_line]=ax1.plot(freq,noisepsd,color=colors[0],ls='--')
[olnpsd_line]=ax1.plot(freq,psdm,color=colors[0],ls='-')
[clpsd_line]=ax1.plot(freq,psdm*etfm,color=colors[1],ls='-.')
ax1t=ax1.twinx()
ax1t.set_yscale('log')
ax1t.set_xscale('log')
ax1t.set_ylabel('optimal ETF, gain='+str(np.round(gm,1))+', leak='+str(np.round(lm,4))+', BW='+str(np.round(bwm,1))+' Hz',y=0.4)
[etf_line]=ax1t.plot(freq,np.abs(Hrej(freq, Twfs, tau, gm,lm)**2),color=colors[2])
bwm_line=ax1.axvline(bwm,color=colors[2])


legend_elements=[]
legend_elements.append(Line2D([0], [0], color=colors[0], ls=':',label='OL dynamic'))
legend_elements.append(Line2D([0], [0], color=colors[0], ls='--',label='OL noise'))
legend_elements.append(Line2D([0], [0], color=colors[0], ls='-',label='OL'))
legend_elements.append(Line2D([0], [0], color=colors[1], ls='-.',label='CL'))
legend_elements.append(Line2D([0], [0], color=colors[2], ls='-',label='ETF'))
ax1.legend(loc='lower right',handles=legend_elements,ncol=2)

im=ax2.imshow(optwfeglarr,origin='lower',extent=(np.log10(1-larr[0]),np.log10(1-larr[-1]),garr[0],garr[-1]),aspect=-1*np.log10(1-larr[-1])/garr[-1])
imcb=fig.colorbar(im,label='output WFE (pm rms)',ax=ax2)
ax2.set_xlabel('log$_{10}$(1-leak)')
ax2.set_ylabel('gain')
cbticks=imcb.get_ticks()

colors=cm.viridis(np.linspace(0,1,4))
phasegrid=np.linspace(-np.pi,np.pi,500)
xunit,yunit=np.cos(phasegrid),np.sin(phasegrid)
ax3.plot(xunit,yunit,color='k',ls=':')
gain_margin_limit=2.5
ax3.set_title('$\\Delta$g='+str(round(-1/gmargin,1))+', $\\Delta\\phi='+str(round(pmargin,1))+'^{\\circ}$',size=size)
ax3.axvline(-1/gain_margin_limit,color=colors[0],ls='--',label='$\\Delta$g=2.5')

ax3.plot(-1,0,'-o',color=colors[1],label='pole')
ax3.set_xlim(-1.1,1.1)
ax3.set_ylim(-1.1,1.1)
ax3.plot(np.linspace(-2,0,10),np.linspace(-2,0,10),color=colors[2],ls='--',label='$\\Delta\\phi=45^{\\circ}$')
ax3.plot(np.linspace(-2,0,10),np.linspace(2,0,10),color=colors[2],ls='--')

[nyquistp_line]=ax3.plot(np.real(Holp),np.imag(Holp),color=colors[3],label='H$_\\mathrm{OL}$')
[nyquistn_line]=ax3.plot(np.real(Holn),np.imag(Holn),color=colors[3])
ax3.legend(loc='best',framealpha=0.7,ncol=2)
ax3.set_ylabel('Imaginary{H$_\\mathrm{OL}$}')
ax3.set_xlabel('Real{H$_\\mathrm{OL}$}')
plt.tight_layout()

fig.subplots_adjust(bottom=0.55)

Twfs_ax  = fig.add_axes([0.25, 0.4, 0.65, 0.03])
tau_ax  = fig.add_axes([0.25, 0.35, 0.65, 0.03])
wfeol_ax = fig.add_axes([0.25, 0.3, 0.65, 0.03])
f0_ax  = fig.add_axes([0.25, 0.25, 0.65, 0.03])
pl_ax  = fig.add_axes([0.25, 0.2, 0.65, 0.03])
mv_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
beta_ax  = fig.add_axes([0.25, 0.1, 0.65, 0.03])


Twfs_slider = Slider(Twfs_ax, 'T$_s$ (s)',0.1e-3,1, valinit=Twfs,valfmt='%.2e')
tau_slider = Slider(tau_ax, '$\\tau$ (s)',0.1e-3,100e-3, valinit=tau,valfmt='%.2e')
wfeol_slider = Slider(wfeol_ax, 'input WFE (pm rms)',1.,1000., valinit=wfeol)
f0_slider = Slider(f0_ax, 'f$_0$ (Hz)',1/30/60,10, valinit=f0)
pl_slider = Slider(pl_ax, '$\\alpha$ (power law)',-4,-1, valinit=pl)
mv_slider = Slider(mv_ax, 'target V mag',-15,7, valinit=mv)
beta_slider = Slider(beta_ax, '$\\beta$ (WFS sensitivity)',0.01,0.5, valinit=beta)

def sliders_on_changed(val):
	Twfs=Twfs_slider.val
	tau=tau_slider.val
	wfeol=wfeol_slider.val
	f0=f0_slider.val
	pl=pl_slider.val
	mv=mv_slider.val
	beta=beta_slider.val

	wfearrm,freq,wfe,noisepsd,psdm,etfm,gm,lm,bwm,optwfeglarr,larr,garr,pmargin,gmargin,Holp,Holn=gendata(Twfs,tau,wfeol,f0,pl,mv,beta)


	ax1.set_xlim(freq[0]*0.5,freq[-1]*1.5)
	ax1t.set_xlim(freq[0]*0.5,freq[-1]*1.5)
	ax1ydata=np.array([wfe,noisepsd,psdm,psdm*etfm]).flatten()
	ax1.set_ylim(np.min(ax1ydata)*0.5,np.max(ax1ydata)*1.5)
	ax1t.set_ylim(np.min(np.abs(Hrej(freq, Twfs, tau, gm,lm)**2))*0.5,np.max(np.abs(Hrej(freq, Twfs, tau, gm,lm)**2))*1.5)
	olpsd_line.set_data(freq,wfe)
	npsd_line.set_data(freq,noisepsd)
	olnpsd_line.set_data(freq,psdm)
	clpsd_line.set_data(freq,psdm*etfm)
	ax1t.set_ylabel('optimal ETF, gain='+str(np.round(gm,1))+', leak='+str(np.round(lm,4))+', BW='+str(np.round(bwm,1))+' Hz',y=0.4)
	etf_line.set_data(freq,np.abs(Hrej(freq, Twfs, tau, gm,lm)**2))
	bwm_line.set_xdata(bwm)

	im=ax2.imshow(optwfeglarr,origin='lower',extent=(np.log10(1-larr[0]),np.log10(1-larr[-1]),garr[0],garr[-1]),aspect=-1*np.log10(1-larr[-1])/garr[-1])
	imcb.set_ticklabels(np.linspace(int(round(np.nanmin(optwfeglarr))),int(round(np.nanmax(optwfeglarr))),len(cbticks)).astype(int))

	ax3.set_title('$\\Delta$g='+str(round(-1/gmargin,1))+', $\\Delta\\phi='+str(round(pmargin,1))+'^{\\circ}$',size=size)
	nyquistp_line.set_data(np.real(Holp),np.imag(Holp))
	nyquistn_line.set_data(np.real(Holn),np.imag(Holn))

	fig.suptitle('output WFE = '+str(int(round(np.min(wfearrm))))+'pm rms',size=size)
	fig.canvas.draw_idle()

Twfs_slider.on_changed(sliders_on_changed)
tau_slider.on_changed(sliders_on_changed)
wfeol_slider.on_changed(sliders_on_changed)
f0_slider.on_changed(sliders_on_changed)
pl_slider.on_changed(sliders_on_changed)
mv_slider.on_changed(sliders_on_changed)
pl_slider.on_changed(sliders_on_changed)
beta_slider.on_changed(sliders_on_changed)


plt.show()
