#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Grace Li (li85)
Date: 7/6/2021

linear_shock class to represent the equations in the
Velikovich et al. 2007 paper: 
''Shock front distortion and Richtmyer-Meshkov-type growth caused by 
a small preshock nonuniformity''
Phys. Plasmas 14, 072706 (2007)


In this code, we replace time t in the paper with t_new = ky*a2*t / 2*pi
so t = 2*pi * t_new / (ky*a2), and this gives omega_new = omega / (ky*a2) = kx*D/ky*a2 
to agree with dimensionless time.

We replace space x in the paper with x_new = ky * x / 2*pi, so x = 2*pi * x_new / ky

With this change, the density perturbation exponent depends on the ratio kx/ky.
The frequency omega also depends on thehe ratio kx/ky.
The remaining sonic/vortex and entropy exponents have no dependence on kx or ky alone
just on the ratio, which is preseent in the omega term.

The density, pressure, and velocity values and amplitude have no ky dependence,
only dependence on the ratio kx/ky.

However, the shock amplitude delta_xs has ky only dependence instead of dependence
on the ratio. This comes from the form of Eqn 32 and 35.

Instead of considering kx and ky, we will consider ky and the ratio kx/ky.

The 2*pi factor in the dimensionless variables allows us to plot from [0,1]
instead of [0, 2*pi] for guranteed full periods if kx and ky are integers

Plots of scalar quantities vs time, or integrated and vs parameters

"""

import numpy as np
from numpy import pi as pi
import matplotlib as mpl
import matplotlib.pyplot as plt
#from fractions import Fraction
import math
#import cmath
import equations as eq

# %% Plot eta vs ratio kx/ky

fontsizes = {'XL': 20, 'L': 15, 'M': 12, 'S':10}

epsilon_k = 1
gamma = (5/3)
Ny = 100

fig, ax = plt.subplots(figsize=(7,5))

for M1 in [1.75, 2.75, 3.75]:
    ratio = []
    eta = []    
    
    for Nx in range(0, 120+1):
        ratio.append(Nx/Ny)
        sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)
        eta.append(sim.eta)
    
    ax.plot(ratio, np.real(eta), label='real, M1=' + str(M1))
    ax.plot(ratio, np.imag(eta), label='imag, M1=' + str(M1))
    
plt.xlabel("Wavenumber Ratio kx/ky", fontsize=fontsizes['M'])
plt.ylabel(r"$\eta$")
plt.legend()
plt.title(r"$\eta$ vs kx/ky ratio", fontsize=fontsizes['L'])

# %% Plot maximum post-shock perturbations while varying kx/ky

fontsizes = {'XL': 20, 'L': 15, 'M': 12, 'S':10}

# Define parameters

ratios = np.linspace(0.001, 1.2, 2000)
Ny = 100
M1 = 1.75
epsilon_k = 1
gamma = (5/3)
gamma_str = '5/3'

#Loop through values of kx/ky ratios
#and get the magnitude of all the amplitudes
p0, delta_xs = [], []
v_x0_s, v_x0_v = [], []
v_y0_s, v_y0_v = [], []
rho_0_s, rho_0_e = [], []
for ratio in ratios:
    Nx = ratio*Ny
    sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)
    p0.append(np.abs(sim.p0))
    if sim.longwavelength:
        t = sim.phi / (sim.omega)
    else:
        t = 1 / (2*sim.omega)
    delta_xs.append(np.abs(sim.delta_xs(t)))
    v_x0_s.append(np.abs(sim.vx_0_s))
    v_x0_v.append(np.abs(sim.vx_0_v))
    v_y0_s.append(np.abs(sim.vy_0_s))
    v_y0_v.append(np.abs(sim.vy_0_v))
    rho_0_s.append(np.abs(sim.rho_0_s))
    rho_0_e.append(np.abs(sim.rho_0_e))    

amplitudes = [p0, delta_xs, v_x0_s, v_x0_v, v_y0_s, v_y0_v, rho_0_s, rho_0_e]
titles = [r"$\tilde{p}_0^{(s)}$", r"$\delta x_s$", r"$\tilde{v}_{x0}^{(s)}$", r"$\tilde{v}_{x0}^{(v)}$",
          r"$\tilde{v}_{y0}^{(s)}$", r"$\tilde{v}_{y0}^{(v)}$", r"$\tilde{\rho}_0^{(s)}$", r"$\tilde{\rho}_0^{(e)}$"]

fig, axs = plt.subplots(4, 2, figsize = (8,12), sharex=True)
for row in range(4):
    for col in range(2):
        index = 2*row + col
        ax = axs[row, col]
        ax.plot(ratios, amplitudes[index])
        ax.set_title(titles[index], fontsize = fontsizes['L'])
        ax.set_xlabel(r"$k_x/k_y$", fontsize = fontsizes['M'])
        ax.set_ylabel("Amplitude Magnitude", fontsize = fontsizes['M'])

plt.tick_params(axis='both', which='major', labelsize=fontsizes['S'])
    
suptitle = r"$N_y=$" +str(Ny) + r", $M_1=$" + str(M1)
suptitle = suptitle + r", $\epsilon_k=$" + str(epsilon_k) + r", $\gamma=$" + gamma_str
fig.suptitle(suptitle, fontsize = fontsizes['XL'])

fig.tight_layout(rect=[0, 0.03, 1, 0.96])

savename = 'plots/amplitudes/ratio--' + 'M1-' + str(M1) + '-Ny-' + str(Ny) + '.png'
plt.savefig(savename, facecolor='white', bbox_inches='tight')


# %% Plot shock amplitude vs time

ax_fontsize = 12
title_fontsize = 15

#Define parameters
Ny = 100
M1 = 1.7
epsilon_k = 1
gamma = (5/3)
gamma_str = '5/3'

#Time range
t0 = 0
t1 = 20
y = 0
t = np.linspace(t0, t1, 500)

## Long wavelength plot
Nx = 30
sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)
delta_xs = sim.delta_xs(t, y)

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(t, delta_xs)
plt.xlabel(r"$\bar{t} = t / k_y a_2$", fontsize=ax_fontsize)
plt.ylabel(r"Shock Amplitude $\delta x_s$")
title = r"Long Wavelength: M_1=" + str(M1) + r", $N_y$=" + str(Ny)
title = title + r", $k_x/k_y=$" + str(Nx/Ny) + r", $1/\omega\approx$" + "{:.2f}".format(1/sim.omega)
plt.title(title, fontsize=title_fontsize)
plt.show()

## Short wavelength plot
Nx = 70
sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)
delta_xs = sim.delta_xs(t, y)

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(t, delta_xs)
plt.xlabel(r"$\bar{t} = t / k_y a_2$", fontsize=ax_fontsize)
plt.ylabel(r"Shock Amplitude $\delta x_s$")
title = r"Short Wavelength: M_1=" + str(M1) + r", $N_y$=" + str(Ny)
title = title + r", $k_x/k_y=$" + str(Nx/Ny) + r", $1/\omega\approx$" + "{:.2f}".format(1/sim.omega)
plt.title(title, fontsize=title_fontsize)
plt.show()

#y range
y0 = 0
y1 = 2*pi
t = 0
y = np.linspace(y0, y1, 500)

## Long wavelength plot
Nx = 30
sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)
delta_xs = sim.delta_xs(t, y)

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(y, delta_xs)
plt.xlabel("y", fontsize=ax_fontsize)
plt.ylabel(r"Shock Amplitude $\delta x_s$")
title = r"Long Wavelength: M_1=" + str(M1) + r", $N_y$=" + str(Ny)
title = title + r", $k_x/k_y=$" + str(Nx/Ny) + r", $1/\omega\approx$" + "{:.2f}".format(1/sim.omega)
plt.title(title, fontsize=title_fontsize)
plt.show()

## Short wavelength plot
Nx = 70
sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)
delta_xs = sim.delta_xs(t, y)

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(y, delta_xs)
plt.xlabel("y", fontsize=ax_fontsize)
plt.ylabel(r"Shock Amplitude $\delta x_s$")
title = r"Short Wavelength: M_1=" + str(M1) + r", $N_y$=" + str(Ny)
title = title + r", $k_x/k_y=$" + str(Nx/Ny) + r", $1/\omega\approx$" + "{:.2f}".format(1/sim.omega)
plt.title(title, fontsize=title_fontsize)
plt.show()

# %% Plot amplitudes while varying kx/ky

fontsizes = {'XL': 20, 'L': 15, 'M': 12, 'S':10}

# Define parameters

ratios = np.linspace(0.001, 1.2, 2000)
Ny = 100
M1 = 1.7
epsilon_k = 1
gamma = (5/3)
gamma_str = '5/3'

#Loop through values of kx/ky ratios
#and get the magnitude of all the amplitudes
p0, delta_xs = [], []
v_x0_s, v_x0_v = [], []
v_y0_s, v_y0_v = [], []
rho_0_s, rho_0_e = [], []
for ratio in ratios:
    Nx = ratio*Ny
    sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)
    p0.append(np.abs(sim.p0))
    if sim.longwavelength:
        t = sim.phi / (sim.omega)
    else:
        t = 1 / (2*sim.omega)
    delta_xs.append(np.abs(sim.delta_xs(t)))
    v_x0_s.append(np.abs(sim.vx_0_s))
    v_x0_v.append(np.abs(sim.vx_0_v))
    v_y0_s.append(np.abs(sim.vy_0_s))
    v_y0_v.append(np.abs(sim.vy_0_v))
    rho_0_s.append(np.abs(sim.rho_0_s))
    rho_0_e.append(np.abs(sim.rho_0_e))    

amplitudes = [p0, delta_xs, v_x0_s, v_x0_v, v_y0_s, v_y0_v, rho_0_s, rho_0_e]
titles = [r"$\tilde{p}_0^{(s)}$", r"$\delta x_s$", r"$\tilde{v}_{x0}^{(s)}$", r"$\tilde{v}_{x0}^{(v)}$",
          r"$\tilde{v}_{y0}^{(s)}$", r"$\tilde{v}_{y0}^{(v)}$", r"$\tilde{\rho}_0^{(s)}$", r"$\tilde{\rho}_0^{(e)}$"]

fig, axs = plt.subplots(4, 2, figsize = (8,12), sharex=True)
for row in range(4):
    for col in range(2):
        index = 2*row + col
        ax = axs[row, col]
        ax.plot(ratios, amplitudes[index])
        ax.set_title(titles[index], fontsize = fontsizes['L'])
        ax.set_xlabel(r"$k_x/k_y$", fontsize = fontsizes['M'])
        ax.set_ylabel("Amplitude Magnitude", fontsize = fontsizes['M'])

plt.tick_params(axis='both', which='major', labelsize=fontsizes['S'])
    
suptitle = r"$N_y=$" +str(Ny) + r", $M_1=$" + str(M1)
suptitle = suptitle + r", $\epsilon_k=$" + str(epsilon_k) + r", $\gamma=$" + gamma_str
fig.suptitle(suptitle, fontsize = fontsizes['XL'])

fig.tight_layout(rect=[0, 0.03, 1, 0.96])

savename = 'plots/amplitudes/ratio--' + 'M1-' + str(M1) + '-Ny-' + str(Ny) + '.png'
plt.savefig(savename, facecolor='white', bbox_inches='tight')

# %% Plot amplitudes while varying Mach number
# for select short/long wavelengths
fontsizes = {'XL': 20, 'L': 15, 'M': 12, 'S':10}

# Define parameters
Nxs = [10, 30, 70, 90]
M1s = np.linspace(1.5, 2.5, 1000)
Ny = 100
epsilon_k = 1
gamma = (5/3)
gamma_str = '5/3'

#For the small selection of kx values
#Loop through values of M1s
#and get the magnitude of all the amplitudes
for Nx in Nxs:
    p0, delta_xs = [], []
    v_x0_s, v_x0_v = [], []
    v_y0_s, v_y0_v = [], []
    rho_0_s, rho_0_e = [], []
    for M1 in M1s:
        sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)
        p0.append(np.abs(sim.p0))
        if sim.longwavelength:
            t = sim.phi / (sim.omega)
        else:
            t = 1 / (2*sim.omega)
        delta_xs.append(np.abs(sim.delta_xs(t)))
        v_x0_s.append(np.abs(sim.v_x0_s))
        v_x0_v.append(np.abs(sim.v_x0_v))
        v_y0_s.append(np.abs(sim.v_y0_s))
        v_y0_v.append(np.abs(sim.v_y0_v))
        rho_0_s.append(np.abs(sim.rho_0_s))
        rho_0_e.append(np.abs(sim.rho_0_e))    
    
    amplitudes = [p0, delta_xs, v_x0_s, v_x0_v, v_y0_s, v_y0_v, rho_0_s, rho_0_e]
    titles = [r"$\tilde{p}_0^{(s)}$", r"$\delta x_s$", r"$\tilde{v}_{x0}^{(s)}$", r"$\tilde{v}_{x0}^{(v)}$",
              r"$\tilde{v}_{y0}^{(s)}$", r"$\tilde{v}_{y0}^{(v)}$", r"$\tilde{\rho}_0^{(s)}$", r"$\tilde{\rho}_0^{(e)}$"]
    
    fig, axs = plt.subplots(4, 2, figsize = (8,12), sharex=True)
    for row in range(4):
        for col in range(2):
            index = 2*row + col
            ax = axs[row, col]
            ax.plot(M1s, amplitudes[index])
            ax.set_title(titles[index], fontsize = fontsizes['L'])
            ax.set_xlabel(r"$M_1$", fontsize = fontsizes['M'])
            ax.set_ylabel("Amplitude Magnitude", fontsize = fontsizes['M'])
    
    plt.tick_params(axis='both', which='major', labelsize=fontsizes['S'])
        
    suptitle = r"$k_x/k_y=$" + str(Nx/Ny) + r", $N_y=$" +str(Ny)
    suptitle = suptitle + r", $\epsilon_k=$" + str(epsilon_k) + r", $\gamma=$" + gamma_str
    fig.suptitle(suptitle, fontsize = fontsizes['XL'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    savename = 'plots/amplitudes/M1--' + 'Nx-' + str(Nx) + '-Ny-' + str(Ny) + '.png'
    plt.savefig(savename, facecolor='white', bbox_inches='tight')
    
# %% Plot amplitudes while epsilon_k
# for select short/long wavelengths - to check scaling
fontsizes = {'XL': 20, 'L': 15, 'M': 12, 'S':10}

# Define parameters
Nxs = [10, 30, 70, 90]
log_epsilons = np.linspace(-10, 2, 1000)
Ny = 100
M1 = 1.7
gamma = (5/3)
gamma_str = '5/3'

#For the small selection of kx values
#Loop through values of M1s
#and get the magnitude of all the amplitudes
for Nx in Nxs:
    p0, delta_xs = [], []
    v_x0_s, v_x0_v = [], []
    v_y0_s, v_y0_v = [], []
    rho_0_s, rho_0_e = [], []
    for log_ep in log_epsilons:
        epsilon_k = 10**log_ep
        sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)
        p0.append(np.abs(sim.p0))
        if sim.longwavelength:
            t = sim.phi / (sim.omega)
        else:
            t = 1 / (2*sim.omega)
        delta_xs.append(np.abs(sim.delta_xs(t)))
        v_x0_s.append(np.abs(sim.v_x0_s))
        v_x0_v.append(np.abs(sim.v_x0_v))
        v_y0_s.append(np.abs(sim.v_y0_s))
        v_y0_v.append(np.abs(sim.v_y0_v))
        rho_0_s.append(np.abs(sim.rho_0_s))
        rho_0_e.append(np.abs(sim.rho_0_e))    
    
    amplitudes = [p0, delta_xs, v_x0_s, v_x0_v, v_y0_s, v_y0_v, rho_0_s, rho_0_e]
    titles = [r"$\tilde{p}_0^{(s)}$", r"$\delta x_s$", r"$\tilde{v}_{x0}^{(s)}$", r"$\tilde{v}_{x0}^{(v)}$",
              r"$\tilde{v}_{y0}^{(s)}$", r"$\tilde{v}_{y0}^{(v)}$", r"$\tilde{\rho}_0^{(s)}$", r"$\tilde{\rho}_0^{(e)}$"]
    
    fig, axs = plt.subplots(4, 2, figsize = (8,12), sharex=True)
    for row in range(4):
        for col in range(2):
            index = 2*row + col
            ax = axs[row, col]
            ax.plot(log_epsilons, np.log(amplitudes[index]))
            ax.set_title(titles[index], fontsize = fontsizes['L'])
            ax.set_xlabel(r"$log(\epsilon_k)$", fontsize = fontsizes['M'])
            ax.set_ylabel("log(Amplitude Magnitude)", fontsize = fontsizes['M'])
    
    plt.tick_params(axis='both', which='major', labelsize=fontsizes['S'])
        
    suptitle = r"$k_x/k_y=$" + str(Nx/Ny) +  r", $N_y=$" + str(Ny)
    suptitle = suptitle + r", $M_1=$" + str(M1) + r", $\gamma=$" + gamma_str
    fig.suptitle(suptitle, fontsize = fontsizes['XL'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    savename = 'plots/amplitudes/epsilon--' + 'Nx-' + str(Nx) 
    savename = savename + '-Ny-' + str(Ny) + '-M1-' + str(M1) + '.png'
    plt.savefig(savename, facecolor='white', bbox_inches='tight')
    
    
# %% Plot velocity amplitude
# To compare to Figure 9 in the paper
# x axis is D / a2 * t


ax_fontsize = 12
title_fontsize = 15

#Define parameters
Ny = 100
M1 = 10
epsilon_k = 1
gamma = (5/3)

#Time range fro plot
t_plot = np.linspace(0, 60, 500)

## Long wavelength plot kx/ky = 1/5
Nx = 20
t = t_plot * sim.a2 / sim.D
sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)

vx = sim.delta_vx(t, x=0, y=0)
vx_normalized = vx / (epsilon_k*sim.D)

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(t, vx_normalized)
plt.xlabel(r"$(D / a_2) t$", fontsize=ax_fontsize)
plt.ylabel(r"$\delta v_x / \epsilon_k D (t,x=0,y=0)$")
plt.title('Long wavelength', fontsize=title_fontsize)
plt.show()

## Short wavelength plot kx/ky = 1
Nx = 100
t = t_plot * sim.a2 / sim.D
sim = eq.linear_shock(Nx, Ny, M1, epsilon_k, gamma)

vx = sim.delta_vx(t, x=0, y=0)
vx_normalized = vx / (epsilon_k*sim.D)

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(t, vx_normalized)
plt.xlabel(r"$(D / a_2) t$", fontsize=ax_fontsize)
plt.ylabel(r"$\delta v_x / \epsilon_k D (t,x=0,y=0)$")
plt.title('Short wavelength', fontsize=title_fontsize)
plt.show()