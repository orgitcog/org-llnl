#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Grace Li (li85)
Date: 8/10/2021

Plot integral of perturbed kinetic energy evaluated in space for 
x \in [0,Lx] and y \in [0,Ly], and plotted over time 

"""

import numpy as np
from numpy import pi as pi
# import matplotlib as mpl
import matplotlib.pyplot as plt
import math

import sys 
sys.path.append('..') #go up a directory to the GraceGrains parent directory
from LinearAnalysis.SingleMode import SingleModeSolution

# Define fixed parameters
epsilon_k = 0.1 #epsilon_k scales all values, so we leave it fixed at 1
gamma = (5/3)
str_gamma = '5/3' #to display gamma fraction in plot titles

#Whether to normalize the integrals or not
normalize = True


# %% Single plot of components of energy integral over time

M1 = 1.75
Ny = 20
Nx = 20
t_interval = 0.1 #time interval to view longwavelength plot

# Define Fourier mode simulation
sim = SingleModeSolution(Nx, Ny, M1, epsilon_k=epsilon_k, gamma=gamma)

## Calculate t value that would work to push the shock compltely out of the integration region
# This depends on Ny and M1 only, not Nx
#Calculate a t value that would just push the shock completely out of the plot box
if sim.longwavelength: 
    if sim.Nx == 0:
        max_delta_xs = sim.epsilon_k * sim.R * sim.M2**2 / (sim.ky * math.sqrt(1-sim.M2**2))
    else:
        max_delta_xs = np.abs(np.real(sim.delta_xs(sim.phi / sim.omega)))
else:
    max_delta_xs = np.abs(np.real(sim.delta_xs((pi / 2) * sim.omega)))
t_shock = (sim.Lx + max_delta_xs) / (sim.M2 * sim.a2)
print("\nNy = ", Ny, ", Nx = ", Nx, ", and M1 = ", M1)
print('Dimensional time for shock to completely exit box: ', t_shock)

## Define the time range to plot
n_points = 100
# View t_interval if long wavelength condition
t_interval = 0.05
if sim.longwavelength:
    ts = np.linspace(t_shock, t_shock + t_interval, n_points) 
#View 2 periods if the short wavelength condition
else:
    if sim.Ny == 0:
        period = 2*pi / (sim.omega + np.imag(sim.c) * sim.M2 * sim.a2)
    else:
        period = 2*pi / (sim.omega + sim.ky * np.imag(sim.eta) * sim.M2 * sim.a2)
    ts = np.linspace(t_shock, t_shock + 2*period, n_points) 
        

## Calculate the analytic KE integral as a function of t and compare
# We will plot and compare the sonic, vortex, mixed, and total integrals

# integral_s = []
# integral_v = []
# integral_mix = []
# integral_total = []
# for t in ts:
#     integral_s.append(sim.integral_KE_s(t))
#     integral_v.append(sim.integral_KE_v())
#     integral_mix.append(sim.integral_KE_mix(t))
#     integral_total.append(sim.integral_KE(t))
    
integral_s = sim.integral_KE_s(ts)
integral_v = sim.integral_KE_v() * np.ones(ts.shape)
integral_mix = sim.integral_KE_mix(ts)
integral_total = sim.integral_KE(ts)

#Normalize if specified
if normalize:
    baseline_KE = 0.5 * sim.rho2 * sim.U**2
    integral_s = np.array(integral_s) / baseline_KE
    integral_v = np.array(integral_v) / baseline_KE
    integral_mix = np.array(integral_mix) / baseline_KE
    integral_total = np.array(integral_total) / baseline_KE


#Plot the KE integral over time
fig, ax = plt.subplots(figsize=(7,5))
# ax.plot(ts, integral_s, label='sonic')
# ax.plot(ts, integral_v, label='vortex')
# ax.plot(ts, integral_mix, label='mixed')
# ax.plot(ts, np.array(integral_s) + np.array(integral_v), label='sonic+vortex')
ax.plot(ts, integral_total, label='total')
ax.set_xlabel('Time t')
# plt.legend()
heading = "Normalized Integral of Total Kinetic Energy\n"
plt.title(heading + 'kx/ky = ' + "{0:.2g}".format(Nx/Ny) + ', M1 = ' + str(M1))
# plt.title(heading + 'Nx = ' + str(Nx) + ', Ny = ' + str(Ny) + ', M1 = ' + str(M1))
if normalize:
    ax.set_ylabel('Normalized Kinetic Energy Integral')
else:
    ax.set_ylabel('Kinetic Energy Integral')

test = sim.long_time_average_integral_KE(component = 'total')
print(test)

# %% Plot components of kinetic energy integral over time

Nys = [0,5,20,100] #[5,10,20,100]
t_interval = {'5':0.3, '10':0.2, '20':0.1} #how long of a time interval to plot based on Ny
M1s = [1.5, 1.75, 2.0]

for Ny in Nys:
    if Ny == 100:
        Nxs = np.array(range(0,math.floor(1.3*10)+1)) * 10
    elif Ny == 0:
        Nxs = [5,20,100]
    else:
        Nxs = list(range(0,math.floor(1.3*Ny)+1))
    for Nx in Nxs:
        for M1 in M1s:
            
            # Define Fourier mode simulation
            sim = SingleModeSolution(Nx, Ny, M1, epsilon_k=epsilon_k, gamma=gamma)
            
            ## Calculate t value that would work to push the shock compltely out of the integration region
            # This depends on Ny and M1 only, not Nx
            #Calculate a t value that would just push the shock completely out of the plot box
            if sim.longwavelength: 
                if sim.Nx == 0:
                    max_delta_xs = sim.epsilon_k * sim.R * sim.M2**2 / (sim.ky * math.sqrt(1-sim.M2**2))
                else:
                    max_delta_xs = np.abs(np.real(sim.delta_xs(sim.phi / sim.omega)))
            else:
                max_delta_xs = np.abs(np.real(sim.delta_xs((pi / 2) * sim.omega)))
            t_shock = (sim.Lx + max_delta_xs) / (sim.M2 * sim.a2)
            print("\nNy = ", Ny, ", Nx = ", Nx, ", and M1 = ", M1)
            print('Dimensional time for shock to completely exit box: ', t_shock)
            
            ## Define the dimensional time range to plot
            n_points = 100
            #Pick a suitable time to see the initial drop off for the longwavelength case
            interval = 0.05
            if str(Ny) in t_interval:
                interval = t_interval[str(Ny)]
            if sim.longwavelength:
                ts = np.linspace(t_shock, t_shock + interval, n_points) 
                
            #View 2 periods if the short wavelength condition
            else:
                if sim.Ny == 0:
                    period = np.imag(sim.c) * sim.M2*sim.a2 + sim.omega
                else:
                    period = np.imag(sim.eta)*sim.ky * sim.M2*sim.a2 + sim.omega
                period = np.abs(2* pi / period)
                ts = np.linspace(t_shock, t_shock + 2*period, n_points) 
            
            ## Calculate the analytic KE integral as a function of t and compare
            # We can look at the sonic, vortex, mixed, and total integrals
            
            #Calculate baseline KE for normalization
            baseline_KE = 0.5 * sim.rho2 * sim.U**2
            
            # integral_s = []
            # integral_v = []
            # integral_mix = []
            # integral_total = []
            # for t in ts:
            #     integral_s.append(sim.integral_KE_s(t))
            #     integral_v.append(sim.integral_KE_v())
            #     integral_mix.append(sim.integral_KE_mix(t))
            #     integral_total.append(sim.integral_KE(t))
            
            integral_s = sim.integral_KE_s(ts)
            integral_v = sim.integral_KE_v() * np.ones(ts.shape)
            integral_mix = sim.integral_KE_mix(ts)
            integral_total = sim.integral_KE(ts)
                
            #Normalize if specified
            if normalize:
                baseline_KE = 0.5 * sim.rho2 * sim.U**2
                integral_s = np.array(integral_s) / baseline_KE
                integral_v = np.array(integral_v) / baseline_KE
                integral_mix = np.array(integral_mix) / baseline_KE
                integral_total = np.array(integral_total) / baseline_KE
            
            #Plot energy fluctation
            fig, ax = plt.subplots(figsize=(7,5))
            # ax.plot(t_dim, integral_s, label='sonic')
            # ax.plot(t_dim, integral_v, label='vortex')
            # ax.plot(t_dim, integral_mix, label='mixed')
            ax.plot(ts, np.array(integral_s) + np.array(integral_v), label='sonic+vortex')
            ax.plot(ts, integral_total, label='total')
            ax.set_xlabel('Time t')
            
            plt.legend()
            plt.title('Nx = ' + str(Nx) + ', Ny = ' + str(Ny) + ', M1 = ' + str(M1))
            
            if normalize:
                ax.set_ylabel('Normalized Perturbed Kinetic Energy Integral')
                filename = 'plots/KE_integral_in_time_normalized/'
            else:
                ax.set_ylabel('Perturbed Kinetic Energy Integral')
                filename = 'plots/KE_integral_in_time/'
            
            filename = filename +'Ny-'+ str(Ny) +'--Nx-' +str(Nx) +'--M1-'+ str(M1) +'.png'
            plt.savefig(filename, facecolor='white', bbox_inches='tight')
            
            plt.close()


