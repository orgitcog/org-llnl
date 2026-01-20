#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Grace Li (li85)
Date: 8/10/2021

Plot long-time average of the integral of perturbed kinetic energy 
evaluated in space for x \in [0,Lx] and y \in [0,Ly].
Varying parameters of kx/ky and the Mach number M1.

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

#Specify whether to normalize the integrals or not
normalize = True
    
# %% Plot long-time average of kinetic energy integral vs ratio kx/ky

## Plot Integral vs ratio kx/ky with Ny, M1 constant
fontsizes = {'XL': 20, 'L': 15, 'M': 13, 'S':12}

#Define integral types to plot
integral_types = ['total', 'sonic', 'vortex', 'mixed']

#Define the Mach numbers to plot
M1s = [1.5, 1.75, 2.0]

# ## Calculate M1 needed to give transition at kx/ky=0.55 by finding root numerically
# from scipy import optimize
# #Define function to find zero of to get M1**2
#     value = (0.55)**2 * ( (gamma+1)*(M12) / ((gamma-1)*(M12) + 2) )**2 + 1
#     value = value - (2*gamma*(M12) - gamma + 1) / ( (gamma-1)*(M12) + 2 )
#     return value
# M12 = optimize.root_scalar(f, bracket=[4,9])  
# M1 = math.sqrt(M12.root)
# M1s = [M1]

#Define Ny values - we will overlay multiple curves for values of Ny
Nys = [500, 100, 50, 10]

# integral_types = ['total']
# M1s = [1.75]

for integral_type in integral_types:
    plt.close('all')
    for M1 in M1s:
        #Calculate long wavelength transition, which just depends on M1 and gamma not kx or ky
        sim = SingleModeSolution(Nx=1, Ny=1, M1=M1, epsilon_k = epsilon_k, gamma=gamma)
        wavelength_transition = math.sqrt(1-sim.M2**2) / (sim.R*sim.M2)
    
        #Define lists to store ratios and integrals for plotting for each Ny
        all_ratios = []
        all_integrals = []
    
        #For each Ny, calculate the desired integral
        for Ny in Nys:
            Nxs = list(range(1,math.floor(1.4*Ny)+1))
            ratio = np.array(Nxs) / Ny 
            all_ratios.append(ratio)
            
            #Calculate and store the values of the KE integrals for each kx/ky ratio
            integral = []
            for Nx in Nxs:
                sim = SingleModeSolution(Nx, Ny, M1, epsilon_k=epsilon_k, gamma=gamma)
                if normalize:
                    baseline_KE = 0.5 * sim.rho2 * sim.U**2
                    integral.append(sim.long_time_average_integral_KE(integral_type) / baseline_KE)
                else:
                    integral.append(sim.long_time_average_integral_KE(integral_type))
            all_integrals.append(integral)
            
        ## Generate plot of KE integral long time average
        fig, ax = plt.subplots(figsize=(8,6))
        
        #Plot each of the integral vs kx/ky ratio curves
        for i in range(len(Nys)):
            if i in [0,1]:
                ax.plot(all_ratios[i], all_integrals[i], '-', label = "Ny = " + str(Nys[i]))
            else:
                ax.plot(all_ratios[i], all_integrals[i], '.', label = "Ny = " + str(Nys[i]))
        # Add a vertical line for the wavelength transition
        plt.axvline(x = wavelength_transition, color='k')
        # Add transition annotation
        annotation_y = all_integrals[-1][-2]
        if integral_type == 'sonic':
            annotation_y = 0.5 * (max(all_integrals[0]) + min(all_integrals[0]))
        if integral_type == 'mixed':
            annotation_y = min( [value for integral in all_integrals for value in integral])
        ax.annotate('long-wavelength\n transition', 
                    xy=(wavelength_transition, annotation_y), xycoords = 'data', 
                    xytext=(-20,0), textcoords = 'offset points',
                    arrowprops=dict(facecolor='black', width = 4, headwidth = 12),
                    horizontalalignment='right', verticalalignment='center',
                    fontsize = fontsizes['M']
                    )
            
        ax.set_xlabel('Wavenumber Ratio (kx/ky)', fontsize=fontsizes['M'])
        ax.tick_params(axis='x', labelsize=fontsizes['S'])
        ax.tick_params(axis='y', labelsize=fontsizes['S'])
        plt.legend(fontsize=fontsizes['M'])
        title = integral_type.capitalize() + ' Kinetic Energy Integral\n'
        title = title + 'M1 = ' + str(M1) + r', $\epsilon_k$ = ' + str(epsilon_k) + r', $\gamma$ = ' + str_gamma
        plt.title(title, fontsize=fontsizes['L'])
    
        if normalize:
            ax.set_ylabel('Normalized Long-Time Average Kinetic Energy Integral', fontsize=fontsizes['M'])
            filename = 'plots/KE_integral_average_normalized/'
        else:
            ax.set_ylabel('Long-Time Average Kinetic Energy Integral', fontsize=fontsizes['M'])
            filename = 'plots/KE_integral_long_time_average/'
    
        filename = filename + integral_type + '--ratio--M1-'+ str(M1) + '.png'
        plt.savefig(filename, facecolor='white', bbox_inches='tight')


        
# %% Plot long-time average of kinetic energy integral vs Mach Number
## Plot Integral vs Mach number M1 with Ny and ratio kx/ky constant
fontsizes = {'XL': 20, 'L': 15, 'M': 13, 'S':12}

#Define integral types to plot
integral_types = ['total', 'sonic', 'vortex', 'mixed']

#Define the Mach numbers to plot
# M1s = np.linspace(1.5, 5, 15)
M1s = np.linspace(1.1, 2.5, 281)

#Define kx/ky ratio values - we will overlay multiple curves for values of Ny
ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, "infty"]
Ny = 100 #define fixed Ny

for integral_type in integral_types:
    plt.close('all')
    #Define lists to integrals for plotting for each ratio
    all_integrals = []
    
    for ratio in ratios:
        
        if ratio == "infty":
            Nx = Ny
        else:
            Nx = ratio * Ny
        
        #For each Mach number M1, calculate the desired integral and store
        integral = []
        for M1 in M1s:
            if ratio == "infty":
                sim = SingleModeSolution(Nx, 0, M1, epsilon_k=epsilon_k, gamma=gamma)
            else:
                sim = SingleModeSolution(Nx, Ny, M1, epsilon_k=epsilon_k, gamma=gamma)
                
            if normalize:
                baseline_KE = 0.5 * sim.rho2 * sim.U**2
                integral.append(sim.long_time_average_integral_KE(integral_type) / baseline_KE)
            else:
                integral.append(sim.long_time_average_integral_KE(integral_type))
        all_integrals.append(integral)
            
    # Generate plot
    fig, ax = plt.subplots(figsize=(8,6))
    
    #Plot each of the integral vs kx/ky ratio curves
    for i in range(len(ratios)):
        if ratios[i] == "infty":
            ax.plot(M1s, all_integrals[i], '-', label = r"kx/ky = $\infty$")
        else:
            ax.plot(M1s, all_integrals[i], '-', label = "kx/ky = " + str(ratios[i]))
    
    #Plot labels and formatting
    ax.set_xlabel(r"Mach Number ($M_1$)", fontsize=fontsizes['M'])
    ax.tick_params(axis='x', labelsize=fontsizes['S'])
    ax.tick_params(axis='y', labelsize=fontsizes['S'])
    plt.legend(fontsize=fontsizes['M'])
    title = integral_type.capitalize() + ' Kinetic Energy Integral\n'
    title = title + 'Ny = ' + str(Ny) + r', $\epsilon_k$ = ' + str(epsilon_k) + r', $\gamma$ = ' + str_gamma
    plt.title(title, fontsize=fontsizes['L'])

    if normalize:
        ax.set_ylabel('Normalized Long-Time Average Kinetic Energy Integral', fontsize=fontsizes['M'])
        filename = 'plots/KE_integral_average_normalized/'
    else:
        ax.set_ylabel('Long-Time Average Kinetic Energy Integral', fontsize=fontsizes['M'])
        filename = 'plots/KE_integral_long_time_average/'
    
    #Save figure
    filename = filename + integral_type + '--M1--Ny-'+ str(Ny) + '.png'
    plt.savefig(filename, facecolor='white', bbox_inches='tight')