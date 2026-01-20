#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Grace Li (li85)
Date: 8/23/2021

Python script to generate the percent perturbation (calculated as an average
in space) at each time coordinate of the magnitudes of energy and momentum quantities 
while scanning various parameters including aspect ratio, interstitial width, 
interstitial density, and grain size.

The internal energy space integral is calculated analytically instead, so see
grains_IE_over_time.py instead for that.
                             
After running this script, you can use grains_integral_plots.py to generate
the plots of the resulting time-averaged percent perturbations vs the parameters
scanned in the investigation(s).

This script generates and saves a matfile containing the space-integral at each
time point, and also a plot of the space-integral over time for each set of parameters.

Change the "integration_quantity" (which magnitude of energy/momentum to look at),
whether perturbations are measured from the lab reference frame ("lab_frame" True or False),
which "investigation"

Then, control F the "investigation" type you want and under the "if investigation =="
statement, change the particular parameters of the investigation. THe parameters
include a "resolution" parameter, which determines the number of uniform intervals
in x we will use to approximate the space integral. A Riemann-sum of bottom-left
boxes is used to approximate the integral.

"""

import numpy as np
from numpy import pi as pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import time
from scipy.integrate import dblquad
from scipy import io

from LinearAnalysis.Grains import GrainsSolution
# from LinearAnalysis.SingleMode import SingleModeSolution

## Define parameters
# Lg_x = 200 #Grain length scale in x-direction
# Lg_y = 200 #Grain length scale in y-direction
# Ls = 10 #Interstitial space length scale
# Lt = 0.5*Ls #Transition length scale

rho_diamond = 3.5 
rho_graphite = 2.2
M1 = 1.75 #Mach number
rho_reference = rho_diamond


# %% Specify parameters for looping through

# Specify which integral of magnitude we are looking at. The options are:
# "Mtotal" (magnitude of total momentum with both x and y components),
# "Mx" (absolute value of x-momentum), "My" (absolute value of y-momentum),
# "KE" (absolute value of kinetic energy)
integration_quantity = "KE"

# Specify if we want the momentum in the lab reference frame 
# or not (calculates in the postshock stationary frame)
# For Mtotal and Mx only - not for Mx
lab_frame = True

# Specify which parameter we are investigating. The options are:
# "aspect_ratio", "aspect_ratio-fixed_Lx", "Lg_x", "rho_s",
# "Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain",
# "conserve_density-Lt_ratio", "conserve_density-Lt_constant"
investigation = "Ls-Lt_ratio"

# Specify if perturbation should be calculated from mean density
density_perturb_from_mean = False

## --------------------------------------------------------------------------
## Investigation == "aspect_ratio" ------------------------------------------
## Vary the aspect ratio while fixing the grain area Lg-x x Lg_y. 
## The ratio kx/ky = Lg_y / Lg_x. Lg_y = ratio*Lg_x

if investigation == "aspect_ratio":
    # In this case, the grain area, interstitial and transition lengths are fixed
    sqrt_area = 200
    area = sqrt_area**2
    Ls = 10
    Lt = 0.5*Ls
    rho_g = rho_diamond
    rho_s = rho_graphite
    
    #Specify aspect ratio parameters
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ratios = np.linspace(0.05, 1, 20)
    ratios = np.concatenate( (ratios, np.array([4/25, 4/9, 9/16, 16/25])) ) 
    # ratios = [1.0, 0.5, 0.1]
    
    #Calculate the corresponding x and y direction grain widths
    Lg_xs = np.sqrt(area / np.array(ratios))
    
    # Number of points in x direction in grid to average over
    resolutions = [300]
    # resolutions = [200, 300, 400]
    
## --------------------------------------------------------------------------
## Investigation == "aspect_ratio-fixed_Lx" ------------------------------------------
## Vary the aspect ratio while fixing Lx. The ratio kx/ky = Lg_y / Lg_x

if investigation == "aspect_ratio-fixed_Lx":
    # In this case, the x-grain, interstitial and transition lengths are fixed
    Lg_x = 50
    Ls = 10
    Lt = 0.5*Ls
    rho_g = rho_diamond
    rho_s = rho_graphite
    
    #Specify aspect ratio parameters
    ratios = [1.0, 0.5, 0.7, 0.3, 0.9, 0.1, 0.8, 0.2]
    # ratios = [0.95, 0.55, 0.75, 0.35, 0.05, 0.15, 0.85, 0.25]
    Lg_ys = np.array(ratios) * Lg_x
    
    # Number of points in x direction in grid to average over
    resolutions = [200, 300]

## ---------------------------------------------------------------------------
## Investigation == "Ls-Lt_ratio", "Ls-Lt_constant"--------------------------
## Vary the interstitial space width Ls with either scaled or constant transition Lt
## Scale the grains as well so that the total number of grains in an area is the same
## i.e. the periods Lx and Ly are constant

if investigation in ["Ls-Lt_ratio", "Ls-Lt_constant"]:
    # In this case, the period in x and y is fixed. We will use the original grain
    # lengths to calculate the new ones as the interstitial space Ls gets wider
    Lg_x_original = 200
    Lg_y_original = 200
    Ls_original = 4
    rho_g = rho_diamond
    rho_s = rho_graphite
    
    #If investigation is "Ls-Lt_ratio-fixed_grain", then Lt scales with Ls
    #Specify the ratio of the transition to interstitial Lt/Ls
    if investigation == "Ls-Lt_ratio":
        Lt_ratio = 0.5
        #Specify interstitial space width parameters
        Lss = [4, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        Lss.reverse()
    
    #If investigation is "Ls-Lt_constant-fixed_grain", then the transition length is fixed
    if investigation == "Ls-Lt_constant":
        Lt = 5
        #Specify interstitial space width parameters
        Lss = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        Lss.reverse()

    # Number of points in x direction in grid to average over
    resolutions = [300]


## ---------------------------------------------------------------------------
## Investigation == "Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain"--------------------------
## Vary the interstitial space width Ls with either scaled or constant transition Lt
## Keep the grain size fixed in this case

if investigation in ["Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain"]:
    # In this case, the x-grain and y-grain lengths are fixed
    Lg_x_original = 200
    Lg_y_original = 200
    rho_g = rho_diamond
    rho_s = rho_graphite
    
    #If investigation is "Ls-Lt_ratio-fixed_grain", then Lt scales with Ls
    #Specify the ratio of the transition to interstitial Lt/Ls
    if investigation == "Ls-Lt_ratio-fixed_grain":
        Lt_ratio = 0.5
        #Specify interstitial space width parameters
        Lss = [4, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        Lss.reverse()
    
    #If investigation is "Ls-Lt_constant-fixed_grain", then the transition length is fixed
    if investigation == "Ls-Lt_constant-fixed_grain":
        Lt = 5
        #Specify interstitial space width parameters
        Lss = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        Lss.reverse()

    # Number of points in x direction in grid to average over
    resolutions = [200]

## --------------------------------------------------------------------------
## Investigation == "Lg_x"----------------------------------------------------
## Vary the grain width in the x-direction

if investigation == "Lg_x":
    # In this case, the aspect ratio, interstitial and transition lengths are fixed
    aspect_ratio = 1 #aspect_ratio is defined as Lg_y / Lg_x
    Ls = 10
    Lt = 5
    rho_g = rho_diamond
    rho_s = rho_graphite

    #Grid for averaging has n_x_intervals_scaling * Lg_x points in x-direction
    n_x_intervals_scaling = 0.5
    
    #Specify x-grain width parameters
    Lg_xs = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Number of points in x direction in grid to average over
    resolutions = [200]

## --------------------------------------------------------------------------
## Investigation == "rho_s"----------------------------------------------------
## Vary the density in the interstitial space

if investigation == "rho_s":
    # In this case, the grain and interstitial space dimensions are fixed
    Lg_x = 200 #Grain length scale in x-direction
    Lg_y = 200 #Grain length scale in y-direction
    Ls = 10 #Interstitial space length scale
    Lt = 0.5*Ls #Transition length scale
    rho_g = rho_diamond
    
    #Specify interstitial space density parameters
    rho_ss = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4]
    rho_ss.reverse()
    
    # Number of points in x direction in grid to average over
    resolutions = [300]

## --------------------------------------------------------------------------
## Investigation in ["conserve_density-Lt_ratio", "conserve_density-Lt_constant"]
## Scale the interstitial space wider and the interstitial density higher
## while keeping the average density (in 1-D) conserved.
## We also conserve the number of grains, so Ls + Lg = Ls_new + Lg_new.
## Transition region scales with interstitial space if "conserve_density-Lt_ratio"
## Transition region is fixed if "conserve_density-Lt_constant"

if investigation in ["conserve_density-Lt_ratio", "conserve_density-Lt_constant"]:
    # In this case, the grain dimensions are fixed
    Lg_x_original = 200 #Grain length scale in x-direction
    Lg_y_original = 200 #Grain length scale in y-direction
    Ls_original = 4 #Interstitial space width
    rho_g = rho_diamond
    
    #Specify interstitial space width parameters
    Lss = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    Lss.reverse()
    
    #Specify the ratio Lt/Ls if we are scaling Lt with Ls
    if investigation == "conserve_density-Lt_ratio":
        Lt_ratio = 0.5
        #Specify interstitial space width parameters
        Lss = [4, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        Lss.reverse()

    #Otherwise specify the transition width Lt if we are keeping it constant
    elif investigation == "conserve_density-Lt_constant":
        Lt = 5
        #Specify interstitial space width parameters
        Lss = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        Lss.reverse()
    
    # Number of points in x direction in grid to average over
    resolutions = [300]

## --------------------------------------------------------------------------
if investigation == "other":
    ## Define parameters
    Lg_x = 800 #Grain length scale in x-direction
    Lg_y = 100 #Grain length scale in y-direction
    Ls = 4 #Interstitial space length scale
    Lt = 0.5*Ls #Transition length scale
    rho_g = rho_diamond #Grain density in g/cm3
    rho_s = rho_graphite #Interstitial space density in g/cm3
    M1 = 1.75 #Mach number
    
    # Number of points in x direction in grid to average over
    resolutions = [400, 500]
    
    parameters = [1]

## --------------------------------------------------------------------------
## Specify additional parameters ---------------------------------------------

# Number of time points to calculate
n_time_points =  30
# n_time_points = 100

fontsizes = {'XL': 20, 'L': 15, 'M': 13, 'S':12}

#Sort out which values we're investigating -----------------------------------

if investigation == "aspect_ratio":
    parameters = Lg_xs
if investigation == "aspect_ratio-fixed_Lx":
    parameters = Lg_ys
elif investigation in ["Ls-Lt_ratio", "Ls-Lt_constant"]:
    parameters = Lss
elif investigation in ["Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain"]:
    parameters = Lss
elif investigation == "Lg_x":
    parameters = Lg_xs
elif investigation == "rho_s":
    parameters = rho_ss
elif investigation in ["conserve_density-Lt_ratio", "conserve_density-Lt_constant"]:
    parameters = Lss
    
# %% Integrate by averaging grid over the Lx x Ly region (2D midpoint rule)

if integration_quantity == "Mx":
    print("|Mx| Integral over Time (by averaging over grid)")
elif integration_quantity == "My":
    print("|My| Integral over Time (by averaging over grid)")
elif integration_quantity == "Mtotal":
    print("|M-total| Integral over Time (by averaging over grid)")
elif integration_quantity == "KE":
    print("|KE| Integral over Time (by averaging over grid)")
    
print('Investigating ', investigation)

for parameter in parameters:
    
    #Sort out the grain parameters based on what kind of investigation we're doing
    if investigation == "aspect_ratio":
        Lg_x = parameter
        Lg_y = area / Lg_x
    elif investigation == "aspect_ratio-fixed_Lx":
        Lg_y = parameter
    elif investigation == "Ls-Lt_ratio":
        Ls = parameter
        Lt = Lt_ratio * Ls
        Lg_x = Lg_x_original + Ls_original*(1 + 2*Lt_ratio) - Ls*(1 + 2*Lt_ratio)
        Lg_y = Lg_y_original + Ls_original*(1 + 2*Lt_ratio) - Ls*(1 + 2*Lt_ratio)
    elif investigation == "Ls-Lt_constant":
        Ls = parameter
        Lg_x = Lg_x_original + Ls_original - Ls
        Lg_y = Lg_y_original + Ls_original - Ls
    elif investigation in ["Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain"]:
        Ls = parameter
        if investigation == "Ls-Lt_ratio-fixed_grain":
            Lt = Lt_ratio * Ls
    elif investigation == "Lg_x":
        Lg_x = parameter
        Lg_y = aspect_ratio * Lg_x
        if Lg_x > 200:
            resolutions = [int(math.ceil(Lg_x * n_x_intervals_scaling))]
    elif investigation == "rho_s":
        rho_s = parameter
    elif investigation == "conserve_density-Lt_ratio":
        Ls = parameter
        Lt = Lt_ratio * Ls
        Lg_x = Lg_x_original + Ls_original*(1 + 2*Lt_ratio) - Ls*(1 + 2*Lt_ratio)
        Lg_y = Lg_y_original + Ls_original*(1 + 2*Lt_ratio) - Ls*(1 + 2*Lt_ratio)
        rho_s = rho_g - (rho_g - rho_graphite) * (Ls_original / Ls)
    elif investigation == "conserve_density-Lt_constant":
        Ls = parameter
        Lg_x = Lg_x_original + Ls_original - Ls
        Lg_y = Lg_x_original + Ls_original - Ls
        rho_s = rho_g - (rho_g - rho_graphite) * (Ls_original / Ls)
    
    #Keep track of runtime
    start = time.time()
    
    #Define grains solution class
    sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1)
    
    #Print parameters and number of modes to get a sense of runtime
    print('\nLg_x =', Lg_x, ', Lg_y =', Lg_y, ', Ls =', Ls, ', Lt =', Lt)
    print('Number of x-modes =', sim.n_x_modes, 'and Number of y-modes =', sim.n_y_modes)

    #Define time interval to plot over
    R = sim.rho2 / sim.rho1
    xs = (1.5*sim.Lx + sim.Ls/2 + sim.Lt) / R
    t0 = xs * (1 + 1/sim.M2) / sim.a2
    period = sim.Lx / ( sim.D * (1 - sim.M2 / (sim.M2 +1)) )
    t1 = t0 + 2*period
    ts = np.linspace(t0, t1, n_time_points)
    
    # t0 = sim.Lx / (sim.M2 * sim.a2)
    # ts = np.linspace(t0, t0 + n_Lx_lengths*sim.Lx, n_time_points) 
    
    #Save the approx integrals for both resolutions tested so we can estimate error
    integrals = []
    
    for index in range(len(resolutions)):
        
        n_x_intervals = resolutions[index]
        
        if investigation == "aspect_ratio":
            n_x_intervals = int(math.ceil(n_x_intervals * Lg_x / sqrt_area))
        
        # Generate 2 2D grids for the x & y bounds to average the integration_quantity over
        n_y_intervals = int(math.ceil(n_x_intervals * sim.Ly / sim.Lx))
        
        dx = sim.Lx / n_x_intervals
        dy = sim.Ly / n_y_intervals
        
        x = np.linspace(0, sim.Lx - dx, n_x_intervals)
        y = np.linspace(0, sim.Ly - dy, n_y_intervals)
        
        x, y = np.meshgrid(x,y)
        
        #For each time calculate the integration_quantity on the grid and integral using
        # a Riemann sum of bottom-left boxs as an approximation
        integral = []
        
        for t in ts:
            if integration_quantity == "Mx":
                abs_value = np.abs(sim.tilde_Mx(t,x,y,lab_frame = lab_frame, perturb_from_mean = density_perturb_from_mean)) 
            elif integration_quantity == "My":
                abs_value = np.abs(sim.tilde_My(t,x,y, perturb_from_mean = density_perturb_from_mean))
            elif integration_quantity == "Mtotal":
                abs_value = sim.tilde_abs_M(t,x,y,lab_frame = lab_frame, perturb_from_mean = density_perturb_from_mean) 
            elif integration_quantity == "KE":
                abs_value = np.abs(sim.tilde_KE(t,x,y,lab_frame = lab_frame, perturb_from_mean = density_perturb_from_mean)) 
            space_integral = np.sum(abs_value) * dx*dy / (sim.Lx * sim.Ly)
            integral.append(space_integral)
            
        integrals.append(np.array(integral))
            
        #Specify base savename for figure and matfile of data
        savename = integration_quantity + ("_lab" * lab_frame) + "_over_time/"
        savename = savename + investigation + ("---density_perturb_from_mean" * density_perturb_from_mean)
        savename = savename + "/Ls-" + str(Ls)
        if investigation in ["Ls-Lt_constant", "Ls-Lt_constant-fixed_grain", "conserve_density-Lt_constant"]:
            savename = savename + "--Lt-" + "{0:.2g}".format(Lt)
        else:
            savename = savename + "--Lt_ratio-" + "{0:.2g}".format(Lt/Ls)
        if investigation == "aspect_ratio":
            savename = savename + "--sqrt_area" + str(sqrt_area)
            savename = savename + "--aspect_ratio" + "{0:.2f}".format(Lg_y/Lg_x)
            savename = savename + "--n_x_intervals-" + str(resolutions[index])
        else:
            savename = savename + "--Lg_x" + str(Lg_x) + "--Lg_y" + "{0:.1f}".format(Lg_y)
            if investigation == "rho_s":
                savename = savename + "--rho_s" + "{0:.1f}".format(rho_s)
            savename = savename + "--n_x_intervals-" + str(n_x_intervals)
        
        #Save matfile of calculated integrals
        save_dict = {
                    'integration_quantity': integration_quantity,
                    't': ts, 'integral': integral,
                    'Ls': Ls, 'Lt' : Lt,
                    'Lg_x': Lg_x, 'Lg_y': Lg_y,
                    'n_x_intervals': n_x_intervals
                     }
        
        matfilename = "matfiles/" + savename + ".mat"
        io.savemat(matfilename, save_dict)
        
        #Plot the integral over time
        fig, ax = plt.subplots(figsize=(7,6))
        ax.plot(ts, 100*np.array(integral))
        
        titles = {"Mx": "x-Momentum", "My": "y-Momentum", 
                  "Mtotal": "Total Momentum",
                  "KE": "Kinetic Energy"}
        
        title = "Percent Perturbation of Magnitude of " + titles[integration_quantity] + "\n"
        if investigation == "aspect_ratio":
            title = title + r"$\sqrt{Area}=$" + str(sqrt_area)
            title = title + r", $Lg_y/Lg_x=$" + "{0:.2f}".format(Lg_y/Lg_x) 
        else:
            title = title + r"$Lg_x=$" + str(Lg_x) + r", $Lg_y=$" + str(Lg_y)
        title = title + ", Ls=" + str(Ls) + ", Lt=" + str(Lt)
        title = title + "\n" + r"$\rho_g=$" + "{0:.2g}".format(rho_g) 
        title = title + r", $\rho_s=$" + "{0:.2g}".format(rho_s) + r", $M_1=$" + str(M1)
        plt.title(title, fontsize = fontsizes['L'])
        
        ax.set_xlabel('Time t', fontsize=fontsizes['M'])
        ax.set_ylabel("Average Percent Perturbation", fontsize=fontsizes['M'])
        ax.tick_params(axis='x', labelsize=fontsizes['S'])
        ax.tick_params(axis='y', labelsize=fontsizes['S'])
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    
        figname = "plots/" + savename + ".png"
        plt.savefig(figname, facecolor='white', bbox_inches='tight')
        
        print("Average over Lx x Ly grid with number of x-intervals = ", n_x_intervals)
        print('Runtime in seconds: ', time.time() - start,)
         
    if len(resolutions) == 2:
        rel_errors = np.abs((integrals[0] - integrals[1]) / integrals[1])
        print()
        print("Maximum relative error between resolutions = ", np.max(rel_errors))
        try: 
            print('Max ratio of calculated values = ', np.max(np.abs(integrals[0]/integrals[1])))
            print('Min ratio of calculated values = ', np.min(np.abs(integrals[0]/integrals[1])))
            print('Average ratio of calculated values = ', np.average(integrals[0]/integrals[1]))
        except:
            print('Could not calculate ratio of values for different resolutions')
     
    plt.close('all')