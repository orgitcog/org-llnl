#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Grace Li (li85)
Date: 8/17/2021

Contour plots to visualize energy perturbations of grains in the x-y plane.

"""

import numpy as np
from numpy import pi as pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import time

from LinearAnalysis.Grains import GrainsSolution
from LinearAnalysis.SingleMode import SingleModeSolution

## Define parameters
Lg_x = 200 #Grain length scale in x-direction
Lg_y = 200 #Grain length scale in y-direction
Ls = 10 #Interstitial space length scale
Lt = 0.5*Ls #Transition length scale
rho_g = 3.5 #Grain density
rho_s = 2.2 #Interstitial space density
M1 = 1.75 #Mach number

# %% Specify parameters for looping through

# Specify which parameter we are investigating looping through. The options are:
# "aspect_ratio", "aspect_ratio_fixed_Lx", "Ls-Lt_ratio", "Ls-Lt_constant", "Lg_x"
investigation = "Lg_x"

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
    
    #Specify aspect ratio parameters
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ratios = np.linspace(0.1, 1.0, 19)
    ratios = np.concatenate((ratios, np.array([4/25, 14, 4/9, 9/16, 16/25])))
    
    #Calculate the corresponding x and y direction grain widths
    Lg_xs = np.sqrt(area / np.array(ratios))
    
    #Specify the grid resolution for plotting by providing the number of points
    #in the x-direction if it has length sqrt_area
    resolution = 150
    
## --------------------------------------------------------------------------
## Investigation == "aspect_ratio_fixed_Lx" ------------------------------------------
## Vary the aspect ratio while fixing Lx. The ratio kx/ky = Lg_y / Lg_x

if investigation == "aspect_ratio_fixed_Lx":
    # In this case, the x-grain, interstitial and transition lengths are fixed
    Lg_x = 50
    Ls = 10
    Lt = 5
    
    #Specify aspect ratio parameters
    ratios = [1.0, 0.5, 0.7, 0.3, 0.9, 0.1, 0.8, 0.2]
    # ratios = [0.95, 0.55, 0.75, 0.35, 0.05, 0.15, 0.85, 0.25]
    Lg_ys = np.array(ratios) * Lg_x
    
    #Specify the grid resolution for plotting by providing the number of points
    #in the x-direction
    n_x_points = 150


## ---------------------------------------------------------------------------
## Investigation == "Ls-Lt_ratio" or "Ls-Lt_constant"--------------------------
## Vary the interstitial space width Ls with either scaled or constant transition Lt

if investigation in ["Ls-Lt_ratio", "Ls-Lt_constant"]:
    # In this case, the x-grain and y-grain lengths are fixed
    Lg_x = 200
    Lg_y = 200
    
    #Specify interstitial space width parameters
    Lss = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    Lss.reverse()
    
    #If investigation is "Ls-Lt_ratio", then Lt scales with Ls
    #Specify the ratio of the transition to interstitial Lt/Ls
    if investigation == "Ls-Lt_ratio":
        Lt_ratio = 0.5
    
    #If investigation is "Ls-Lt_constant", then the transition length is fixed
    if investigation == "Ls-Lt_constant":
        Lt = 5

    #Specify the grid resolution for plotting by providing the number of points
    #in the x-direction
    n_x_points = 150

## --------------------------------------------------------------------------
## Investigation == "Lg_x"----------------------------------------------------
## Vary the grain width in the x-direction

if investigation == "Lg_x":
    # In this case, the aspect ratio, interstitial and transition lengths are fixed
    aspect_ratio = 1 #aspect_ratio is defined as Lg_y / Lg_x
    Ls = 10
    Lt = 0.5*Ls
    #Grid for averaging has n_x_intervals_scaling * Lg_x points in x-direction
    n_x_points_scaling = 0.5
    
    #Specify x-grain width parameters
    Lg_xs = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    Lg_xs = [50, 100, 200, 300, 400, 500, 600]
    
    #Specify the grid resolution for plotting by providing the number of points
    #in the x-direction
    n_x_points = 150

## --------------------------------------------------------------------------
# Generate a single plot for a set of parameters
if investigation == "other":
    
    ## Define parameters
    Lg_x = 8000 #Grain length scale in x-direction
    Lg_y = 1000 #Grain length scale in y-direction
    Ls = 50 #Interstitial space length scale
    Lt = 0.5*Ls #Transition length scale
    rho_g = 3.5 #Grain density in g/cm3
    rho_s = 2.2 #Interstitial space density in g/cm3
    M1 = 1.75 #Mach number
    
    #Specify the grid resolution for plotting by providing the number of points
    #in the x-direction
    n_x_points = 150
    
    parameters = [1]
    
## --------------------------------------------------------------------------
# Additional Parameters

#Specify how many x-periods it has been since the shock exited the box 
shock_periods_later = 0
    
# Plot fontsize
fontsizes = {'XL': 20, 'L': 15, 'M': 12, 'S':10}

# Plot Colormap
cmap = plt.get_cmap('Spectral')
nbins = 50 #number of levels to use in plot

#Sort out which values we're investigating -----------------------------------

if investigation == "aspect_ratio":
    parameters = Lg_xs
if investigation == "aspect_ratio_fixed_Lx":
    parameters = Lg_ys
elif investigation in ["Ls-Lt_ratio", "Ls-Lt_constant"]:
    parameters = Lss
elif investigation == "Lg_x":
    parameters = Lg_xs

# %% Visualize the energy perturbations in x-y space
# for fixed t to give the shock front xs just exiting (right edge) of the plot

print('Visualize Energy Perturbations')

for parameter in parameters:
    
    start = time.time()
    
    #Sort out the grain parameters based on what kind of investigation we're doing
    if investigation == "aspect_ratio":
        Lg_x = parameter
        Lg_y = area / Lg_x
        n_x_points = int(resolution * Lg_x / sqrt_area)
    elif investigation == "aspect_ratio_fixed_Lx":
        Lg_y = parameter
    elif investigation == "Ls-Lt_ratio":
        Ls = parameter
        Lt = Lt_ratio * Ls
    elif investigation == "Ls-Lt_constant":
        Ls = parameter
    elif investigation == "Lg_x":
        Lg_x = parameter
        Lg_y = aspect_ratio * Lg_x
        if Lg_x > 200:
            n_x_points = int(math.ceil(Lg_x * n_x_points_scaling))
    
    ## Initialize GrainsSolution class
    sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1)
        
    # Generate 2 2D grids for the x & y bounds
    n_y_points = int(math.ceil(n_x_points * sim.Ly / sim.Lx))
    # Plot a single period in the larger of the x and y directions
    x0, x1 = 0, sim.Lx #initial and final x values
    y0, y1 = 0, sim.Ly #initial and final y values
    x = np.linspace(x0, x1, n_x_points)
    y = np.linspace(y0, y1, n_y_points)
    
    #Define meshgrid
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Define the desired middle of shock front location xs
    xs = x1 + sim.Lx * shock_periods_later
    #Calculate corresponding time
    t = xs / (sim.M2 * sim.a2)

    # Get z values to visualize: kinetic and total energy perturbation. 
    titles = ["Normalized\n Kinetic Energy Perturbation", 
              "Normalized\n Internal Energy Perturbation",
              r"Density Perturbation $\delta\rho / \rho_2$"]
    zs = [sim.tilde_KE(t, x_grid, y_grid), 
          sim.tilde_IE(t, x_grid, y_grid), 
          sim.tilde_rho(t, x_grid, y_grid)] 
    
    # #If we want to plot the total energy instead of the internal energy
    # titles = ["Normalized\n Kinetic Energy Perturbation", 
    #           "Normalized\n Total Energy Perturbation",
    #           r"Density Perturbation $\delta\rho / \rho_2$"]
    # zs = [sim.tilde_KE(t, x_grid, y_grid), 
    #       sim.tilde_TE(t, x_grid, y_grid), 
    #       sim.tilde_rho(t, x_grid, y_grid)] 
    
    # #If we want to plot the pre-shock density perturbation instead:
    # titles = ["Normalized Kinetic Energy\n" + r"Perturbation", 
    #           "Normalized Total Energy\n" + r"Perturbation",
    #           "Preshock Density\n" +  r"Perturbation $\delta\rho / \rho_1$"]
    # zs = [sim.tilde_KE(t, x_grid, y_grid), 
    #       sim.tilde_TE(t, x_grid, y_grid), 
    #       sim.fourier_delta_rho(x, y) / sim.rho1] 
        
    #If needed, add periods in y for ease of visualizing the plot to scale
    n_y_periods = math.floor(x1 / sim.Ly)
    if n_y_periods > 1:
        y_grid_new = np.copy(y_grid)
        x_grid_new = np.copy(x_grid)
        zs_new = []
        for item in range(len(zs)):
            zs_new.append(np.copy(zs[item]))
        for i in range(1, n_y_periods):
            y_grid_new = np.concatenate((y_grid_new[:-1, :], y_grid + i*sim.Ly), axis=0)
            x_grid_new = np.concatenate((x_grid_new[:-1, :], x_grid), axis=0)
            for item in range(len(zs)):
                zs_new[item] = np.concatenate((zs_new[item][:-1, :], zs[item]), axis=0)
        y_grid = y_grid_new
        x_grid = x_grid_new
        zs = zs_new
    elif n_y_periods == 0:
        n_y_periods = 1
    
    #Plot the perturbation contour plots
    n_cols = 3
    fig, axs = plt.subplots(1, n_cols, figsize=(15,5.2))

    for index in range(n_cols):
        
        ax = axs[index] #specify axis
        z = zs[index] #specify z value

        # pick sensibpick sensible levels, and define a normalization
        # instance which takes data values and translates those into levels.
        levels = mpl.ticker.MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Plot contour plot
        cf = ax.contourf(x_grid, y_grid, z, levels = levels, cmap=cmap)
       
        # #Add a line for the middle of the shock front
        ax.plot([xs, xs], [y0, y1], color='k', linestyle='-', linewidth=0.5)
        
        #Formatting - colorbar
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cbar = fig.colorbar(cf, ax=ax, format=fmt)
        cbar.ax.yaxis.set_offset_position('left') 
        cbar.ax.yaxis.get_offset_text().set_fontsize(fontsizes['S'])
        cbar.ax.tick_params(labelsize=fontsizes['M'])
        cbar.update_ticks()
        
        #Formatting - axis/title/labels
        ax.set_title(titles[index], fontsize=fontsizes['L'])
        ax.tick_params(axis='x', labelsize=fontsizes['M'])
        ax.tick_params(axis='y', labelsize=fontsizes['M'])
        
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, n_y_periods*sim.Ly)
    
    #Formatting master title and naming save file
    if investigation == "aspect_ratio":
        suptitle = r"$\sqrt{Area}=$" + str(sqrt_area)
        suptitle = suptitle + r", Aspect Ratio=" + "{0:.4f}".format(Lg_y/Lg_x) 
    else:
        suptitle = r"$Lg_x=$" + str(Lg_x) + r", $Lg_y=$" + str(Lg_y) 
    suptitle = suptitle + ", Ls=" + str(Ls) + ", Lt=" + str(Lt)
    suptitle = suptitle + r", $\rho_g=$" + "{0:.2g}".format(rho_g) 
    suptitle = suptitle + r", $\rho_s=$" + "{0:.2g}".format(rho_s) + r", $M_1=$" + str(M1)
    suptitle = suptitle + "\nShock front is " + "{0:.2f}".format(shock_periods_later) + " periods past the right edge" 
    
    fig.suptitle(suptitle, fontsize = fontsizes['XL'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.85])

    savename = "plots/energy/" + investigation + "/Ls-" + str(Ls)
    savename = savename + "--Lt_ratio-" + "{0:.2g}".format(Lt/Ls)
    if investigation == "aspect_ratio":
        savename = savename + "--sqrt_area" + str(sqrt_area)
        savename = savename + "--aspect_ratio" + "{0:.2f}".format(Lg_y / Lg_x)
    else:
        savename = savename + "--Lg_x" + str(Lg_x) + "--Lg_y" + "{0:.1f}".format(Lg_y)
    savename = savename + ".png"
    plt.savefig(savename, facecolor='white', bbox_inches='tight')
    
    plt.close()
    
    print('Lg_x = ', Lg_x, ', Lg_y = ', Lg_y, ', Ls = ', Ls, ', Lt = ', Lt)
    print('Number x modes = ', sim.n_x_modes, ' and Number of y modes = ', sim.n_y_modes)
    print('Runtime in seconds: ', time.time() - start)