#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Grace Li (li85)
Date: 8/10/2021

Contour plots to visualize perturbations of grains in the x-y plane.

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
Lg_x = 50 #Grain length scale in x-direction
Lg_y = 50 #Grain length scale in y-direction
Ls = 10 #Interstitial space length scale
Lt = 0.5*Ls #Transition length scale
rho_g = 3.5 #Grain density
rho_s = 2.2 #Interstitial space density
M1 = 1.75 #Mach number

# ## Define parameters
# Lg_x = 200 #Grain length scale in x-direction
# Lg_y = 200 #Grain length scale in y-direction
# Ls = 10 #Interstitial space length scale
# Lt = 5 #Transition length scale
# rho_g = 3.5 #Grain density
# rho_s = 2.2 #Interstitial space density
# M1 = 1.75 #Mach number

# ## Define parameters
# Lg_x = 1000 #Grain length scale in x-direction
# Lg_y = 1000 #Grain length scale in y-direction
# Ls = 10 #Interstitial space length scale
# Lt = 0.5*Ls #Transition length scale
# rho_g = 3.5 #Grain density
# rho_s = 2.2 #Interstitial space density
# M1 = 1.75 #Mach number

# Define the periods in x and y direction
Lx = Lg_x + Ls + 2*Lt
Ly = Lg_y + Ls + 2*Lt

# Initialize GrainsSolution class
sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1)

# # %% Visualize density in x-y space
# # for fixed t to give the shock front xs in the middle of the plot

# start = time.time()

# fontsizes = {'XL': 20, 'L': 15, 'M': 12, 'S':10}

# #Colormap
# cmap = plt.get_cmap('Spectral')
# nbins = 50 #number of levels to use in plot

# # Define grains solution class  
# sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt,
#                       rho_g, rho_s, M1)

# R = sim.rho2 / sim.rho1

# # Calculuate time value to give desired middle of the shock front location xs
# xs = (Lx + Lg_x/2 + Ls + 2*Lt)/R #desired middle of shock front x-coordinate
# t = xs / (sim.M2 * sim.a2)
# t1 = xs * (1/(sim.M2*sim.a2) + 1/sim.a2)
# t = t1
# xs = t1 * sim.M2 * sim.a2
# # t = t1 + sim.M2 *4

# # Generate 2 2D grids for the x & y bounds
# x0, x1 = 0, Lx + Lx/2 #initial and final x values
# y0, y1 = 0, Ly #initial and final y values

# x = np.linspace(x0, x1, 75)
# y = np.linspace(x0, x1, 50)

# # #Define the x coordinates for the meshgrid
# # n_points_grain = 20 #100
# # n_points_space = 10 #50
# # n_points_sonic = 10 #50
# # #post shock parts
# # sonic_edge = [xs - Lg_x/(2 * sim.M2), xs - Lg_x/(2 * sim.M2) + 3*Ls]
# # part1 = np.linspace(0, (Lg_x/2)/R, int(n_points_grain/2))
# # part2 = np.linspace(Lg_x/2 * 1/R, sonic_edge[1], n_points_space + n_points_sonic)
# # part3 = np.linspace(sonic_edge[1], (Lx + Lg_x/2)/R, n_points_grain)
# # part4 = np.linspace((Lx + Lg_x/2)/R, (Lx + Lg_x/2 + Lt + Ls/2)/R, n_points_space)
# # part5 = np.linspace((Lx + Lg_x/2 + Lt + Ls/2)/R, xs, int(n_points_grain/2))
# # x = np.concatenate((part1[:-1], part2[:-1], part3[:-1], part4[:-1], part5[:-1]))
# # #pre shock parts
# # part6 = np.linspace(xs, xs + Lg_x/2, int(n_points_grain/2))
# # part7 = np.linspace(xs + Lg_x/2, xs + Lg_x/2 + Lt + Ls/2, int(n_points_space/2))
# # x = np.concatenate((x, part6[:-1], part7))

# # #Define the y coordinates for the meshgrid
# # part1 = np.linspace(0, Lg_y/2, int(n_points_grain/2))
# # part2 = np.linspace(Lg_y/2, Lg_y/2 + 2*Lt +Ls, n_points_space)
# # part3 = np.linspace(Lg_y/2 + 2*Lt +Ls, Ly, int(n_points_grain/2))
# # y = np.concatenate((part1[:-1], part2[:-1], part3))

# #Define meshgrid
# x_grid, y_grid = np.meshgrid(x, y)

# # min_steps = math.ceil(x1 / Lt)
# # dx, dy = (x1-x0)/(2*min_steps), (y1-y0)/(2*min_steps) # make these smaller to increase the resolution
# # x, y = np.mgrid[slice(x0, x1 + 2*dx, dx), slice(y0, y1 + 2*dy, dy)]

# # Get z values
# z = sim.tilde_rho(t, x_grid, y_grid)

# #If needed, add periods in y for ease of visualizing the plot to scale
# n_y_periods = math.floor(x1 / Ly)
# if n_y_periods > 1:
#     y_grid_new = np.copy(y_grid)
#     x_grid_new = np.copy(x_grid)
#     z_new = np.copy(z)
#     for i in range(1, n_y_periods):
#         y_grid_new = np.concatenate((y_grid_new[:-1, :], y_grid + i*Ly), axis=0)
#         x_grid_new = np.concatenate((x_grid_new[:-1, :], x_grid), axis=0)
#         z_new = np.concatenate((z_new[:-1, :], z), axis=0)
#     y_grid = y_grid_new
#     x_grid = x_grid_new
#     z = z_new

# # Plot the perturbation contour plot
# fig, ax = plt.subplots(figsize=(8, 8*n_y_periods*Ly/x1))

# # pick sensible levels, and define a normalization
# # instance which takes data values and translates those into levels.
# levels = mpl.ticker.MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())
# norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# # Plot contour plot
# cf = ax.contourf(x_grid, y_grid, z, levels = levels, cmap=cmap)
# # cf = ax.contourf(x_grid[:-1, :-1] + dx/2., 
# #                   y_grid[:-1, :-1] + dy/2., 
# #                   z, levels=levels, cmap=cmap)

# #Add a line for the middle of the shock front
# ax.plot([xs, xs], [y0, n_y_periods*Ly], color='k', linestyle='-', linewidth=0.5)

# # #Test location of the propagating sonic wave
# # ax.plot([sonic_edge[1], sonic_edge[1]], [y0, n_y_periods*Ly], color='m', linestyle='-', linewidth=1)

# #Formatting - colorbar
# fmt = mpl.ticker.ScalarFormatter(useMathText=True)
# fmt.set_powerlimits((0, 0))
# cbar = fig.colorbar(cf, ax=ax, format=fmt)
# cbar.ax.yaxis.set_offset_position('left') 
# cbar.ax.yaxis.get_offset_text().set_fontsize(fontsizes['S'])
# cbar.ax.tick_params(labelsize=fontsizes['M'])
# cbar.update_ticks()

# #Formatting - axis/title/labels
# ax.set_title(r"Density perturbation $\delta\rho / \rho_2$", fontsize=fontsizes['L'])
# ax.tick_params(axis='x', labelsize=fontsizes['M'])
# ax.tick_params(axis='y', labelsize=fontsizes['M'])

# ax.set_xlim(x0, x1)
# ax.set_ylim(y0, n_y_periods*Ly)

# fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# # plt.savefig('test_scale.png', facecolor='white', bbox_inches='tight')

# print('Lg_x = ', Lg_x, ', Lg_y = ', Lg_y, ', Ls = ', Ls, ', Lt = ', Lt)
# print('Number x modes = ', sim.n_x_modes, ' and Number of y modes = ', sim.n_y_modes)
# print('Runtime in seconds: ', time.time() - start)

# %% Visualize pressure, density and velocity perturbations in x-y space
# for fixed t to give the shock front xs in the middle of the plot

#The ratio kx/ky = Lg_y / Lg_x
ratios = [1.0, 0.5, 0.7, 0.3, 0.9, 0.1, 0.8, 0.2]
Lg_ys = np.array(ratios) * Lg_x

#Number of points for plot resolution
n_points_grain = 100
n_points_space = 30

fontsizes = {'XL': 20, 'L': 15, 'M': 12, 'S':10}

#Colormap
cmap = plt.get_cmap('Spectral')
nbins = 50 #number of levels to use in plot

print('Visualize 4 Perturbations')

for Lg_y in Lg_ys:        
    start = time.time()
    
    #Calculate the period in x and y
    Lx = Lg_x + Ls + 2*Lt
    Ly = Lg_y + Ls + 2*Lt
    
    ## Initialize GrainsSolution class
    sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1)
    
    #Calculate the compression ratio, which we need to determine postshock compression
    R = sim.rho2 / sim.rho1

    # Calculuate time value to give desired middle of the shock front location xs
    xs = Lx * (2/R) #desired middle of shock front x-coordinate
    # xs = Lx * (2/R) + 0.06*Lx #to get sonic wave to intersect leftmost vertical interstitial
    t = xs / (sim.M2 * sim.a2)
    
    # Generate 2 2D grids for the x & y bounds
    x0, x1 = 0, Lx + Lx/2 #initial and final x values
    y0, y1 = 0, Ly #initial and final y values
    
    #Define the x coordinates for the meshgrid
    #post shock parts
    sonic_edge = [xs - Lg_x/(2 * sim.M2), xs - Lg_x/(2 * sim.M2) + 3*Ls]
    part1 = np.linspace(0, (Lg_x/2)/R, int(n_points_grain/2))
    part2 = np.linspace(Lg_x/2 * 1/R, sonic_edge[1], 2*n_points_space)
    part3 = np.linspace(sonic_edge[1], (Lx + Lg_x/2)/R, n_points_grain)
    part4 = np.linspace((Lx + Lg_x/2)/R, (Lx + Lg_x/2 + Lt + Ls/2)/R, n_points_space)
    part5 = np.linspace((Lx + Lg_x/2 + Lt + Ls/2)/R, xs, int(n_points_grain/2))
    x = np.concatenate((part1[:-1], part2[:-1], part3[:-1], part4[:-1], part5[:-1]))
    #pre shock parts
    part6 = np.linspace(xs, xs + Lg_x/2, int(n_points_grain/2))
    part7 = np.linspace(xs + Lg_x/2, xs + Lg_x/2 + Lt + Ls/2, int(n_points_space/2))
    x = np.concatenate((x, part6[:-1], part7))
    
    #Define the y coordinates for the meshgrid
    part1 = np.linspace(0, Lg_y/2, int(n_points_grain/2))
    part2 = np.linspace(Lg_y/2, Lg_y/2 + 2*Lt +Ls, n_points_space)
    part3 = np.linspace(Lg_y/2 + 2*Lt +Ls, Ly, int(n_points_grain/2))
    y = np.concatenate((part1[:-1], part2[:-1], part3))
    
    #Define meshgrid
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Get z values: pressure, density, x and y velocity perturbations
    # Normalize following equation 5 in the paper
    # x and t are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    titles = [r"Pressure Perturbation $\delta p / (\gamma p_2)$", 
              r"Density Perturbation $\delta \rho / \rho_2$", 
              r"x-velocity Perturbation $\delta v_x / a_2$", 
              r"y-velocity Perturbation $\delta v_y / a_2$"]
    zs = [sim.tilde_p(t, x_grid, y_grid), 
          sim.tilde_rho(t, x_grid, y_grid), 
          sim.tilde_vx(t, x_grid, y_grid), 
          sim.tilde_vy(t, x_grid, y_grid)]   
    
    #If needed, add periods in y for ease of visualizing the plot to scale
    n_y_periods = math.floor(x1 / Ly)
    if n_y_periods > 1:
        y_grid_new = np.copy(y_grid)
        x_grid_new = np.copy(x_grid)
        zs_new = []
        for item in range(len(zs)):
            zs_new.append(np.copy(zs[item]))
        for i in range(1, n_y_periods):
            y_grid_new = np.concatenate((y_grid_new[:-1, :], y_grid + i*Ly), axis=0)
            x_grid_new = np.concatenate((x_grid_new[:-1, :], x_grid), axis=0)
            for item in range(len(zs)):
                zs_new[item] = np.concatenate((zs_new[item][:-1, :], zs[item]), axis=0)
        y_grid = y_grid_new
        x_grid = x_grid_new
        zs = zs_new
    elif n_y_periods == 0:
        n_y_periods = 1
    
    #Plot the perturbation contour plots
    n_rows, n_cols = 2,2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10)) #*Lg_y/Lg_x))
    
    for row in range(n_rows):
        for col in range(n_cols):
            
            ax = axs[row,col] #specify axis
            index = row*n_rows + col
            z = zs[index] #specify z value
    
            # pick sensible levels, and define a normalization
            # instance which takes data values and translates those into levels.
            levels = mpl.ticker.MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())
            norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
            # Plot contour plot
            cf = ax.contourf(x_grid, y_grid, z, levels = levels, cmap=cmap)
           
            #Add a line for the middle of the shock front
            ax.plot([xs, xs], [y0, n_y_periods*Ly], color='k', linestyle='-', linewidth=0.5)
            
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
            ax.set_ylim(y0, n_y_periods*Ly)
                
    #Formatting master title and naming savefile
    
    suptitle = r"$Lg_x=$" + str(Lg_x) + r", $Lg_y=$" + str(Lg_y) 
    suptitle = suptitle + ", Ls=" + str(Ls) + ", Lt=" + str(Lt)
    suptitle = suptitle + "\n" + r"$\rho_g=$" + "{0:.2g}".format(rho_g) 
    suptitle = suptitle + r", $\rho_s=$" + "{0:.2g}".format(rho_s) + r", $M_1=$" + str(M1)
    fig.suptitle(suptitle, fontsize = fontsizes['XL'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    savename = "plots/perturbations/"
    savename = savename + "Ls-" + str(Ls) + "--Lt_ratio-" + "{0:.2g}".format(Lt/Ls) + "/" 
    savename = savename + "Lg_x" + str(Lg_x) + "--Lg_y" + str(Lg_y) + '--sonic-space-overlap.png' 
    plt.savefig(savename, facecolor='white', bbox_inches='tight')
    
    plt.close()
    
    print('Lg_x = ', Lg_x, ', Lg_y = ', Lg_y, ', Ls = ', Ls, ', Lt = ', Lt)
    print('Number x modes = ', sim.n_x_modes, ' and Number of y modes = ', sim.n_y_modes)
    print('Runtime in seconds: ', time.time() - start)
