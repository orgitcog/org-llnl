#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:50:38 2021

@author: li85

Verify that the fuction to represent the actual density perturbation looks correct,
and that the Fourier 
"""

# Required Imports
import warnings

import math
import numpy as np
from numpy import pi as pi

import matplotlib.pyplot as plt
import matplotlib as mpl

from LinearAnalysis.SingleMode import SingleModeSolution
from LinearAnalysis.Grains import GrainsSolution

# ## Define parameters
# Lg_x = 1e-4 #Grain length scale in x-direction in cm
# Lg_y = 1e-4 #Grain length scale in y-direction in cm
# Ls = 2e-5 #Interstitial space length scale in cm
# Lt = 0.5*Ls #Transition length scale in cm
# rho_g = 3.5 #Grain density in g/cm3
# rho_s = 2.2 #Interstitial space density in g/cm3
# M1 = 1.75 #Mach number (dimensionless)

## Define parameters
Lg_x = 50 #Grain length scale in x-direction
Lg_y = 50 #Grain length scale in y-direction
Ls = 10 #Interstitial space length scale
Lt = 0.5*Ls #Transition length scale
rho_g = 3.5 #Grain density in g/cm3
rho_s = 2.2 #Interstitial space density in g/cm3
M1 = 1.75 #Mach number

# Define the periods in x and y direction
Lx = Lg_x + Ls + 2*Lt
Ly = Lg_y + Ls + 2*Lt

## Initialize GrainsSolution class
sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1)


fontsizes = {'XL': 25, 'L': 22, 'M': 20, 'S':15}

# %% Test plot the density perturbation in x only

x = np.linspace(0, Lx, 10000) #x-coordinate

fig, ax = plt.subplots(figsize=(12,8))

#Plot some truncated Fourier series to test
#THIS MUST BE SORTED SMALLEST TO LARGEST
n_modes = [0,4,10,22,50,100]

#Calculate and store the truncated Fourier series
series_number = 0
Fxs = []
Fx = sim.cx0 * np.ones(x.shape)
if n_modes[0] == 0:
    Fxs.append(Fx)
    series_number = 1
for i in range(1, max(n_modes)+1):
    Fx = Fx + sim.x_mode(x, i)
    if i == n_modes[series_number]:
        Fxs.append(Fx)
        series_number += 1
        
for i in range(len(n_modes)):
    if i == len(n_modes) - 1:
        ax.plot(x, Fxs[i], 'k', linewidth = 2.5, label = str(n_modes[i])+' modes')
    else:
        ax.plot(x, Fxs[i], label = str(n_modes[i]) + ' modes')

#Plot grain density in x-direction
ax.plot(x, sim.rho_x(x), 'm', linewidth=2)

legend = plt.legend(title = 'Number of Modes', bbox_to_anchor=(1.02, 1), 
            loc='upper left', fontsize = fontsizes['M'])
plt.setp(legend.get_title(), fontsize = fontsizes['M'])

plt.xlabel("x", fontsize=fontsizes['M'])
plt.ylabel("Density Perturbation", fontsize=fontsizes['M'])
ax.tick_params(axis='x', labelsize=fontsizes['S'])
ax.tick_params(axis='y', labelsize=fontsizes['S'])

title = 'Lg_x = ' + str(Lg_x) + ', Ls = ' + str(Ls) + ', Lt = ' + str(Lt)
plt.title(title, fontsize = fontsizes['XL'])

# %% 2-D density perturbation plot 

# Generate 2 2D grids for the x & y bounds
x0, x1 = 0, 2*Lx #initial and final x values
y0, y1 = 0, 2*Ly #initial and final y values
dx, dy = (x1-x0)/1000, (y1-y0)/1000 # make these smaller to increase the resolution
x, y = np.mgrid[slice(x0, x1 + dx, dx), slice(y0, y1 + dy, dy)]

#Get normalized density perturbation
z = sim.preshock_delta_rho(x,y) / sim.rho1
z = z[:-1, :-1] 

#Plot
if Ly / Lx > 0.5:
    fig, ax = plt.subplots(figsize=(10,9*Ly/Lx))
else:
    fig, ax = plt.subplots(figsize=(10,5))

#Colormap
cmap = plt.get_cmap('Spectral')
nbins = 50 #number of levels to use in plot

# pick sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
levels = mpl.ticker.MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# Plot contour plot
cf = ax.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1] + dy/2., 
                  z, levels=levels, cmap=cmap)

fmt = mpl.ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cbar = fig.colorbar(cf, ax=ax, format=fmt)
cbar.ax.yaxis.get_offset_text().set_fontsize(fontsizes['S'])
cbar.ax.tick_params(labelsize=fontsizes['S'])
#cbar.formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_offset_position('left')              
# cbar.ax.set_title(r"$\epsilon_k$", fontsize=fontsizes['M'])           
cbar.update_ticks()
ax.set_title('Normalized Density Perturbation', fontsize=fontsizes['L'])
ax.set_xlim(x0,x1)
ax.set_ylim(y0,y1)
ax.set_xlabel('x', fontsize=fontsizes['M'])
ax.set_ylabel('y', fontsize=fontsizes['M'])
ax.tick_params(axis='x', labelsize=fontsizes['S'])
ax.tick_params(axis='y', labelsize=fontsizes['S'])


# %% Plot the fourier decomposition of density perturbation

Nx = sim.n_x_modes #number of x-direction modes
Ny = sim.n_y_modes #number of y-direction modes


# Generate 2 2D grids for the x & y bounds
x0, x1 = 0, 2*sim.Lx #initial and final x values
y0, y1 = 0, 2*sim.Ly #initial and final y values
dx, dy = (x1-x0)/2000, (y1-y0)/2000 # make these smaller to increase the resolution
x = np.arange(x0, x1 + dx, dx)
y = np.arange(y0, y1 + dy, dy)

#Get the Fourier representation of normalized density perturbations and normalize
z = sim.fourier_delta_rho(x,y)
z = z[:-1, :-1] 
z = z / sim.rho1

#Set up array for contour plot
x, y = np.mgrid[slice(x0, x1 + dx, dx), slice(y0, y1 + dy, dy)]

#Plot
if Ly / Lx > 0.5:
    fig, ax = plt.subplots(figsize=(10,9*Ly/Lx))
else:
    fig, ax = plt.subplots(figsize=(10,5))

#Colormap
cmap = plt.get_cmap('Spectral')
nbins = 50 #number of levels to use in plot

# pick sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
levels = mpl.ticker.MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# Plot contour plot
cf = ax.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1] + dy/2., 
                  z, levels=levels, cmap=cmap)

fmt = mpl.ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cbar = fig.colorbar(cf, ax=ax, format=fmt)
cbar.ax.yaxis.get_offset_text().set_fontsize(fontsizes['S'])
cbar.ax.tick_params(labelsize=fontsizes['S'])
#cbar.formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_offset_position('left')              
# cbar.ax.set_title(r"$\epsilon_k$", fontsize=fontsizes['M'])           
cbar.update_ticks()
ax.set_title('Fourier Representation of\n Normalized Density Perturbation', fontsize=fontsizes['L'])
ax.set_xlim(x0,x1)
ax.set_ylim(y0,y1)
ax.set_xlabel('x', fontsize=fontsizes['M'])
ax.set_ylabel('y', fontsize=fontsizes['M'])
ax.tick_params(axis='x', labelsize=fontsizes['S'])
ax.tick_params(axis='y', labelsize=fontsizes['S'])

# %% Plot the error of the truncated fourier decomposition of density perturbation

Nx = sim.n_x_modes #number of x-direction modes
Ny = sim.n_y_modes #number of y-direction modes

# Generate 2 2D grids for the x & y bounds
x0, x1 = 0, 2*Lx #initial and final x values
y0, y1 = 0, 2*Ly #initial and final y values
dx, dy = (x1-x0)/1000, (y1-y0)/1000 # make these smaller to increase the resolution

x = np.arange(x0, x1 + dx, dx)
y = np.arange(y0, y1 + dy, dy)

#Get the Fourier representation of normalized density perturbations,
#and subtract the actual density perturbation. Then, normalize.
z = sim.fourier_delta_rho(x,y)
#Make x-y grid to calculate the preshock density
x, y = np.mgrid[slice(x0, x1 + dx, dx), slice(y0, y1 + dy, dy)]
z = z - sim.preshock_delta_rho(x,y)
z = z[:-1, :-1] 
z = z / sim.rho1

print(np.max(z))

#Set up array for contour plot
x, y = np.mgrid[slice(x0, x1 + dx, dx), slice(y0, y1 + dy, dy)]

#Plot
if Ly / Lx > 0.5:
    fig, ax = plt.subplots(figsize=(10,9*Ly/Lx))
else:
    fig, ax = plt.subplots(figsize=(10,5))

#Colormap
cmap = plt.get_cmap('Spectral')
nbins = 50 #number of levels to use in plot

# pick sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
levels = mpl.ticker.MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# Plot contour plot
cf = ax.contourf(x[:-1, :-1] + dx/2., y[:-1, :-1] + dy/2., 
                  z, levels=levels, cmap=cmap)

fmt = mpl.ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cbar = fig.colorbar(cf, ax=ax, format=fmt)
cbar.ax.yaxis.get_offset_text().set_fontsize(fontsizes['S'])
cbar.ax.tick_params(labelsize=fontsizes['S'])
#cbar.formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_offset_position('left')              
# cbar.ax.set_title(r"$\epsilon_k$", fontsize=fontsizes['M'])           
cbar.update_ticks()
ax.set_title('Error in Normalized Density Perturbation', fontsize=fontsizes['L'])
ax.set_xlim(x0,x1)
ax.set_ylim(y0,y1)
ax.set_xlabel('x', fontsize=fontsizes['M'])
ax.set_ylabel('y', fontsize=fontsizes['M'])
ax.tick_params(axis='x', labelsize=fontsizes['S'])
ax.tick_params(axis='y', labelsize=fontsizes['S'])

# # %% Test triangle of Fourier modes and 2D density perturbation error

# #Define x and y range near the interstitial space to check error
# n_points = 1000
# x1, x2 = Lg_x/2 * 0.9, Lx - (Lg_x/2 * 0.9)
# y1, y2 = Lg_y/2 * 0.9, Ly - (Lg_y/2 * 0.9)
# x = np.linspace(x1, x2, n_points)
# y = np.linspace(y1, y2, n_points)

# print(x.shape)
# print(y.shape)

# #Define meshgrid version to calculate the actual density
# x_grid, y_grid = np.meshgrid(x,y)

# print(x_grid.shape)
# print(y_grid.shape)

# n_x_modes = sim.n_x_modes
# n_y_modes = sim.n_y_modes
# tol = 0.1 * (rho_g - rho_s)

# #Keep track of if tolerance was reached
# tol_reached  = False
# counter = 0

# while tol_reached == False:
#     tol_reached = True
#     print('in loop')
    
#     #Calculate the Fourier representation of density perturbation by summing up
#     #the contributions from a "triangle" of modes 
#     fourier_delta_rho = np.zeros(x_grid.shape)
#     for i in range(n_x_modes):
#         # for j in range(n_y_modes):
#         for j in range(int(math.ceil((n_x_modes + n_y_modes)/2)) - i):
#             if not (i==0 and j==0):
#                 mode_perturbation = np.outer(sim.y_mode(y,j), sim.x_mode(x,i))
#                 fourier_delta_rho = fourier_delta_rho + mode_perturbation
                
#     # fourier_delta_rho = fourier_delta_rho
#     actual_delta_rho = sim.preshock_delta_rho(x_grid, y_grid)
#     error = abs(fourier_delta_rho - actual_delta_rho)
#     if np.max(error) > tol:
#         print('oh no!')
#         print(np.max(error))
#         n_x_modes = n_x_modes + 1
#         n_y_modes = n_y_modes + 1
#         counter = counter + 1
#         print(counter)
#         tol_reached = False
