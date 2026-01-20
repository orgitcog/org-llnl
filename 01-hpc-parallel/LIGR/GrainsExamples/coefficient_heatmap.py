#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 16:58:52 2021

@author: li85
"""

import numpy as np
from numpy import pi as pi
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

from LinearAnalysis.Grains import GrainsSolution
# from LinearAnalysis.SingleMode import SingleModeSolution

fontsizes = {'XL': 20, 'L': 18, 'M': 15, 'S':10}

## Define parameters
Lg_x = 50 #Grain length scale in x-direction
Lg_y = 50 #Grain length scale in y-direction
Ls = 4 #Interstitial space length scale
Lt = 0.5*Ls #Transition length scale
rho_g = 3.5 #Grain density
rho_s = 2.2 #Interstitial space density
M1 = 1.75 #Mach number


# Initialize GrainsSolution class
sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1)


# %%Get Fourier coefficients and calculate density perturbation magnitude epsilon_k
epsilon_matrix = np.outer(sim.y_coeff, sim.x_coeff)
epsilon_matrix = abs(epsilon_matrix) / sim.rho1
epsilon_matrix[0,0] = 0

epsilon_df = pd.DataFrame(epsilon_matrix, 
                          index = list(range(sim.n_y_modes)),
                          columns = list(range(sim.n_x_modes)))
# epsilon_df = epsilon_df.drop([0]) #drop first row
# epsilon_df = epsilon_df.drop(columns=[0]) #drop first column

# Plot heatmap
if Lg_y/Lg_x > 0.6:
    fig, ax = plt.subplots(figsize = (10, 10*Lg_y/Lg_x))
else:
    fig, ax = plt.subplots(figsize = (10, 6))

#Set tick spacing to show a reasonable number of ticks for readability
n_ticks = min(12, sim.n_x_modes, sim.n_y_modes) #Number of ticks to show
#Integer spacing of x and y ticsk to show approximately n_ticks
xtick_space = math.floor(sim.n_x_modes / n_ticks)
ytick_space = math.floor(sim.n_y_modes / n_ticks)

#Exponential format for colorbar
fmt = mpl.ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))

heatmap = sns.heatmap(epsilon_df, ax = ax,
                  xticklabels = xtick_space, yticklabels = ytick_space,
                  cbar_kws = {"format": fmt})

#Formatting colorbar
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_offset_position('left') 
cbar.ax.yaxis.get_offset_text().set_fontsize(fontsizes['M'])
cbar.ax.tick_params(labelsize=fontsizes['M'])
cbar.update_ticks()

#Formating axis
plt.yticks(rotation=0) 
ax.set_xlabel('Nx', fontsize=fontsizes['M'])
ax.set_ylabel('Ny', fontsize=fontsizes['M'])
ax.tick_params(axis='x', labelsize=fontsizes['L'])
ax.tick_params(axis='y', labelsize=fontsizes['L'])

#Formatting Title
title = r"$\epsilon_k$ Density Perturbation per Fourier Mode" + "\n"
title = title + r"$Lg_x=$" + str(Lg_x) + r", $Lg_y=$" + str(Lg_y) 
title = title + ", Ls=" + str(Ls) + ", Lt=" + str(Lt)
title = title + "\n" + r"$\rho_g=$" + "{0:.2g}".format(rho_g) 
title = title + r", $\rho_s=$" + "{0:.2g}".format(rho_s)
title = title +", x-modes = " + str(sim.n_x_modes) +", y-modes = " + str(sim.n_y_modes)
plt.title(title, fontsize = fontsizes['XL'])

fig.tight_layout(rect=[0, 0.03, 1, 0.88])

savename = "plots/coefficient_heatmaps/"
savename = savename + "Lg_x-" + str(Lg_x) + "--Lg_y-" + str(Lg_y) 
savename = savename + "--Ls-" + str(Ls) + "--Lt_ratio-" + "{0:.2g}".format(Lt/Ls) +'.png' 
# plt.savefig(savename, facecolor='white', bbox_inches='tight')