#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 21:11:11 2021

@author: li85
"""

import numpy as np
from numpy import pi as pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy import io

from LinearAnalysis.Grains import GrainsSolution

## Define parameters
Lg_x = 200 #Grain length scale in x-direction
Lg_y = 200 #Grain length scale in y-direction
Ls = 10 #Interstitial space length scale
Lt = 0.5*Ls #Transition length scale
rho_g = 3.5 #Grain density
rho_s = 2.2 #Interstitial space density
M1 = 1.75 #Mach number

# Specify which shock-front related plot to generate
# Options are: "delta_xs"
plot_type = "delta_xs"

# %% Change plot parameters here
fontsizes = {'XL': 23, 'L': 20, 'M': 19, 'S':17}

## --------------------------------------------------------------------------
## Plot the shock perturbation delta_xs vs the y-coordinate over two periods
if plot_type == "delta_xs":
    
    ## Define parameters
    Lg_x = 200 #Grain length scale in x-direction
    Lg_y = 200 #Grain length scale in y-direction
    Ls = 4 #Interstitial space length scale
    Lt = 0.5*Ls #Transition length scale
    rho_g = 3.5 #Grain density
    rho_s = 2.2 #Interstitial space density
    M1 = 1.75 #Mach number
    
    #Specify the x-coordinates for the shock front to be plotted
    #If multiple x-coordinates are to be plotted labels for the legend must be defined
    xs = [Lg_x/4]
    
    # xs = [Lg_x/4, 
    #       Lg_x * (3/8), 
    #       Lg_x/2, 
    #       Lg_x/2 + Lt, 
    #       sim.Lx/2,
    #       sim.Lx + Ls/2,
    #       sim.Lx + Ls/2 + Lt] 
    
    # labels = [r"$Lg_x$/4", 
    #           r"$3 Lg_x$/8", 
    #           r"$Lg_x$/2",
    #           r"$Lg_x$/2 + Lt", 
    #           r"Lx/2",
    #           r"Lx/2 + Ls/2", 
    #           r"Lx/2 + Ls/2 + Lt"]

## --------------------------------------------------------------------------
# Plot the perturbation range against the shock x-position
if plot_type == "aspect_ratio":
    # In this case, the grain area, interstitial and transition lengths are fixed
    sqrt_area = 200
    area = sqrt_area**2
    Ls = 10
    Lt = 0.5*Ls
    
    #Specify aspect ratio parameters
    ratios = np.linspace(0.05, 1, 50)
    
    #Calculate the corresponding x and y direction grain widths
    Lg_xs = np.sqrt(area / np.array(ratios))
    
## --------------------------------------------------------------------------
if plot_type == "Ls-Lt_ratio":

    # In this case, the x-grain and y-grain lengths are fixed
    Lg_x = 200
    Lg_y = 200
    
    Lt_ratio = 0.5
    
    #Specify interstitial space width parameters
    Lss = np.linspace(4, 100, 49)
    # Lss = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
## --------------------------------------------------------------------------
if plot_type == "Lg_x":
    # In this case, the aspect ratio, interstitial and transition lengths are fixed
    aspect_ratio = 1 #aspect_ratio is defined as Lg_y / Lg_x
    Ls = 10
    Lt = 5
    
    #Specify x-grain width parameters
    Lg_xs = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    Lg_xs = np.linspace(100, 1000, 19)
    
# %% Plot the shock front over time over two periods in y
if plot_type == "delta_xs":

    ## Initialize GrainsSolution class
    sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1, max_modes = 1000000)
    
    ## Define y range to plot
    y = np.linspace(0, 2*sim.Ly, 500)
    
    #Calculate the corresponding t values for the shock
    ts = np.array(xs) / (sim.M2 * sim.a2)
    
    #Plot the shock front for each time
    fig, ax = plt.subplots(figsize=(10,6))
    
    for i in range(len(xs)):
        t = ts[i]
        shock_perturbation = sim.delta_xs(t,y)
        
        #Plot the long-time-average integral value vs the aspect ratio
        try:
            ax.plot(y, shock_perturbation, label=labels[i])
        except:
            ax.plot(y, shock_perturbation)
            
    title = "Shock front perturbations over time\n"
    if len(xs) == 1:
        title = title + "$x_s$=" + str(xs[0]) + ", "
    title = title + r"$Lg_x=$" + str(Lg_x) + r", $Lg_y=$" + str(Lg_y) 
    title = title + ", Ls=" + str(Ls) + ", Lt=" + str(Lt)
    plt.title(title, fontsize = fontsizes['XL'], y=1.01)
    
    ax.set_xlabel("y", fontsize=fontsizes['M'])
    ax.set_ylabel("Shock front perturbation $\delta x_s$", fontsize=fontsizes['M'])
    ax.tick_params(axis='x', labelsize=fontsizes['S'])
    ax.tick_params(axis='y', labelsize=fontsizes['S'])
    
    if len(xs) > 1:
        legend = plt.legend(title = "x-coordinate of shock", bbox_to_anchor=(1.02, 1), 
                    loc='upper left', fontsize = fontsizes['M'])
        plt.setp(legend.get_title(), fontsize = fontsizes['L'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    figname = "plots/shock_front/delta_xs/"
    if len(xs) == 1:
        figname = figname + "single_front--"
    figname = figname + "Ls" + str(Ls) + "--Lt" + str(Lt)
    figname = figname + "--Lg_x" + str(Lg_x) + "--Lg_y" + str(Lg_y) + ".png"
    plt.savefig(figname, facecolor='white', bbox_inches='tight')
    
    print("Lg_x=", Lg_x, ", Lg_y=", Lg_y)
    print("Ls=", Ls, ", Lt=", Lt)
    print("rho_g=", rho_g, ", rho_s=", rho_s)
    print("shock front perturbation:", max(shock_perturbation) - min(shock_perturbation))
    
    # plt.close()

# %% Plot the perturbation range against the shock x-position
# (Partly to find what shock front location gives maximum perturbation.)
if plot_type == "range_vs_xs":

    ## Initialize GrainsSolution class
    sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1)
    
    #Define x-coordinates of interest for the shock front
    xs = np.linspace(0,1,100) * sim.Lx
    
    #Calculate the corresponding t values for the shock
    ts = np.array(xs) / (sim.M2 * sim.a2)
    
    #y-coordinate in the middle of interstitial space gives max shock perturbation
    ymax = sim.Ly / 2
    #y-coordinate in the middle of grain gives min shock perturbation
    ymin = sim.Ly
    
    #Store the perturbation range for the shock front
    perturbation_range = []
    for t in ts:
        values = sim.delta_xs(t, np.array([ymin, ymax]))
        perturbation_range.append(values[1] - values[0])
        
    # Plot the shock-front perturbation range vs the shock location in x
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.plot(xs, perturbation_range)
    
    #Get y-limits of plot
    y1, y2 = ax.get_ylim()
    
    #Plot the locations of the transition region and interstitial space boundaries
    ax.plot([Lg_x/2, Lg_x/2], [y1, y2], 'k')
    ax.plot([Lg_x/2 + Lt, Lg_x/2 + Lt], [y1, y2], 'r')
    ax.plot([Lg_x/2 + Lt + Ls, Lg_x/2 + Lt + Ls], [y1, y2], 'r')
    ax.plot([Lg_x/2 + 2*Lt + Ls, Lg_x/2 + 2*Lt + Ls], [y1, y2], 'k')
        
    title = "Range of Shock Perturbation vs x-Coordinate\n"
    title = title + r"$Lg_x=$" + str(Lg_x)  + r", $Lg_y=$" + str(Lg_y)
    title = title + ", Ls=" + str(Ls) + ", Lt=" + str(Lt)
    plt.title(title, fontsize = fontsizes['XL'], y=1.01)
    
    ax.set_xlabel("x-Coordinate of Shock Front", fontsize=fontsizes['M'])
    ax.set_ylabel("Range of Shock-Front Perturbation", fontsize=fontsizes['M'])
    ax.tick_params(axis='x', labelsize=fontsizes['S'])
    ax.tick_params(axis='y', labelsize=fontsizes['S'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    figname = "plots/shock_front/range_vs_xs/"
    figname = figname + "Lg_x" + str(Lg_x) + "--Lg_y" + str(Lg_x)
    figname = figname + "--Ls" + str(Ls) + "--Lt" + str(Lt) + ".png"
    plt.savefig(figname, facecolor='white', bbox_inches='tight')

    
# %% Plot maximum perturbation range against aspect ratio
if plot_type == "aspect_ratio":
    
    #Store the perturbation range for the shock front
    perturbation_range = []
    
    for Lg_x in Lg_xs:
    
        Lg_y = area / Lg_x    
    
        ## Initialize GrainsSolution class
        sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1)
        
        #Calculate the perturbation range as the shock leaves the grain
        x = sim.Lg_x / 2
        
        #Calculate the corresponding t value for the shock
        t = np.array(x) / (sim.M2 * sim.a2)
        
        #y-coordinate in the middle of interstitial space gives max shock perturbation
        ymax = sim.Ly / 2
        #y-coordinate in the middle of grain gives min shock perturbation
        ymin = sim.Ly
        
        values = sim.delta_xs(t, np.array([ymin, ymax]))
        perturbation_range.append(values[1] - values[0])
        
    # Plot the shock-front perturbation range vs the aspect ratio
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.plot(ratios, perturbation_range)
    
    title = "Shock Perturbation vs Aspect Ratio\n"
    title = title + r"$\sqrt{Area}=$" + str(sqrt_area)
    title = title + r", $Lg_y/Lg_x=$" + "{0:.2f}".format(Lg_y/Lg_x) 
    title = title + ", Ls=" + str(Ls) + ", Lt=" + str(Lt)
    plt.title(title, fontsize = fontsizes['XL'], y=1.01)
    
    ax.set_xlabel(r"Aspect Ratio $Lg_y/Lg_x$", fontsize=fontsizes['M'])
    ax.set_ylabel("Shock Front Perturbation", fontsize=fontsizes['M'])
    ax.tick_params(axis='x', labelsize=fontsizes['S'])
    ax.tick_params(axis='y', labelsize=fontsizes['S'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    figname = "plots/shock_front/range/aspect_ratio---"
    figname = figname + "Lg_x" + str(Lg_x) + "--Lg_y" + str(Lg_x)
    figname = figname + "--Ls" + str(Ls) + "--Lt" + str(Lt) + ".png"
    plt.savefig(figname, facecolor='white', bbox_inches='tight')

# %% Plot maximum perturbation range against interstitial space width
# with fixed Lt/Ls ratio (Ls-Lt_ratio)

if plot_type == "Ls-Lt_ratio":
    
    #Store the perturbation range for the shock front
    perturbation_range = []
    
    for Ls in Lss:
    
        #Calculate transition width
        Lt = Lt_ratio * Ls   
    
        ## Initialize GrainsSolution class
        sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1)
        
        #Calculate the perturbation range as the shock leaves the grain
        x = sim.Lg_x / 2
        
        #Calculate the corresponding t value for the shock
        t = np.array(x) / (sim.M2 * sim.a2)
        
        #y-coordinate in the middle of interstitial space gives max shock perturbation
        ymax = sim.Ly / 2
        #y-coordinate in the middle of grain gives min shock perturbation
        ymin = sim.Ly
        
        values = sim.delta_xs(t, np.array([ymin, ymax]))
        perturbation_range.append(values[1] - values[0])
        
    # Plot the shock-front perturbation range vs the aspect ratio
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.plot(Lss, perturbation_range)
    
    title = "Shock Perturbation vs Intersitial Width\n"
    title = title + r"$Lg_x=$" + str(Lg_x) + r", $Lg_y=$" + "{0:.1f}".format(Lg_y)
    title = title + ", Lt/Ls =" + str(Lt_ratio)
    plt.title(title, fontsize = fontsizes['XL'], y=1.01)
    
    ax.set_xlabel(r"Interstitial Width Ls", fontsize=fontsizes['M'])
    ax.set_ylabel("Shock Front Perturbation", fontsize=fontsizes['M'])
    ax.tick_params(axis='x', labelsize=fontsizes['S'])
    ax.tick_params(axis='y', labelsize=fontsizes['S'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    figname = "plots/shock_front/range/Ls-Lt_ratio"
    figname = figname + "--Lg_x" + str(Lg_x) + "--Lg_y" + "{0:.1f}".format(Lg_y)
    figname = figname + "--Lt_ratio" + str(Lt_ratio) + ".png"
    plt.savefig(figname, facecolor='white', bbox_inches='tight')

# %% Plot maximum perturbation range against grain width (Lg_x)
# Plot the perturbation range against the shock x-position

if plot_type == "Lg_x":
    
    #Store the perturbation range for the shock front
    perturbation_range = []
    
    for Lg_x in Lg_xs:
    
        #Calculate the y grain width
        Lg_y = Lg_x * aspect_ratio
    
        ## Initialize GrainsSolution class
        sim = GrainsSolution(Lg_x, Lg_y, Ls, Lt, rho_g, rho_s, M1)
        
        #Calculate the perturbation range as the shock leaves the grain
        x = sim.Lg_x / 2
        
        #Calculate the corresponding t value for the shock
        t = np.array(x) / (sim.M2 * sim.a2)
        
        #y-coordinate in the middle of interstitial space gives max shock perturbation
        ymax = sim.Ly / 2
        #y-coordinate in the middle of grain gives min shock perturbation
        ymin = sim.Ly
        
        values = sim.delta_xs(t, np.array([ymin, ymax]))
        perturbation_range.append(values[1] - values[0])
        
    # Plot the shock-front perturbation range vs the aspect ratio
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.plot(Lg_xs, perturbation_range)
    
    title = "Shock Perturbation vs Grain Width\n"
    title = title + r"$Lg_y/Lg_x=$" + str(aspect_ratio)
    title = title + ", Ls =" + str(Ls) + ", Lt =" + str(Lt)
    plt.title(title, fontsize = fontsizes['XL'], y=1.01)
    
    ax.set_xlabel(r"x-Direction Grain Width $Lg_x$", fontsize=fontsizes['M'])
    ax.set_ylabel("Shock Front Perturbation", fontsize=fontsizes['M'])
    ax.tick_params(axis='x', labelsize=fontsizes['S'])
    ax.tick_params(axis='y', labelsize=fontsizes['S'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    figname = "plots/shock_front/range/Lg_x"
    figname = figname + "--aspect_ratio" + str(aspect_ratio)
    figname = figname + "--Ls" + str(Ls) + "--Lt" + str(Lt) + ".png"
    plt.savefig(figname, facecolor='white', bbox_inches='tight')