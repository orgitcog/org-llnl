#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Grace Li (li85)
Date: 9/1/2021

Python script to generate the plots of percent perturbation of various
energy and momentum quantities while scanning various parameters including
aspect ratio, interstitial width, interstitial density, and grain size.
                             
First run either grains_integrals_over_time.py or grains_IE_over_time
to get the saved matfiles with the saved average in space (approximation for
integral) of the energy or momentum quantity for each time coordinate.

Change the "integration_quantity" (which magnitude of energy/momentum to look at),
whether perturbations are measured from the lab reference frame ("lab_frame" True or False),
which "investigation", and what resolution ("n_x_intervals") we took the integral at
in the first section of this code.

Then, control F the "investigation" type you want and under the "if investigation =="
statement, change the particular parameters of the investigation. 
"""

import numpy as np
from numpy import pi as pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy import io
from scipy.optimize import curve_fit

from LinearAnalysis.Grains import GrainsSolution

rho_diamond = 3.5
rho_graphite = 2.2
M1 = 1.75 #Mach number

# Specify which integral we're looking at. The options are:
# "KE" (kinetic energy), "IE" (internal energy),
# "Mx" (magnitude of x-momentum), "My" (magnitude of y-momentum), 
# and "Mtotal" (magnitude of total momentum)
integration_quantity = "Mtotal"

# Specify if we want the quantity in the lab reference frame 
# or not (calculates in the postshock stationary frame)
# For KE, Mtotal and Mx only - not for My or IE
lab_frame = False

# Specify which parameter we are investigating. The options are:
# "aspect_ratio", "Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain", "Lg_x"
investigation = "conserve_density-Lt_ratio"

# Specify the resolution we ran at
n_x_intervals = 300

# Specify if perturbation should be calculated from mean density
density_perturb_from_mean = False

fontsizes = {'XL': 20, 'L': 17, 'M': 15, 'S':13}

# %% Change plot parameters for each investigation type here

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
    ratios = np.linspace(0.1, 1.0, 19)
    
    #Specify some aspect ratios with whole sides to annotate
    annotated_ratios = [4/25, 1/4, 16/25, 1]
    # annotated_ratios = []
    
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
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ratios = np.linspace(0.05, 1, 20)
    Lg_ys = np.array(ratios) * Lg_x


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
        Lss = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        
        Lx_original = Lg_x_original + Ls_original * (1 + 2*Lt_ratio)
        Ly_original = Lg_y_original + Ls_original * (1 + 2*Lt_ratio)
    
    #If investigation is "Ls-Lt_constant-fixed_grain", then the transition length is fixed
    if investigation == "Ls-Lt_constant":
        Lt = 5
        
        #Specify interstitial space width parameters
        Lss = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        Lx_original = Lg_x_original + Ls_original + 2*Lt
        Ly_original = Lg_y_original + Ls_original + 2*Lt


## ---------------------------------------------------------------------------
## Investigation == "Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain"--------------------------
## Vary the interstitial space width Ls with either scaled or constant transition Lt
## Keep the grain size fixed in this case

if investigation in ["Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain"]:
    # In this case, the x-grain and y-grain lengths are fixed
    Lg_x = 200
    Lg_y = 200
    rho_g = rho_diamond
    rho_s = rho_graphite
    
    #If investigation is "Ls-Lt_ratio-fixed_grain", then Lt scales with Ls
    #Specify the ratio of the transition to interstitial Lt/Ls
    if investigation == "Ls-Lt_ratio-fixed_grain":
        Lt_ratio = 0.5
        
        #Specify interstitial space width parameters
        Lss = [4, 10, 15, 20, 30, 40, 45, 50]
    
    #If investigation is "Ls-Lt_constant-fixed_grain", then the transition length is fixed
    if investigation == "Ls-Lt_constant-fixed_grain":
        Lt = 5
       
        #Specify interstitial space width parameters
        Lss = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
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
    n_x_intervals = 200
    
    #Specify x-grain width parameters
    # Lg_xs = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    Lg_xs = [50, 100, 200, 300, 400, 500, 600]#

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
    
    #Specify the ratio Lt/Ls if we are scaling Lt with Ls
    if investigation == "conserve_density-Lt_ratio":
        Lt_ratio = 0.5
        
        #Specify interstitial space width parameters
        Lss = [4, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        Lss = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        
        Lx_original = Lg_x_original + Ls_original * (1 + 2*Lt_ratio)
        Ly_original = Lg_y_original + Ls_original * (1 + 2*Lt_ratio)
    
    #If investigation is "Ls-Lt_constant-fixed_grain", then the transition length is fixed
    if investigation == "conserve_density-Lt_constant":
        Lt = 5
        
        #Specify interstitial space width parameters
        Lss = [4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        Lx_original = Lg_x_original + Ls_original + 2*Lt
        Ly_original = Lg_y_original + Ls_original + 2*Lt

## --------------------------------------------------------------------------
if investigation == "other":
    ## Define parameters
    Lg_x = 800 #Grain length scale in x-direction
    Lg_y = 100 #Grain length scale in y-direction
    Ls = 4 #Interstitial space length scale
    Lt = 0.5*Ls #Transition length scale
    rho_g = 3.5 #Grain density in g/cm3
    rho_s = 2.2 #Interstitial space density in g/cm3
    M1 = 1.75 #Mach number
    
    # Number of points in x direction in grid to average over
    n_x_intervals = 500
    
    parameters = [1]
    
    #Integration quantity to print out
    integration_quantities = ["KE", "My", "Mx", "Mtotal",
                              "KE_lab", "Mx_lab", "Mtotal_lab"]
    
    for integration_quantity in integration_quantities:
    
        try: 
            #Load the corresponding matfile
            filename = "matfiles/" + integration_quantity + "_over_time/" 
            filename = filename + investigation + ("---density_perturb_from_mean" * density_perturb_from_mean)
            filename = filename + "/Ls-" + str(Ls)
            filename = filename + "--Lt_ratio-" + "{0:.2g}".format(Lt/Ls)
            filename = filename + "--Lg_x" + str(Lg_x) + "--Lg_y" + "{0:.1f}".format(Lg_y)
            if (integration_quantity != 'IE'):
                filename = filename + "--n_x_intervals-" + str(n_x_intervals) + ".mat"
            
            matdict = io.loadmat(filename)
            
            #Load the time and integrals at each time
            t = matdict["t"][0]
            try:
                integrals = matdict["integral"][0]
            except:
                #If the file is older, than the key for the integrals was different
                if integration_quantity in ['Mx', 'My', 'Mtotal']:
                    key = "M_integral"
                else:
                    key = integration_quantity + "_integral"
            
            #Take the average over all the time points and add to long-time average list
            print("Long-Time-Average Integral of", integration_quantity)
            print(np.average(integrals))
        
        except:
           print("No data generated yet for ", integration_quantity) 

## --------------------------------------------------------------------------
# Sort out which values we're investigating -----------------------------------

#Specify if we should plot the power law fit for the below Ls investigations
fit_power_law = True

# List of investigations where the parameter varied is Ls
Ls_investigations = ["Ls-Lt_ratio", "Ls-Lt_constant",
                     "Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain",
                     "conserve_density-Lt_ratio", "conserve_density-Lt_constant"]

if investigation == "aspect_ratio":
    parameters = Lg_xs
if investigation == "aspect_ratio-fixed_Lx":
    parameters = Lg_ys
elif investigation == "Lg_x":
    parameters = Lg_xs
elif investigation == "rho_s":
    parameters = rho_ss
elif investigation in Ls_investigations:
    parameters = Lss
    
    
# %% Generate the desired plots based on the parameters and investigation type
# specified above.

# Create a list to store the average (approximation of the integral) in time
long_time_average = []   

#For each parameter, we open the corresponding file and calculate the long time average
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
            resolutions = int(math.ceil(Lg_x * n_x_intervals_scaling))
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
           
    #Load the corresponding matfile
    filename = "matfiles/" + integration_quantity 
    filename = filename + ("---density_perturb_from_mean" * density_perturb_from_mean)
    filename = filename + ("_lab" * lab_frame) + "_over_time/" + investigation
    filename = filename + "/Ls-" + str(Ls)
    
    if investigation in ["Ls-Lt_constant", "Ls-Lt_constant-fixed_grain", "conserve_density-Lt_constant"]:
        filename = filename + "--Lt-" + str(Lt)
    else:
        filename = filename + "--Lt_ratio-" + "{0:.2g}".format(Lt/Ls)
        
    if investigation == "aspect_ratio":
        filename = filename + "--sqrt_area" + str(sqrt_area)
        filename = filename + "--aspect_ratio" + "{0:.2f}".format(Lg_y/Lg_x)
    else:
        filename = filename + "--Lg_x" + str(Lg_x) + "--Lg_y" + "{0:.1f}".format(Lg_y)
        if investigation == "rho_s":
                filename = filename + "--rho_s" + "{0:.1f}".format(rho_s)
    if (integration_quantity != 'IE'):
        filename = filename + "--n_x_intervals-" + str(n_x_intervals) + ".mat"
    
    matdict = io.loadmat(filename)
    
    #Load the time and integrals at each time
    if integration_quantity in ['Mx', 'My', 'Mtotal']:
        key = "M_integral"
    else:
        key = integration_quantity + "_integral"
    t = matdict["t"][0]
    integrals = matdict[key][0]
    
    #Take the average over all the time points and add to long-time average list
    long_time_average.append(np.average(integrals))
    
    #For the Ls-Lt_ratio-fixed_grain investigation, 
    #save the 4nm integrated value to try and fit a power law
    #Multiply the value by 100 to convert to a percent
    if (investigation in Ls_investigations) and (Ls == min(Lss)):
        base_Ls_nm_value = 100*np.average(integrals)

#Convert the average density of perturbations to a percent (x 100%)
long_time_average = 100 * np.array(long_time_average)
    
#Plot the long-time-average integral value vs the parameter
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(parameters, long_time_average, '-o')

#For the aspect ratio with fixed grain area,
#Try annotating key ratios which give "nice" grain dimensions if provided
if investigation == "aspect_ratio" and len(annotated_ratios) > 0:
    for i in range(len(annotated_ratios)):
        ratio = annotated_ratios[i]
        Lg_x = sqrt_area / math.sqrt(ratio)
        Lg_y = area / Lg_x
        
        #Load the corresponding matfile
        filename = "matfiles/" + integration_quantity 
        filename = filename + ("---density_perturb_from_mean" * density_perturb_from_mean)
        filename = filename + ("_lab" * lab_frame) + "_over_time/" + investigation
        filename = filename + "/Ls-" + str(Ls)
        filename = filename + "--Lt_ratio-" + "{0:.2g}".format(Lt/Ls)
        if investigation == "aspect_ratio":
            filename = filename + "--sqrt_area" + str(sqrt_area)
            filename = filename + "--aspect_ratio" + "{0:.2f}".format(Lg_y/Lg_x)
        else:
            filename = filename + "--Lg_x" + str(Lg_x) + "--Lg_y" + "{0:.1f}".format(Lg_y)
        
        if (integration_quantity != 'IE'):
            filename = filename + "--n_x_intervals-" + str(n_x_intervals) + ".mat"
        
        matdict = io.loadmat(filename)
        
        #Load the time and integrals at each time
        if integration_quantity in ['Mx', 'My', 'Mtotal']:
            key = "M_integral"
        else:
            key = integration_quantity + "_integral"
        t = matdict["t"][0]
        integrals = matdict[key][0]
    
        #Take the average over all the time points and add to long-time average list
        #Convert the average density of perturbations to a percent (x 100%)
        average = 100 * np.average(integrals)
    
        #Plot the point to annotate in a different color
        ax.scatter(ratio, average, color = 'r', s=100)
    
        #Annotate the points with grain dimensions
        annotation = r"$Lg_x=$" + str(round(Lg_x)) + "nm\n"
        annotation = annotation + r"$Lg_y=$" + str(round(Lg_y)) + "nm"
        y_range = max(long_time_average) - min(long_time_average)
        if integration_quantity == "IE":
            x_text = ratio - 0.09*(i==len(annotated_ratios)-1)
            y_text = average + y_range*(-0.17 + 0.55*(i==0))
        else:
            x_text = ratio - 0.1*(i==len(annotated_ratios)-1)
            y_text = average + y_range*(0.40 - 0.85*(i==0))
        ax.annotate(annotation, 
                    xy=(ratio, average),  
                    xycoords='data',
                    xytext=(x_text, y_text), 
                    textcoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.1, width=2, headwidth=8),
                    horizontalalignment= 'center',
                    verticalalignment='top',
                    size = fontsizes['S']
                    )

#Convert quantity appreviation to a full title name
title_names = {"KE": "Kinetic Energy", 
               "IE": "Internal Energy",
               "Mx": "Magnitude of x-Momentum",
               "My": "Magnitude of y-Momentum",
               "Mtotal": "Magnitude of Total Momentum"}

#Generate plot title
title = "Percent Perturbation of " + title_names[integration_quantity] + "\n"
if investigation == "aspect_ratio":
    title = title + r"$\sqrt{Area}=$" + str(sqrt_area)
    title = title + ", Ls=" + str(Ls) + ", Lt=" + str(Lt)
elif investigation in ["Ls-Lt_ratio", "Ls-Lt_constant"]:
    title = title + r"$Lx=$" + str(Lx_original) + r", $Ly=$" + str(Ly_original)
    if investigation == "Ls-Lt_ratio-fixed_grain":
        title = title + ", Lt/Ls=" + str(Lt_ratio)
    elif investigation == "Ls-Lt_constant-fixed_grain":
        title = title + ", Lt=" + str(Lt)
elif investigation in ["Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain"]:
    title = title + r"$Lg_x=$" + str(Lg_x) + r", $Lg_y=$" + "{0:.1f}".format(Lg_y)
    if investigation == "Ls-Lt_ratio-fixed_grain":
        title = title + ", Lt/Ls=" + str(Lt_ratio)
    elif investigation == "Ls-Lt_constant-fixed_grain":
        title = title + ", Lt=" + str(Lt)
elif investigation == "Lg_x":
    title = title + r"$Lg_y/Lg_x=$" + str(aspect_ratio)
    title = title + ", Ls =" + str(Ls) + ", Lt =" + str(Lt)
elif investigation == "aspect_ratio-fixed_Lx":
    title = title + r"$Lg_x=$" + str(Lg_x)
    title = title + ", Ls=" + str(Ls) + ", Lt=" + str(Lt)
elif investigation == "rho_s":
    title = title + r"$Lg_x=$" + str(Lg_x) + r", $Lg_y=$" + "{0:.1f}".format(Lg_y)
    title = title + ", Ls =" + str(Ls) + ", Lt =" + str(Lt)
elif investigation in ["conserve_density-Lt_ratio", "conserve_density-Lt_constant"]:
    title = title + r"$Lx=$" + str(Lx_original) + r", $Ly=$" + str(Ly_original)
    if investigation == "conserve_density-Lt_ratio":
        title = title + ", Lt/Ls=" + str(Lt_ratio)
    elif investigation == "conserve_density-Lt_constant":
        title = title + ", Lt=" + str(Lt)

if investigation not in ["rho_s", "conserve_density-Lt_ratio", "conserve_density-Lt_constant"]:
    title = title + r", $\rho_s=$" + "{0:.2g}".format(rho_s)
title = title + r", $\rho_g=$" + "{0:.2g}".format(rho_g) + r", $M_1=$" + str(M1)

plt.title(title, fontsize = fontsizes['L'], y=1.05)

#Generate x-axis label
if investigation in ["aspect_ratio", "aspect_ratio-fixed_Lx"]:
    ax.set_xlabel(r"Grain Aspect Ratio ($Lg_y/Lg_x$)", fontsize=fontsizes['M'])
elif investigation in Ls_investigations:
    ax.set_xlabel(r"Interstitial Space Width (Ls)", fontsize=fontsizes['M'])
elif investigation == "Lg_x":
    ax.set_xlabel(r"x-Direction Grain Width ($Lg_x$)", fontsize=fontsizes['M'])
elif investigation == "rho_s":
    ax.set_xlabel(r"Interstitial Space Density ($\rho_s$)", fontsize=fontsizes['M'])
    
#Set y-axis label and axis tick size
ax.set_ylabel("Average Percent Perturbation", fontsize=fontsizes['M'])
ax.tick_params(axis='x', labelsize=fontsizes['S'])
ax.tick_params(axis='y', labelsize=fontsizes['S'])

#Tight layout to fit the plot better
fig.tight_layout(rect=[0, 0.03, 1, 0.92])

#Save the plot
figname = "plots/long_time_average_integrals/"
figname = figname + integration_quantity + ("_lab" * lab_frame)
figname = figname + "--" + investigation + "-"
if investigation == "aspect_ratio":
    figname = figname + "--sqrt_area" + str(sqrt_area) + "--Ls" + str(Ls) + "--Lt" + str(Lt)
elif investigation == "aspect_ratio-fixed_Lx":
    figname = figname + "--Lg_x" + str(Lg_x) + "--Ls" + str(Ls) + "--Lt" + str(Lt) 
elif investigation in ["Ls-Lt_ratio", "Ls-Lt_constant"]:
    figname = figname + "--Lx" + str(Lx_original) + "--Ly" + str(Ly_original)
    if investigation == "Ls-Lt_ratio":
        figname = figname + "--Lt_ratio" + str(Lt_ratio)
    elif investigation == "Ls-Lt_ratio":
        figname = figname + "--Lt" + str(Lt)
elif investigation in ["Ls-Lt_ratio-fixed_grain", "Ls-Lt_constant-fixed_grain"]:
    figname = figname + "--Lg_x" + str(Lg_x) + "--Lg_y" + "{0:.1f}".format(Lg_y)
    if investigation == "Ls-Lt_ratio-fixed_grain":
        figname = figname + "--Lt_ratio" + str(Lt_ratio)
    elif investigation == "Ls-Lt_ratio-fixed_grain":
        figname = figname + "--Lt" + str(Lt)
elif investigation == "Lg_x":
    figname = figname + "--aspect_ratio" + str(aspect_ratio)
    figname = figname + "--Ls" + str(Ls) + "--Lt" + str(Lt)
elif investigation == "rho_s":
    figname = figname + "--Lg_x" + str(Lg_x) + "--Lg_y" + str(Lg_y)
    figname = figname + "--Ls" + str(Ls) + "--Lt" + str(Lt)
elif investigation in ["conserve_density-Lt_ratio", "conserve_density-Lt_constant"]:
    figname = figname + "--Lx" + str(Lx_original) + "--Ly" + str(Ly_original)
    if investigation == "conserve_density-Lt_ratio":
        figname = figname + "--Lt_ratio" + str(Lt_ratio)
    elif investigation == "conserve_density-Lt_ratio":
        figname = figname + "--Lt" + str(Lt)

figname + figname + ("---density_perturb_from_mean" * density_perturb_from_mean)

plt.savefig(figname + ".png", facecolor='white', bbox_inches='tight')    


## If we're investigating Ls-Lt_ratio-fixed_grain, then
## Fit a power law function to the plot for comparison with simulations
if investigation in Ls_investigations and fit_power_law:
    base_Ls = min(Lss)
    def power_law(Ls, p):
        return base_Ls_nm_value * np.power(Ls/base_Ls, p)
    
    p, cov = curve_fit(f=power_law, 
                          xdata=Lss, ydata=long_time_average,
                          bounds=(-10,10)) 
    
    print(investigation, ' investigation with ', integration_quantity)
    print('Power law exponent = ', p[0])
    
    p_rounded = round(p[0]*100) / 100 
    print('Rounded to ', p_rounded)
    
    #Plot power law to verify fit
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(Lss, long_time_average, '-o', label="Simulated Value")
    power_law_label = "Power Law Fit\n" + r"|" + integration_quantity
    power_law_label = power_law_label + r"$|_{" + str(min(Lss)) + r"nm}$" 
    power_law_label = power_law_label + "$(Ls/" + str(min(Lss)) + r" nm)^p$"
    power_law_label = power_law_label + "\n" + "p = " + "{0:.2f}".format(p[0])
    ax.plot(Lss, power_law(np.array(Lss), p_rounded), label=power_law_label)
    
    ax.set_xlabel(r"Interstitial Space Width (Ls)", fontsize=fontsizes['M'])
    ax.set_ylabel("Average Percent Perturbation", fontsize=fontsizes['M'])
    ax.tick_params(axis='x', labelsize=fontsizes['S'])
    ax.tick_params(axis='y', labelsize=fontsizes['S'])
        
    plt.title(title, fontsize = fontsizes['L'], y=1.05)
    plt.legend(fontsize = fontsizes['S'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(figname + "--power_law.png", facecolor='white', bbox_inches='tight')
         