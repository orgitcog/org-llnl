# Copyright 2024 Lawrence Livermore National Security, LLC.
# See the top-level LICENCE file for details.
#
# SPDX-License-Identifier: MIT

import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy
import sys
fmt = 'pdf'
trans = False

from data import *

showPlot = False
saveKW = {}
saveKW["format"] =fmt
saveKW["transparent"]=trans
saveKW["bbox_inches"]='tight'

def plt_show():
    global showPlot
    if showPlot:
        plt.show()
    plt.close()
    plt.clf()

def plotDetailedTemp(temp):
    for key in detailedKeys:
        name,rho,T = key
        if T != temp:
            continue
        eps = numpy.array(epsilon[key])
        sigma = rho*numpy.array(kappa[key])
        plt.loglog(eps,sigma,label=f'$\\sigma(\\epsilon)$ {name}')
        plt.xlabel(r'Photon energy, $\epsilon$ [kev]')
        plt.ylabel(r'Opacity, $\sigma = \rho \kappa$ [cm$^{-1}$]')

plotDetailedTemp(0.5)
plt.xlim([0.01,20.0])
plt.ylim([0.1,1e6])
plt.legend(fontsize='small',frameon=False, loc='best')
plt.savefig(f'detailedTemp0.5.{fmt:}', **saveKW)
plt_show()

#showPlot = False

def plotDetailedMat(mat):
    for key in detailedKeys:
        name,rho,T = key
        if name != mat:
            continue
        eps = numpy.array(epsilon[key])
        sigma = rho*numpy.array(kappa[key])
        plt.loglog(eps,sigma,label=f'$\\sigma(\\epsilon) T={T:}$ keV')
        plt.xlabel(r'Photon energy, $\epsilon$ [kev]')
        plt.ylabel(r'Opacity, $\sigma = \rho \kappa$ [cm$^{-1}$]')

plotDetailedMat('foam')
plt.xlim([0.02,10.0])
plt.ylim([0.1,1e5])
plt.legend(fontsize='small',frameon=False, loc='best')
plt.savefig(f'detailedCarbon.{fmt:}', **saveKW)
#plt.xlim([5,8])
#plt.ylim([1,1e6])
#plt.savefig(f'detailedCarbonIron.{fmt:}', **saveKW)
plt_show()


plotDetailedMat('carbon')
plt.xlim([0.02,10.0])
plt.ylim([0.1,1e7])
plt.legend(fontsize='small',frameon=False, loc='best')
plt.savefig(f'detailedCarbon.{fmt:}', **saveKW)
#plt.xlim([5,8])
#plt.ylim([1,1e6])
#plt.savefig(f'detailedCarbonIron.{fmt:}', **saveKW)
plt_show()

plotDetailedMat('iron')
plt.xlim([0.02,50.0])
plt.ylim([1,5e8])
plt.legend(fontsize='small',frameon=False, loc='best')
plt.savefig(f'detailedIron.{fmt:}', **saveKW)
plt.xlim([5,8])
plt.ylim([1,1e6])
plt.savefig(f'detailedZoomIron.{fmt:}', **saveKW)
plt_show()

def plotSigmaNu( key, lowEps = None, highEps = None, pname = "sigma", plotPlanck = False):

    name,rho,T = key

    eps = numpy.array(epsilon[key])
    sigma = rho*numpy.array(kappa[key])

    sigpg = rho*numpy.array(kappaP_g[key])
    sigrg = rho*numpy.array(kappaR_g[key])

    minEps = 0
    if lowEps is not None:
        minEps = numpy.searchsorted(eps,lowEps)
    maxEps = len(eps)-1
    if highEps is not None:
        maxEps = numpy.searchsorted(eps,highEps)

    fig, ax1 = plt.subplots()

    ax1.loglog(eps,sigma,label=r'$\sigma(\epsilon)$')
    ax1.stairs(sigpg,groupBounds,label=r'$\sigma_{P,g}$')
    ax1.stairs(sigrg,groupBounds,label=r'$\sigma_{R,g}$')
    ax1.set_xlabel(r'Photon energy, $\epsilon$ [kev]')
    ax1.set_ylabel(r'Opacity, $\sigma = \rho \kappa$ [cm$^{-1}$]')
    ax1.set_xlim([lowEps,highEps])
    limKappa = sigma[minEps:maxEps]
    mn = math.pow(10,math.floor(math.log10(min(limKappa))))
    mx = math.pow(10,math.ceil(math.log10(max(limKappa))))
    ax1.set_ylim([mn,mx])
    ax1.legend(fontsize='small',frameon=False, loc='center left')

    if plotPlanck:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized weights')

        beps = b[key]
        reps = r[key]

        ax2.semilogx(eps,beps,label=r'$b(\epsilon)$',color='black',linewidth=0.5)
        ax2.semilogx(eps,reps,label=r'$db/dT(\epsilon)$',color='black',linewidth=0.5, linestyle="--")
        ax2.legend(fontsize='small',frameon=False, loc='upper right')

    fig.tight_layout()
    plt.savefig(f'{pname}_{name}_{rho}_{T}.{fmt:}', **saveKW)
    plt_show()

for key in detailedKeys:
    plotSigmaNu( key, lowEps = 1.0e-3, highEps = 50.0, pname = "sigma", plotPlanck=True)

key = ('iron',6,1.0)
name,rho,T = key
plotSigmaNu( key, lowEps = 2.5, highEps = 8.0, pname = "sigmaZoom", plotPlanck=False)

i = 0
for key in greyKeys:

    name,rho = key
    color = f'C{i}'
    i+=1

    kp = rho*numpy.array(kappaP[key])
    kr = rho*numpy.array(kappaR[key])

    plt.loglog(greyTemps,kp,label=f'{name} $\\rho={rho:}$ $\\sigma_P$',color=color)
    plt.loglog(greyTemps,kr,label=f'{name} $\\rho={rho:}$ $\\sigma_R$',color=color, linestyle="--")
plt.legend(fontsize='small',frameon=False, loc='best')
plt.xlabel(r'Material temperature, $T$ [kev]')
plt.ylabel(r'Grey opacity, $\sigma$ [cm$^{-1}$]')
#plt.xlim([1.e-2,50.0])
#limKappa = kap[0:maxEps]
#mn = math.pow(10,math.floor(math.log10(min(limKappa))))
#mx = math.pow(10,math.ceil(math.log10(max(limKappa))))
#plt.ylim([mn,mx])
plt.savefig(f'greySigmas.{fmt:}', **saveKW)
plt_show()


