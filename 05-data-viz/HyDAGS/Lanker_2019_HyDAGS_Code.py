#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
#
# SPDX-License-Identifier: MIT

"""
@author: Cory Lanker, lanker1@llnl.gov, March 9, 2019

License Information:

  This work is covered by the MIT Licence.  Please see the file LICENSE in this
  directory.  LLNL-CODE-773161

*******************************************************************************
*******************************************************************************
**      This research is funded by Hyperspectral Advanced Research and       **
**    Development for Solids (PL14-FY14-112-PD3WA) and is performed under    **
**    the auspices of the U.S. Department of Energy by Lawrence Livermore    **
**  National Laboratory under Contract DE-AC52-07NA27344. LLNL-JRNL-753011.  **
*******************************************************************************
*******************************************************************************


Citation Policy:

  If you publish material based on this method, then, in your acknowledgements,
  please note the assistance you received by using this method and its code.
  The authors would appreciate you citing the following article:

      Lanker, Cory, and Milton O. Smith. A sparse Gaussian sigmoid basis
      function approximation of hyperspectral data for detection of solids.
      Statistical Analysis and Data Mining, 2019;00:xx-xx.


Table of Contents:

  1. Functions I've written that are necessary for running our method
  2. Analysis of Dr. Melissa Lane's calcite measurements and creation
     of the plots of Section 3.1 of our journal article.
  3. A set of simulation studies, analyses, and creation of the plots of
     Section 3.2 of our journal article.

"""


import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
from sklearn import discriminant_analysis
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.lines import Line2D


"""
*****************************************************************************
    1. Functions I've written that are necessary for running the method
*****************************************************************************
"""


def generate_spectral_var(wvl, M, nlev1, nlev2, seed):
    """
    A function for the simulation studies.
    Generates M samples of wavelength vector wvl incorporating
        spectral variability based on Gaussian processes

    Gaussian process 1 is Gaussian kernel with variance 0.25
      - generates broad variability with emissivity amplitude nlev1
    Gaussian process 2 is Gaussian kernel with variance 0.00125
      - generates narrow variability with emissivity amplitude nlev2

    Seed sets the numpy.random seed
    """
    np.random.seed(seed)
    eps = 1e-5
    n = len(wvl)

    K = np.asarray([[np.exp(-0.5 * (a-b)**2 / 0.25)
                     for a in wvl] for b in wvl])
    L = np.linalg.cholesky((1-eps) * K + eps * np.eye(n))
    f_broad = nlev1 * np.dot(L, np.random.normal(size=(n, M))).T
    f_broad -= np.mean(f_broad, axis=1).reshape(-1, 1)

    K2 = np.asarray([[np.exp(-0.5 * (a-b)**2 / 0.00125)
                      for a in wvl] for b in wvl])
    L2 = np.linalg.cholesky((1-eps) * K2 + eps * np.eye(n))
    f_narrow = nlev2 * np.dot(L2, np.random.normal(size=(n, M))).T
    f_narrow -= np.mean(f_narrow, axis=1).reshape(-1, 1)

    return f_broad + f_narrow


def generate_target(wvl, comp):
    """
    A function for the simulation studies.
    Generates the target spectrum for target feature component comp.

    The five features, value of comp:
      1: 0.1 * Gaussian(8.8, 0.1) will span 8.55 to 9.05
      2: 0.2 * Lorentzian(9.3, 0.1) will span 9.05 to 9.55
      3: 0.2 * Gaussian(10.1, 0.25 on left, 0.125 on right)
         will span 9.6 to 10.4
      4: 0.1 * Gaussian(10.8, 0.125) + 0.1 * Gaussian(11, 0.125)
         will be flat-topped feature spanning 10.5 to 11.4
      5: 4th order polynomial will be negative volume scattering feature
         starting at 11.8 until 12.4 of height -0.1.
    """
    f1 = 0.1 * (sp.stats.norm.pdf(wvl, 8.8, 0.1)*np.sqrt(2*np.pi)*0.1)
    f2 = 0.2 * (0.25*(0.1)**2/((wvl-9.3)**2 + (0.5*0.1)**2))
    f3 = 0.2 * np.append(sp.stats.norm.pdf(wvl[:41], 10.1, 0.25) *
                         np.sqrt(2*np.pi)*0.25,
                         sp.stats.norm.pdf(wvl[41:], 10.1, 0.125) *
                         np.sqrt(2*np.pi)*0.125)
    f4 = 0.1 * (sp.stats.norm.pdf(wvl, 10.8, 0.125)*np.sqrt(2*np.pi)*0.125)
    f4 += 0.1 * (sp.stats.norm.pdf(wvl, 11.0, 0.125)*np.sqrt(2*np.pi)*0.125)
    f5 = 2.0 * ((wvl-12.1)**2 - (1.5*(wvl-12.1))**4)
    f5[:82] = f5[82]
    f5[99:] = f5[98]

    target = np.zeros(len(wvl))
    f_list = [f1, f2, f3, f4, f5]
    if comp == 'all':
        comp = list(range(1, 6))
    for n in comp:
        target += f_list[n-1]

    return target


def generate_data(M, nlev1, nlev2, seed, comp):
    """
    A function for the simulation studies.
    Generates M observations of emissivity with (em) and without (em0)
        the target feature specified by component comp.
    Also output are wvl, the wavelength spectral bins, and target,
        the target spectrum
    """
    wvl = np.linspace(8.5, 12.5, 101)
    # This chosen wavelength range is a generic subset of the LWIR range
    # approximately in the main region of atmospheric transmission.
    # See John Schott, Remote Sensing: The Image Chain Approach, 2006,
    # where Figure 3.23 on p.105 shows the atmospheric transmission
    # range above 0.75 as 8.6 to 12.3 um.
    em = generate_spectral_var(wvl, M, nlev1, nlev2, seed)
    em0 = em.copy()
    target = generate_target(wvl, comp)
    target -= np.mean(target)
    em += target
    return wvl, em, em0, target


def generate_bases(wvl, wid=[1, 2, 4, 8]):
    """
    Generates basis function collection based on wvl and widths
        based on diff(wvl). Centers set to observed wvl.
        The wider basis functions are thinned relative to width.
    Inputs:
        wvl = wavelength vector, these will be the centers
            of the basis functions
        wid = integer list of widths relative to diff(wvl) spacing
    Outputs:
        parval = parameter matrix of basis function centers and sigmas,
            total width for a basis function is approximately 4*sigma
        B = basis functions,
            rows are basis functions, columns are wavelengths
    """
    parval = []
    for k, j in enumerate(wid):
        for i in wvl[(k+1):-1*(k+1):2**k]:
            parval.append([i, j])

    # These lines turn the parameter matrix parval into basis functions:
    parval = np.array(parval)
    B = []
    for k, x in enumerate(parval):
        index = np.argmin(np.fabs(wvl - x[0]))
        parval[k, 1] *= np.mean(np.diff(wvl[index-1:index+2]))
        B.append(parval[k, 1] * sp.stats.norm.cdf(wvl, loc=x[0],
                                                  scale=parval[k, 1]))
    B = np.array(B)

    # demean rows of B
    for i in range(B.shape[0]):
        B[i, ] -= np.mean(B[i, ])

    return parval, B


def coefSmooth(pars1, pars2, smoothSigma=1.0, eps=1e-3):
    """
    Here pars1 and pars2 are instances of parval. If pars1 = pars2, then
        the parval coefficients become a smoothed version onto itself.
    If X1 has parameter array pars1 and fit coefficients in matrix b, then
        b.dot(transformmatrix) yields smoothed fits in parameter array pars2.
    smoothSigma = smoothing factor, 1.0 should work well universally.
        (values: 0.5 for only immediate width neighbor,
                 1.0 for reasonable smoothing along center and width,
                 1.5+ for broader smoothing, if desired)
    """
    transformmatrix = np.zeros((pars1.shape[0], pars2.shape[0]))
    for i in range(pars2.shape[0]):
        temp = np.array([[(pars2[i, 1]**2)/4. * smoothSigma, 0],
                         [0, (pars2[i, 1]**2)/16. * smoothSigma]])
        temp = sp.stats.multivariate_normal.pdf(pars1, mean=pars2[i, ],
                                                cov=temp)
        transformmatrix[:, i] = temp / (np.sum(temp) + 1e-20)

    # next loop forces rows of transformmatrix to add to 1
    for i in range(pars1.shape[0]):
        if sum(transformmatrix[i, ]) > 0:
            transformmatrix[i, ] /= np.sum(transformmatrix[i, ])

    # eliminate small values
    transformmatrix[transformmatrix < eps] = 0.

    return transformmatrix


def gmvector(M):
    """
    Quickly solves the Ax=b for vectorized A using Gauss-Markov elimination.
    This function is required for function runL1 (LARS-LASSO fitting).
    """
    # Note that A cannot be singular.
    n = np.int(np.sqrt(M.shape[1]))
    D = M.copy(order='F')
    # This loops ensures there aren't empty rows (making A singular)
    for i in range(0, n):
        D[D[:, i*(n+1)+i] <= np.finfo(float).eps, i*(n+1)+i] = 1

    # This loop will make A upper triangular
    for i in range(0, n):
        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            c = -D[:, (n+1)*k+i]/D[:, (n+2)*i]
            for j in range(i, n+1):
                if i == j:
                    D[:, (n+1)*k+j] = 0
                else:
                    D[:, (n+1)*k+j] += c * D[:, (n+1)*i+j]

    # Solves equation Ax=b for an upper triangular matrix A
    x = np.zeros((M.shape[0], n))
    for i in range(n-1, -1, -1):
        x[:, i] = D[:, (n+1)*i+n]/D[:, (n+2)*i]
        for k in range(i-1, -1, -1):
            D[:, (n+1)*k+n] -= D[:, (n+1)*k+i] * x[:, i]
    return x


def runL1(y, X, thresh):
    """
    Finds LASSO fit coefficient values given data y and basis functions X
        using LARS (least angle regression) methodology.
    Important note:
        This code will not find the exact LARS solutions due to two
        differences. First, when the L1 penalty is reached, the last
        included basis function is allowed to accumulate as much coefficient
        as necessary, regardless of passing the penalty threshold. Second,
        when the L1 penalty is reached in removing a basis function (if
        possible) the code allows one more basis function to be added, i.e.,
        the code may not end its fitting on a removal. There are reasons
        relevant to the application of LARS-LASSO for allowing these two
        differences.
      ***************************  CAUTION  ********************************
      ****   The above differences are important to know if modifying   ****
      ****      this code to use for other LARS-LASSO algorithms.       ****
      *********************************************************************
    Reference:
        Efron, B., T. Hastie, I. Johnstone, R. Tibshirani, et al., (2004).
        Least angle regression. The Annals of Statistics, 32, no.2, 407–499.

    Inputs:
        y = emissivity data, rows are spectra, columns are spectral bins
        X = basis functions, rows are bases, columns are spectral bins
        thresh = cutoff for LARS search, a function of the L1 penalty
    Output:
        b = coefficient vector per pixel
        t = total runtime
    """

    # demean rows of y, if necessary
    if np.any(~np.isclose(np.mean(y, axis=1), 0)):
        for i in range(y.shape[0]):
            y[i, ] -= np.mean(y[i, ])
    # demean rows of X, if necessary
    if np.any(~np.isclose(np.mean(X, axis=1), 0)):
        for i in range(X.shape[0]):
            X[i, ] -= np.mean(X[i, ])
    startTime = time.time()  # to get computation time
    S = X.dot(X.T)   # X'X covariance matrix
    k, n = X.shape   # k=number of bases, n=spectral bins
    p = y.shape[0]   # number of pixels

    # important variables:
    b_save = np.zeros((len(thresh), p, k))
    b = np.zeros((p, k))  # coefficients, the LASSO fit
    Ind = np.array(np.zeros((p, k)), dtype='?')  # active indices
    TI = np.array(np.ones((p, len(thresh))), dtype='?')  # threshold indices
    pixel_active = np.max(TI, axis=1)  # pixel is active if TI is True for any

    r = y - b.dot(X)  # residuals
    t = r.dot(X.T)  # benefit for b[k] (relative to reduction in SSE)
    # next_index = choose t that's most beneficial (in terms of SSE reduction)
    next_index = np.argmax(np.fabs(t), axis=1)
    for i in range(p):
        Ind[i, next_index[i]] = True
    loopno = 0  # loop counter
    while np.min(np.sum(Ind, axis=1)) < (k*0.75):
        # Stop when all pixels have at least 3/4 basis functions in fit or
        #   all pixel coefficients optimized according to l1_penalty

        pixel_active = np.max(TI, axis=1)  # pixel active if TI is True for any
        if not np.any(pixel_active):
            break
        AI = np.sum(Ind, axis=1)  # num of active indices per pixel
        lm = np.int(np.percentile(AI[AI < k], 75))
        # lm: prevents inefficiency if only a few pixels require large Cmat

        # Cmat = Gauss-Markov matrix for solving Ax=b
        #   where values are X'X[active indices, active indices] (the 'A')
        #   with the t values in an appended column (the 'b')
        #   and then vectorized by row.  This solves for the x in Ax=b.
        Cmat = np.zeros((np.sum(AI <= lm), (lm+1)*lm))
        for i1, i2 in enumerate(np.where(AI <= lm)[0]):
            for j1, j2 in enumerate(np.where(Ind[i2, ])[0]):
                Cmat[i1, (lm+1)*j1:(lm+1)*j1+np.sum(Ind[i2, ])] = \
                   S[j2, Ind[i2, ]]
            Cmat[i1, (lm+2)*(j1+1)::lm+2] = 1  # makes A full rank in Ax=b
            Cmat[i1, lm:lm+(lm+1)*np.sum(Ind[i2, ]):lm+1] = t[i2, Ind[i2, ]]
        # del_b = change in beta coefficients to solve OLS for active bases
        del_b = np.zeros((np.sum(AI <= lm), b.shape[1]))
        del_sub = gmvector(Cmat)
        for i1, i2 in enumerate(np.where(AI <= lm)[0]):
            del_b[i1, Ind[i2, ]] = del_sub[i1, :np.sum(Ind[i2, ])]
        # opt_* = b,r,t at OLS solution for y and X[active basis functions]
        opt_b = b[AI <= lm, ] + del_b
        opt_r = y[AI <= lm, ] - opt_b.dot(X)
        opt_t = opt_r.dot(X.T)
        # v and w help find where to stop between b and opt_b
        # stopping point is where another basis function is as beneficial
        #   as current active bases
        v = np.fabs(opt_t)
        w = (opt_t - t[AI <= lm, ]) * np.sign(opt_t)
        w += np.tile(np.max(np.fabs(t[AI <= lm, ]), axis=1), (v.shape[1], 1)).T
        v[~Ind[AI <= lm, ], ] /= w[~Ind[AI <= lm, ], ]
        v *= ~Ind[AI <= lm, ]
        # next_index = next basis function to be added at next_dist
        next_index = np.argmax(v, axis=1)
        next_dist = np.max(v, axis=1)
        # next_* = b,r,t at resting point where another basis function added
        next_b = b[AI <= lm, ] + np.tile(1 - next_dist,
                                         (b.shape[1], 1)).T * del_b
        next_r = y[AI <= lm, ] - next_b.dot(X)
        next_t = next_r.dot(X.T)
        # next_t contains the highest alpha (L1 penalty) required for
        #   that basis function to be included in the fit
        # Therefore, if the highest next_t is beneath the L1 threshold
        #   then the algorithm stops fitting (but adds the full value of
        #   the next basis function)

        # Determine if any beta cross zero from b to next_b
        b_change = np.matrix(np.sign(b[AI <= lm, ]) != np.sign(next_b))
        b_change[b[AI <= lm, ] == 0] = False
        g2 = np.fabs(b[AI <= lm, ]) + np.fabs(next_b)
        g1 = np.fabs(next_b)
        g1[~b_change] = 0
        g1[b_change] /= g2[b_change]
        # if any g1 > 0 then a coefficient crosses zero and should be removed
        #   in which case, next_index is not added, instead
        #   basis function rem_index is removed
        rem_index = np.argmax(g1, axis=1)
        br1 = np.where(AI <= lm)[0]
        br1pos = np.where(np.max(g1, axis=1) > 0.)[0]
        br2 = br1[br1pos]
        br1neg = np.where(np.max(g1, axis=1) <= 0.)[0]
        br2c = br1[br1neg]

        v2 = np.asarray([g1[i, j] for i, j in enumerate(rem_index)])
        v2 = np.tile(v2, (b.shape[1], 1)).T
        b_zero = b[br1, ] * v2 + next_b * (1.-v2)
        b[br2, ] = b_zero[br1pos, ]
        for i, j in zip(br2, rem_index[br1pos]):
            b[i, j] = 0.
            Ind[i, j] = False

        t_val = np.max(np.fabs(next_t), axis=1)
        cutoff = (np.sum(Ind[br1, ], axis=1) >= (k*0.75))
        t_val[cutoff] = 0.
        Ind[br1[cutoff], :] = True

        ti_sat = (np.tile(t_val[br1neg], (len(thresh), 1)).T <
                  np.tile(thresh, (len(br2c), 1))) * TI[br2c, :]

        for z1, z2 in enumerate(np.transpose(ti_sat)):
            b_save[z1, br2c[z2], :] = next_b[br1neg[z2], :]
            TI[br2c[z2], z1] = False

        Ind[np.all(np.invert(TI), axis=1), :] = True
        b[br2c, ] = next_b[br1neg, ]
        for i, j in zip(br2c, next_index[br1neg]):
            Ind[i, j] = True

        r = y - b.dot(X)
        t = r.dot(X.T)
        loopno += 1

    return np.asarray(b_save, dtype='float32'), time.time()-startTime


def reduce_lists(ids, metric, verbose=False):
    """
    This function reduces nearby basis functions of the same sign
        to a single representative basis function for that feature.
    """
    ids = list(ids)
    temp = ids.copy()
    for j in temp:
        if (j-1) in ids and metric[j] > metric[j-1]:
            if verbose:
                print('j-1 removed from id list for', j)
            ids.remove(j-1)
        if (j+1) in ids and metric[j] > metric[j+1]:
            if verbose:
                print('j+1 removed from id list for', j)
            ids.remove(j+1)
    return ids


def get_slopes(x0, x1, lim, cutoff, f_no, parval, wvl, cval, lowerlim, intlen):
    """
    A function for the simulation studies.
    Reduces the important common features for the LDA classifier.
    """
    v = np.mean(x1 > lim, axis=0)
    v -= np.mean(x0 > lim, axis=0)
    idp = np.where(v > cutoff)[0]
    idp = reduce_lists(idp, v)
    v = np.mean(x1 < -1*lim, axis=0)
    v -= np.mean(x0 < -1*lim, axis=0)
    idn = np.where(v > cutoff)[0]
    idn = reduce_lists(idn, v)

    # remove ultra-broad basis functions
    idp = [x for x in idp if parval[x, 1] < 0.25]
    idn = [x for x in idn if parval[x, 1] < 0.25]
    if f_no < 4:
        idp = [x for x in idp if parval[x, 0] <= cval[f_no] and
               parval[x, 0] >= wvl[lowerlim[f_no]-2] and
               parval[x, 0] <= wvl[lowerlim[f_no]+intlen[f_no]+2]]
        idn = [x for x in idn if parval[x, 0] >= cval[f_no] and
               parval[x, 0] >= wvl[lowerlim[f_no]-2] and
               parval[x, 0] <= wvl[lowerlim[f_no]+intlen[f_no]+2]]
    else:
        idn = [x for x in idn if parval[x, 0] <= cval[f_no] and
               parval[x, 0] >= wvl[lowerlim[f_no]-2] and
               parval[x, 0] <= wvl[lowerlim[f_no]+intlen[f_no]+2]]
        idp = [x for x in idp if parval[x, 0] >= cval[f_no] and
               parval[x, 0] >= wvl[lowerlim[f_no]-2] and
               parval[x, 0] <= wvl[lowerlim[f_no]+intlen[f_no]+2]]

    return idp, idn


"""
*****************************************************************************
    2. Analysis of Dr. Melissa Lane's calcite measurements and creation
       of the plots of Section 3.1 of our journal article.


Acknowledgment:

    I thank Dr. Melissa Lane for graciously allowing these calcite
        emissivity measurements for distribution with this journal article and
        their general distribution for these analyses.

Copyright:

    These calcite measurements provided in file "Lane_1999_Midinfrared_Fig3c_
        Reduced.csv" are copyright to her, 1999, and are displayed in
        Figure 3(c) from her journal article that should be attributed in any
        derivative works:

            Lane, Melissa D.  Midinfrared optical constants of calcite
            and their relationship to particle size effects in thermal
            emission spectra of granular calcite.  Journal of Geophysical
            Research, 1999, 104, no. E6, 14099–14108.

    For complete data of these calcite emissivity measurements, please write
        to Dr. Melissa Lane at <lane@fibergyro.com>. These measurements have
        been reduced from their full size and values in the following ways:
            (1) Our subset of the entire data is the 8.5 to 12.5 um range.
            (2) All wavelengths are rounded to 3 decimals.
            (3) All emissivities are rounded to 4 decimals.

*****************************************************************************
"""

# 2-1. Read in data.
fileName = 'Lane_1999_Midinfrared_Fig3c_Reduced.csv'
with open(fileName, 'r') as f:
    dat = f.readlines()

colNames = dat[5]
d = []
for x in dat[6:]:
    d.append([float(y) for y in x.rstrip().split(',')])
d = np.asarray(d)
d_n = colNames[6:].rstrip().split(',')

# 2-2. Calcite emissivity plot.
# A plot that shows the similarity in emissivities versus
# particle size.
custom_lines = [Line2D([0], [0], color='C0', lw=0.8, ls='-'),
                Line2D([0], [0], color='C2', lw=0.7, ls=':'),
                Line2D([0], [0], color='C4', lw=0.7, ls='--')]

plt.figure(figsize=(3.125, 1.75), dpi=1200)
plt.rcParams['font.size'] = 5
lslist = ['-', '-', ':', '-', '--']
for i in range(5):
    plt.plot(d[:, 0], d[:, i+1], lw=1-0.25*(i % 2), ls=lslist[i])

plt.grid()
plt.xlim(8.5, 12.5)
plt.xlabel('wavelength ($\lambda$)')
plt.ylabel('measured emissivity')
plt.yticks(np.linspace(0.8, 0.98, 10))
plt.legend(custom_lines, ['%s $\mu$m' % x for x in d_n[::2]], fontsize=5)
plt.tight_layout()
plt.savefig('figure1.png')

# 2-3. Run fitting algorithm, find common upslopes (idp) and downslopes (idn)
wvl = d[:, 0]
# create parameters parval (mu, sigma) for basis functions of Xmat
parval, Xmat = generate_bases(wvl)
t_list = [2e-5]  # This value for L1 penalty allows enough detail
                 # without overfitting every small emissivity change.
im = d[:, 1:].copy()  # use 63-90 to 250-355 um ranges
# Above 355um the emissivities are relatively featureless, and though
# the results are good, they aren't as demonstrable as an analysis of
# simply these five ranges:  particle sizes with very different spectral
# shapes while having the same common upslopes/downslopes.
im_means = np.mean(im, axis=0)  # The code will demean the spectra in situ,
                                # so the means are saved for later plotting.
# Runs the LARS-LASSO fitting algorithm:
b, t = runL1(y=im.T, X=Xmat, thresh=t_list)
# Minor smoothing of the coefficients to adjacent basis functions
# in the (mu, sigma) plane helps to link their common features.
tmat = coefSmooth(parval, parval, 1., eps=1e-4)
bf = b[0, :, :].copy().dot(tmat)  # The first index of b is for
# the threshold list, but only one element is used for t_list here.

# idp = column id's for Positive slopes
# idn = column id's for Negative slopes
# A threshold of 1e-5 captures the bulk of any non-zero coefficients:
idp = np.where(np.min(bf, axis=0) > 1e-5)[0]
idn = np.where(np.max(bf, axis=0) < -1e-5)[0]
# The function reduce_lists reduces nearby basis functions of the same
# sign to a single representative basis function for that feature:
idp = reduce_lists(idp, np.mean(bf, axis=0))
idn = reduce_lists(idn, -1*np.mean(bf, axis=0))
# Should result in idp length 31 features and idn length 33 features.

# 2-4. Plot of common features.
plt.figure(figsize=(3.125, 1.25), dpi=1200)
plt.rcParams['font.size'] = 5
isx = [(x in idp) for x in range(len(parval))]
plt.plot(parval[isx, 0], parval[isx, 1], 'rd', ms=1.5)
isx = [(x in idn) for x in range(len(parval))]
plt.plot(parval[isx, 0], parval[isx, 1], 'bs', ms=1.5)
plt.grid()
plt.xlim(8.5, 12.5)
plt.ylim(0.01, 0.3)
plt.ylabel('$\sigma$ of basis function')
plt.xlabel('$\mu$ of basis function')
plt.yscale('log')
plt.tight_layout()
plt.savefig('figure2.png')

# 2-5. Using only the common features, fit a Lasso regression with very
# small penalty with Positive-coefficients only = True, thus getting the true
# fit of spectra with only the common features in the same direction.
plt.figure(figsize=(3.125, 2.), dpi=1200)
plt.rcParams['font.size'] = 5
pl, r2list = [], np.zeros(5)
for i in range(5):
    # plot spectra:
    tp = plt.plot(d[:, 0], d[:, i+1], c='C%d' % i, lw=1-0.25*(i % 2),
                  ls=['-', '-', ':', '-', '--'][i])
    pl.extend(tp)
    # plot linear regression of spectra using common Gaussian sigmoids:
    X = np.column_stack([Xmat[idp, :].T, -1*Xmat[idn, :].T])
    rlm = Lasso(alpha=1e-9, positive=True, tol=1e-9, max_iter=20000)
    rlm.fit(X, d[:, i+1])
    rlm_fit = rlm.predict(X)
    plt.plot(d[:, 0], rlm_fit, c='k', lw=0.75)  # Gaussian sigmoid fit
    r2list[i] = 1 - (np.sum((d[:, i+1] - rlm_fit)**2) /
                     np.sum((d[:, i+1] - np.mean(d[:, i+1]))**2))

plt.grid()
plt.xlim(8.5, 12.5)
plt.ylim(0.77, 0.98)
plt.ylabel('emissivity')
plt.xlabel('wavelength ($\lambda$)')
plt.yticks(np.linspace(0.78, 0.98, 11))
plt.legend(pl, ['%s $\mu$m, R$^2$=%.4f' % (x, y) for x, y
                in zip(d_n, r2list)], fontsize=5, labelspacing=0.2)
plt.tight_layout()
plt.savefig('figure3.png')


"""
*****************************************************************************
    3. A set of simulation studies, analyses, and creation of the plots of
       Section 3.2 of our journal article.
*****************************************************************************
"""

# 3-1. Run 5 target feature simulation studies and save their output
l1pen = [0.001]  # L1 penalty, a value that has balance between
                 # narrow and broad features
lim = 0.01  # threshold for smoothed function coefficients
cutoff = 0.02  # necessary difference in feature presence between

# Variables for common upslope and downslope functions.
# These values have been found to optimize the correlation scores.
# Accordingly, I also used these values for the range of acceptable
# basis functions to be input into the LDA classifier.
cval = [8.8, 9.3, 10.1, 10.9, 12.1]  # center wavelength of target features
lowerlim = [2, 15, 30, 52, 83]  # first index of important correlation region
intlen = [12, 11, 16, 17, 15]  # length of correlation region interval

N, R = 2000, 50
# stores all the predicted values for ROC curve plotting:
predvals = np.zeros((5, R, 2*N))
# stores all target correlation values for ROC curve plotting:
corrvals = np.zeros((5, R, 2*N))
# store stats of the fitting, time in 6th column is for 4*N spectra:
te_stats = np.zeros((5, R, 7))
# raw counts of significant features
fp = np.zeros((5, R, 184), dtype='?')   # 184 = number of basis functions
fn = fp.copy()
for rep in range(1, R+1):
    print('Starting repetition', rep, '...')

    for f_no in range(5):
        wvl, im, im0, target = generate_data(4*N, nlev1=0.125, nlev2=0.025,
                                             seed=100*rep+f_no, comp=[f_no+1])

        # essentially im[2*N:, ] and im0[:2*N, ] are thrown away, but this
        # ensures the background spectra are different between classes
        dwvl = np.mean(np.diff(wvl))
        parval, Xmat = generate_bases(wvl)
        b, t = runL1(y=im[:2*N, ], X=Xmat, thresh=l1pen)
        b0, t0 = runL1(y=im0[2*N:4*N, ], X=Xmat, thresh=l1pen)
        tmat = coefSmooth(parval, parval, 1.)
        bf = b[0, :, :].dot(tmat)
        bg = b0[0, :, :].dot(tmat)

        # Training set:
        x1 = bf[:N, :].copy()
        x0 = bg[:N, :].copy()
        X = np.row_stack([x1, x0])

        # Get list of indices
        idp, idn = get_slopes(x0, x1, lim, cutoff, f_no, parval, wvl,
                              cval, lowerlim, intlen)
        # Zero all features in opposite direction of the target
        # This is to prevent learning the background rather than the target.
        for j in idp:
            X[X[:, j] < 0, j] = 0
        for j in idn:
            X[X[:, j] > 0, j] = 0
        idt = np.union1d(idp, idn)
        fp[f_no, rep-1, idp] = True
        fn[f_no, rep-1, idn] = True

        # form LDA classification model based on idt features
        clf = discriminant_analysis.LinearDiscriminantAnalysis()
        Y = np.append(np.ones(N), np.zeros(N))
        clf.fit(X[:, idt], Y)
        pred = clf.predict_proba(X[:, idt])[:, 1]
        tr_auc = roc_auc_score(Y, pred)

        # Testing set:
        x1 = bf[N:, :].copy()
        x0 = bg[N:, :].copy()
        Xte = np.row_stack([x1, x0])
        Yte = np.append(np.ones(N), np.zeros(N))

        # Also zero features in opposite direction of target relationship
        for j in idp:
            Xte[Xte[:, j] < 0, j] = 0
        for j in idn:
            Xte[Xte[:, j] > 0, j] = 0
        # Predict probabilities of presence of target in spectra
        #   based on the trained model and learned target features
        #   using similar fits of test set spectra
        pred = clf.predict_proba(Xte[:, idt])[:, 1]
        predvals[f_no, rep-1, :] = pred  # save predicted probabilities
        te_stats[f_no, rep-1, :6] = [len(idp), len(idn), len(idt),
                                     roc_auc_score(Yte, pred), tr_auc, t+t0]

        istarget = list(range(lowerlim[f_no], lowerlim[f_no]+intlen[f_no]))
        p1 = [np.corrcoef(x[istarget], target[istarget])[0, 1] for
              x in im[N:2*N, ]]  # N:2N is test set for target class
        p0 = [np.corrcoef(x[istarget], target[istarget])[0, 1] for
              x in im0[3*N:4*N, ]]  # 3N:4N is test set for bkgd class
        corrvals[f_no, rep-1, :] = np.append(p1, p0)
        te_stats[f_no, rep-1, 6] = roc_auc_score(Yte, corrvals[f_no, rep-1, :])

        out = np.mean(te_stats[f_no, :rep, :], axis=0)
        print('rep %2i  f_no %i   ROC AUC %.6f vs. %.6f  [%.1f, %.1f] t=%.2f' %
              (rep, f_no+1, out[3], out[6], out[0], out[1], out[5]))

    np.save('simstudy_output.npy',
            {'predvals': predvals, 'corrvals': corrvals,
             'te_stats': te_stats, 'fp': fp, 'fn': fn})

temp = np.load('simstudy_output.npy').tolist()
predvals = temp['predvals']
corrvals = temp['corrvals']
te_stats = temp['te_stats']
fp = temp['fp']
fn = temp['fn']

# 3-2. Plot examples of background spectra
fig = plt.figure(figsize=(3.125, 1.5), dpi=1200)
plt.rcParams['font.size'] = 5
wvl, _, em0, target = generate_data(8, nlev1=0.125, nlev2=0.025,
                                    seed=2, comp=[1])
lslist = ['-', '--', '-.', ':']
for j in range(8):
    plt.plot(wvl, em0[j], ls=lslist[j % 3], lw=0.25, c='C%d' % (5+2*(j//4)))

for f_no in range(5):
    wvl, _, _, target = generate_data(10, 0, 0, 0, comp=[f_no+1])
    dt = np.percentile(target, 50)
    plt.plot(wvl, target - dt, c='k', ls='--', lw=0.5)
    l1 = lowerlim[f_no]
    l2 = lowerlim[f_no] + intlen[f_no]
    plt.plot(wvl[l1:l2], target[l1:l2] - dt, c='C%d' % f_no, lw=1)
    plt.text(cval[f_no]-0.025, 0.01, str(f_no+1), fontsize=5)

plt.xlabel('wavelength ($\lambda$)')
plt.ylabel('relative emissivity')
plt.xlim(8.5, 12.5)
plt.ylim(-0.3, 0.3)
plt.yticks(np.linspace(-0.3, 0.3, 7))
plt.grid()
plt.tight_layout()
fig.savefig('figure4.png')

# 3-3. Plotting common features per simulation study:
mList = ['#1 Small Gaussian', '#2 Lorentzian',
         '#3 Different slopes', '#4 Double Gaussian',
         '#5 Quartic curve']
wvl, _, _, _ = generate_data(10, 0, 0, 0, comp=[1])
parval, Xmat = generate_bases(wvl)
tmat = coefSmooth(parval, parval, 1.)

fig = plt.figure(figsize=(3.125, 1.25), dpi=1200)
plt.rcParams['font.size'] = 5
for j in range(5):
    wvl, _, _, target = generate_data(10, 0, 0, 0, comp=[j+1])
    v = 0.1-0.025*j
    target = target*0.25
    target -= np.percentile(target, 50)
    plt.plot(wvl, target+v, lw=0.5, c='k', ls='-')
    plt.plot([8.5, 12.5], [v, v], 'k--', lw=0.25)
    if j < 3:
        plt.text(12.4, v+0.003, mList[j], horizontalalignment='right')
    else:
        plt.text(8.6, v+0.003, mList[j], horizontalalignment='left')

    for k in np.where(np.mean(fp[j, :, :], axis=0) > 0.5)[0][::-1]:
        plt.plot([parval[k, 0]], [v], 'rd', mfc='w', ms=40*parval[k, 1])
    for k in np.where(np.mean(fn[j, :, :], axis=0) > 0.5)[0][::-1]:
        plt.plot([parval[k, 0]], [v], 'bs', mfc='w', ms=35*parval[k, 1])
plt.xlim(8.5, 12.5)
plt.xlabel('wavelength ($\lambda$)')
plt.yticks([])
plt.tight_layout()
plt.savefig('figure5.png')

# 3-4. Plot ROC curves for the simulations studies.
nf = [np.mean(np.sum(fn[j, ], axis=1) + np.sum(fp[j, ], axis=1))
      for j in range(5)]

fig = plt.figure(figsize=(2.5, 2.5), dpi=1200)
plt.rcParams['font.size'] = 5
lw = [4/3*x for x in [0.6, 0.75, 0.6, 0.4, 0.5]]
ls = ['--', '-', '-.', ':', '-']
for j in range(5):
    Y = [int(x) for x in np.append(np.ones(N*R), np.zeros(N*R))]
    pred = predvals[j, ].T.reshape(-1, 1)
    auc = roc_auc_score(Y, pred)
    xval, yval, _ = roc_curve(Y, pred)
    pred2 = corrvals[j, ].T.reshape(-1, 1)
    auc2 = roc_auc_score(Y, pred2)
    xval2, yval2, _ = roc_curve(Y, pred2)
    # Plot our method's results:
    plt.plot(xval, yval, lw=lw[j], ls=ls[j],
             c=plt.cm.bwr(20*j),
             label='%s\n       (AUC = %.4f, ' % (mList[j], auc) +
             '%.1f functions)' % nf[j])
    # Plot results from using correlations:
    plt.plot(xval2, yval2, lw=lw[j]*0.7, ls=ls[j],
             c=plt.cm.bwr(255-20*j),
             label='%s (correlation AUC = %.4f)' % (mList[j][:2], auc2))

plt.xscale('log')
plt.yscale('log')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(1e-5, 1.015)
plt.ylim(1e-5, 1.015)
plt.legend(fontsize=5, labelspacing=1./3)
plt.grid()
plt.tight_layout()
fig.savefig('figure6.png')
