
Quantile Kriging is a method to model the uncertainty of a stochastic simulation by modelling both the overall simulation response and the output distribution at each sample point.  The output distribution is characterized by dividing it into quantiles, where the division of each quantile is determined by kriging.  
This library is our re-implementation of Quantile Kriging as described by Matthew Plumlee & Rui Tuo in their 2014 paper "Building Accurate Emulators for Stochastic Simulations via Quantile Kriging."  With computational savings when dealing with replication from the recent paper "Practical heteroskedastic Gaussian process modeling for large simulation experiments " by Binois, M., Gramacy, R., and Ludovski, M. it is now possible to apply Quantile Kriging to a wider class of problems.  In addition to fitting the model, other useful tools are provided such as the ability to automatically perform leave-one-out cross validation.

Getting Started
----------------
Quantkriging can be downloaded and installed inside of RStudio by using the devtools package.  If you do not have the devtools package run:
> install.package("devtools")

once devtools is installed, run:
> library(devtools)
> install_github("LLNL/quantkriging")


Documentation
----------------

Documentation is included in the package.  It can be viewed in RStudio with the command: help(quantkriging)

Contributing
------------------------
Just send us a [pull request](https://help.github.com/articles/using-pull-requests/). 

Authors
----------------

This implementation of quantile kriging was developed by Kevin Quinlan, quinlan5@llnl.gov.


Licenses
----------------

quantkriging is distributed under the terms of the MIT license

SPDX-License-Identifier: MIT

LLNL-CODE-796243

Copyright (c) 2019 Lawrence Livermore National Security

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Notice
----------------

Title: Quantkriging (Quantile Kriging R Package) , Version: 0.1.0
Author(s) Kevin R. Quinlan, James R. Leek

This work was produced under the auspices of the U.S. Department of
Energy by Lawrence Livermore National Laboratory under Contract
DE-AC52-07NA27344.

This work was prepared as an account of work sponsored by an agency of
the United States Government. Neither the United States Government nor
Lawrence Livermore National Security, LLC, nor any of their employees
makes any warranty, expressed or implied, or assumes any legal liability
or responsibility for the accuracy, completeness, or usefulness of any
information, apparatus, product, or process disclosed, or represents that
its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service
by trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the
United States Government or Lawrence Livermore National Security, LLC.

The views and opinions of authors expressed herein do not necessarily
state or reflect those of the United States Government or Lawrence
Livermore National Security, LLC, and shall not be used for advertising
or product endorsement purposes.
