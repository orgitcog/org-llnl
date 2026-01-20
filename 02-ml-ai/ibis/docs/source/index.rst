Welcome to LLNL's IBIS Documentation!
==============================================

Interactive Bayesian Inference and Sensitivity, or IBIS, is a Python-based scientific tool for analyzing concurrent UQ simulations on high-performance computers.
Using a simple, non-intrusive interface to simulation models, it provides the following capabilities:

* generating parameter studies
* generating one-at-a-time parameter variation studies
* sampling high dimensional uncertainty spaces
* constructing surrogate models
* performing sensitivity studies
* performing statistical inferences
* estimating parameter values and probability distributions

IBIS
====

IBIS has been used for simulations in the domains of Inertial Confinement Fusion, National Ignition Facility experiments, climate, as well as other programs and projects. 
The `ibis` package is designed to be used after a number of simulations have run to completion. 
This package is used to predict the results of future simulation runs and to assess the sensitivity each
input parameter has on the output.

The ``ibis`` package works with Python 3 and is made up of 7 modules:
 * ``filter``
 * ``likelihoods``
 * ``mcmc``
 * ``mcmc_diagnostics``
 * ``sensitivity``
 * ``pce_model``
 * ``plots``

Installation:

::

        pip install llnl-ibis

Demo usage :

.. code:: python

        from ibis import mcmc, pce_model

.. toctree::
   :maxdepth: 2
   :hidden:
   
   mcmc_tutorial
   sensitivity_tutorial
   examples
   plots
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`