.. trata documentation master file, created by
   sphinx-quickstart on Thu Oct 11 15:43:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the LLNL Trata Documentation!
====================================================

Trata
=====

Trata is used to generate sample points in order to explore a parameter space.
For instance, if a simulation takes two inputs, x and y, and you want to run a set of simulations with x-values
between 5 and 20 and y-values between 0.1 and 1000, the sampling component can generate sample points (which in
this case means (x,y) pairs) for you. You can specify how many total sample points you want, and how you want
them to be chosen--the sampling component offers a large number of different sampling strategies.
If, on the other hand, you already have sample points you wish to use,
the component can simply read them in from a file.


The ``trata`` package works with Python 3 and is made up of 4 modules:
   * ``sampler``
   * ``composite_samples``
   * ``adaptive_sampler``
   * ``kosh_sampler``

Installation:

::

         pip install trata
 
Demo usage:

.. code:: python

         from trata import composite_samples

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorial
   sampler
   composite_sampling
   adaptive_sampling
   kosh_sampling

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
