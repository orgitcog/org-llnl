Welcome to slewpy's documentation!
==================================

slewpy is a Python package that allows the simulation of the science operations of an astrophysics space satellite mission. 
slewpy allows users to specify an astronomical target list with observing priorities and a satellite configuration 
(i.e., orbit and various satellite parameters). Taking these inputs, slewpy can be used to run a time-resolved 
simulation of an astrophysics mission and outputs simulated target observations given constraints 
such as satellite slewing rates between targets, observing time on a given target, and Sun-, Earth-, 
and Moon-limb observing constraints. slewpy provides the tools to test astrophysics space satellite mission 
designs against observing requirements for a given science case.


slewpy is being actively
developed on `GitHub <https://github.com/LLNL/slewpy>`_.

.. figure:: slewpy_sim.gif

   Example visualization of a slewpy satellite simulation. The figure shows astrophysical transients appearing and disappearing on the celestial sphere (colored circles). Every        time an unfilled circle appears around a transient, it indicates that the transient was observed by a satellite in a 600 km altitude sun-synchronous polar orbit. Shaded regions     show various observing constraints: Moon, Sun, and Earth-limb constraints, as well as the high source density region of the Galactic plane. Transients are not observed when they    are in one of the constrained regions.

A good place to get started is with the installation guide, getting started page and
the the tutorial examples.


.. note:: Finding your way around

   A good place to get started is with the installation guide, getting started page and
   the the tutorial examples.

   If you are here to develop on popclass, please head over to the contributing guide.

   If you have issues with the software or need support please open a `github issue <https://github.com/LLNL/slewpy/issues>`_.

   If you are here to develop on slewpy, please head over to the contributing guide.

Contents
--------

.. toctree::
   :titlesonly:
   :maxdepth: 1

   installation
   getting_started.ipynb
   background
   tutorials
   library
   acknowledgements
   contributing
   developer
   changelog
   references
   api
