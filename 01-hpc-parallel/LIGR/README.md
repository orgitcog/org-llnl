LIGR (Linear Grains) Version: 1.0

LIGR (Linear Grains) is a Python package that implements a linear (Fourier mode) analysis of a shock propagating through density perturbations. It has two primary capabilities. First, it calculates the shock front distortion and post-shock flow variables (density, velocity, pressure) for single Fourier mode pairs in two dimensions (a pair of x and y modes). Second, it calculates these same quantities for a grains-like pre-shock density perturbation, which consists of a sum of many Fourier modes. This grains-like perturbation is a periodic grid of rectangles at a specifiable density and size (the grains), separated by regions of a different specifiable size and density (interstitial spaces between the grains).
The single-mode analysis is based on the work of Velikovich et al., Phys. Plasmas 14 072706 (2007).

Usage
----------------
The primary functionality is contained in the LinearAnalysis directory. This contains the single mode and grains-like implementations described above.

Various visualization and analysis scripts, which utilize these core functionalities, are contained in the SingleModeVisualization and GrainsExamples directories.


Authors
----------------
LIGR was developed by Grace J Li (li85@llnl.gov)


Acknowledgements
----------------
Seth Davidovits (davidovits1@llnl.gov) supported the development of LIGR and the various example cases.


License
----------------

LIGR is distributed under the terms of the BSD 3-Clause License.

All new contributions must be made under the same license.

LLNL-CODE-832565
