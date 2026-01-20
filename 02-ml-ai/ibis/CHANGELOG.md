# Release Notes

* [1.2.0](#120-release)
* [1.1.1](#111-release)
* [1.1.0](#110-release)

## 1.2.0 Release

### Description

This release features new plots and some bug fixes.

### New in this release

Sensitivity methods new score and rank plots

* One-at-a-time: calculates each parameters individual effect shows the magnitude in a score plot, and compares features in a rank plot. `ibis.sensitivity.oat_score_plot` and `ibis.sensitivity.oat_rank_plot`
* Sobol Indices: Calculates contributions of individual parameters and their interactions. The score plot shows first order or total order sensitivity indices. The rank plot compares them across the outputs. `ibis.sensitivity.sobol_score_plot` and `ibis.sensitivity.sobol_rank_plot`
* Morris method: Calculates elementary effects for each path through the parameter space, and calculates statistics mu* and sigma across all the trajectories to understand main effects and interactions. Score plot can show mu* and sigma, and rank plot just compares parameters based on mu* or sigma. `ibis.sensitivity.morris_score_plot` and `ibis.sensitivity.morris_rank_plot`
* These plots are also available as Kosh operators from `ibis.kosh_operators.KoshSensitivityPlots`

### Bug fixes

* Updated some indexing and variable assignment methods to be compatible with numpy > 2.0
* Other fixes and updates to the testing suite

## 1.1.1 Release

### Description

This release includes some bug fixes and an update to the basic descriptions.

### Improvements

* Updated descriptions for the main modules
* More explanation for discrepancy MCMC sampling

### Bug fixes

* KoshMCMC function only has experimental data coming from a Kosh store.
* Fixed loop to allow for any number of quantity of interest and experiments in the KoshMCMC function.
* Allow for the case of only one QOI in the sensitivity plot, variance_network_plot, and rank plot.

## 1.1.0 Release

This release introduces some new features

### New in this release

Added Kosh operators to Ibis to be able to use
IBIS UQ and sensitivity methods with Kosh datasets.

* KoshMCMC
* KoshOneAtATimeEffects
* KoshSensitivityPlots

A sobol_indices function has been added to the sensitivity module. It's meant
to be used with the SobolIndexSampler in the Trata sampler module or similar.