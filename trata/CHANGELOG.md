# Release Notes

* [1.2.1](#121-release)
* [1.2.0](#120-release)
* [1.1.1](#111-release)
* [1.1.0](#110-release)
* [1.0.3](#103-release)
* [1.0.2](#102-release)
* [1.0.1](#101-release)


## 1.2.1 Release

### Description

### Changed

The `QuasiRandomNumberSampler` has been updated to use scipy's implementation for Sobol and Halton sequences.

Benefits include:
 * Continue the sequence from a certain point with the `sequence_offset` option.
 * A `scramble` option for better randomization
 * The directional numbers for generating a Sobol sequence are updated to allow for higher dimensions. [Joe and Kuo 2008](https://epubs.siam.org/doi/10.1137/070709359) This gives different but still valid results compared to the previous version.


## 1.2.0 Release

### Description

### Added

* BestCandidateSampler: A new option in `trata.adaptive_sampler` that chooses new samples that add most to diversity of existing samples.
* All adaptive samples can be passed a random seed for reproducability.
* The sampling documentation notebook has been updated to include `adaptive_sampler` examples.
* The `ProbabilityDensityFunctionSampler` will automatically scale values based on ranges passed with the "box" parameter.

### Changed

* Adaptive samplers use the same variable names as the rest of the modules. Only "values" for the inputs/X, and "output" for response/Y.

### Bug fixes

* Updated some indexing and variable assignment methods to be compatible with numpy > 2.0
* One-at-a-time sampler has input validation and avoids duplicating default values.
* Other fixes and updates to the testing suite


## 1.1.1 Release

### Description

This release includes a minor bug fix.

### Bug fixes

* In the `sampler` module, checking for empty lists was fixed.

## 1.1.0 Release

### Description

This release introduces a new kosh_sampler module

### New in this release

* The kosh_sampler module wraps the adaptive sampling functions in a Kosh operator. Users with existing Kosh datasets can easily find the next best set of samples based on their model's error or sensitivity. 
* A Sobol index sampler has been added to the sampler module. It creates samples to be used in
IBIS's Sobol indices function in the sensitivity module.

## 1.0.3 Release

This is a minor release with a few bug fixes.

### Bug fixes
* fixed adaptive sampling tests to be more robust to numpy versions.
* fixed hanging problem in composite samples.
* A few other minor changes related to testing

## 1.0.2 Release

### Description

This release is a minor release with a few bux fixes and new features. We encourage users to upgrade.

### New in this release

* Added `**kwargs` to Morris sampler to catch extra arguments.

### Improvements
* Added code of conduct contributing documents


### Bug fixes
* No bug fixes


## 1.0.1 Release

### Description

This release is a minor release with a few bux fixes and new features. We encourage users to upgrade.
