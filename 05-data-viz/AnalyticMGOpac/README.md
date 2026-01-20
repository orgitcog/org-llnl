# AnalyticMGOpac: An analytic opacity integrator for multigroup calculations

This C++14 code numerically integrates the opacity formulas for the test problem described
in
* Thomas A. Brunner, "A Family of Multi-Dimensional Thermal Radiative Transfer Test Problems",
  2023, LLNL-TR-858450-REV-1, [DOI 10.2172/2280904]( https://doi.org/10.2172/2280904)

Please see that document for a full description.

## Getting started

This code mainly consists of two implementation files for the opacity and integrator.
The there is a test file that will output sample data used in LLNL-TR-858450 and a
Python file to make the plots.


* `AnalyticEdgeOpacity.cc` and `AnalyticEdgeOpacity.hh`: C++14 code to compute the frequency
  dependent opacity for a given set of parameters
* `MultiGroupIntegrator.cc` and `MultiGroupIntegrator.hh`: C++14 code to integrate a given
  instance of `AnalyticEdgeOpacity` for a particular set of group bounds.
* `opacTest.cc`: An optional C++20 test driver.  This language version was chosen to help
  reduce some boilerplate in order to help make it clearer how to use the library.
* `Makefile`: A simple makefile to build the test driver, assuming you have `clang++` installed.
* `plotOpacs.py`: A [matplotlib](https://matplotlib.org/) based script to plot the results from
  `opacTest.cc` which can be seen in LLNL-TR-858450.

### Prerequisites
The opacity integrator only requires C++14.  The test file `opacTest.cc` which documents
the usage of the integrator relies on the [`{fmt}`](https://fmt.dev) library.

### Installing
A simple Makefile is included for this project.  You can edit it to fit your compiler.
```
git clone git@github.com:fmtlib/fmt.git
make
```

### Running the tests
After building the test code, you can generate opacity plots by running
```
% ./opacTest
% python3 plotOpacs.py
```

## Release
The code of this site is released under the MIT License. For more details, see the
[LICENSE](LICENSE) file.  All new contributions must be made under this license.

See the [NOTICE](NOTICE) file for other important information.

LLNL-CODE-858992
