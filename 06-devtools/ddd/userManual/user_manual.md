Delaunay Density Diagnostic User Manual
----------------

   Version 2.1, September 2024.

   This code implements algorithms described in:\
   **Algorithm XXXX: The Delaunay Density Diagnostic**\
   under review at ACM Transactions on Mathematical Software\
   original title: ``Data-driven geometric scale detection via Delaunay interpolation''
   Andrew Gillette and Eugene Kur, 2022 \
   https://arxiv.org/abs/2203.05685

Authors
----------------
The Delaunay density diagnostic code was created by Andrew Gillette, gillette7@llnl.gov, with input from Eugene Kur, kur1@llnl.gov.


Overview
-----------------

The Delaunay Density Diagnostic code is a tool to assess a function with *d* numerical inputs and a single numerical output, for *d* in the range 2-10.  
The user provides either a static dataset of input-output pairs or access to a function that can be evaluated on inputs within a user-provided domain. The diagnostic then returns "rate estimates" for various degrees sampling densities and saves a figure.

High level takeaway: 
- If the sampling density *is* sufficient to recover geometric features present in the function, the rate will be close to the *dotted green line* in the figure.
- If the sampling density is **not** suffficinet to distinguish the geometric features from random noise, the rate will be close to the *dashed red line* in the figure.


<!-- ![Diagram of the process](../ddd-figure-griewank-repo.png) -->

<img src="../ddd-figure-griewank-repo.png" alt="Alt text" width="300" height="500">
<img src="../ddd-figure-static-repo.png" alt="Alt text" width="300" height="500">

The example figures shown above are also included with the repository.   The code in the repository can be used to generate these exact figures.

Follow the steps described in Usage below to generate the figures.  Then run
   ~~~~
   python delaunay_density_diagnostic.py --help
   ~~~~
to read about the command line options.  The paper linked at the top of this user manual describes the algorithm and case studies in detail.



Requirements
-----------------

python>=3.7

numpy>=1.21.5 

pandas>=1.3.5

matplotlib>=3.5.3



Usage
----------------

1. Activate a python environment that includes the packages listed in the REQUIREMENTS.txt file.  

2. Ensure that the `gfortran` compiler is installed.

3. Run the driver script for the Griewank and/or static data examples:
   ~~~~
   python run_ddd_griewank.py
   ~~~~
   The above script will run a total of 100 trials of the `delaunay_density_diagnostic.py` script,
      using data from the 2D Griewank function.  The results are saved as `.csv` files.  Then the script `generate_ddd_figures.py` is called to generate a `.png` figure called `ddd-figure-griewank.png`.  The figure should match the file `ddd-figure-griewank-repo.png` that is contained in the repository.  More details can be found in the header of  `run_ddd_griewank.py`.

   A typical run time for a single trial is a few seconds, so the whole script should complete
      in 5-10 minutes.

   ~~~~
   python run_ddd_static.py
   ~~~~
   The above script will run a total of 100 trials of the `delaunay_density_diagnostic.py` script,
      using data from the static topography dataset described in the paper (and stored in the subfolder `staticdata/`).   The results are saved as `.csv` files.  Then the script `generate_ddd_figures.py` is called to generate a `.png` figure called `ddd-figure-static.png`.  The figure should match the file `ddd-figure-static-repo.png` that is contained in the repository.  More details can be found in the header of  `run_ddd_static.py`.


4. If the figures generates correctly, run
   ~~~~
   python delaunay_density_diagnostic.py --help
   ~~~~
   to see the command line options that can be added to the driver scripts for
   user-specified experiments.

Debugging notes
----------------

The package includes source files in Fortran that impmlement a version of TOMS Algorithm 1012:
DELAUNAYSPARSE.  This version that has been updated from the original submission to more easily allow python wrapping.  Running the script `delaunay_density_diagnostic.py` will compile the relevant files using `gfortran`.  

During compiling, this type of warning may occur:
~~~~
Warning: Rank mismatch between actual argument at (1) and actual argument at (2)
~~~~
This warning is issued by the `slatec` library that is included with the DELAUNAYSPARSE source code and is not easily suppressed.  However, this warning is only due to a change in Fortran conventions since the original publication of TOMS 1012 and does not cause any issues in regards to the results.



License
----------------

Delaunay density diagnostic is distributed under the terms of the MIT license.

All new contributions must be made under the MIT license.

See [LICENSE](https://github.com/ddd/blob/main/LICENSE) and
[NOTICE](https://github.com/ddd/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE-833036
