Delaunay Density Diagnostic
----------------
   Version 2.1, September 2024.

   This code implements algorithms described in:\
   **Algorithm XXXX: The Delaunay Density Diagnostic**\
   under review at ACM Transactions on Mathematical Software\
   original title: ``Data-driven geometric scale detection via Delaunay interpolation''
   Andrew Gillette and Eugene Kur, 2022 \
   https://arxiv.org/abs/2203.05685


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

Authors
----------------
The Delaunay density diagnostic code was created by Andrew Gillette, gillette7@llnl.gov, with input from Eugene Kur, kur1@llnl.gov.

Citation information
----------------
If you are referring to this code in a publication, please cite the following paper:

Andrew Gillette and Eugene Kur.  *Data-driven geometric scale detection via Delaunay interpolation*.  Submitted.  2022.  LLNL-JRNL-832470.

~~~~
@article{GK2022,
  author = Gillette, Andrew and Kur, Eugene},
  title = {Data-driven geometric scale detection via Delaunay interpolation},
  journal = {Submitted. Preprint at arXiv:2203.05685},
  year = {2022},
}
~~~~

If you wish to cite the code specifically, please use:

~~~~
@misc{ doecode_72093,
title = {Delaunay density diagnostic},
author = {Gillette, Andrew K.},
url = {https://doi.org/10.11578/dc.20220324.3},
howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20220324.3}},
year = {2022},
month = {mar}
}
~~~~

The DOI for this repository is:  https://doi.org/10.11578/dc.20220324.3


License
----------------

Delaunay density diagnostic is distributed under the terms of the MIT license.

All new contributions must be made under the MIT license.

See [LICENSE](https://github.com/ddd/blob/main/LICENSE) and
[NOTICE](https://github.com/ddd/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE-833036
