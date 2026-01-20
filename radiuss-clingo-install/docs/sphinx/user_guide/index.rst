.. ##
.. ## Copyright (c) 2022, Lawrence Livermore National Security, LLC and
.. ## other RADIUSS Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (MIT)
.. ##

##########
User Guide
##########

Projects using Spack have to setup Clingo, which Spack uses in its concretizer.
It is common, and recommended for projects to freeze Spack version for
reproducibility and reliability.  In the end, two projects may not use the same
Spack version, and syncing/updating can become a hassle.

In addition to that, using Spack in CI also requires to setup Clingo in a
reliable manner on any machine of interest.

We created this repo to help testing the Clingo install with any Spack version
and automating the installation on several machines. This documentation will
demonstrate how to achieve both.

.. toctree::
   :maxdepth: 2

   getting_started
   project_configuration
   ci_configuration
