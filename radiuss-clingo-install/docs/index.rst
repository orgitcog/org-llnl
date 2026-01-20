.. ##
.. ## Copyright (c) 2022, Lawrence Livermore National Security, LLC and
.. ## other RADIUSS Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (MIT)
.. ##

######################
Radiuss Clingo Install
######################

RADIUSS Clingo Install is a sub-project from the RADIUSS initiative providing a
testing infrastructure to test that Clingo installs correctly on any LC
machines of interest. It does so automatically in GitLab and can track changes
in Spack.

LLNL's RADIUSS project (Rapid Application Development via an Institutional
Universal Software Stack) aims to broaden usage across LLNL and the open source
community of a set of libraries and tools used for HPC scientific application
development.


=========================
Background and Motivation
=========================

In this repo, we configure Gitlab CI to generate CI pipelines that build Spack
and Clingo in various configurations on various machines.

The intended use is to setup schedules to build Clingo for Spack on a regular
basis and check that this is not broken.

This can also be used to populate and update a Spack bootstrap store for a
frozen Spack version (for reproducibility / reliability).


.. toctree::
   :hidden:
   :caption: User Documentation

   sphinx/user_guide/index

.. toctree::
   :hidden:
   :caption: Developer Documentation

   sphinx/dev_guide/index
