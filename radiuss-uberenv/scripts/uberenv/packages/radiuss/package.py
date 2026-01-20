# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Radiuss(BundlePackage):
    """LLNL's RADIUSS project—Rapid Application Development via an
    Institutional Universal Software Stack—aims to broaden usage across LLNL
    and the open source community of a set of libraries and tools used for
    HPC scientific application development."""

    homepage = "https://software.llnl.gov/radiuss/"

    version('1.0.0')

    depends_on('mfem')
    depends_on('hypre')
    depends_on('sundials')
    depends_on('samrai')
    depends_on('xbraid')
    depends_on('umpire')
    depends_on('raja')
    depends_on('caliper')
    depends_on('py-maestrowf')
    depends_on('conduit')
    depends_on('ascent')
    depends_on('zfp')
    depends_on('scr')

    depends_on('hdf5@1.8.21')

    phases= ['hostconfig']

    def hostconfig(self, spec, prefix, py_site_pkgs_dir=None):
        pass
