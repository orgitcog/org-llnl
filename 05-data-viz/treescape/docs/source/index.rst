..
   Copyright 2025 Lawrence Livermore National Security, LLC and other
   TreeScape Project Developers. See the top-level LICENSE file for details.

   SPDX-License-Identifier: MIT

.. TreeScape documentation master file, created by
   sphinx-quickstart on Wed Mar 13 14:02:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TreeScape's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

TreeScape shows you line charts and flamegraphs of the data you provide.

Example Inputs
~~~~~~~~~~~~~~

You can select runs by placing them in a directory and creating a th_ens and profiles object.
This is an example of how to build a th_ens object.   The th_ens object is then
used to populate the TreeScapeModel with runs.

.. code-block:: python

   import sys
   import platform
   import datetime as dt

   input_deploy_dir_str = "/usr/gapps/spot/dev/"
   machine = platform.uname().machine

   sys.path.append("/Users/aschwanden1/thicket")

   import re

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from IPython.display import display
   from IPython.display import HTML

   import hatchet as ht
   import thicket as tt

   # get a list of all cali files in subdirectory - recursively
   from glob import glob
   import os

   PATH = '/Users/aschwanden1/lulesh_gen/100/'
   profiles = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.cali'))]

   #  this contains some metadata we need.
   #  also contains the tree data.
   th_ens = tt.Thicket.from_caliperreader(profiles)



Profiles
~~~~~~~~~
Profiles is a list of \*.cali input files thicket requires for the caliperreader.
It looks like this:

.. code-block:: python

   example = [
      '/Users/aschwanden1/lulesh_gen/100/49.cali',
      '/Users/aschwanden1/lulesh_gen/100/73.cali',
      '/Users/aschwanden1/lulesh_gen/100/24.cali',
      '/Users/aschwanden1/lulesh_gen/100/32.cali'
      ]



Usage
~~~~~~~~
In a Jupyter notebook use the following python code.  Above, you created two variables needed by graph renderer
profiles and th_ens.

.. code-block:: python

   import sys

   sys.path.append("python")
   sys.path.append("viz")

   from StackedLine import StackedLine
   from ThicketWrapper import ThicketWrapper, get_th_ens
   from TreeScapeModel import TreeScapeModel

   th_ens, profiles = get_th_ens()

   model = TreeScapeModel(th_ens, profiles)
   model.setVisibleAnnotations(['main'])
   model.setXAxis("launchdate")

   sl = StackedLine()
   sl.render("Line", model )


Tree
~~~~~~
.. toctree::
   :maxdepth: 2

   TreeScapeModel
   StackedLine




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
