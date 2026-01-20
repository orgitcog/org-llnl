#!/usr/bin/env python3
#
# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import sys

sys.path.append(
    "/usr/gapps/spot/live/hatchet-venv/x86_64/lib/python3.9/site-packages/"
)  # <-- Python packages
sys.path.append("/usr/gapps/spot/live/hatchet/x86_64/")  # <-- Hatchet
sys.path.append("/usr/gapps/spot/live/thicket-playground-dev/")  # <-- Thicket

from glob import glob

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

import hatchet as ht
import thicket as th

#
