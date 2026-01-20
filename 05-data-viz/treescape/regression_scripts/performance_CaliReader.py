#!/usr/bin/env python3
#
# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

# cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test_plus_80"
# cali_file_loc = "/Users/aschwanden1/datasets/newdemo/mpi"
# cali_file_loc = "/g/g0/pascal/datasets/newdemo/mpi"

#  this one should be available for everyone
cali_file_loc = "/usr/gapps/spot/datasets/newdemo/mpi/"

xaxis = "launchday"
metadata_key = "test"
processes_for_parallel_read = 15
initial_regions = ["main"]

import sys

sys.path.append("/Users/aschwanden1/min-venv-local/lib/python3.9/site-packages")
sys.path.append("/")

import treescape as tr

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    inclusive_strs = [
        "min#inclusive#sum#time.duration",
        "max#inclusive#sum#time.duration",
        "avg#inclusive#sum#time.duration",
        "sum#inclusive#sum#time.duration",
    ]

    caliReader = tr.CaliReader(
        cali_file_loc, processes_for_parallel_read, inclusive_strs
    )
    tsm = tr.TreeScapeModel(caliReader)
    # Always be sure to sort your data into some reasonable way.
    alltests = sorted(tsm, key=lambda x: x.metadata[xaxis])
