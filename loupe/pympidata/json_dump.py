##############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# Written by Emilio Castillo <ecastill@bsc.es>.
# LLNL-CODE-745958. All rights reserved.
#
# This file is part of Loupe. For details, see:
# https://github.com/LLNL/loupe
# Please also read the LICENSE file for the MIT License notice.
##############################################################################

from pympidata import dataset
from pympidata import parser
import sys
import json

dr = parser.DataReader() 
ds = dataset.Dataset(dr.read_files(sys.argv[1]))
ds.averages()

print json.dumps(ds.metrics(),indent=4)
