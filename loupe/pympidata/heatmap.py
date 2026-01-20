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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


dr = parser.DataReader() 
ds = dataset.Dataset(dr.read_files(sys.argv[1]))
ds.averages()
ds.heat_map('bytes')
pattern = ds._metrics['pattern_bytes']
#plt.imshow(pattern, cmap='Reds', interpolation='nearest')
fig, axis = plt.subplots()
axis.invert_yaxis()
heatmap = axis.pcolormesh(pattern, cmap='Blues')
plt.colorbar(heatmap)
plt.savefig('pattern_bytes.png', bbox_inches='tight')
ds.heat_map('#calls')
pattern = ds._metrics['pattern_#calls']
fig, axis = plt.subplots()
axis.invert_yaxis()
heatmap = axis.pcolormesh(pattern, cmap='Blues')
plt.colorbar(heatmap)
#print pattern
#plt.imshow(pattern, cmap='Reds', interpolation='nearest')
plt.savefig('pattern_calls.png', bbox_inches='tight')
