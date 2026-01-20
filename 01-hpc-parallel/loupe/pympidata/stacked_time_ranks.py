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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dr = parser.DataReader() 
ds = dataset.Dataset(dr.read_files(sys.argv[1]))
ds.averages()
metrics = ds.metrics()

N = metrics['total_ranks']
app_time = [metrics[i]['app_time%'] for i in range(N)] + [metrics['avg']['app_time%']]
mpi_time = [metrics[i]['mpi_time%'] for i in range(N)] + [metrics['avg']['mpi_time%']]
N += 1
#menStd = (2, 3, 4, 1, 2)
#womenStd = (3, 5, 2, 3, 3)
ind = range(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, app_time, width, color='#d62728')#, yerr=menStd)
p2 = plt.bar(ind, mpi_time, width,
             bottom=app_time)#, yerr=womenStd)

plt.ylabel('Time (%)')
plt.title('Per Rank Time Breakdown')
plt.xticks(ind, map(str,range(N-1))+['AVG'])
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('App Time', 'MPI Time'))

plt.savefig('per_rank_time.png', bbox_inches='tight')

