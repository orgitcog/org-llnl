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

from collections import defaultdict
import numpy as np

class Dataset(object):
  
    def __init__(self, metrics):
        self._metrics = metrics

    def averages(self):
        self._metrics['avg'] = {}
        self._metrics['avg']['calls'] = self.__average_section('calls')
        self._metrics['avg']['callsites'] = self.__average_section('callsites')
        self.__average_header()
 
    def totals(self):
        self._metrics['total'] = {}
        self._metrics['total']['calls'] = self.__total_section('calls')
        self._metrics['total']['callsites'] = self.__total_section('callsites')

    def __average_header(self):
        self._metrics['avg']['app_time']=np.average(self._metrics['app_time'])
        self._metrics['avg']['mpi_time']=np.average(self._metrics['mpi_time'])

        self._metrics['avg']['mpi_time%'] = float(self._metrics['avg']['mpi_time'])/self._metrics['avg']['app_time']*100 
        self._metrics['avg']['app_time%'] = 100.0-self._metrics['avg']['mpi_time%'] 
         
    def __average_section(self,section):
        calls = {}
        fields = ('#calls','acc_time','kbytes','time_per_call','bytes_per_call')
        # Look for all the calls in the table
        #Some process might not be calling some mpi ops
        for rank in range(self._metrics['total_ranks']):
            for call in self._metrics[rank][section]:
                if not call in calls:
                    calls[call] = {}
                    for field in fields:
                         calls[call][field] = 0.0
                    calls[call]['samples'] = 0.0

        for rank in range(self._metrics['total_ranks']):
            for call in self._metrics[rank][section]:
                    for field in fields:
                        x_i = self._metrics[rank][section][call][field]
                        u_i = calls[call][field]
                        n = calls[call]['samples']+1
                        calls[call][field] = u_i+1/n*(x_i-u_i)
                    calls[call]['samples']+=1
        return calls

    def __total_section(self,section):
        calls = {}
        fields = ('#calls','acc_time','kbytes','time_per_call','bytes_per_call')
        # Look for all the calls in the table
        #Some process might not be calling some mpi ops
        for rank in range(self._metrics['total_ranks']):
            for call in self._metrics[rank][section]:
                if not call in calls:
                    calls[call] = {}
                    for field in fields:
                         calls[call][field] = 0.0

        for rank in range(self._metrics['total_ranks']):
            for call in self._metrics[rank][section]:
                    for field in fields:
                        calls[call][field] += self._metrics[rank][section][call][field]
        return calls

    #param can be bytes or #calls
    def heat_map(self, param):
        nranks = self._metrics['total_ranks']
        self._metrics['pattern_'+param] = [[0 for j in range(nranks)] for i in range(nranks)]
        for rank in range(self._metrics['total_ranks']):
            # USE only comm world, we cant translate other comms yet
            for dest in self._metrics[rank]['pattern']:
                self._metrics['pattern_'+param][rank][dest]+=self._metrics[rank]['pattern'][dest][param]
         
    def heat_map_freq(self):
        pass

    
    def metrics(self):
        return self._metrics 
