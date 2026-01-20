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

import h5py
import numpy as np

class DataReader(object) :


    def read_files(self, path):
        h5_file = h5py.File(path+'.h5', 'r')
        self._symbols = self.__read_symbols(path+'.sym')   
        self._metrics = {}
        #This is a hack for the json module to output the np arrays
        self._metrics['app_time'] = list(np.array(h5_file['app_time']))
        self._metrics['mpi_time'] = list(np.array(h5_file['mpi_time'])/1000)
        ranks = len(self._metrics['app_time'])  
        self._metrics['total_ranks'] = ranks

        for rank in range(self._metrics['total_ranks']):
            rank_stats = {} 
            rank_stats['calls'] = {}
            rank_stats['comms'] = {}
            rank_stats['callsites'] = {}
            rank_stats['pattern'] = {}
            rank_stats['app_time'] = self._metrics['app_time'][rank]
            rank_stats['mpi_time'] = self._metrics['mpi_time'][rank]
            mpi_perct = float(rank_stats['mpi_time'])/rank_stats['app_time']*100

            rank_stats['mpi_time%'] =  mpi_perct
            rank_stats['app_time%'] = 100.0-rank_stats['mpi_time%']

	    self.__get_calls_data(rank_stats['calls'],np.array(h5_file['calls'])[rank]) 
            self.__get_callsites_data(rank_stats['callsites'],np.array(h5_file['callsites'])[rank]) 
            self.__get_pattern_data(rank_stats['pattern'],np.array(h5_file['pattern'])[rank]) 

            self._metrics[rank]= rank_stats

        #self._metrics['pattern'] = [[0 for j in range(ranks)] for i in range(ranks)]
        h5_file.close()
        return self._metrics

    def __read_symbols(self, path):
        symbols = {}
        with open(path,'r') as f:
            for line in f:
                tokens = line.split()
                symbols[int(tokens[0])] = ''.join(tokens[1:])  
        return symbols

    def __get_calls_data(self, rank_stats, dataset):
        # TODO: Pandas dataframe?
        for call in dataset:
            key = self._symbols[call[0]]
            rank_stats[key] = {}
            rank_stats[key]['#calls']   = call[1]
            rank_stats[key]['acc_time'] = call[2]
            rank_stats[key]['kbytes']   = call[3]

            rank_stats[key]['time_per_call']  = call[2]/call[1]
            rank_stats[key]['bytes_per_call'] = call[3]/call[1]

    def __get_callsites_data(self, rank_stats, dataset):
        # TODO: Pandas dataframe?
        for call in dataset:
            key = '%s %s'%(self._symbols[call[0]],self._symbols[call[1]]) #TODO MAP SYMBOLS
            rank_stats[key] = {}
            rank_stats[key]['#calls']   = call[2]
            rank_stats[key]['acc_time'] = call[3]
            rank_stats[key]['kbytes']   = call[4]

            rank_stats[key]['time_per_call']  = call[2]/call[1]
            rank_stats[key]['bytes_per_call'] = call[3]/call[1]

    def __get_pattern_data(self, rank_stats, dataset):
        for pat in dataset:
            dest = pat[0]
            if not dest in rank_stats:
                rank_stats[dest] = { 'bytes'  : pat[1],
                                     '#calls' : pat[2]}

            rank_stats[dest]['bytes']  += pat[1]
            rank_stats[dest]['#calls'] += pat[2]

         

