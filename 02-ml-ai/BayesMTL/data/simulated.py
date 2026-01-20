#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulated dataset class.

@author: zhu18
"""
import os
import numpy as np

from abc import ABCMeta, abstractmethod

data_subtypes = ["Ultra sparse","Sparse","Dense"]


class simulated(object):
    """ Simulated dataset.
        This class stores all information about the simulated dataset
    """
    __metaclass__ = ABCMeta

    def __init__(self, dirpath, data_subtype, r_i=0,
                 problem_type='classification', min_samples=0):

        assert data_subtype in data_subtypes, \
                     'Unknown data type {}.'.format(data_subtype)

        self.dirpath = dirpath
        self.problem_type = problem_type
        self.min_samples = min_samples
        self.r_i = r_i

        self.data_split = {'train': {'x': list(),
                        'y': list()},
                'test': {'x': list(),
                        'y': list()},
            }

        # change to your appropriate file locations
        input_dir = os.path.join(self.dirpath, data_subtype)

        self.data_split['train']['x'] = list(np.load(input_dir+"/data_x_train{}.npy".format(r_i),allow_pickle=True))
        self.data_split['train']['y'] = list(np.load(input_dir+"/data_y_train{}.npy".format(r_i),allow_pickle=True))

        self.data_split['test']['x'] = list(np.load(input_dir+"/data_x_test{}.npy".format(r_i),allow_pickle=True))
        self.data_split['test']['y'] = list(np.load(input_dir+"/data_y_test{}.npy".format(r_i),allow_pickle=True))

        self.data_split["weight"] = np.load(input_dir+"/weight{}.npy".format(r_i),allow_pickle=True)
        self.data_split["beta_vec"] = np.load(input_dir+"/beta_vec{}.npy".format(r_i),allow_pickle=True)