#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jan  26 14:00:14 2023

@author: zhu18
"""
import numpy as np
from mpm.design import Method
import scipy


class GroundTruthClassifier(Method):
    """
    Ground Truth Classifier, 
    use the parameters from the generative model to perfrom 
    classifications.
    """
    def __init__(self, name=''):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            name (str): name of the method
        """
        # set method's name and paradigm
        super().__init__(name, paradigm='GT')

        self.output_directory = ''

    def fit(self, x, y, weight, beta_vec, **kwargs):
        """
            load in the parameters 
        """
        self.logger.info('Traning process is about to start.')


        self.ntasks = len(x)  # get number of tasks
        self.dimension = x[0].shape[1]  # get problem dimension

        for t in range(self.ntasks):
            x[t] = x[t].astype(np.float64)
            y[t] = y[t].astype(np.int32).ravel()
        self.m = weight
        self.phi = beta_vec


    def predict(self, x, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (numpy.array): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        yhat = [None]*len(x)
        for t in range(len(x)):
            yhat[t] = scipy.special.expit(np.dot(x[t], self.m[t, :]*self.phi))
            yhat[t] = np.around(yhat[t]).astype(np.int32)
        return yhat

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (string): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
