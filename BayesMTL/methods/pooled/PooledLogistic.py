#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 08:59:14 2018

@author: goncalves1
"""
import os

import numpy as np
import pandas as pd
from sklearn import linear_model
from mpm.design import Method

import pickle

class PooledLogisticClassifier(Method):
    """
    Implement an pooled Logistic Classifier: train one single logistic
    classifier for all tasks.

    """
    def __init__(self, alpha_l1=0.5, name='Pooled-LC'):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            alpha_1 (float): l1 penalization hyper-parameter
        """
        # set method's name and paradigm
        super().__init__(name, paradigm='Pooled')

        self.model = linear_model.LogisticRegression(penalty='l1', C=alpha_l1,solver='saga')
        self.alpha_l1 = alpha_l1
        self.output_directory = ''

    def fit(self, x, y,column_names=None,**kwargs):
        """
        Train model on given data x and y.
        Args:
            x (pandas.DataFrame): input data matrix (covariates).
            y (pandas.DataFrame): label vector (outcome).
        Returns:
            None.
        """
        if self.mode == 'train':
            self.logger.info('Traning process is about to start.')
            self.feature_names = column_names

        self.ntasks = len(x)  # get number of tasks
        self.dimension = x[0].shape[1]  # get problem dimension

        for t in range(self.ntasks):
            x[t] = x[t].astype(np.float64)
            y[t] = y[t].astype(np.int32).ravel()

        xpooled = np.row_stack(x)
        ypooled = np.concatenate(y)

        # Train the model using the training sets
        self.model.fit(xpooled, ypooled)
        if self.mode == 'train':
            self.logger.info('Training process finalized.')
            fname = os.path.join(self.output_directory, '%s.mdl' % self.__str__())
            # weights = self.model.coef_[0]
            weights = np.concatenate((np.concatenate(self.model.coef_),self.model.intercept_))
            with open(fname, 'wb') as fh:
                pickle.dump([weights], fh)

    def predict(self, x, prob=False, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (numpy.array): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        yhat = [None]*len(x)
        if prob:
            for t in range(len(x)):
                x[t] = x[t].astype(np.float64)
                yhat[t] = self.model.predict_proba(x[t])[:,1]
        else:
            for t in range(len(x)):
                x[t] = x[t].astype(np.float64)
                yhat[t] = np.around(self.model.predict(x[t])).astype(int)
        return yhat

    def feature_importance(self):
        """ Compute/Extract feature importance from the trained model
            and store it as a pd.Series object.
        Args:
            None
        Returns:
            pd.Series with feature importance values.
        """
        importance = np.exp(self.model.coef_[0]) * np.sign(self.model.coef_[0])
        df = pd.Series(importance, index=self.feature_names)
        return df

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.alpha_l1 = params['alpha_l1']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'alpha_l1': self.alpha_l1}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        alphas_l1 = np.logspace(-6, 3, 50)
        for alpha_l1 in alphas_l1:
            yield {'alpha_l1': alpha_l1}

    def load_params(self, params):
        """
        load previously trained parameters to be used in the execution.
        Args:
            params (dict): dict with parameter values.
        """
        self.model.coef_ = np.array([params[:-1]])
        self.model.intercept_ = params[-1]
        return

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (string): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
