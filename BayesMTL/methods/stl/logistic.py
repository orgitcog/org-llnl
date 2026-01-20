#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:42:03 2018

@author: goncalves1
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
from mpm.design import Method


# suppressing the convergence warning for debug mode
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)



class LogisticClassifier(Method):
    """
    Implement an Logistic (Regression) Classifier model (no-regularized).

    Attributes:
        batch_size (int): size of the mini-batch.
        nb_epochs (int): number of epochs for training
    """
    def __init__(self, penalty='l2', C=1.0,l1_ratio=1.0, name='LC'):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            batch_size (int): mini-batch size.
            nb_epochs (int): number of training epochs.
        """
        # set method's name and paradigm
        super().__init__(name, 'STL')
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        if self.penalty is None:
            self.model = linear_model.LogisticRegression(penalty='none',class_weight="balanced",max_iter=3000)
        elif self.penalty == 'elasticnet':
            self.model = linear_model.LogisticRegression(penalty=penalty, class_weight="balanced",C=C, l1_ratio=self.l1_ratio, max_iter=3000,solver='saga')
        else:
            self.model = linear_model.LogisticRegression(penalty=penalty, class_weight="balanced",C=C, max_iter=3000,solver='saga')
        self.model.class_weight = None
        self.feature_names = None
        self.output_directory = ''

    def fit(self, x, y, **kwargs):
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
            self.feature_names = kwargs['column_names']
        # convert to numpy array
        x = x.astype(np.float64)
        y = y.astype(np.int32).ravel()
        # Train the model using the training sets
        self.model.fit(x, y)

        if self.mode == 'train':
            self.logger.info('Training process finalized.')

    def predict(self, x, prob=False, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (numpy.array): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        # convert to numpy array
        x = x.astype(np.float64)
        # map output prediction into 0 or 1
        if prob:
            return self.model.predict_proba(x)[:,1]
        else:
            return np.around(self.model.predict(x)).astype(int)

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
        self.penalty = params['penalty']
        self.C = params['C']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'penalty':self.penalty,'C': self.C}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        Cs = np.logspace(-6, 3, 50)
        for C in Cs:
            yield {'C': C}

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

    def return_weight(self):
        """ Set output folder path.
        Args:
            output_dir (string): path to output directory.
        """
        return np.concatenate((np.concatenate(self.model.coef_),self.model.intercept_))
    def set_mode(self, mode):
        self.mode = mode
        """
            Depends on the mode initialize model class with
            different max_iter parameters
        """
        self.model.class_weight = None
        if self.mode == "train":
            if self.penalty is None:
                self.model = linear_model.LogisticRegression(penalty='none',class_weight="balanced",max_iter=10000)
            elif self.penalty == 'elasticnet':
                self.model = linear_model.LogisticRegression(penalty=self.penalty, class_weight="balanced",C=self.C, l1_ratio = self.l1_ratio,max_iter=10000,solver='saga')
            else:
                self.model = linear_model.LogisticRegression(penalty=self.penalty, class_weight="balanced",C=self.C, max_iter=10000,solver='saga')
        else:
            if self.penalty is None:
                self.model = linear_model.LogisticRegression(penalty='none',class_weight="balanced",max_iter=20)
            elif self.penalty == 'elasticnet':
                self.model = linear_model.LogisticRegression(penalty=self.penalty, class_weight="balanced",C=self.C, l1_ratio=self.l1_ratio,max_iter=20,solver='saga')
            else:
                self.model = linear_model.LogisticRegression(penalty=self.penalty, class_weight="balanced",C=self.C, max_iter=20,solver='saga')
