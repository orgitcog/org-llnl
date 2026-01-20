#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

L21 Joint Feature Learning with Logistic Loss.

OBJECTIVE
   argmin_{W,C} { sum_i^t (- sum(log (1./ (1+ exp(-X{i}*W(:, i) - Y{i} .*
                  C(i)))))/length(Y{i})) + opts.rho_L2 * \|W\|_2^2 +
                  rho1 * \|W\|_{2,1} }

INPUT:
 X: {n * d} * t - input matrix
 Y: {n * 1} * t - output matrix
 rho1: L2,1-norm group Lasso parameter.
 optional:
   opts.rho_L2: L2-norm parameter (default = 0).

OUTPUT:
 W: model: d * t
 C: model: 1 * t
 funcVal: function value vector.

RELATED PAPERS:
   [1] Evgeniou, A. and Pontil, M. Multi-task feature learning, NIPS 2007.
   [2] Liu, J. and Ye, J. Efficient L1/Lq Norm Regularization, Technical
       Report, 2010.

@author: goncalves1
"""
import os
import pickle

import pandas as pd

import numpy as np

from scipy.special import expit

from mpm.design import Method

clip = np.clip
sumn = np.sum
norm = np.linalg.norm 
maximumnn = np.maximum

hstack = np.hstack

def sigmoid(a, epsilon=1e-15):
    """ sigmoid function for logistic regression
    
    inputs:
        a: logit inputs 
        epsilon: overflow-underflow tolerance 
    """
    return clip(expit(a),epsilon, 1 - epsilon)

def proximal_l21(W, lambda_1):
    """Proximal operator for the l2,1 norm
    """
    norm2 = norm(W, axis=0)
    multiplier = maximumnn(0, 1 - lambda_1 / (norm2 + 1e-10)) # Adding a small constant to avoid division by zero
    return W * multiplier

def proximal_frobenius(W, lambda_2):
    """ Proximal operator for the Frobenius norm
    """
    return W / (1 + 2*lambda_2)

def l21_norm(W):
    """ compute l21 norm 
    """
    return sumn(norm(W[:-1], axis=1))

def frobenius_norm(W):
    """ frobenius norm squared
    """
    return norm(W[:-1], 'fro')

def soft_thresholding_operator(z, lambda1):
    """ soft-thresholding operator 
    """
    norm_val = norm(z, 2)
    if norm_val == 0:
        return z
    return maximumnn(0, 1 - lambda1 / norm_val) * z


class JFLMTLClassifier(Method):
    """
    Implement the L21 Joint Feature Learning classifier.

    Attributes:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
    """
    def __init__(self, rho_L21=0.1, rho_L2=0, name='JFLMTL', normalize_data=True):
        """ Initialize object with the informed hyper-parameter values.

        Args:
            rho_L21 (float): l2,1 penalization hyper-parameter
            rho_L2 (float): l2 penalization hyper-parameter
        """
        # set method's name and paradigm
        super().__init__(name, 'MTL')

        self.rho_L21 = rho_L21
        self.rho_L2 = rho_L2
        self.max_iters = 60000
        self.tol = 1e-6  # minimum tolerance: eps * 100
        self.ntasks = -1
        self.ndimensions = -1
        self.W = None 
        self.W0 = None 
        self.output_directory = ''
        self.fitted= False
        self.offsets = None
        self.intercept = True
        self.test = True
        self.normalize_data = normalize_data

    def __preprocess_data(self, x, y):
        """Preprocess the data (normalization, add constant feature if necessary)
            
        Aargs:
            x (list of data matrix): len(.) = number of tasks 
            y (list of vectors): len(.) = number of tasks 
        """
        # make sure y is in correct shape
        for t in range(self.ntasks):
            x[t] = x[t].astype(np.float64)
            y[t] = y[t].astype(np.float64).ravel()
            if len(y[t].shape) < 2:
                y[t] = y[t][:, np.newaxis]

        offsets = {'x_offset': list(),
                   'x_scale': list()}
        for t in range(self.ntasks):
            if self.normalize_data:
                offsets['x_offset'].append(x[t].mean(axis=0))
                std = x[t].std(axis=0)
                std[std == 0] = 1
                offsets['x_scale'].append(std)
            else:
                offsets['x_offset'].append(np.zeros(x[t].mean(axis=0).shape))
                std = np.ones((x[t].shape[1],))
                offsets['x_scale'].append(std)
            x[t] = (x[t] - offsets['x_offset'][t]) / offsets['x_scale'][t]
            if self.intercept:
                # add intercept term, 0.79788 is average absolute deviation (AAD) of
                # a standard normal distribution
                x[t] = hstack((x[t], 0.79788*np.ones((x[t].shape[0], 1))))
        return x, y, offsets  

    def init_with_data(self, x, y, seed=0, init_param=True, intercept=True):
        """ Initialize the model with the data, perform normalization of the data by 
            calling _preprocess_data function 

        Args:
          seed:  The seed for random initialization 
          intercept: whether to add a bias term (i.e., augment the data by add constant feature)
          init_param: whether to intialize the model parameters 
        """

        self.ntasks = len(x)  # get number of tasks
        self.ndimensions = x[0].shape[1]  # dimension of the data
        if intercept:
            self.ndimensions += 1  # if consider intercept, add another feat +1
        self.W0 = np.ones((self.ndimensions, self.ntasks)) # initialize a starting point
        self.intercept = intercept

        x, y, offsets = self.__preprocess_data(x, y)

        # compute the initial step size based on Lipschitz constant
        sum_matrix = np.zeros((x[0].shape[1], x[0].shape[1]))
        for t in range(self.ntasks):
            sum_matrix += x[t].T.dot(x[t])
        l1_norm = np.max(np.sum(np.abs(sum_matrix), axis=0))
        l_inf_norm = np.max(np.sum(np.abs(sum_matrix), axis=1))
        # learning rate initialized based on upper bound on Lipschitz constant 
        self.lr = 4/np.sqrt(l1_norm*l_inf_norm) 
        self.offsets = offsets
        self.fitted= True
        self.test = True
        return

    def fit(self, x, y, column_names=None, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (list): list of pandas.DataFrame input data matrix (covariates).
            y (list): list of pandas.DataFrame label vector (outcome).
        Returns:
            None.
        """
        max_iters = 0
        if self.mode == 'train':
            self.logger.info('Traning process is about to start.')
            self.feature_names = column_names
            self.max_iters = 60000
            self.tol = 1e-6
            if os.path.exists(self.output_directory+'/max_iter.npy'):
                max_iters = np.load(self.output_directory+'/max_iter.npy')
        else:
            self.max_iters = 20000
            self.tol = 1e-5
        if not self.fitted:
            x, y, offsets = self.__preprocess_data(x, y)
            self.offsets = offsets

        if self.mode == 'train':
            self.logger.info('Training process started.')

        if self.W is not None:
            W0 = self.W
        else:
            W0 = self.W0
        W = self.fista(np.copy(W0), x, y, lambda1= self.rho_L21, lambda2=self.rho_L2, lr=self.lr, max_iter=self.max_iters-max_iters, tol=self.tol)
        self.fitted = True

        self.W = np.copy(W)

        if self.mode=="train":
            np.save(self.output_directory+'/max_iter.npy',self.max_iters)
            # save model into pickle file
            fname = os.path.join(self.output_directory, '%s.mdl' % self.__str__())
            with open(fname, 'wb') as fh:
                pickle.dump([self.rho_L21, self.rho_L2, self.W], fh)
        return

    def predict(self, x, prob=False,**kwargs):
        """ Predict regression value for the input x.
        Args:
            x (pandas.DataFrame): list of pandas.DataFrame input data matrix.
        Returns:
            list of numpy.array with the predicted values.
        """
        if self.test and self.mode=="train":
            for t in range(self.ntasks):
                x[t] = x[t].astype(np.float64)
                if x[t].shape[1]==len(self.offsets['x_offset'][t]):
                    x[t] = (x[t]-self.offsets['x_offset'][t])
                    if self.normalize_data:
                        x[t] = x[t]/self.offsets['x_scale'][t]
                    if self.intercept:
                        x[t] = np.hstack((x[t], 0.79788*np.ones((x[t].shape[0], 1))))
            self.test = False # since data is passed by reference only need
                            # to be pre-processed once
        yhat = [None]*len(x)
        for t in range(self.ntasks):
            yhat[t] = sigmoid(np.dot(x[t], self.W[:, t]))
            if not prob:
                yhat[t] = np.around(yhat[t]).astype(np.int32)
        return yhat

    def feature_importance(self):
        """ Compute/Extract feature importance from the trained model
            and store it as a pd.Series object.
        Args:
            None
        Returns:
            pd.Series with feature importance values.
        """
        dfs = list()
        for t in range(self.W.shape[1]):
            importance = np.exp(self.W[:-1, t]) * np.sign(self.W[:-1, t])
            dfs.append(pd.Series(importance, index=self.feature_names))
        return dfs

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.rho_L21 = params['rho_L21']
        self.rho_L2 = params['rho_L2']

    def return_support(self):
        if self.intercept:
            return np.sum(np.abs(self.W[:-1,:]),axis=1)
        else:
            return np.sum(np.abs(self.W),axis=1)

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'rho_L21': self.rho_L21,
               'rho_L2': self.rho_L2}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        rho_L21 = np.logspace(-10, 1, 10)
        rho_L2 = np.logspace(-10, 1, 10)
        for r0 in rho_L21:
            for r1 in rho_L2:
                yield {'rho_L21': r0,
                       'rho_L2': r1}

    def load_params(self, params):
        """
        load previously trained parameters to be used in the execution.
        Args:
            params (dict): dict with parameter values.
        """
        self.rho_L21 = params["rho_L21"]
        self.rho_L2 = params["rho_L2"]
        self.W = params["W"]

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())

    def logistic_loss(self,W, X, Y):
        """Compute the Logistic Loss 
        """
        loss = 0
        epsilon = 1e-15
        for t in range(self.ntasks):
            pred = sigmoid(X[t].dot(W[:, t]))
            pred = np.clip(pred, epsilon, 1 - epsilon)
            loss += -np.sum(np.squeeze(Y[t]) * np.log(pred) + (1 - np.squeeze(Y[t])) * np.log(1 - pred))
        return loss

    def objective(self, W, X, Y, lambda1, lambda2):
        """Compute the objective function 
        """
        return self.logistic_loss(W, X, Y) + lambda1 * l21_norm(W) + lambda2 * (frobenius_norm(W)**2)

    def gradient(self, W, X, Y, lambda2):
        """Compute the gradient
        """
        grad = np.zeros_like(W)
        epsilon = 1e-15
        for t in range(self.ntasks):
            pred = sigmoid(X[t].dot(W[:, t]))
            pred = np.clip(pred, epsilon, 1 - epsilon)
            grad[:, t] += X[t].T.dot(pred - np.squeeze(Y[t]))
        grad[:-1] += 2 * lambda2 * W[:-1]
        return grad

    def fista(self, W, X, Y, lambda1=0.1, lambda2=0.1, lr=0.01, max_iter=100, tol=1e-4):
        """Optimization scheme based on FISTA, see Beck, Amir, and Marc Teboulle. 
            "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." 
            SIAM journal on imaging sciences 2.1 (2009): 183-202.
        """
        W_prev = np.copy(W)
        theta = 1
        theta_prev = 1
        Y_temp = np.copy(W)
        obj_prev = float('inf')

        for _ in range(max_iter):

            grad = self.gradient(Y_temp, X, Y, lambda2)
            W_new  = Y_temp - lr * grad

            # Proximal update for l2,1-norm
            W_new = proximal_l21(W_new, lambda1 * lr)

            # Proximal update for Frobenius norm
            W_new = W_new / (1 + 2 * lambda2 * lr)

            theta = (1 + np.sqrt(1 + 4 * theta**2)) / 2
            Y_temp = W_new + ((theta_prev - 1) / theta) * (W_new - W_prev)

            theta_prev = theta
            W_prev = np.copy(W_new)

            obj_curr = self.objective(W_new, X, Y, lambda1, lambda2)

            if obj_curr-obj_prev>0:
                theta = 1
                theta_prev = 1
                Y_temp = np.copy(W_new)

            if abs(obj_prev - obj_curr)/abs(obj_curr) < tol:
                break

            obj_prev = obj_curr

        return W_new

