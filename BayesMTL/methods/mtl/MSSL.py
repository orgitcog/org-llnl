#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 08:51:38 2018

@author: goncalves1
"""
import os
import pickle
import numpy as np
import pandas as pd

import scipy.special
import scipy.optimize

from sklearn import linear_model
from mpm.design import Method

from numpy.random import default_rng, SeedSequence

inv = np.linalg.inv
norm = lambda z: np.sqrt(np.sum(z**2))

einsum = np.einsum

def logloss(w, x, y, Omega, lambda_reg):
    ''' MSSL with logloss function '''

    # t0 = time()
    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # make sure the data is in the correct format
    for t in range(ntasks):
        if len(y[t].shape) > 1:
            y[t] = np.squeeze(y[t])

    # cost function for each task
    cost = 0
    for t in range(ntasks):
        h_t_x = np.minimum(sigmoid(np.dot(x[t], wmat[:, t])), 0.98)
#       h_t_x = scipy.special.expit(np.dot(x[t], wmat[:, t]))
        f1 = np.multiply(y[t], np.log(h_t_x))
        f2 = np.multiply(1-y[t], np.log(1-h_t_x))
        cost += -(f1 + f2).mean()

    # gradient of regularization term
    # cost += (0.5*lambda_reg) * np.trace(np.dot(np.dot(wmat, Omega), wmat.T))
    cost += (0.5*lambda_reg) * einsum("ik,kj,ij",wmat,Omega,wmat)
    # print('logloss time: {} secs'.format(time()-t0))
    return cost


def logloss_der(w, x, y, Omega, lambda_reg):
    ''' Gradient of the MSSL with logloss function '''

    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # make sure data is in correct format
    for t in range(ntasks):
        if len(y[t].shape) > 1:
            y[t] = np.squeeze(y[t])

    # gradient of logloss term
    grad = np.zeros(wmat.shape)
    for t in range(ntasks):
#        sig_s = scipy.special.expit(np.dot(x[t], wmat[:, t])) #[:, np.newaxis]
        sig_s = sigmoid(np.dot(x[t], wmat[:, t]))
        grad[:, t] = np.dot(x[t].T, (sig_s-y[t]))/x[t].shape[0]
    # gradient of regularization term
    grad += lambda_reg * wmat@Omega
    grad = np.reshape(grad, (ndimension*ntasks, ), order='F')
    return grad


def sigmoid(a):
    # Logit function for logistic regression
    # x: data point (array or a matrix of floats)
    # w: set of weights (array of float)

    # treating over/underflow problems
    a = np.maximum(np.minimum(a, 50), -50)
    #f = 1./(1+exp(-a));
    f = np.exp(a) / (1+np.exp(a))
    return f


def shrinkage(a, kappa):
    return np.maximum(0, a-kappa) - np.maximum(0, -a-kappa)


class MSSLClassifier(Method):
    """
    Implement the MSSL classifier.

    Attributes:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
    """
    def __init__(self, lambda_1=0.1, lambda_2=0, init_params='random', name='MSSL',normalize_data=True):
        """ Initialize object with the informed hyper-parameter values.
        Args:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
        """
        # set method's name and paradigm
        super().__init__(name, 'MTL')

        self.lambda_1 = lambda_1  # trace term
        self.lambda_2 = lambda_2  # omega sparsity
        # self.max_iters = 100      # change to 3000 later
        self.max_iters = 3000      # 100 for initial run
        self.tol = 1e-4  # minimum tolerance: eps * 100

        self.normalize_data = normalize_data
        self.admm_rho = 20000  # ADMM parameter
        self.eps_theta = 1e-3  # stopping criteria parameters
        self.eps_w = 1e-3  # stopping criteria parameters

        self.init_params = init_params
        self.ntasks = -1
        self.ndimensions = -1
        self.W0 = None
        self.Omega0 = None
        self.W = None
        self.Omega = None
        self.fitted = False
        self.offsets = None
        self.intercept = True
        self.test = True
        self.output_directory = ''

    def init_with_data(self,x,y, seed=0, init_param=True,intercept=True):
        self.ntasks = len(x)  # get number of tasks
        self.ndimensions = x[0].shape[1]  # dimension of the data
        if intercept:
            self.ndimensions += 1  # if consider intercept, add another feat +1
        self.intercept = intercept

        x, y, offsets = self.__preprocess_data(x, y)
        self.offsets = offsets
        self.W0, self.Omega0 = self.__init_parameters(mode=self.init_params, seed=seed,X=x, Y=y)
        self.fitted= True
        self.test = True
        return

    def fit(self, x, y, column_names=None, **kwargs):
        max_iters = 0
        if self.mode == 'train':
            self.logger.info('Traning process is about to start.')
            self.feature_names = column_names
            self.max_iters = 3000 #100
            if os.path.exists(self.output_directory+'/max_iter.npy'):
                max_iters = np.load(self.output_directory+'/max_iter.npy')
        else:
            self.max_iters = 10
        if not self.fitted:
            x, y, offsets = self.__preprocess_data(x, y)
            # self.ntasks = len(x)  # get number of tasks
            # self.ndimensions = x[0].shape[1]  # dimension of the data
            self.offsets = offsets
            self.W0, self.Omega0 = self.__init_parameters(mode=self.init_params,X=x, Y=y)

        # model train
        W, Omega = self.__mssl_train(x, y,start=max_iters)
        self.fitted = True

        self.W = W.copy()
        self.Omega = Omega.copy()
        if self.mode == 'train':
            np.save(self.output_directory+'/max_iter.npy',self.max_iters)
            fname = os.path.join(self.output_directory, '%s.mdl' % self.__str__())
            with open(fname, 'wb') as fh:
                pickle.dump([self.lambda_1,self.lambda_2,self.W, self.Omega], fh)
        return

    def predict(self, x, prob=False, **kwargs):
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
        for t in range(len(x)):
            yhat[t] = scipy.special.expit(np.dot(x[t], self.W[:, t]))
            if not prob:
                yhat[t] = np.around(yhat[t]).astype(np.int32)
        return yhat

    def __preprocess_data(self, x, y):
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
                x[t] = np.hstack((x[t], 0.79788*np.ones((x[t].shape[0], 1))))
        return x, y, offsets

    def __init_parameters(self, seed=0, mode='random', X=None, Y=None):

        rng = default_rng(seed)
        if mode == 'random':
            W0 = -0.05 + 0.05*rng.random((self.ndimensions, self.ntasks))
            Omega0 = np.eye(self.ntasks)

        elif mode == 'warm_start':
            W0 = np.zeros((X[0].shape[1], len(X)))
            for t, (x, y) in enumerate(zip(X, Y)):

                model_ = linear_model.LogisticRegression(penalty='none',
                                                         fit_intercept=False)
                model_.fit(x, y.ravel())

                W0[:, t] = model_.coef_

            Omega0 = np.eye(self.ntasks)

        else:
            raise('Unknown init_parameters mode '.format(mode))

        return W0, Omega0

    def __mssl_train(self, x, y,start=0):

        # initialize learning parameters
        # np.linalg.solve(xmat, ymat)  #  regression warm start
        W = self.W0
        Omega = self.Omega0

        # scipy opt parameters
        opts = {'maxiter': 1000, 'disp': False}
        for _ in range(start,self.max_iters):
            # print('MSSL iter: %d' % it)
            # Minimization step
            W_old = W.copy()
            Wvec = np.reshape(W, (self.ndimensions*self.ntasks, ), order='F')

#            r = scipy.optimize.check_grad(logloss,
#                                          logloss_der,
#                                          Wvec, x, y, Omega, self.lambda_1)

            additional = (x, y, Omega, self.lambda_1)
            res = scipy.optimize.minimize(logloss, x0=Wvec,
                                          args=additional,
                                          jac=logloss_der,
                                          method='L-BFGS-B',
                                          options=opts)

            # put it in matrix format, where columns are coeff for each task
            W = np.reshape(res.x.copy(),
                           (self.ndimensions, self.ntasks), order='F')
            # Omega step:
            Omega_old = Omega

            # Learn relationship between tasks (inverse covariance matrix)
            # t0 = time()
            Omega = self.__omega_step(np.cov(W, rowvar=False),
                                      self.lambda_2, self.admm_rho)
            # print('Omega optim: {} secs'.format(time()-t0))
            # checking convergence of Omega and W
            diff_Omega = np.linalg.norm(Omega-Omega_old)
            diff_W = np.linalg.norm(W-W_old)

            # if difference between two consecutive iterations are very small,
            # stop training
            if (diff_Omega < self.eps_theta) and (diff_W < self.eps_w):
                break

        return W, Omega


    def __omega_step(self, S, lambda_reg, rho):
        '''
        ADMM for estimation of the precision matrix.

        Input:
           S: sample covariance matrix
           lambda_reg: regularization parameter (l1-norm)
           rho: dual regularization parameter (default value = 1)
        Output:
           omega: estimated precision matrix
        '''
        # global constants and defaults
        max_iters = 10
        abstol = 1e-5
        reltol = 1e-5
        alpha = 1.4

        # varying penalty parameter (rho)
        mu = 10
        tau_incr = 2
        tau_decr = 2

        # get the number of dimensions
        ntasks = S.shape[0]

        # initiate primal and dual variables
        Z = np.zeros((ntasks, ntasks))
        U = np.zeros((ntasks, ntasks))

#        print('[Iters]   Primal Res.  Dual Res.')
#        print('------------------------------------')

        for k in range(0, max_iters):

            # x-update
            # numpy returns eigc_val,eig_vec as opposed to matlab's eig
            eig_val, eig_vec = np.linalg.eigh(rho*(Z-U)-S)

            # check eigenvalues
            if isinstance(eig_val[0], complex):
                print("Warning: complex eigenvalues. Check covariance matrix.")

            # eig_val is already an array (no need to get diag)
            xi = (eig_val + np.sqrt(eig_val**2 + 4*rho)) / (2*rho)
            X = (eig_vec*xi)@(eig_vec.T)
            # X = np.dot(np.dot(eig_vec, np.diag(xi, 0)), eig_vec.T)

            # z-update with relaxation
            Zold = Z.copy()
            X_hat = alpha*X + (1-alpha)*Zold
            Z = shrinkage(X_hat+U, lambda_reg/rho)
#            Z = Z - np.diag(np.diag(Z)) + np.eye(Z.shape[0])
            # dual variable update
            U = U + (X_hat-Z)

            # diagnostics, reporting, termination checks
            r_norm = np.linalg.norm(X-Z, 'fro')
            s_norm = np.linalg.norm(-rho*(Z-Zold), 'fro')

#            if r_norm > mu*s_norm:
#                rho = rho*tau_incr
#            elif s_norm > mu*r_norm:
#                rho = rho/tau_decr

            eps_pri = np.sqrt(ntasks**2)*abstol + reltol*max(np.linalg.norm(X,'fro'), np.linalg.norm(Z,'fro'))
            eps_dual = np.sqrt(ntasks**2)*abstol + reltol*np.linalg.norm(rho*U,'fro')

            # keep track of the residuals (primal and dual)
#            print('   [%d]    %f     %f ' % (k, r_norm, s_norm))
            if r_norm < eps_pri and s_norm < eps_dual:
                break

        return Z

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
            importance = self.W[:-1, t] * np.sign(self.W[:-1, t])
            # importance = np.exp(self.W[:-1, t]) * np.sign(self.W[:-1, t])
            dfs.append(pd.Series(importance, index=self.feature_names))
        return dfs
    def return_support(self):
        support = np.sum(np.abs(self.W),axis=1)
        return support

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.lambda_1 = params['lambda_1']
        self.lambda_2 = params['lambda_2']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'lambda_1': self.lambda_1,
               'lambda_2': self.lambda_2}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        lambda_1 = np.logspace(-1, 3, 10)
        lambda_2 = np.logspace(-5, 2, 10)
        for r0 in lambda_1:
            for r1 in lambda_2:
                yield {'lambda_1': r0,
                       'lambda_2': r1}

    def load_params(self, params):
        """
        load previously trained parameters to be used in the execution.
        Args:
            params (dict): dict with parameter values.
        """
        self.lambda_1 = params["lambda_1"]
        self.lambda_2 = params["lambda_2"]
        self.W = params["W"]
        self.Omega = params["Omega"]

    def return_params(self):
        """ Return parameters used in the execution.
        Return:
            params (dict): dict containing the params values.
        """
        ret = {'W': self.W,
               'Omega': self.Omega}
        return ret

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())

