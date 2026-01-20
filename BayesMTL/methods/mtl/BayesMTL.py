
import argparse

import os
from copy import deepcopy

import pickle
import pandas as pd

import numpy as np
from numpy.random import default_rng

import seaborn as sns
import matplotlib.pyplot as plt

from bayesmtl.methods.method import Method # abstract method class


from scipy.special import expit
from scipy.stats import wishart
from scipy.special import logsumexp,digamma,multigammaln,beta as betaf

sumn = np.sum
sqrt = np.sqrt 
log = np.log
clip = np.clip 
hstack = np.hstack
eps = np.finfo(float).eps

trace = np.trace 
inv = np.linalg.inv
pinv = np.linalg.inv
det = np.linalg.det

logn = lambda z: log(z+eps)
norm = lambda z: sqrt(sumn(z**2))


def multidigamma(a,t):
    """
        Multidigamma gamma function     
    """
    return sumn ([digamma(a-i/2)  for i in range(t)])

def sigmoid(a, epsilon=1e-15):
    """ sigmoid function for logistic regression
    
    inputs:
        a: logit inputs 
        epsilon: overflow-underflow tolerance 
    """
    return clip(expit(a),epsilon, 1 - epsilon)

class BayesMTLClassifier(Method):
    """
    Implementation of the BayesMTL classifier.

    Attributes:
        Clustering (dictionary): the clustering of different tasks, a
                    dictionary where keys are the cluster
                    indices and values are the task indices
                    (e.g., {1: [1,2,3], 2:[4,5]}, where 1 is cluster indice, and the values [1,2,3]
                    is a list of task indices
                    )
        Clusters (dictionary):   the clustering of different tasks, a
                    dictioonary where keys are the task
                    indices and values are the cluster
                    assignment
                    (e.g., {1: 1, 2:1, 3:2}, where 1 is the task indice, and value 1 is the cluster 
                    assignment 
                    )

        Model parameters:
            alp0,beta0 (list of positive floats): 
                    priors for the Bernoulli parameter, len(.)= number of clusters 
            v (list of positive float), V (list of PD T by T matrix): 
                    prior for the common precision matrix across the tasks within a given
                    cluster, len(.)= number of clusters 

        Data parameters: 
            n (a list of floats): weights based on number of samples used to initialize V (to 
                                  accounts for various sample sizes acrosss tasks), 
                                  len(.) = number of tasks 
            nsamples (a list of ints): number of samples from each task, len(.) = number of tasks

        Control variables: 
            fitted (bool): true if model is already fitted 
            test (bool):   true if model is in test model 
            bagging (bool): true if we are using bagging for training
            stochastic (bool): true if we are using stochastic coordinate ascent  

        
        Log-file parameters: 
            output_directory (str of path):  output directory to save the models and log-files 
    """
    def __init__(self, init_params='random', name='BayesMTL', normalize_data=True, bagging=True):
        """ Initialize the model object

        Args:
          init_params (str): the mode of how to initialize the parameters 
          normalize_data (bool): true if the data needed to be normalized 
        """
        # set method's name and paradigm
        super().__init__(name, 'MTL')

        self.interept = True

        self.max_iters = 100
        print("Max number of iterations is {}".format(self.max_iters),flush=True)
        self.tol = 1e-4  # minimum tolerance: eps * 100

        self.normalize_data = normalize_data

        self.init_params = init_params
        self.ntasks = -1
        self.ndimensions = -1
        self.output_directory = ''
        self.n = None 
        self.nsamples = None
        self.fitted=False
        self.offsets = None
        self.test = True
        self.clustering = None
        self.clusters = None
        self.alp0 = None 
        self.beta0 = None 
        self.alp = None
        self.beta = None
        self.v = None
        self.V = None
        self.theta = None
        self.phi = None
        self.m = None
        self.prec = None
        self.sigma = None 
        self.bagging = bagging 
        self.stochastic = True 

    def init_with_data(self, x, y, clustering=None, clusters=None, seed=0, init_param=True, intercept=True):
        """ Initialize the model with the data, perform normalization of the data by 
            calling _preprocess_data function 

        Args:
          clustering: the clustering of different tasks
                      dictioonary where keys are the cluster
                      indices and values are the task indices.
                      Default-all tasks are in one cluster 

          clusters:  the clustering of different tasks
                      dictionary where keys are the task
                      indices and values are the cluster
                      assignment
                      Default-all tasks are in one cluster 

          seed:      The seed for random initialization 
          intercept: whether to add a bias term (i.e., augment the data by add constant feature)
          init_param: whether to intialize the model parameters 
        """
        self.ntasks = len(x)  # get number of tasks
        self.ndimensions = x[0].shape[1]  # dimension of the data
        if self.clustering is None: 
            self.clustering = {0: list(range(self.ntasks))}
            self.clusters = {i: 0 for i in range(self.ntasks)}
        else:
            self.clustering = clustering
            self.clusters= clusters
        if intercept:
            self.ndimensions += 1  # if consider intercept, add another feat +1
        self.intercept = intercept

        x, y, offsets = self.__preprocess_data(x, y)
        self.offsets = offsets
        self.fitted = True
        self.test = True
        self.nsamples = np.array([len(Y) for Y in y],dtype=int)
        self.n = [1/(len(Y))**2 for Y in y]
        if init_param:
            rng = default_rng(seed)
            self.m= self.__init_parameters(mode=self.init_params, X=x, Y=y, rng=rng)
        return

    def Compute_ELBO(self, v0, V0, v, V, alp0, beta0, alp, beta, m, y_pred, sigma, phi, x, y): 
        
        """ Compute the ELBO to track progress of the algorithm 
        """    
        elbo = 0 
        for key,value in self.clustering.items():
            elbo += -(0.5*v[key])*trace(inv(V[key]@V0[key]))-0.5*(v0[key]+self.ndimensions)*logn(det(V[key]))\
                    +0.5*(v0[key]+self.ndimensions-v[key])*multidigamma(v[key]*0.5,len(value))+v[key]*len(value)/2\
                    +multigammaln(0.5*v[key],len(value))+(alp0[key]+sumn(phi[key])-alp[key])*digamma(alp[key])\
                    +(beta0[key]+self.ndimensions-sumn(phi[key])-beta[key])*digamma(beta[key])\
                    +logn(betaf(alp[key],beta[key]))+(alp[key]+beta[key]-self.ndimensions-alp0[key]-beta0[key])*digamma(alp[key]+beta[key])\
                    -(0.5*v[key])*(sumn([trace(inv(V[key])@sigma[key][j]) for j in range(self.ndimensions)])+sumn([m[value,j].dot(inv(V[key]).dot(m[value,j])) for j in range(self.ndimensions)]))\
                    +sumn([(np.squeeze(y[t],axis=1)-sigmoid(y_pred[t])).dot(x[t]@(m[t,:]*phi[key])) for t in value])\
                    -sumn ([sumn(logsumexp(hstack((y_pred[t],np.zeros(self.ndimensions))))) for t in value])\
                    +sumn ([sigmoid(y_pred[t]).dot(y_pred[t]) for t in value])\
                    -0.125*sumn ([sumn(y_pred[t]**2) for t in value])\
                    +0.25*sumn ([y_pred[t].dot(x[t]@(m[t,:]*phi[key])) for t in value])\
                    -0.125*sumn ([sumn((x[t]@(m[t,:]*phi[key]))**2) for t in value])\
                    +0.125*sumn ([(m[t,:]**2*(phi[key]-1)*phi[key]).dot(sumn(x[t]**2, axis=0)) for t in value])\
                    -0.125*sumn ([np.array([sigma[key][j][t_index,t_index] for j in range(self.ndimensions)]).dot(phi[key] * sumn (x[t]**2, axis=0)) for (t_index,t) in enumerate(value)])\
                    +0.5*sumn ([logn(det(sigma[key][j])) for j in range(self.ndimensions)])-sumn(phi[key]*logn(phi[key]))-sumn((1-phi[key])*logn(1-phi[key]))
        return elbo


    def fit(self, x, y, column_names=None, **kwargs):
        """fit the model, has two modes:
            if in the train mode using the standard coordinate ascent 
            algorithm otherwise use stochastic coordinate ascent algorithm 

          Aargs:
            x (list of data matrix): len(.) = number of tasks 
            y (list of vectors): len(.) = number of tasks 
            column_names (optional, list of strings): feature names 
        """
        if self.mode == 'train':
            self.stochastic = False 
            if not self.fitted:
                x, y, offsets = self.__preprocess_data(x, y)
                self.offsets = offsets
            # self.logger.info('Traning process is about to start.')
            self.feature_names = column_names

            # load the training information if exists 
            # if os.path.exists(self.output_directory+'/max_iter.npy'):
            #     max_iters = np.load(self.output_directory+'/max_iter.npy')
            # else:
            #     max_iters = 0
            max_iters = 0

            # model train
            alp, beta, v, V, phi, m, sigma = self.__mssl_train(x, y, self.alp0,self.beta0, 
                                                            self.v0, self.V0, self.phi, self.m, 
                                                            start=max_iters, 
                                                            niter=self.max_iters)
            # np.save(self.output_directory+'/max_iter.npy',self.max_iters)
            # fname = os.path.join(self.output_directory, '%s.mdl' % self.__str__())
            # with open(fname, 'wb') as fh:
            #     pickle.dump([self.alp0,self.beta0, alp, beta, self.v0,self.V0,v, V, phi, m, sigma], fh)
        else:
            self.stochastic = True 
            if not self.fitted:
                x, y, offsets = self.__preprocess_data(x, y)
                self.offsets = offsets
            # stochastic training for hyperparameter selection
            alp, beta, v, V, phi, m, sigma = self.__mssl_train(x, y, self.alp0,
                            self.beta0, self.v0, self.V0, self.phi, self.m,
                            niter=40,ndim=np.min((4000,self.ndimensions))) # 4000 is set for computational purposes
        self.alp = alp
        self.beta = beta
        self.v = v
        self.V = V.copy()
        self.theta = alp/(alp+beta)
        self.phi = phi.copy()
        self.m = m.copy()
        self.sigma = deepcopy(sigma)
        self.prec = [inv(sigma_temp) for sigma_temp in sigma]
        self.fitted = True

    def predict(self, x, r_i=0, seed=0, prob=False, if_sampling=False, **kwargs):

        """perform prediction given the data 
        
          Aargs:
            x (list of data matrix): len(.) = number of tasks 
            column_names (optional, list of strings): feature names 
            prob (optional, bool): if true return probability othersie 
                                   rounded to the most-likely class
            if_sampling (optional, bool): if true use monte-carlo method to 
                                          peroform the posterior predictions, 
                                          otherwise use the mean parameters 
                                          for predictions 
            n_samples (optional, int): number of posterior samples used for
                                       predictions, only applicable if if_sampling
                                       is true 
            seed (optional, int):   random seeds for posterior sampling
        """
        if self.test and self.mode == "train":
            # if first time on test data need to 
            # transform the data first (only once)
            for t in range(self.ntasks):
                x[t] = x[t].astype(np.float64)
                x[t] = (x[t]-self.offsets['x_offset'][t])
                if self.normalize_data:
                    x[t] = x[t]/self.offsets['x_scale'][t]
                if self.intercept:
                    x[t] = hstack((x[t],0.79788*np.ones((x[t].shape[0], 1))))
            self.test = False # since data is passed by reference only need
                            # to be pre-processed once
        if if_sampling:
            w = np.empty_like(self.m)
            z = np.ones(self.m.shape[1],dtype=bool)
            yhat = [None]*len(x)
            rng = default_rng(seed)
            nsamples = 100*self.ndimensions
            for n in range(nsamples):
                for j in range(self.m.shape[1]):
                    w[:,j] = rng.multivariate_normal(self.m[:,j], self.sigma[j])
                    z[j] = rng.binomial(1,self.phi[j])
                if n==0:
                    for t in range(len(x)):
                        yhat[t] = sigmoid(x[t].dot(w[t,:]*z))
                else:
                    for t in range(len(x)):
                        yhat[t] = yhat[t]*n/(n+1)+sigmoid(x[t].dot(w[t,:]*z))/(n+1)
        else:
            yhat = [None]*len(x)
            for t in range(len(x)):
                # yhat[t] = clip(expit(x[t].dot(self.m[t, :]*self.phi[self.clusters[t]])),epsilon, 1 - epsilon)
                yhat[t] = sigmoid(x[t].dot(self.m[t, :]*self.phi[self.clusters[t]]))
        if prob:
            return yhat
        else:
            for t in range(len(x)):
                yhat[t] = np.around(yhat[t]).astype(np.int32)
            return yhat

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

    def __init_parameters(self, mode='random', X=None, Y=None, rng=default_rng(0)):
        """initialze the mean parameters based on the mode 
        """
        if mode == 'random':
            m= np.empty((self.ntasks, self.ndimensions))
            for t in range(self.ntasks):
                m[t, :] = np.squeeze(X[t].T@Y[t]) # initialize using matrix product
            m += rng.random((self.ntasks,self.ndimensions)) # add noise 
        else:
            m = np.ones((self.ntasks,self.ndimensions)) 

        return m

    def __mssl_train(self, x, y, alp0, beta0, v, V, phi, m, start=0, seed=1234, niter=80, ndim=400):
        """function to train the model

        Aargs:
            x (list of data matrix): 
                len(.) = number of tasks 
            y (list of vectors): 
                len(.) = number of tasks 
            alp0,beta0 (list of positive floats): 
                priors for the Bernoulli parameter, len(.)= number of clusters 
            v (list of positive float), V (list of PD matrix): 
                prior for the common precision matrix across the tasks within a given
                cluster, len(.)= number of clusters 
                Note (V=V^-1 of the paper, i.e., it's inverse-wishart parameter)
            phi (list of float vectors): 
                sparsity parameters, len(.)= number of clusters 
            m (regression coefficient matrix)
            seed (optional): random seed 
            niter (optional): number of iterations
            ndim (optional, int): dimension size for stochastic coordinate ascent
                                  for each sampling step 
            start (optional, int): if previous training already exists, the start
                                point to resume training 

        Imediate Variables: 
            y_pred (a list of vectors): 
                the predicted logit values for all the samples, same
                shape as y 
            y_pred_prev (a list of vectors): 
                the predicted logit values for all the samples in the previous iteration, same
                shape as y, needed to compute ELBO approximations 
            y_pred_m (a list of vectors): 
                the contribution of particular feature to the 
                predicted logit values for all the samples, same
                shape as y. (This is introduced for more efficient update
                of y_pred)
            idx_choosen: the indices of features choosen for stochastic coordinate ascent
                         algorithm, default to enumeration for standard coordinate ascent 
        Return: alp, beta, v, V, phi, m, sigma (model parameters)
        """
        rng = default_rng(seed)
        y_tilde = np.zeros((self.ntasks, self.ndimensions))
        sig_tilde= np.zeros((self.ntasks, self.ndimensions))
        y_pred = [np.zeros_like(Y).reshape(-1) for Y in y]
        y_pred_prev = [np.zeros_like(Y).reshape(-1) for Y in y]
        y_pred_m = [np.zeros_like(Y).reshape(-1) for Y in y]
        idx_choosen = range(self.ndimensions)

        # if only given a single value of v duplicate it 
        # to a list 
        if not isinstance(v, (list, np.ndarray)):
            v = [v] * len(self.clustering)

        v0 = deepcopy(v) # deep copy avoid pass by reference 
        V0 = deepcopy(V) # deep copy avoid pass by reference

        # if only given as scalars for  alp0,beta0 duplicate them 
        # to lists of proper length
        if isinstance(alp0, (list, np.ndarray)):
            alp0 = np.array(alp0) 
            alp = deepcopy(alp0)
            beta = deepcopy(beta0)
        else:
            alp0 = np.array([alp0] * len(self.clustering))
            beta0 = np.array([beta0] * len(self.clustering) )
            alp = deepcopy(alp0)
            beta = deepcopy(beta0)

        max_sz = np.max(self.nsamples)

        if self.bagging:
            indices = [rng.choice(nsample, size=max_sz, replace=(nsample != max_sz)) for nsample in self.nsamples]
        else:
            indices = [range(nsample) for nsample in self.nsamples] # default without bagging
        phi_clusters = [phi[self.clusters[t]] for t in range(self.ntasks)]

        for t in range(self.ntasks):
            idx = indices[t]
            y_pred[t][idx] = x[t][idx, :] @ (m[t, :] * phi_clusters[t])
            y_pred_prev[t][idx] = deepcopy(y_pred[t][idx])
            y_tilde[t, :] = np.squeeze(x[t][idx, :].T @ y[t][idx]) * phi_clusters[t]

        for i_iter in range(start, niter):

            # choose the features to be updated 
            if self.stochastic:
                idx_choosen = rng.choice(range(self.ndimensions),size=ndim,replace=False)

            # update the precision matrix 
            for t in range(self.ntasks):
                sig_tilde[t, :] = 0.25 * (phi[self.clusters[t]] * sumn (x[t][indices[t],:]**2, axis=0))
            prec0 = [v[i]*inv(V_temp) for i,V_temp in enumerate(V)]
            prec = [[prec0_temp.copy() for _ in range(self.ndimensions)] for prec0_temp in prec0]
            
            for i in idx_choosen:
                for key,value in self.clustering.items():
                    prec[key][i] += np.diag(sig_tilde[value,i])
            sigma = [inv(prec_temp) for prec_temp in prec]

            # update the mean paramters
            for j in idx_choosen:
                m_values = m[:, j]
                for t in range(self.ntasks):
                    y_pred_m[t][indices[t]] = x[t][indices[t], j] * (m_values[t] * phi_clusters[t][j])
                for key, value in self.clustering.items():
                    # m[value, j] = sigma[key][j] @ (y_tilde[value, j] - np.array([phi_clusters[t][j] * np.clip(sigmoid(y_pred[t][indices[t]]), epsilon, 1 - epsilon).dot(x[t][indices[t], j]) for t in value]) + np.array([(0.25 * phi_clusters[t][j]) * (y_pred_m[t][indices[t]]).dot(x[t][indices[t], j]) for t in value]))
                    m[value, j] = sigma[key][j] @ (y_tilde[value, j] - np.array([phi_clusters[t][j] * sigmoid(y_pred[t][indices[t]]).dot(x[t][indices[t], j]) for t in value]) + np.array([(0.25 * phi_clusters[t][j]) * (y_pred_m[t][indices[t]]).dot(x[t][indices[t], j]) for t in value]))
                for t in range(self.ntasks):
                    y_pred[t][indices[t]] -= y_pred_m[t][indices[t]]
                    y_pred[t][indices[t]] += x[t][indices[t], j] * (m_values[t] * phi_clusters[t][j])
            # update the sparsity paramters
            theta = digamma(alp)-digamma(beta)
            for j in idx_choosen:
                if j != self.ndimensions - 1 or not self.intercept:
                    temp = theta.copy()
                    for t in range(self.ntasks):
                        # temp[self.clusters[t]] += (x[t][indices[t], j].dot(np.squeeze(y[t][indices[t]]) - np.clip(sigmoid(y_pred[t][indices[t]]), epsilon, 1 - epsilon)) * m_values[t] + ((m_values[t]**2 * (2 * phi_clusters[t][j] - 1) - sigma[self.clusters[t]][j][self.clustering[self.clusters[t]].index(t), self.clustering[self.clusters[t]].index(t)]) / 8) * sumn (x[t][indices[t], j]**2))
                        temp[self.clusters[t]] += (x[t][indices[t], j].dot(np.squeeze(y[t][indices[t]]) - sigmoid(y_pred[t][indices[t]])) * m[t, j] + ((m[t, j]**2 * (2 * phi_clusters[t][j] - 1) - sigma[self.clusters[t]][j][self.clustering[self.clusters[t]].index(t), self.clustering[self.clusters[t]].index(t)]) / 8) * sumn (x[t][indices[t], j]**2))
                    for index, value in enumerate(temp):
                        phi[index][j] = sigmoid(value)
                    for t in range(self.ntasks):
                        y_pred[t][indices[t]] -= y_pred_m[t][indices[t]]
                        y_pred[t][indices[t]] += x[t][indices[t], j] * (m[t, j] * phi_clusters[t][j])
                        
            # update the hyperparameters
            # alp = alp0 + sumn ([value[:-1] for value in phi], axis=1)
            # beta = beta0 + self.ndimensions - 1 - sumn ([value[:-1] for value in phi], axis=1)
            alp = alp0 + sumn ([value for value in phi], axis=1)
            beta = beta0 + self.ndimensions - 1 - sumn ([value for value in phi], axis=1)
            v = [v0_temp + self.ndimensions for v0_temp in v0]
            V = [inv(V0_temp).copy() for V0_temp in V0]
            for j in idx_choosen:
                for key, value in self.clustering.items():
                    V[key] += np.outer(m[value, j], m[value, j]) + sigma[key][j]

            # update the sample selection for next bagging 
            if self.bagging:
                indices = [rng.choice(nsample, size=max_sz, replace=(nsample != max_sz)) for nsample in self.nsamples]

            # compute the y_pred, y_tilde for the next round of samples 
            for t in range(self.ntasks):
                y_pred[t][indices[t]] = x[t][indices[t], :] @ (m[t, :] * phi_clusters[t])
                y_pred_prev[t][indices[t]] = deepcopy(y_pred[t][indices[t]])
                y_tilde[t, :] = np.squeeze((x[t][indices[t], :].T @ y[t][indices[t]])) * phi_clusters[t]
        return alp, beta, v, V, phi, m, sigma


    def feature_importance(self):
        """ Compute/Extract feature importance from the trained model
            and store it as a pd.Series object.
        Args:
            None
        Returns:
            pd.Series with feature importance values.
        """
        dfs = list()
        for t in range(self.m.shape[0]):
            if self.intercept:
                importance = self.m[t,:-1]*self.phi[self.clusters[t]][:-1] * np.sign(self.m[t, :-1])
            else:
                importance = self.m[t,:]*self.phi[self.clusters[t]] * np.sign(self.m[t, :])
            dfs.append(pd.Series(importance, index=self.feature_names))
        return dfs
    
    def return_support(self):
        support_score = 0
        for (key,phi) in enumerate(self.phi):
            supprt_per_task = phi*sumn (self.m[self.clustering[key],:],axis=0)
            support_score += sumn (supprt_per_task>1e-10)
        support_score = support_score/len(self.phi)
        return support_score

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.alp0 = params['alp0']
        self.beta0 = params['beta0']
        self.v0 = params['v0']
        if params['V0']=="diagonal":
            self.V0 = list()
            for key in sorted(self.clustering.keys()):
              self.V0.append(np.eye(len(self.clustering[key])))
        elif params["V0"]=="weighted":
            norm_vec = self.n/(sumn (self.n)/self.ntasks)
            self.V0 = list()
            for key in sorted(self.clustering.keys()):
              self.V0.append(np.diag(norm_vec[self.clustering[key]]))
        else:
            raise ValueError("invalid hyperparameter value for V0")
        if isinstance(self.alp0,list) or isinstance(self.alp0,np.ndarray):
            if isinstance(self.alp0,list):
                self.alp0 = np.array(self.alp0)
                self.beta0 = np.array(self.beta0)
            self.phi = [self.alp0[j]/(self.alp0[j]+self.beta0[j])*np.ones(self.ndimensions) for j in range(len(self.clustering))]
        else:
            self.phi = [self.alp0/(self.alp0+self.beta0)*np.ones(self.ndimensions) for _ in range(len(self.clustering))]
        if self.intercept:
            for t in range(len(self.clustering)):
                self.phi[t][-1] = 1
        return

    def get_params(self):
        """ Return hyper-parameters used in the execution.

        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {
               'alp0': self.alp0,
               'beta0': self.beta0,
               'v0': self.v,
               'V0': self.V0,
               }
        return ret

    def return_params(self):
        """ Return all parameters used in the execution.
        
        Return:
            params (dict): dict containing the params values.
        """
        ret = {
               'alp0': self.alp0,
               'beta0': self.beta0,
               'alp': self.beta,
               'beta': self.alp,
               'v0': self.v,
               'V': self.V0,
               'v': self.v,
               'V': self.V,
               'phi': self.phi,
               'm': self.m,
               'sigma': self.sigma
               }
        return ret

    def load_params(self, params):
        """load previously trained parameters to be used in the execution.

        Args:
            params (dict): dict with parameter values.
        """
        self.alp0 = params['alp0']
        self.beta0 = params['beta0']
        self.alp = params['alp']
        self.beta = params['beta']
        self.v0 = params['v0']
        self.V0 = params['V0']
        self.v = params['v']
        self.V = params['V']
        self.phi = params['phi']
        self.m = params['m']
        self.sigma = params['sigma']
        self.theta = self.alp/(self.alp+self.beta)
        return

    def get_hyperparameter_space(self):
        """Default hyperparameter values 
        """
        params = {
               'alp0': 1,
               'beta0': 1,
               'v0': 1,
               'V0': "diagonal"
        }
        return params

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())

if __name__ == '__main__':
    ''''
    # Simple use case 
        There are two settings to generate sample size:
            1. Poisson distribution with mean N
            2. Negative binomial distribution with parameters
                rho (probability of success) and r(numberr of failure till stop)
        note:
            -poisson distribution mean=var=N
            -negative_binomial
                mean = r*(1-rho)/rho
                var = r*(1-rho)/rho^2
    '''''
    modes = {"Pois", "NB"}

    parser = argparse.ArgumentParser(description='The data distribution of choice')
    parser.add_argument('--d', type=str, required=True, help='The data distribution for the data Pois for Poisson and NB for negative binomial')
    args = parser.parse_args()

    d = 20 # feature dimension
    T = 10 # number of tasks
    rho = 0.025
    r = 2
    N = 1+r*(1-rho)/rho # sample size, poisson parameter
    p = 0.9 # sparsity bernoulli paramter

    rng = default_rng(1)

    A = rng.random((T, T))
    _, U = np.linalg.eigh(A.T@A)
    A = U.T@U

    (v, V) = (T, A)  #hyper prior for weights covariance matrix

    if args.d == "Pois":
        n = rng.poisson(N, T) # sample sizes for different tasks
    elif args.d == "NB":
        n = 1+rng.negative_binomial(r, rho, T)
    else:
        raise ValueError("Unexpected distributiohn name: %s" % args.d)
    
    x_train = [rng.multivariate_normal(np.zeros(d),np.diag(np.ones(d)),size=j) for _,j in enumerate(n)]

    beta_vec = rng.binomial(1,p,size=d) # sparsity of weights

    Sigma = wishart.rvs(df=v, scale=V, random_state=rng)
    w = rng.multivariate_normal(np.zeros(T), Sigma, size=d).T

    y_train = [rng.binomial(n=1,p=sigmoid(x_train[t]@(beta_vec* w[t, :]))) for t in range(T)]

    n_test = rng.poisson(N, T) # sample sizes for test, this is balanced dataset
    x_test = [rng.multivariate_normal(np.zeros(d),np.diag(np.ones(d)),size=j) for _,j in enumerate(n_test)]
    y_test = [rng.binomial(n=1,p=sigmoid(x_test[t]@(beta_vec* w[t, :]))) for t in range(T)]


    method = BayesMTLClassifier(normalize_data=False)
    method.set_mode('train')
    method.init_with_data(x_train,y_train, intercept=False)
    params = {'alp0': 1, 
              'beta0': 1,
              'v0': 1,
              "V0": "diagonal"
              }
    method.set_params(params=params)
    method.fit(x_train,y_train,rng=default_rng(0),intercept=False)
    method.set_mode('test')

    y_hat = method.predict(x_train)
    acc=0
    acc_pt = np.empty(T)
    for t in range(T):
        acc_pt[t] = sumn(np.squeeze(y_hat[t])==np.squeeze(y_train[t]))/n[t]
        acc +=sumn (np.squeeze(y_hat[t])==np.squeeze(y_train[t]))
    acc /= sumn (n)
    print("training accuracy per task is: {}".format(acc_pt))
    print("training accuracy is: {}".format(acc))
    print("training accuracy per task sorted by sample size per task is: {}".format(acc_pt[np.argsort(n)]))

    y_hat = method.predict(x_test)
    acc=0
    acc_pt = np.empty(T)
    for t in range(T):
        acc_pt[t] = sumn (np.squeeze(y_hat[t])==np.squeeze(y_test[t]))/n_test[t]
        acc +=sumn (np.squeeze(y_hat[t])==np.squeeze(y_test[t]))
    acc /= sumn (n_test)
    print("test accuracy per task is: {}".format(acc_pt))
    print("test accuracy is: {}".format(acc))
    print("test accuracy per task sorted by number of training samples is: {}".format(acc_pt[np.argsort(n)]))