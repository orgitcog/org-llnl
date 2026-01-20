import os

import numpy as np
from numpy.random import default_rng

from scipy.stats import wishart
from scipy.special import expit

clip = np.clip 

def sigmoid(a, epsilon=1e-15):
    """ sigmoid function for logistic regression
    
    inputs:
        a: logit inputs 
        epsilon: overflow-underflow tolerance 
    """
    return clip(expit(a),epsilon, 1 - epsilon)

def generate_data(d=100, T=10, N=24, n_run=10, data_types=["balanced","unbalanced"], data_subtypes = ["Ultra sparse","Sparse","Dense"], theta_values = [0.05, 0.2, 0.8], path_data_source="simulated"):
  """ function to generate the synthetic data 

  inputs: 
    d: feature dimension
    T: number of tasks
    N: Expected sample sizes per task-6 (subtracted by 6)
    n_run:  Numer of runs per setting
    data_types: balanced vs unbalanced
    data_subtypes: different sparsity structures
    theta_values: different overall sparsity parameters
    path_data_source: source directory of the data
  """

  for data_type in data_types:
    path_data = os.path.join(path_data_source, data_type)
    os.makedirs(path_data, exist_ok=True)
    for seed,(data_subtype,p) in enumerate(zip(data_subtypes,theta_values)):
      path_data_subtype = os.path.join(path_data, data_subtype)
      os.makedirs(path_data_subtype, exist_ok=True)
      rng = default_rng(seed)

      if data_type=="balanced":
        n = 16+rng.poisson(N, T) # sample sizes for different tasks
      else:
        rho = 0.04
        r = 1
        n = 16+rng.negative_binomial(r, rho, T)

      # legacy to generate balanced_large 
      # n = 976+rng.poisson(N, T) # sample sizes for different tasks
      for run in range(n_run):
        A = rng.random((T, T))
        _, U = np.linalg.eigh(A.T@A)
        A = U.T@U

        (v, V) = (T, A) #hyper prior for weights covariance matrix

        data_x_train = [rng.multivariate_normal(np.zeros(d),np.diag(np.ones(d)),size=j) for _,j in enumerate(n)]
        data_x_test = [rng.multivariate_normal(np.zeros(d),np.diag(np.ones(d)),size=j) for _,j in enumerate(n)]

        beta_vec = rng.binomial(1,p,size=d) # sparsity of weights

        Sigma = wishart.rvs(df=v, scale=V, random_state=rng)
        w = rng.multivariate_normal(np.zeros(T), Sigma, size=d).T

        data_y_train = [rng.binomial(n=1,p=sigmoid(data_x_train[t]@(beta_vec* w[t, :]))) for t in range(T)]
        data_y_test = [rng.binomial(n=1,p=sigmoid(data_x_test[t]@(beta_vec* w[t, :]))) for t in range(T)]

        np.save(os.path.join(path_data_subtype,"beta_vec{}.npy".format(run)), beta_vec)
        np.save(os.path.join(path_data_subtype,"weight{}.npy".format(run)), w)
        np.save(os.path.join(path_data_subtype,"Sigma{}.npy".format(run)), Sigma)
        np.save(os.path.join(path_data_subtype,"V{}.npy".format(run)), V)
        np.save(os.path.join(path_data_subtype,"data_x_train{}.npy".format(run)), np.array(data_x_train,dtype=object))
        np.save(os.path.join(path_data_subtype,"data_y_train{}.npy".format(run)), np.array(data_y_train,dtype=object))
        np.save(os.path.join(path_data_subtype,"data_x_test{}.npy".format(run)), np.array(data_x_test,dtype=object))
        np.save(os.path.join(path_data_subtype,"data_y_test{}.npy".format(run)), np.array(data_y_test,dtype=object))

  return 

if __name__ == '__main__':

  generate_data()