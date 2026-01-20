#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Availabe preprocess functions:
    1. impute(X_train, X_test, strategy, seed=42)
        - mean
        - median
        - most_frequent
        - fill_zero
        - bayes_ridge
        - tree_regressor
    
    2. scale(X_train, X_test, strategy)
        - z-score
        - l2
        - minmax
        
    3. oversample(X, y, strategy, param)
        - smote
        - smotenc
        - smote_kmean2 
        - smote_kmeans5
        - smote_kmeans10
        - ADASYN
        
    4. undersample(X, y, strategy, param)
        - random
    
    5. add_noise(X, noise_type, noise_param)
        - gaussian
    
"""
import numpy as np

from imblearn.over_sampling import (
    SMOTE,
    SMOTENC,
    KMeansSMOTE,
    ADASYN
)
from imblearn.under_sampling import RandomUnderSampler

#from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn import preprocessing



    
def impute(X_train, X_test, strategy, seed=42):
    if strategy == 'mean':
        imputer = SimpleImputer(missing_values=np.nan, 
                                strategy='mean')
    elif strategy == 'median':
        imputer = SimpleImputer(missing_values=np.nan, 
                                strategy='median')
    elif strategy == 'most_frequent':
        imputer = SimpleImputer(missing_values=np.nan, 
                                strategy='most_frequent')
    elif strategy == 'fill_zero':
        imputer = SimpleImputer(missing_values=np.nan, 
                                strategy='constant', fill_value=0)
    elif strategy == "mice":
        _estimator = BayesianRidge()
        imputer = IterativeImputer(max_iter=50, 
                                   random_state=seed, 
                                   estimator=_estimator)
    # elif strategy == "tree_regressor":
    #     _estimator = ExtraTreesRegressor(n_estimators=10, 
    #                                      random_state=seed)
    #     imputer = IterativeImputer(max_iter=50, 
    #                                random_state=seed, 
    #                                estimator=_estimator)
    else:
        raise NameError(f'Imputation strategy {strategy} not found.')
    imputer.fit(np.array(X_train))
    X_train_imp = imputer.transform(np.array(X_train))
    X_test_imp = imputer.transform(np.array(X_test))
    return X_train_imp, X_test_imp


def scale(X_train, X_test, strategy):
    if strategy == "z-score":
#         columns_trans = ColumnTransformer(
#                 [('scaler', preprocessing.StandardScaler(), [idx for idx, val in enumerate(numerical) if val == 1])], remainder='passthrough'
#         )
        columns_trans = preprocessing.StandardScaler()
    elif strategy == "l2":
#        columns_trans = ColumnTransformer(
#                [('scaler', preprocessing.Normalizer(), [idx for idx, val in enumerate(numerical) if val == 1])], remainder='passthrough'
#        )
        columns_trans = preprocessing.Normalizer()
    elif strategy == 'minmax':
#        columns_trans = ColumnTransformer(
#                [('scaler', preprocessing.MinMaxScaler(), [idx for idx, val in enumerate(numerical) if val == 1])], remainder='passthrough'
#        )
        columns_trans = preprocessing.MinMaxScaler()
    else:
        raise NameError("scalar strategy not found")
    X_train_scaled = columns_trans.fit_transform(X_train)
    X_test_scaled = columns_trans.fit_transform(X_test)
    return X_train_scaled, X_test_scaled


def oversample(X, y, strategy, param):
    if strategy == 'smote':
        oversample = SMOTE(sampling_strategy=param, random_state=42)
    elif strategy == 'smotenc':
        oversample = SMOTENC([1, 2], sampling_strategy=param, random_state=42)
    elif strategy == 'smote_kmeans2':
        oversample = KMeansSMOTE(sampling_strategy=param, k_neighbors=2, random_state=42)
    elif strategy == 'smote_kmeans5':
        oversample = KMeansSMOTE(sampling_strategy=param, k_neighbors=5, random_state=42)
    elif strategy == 'smote_kmeans10':
        oversample = KMeansSMOTE(sampling_strategy=param, k_neighbors=10, random_state=42)
    elif strategy == "ADASYN":
        oversample = ADASYN(sampling_strategy=param, random_state=42)
    else:
        raise NameError(f'Oversampling strategy {strategy} not found.')
    X_over, y_over = oversample.fit_resample(X, y)
    return X_over, y_over


def undersample(X, y, strategy, param):
    if strategy == 'random':
        undersample = RandomUnderSampler(sampling_strategy=param, random_state=42)
    X_under, y_under = undersample.fit_resample(X, y)
    return X_under, y_under


def add_noise(X, noise_type, noise_param):
    if noise_type == 'gaussian':
        noise = np.random.normal(loc=0, scale=noise_param, size=X.shape)
        print(noise)
    else:
        raise NameError(f'Noise type {noise_type} not found')
    X_noise = X + noise
    return X_noise
