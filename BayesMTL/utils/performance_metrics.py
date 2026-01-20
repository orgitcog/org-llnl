#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Dictionary of performance metrics. Any of these metrics can be used
    in Experiment.

@author: widemann1, goncalves1
"""
# import matplotlib.pyplot as plt
# import sklearn
# from sklearn import metrics
# from lifelines.utils import concordance_index
# from utils.censoring import inverse_probability_of_censoring_weights

import numpy as np
from sklearn.metrics import average_precision_score, \
                            precision_score, recall_score, \
                            roc_curve, auc, accuracy_score, \
                            f1_score, balanced_accuracy_score, \
                            matthews_corrcoef, fbeta_score

f2_score = lambda y_true, y_pred: fbeta_score(y_true, y_pred,beta=2)

def mcc(y_pred, y_true, **kwargs):
    """ Compute the Matthews correlation coefficient (MCC) """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return matthews_corrcoef(y_true, y_pred)


def balanced_accuracy(y_pred, y_true, **kwargs):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    return balanced_accuracy_score(y_true, y_pred)


def area_under_curve(y_pred, y_true, **kwargs):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)


def f1score(y_pred, y_true, **kwargs):
    average_precision = f1_score(y_true, y_pred)
    return average_precision

def f2score(y_pred, y_true, **kwargs):
    average_precision = f2_score(y_true, y_pred)
    return average_precision

def precision(y_pred,y_true,**kwargs):
    return precision_score(y_true, y_pred)

def recall(y_pred,y_true,**kwargs):
    return recall_score(y_true, y_pred)

def avg_precision(y_pred,y_true,**kwargs):
    average_precision = average_precision_score(y_true, y_pred)
    return average_precision



def rmse(y_pred, y_true, **kwargs):
    """ Compute Root Mean Squared Error."""
    # prepare input data to make sure they have the same dimension
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    return np.sqrt(((y_pred-y_true)**2).mean())


def nmse(y_pred, y_true, **kwargs):
    """ Compute Normalized-MSE.
    The normalized mean squared error (NMSE), which is defined as the
    MSE divided by the variance of the ground truth.
    See paper: https://arxiv.org/pdf/1206.4601.pdf
    """
    # prepare input data to make sure they have the same dimension
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    return ((y_pred-y_true)**2).mean() / np.var(y_true)


def accuracy(y_pred, y_true, **kwargs):
    """ Compute classification accuracy. """
    # prepare input data to make sure they have the same dimension
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    return accuracy_score(y_true, y_pred)


def accuracy_per_class(y_pred, y_true, **kwargs):
    """ Compute classification accuracy per class. """
    # prepare input data to make sure they have the same dimension
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    set_y = set(y_true)  # this should sort too - unique classes
    y_pred = np.round(y_pred).astype(int)  # just to make sure it's int
    out = []
    # for each class compute accuracy
    for c in set_y:
        inds = np.where(y_true == c)
        acc = accuracy(y_pred[inds], y_true[inds])
        out.append((c, acc))
    return out


def rmse_survival(y_pred, y_true, **kwargs):
    """ Compute Root Mean Squared Error."""
    # prepare input data to make sure they have the same dimension
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    d = np.array(kwargs['censor_flag'].astype(np.int8).ravel())  # censor flag
    #return np.sqrt( (((d == 1).astype(float)*(y_pred-y_true)**2)).mean()  )
    return np.sqrt(  ((y_pred[d == 1]-y_true[d == 1])**2).mean() )


def mse_survival(y_pred, y_true, **kwargs):
    """ Compute MSE. """
    # prepare input data to make sure they have the same dimension
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    d = np.array(kwargs['censor_flag'].astype(np.int8).ravel())  # censor flag
    #return ((d == 1).astype(float)*(y_pred-y_true)**2).mean()
    return ((y_pred[d == 1]-y_true[d == 1])**2).mean()


def c_index_ours(y_pred, y_true, **kwargs):
    """  Compute a concordance-index.
    The c-index is a measure of accuracy similar to the area under the
    ROC (AUC). It is computed by assessing relative risks (orderings of
    survival times across patients) where comparisons are restricted based
    on censoring. See paper http://dmkd.cs.vt.edu/papers/CSUR17.pdf for details

    " It can be interpreted as the fraction of all pairs of
      subjects whose predicted survival times are correctly ordered
      among all subjects that can actually be ordered."
      This quote is from https://papers.nips.cc/paper/3375-on-ranking-in-survival-analysis-bounds-on-the-concordance-index.pdf

    """
    # prepare input data to make sure they have the same dimension
    d = np.array(kwargs['censor_flag'].astype(np.int8).ravel())  # censor flag
    v = np.array(kwargs['survival_time'].astype(np.float32).ravel())  # survival time

    numerator_denominator = np.sum([  np.sum([  [float(yi_pred < yj_pred),1.]  for yj_pred in y_pred[v > vi] ], axis=0) for vi, yi_pred, yi_true in zip(v[d==1], y_pred[d==1], y_true[d==1]) ], axis=0)
    C = numerator_denominator[0]/numerator_denominator[1]

    return C


def mae_survival(y_pred, y_true, **kwargs):
    """ Compute Mean Absolute Error for uncensored data. """
    # prepare input data to make sure they have the same dimension
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    d = np.array(kwargs['censor_flag'].astype(np.int8).ravel())  # censor flag
    #return ((d == 1).astype(float) * np.abs(y_pred-y_true)).mean()
    return ( np.abs(y_pred[d == 1]-y_true[d == 1])).mean()
