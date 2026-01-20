#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 22:07:17 2021

@author: chan59
"""

import os
import json
import torch
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
#    confusion_matrix, 
#    precision_recall_curve, 
    average_precision_score, 
    recall_score, 
    precision_score,
    roc_auc_score,
#    brier_score_loss,
)

def load_data(version, outcome, skip_columns=[], missing_threshold=1.):
    '''Load data with specified version and outomce. '''
    path = os.path.join('/data/covid-ehr-ldrd/extracted', 
                        f'entrance_admission_v{version}.pt')
    data_dict = torch.load(path)
    data, labels, df = data_dict[outcome]
    if outcome == 'vent_invasive_noninvasive':
        # change 1 to 0 and 2 to 1: noninvasive to negative and invasive to positive
        labels[labels==1] = 0
        labels[labels==2] = 1
        
    print(version, outcome, Counter(labels)) 
    df = df.drop(columns=skip_columns)
    
    # drop columns based on some missing threshold
    missing = df.isna().sum()

    n = data.shape[0]
    missing_drop = []
    for col, count in zip(missing.index, missing.values):
        if (count / n) > missing_threshold:
            missing_drop.append(col)

    print(f"Drop these columns with more missing ratio than {missing_threshold}")
    print(missing_drop)
    df = df.drop(missing_drop, axis=1)
    print("Using columns:")
    print(df.columns)
    
    return df.to_numpy(), labels, df


def get_results(classifier, X, y):
    """Put predictions and scores in dict. """
    preds = {
        'y_true': y.tolist(),
        'y_pred': classifier.predict(X).tolist(),
        'y_prob': classifier.predict_proba(X)[:, 1].tolist()
    }    
    scores = {'accuracy': accuracy_score(preds['y_true'], preds['y_pred']),
              'average_precision': average_precision_score(preds['y_true'], preds['y_prob']),
#               'confusion_matrix': confusion_matrix(preds['y_true'], preds['y_pred']),
              'precision': precision_score(preds['y_true'], preds['y_pred']),
              'recall': recall_score(preds['y_true'], preds['y_pred']),
              'roc_auc': roc_auc_score(preds['y_true'], preds['y_prob'])}
    return preds, scores


def save_dict(model_dir, _dict, filename):
    path = os.path.join(model_dir, filename)
    with open(path, 'w') as f:
        json.dump(_dict, f, indent=2, sort_keys=True)