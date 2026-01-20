#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 00:55:49 2021

@author: chan59
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
import shap
import torch


def plot_roc(classifiers, test_data, test_labels, save_dir, tail=''):
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    for j in range(len(classifiers)):
        viz = plot_roc_curve(classifiers[j], test_data[j], test_labels[j],
                             name='ROC fold {}'.format(j), alpha=0.8, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='random')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'roccurve{}.jpg'.format(tail)))
    
    
def plot_roc_cv(classifiers, train_splits, splits, df, save_dir, tail=''):
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    
    for j, (classifier, split) in enumerate(zip(classifiers, splits)):
        X_test, y_test = split
        
        viz = plot_roc_curve(classifier, X_test, y_test,
                             name='ROC fold {}'.format(j), alpha=0.8, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
            
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='random')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'roccurve{}.jpg'.format(tail)))
    
    
def plot_shap(classifiers, X_test, y_test, df, save_dir, tail=''):
    for j, classifier in enumerate(classifiers):
        mybooster = classifier.get_booster()
        model_bytearray = mybooster.save_raw()[4:]
        def myfun(self=None):
            return model_bytearray
        mybooster.save_raw = myfun
        
        # scatter plot
        explainer = shap.TreeExplainer(mybooster)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(7, 13), dpi=150)
        shap.summary_plot(shap_values, X_test, feature_names=df.columns, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap-scatter-model{}{}.jpg'.format(j, tail)))
        
        # bar plot
        plt.figure(figsize=(7, 13), dpi=150)
        shap.summary_plot(shap_values, 
                          X_test, 
                          feature_names=df.columns,
                          plot_type='bar',
                          show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap-summary-model{}{}.jpg'.format(j, tail)))
        
        
def plot_shap_cv(classifiers, train_splits, splits, df, save_dir, tail=''):
    for j, (classifier, train_split, split) in enumerate(zip(classifiers, train_splits, splits)):
        X_train, y_train= train_split
        X_test, y_test = split
        mybooster = classifier.get_booster()
        model_bytearray = mybooster.save_raw()[4:]
        def myfun(self=None):
            return model_bytearray
        mybooster.save_raw = myfun

        background = X_train  # TODO: take a number of sample instead?
        # scatter plot
        explainer = shap.TreeExplainer(mybooster, background)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(7, 13), dpi=150)
        shap.summary_plot(shap_values, X_test, feature_names=df.columns, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap-scatter-model{}{}.jpg'.format(j, tail)))

        # bar plot
        plt.figure(figsize=(7, 13), dpi=150)
        shap.summary_plot(shap_values, 
                          X_test, 
                          feature_names=df.columns,
                          plot_type='bar',
                          show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap-summary-model{}{}.jpg'.format(j, tail)))

def save_shap_cv(classifiers, train_splits, splits, df, save_dir, tail=''):
    shaps = []
    for j, (classifier, train_split, split) in enumerate(zip(classifiers, train_splits, splits)):
        X_train, y_train= train_split
        X_test, y_test = split
        mybooster = classifier.get_booster()
        model_bytearray = mybooster.save_raw()[4:]
        def myfun(self=None):
            return model_bytearray
        mybooster.save_raw = myfun

        background = X_train  # TODO: take a number of sample instead?
        # scatter plot
        explainer = shap.TreeExplainer(mybooster, background)
        shap_values = explainer.shap_values(X_test)
        shaps.append(shap_values)
        
    save_dct = {'shaps': shaps, 'columns': df.columns}
    torch.save(save_dct, os.path.join(save_dir, 'shap_values.pt'))
