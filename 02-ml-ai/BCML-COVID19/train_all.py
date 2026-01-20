#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""
import argparse
import os
import pandas as pd
import numpy as np
import optuna
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

import plot
import preprocess
import utils



def tune_logreg(trial, splits):
    alpha = trial.suggest_loguniform('alpha', 1e-4, 1.)
    classifier = SGDClassifier(loss='log', penalty='l2', max_iter=300, alpha=alpha, random_state=10)
    scores = []
    for split in splits:
        X_train, y_train, X_test, y_test = split
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        score = metrics.average_precision_score(y_test, y_pred)
        scores.append(score)    
    return np.asarray(scores).mean()

def tune_rf(trial, splits):
    estimator = trial.suggest_int('n_estimators', 10, 50)
    classifier = RandomForestClassifier(n_estimators=estimator)
    scores = []
    for split in splits:
        X_train, y_train, X_test, y_test = split
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        score = metrics.average_precision_score(y_test, y_pred)
        scores.append(score)    
    return np.asarray(scores).mean()

def tune_xgboost(trial, splits):
    n_estimators = trial.suggest_int('n_estimators', 30, 200)
    max_depth = trial.suggest_int('max_depth', 1, 15)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1.)
    tree_method = trial.suggest_categorical('tree_method', ['auto', 'exact'])
    reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-4, 10.)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-1, 10.)
    min_split_loss = trial.suggest_loguniform('min_split_loss', 1e-4, 10.)
    subsample = trial.suggest_uniform('subsample', 0.1, 1.)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.1, 1.)

    classifier = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, tree_method=tree_method, scale_pos_weight=SCALE_POS_WEIGHT, reg_alpha=reg_alpha, reg_lambda=reg_lambda, min_split_loss=min_split_loss, subsample=subsample, colsample_bytree=colsample_bytree)
    scores = []
    for split in splits:
        X_train, y_train, X_test, y_test = split
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        #score = metrics.average_precision_score(y_test, y_pred)
        # ROC AUC score
        score = metrics.roc_auc_score(y_test, y_pred)
        scores.append(score)    
    return np.asarray(scores).mean()

def tune_GP(trial, splits):
    length_scale = trial.suggest_loguniform('length_scale', 1e-4, 1.)
    classifier = GaussianProcessClassifier(1.0 * RBF(length_scale))
    scores = []
    for split in splits:
        X_train, y_train, X_test, y_test = split
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        score = metrics.average_precision_score(y_test, y_pred)
        scores.append(score)    
    return np.asarray(scores).mean()

def tune_MLP(trial, splits):
    alpha = trial.suggest_loguniform('alpha', 1e-4, 1.0)
    learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-4, 1.0)
    classifier = MLPClassifier(hidden_layer_sizes=[32, 32], 
                               alpha=alpha,
                               learning_rate_init=learning_rate_init,
                               random_state=10)
    scores = []
    for split in splits:
        X_train, y_train, X_test, y_test = split
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        score = metrics.average_precision_score(y_test, y_pred)
        scores.append(score)    
    return np.asarray(scores).mean()

def tune_xgboostrf(trial, splits):
    n_estimators = trial.suggest_int('n_estimators', 10, 150)
    max_depth = trial.suggest_int('max_depth', 1, 5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1.)

    classifier = XGBRFClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    scores = []
    for split in splits:
        X_train, y_train, X_test, y_test = split
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        score = metrics.average_precision_score(y_test, y_pred)
        scores.append(score)    
    return np.asarray(scores).mean()

def select_impute(method, X_split, y_split, X_eval, y_eval, scale=True):
    if method == 'mean':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'mean')
        if scale:
            X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
    elif method == 'median':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'median')
        if scale:
            X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
    elif method == 'mice':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'mice')
        if scale:
            X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
    elif method == 'mice+smote':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'mice')
        if scale:
            X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
        X_split, y_split = preprocess.oversample(X_split, y_split, strategy='ADASYN', param=0.5)
    else:
        raise NameError('Imputation method not found.')
    return X_split, y_split, X_eval, y_eval
        

def train_classifier(choice, splits, n_trials, random_state):
    study = optuna.create_study(direction='maximize')
    if choice == 'logreg':
        study.optimize(lambda trial: tune_logreg(trial, splits), 
                       n_trials=n_trials)
        classifier = SGDClassifier(loss='log', 
                                   penalty='l2', 
                                   max_iter=300,
                                   alpha=study.best_params['alpha'],
                                   random_state=10)
    elif choice == 'rf':
        study.optimize(lambda trial: tune_rf(trial, splits),
                       n_trials=n_trials)
        classifier = RandomForestClassifier(**study.best_params)
    elif choice == 'xgboost':
        study.optimize(lambda trial: tune_xgboost(trial, splits),
                       n_trials=n_trials)
        classifier = XGBClassifier(**study.best_params, scale_pos_weight=SCALE_POS_WEIGHT)
    elif choice == 'gp':
        study.optimize(lambda trial: tune_GP(trial, splits), n_trials=n_trials)
        classifier = GaussianProcessClassifier(1.0 * RBF(**study.best_params))  
    elif choice == 'mlp':
        study.optimize(lambda trial: tune_MLP(trial, splits), n_trials=n_trials)
        classifier = MLPClassifier(hidden_layer_sizes=[32, 32], **study.best_params)
    elif choice == 'xgboostrf':
        study.optimize(lambda trial: tune_xgboost(trial, splits),
                       n_trials=n_trials)
        classifier = XGBRFClassifier(**study.best_params)
    return classifier, study.best_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True,
                        help='data version.')
    parser.add_argument('--outcome', type=str, required=True,
                        help='prediction outcome.')
    parser.add_argument('--model', type=str, required=True,
                        help='choice of model.')
    parser.add_argument('--impute', type=str, required=True,
                        help='type of imputation method.')
    parser.add_argument('--scale', type=int, default=1,
                        help='whether to use scale or not')
    parser.add_argument('--n_test_splits', type=int, default=5,
                        help='number of train/test splits; outer cross validation loop.')
    parser.add_argument('--n_val_splits', type=int, default=4,
                        help='number of cv/val splits; inner cross validation loop.')
    parser.add_argument('--n_trials', type=int, default=150,
                        help='number of trails for optuna hyperparamter tuning.')
    parser.add_argument('--random_state', type=int, default=10,
                        help='seed for randomizing experiments.')
    parser.add_argument('--save_dir', type=str, default='./saved_models/all_final/',
                        help='base directory for saving model.')
    args = parser.parse_args()
    

    
    # data loading
    if args.outcome == 'death':
        col2remove = ["WEIGHT/SCALE", "HEIGHT",     #'% O2 Sat', 'Insp. O2 Conc.', 'PCO2', 'SPO2', 'PO2', 'R FIO2',
                  'diag_A', 'diag_B', 'diag_C', 'diag_D', 'diag_E', 'diag_F', 'diag_G', 'diag_H',
                  'diag_I', 'diag_J', 'diag_K', 'diag_L', 'diag_M', 'diag_N', 'diag_O', 'diag_P', 
                  'diag_Q', 'diag_R', 'diag_S', 'diag_T', 'diag_U', 'diag_V', 'diag_W', 'diag_X', 
                  'diag_Y', 'diag_Z', 'URINE OUTPUT',
    #               'gender', 'race', 'Procalcitonin', 'D-dimer', 
                 ]
        
        SCALE_POS_WEIGHT = 10.0
    elif args.outcome == 'vent':
        col2remove = ["WEIGHT/SCALE", "HEIGHT", '% O2 Sat', 'Insp. O2 Conc.', 'PCO2', 'SPO2', 'PO2', 'R FIO2',
          'diag_A', 'diag_B', 'diag_C', 'diag_D', 'diag_E', 'diag_F', 'diag_G', 'diag_H',
          'diag_I', 'diag_J', 'diag_K', 'diag_L', 'diag_M', 'diag_N', 'diag_O', 'diag_P', 
          'diag_Q', 'diag_R', 'diag_S', 'diag_T', 'diag_U', 'diag_V', 'diag_W', 'diag_X', 
          'diag_Y', 'diag_Z', 'URINE OUTPUT',
#               'gender', 'race', 'Procalcitonin', 'D-dimer', 
         ]
        
        SCALE_POS_WEIGHT = 5.0
    else:
        raise NameError('No such outcome bruh.')
        
    # setup 
    model_dir = os.path.join(args.save_dir, f'{args.version}_{args.outcome}', f'{args.model}_{args.impute}_ntestsplits{args.n_test_splits}_nvalsplits{args.n_val_splits}_ntrials{args.n_trials}_randomstate{args.random_state}_spw{SCALE_POS_WEIGHT}_auc')
    os.makedirs(model_dir, exist_ok=True)
    
    ## Data
    X, y, df = utils.load_data(args.version, args.outcome, col2remove)
    
    ## Training
    classifiers, scores = [], []
    data_split, data_eval = [], []
    labels_split, labels_eval = [], []
    ## STEP 1: split data into train/test
    sss_traintest = StratifiedKFold(n_splits=args.n_test_splits,
                                    shuffle=True,
                                    random_state=args.random_state)
    for fold, (idx_train, idx_test) in enumerate(sss_traintest.split(X, y)):
        sss_cveval = StratifiedKFold(n_splits=args.n_val_splits, random_state=args.random_state, shuffle=True)
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        
        ## STEP 2: split training data into cv/val
        splits = []
        for idx_cv, idx_val in sss_cveval.split(X_train, y_train):
            # data
            X_cv, y_cv = X_train[idx_cv], y_train[idx_cv]
            X_val, y_val = X_train[idx_val], y_train[idx_val]
            X_cv, y_cv, X_val, y_val = select_impute(args.impute, X_cv, y_cv, X_val, y_val, scale=args.scale)
            splits.append((X_cv, y_cv, X_val, y_val))
  
        ## STEP 3: train the best parameters
        print(f"############################")
        print(f"##### FOLDD: {fold} ########")
        print(f"############################")
        best_classifier, best_params = train_classifier(args.model, splits, args.n_trials, args.random_state)
        X_train, y_train, X_test, y_test = select_impute(args.impute, X_train, y_train, X_test, y_test, scale=args.scale)
        best_classifier.fit(X_train, y_train)
        pred, score = utils.get_results(best_classifier, X_test, y_test)

        # save for post-process
        data_split.append(X_train)
        data_eval.append(X_test)
        labels_split.append(y_train)
        labels_eval.append(y_test)    
        classifiers.append(best_classifier)
        scores.append(score)
        
        
    # evaluation
    scores = pd.concat(pd.DataFrame(score, index=[i]) for i, score in enumerate(scores))
    scores.to_csv(os.path.join(model_dir, 'scores_cv.csv'))
    scores_mean, scores_std = scores.mean(0), scores.std(0)
    pd.DataFrame(scores_mean).to_csv(os.path.join(model_dir, 'scores_mean.csv'))
    pd.DataFrame(scores_std).to_csv(os.path.join(model_dir, 'scores_std.csv'))
    utils.save_dict(model_dir, best_params, 'best_params.json')
    utils.save_dict(model_dir, vars(args), 'hparams.json')
    indices = {'traintest': list(sss_traintest.split(X, y)),
               'cveval': list(sss_cveval.split(X_train, y_train))}
    torch.save(indices, os.path.join(model_dir, 'index.pt'))

    print('scores - cv')
    print(scores)
    print('scores - mean')
    print(scores_mean)
    print('scores - std')
    print(scores_std)
    
    # Put into splits and plot
    test_splits = []
    for data, label in zip(data_eval, labels_eval):
        test_splits.append((data, label))
    # Include train data for background (SHAP)
    train_splits = []
    for data, label in zip(data_split, labels_split):
        train_splits.append((data, label))
        
        
    try:
        plot.plot_roc_cv(classifiers, train_splits, test_splits, df, model_dir, tail='')
    
    except:
        print("Model doesn't support ROC curve.")
        
    try:
        plot.save_shap_cv(classifiers, train_splits, test_splits, df, model_dir, tail='')
    except:
        print("Model doesn't support SHAP.")
        