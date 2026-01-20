#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""
import os
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import plot
import preprocess
import utils


def tune_logreg(trial, X_train, X_test, y_train, y_test):
    alpha = trial.suggest_loguniform('alpha', 1e-4, 1.)
    classifier = SGDClassifier(loss='log', penalty='l2', max_iter=300, alpha=alpha, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)[:, 1]
    score = metrics.average_precision_score(y_test, y_pred)
    return score

def tune_rf(trial, X_train, X_test, y_train, y_test):
    estimator = trial.suggest_int('n_estimators', 10, 50)
    classifier = RandomForestClassifier(n_estimators=estimator)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)[:, 1]
    score = metrics.average_precision_score(y_test, y_pred)
    return score

def tune_xgboost(trial, X_train, X_test, y_train, y_test):
    n_estimators = trial.suggest_int('n_estimators', 10, 150)
    max_depth = trial.suggest_int('max_depth', 1, 5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1.)
    
    classifier = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)[:, 1]
    score = metrics.average_precision_score(y_test, y_pred)
    return score

def tune_GP(trial, X_train, X_test, y_train, y_test):
    length_scale = trial.suggest_loguniform('length_scale', 1e-4, 1.)
    classifier = GaussianProcessClassifier(1.0 * RBF(length_scale))
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)[:, 1]
    score = metrics.average_precision_score(y_test, y_pred)
    return score

def select_impute(method, X_split, y_split, X_eval, y_eval):
    if IMPUTE == 'mean':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'mean')
        X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
    elif IMPUTE == 'median':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'median')
        X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
    elif IMPUTE == 'mice':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'mice')
        X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
    elif IMPUTE == 'mice+smote':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'mice+smote')
        X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
        X_split, y_split = preprocess.oversample(X_split, y_split, strategy='ADASYN', param=0.5)
    else:
        raise NameError('Imputation method not found.')
    return X_split, y_split, X_eval, y_eval
        
def select_classifier(choice, X_split, y_split, X_eval, y_eval):
    study = optuna.create_study(direction='maximize')
    if MODEL == 'logreg':
        study.optimize(lambda trial: tune_logreg(trial, X_split, X_eval, y_split, y_eval), 
                       n_trials=N_TRIALS)
        classifier = SGDClassifier(loss='log', 
                                   penalty='l2', 
                                   max_iter=300, 
                                   random_state=RANDOM_STATE, 
                                   alpha=study.best_params['alpha'])
    elif MODEL == 'rf':
        study.optimize(lambda trial: tune_rf(trial, X_split, X_eval, y_split, y_eval),
                       n_trials=N_TRIALS)
        classifier = RandomForestClassifier(**study.best_params)
    elif MODEL == 'xgboost':
        study.optimize(lambda trial: tune_xgboost(trial, X_split, X_eval, y_split, y_eval),
                       n_trials=N_TRIALS)
        classifier = XGBClassifier(**study.best_params)
    elif MODEL == 'gp':
        study.optimize(lambda trial: tune_GP(trial, X_split, X_eval, y_split, y_eval), n_trials=50)
        classifier = GaussianProcessClassifier(1.0 * RBF(**study.best_params))  
    return classifier

if __name__ == '__main__':
    
    SAVE_DIR = './saved_models/'
    # DATA_VER = 4.1
    DATA_VER = 4.0
    # OUTCOME = 'death'
    OUTCOME = 'vent'
    N_SPLITS = 3
    RANDOM_STATE = 10
    TEST_SIZE = 0.3
    N_TRIALS = 200
    IMPUTE = 'mean'
#    IMPUTE = 'median'
    # IMPUTE = 'mice'
#    MODEL = 'logreg'
#    MODEL = 'rf'
    MODEL = 'xgboost'
#    MODEL = 'gps'
    
    # setup 
    model_dir = os.path.join(SAVE_DIR,
                             '{}-{}'.format(DATA_VER, OUTCOME),
                             '{}_impute{}_nsplit{}_testsize{}_rs{}'.format(MODEL, 
                                                                           IMPUTE,
                                                                           N_SPLITS,
                                                                           TEST_SIZE,
                                                                           RANDOM_STATE))
    os.makedirs(model_dir, exist_ok=True)
    
    # data loading
    col2remove = ["WEIGHT/SCALE", "HEIGHT", '% O2 Sat', 'Insp. O2 Conc.', 'PCO2', 'SPO2', 'PO2', 'R FIO2',
              'diag_A', 'diag_B', 'diag_C', 'diag_D', 'diag_E', 'diag_F', 'diag_G', 'diag_H',
              'diag_I', 'diag_J', 'diag_K', 'diag_L', 'diag_M', 'diag_N', 'diag_O', 'diag_P', 
              'diag_Q', 'diag_R', 'diag_S', 'diag_T', 'diag_U', 'diag_V', 'diag_W', 'diag_X', 
              'diag_Y', 'diag_Z', 'URINE OUTPUT',
#               'gender', 'race', 'Procalcitonin', 'D-dimer', 
             ]
    X, y, df = utils.load_data(DATA_VER, OUTCOME, col2remove)

    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, random_state=RANDOM_STATE, test_size=TEST_SIZE)
    
    # cross validation
    classifiers, scores = [], []
    data_split, data_eval = [], []
    labels_split, labels_eval = [], []
    for idx_split, idx_eval in sss.split(X_train, y_train):
        # data
        X_split, y_split = X_train[idx_split], y_train[idx_split]
        X_eval, y_eval = X_train[idx_eval], y_train[idx_eval]
        X_split, y_split, X_eval, y_eval = select_impute(IMPUTE, X_split, y_split, X_eval, y_eval)
        
        # train
        classifier = select_classifier(MODEL, X_split, y_split, X_eval, y_eval)
        classifier.fit(X_split, y_split)
        pred, score = utils.get_results(classifier, X_eval, y_eval)

        # save for post-process
        data_split.append(X_split)
        data_eval.append(X_eval)
        labels_split.append(y_split)
        labels_eval.append(y_eval)    
        classifiers.append(classifier)
        scores.append(score)
    
    # evaluation
    scores = pd.concat(pd.DataFrame(score, index=[i]) for i, score in enumerate(scores))
    scores.to_csv(os.path.join(model_dir, 'scores_cv.csv'))
    scores_mean, scores_std = scores.mean(0), scores.std(0)
    pd.DataFrame(scores_mean).to_csv(os.path.join(model_dir, 'scores_mean.csv'))
    pd.DataFrame(scores_std).to_csv(os.path.join(model_dir, 'scores_std.csv'))
    
    
    print('scores - cv')
    print(scores)
    print('scores - mean')
    print(scores_mean)
    print('scores - std')
    print(scores_std)
    
    plot.plot_roc(classifiers, data_eval, labels_eval, model_dir, tail='')
    X_train, y_train, X_test, y_test = select_impute(IMPUTE, X_train, y_train, X_test, y_test)
    plot.plot_shap(classifiers, X_test, y_test, df, model_dir, tail='')
    
