#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics

from xgboost import XGBClassifier
import optuna
import torch
import preprocess
import utils

feature_default = ['age', 'gender', 'race', 'sdoh', 'RESPIRATIONS', 'R BMI', 'TEMPERATURE', 'SYSTOLIC BLOOD PRESSURE', 
                    'DIASTOLIC BLOOD PRESSURE', 'PULSE OXIMETRY', 'PULSE',
                    'comor_asthma', 'comor_chronic_kidney_disease', 'comor_diabetes', 'comor_hypertension']

feature_by_cost = {
    1: [['Sodium', 'Chloride', 'Glucose', 'Creatinine', 'BUN', 'Calcium', 'Potassium, Bld'],
        ['Hemoglobin', 'Lymphocytes Absolute', 'MCHC', 'Hematocrit', 'White Blood Cells', 'Platelets', 'MCH',  'MCV', 'MPV']],
    2: [['D-dimer'],
        ['LDH'],
        ['Sed Rate'],
        ['CRP']],
    3: [['BNP'], ['Troponin I'], ['Procalcitonin'], ['Ferritin']] 
    }
features_all = list(itertools.chain.from_iterable(feature_by_cost.values()))

features = [['Sodium', 'Chloride', 'Glucose', 'Creatinine', 'BUN', 'Calcium', 'Potassium, Bld'],
            ['Hemoglobin', 'Lymphocytes Absolute', 'MCHC', 'Hematocrit', 'White Blood Cells', 'Platelets', 'MCH',  'MCV', 'MPV'],
            ['D-dimer'],
            ['LDH'],
            ['Sed Rate'],
            ['CRP'],
            ['BNP'], 
            ['Troponin I'], 
            ['Procalcitonin'], 
            ['Ferritin']]
costs = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3]



def select_impute(method, X_split, y_split, X_eval, y_eval):
    if method == 'mean':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'mean')
        X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
    elif method == 'median':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'median')
        X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
    elif method == 'mice':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'mice')
        X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
    elif method == 'mice+smote':
        X_split, X_eval = preprocess.impute(X_split, X_eval, 'mice')
        X_split, X_eval = preprocess.scale(X_split, X_eval, 'z-score')
        X_split, y_split = preprocess.oversample(X_split, y_split, strategy='ADASYN', param=0.5)
    else:
        raise NameError('Imputation method not found.')
    return X_split, y_split, X_eval, y_eval

def train_classifier_cv(choice, splits, n_trials):
    assert choice == "xgboost", "Only xgboost implemented"
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: tune_xgboost_cv(trial, splits),
                   n_trials=n_trials)
    classifier = XGBClassifier(**study.best_params, scale_pos_weight=SCALE_POS_WEIGHT)
    return classifier, study.best_params

def tune_xgboost_cv(trial, splits):
    # splits: X_train, X_test, y_train, y_test = splits[0]
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
        # X_train, X_test must already be imputed
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        score = metrics.average_precision_score(y_test, y_pred)
        scores.append(score)    
    return np.asarray(scores).mean()


if __name__ == '__main__':

    # Hyperparamter
    hparams = {
        'n_test_split': 5,
        'n_val_split': 4,
        'random_state': 10,
        
        'n_trials': 150,
        
        # permutations
        'max_k': 0,
        'min_k': -1,
        
        'model': 'xgboost',
        'impute': 'median',
        
        'data_ver': '4.0_sdoh_lymphocytesabs',
        'outcome': 'vent'
    }
    # data loading
    if hparams['outcome'] == 'death':
        col2remove = ["WEIGHT/SCALE", "HEIGHT",     #'% O2 Sat', 'Insp. O2 Conc.', 'PCO2', 'SPO2', 'PO2', 'R FIO2',
                  'diag_A', 'diag_B', 'diag_C', 'diag_D', 'diag_E', 'diag_F', 'diag_G', 'diag_H',
                  'diag_I', 'diag_J', 'diag_K', 'diag_L', 'diag_M', 'diag_N', 'diag_O', 'diag_P', 
                  'diag_Q', 'diag_R', 'diag_S', 'diag_T', 'diag_U', 'diag_V', 'diag_W', 'diag_X', 
                  'diag_Y', 'diag_Z', 'URINE OUTPUT',
    #               'gender', 'race', 'Procalcitonin', 'D-dimer', 
                 ]
        assert hparams['data_ver'] == '4.1_sdoh_lymphocytesabs'
        SCALE_POS_WEIGHT = 10.0
        
    elif hparams['outcome'] == 'vent':
        col2remove = ["WEIGHT/SCALE", "HEIGHT", '% O2 Sat', 'Insp. O2 Conc.', 'PCO2', 'SPO2', 'PO2', 'R FIO2',
          'diag_A', 'diag_B', 'diag_C', 'diag_D', 'diag_E', 'diag_F', 'diag_G', 'diag_H',
          'diag_I', 'diag_J', 'diag_K', 'diag_L', 'diag_M', 'diag_N', 'diag_O', 'diag_P', 
          'diag_Q', 'diag_R', 'diag_S', 'diag_T', 'diag_U', 'diag_V', 'diag_W', 'diag_X', 
          'diag_Y', 'diag_Z', 'URINE OUTPUT',
#               'gender', 'race', 'Procalcitonin', 'D-dimer', 
         ]
        
        assert hparams['data_ver'] == '4.0_sdoh_lymphocytesabs'
        SCALE_POS_WEIGHT = 5.0
        
    else:
        raise NameError('No such outcome bruh.')
    
    X, y, df = utils.load_data(hparams['data_ver'], hparams['outcome'], col2remove)
    model_dir = os.path.join(f'./saved_models/budget_final_sdoh_lympabs',
                             f'ver{hparams["data_ver"]}_{hparams["outcome"]}',
                             f'{hparams["model"]}_{hparams["impute"]}')
    os.makedirs(model_dir, exist_ok=True)
    print(model_dir)
    
    # perform n choose k feature selection
    for k in range(hparams['max_k'], hparams['min_k']-1, -1):
        if k == 0:  # selecting nothing
            pool = [(-1)]
        else:
            pool = itertools.combinations(np.arange(len(features)), k)
            
        for idx_select in pool:     

            ## STEP 1: split data into train/test
            sss_traintest = StratifiedKFold(n_splits=hparams['n_test_split'],
                                            shuffle=True,
                                            random_state=hparams['random_state'])
            for fold, (idx_train, idx_test) in enumerate(sss_traintest.split(X, y)):
                print('Running train/test fold:', fold)
                print(hparams)
        
                ## STEP 2: split train into split/eval
                sss_cveval = StratifiedKFold(n_splits=hparams['n_val_split'], 
                                             shuffle=True,
                                             random_state=hparams['random_state'])
                X_train, y_train = X[idx_train], y[idx_train]
                
                ## STEP 3: form splits for CV
                cv_splits = []
                for idx_cv, idx_val in sss_cveval.split(X_train, y_train):  
                    if idx_select != -1:
                        feature_keep = [a for b in np.array(features)[list(idx_select)].ravel().tolist() for a in b]
                    else:
                        feature_keep = []
                    feature_select = feature_default + feature_keep
                    X_cv, y_cv = df[feature_select].to_numpy()[idx_train][idx_cv], y[idx_train][idx_cv]
                    X_val, y_val = df[feature_select].to_numpy()[idx_train][idx_val], y[idx_train][idx_val]
                    X_cv, y_cv, X_eval, y_eval = select_impute(hparams['impute'], X_cv, y_cv, X_val, y_val)
                    cv_splits.append([X_cv, y_cv, X_eval, y_eval])
                

                ## STEP 4: obtain the best hyperparams, train and test model on Train/Test split
                best_classifier, best_params = train_classifier_cv(hparams['model'], cv_splits, hparams['n_trials'])
                X_train, y_train = df[feature_select].to_numpy()[idx_train], y[idx_train]
                X_test, y_test = df[feature_select].to_numpy()[idx_test], y[idx_test]
                X_train, y_train, X_test, y_test = select_impute(hparams['impute'], X_train, y_train, X_test, y_test)
                best_classifier.fit(X_train, y_train)
                pred, score = utils.get_results(best_classifier, X_test, y_test)
                
                ##  STEP 5: save
                if isinstance(idx_select, int):
                    folder_name = f"k{k}_fold{fold}_{idx_select}"
                else:
                    folder_name = f"k{k}_fold{fold}_"+"+".join(map(str, idx_select))
                save_dir = os.path.join(model_dir, folder_name)
                os.makedirs(save_dir, exist_ok=True)
                utils.save_dict(save_dir, pred, 'pred.json')
                utils.save_dict(save_dir, score, 'score.json')
                utils.save_dict(save_dir, best_params, 'best_params.json')
                utils.save_dict(save_dir, {'index': np.int32(idx_select).tolist(),
                                           'features': feature_select}, 'features.json')
                utils.save_dict(save_dir, hparams, 'hyperparams.json')
                indices = {'traintest': list(sss_traintest.split(X, y)),
                           'cveval': list(sss_cveval.split(X_train, y_train))}
                torch.save(indices, os.path.join(save_dir, 'index.pt'))
