# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:38:57 2023

@author: azhar2
"""
from separating_with_tiers import separating_with_tiers
from for_log_var import for_log_vars
from imputing_features import imputing_features

from z_score_norm import z_score_norm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def data_prep_imputation_normalizing(df,df_columns,tiers_to_remove=[],drop_missing_labels=True,save_fig=False,lognormal=True,show_distribution=False,z_score=False,impute=True):
    """"
    This function was last updated on 08/22/23 (after a while and high dim visualization including tsne, umap were removed from this function)! if future use is needed, those functions would be incorporated separately. The original version of this file is stored as 'data_prep_imputation_normalizing_obsolete.py'
    
    This function takes as input the dataframe df and the meta dataframe df_columns and does the following:
        - Removes unwanted columns from the dataframe df (defaults to removing groups now as opposed to tiers)
        - performs log-normal transformation (if lognormal == True) (for_log_vars) -> details on which columns require lognormal is stored in df_columns['Distribution']
        - plots feature distribution (feature_space_distribution) (if show_distribution == True)
        - drops missing labels (if drop_missing_labels==True)
        - Splits dataframe into X,Y (labels)
        - Splits the dataset into train-test split (80-20 ratio for now by default)
        - Imputes the dataframes (train and test) (imputing_features)
        - Performs z-score-normalization


    Parameters:
    -----------
    df: Pandas DataFrame
        main datafile 

    df_columns : Pandas DataFrame
        meta deta dataframe

    tiers_to_remove : list (with int elems)
        int values of tiers/groups to be dropped from the dataframe passed as a list e.g., [0,1,...]

    drop_missing_labels : boolean
        Drops entries with missing label values ('age')

    lognormal : boolean
        Computes logarithm of those columns who's distribution is defined as lognormal in df_columns['Distribution']

    z-score : boolean
        Performs z-score normalization on the dataset

    impute : boolean
        Performs imputation (mean/mode) on missing value entries in the dataframe (look at imputing_features() for more details)


    save_fig : boolean
        saves the histograms if show_distribution is True


    Returns:
    --------
    output : dict
        X_train : Train set 
        X_test : Test set 
        y_train : Train labels
        y_test : Test labels 
    
    """
    df_filtered = separating_with_tiers(df,df_columns,tiers=tiers_to_remove,col_name='group') # entries included in the list groups/tiers are the variables removed
    
    if lognormal: df_filtered = for_log_vars(df_filtered,df_columns)    
    
    if drop_missing_labels: 
        df_filtered = df_filtered.dropna(subset='age') 
        print('Missing label samples dropped!')

    Y,X = df_filtered['age'],df_filtered.copy().drop(['age'],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    if impute:
        X_train_imputed, impute_train_vals = imputing_features(X_train,df_columns,method='mean')
        X_test_imputed = X_test.copy()
    
        for index,col in enumerate(X_test_imputed.columns):
            X_test_imputed[col].fillna(impute_train_vals[index],inplace=True)
        print('Returning imputed train-test split')
    else:
        X_test_imputed = X_test.copy()
        X_train_imputed = X_train.copy()
        print('Dataset\'s missing values not imputed')

    if z_score:
        X_train_imputed_norm,means,stds,df_reference= z_score_norm(X_train_imputed,df_columns,is_train=True,means=[],stds=[])
        X_test_imputed_norm = z_score_norm(X_test_imputed,df_columns,is_train=False,means=means,stds=stds)[0]
        print('Z-score normalized')
    else:
        X_train_imputed_norm = X_train_imputed
        X_test_imputed_norm = X_test_imputed
        print('Not z-score normalized')
        

    output = {
         'X_train':X_train_imputed_norm,
         'y_train':y_train,
         'X_test':X_test_imputed_norm,
         'y_test':y_test,
    }

    return output


