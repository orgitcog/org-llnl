# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:25:04 2023

@author: azhar2
"""
import numpy as np
import pandas as pd

def z_score_norm(df,df_var_types,is_train=True,means=[],stds=[]):
    """
    This function performs z-score-normalization. The idea is to call this funciton separately for train and test dataset
    
    
    Input:
        df: dataframe as input (without z-score-normalized features)
        df_var_types: metadata file containing information on features (columns)
        is_train: Bool variable. If training data is passed into this function, set this as True otherwise False
        means: list which is either empty if train data is passed or contains mean values for each features (if test data is passed)
        stds: list which is either empty if train data is passed or contains std values for each features (if test data is passed)
        
    Output:
        df: dataframe with features z-score normalized
        means: list containing means of each feature
        stds: list containing stds of each feature
        df_reference : reference dataframe with means and stds with column names
    """
    columns = df.columns
    df_reference = pd.DataFrame()

    for index,col in enumerate(columns):
        var_type = df_var_types.loc[df_var_types['variable']==col]['Variable Type'].values[0]
        if var_type == 'Continous':
            if is_train == True:
                mean = df[col].mean()
                std = df[col].std()
                means.append(mean),stds.append(std)
            else:
                mean,std = means[index],stds[index] # normalizing test dataset
        elif var_type == 'GPS':
            #df[col] = np.radians(df[col]) # converts degrees to radians 
            if is_train == True:
                mean,std = df[col].mean(),df[col].std() # computes the mean/std of radian lat,lngs
                means.append(mean),stds.append(std)
            else:
                mean,std = means[index],stds[index]        
        elif var_type == 'Timeseries':
            df[col] = pd.to_numeric(df[col])
            if is_train == True:
                mean,std = df[col].mean(),df[col].std()
                means.append(mean),stds.append(std)
            else:
                mean,std = means[index],stds[index]
        elif var_type == 'Ambiguous (SUBBSN_)':
            temp_df = pd.DataFrame()
            df[col] = df[col].astype(str)
            temp_df[['C1','C2']] = df[col].str.split('-',expand=True)
            df[col] = temp_df['C2'].astype(float)
            del temp_df
            if is_train == True:
                mean = df[col].mean()
                std = df[col].std()
                means.append(mean),stds.append(std)
            else:
                mean,std = means[index],stds[index] # normalizing test dataset
        else:
            #print(f"Not normalizing variable {col} as it is not continous")
            if is_train == True:
                mean,std = 0,1
                means.append(mean),stds.append(std)
            else: 
                continue
            
        try:    
            df[col] = (df[col]-mean)/std
        except:
            print(col)
           
    df_reference['cols'],df_reference['means'],df_reference['stds'] = columns,means,stds
            
        
    return df,means,stds,df_reference
