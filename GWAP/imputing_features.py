# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:19:25 2023

@author: azhar2
"""

def imputing_features(df_train,df_var_types,method='mean'):
    """
    This function takes as input a non-normalized dataframe and imputes missing values based on the 
    variable type. Returns normalized train dataset as well as the statistics for each column (as a list)
    for imputation on test dataset features (imputation statistics should be computed on train dataset only)
    
    Input: 
        df_train: training main dataset upon which imputation is to be performed
        df_var_types: metadata dataframe including information about features (columns)
        method: string that decides on the method of imputation. Default is mean for continous variables
    
    Output: 
        df_train: output dataframe that has missing values imputed (contains no nan values)
        values: list containing imputed value of all the features in the dataframe. Returned for imputation
                for the test dataset
    """
    columns = df_train.columns
    values = []
    for col in columns:
        try:
            var_type = df_var_types.loc[df_var_types['variable']==col]['Variable Type'].values[0]
        except IndexError:
            df_train.drop([col],axis=1)
            print(f"{col} dropped!")
            continue

        if var_type == 'Continous':
            if method == 'mean':
                value = df_train[col].mean()                
            elif method == 'mode':
                value = df_train[col].mode()[0] #picking the first most occuring value
        else:
            try:
                value = df_train[col].mode()[0]
            except:
                value = 0
        
        df_train[col].fillna(value,inplace=True)
        values.append(value)
    return df_train, values
