# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:14:43 2023

@author: azhar2
"""
import pandas as pd
import numpy as np

def for_log_vars(df,df_columns):
    """
    Input: 
        df: main dataframe
        df_columns: additional dataframe that contains metadata for features (columns)
        
    Output:
        df_transformed: returns the data with lognormalized columns for the specified 
            columns (information in df_columns dataframe)
    """
    df_transformed = pd.DataFrame()
    
    def individual_log(value):
        if value == 0.0:
            return 0
        else:
            return np.log(value)
    
    for col in df.columns:
        
        try:
            row = list(df_columns.loc[df_columns['variable']==col]['Distribution'])[0]
        except IndexError:
            df.drop([col],axis=1)
            print(f"{col} dropped!")
            continue
      
        if row == 'Log':
            df_transformed[col] = df[col].apply(individual_log)
        else:
            df_transformed[col] = df[col].copy()
    
    print('Lognormalized!')
            
    return df_transformed
