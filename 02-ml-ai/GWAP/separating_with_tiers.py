# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:16:50 2023

@author: azhar2
"""

def separating_with_tiers(df,df_columns,tiers=[],col_name='tier'):
    """
    Input: 
        df: main dataframe of the dataset
        df_columns: metadata dataframe having information on the features (columns)
        tiers: list containing tiers to be removed [0,1,2,3,'Undefined','Label']
        col_name: string that specifies the column name in the dataframe df_columns from which separation of tiers/groups will take place. 
        
    Output:
        df_filtered: returns a subset of the dataframe df by removing the tiers specified 
            which information is pulled from the df_columns dataframe
        
    """
    to_drop = [] # list of columns to be retained following this filtering 
    
    for tier in tiers:
        to_drop.extend(list(df_columns.loc[df_columns[col_name]==tier]['variable']))
        
    df_filtered = df.copy()

    print(f"Number of columns to be dropped: {len(to_drop)}")

    for elem in to_drop:
        try:
            df_filtered = df_filtered.drop(elem,axis=1)
        except:
            continue # looped since columns of df may vary 
    
    print(f"After removing {col_name}: {tiers} columns, the remaining dataset has: {df_filtered.shape[0]} rows and {df_filtered.shape[1]} columns")
    
    return df_filtered

