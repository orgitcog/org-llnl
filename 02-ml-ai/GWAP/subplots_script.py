# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 07:52:07 2023

@author: azhar2
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#plt.rcParams.update({'font.size': 5})
plt.rcParams.update({'font.size': 12})


def script_subplots(path_train="../Result Files/Train_Model_Performances.xlsx",path_test="../Result Files/Test_Model_Performances.xlsx",train=True):
    """
    Script to generate subplots for the various regression approaches tested in this paper. 

    Parameters:
    -----------
    path_train: str
        path file that contains the DataFrame (.xlsx format) for the Train Model Performances 

    path_test: str
        Path file that contains the DataFrame (.xlsx format) for the Test Model Performances

    train: boolean
        Boolean variable that determines the path file name for the plot to be saved 

    Returns:
    --------
    None
    """

    df_train = pd.read_excel(path_train)
    df_groundtruth_train = df_train['groundtruth']
    df_train.drop(labels=['groundtruth'],axis=1,inplace=True)

    df_test = pd.read_excel(path_test)
    df_groundtruth_test = df_test['groundtruth']
    df_test.drop(labels=['groundtruth'],axis=1,inplace=True)
    
    n_plots = df_train.shape[1]
    n_cols = 2
    n_rows = (n_plots+1)//n_cols

    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(n_rows,n_cols,figsize=(10,10))
    fig.tight_layout(pad=5.0) 
    check = 0

    print(f"Train DF shape: {df_train.shape}")
    print(f"Test DF shape: {df_test.shape}")

    titles = [
        'a',
        'b',
        'c',
        'd',
        'e',
        'f'
        ]

    labels = ['Train','Test']

    for row in range(n_rows):
        for col in range(n_cols):
            temp = row+col
            
            x_ = range(-10,80)
            y_ = x_
            axs[row,col].plot(x_,y_,label='y=x',color='black',linewidth=0.5)

            y_pred_train = df_train.iloc[:,temp+check]
            y_pred_test = df_test.iloc[:,temp+check]

            rmse_train = np.round(np.sqrt(mean_squared_error(df_groundtruth_train, y_pred_train)),decimals=2)
            rmse_test = np.round(np.sqrt(mean_squared_error(df_groundtruth_test, y_pred_test)),decimals=2)
            
            axs[row,col].scatter(df_groundtruth_test,y_pred_test,marker='o',alpha=1.0,s=1,color='teal',label='Test')
            axs[row,col].scatter(df_groundtruth_train,y_pred_train,marker='.',alpha=0.7,s=0.5,color='slategray',label='Train')

            axs[row,col].set_title(titles[0])
            #axs[row,col].set_title(f"{y.name} RMSE: {rmse}")
            axs[row,col].set_xlim([-10,80])
            axs[row,col].set_ylim([-10,80])
            axs[row,col].set_xlabel('GroundTruth')
            axs[row,col].set_ylabel('Prediction')
            axs[row,col].set_aspect('equal', 'box')
            #axs[row,col].legend()
            titles.pop(0)
            
              
        check += 1
    
    lines = [] 
    labels = [] 
    
    for ax in fig.axes: 
        Line, Label = ax.get_legend_handles_labels() 
        lines.extend(Line) 
        labels.extend(Label) 
    fig.legend(Line, Label,fontsize='15',markerscale=3) 

    if train:
        fig.suptitle('Overall Model Performances')
   
        fig.tight_layout()
        plt.savefig('Training.png',dpi=1080)
    else:
        fig.suptitle('Test Set Performance')
        fig.tight_layout()
        plt.savefig('Test.png',dpi=1080)
    return None
    

def train_test_subplot(train_set,test_set,regressor='DecisionTreeBaggingRegressor'):
    """
    """

    y_train_groundtruth = train_set['groundtruth'] 
    y_train = train_set[regressor]

    y_test_groundtruth = test_set['groundtruth']
    y_test = test_set[regressor]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    #fig, axs = plt.subplots(1,2)
    # fig.tight_layout(pad=5.0) 

    x_ = range(-10,80)
    y_ = x_

    rmse_train = np.round(np.sqrt(mean_squared_error(y_train_groundtruth, y_train)),decimals=2)
    rmse_test = np.round(np.sqrt(mean_squared_error(y_test_groundtruth, y_test)),decimals=2)

    ax1.plot(x_,y_,label='y=x', color='gray',linewidth=0.5)
    ax1.scatter(y_train_groundtruth,y_train,marker='o',alpha=0.5,s=1,color='slategray')
    ax1.set_title(f"Train RMSE: {rmse_train}")
    ax1.set_xlim([-10,80])
    ax1.set_ylim([-10,80])
    ax1.set_xlabel('GroundTruth')
    ax1.set_ylabel('Prediction')
    ax1.set_aspect('equal', 'box')

    ax2.plot(x_,y_,label='y=x',color='gray',linewidth=0.5)
    ax2.scatter(y_test_groundtruth,y_test,marker='o',alpha=0.5,s=1,color='teal')
    ax2.set_title(f"Test RMSE: {rmse_test}")
    ax2.set_xlim([-10,80])
    ax2.set_ylim([-10,80])
    ax2.set_xlabel('GroundTruth')
    ax2.set_ylabel('Prediction')
    ax2.set_aspect('equal', 'box')


    fig.suptitle(f"{regressor} model performance train and test")
    fig.tight_layout()
    save_path = f"{regressor} train_test.png"
    plt.savefig(save_path,dpi=1080)

#train_test_subplot(train_set,test_set,regressor='DecisionTreeBaggingRegressor')



def rmse_bucketing_code(path="/Users/azhara001/Documents/MAR Project/Paper Drafts/Datasets/DecisionTreeBaggingRegressor_Bucketing_RMSEs.xlsx"):
    df = pd.read_excel(path)
    print(df)

    train_rmse = df['Train_RMSE']
    test_rmse = df['Test_RMSE']
    x = [0,20,40,60,80]

    plt.plot(x,train_rmse,label='Train RMSE')
    plt.plot(x,test_rmse,label='Test RMSE')
    train_ref = float(train_rmse.tail(1))
    test_ref = float(test_rmse.tail(1))
    plt.axhline(y = train_ref, color = 'blue', linestyle = '--',label='Final Train RMSE',xmin=0,xmax=8) 
    plt.axhline(y = test_ref, color = 'r', linestyle = '--',label='Final Test RMSE',xmin=0,xmax=8) 
    plt.xlabel('Number of features x10')
    plt.ylabel('RMSE (years)')

    plt.legend()
    plt.savefig('Leave one bucket out RMSE.png',dpi=800)
    plt.show()

