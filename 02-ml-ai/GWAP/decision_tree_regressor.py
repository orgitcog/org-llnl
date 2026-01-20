# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:30:42 2023

@author: azhar2
"""
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import pandas as pd
import datetime


def decision_tree_regressor(X_train,X_test,y_train,y_test,regressor='DecisionTree',feature_imp=True,hyperparameter=True,plot_tree=False,param_grid={},save_scatter=True,random_state=37,save_results=False):
    """
    Runs the decision tree regressor function from sklearn

    This function takes in train and test datasets and, either hyperparameter tunes the model followed by extracting the best model, or uses the default parameters (from sklearn) or uses userdefined params to declare the model and then trains and fits both train and test sets and computes the RMSE, R^2 values and has the functionality of computing feature importance plots, plots tree and returns the best model and the feature importance df. 

    Parameters:
    ----------

    X_train: Pandas DataFrame
        The training dataframe! Should not contain nan values
    
    X_test: Pandas DataFrame
        Test dataframe! Should not contain nan values
    
    y_train: Pandas Series
        Training labels! Should not contain nan values
    
    y_test: Pandas Series
        Testing labels! Should not contain nan values 
    
    regressor: str
        'DecisionTree' or 'GradientBoosting' or 'DecisionTreePruned' or 'DecisionTreeBaggingRegressor' or 'RandomForest' specifies the type of model to use (type: str)

    plot_tree: boolean
        boolean value set as True which plots the decision tree
    
    hyperparameter: boolean
        boolean variable, which if true, will perform hyperparameter tuning otherwise use the default params! 
    
    feature_imp: boolean
        boolean variable, which if set to true, will plot the feature importance plot and save the figure as well (filename specified by the timestamp)
        
    param_grid: dict
        user defined parameters if hyperparameter tune is False. Should be a non-empty dictionary with, each parameter defined as a single element list
        e.g., param_grid ={'max_depth':[None],
                            'min_samples_split':[None],
                            'max_depth = [5]
                            ...
                            }
    
    save_scatter : boolean
        Proceeds to plot the scatter plot on the test and train set predictions!

    save_results : boolean
        Proceeds to save the results in .csv format
                            
    Returns
        -------
        
        best_model: 
            returns the model in this function
        
        cv_results_df:
            cross-validation result with error bars (standard deviations)

        feature_importance_df: 
            returns the dataframe of the feature importance 

        all_tree_importances
            returns a list of all the tree importances in the n_estimators 

    """
    base_path = '../Result Plots/' # define path for saving figure 


    if regressor == 'DecisionTree':
        if hyperparameter:
            print('Performing Decision Tree Regression with hyper-parameter tuning and cross-validation (5): ... ')
            param_grid = {
                        'criterion':['squared_error'],
                        'splitter':['best'],
                        'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,20,None],
                        'min_samples_split':[2,5,10,15,20],
                        'min_samples_leaf': [1,5,10,20],
                        'max_features' : [None],
                        'random_state' : [random_state]
                        }
        else: 
            print('Performing Decison Tree Regression using user-defined hyper-parameters and cross-validation (5): ...')
            if not bool(param_grid):
                param_grid= { 
                            'criterion': ['squared_error'], 
                            'max_depth': [8], 
                            'max_features': [None], 
                            'min_samples_leaf': [20], 
                            'min_samples_split': [2], 
                            'splitter': ['best']
                            }
        estimator = DecisionTreeRegressor(random_state=random_state)
    elif regressor == 'DecisionTreePruned':
        print('Performing Decision Tree Regression via Pruning: ...')
        estimator = DecisionTreeRegressor(random_state=random_state)
    elif regressor == 'DecisionTreeBaggingRegressor':
        estimator_base = DecisionTreeRegressor(criterion="squared_error",max_depth=None,random_state=37) 
        if hyperparameter:
            print('Performing Bagging Regression with hyper-parameter tuning and cross-validation (5): ... ')
            param_grid = { #params added on 08/31/23
                'estimator' : [estimator_base],
                'n_estimators' : [1,10,20,50,100,200,500,1000,2000,3000],
                'max_samples' : [1.0],
                'bootstrap' : [True],
                'n_jobs' : [-1],
                'random_state' : [random_state], # fixing it for reproducibility
                'verbose' : [0], # play around with this afterwards
                }
        else: 
            print('Performing Decision Tree Bagging Regressor using user-defined hyper-parameters and cross-validation (5) ...')
            if not bool(param_grid):
                param_grid = {'estimator':[estimator_base],'n_estimators':[3000],'max_samples':[1.0],'n_jobs':[-1],'random_state':[random_state],'verbose':[0]} #change n_estimators to 3000
        estimator = BaggingRegressor()
    elif regressor == 'RandomForest':
        estimator_base = DecisionTreeRegressor(criterion="squared_error",max_depth=None,random_state=random_state) 
        if hyperparameter:
            print('Performing Random Forest Decision Tree Regression with hyper-parameter tuning and cross-validation (5): ... ')
            param_grid = { 
                'n_estimators' : [1,10,20,50,100,20,500,1000,2000],
                'criterion' : ['squared_error'],
                'max_depth' : [None],
                'max_features' : [1/3],
                'bootstrap' : [True],
                'n_jobs' : [-1],
                'random_state' : [random_state],
                'verbose' : [0]
                }
        else:
            print('Performing Random Forest Decision Tree Regression with user-defined hyper-parameters and cross-validation(5): ...')
            if not bool(param_grid):
                param_grid = {
                    'bootstrap': [True], 
                    'criterion': ['squared_error'], 
                    'max_depth': [None], 
                    'max_features': [0.3333333333333333], 
                    'n_estimators': [2000], 
                    'n_jobs': [-1], 
                    'random_state': [random_state], 
                    'verbose': [0]
                    }
        estimator = RandomForestRegressor()
    elif regressor == 'ADABoost':
        if hyperparameter:
            print('Performing AdaBoost Regression with hyper-parameter tuning and cross-validation (5): \n')
        
            #Note: AdaBoost does not intrinsically require hyperparameter tuning however, for this piece of code, we want to tinker with the number of estimators (to observe overfitting since subsequent trees in boosting increases overfitting)

            estimator_base = DecisionTreeRegressor(criterion='squared_error',max_depth=6,random_state=37) #max depth of 3 is used here but should be tinkered with! 

            param_grid = {
                'estimator' : [estimator_base],
                'n_estimators' : [1,10,20,50,100,200,500,1000],
                'learning_rate' : [0.1,1,10,50],
                'random_state':[random_state]
            }
        else:
            print('Performing adaboost regression using user-defined hyper-parameters and cross-validation (5): ...')
            if not bool(param_grid):
                param_grid = {'n_estimators':[50],'random_state':[random_state]}
        estimator = AdaBoostRegressor()
    elif regressor == 'GradientBoosting':
        if hyperparameter:
            print('Performing Gradient Boosting Regression with hyper-parameter tuning and cross-validation (5): ... ')
            param_grid = {
                        'loss':['squared_error'],
                        'learning_rate':[0.05,0.1,0.5,1],
                        'n_estimators': [50, 100, 200, 500,1000,2000],
                        'max_leaf_nodes': [2,5,10,30],
                        'subsample':[0.5],
                        'verbose':[0],
                        'random_state':[random_state]                        
                        } 
        else:
            print('Performing Gradient Boosting Regression using user-defined hyper-parameters and cross-validation (5): ...')
            if not bool(param_grid):
                param_grid = {
                            'learning_rate':[0.1],
                            'max_depth': [3],
                            'min_samples_split':[2],
                            'random_state':[random_state]
                            }
        estimator = GradientBoostingRegressor()
    if regressor != 'DecisionTreePruned':    
        if regressor != 'DecisionTree':
            print('Building an ensemble learning model using cross-validation \n')
        else:
            print('Building a decision tree using pre-pruning steps i.e., cross validation \n')

        GridSearch = GridSearchCV(estimator,param_grid,cv=5,return_train_score=True,n_jobs=-1,scoring='neg_mean_squared_error',error_score=0) #hyperparameter tuning using parallelism 
    
        GridSearch.fit(X_train,np.ravel(y_train))
        cv_results_df = pd.DataFrame(GridSearch.cv_results_)
        cv_results_df['mean_test_score'] = cv_results_df['mean_test_score'] * -1 
        cv_results_df['mean_train_score'] = cv_results_df['mean_train_score'] * -1

        best_params = GridSearch.best_params_
        best_score = GridSearch.best_score_
        best_model = GridSearch.best_estimator_
        
        print(f"Best parameters: {best_params}")
        print(f"Best score (GridSearchCV.best_score_): {best_score}")
    else:
        print('Building an overfitted decision tree and performing ccp_alpha pruning \n')

        # splitting X_train into two splits: train and validation 
        X_train_1, X_validation, y_train_1, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

        path = estimator.cost_complexity_pruning_path(X_train_1,y_train_1)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        
        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
        ax.set_xlabel("effective alpha")
        ax.set_ylabel("total impurity of leaves")
        ax.set_title("Total Impurity vs effective alpha for training set")  
        #fig.show()

        ## next, we train a decision tree using the effective alphas. The last value in the ccp_alphas is the alpha value that prunes the whole tree, leaving the tree clfs[-1] with one node
        ## code reference: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeRegressor(random_state=random_state,ccp_alpha=ccp_alpha)
            clf.fit(X_train_1,y_train_1)
            clfs.append(clf)
        print(f"Number of nodes in the last tree is {clfs[-1].tree_.node_count} with ccp_alpha as {ccp_alphas[-1]}")

        # removing the last ccp_alpha value as that refers to the tree stump only!
        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        if save_scatter:
            node_counts = [clf.tree_.node_count for clf in clfs]
            depth = [clf.tree_.max_depth for clf in clfs]
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
            ax[0].set_xlabel("alpha")
            ax[0].set_ylabel("number of nodes")
            ax[0].set_title("Number of nodes vs alpha")
            ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
            ax[1].set_xlabel("alpha")
            ax[1].set_ylabel("depth of tree")
            ax[1].set_title("Depth vs alpha")
            fig.tight_layout()
            #fig.show()

        train_scores,test_scores = [],[]
        min_test = 100

        for clf in clfs:
            
            y_pred_train = clf.predict(X_train_1)
            rmse_train = np.sqrt(mean_squared_error(y_pred_train,y_train_1))
            y_pred_test = clf.predict(X_validation)
            rmse_test = np.sqrt(mean_squared_error(y_pred_test,y_validation))
            train_scores.append(rmse_train)
            test_scores.append(rmse_test)
            
            if rmse_test < min_test:
                best_model = clf
                min_test = rmse_test
        print(clf)
            
        if save_scatter:
            fig, ax = plt.subplots()
            ax.set_xlabel("alpha")
            ax.set_ylabel("rmse")
            ax.set_title("Rmse vs alpha for training and testing sets")
            ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
            ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
            ax.legend()
            #plt.show()

        cv_results_df = None

    best_model.fit(X_train, y_train)

    timestamp = datetime.datetime.now()
    year,month,day,hour,min,sec = timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second

    if plot_tree:
        if regressor == 'DecisionTree' or 'DecisionTreePruned':
            print(best_model)
            print('plotting tree: ... ')
            plt.figure(figsize=(10,10))
            sklearn.tree.plot_tree(best_model,max_depth=None,feature_names=list(X_train.columns))
            path = base_path+'Tree Plot'
            plt.savefig(f"{path}_{regressor}_{year}_{month}_{day}_{hour}_{min}_{sec}.png",dpi=1080)
            
    y_pred_train = best_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print("Train RMSE:", rmse)
    r2 = r2_score(y_train, y_pred_train)
    print("Train R-squared Score:", r2)

    if save_scatter:
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(10,10))
        path = base_path+'Train_Scatter'
        x = range(-10,80)
        y = x
        plt.figure()
        plt.plot(x,y,label='y=x')
        plt.scatter(y_train,y_pred_train,marker='.',alpha=0.5)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model\'s Predicted Output')
        plt.title(f"Train Data {regressor} -> RMSE: {np.round(rmse,2)}")
        plt.savefig(f"{path}_{regressor}_{year}_{month}_{day}_{hour}_{min}_{sec}.png",dpi=1080,bbox_inches='tight')
        
    if y_test is not None:
        y_pred_test = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print("Test RMSE:", rmse)
        r2 = r2_score(y_test, y_pred_test)
        print("Test R-squared Score:", r2)
    
    if save_scatter and y_test is not None:
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(10,10))
        path = base_path+'Test_Scatter'
        x = range(-10,80)
        y = x
        plt.figure()
        plt.plot(x,y,label='y=x')
        plt.scatter(y_test,y_pred_test,marker='.',alpha=0.5)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model\'s Predicted Output')
        plt.title(f"{regressor} -> RMSE: {np.round(rmse,2)}")
        plt.savefig(f"{path}_{regressor}_{year}_{month}_{day}_{hour}_{min}_{sec}.png",dpi=1080,bbox_inches='tight')

    feature_importance_df = None
    all_tree_importances = None
    feature_importance_df_orig = None

    if feature_imp: # Plotting Feature Importance
        plt.figure(figsize=(10,10))
        if regressor == 'DecisionTreeBaggingRegressor':
            all_tree_importances = []

            for tree in best_model.estimators_:
                importances = tree.feature_importances_
                all_tree_importances.append(importances)
            plt.rcParams.update({'font.size': 15})
            average_importances = np.mean(all_tree_importances, axis=0)
            std_importances = np.std(all_tree_importances,axis=0)#/np.sqrt(len(all_tree_importances))
            feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': average_importances,'Deviations':std_importances})
            feature_importance_df_orig = feature_importance_df.copy()
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
            feature_importance_df = feature_importance_df.head(12)
            plt.figure(figsize=[20,10])
            plt.title("Feature Importances")
            plt.bar(range(feature_importance_df.shape[0]), feature_importance_df['Importance'], align="center",color='gray')
            plt.errorbar(range(feature_importance_df.shape[0]),feature_importance_df['Importance'],yerr=feature_importance_df['Deviations'],fmt="o",ecolor='black')
            plt.xticks(range(feature_importance_df.shape[0]), feature_importance_df['Feature'], rotation=90)
            plt.ylabel('Relative Importance')
            #plt.xlim([-1, feature_importance_df.shape[0]])
            path = base_path+'Feature_Importance'
            plt.savefig(f"{path}_{regressor}_{year}_{month}_{day}_{hour}_{min}_{sec}.png",bbox_inches='tight',dpi=1080) 
            
        elif regressor == 'DecisionTree':
            importance = best_model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
            feature_importance_df_orig = feature_importance_df.copy()
            feature_importance_df = feature_importance_df.head(10)
            plt.rcParams.update({'font.size': 15})
            plt.figure(figsize=[20,10])
            plt.title("Feature Importances")
            plt.bar(range(feature_importance_df.shape[0]), feature_importance_df['Importance'], align="center",color='gray')
            plt.xticks(range(feature_importance_df.shape[0]), feature_importance_df['Feature'], rotation=90)
            plt.ylabel('Relative Importance')
            #plt.xlim([-1, feature_importance_df.shape[0]])
            path = base_path+'Feature_Importance'
            plt.savefig(f"{path}_{regressor}_{year}_{month}_{day}_{hour}_{min}_{sec}.png",bbox_inches='tight',dpi=1080)             
            all_tree_importances = None


    # saving the results:
    if save_results:
        save_path = '../Result Files/'
    
        df = pd.DataFrame.from_dict({'Train Ground Truth':list(y_train['age']), 'Train Predictions':list(y_pred_train)})
        df.to_csv(f"{save_path}_{regressor}_Train_Predictions.csv")

        if y_test is not None:
            df = pd.DataFrame.from_dict({'Test Ground Truth':list(y_test), 'Test Predictions':list(y_pred_test)})
            df.to_csv(f"{save_path}_{regressor}_Test_Predictions.csv")

        if feature_imp:
            print(f'feature_importance_df is: {feature_importance_df_orig}')
            feature_importance_df_orig.to_csv(f"{save_path}_{regressor}_Feature_Importances.csv")
        else:
            pass
    
    output = {
        'best_model':best_model,
        'cv_results_df':cv_results_df,
        'feature_importance_df':feature_importance_df, # top 10 ranked feature importances 
        'feature_importance_df_orig':feature_importance_df_orig, # all the entries of feature importances
        'all_tree_importances':all_tree_importances
    }

    return output

