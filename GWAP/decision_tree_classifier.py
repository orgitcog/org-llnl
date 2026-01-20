
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:14:32 2023

@author: azhar2
"""

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


def decision_tree_classifier(X_train,X_test,y_train,y_test,classifier='DecisionTreeClassifier',feature_imp=True,hyperparameter=True,plot_tree=False,param_grid={},class_weight=None,save_scatter=True,save_results=False):
    """
    Runs the decision tree classifier function from sklearn

    This function takes in train and test datasets and, either hyperparameter tunes the model followed by extracting the best model, or uses the default parameters (from sklearn) or uses userdefined params to declare the model and then trains and fits both train and test sets and computes the F-1 Score, Recall Score, Precision Score and has the functionality of computing feature importance plots, plots tree and returns the best model and the feature importance df. 

    Parameters:
    ----------

    X_train: Pandas DataFrame
        The training dataframe! Should not contain nan values
    
    X_test: Pandas DataFrame
        Test dataframe! Should not contain nan values
    
    Y_train: Pandas Series
        Binary Training labels! Should not contain nan values
    
    Y_test: Pandas Series
        Binary Testing labels! Should not contain nan values 
    
    classifier : str
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
    
    class_weight : dict
        a dictionary object that specifies the weight given to each class for the classification problem. Defaults to None (sklearn by default). 

    save_scatter : boolean
        Proceeds to plot the scatter plot on the test and train set predictions!

    save_results : boolean
        Proceeds to save all the results in .csv formats for reproducibility 
    
    Returns
        -------
       best_model: 
            returns the model in this function
        
        cv_results_df:
            cross-validation result with error bars (standard deviations)

        feature_importance_df: 
            returns the dataframe of the feature importance 

    """
    base_path = '../Result Plots/' # define path for saving figure 

    # Declaring Classifiers base estimators ... 
    if classifier == 'DecisionTreeClassifier':
        if hyperparameter:
            print('Performing Decision Tree Classification with hyper-parameter tuning and cross-validation (5): ... ')
            param_grid = {
                'criterion' : ['entropy'], # concept of information gain
                'max_depth' : [1,2,3,4,5,6,7,8,9,10,11,12,20,None],
                'min_samples_split' : [2], # letting tree depth dictate this computation
                'random_state' : [37],
                'class_weight' : [class_weight], 
                } 
        else:
            print('Performing Decision Tree Classification using user-defined hyperparameters and cross-validation (5): ...')
            if not bool(param_grid):
                param_grid = {'criterion' : ['entropy'], 
                              'max_depth' : [None], 
                              'random_state':[37],
                              'class_weight':[class_weight]
                              }

        estimator = DecisionTreeClassifier() # declare base estimator
    elif classifier == 'PrunedClassifier':
        estimator = DecisionTreeClassifier()
    elif classifier == 'BaggingClassifier':
        estimator_base = DecisionTreeClassifier(criterion="entropy",max_depth=None,random_state=37,class_weight=class_weight) # declaring a deep decision tree regressor
        if hyperparameter:
            print('Performing Bagging Decision Tree Classification with hyper-parameter tuning and cross-validation (5): ... ')
            param_grid = { #params added on 08/31/23
                'estimator' : [estimator_base],
                'n_estimators' : [1,10,20,50,100,200,500,1000,2000,3000],
                'max_samples' : [1.0],
                'bootstrap' : [True],
                'n_jobs' : [-1],
                'random_state' : [37], # fixing it for reproducibility
                'verbose' : [0] # play around with this afterwards,
                }
        else:
            print('Performing Bagging Decision Tree Classification with user-defined hyper-parameters and cross- validation (5): ...')
            if not bool(param_grid):
                param_grid = {
                    'bootstrap':[True],
                    'estimator' : [estimator_base],
                    'max_samples':[1.0],
                    'n_estimators':[2000],#[2000], #change to 2000
                    'n_jobs':[-1],
                    'random_state':[37],
                    'verbose':[0]
                }
        estimator = BaggingClassifier()
    elif classifier == 'RandomForestClassifier':
        estimator_base = DecisionTreeClassifier(criterion="entropy",max_depth=None,random_state=37) 
        if hyperparameter:
            print('Performing Random Forest Decision Tree Classification with hyper-parameter tuning and cross-validation (5): ... ')
            param_grid = { #params added on 08/31/23}
                'n_estimators' : [1,10,20,50,100,20,500,1000,2000],#,3000], #[500]
                'criterion' : ['entropy'],
                'max_depth' : [None],
                'max_features' : [1/3],#[1/5, 1/4, 1/3, 1/2, 2/3, 3/4, 4/5],#[1/3] fix n_estimator and then pass in this array?
                'bootstrap' : [True],
                'n_jobs' : [-1],
                'random_state' : [37],
                'verbose' : [0],
                'class_weight' : [class_weight]
                }
        else:
            print('Performing Random Forest Decision Tree Classification with user defined hyper-parameters with cross-validation (5): ... ')
            if not bool(param_grid):
                param_grid = {'bootstrap': [True], 
                              'criterion': ['entropy'], 
                              'max_depth': [None], 
                              'max_features': [0.3333333333333333], 
                              'n_estimators': [2000], 
                              'n_jobs':[ -1], 
                              'random_state': [37], 
                              'verbose': [0],
                              'class_weight':[class_weight]
                              } # output of tuned hyperparameters (getting a perfect train score)
        estimator = RandomForestClassifier()
    elif classifier == 'AdaBoostClassifier':
        estimator_base = DecisionTreeClassifier(criterion='entropy',max_depth=6,random_state=37)
        if hyperparameter:
            print('Performing AdaBoost Classification with hyper-parameter tuning and cross-validation (5): ... ')
            param_grid = {
                'estimator' : [estimator_base],
                'n_estimators' : [1,10,20,50,100,200,500,1000],
                'learning_rate' : [0.1,1,10,50],
                'random_state':[37],
                'class_weight' : [class_weight]
            }
        else:
            print('Performing AdaBoost Decision Tree Classifciation with user-defined hyper-parameters and cross-validation (5): ... ')
            if not bool(param_grid):
                param_grid = {'n_estimators':[50]}
        estimator = AdaBoostClassifier()
    elif classifier == 'GradientBoostingClassifier':
        if hyperparameter:
            print('Performing Gradient Boosting Classification with hyper-parameter tuning and cross-validation (5): ... ')
            param_grid = {
                        'loss':['log_loss'],
                        'learning_rate':[0.05,0.1,0.5,1],
                        'n_estimators': [50, 100, 200, 500,1000,2000],
                        'max_leaf_nodes': [2,5,10,30],
                        'subsample':[0.5],
                        'verbose':[1],
                        'class_weight' : [class_weight]                        
                        } 
        else:
            print('Performing Gradient Boosting Classification with user-defined hyper-parameters and cross-validation (5): ...')
            if not bool(param_grid):
                param_grid = {'learning_rate':[0.1],
                              'max_depth': [3],
                              'min_samples_split':[2],
                              'class_weight':[class_weight]
                              }
        estimator = GradientBoostingClassifier()

    # Training the declared Classifier estimators ... 
    if classifier != 'PrunedClassifier':   
        if classifier == 'DecisionTreeClassifier':
            print('Building a decision tree using pre-pruning steps i.e., cross validation \n')
        elif classifier == 'BaggingClassifier' or 'RandomForestClassifier' or 'AdaBoostClassifier' or 'GradientBoostingClassifier':
            print('Building an ensemble learning model using cross-validation \n')

        GridSearch = GridSearchCV(estimator,param_grid,cv=5,return_train_score=True,n_jobs=-1,scoring='f1',error_score=0) #hyperparameter tuning using parallelism 
        GridSearch.fit(X_train,np.ravel(y_train))
        cv_results_df = pd.DataFrame(GridSearch.cv_results_)

        best_params = GridSearch.best_params_
        best_score = GridSearch.best_score_
        best_model = GridSearch.best_estimator_
        
        print(f"Best parameters: {best_params}")
    else:
        print('Building an overfitted decision tree and performing ccp_alpha pruning \n')
        # splitting X_train into two splits: train and validation (required for choosing ccp alpha hyperparameter)
        X_train_1, X_validation, y_train_1, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        path = estimator.cost_complexity_pruning_path(X_train_1,y_train_1)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        
        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
        ax.set_xlabel("effective alpha")
        ax.set_ylabel("total impurity of leaves")
        ax.set_title("Total Impurity vs effective alpha for training set")  
        fig.show()

        ## next, we train a decision tree using the effective alphas. The last value in the ccp_alphas is the alpha value that prunes the whole tree, leaving the tree clfs[-1] with one node
        ## code reference: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=37,ccp_alpha=ccp_alpha)
            clf.fit(X_train_1,y_train_1)
            clfs.append(clf)
        print(f"Number of nodes in the last tree is {clfs[-1].tree_.node_count} with ccp_alpha as {ccp_alphas[-1]}")

        # removing the last ccp_alpha value as that refers to the tree stump only!
        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]
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

        train_scores,test_scores = [],[]
        max_test = 0

        for clf in clfs:
            
            y_pred_train = clf.predict(X_train_1)
            f1_train = f1_score(y_pred_train,y_train_1)
    
            y_pred_test = clf.predict(X_validation)
            f1_test = f1_score(y_pred_test,y_validation)

            train_scores.append(f1_train)
            test_scores.append(f1_test)
            
            if f1_test > max_test:
                best_model = clf
                max_test = f1_test
        
        print(f"Max F-1 Score on Train After Pruning: {max_test}")

        print(clf)
        print(f"Best Model after ccp pruning: {best_model}")
            
        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("F1 Binary")
        ax.set_title("F1 Binary vs alpha for training and testing sets")
        ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.show()

        cv_results_df = None


    best_model.fit(X_train, y_train)

    timestamp = datetime.datetime.now()
    year,month,day,hour,min,sec = timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second

    if plot_tree:
        if classifier == 'DecisionTreeClassifier' or classifier == 'PrunedClassifier': # draws the tree for a single tree approach
            print(best_model)
            tree.plot_tree(best_model,max_depth=None,feature_names=list(X_train.columns))
            path = base_path+'Tree Plot'
            plt.savefig(f"{path}_{classifier}_{year}_{month}_{day}_{hour}_{min}_{sec}.png",dpi=1080)
            

    y_pred_train = best_model.predict(X_train)
    f1_classifier, precision_classifier, recall_classifier = f1_score(y_pred_train,y_train), precision_score(y_pred_train,y_train), recall_score(y_pred_train,y_train)

    print("Train Precision:", precision_classifier)
    print("Train Recall:", recall_classifier)
    print("Train F-1 Score:",f1_classifier)

    if save_scatter:
        plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train,colorbar=False,cmap='gray')
        path = base_path+'Train_Confusion'
        plt.plot()
        plt.title(f"Train Data Confusion Matrix with class_weights: {class_weight}")
        plt.savefig(f"{path}_{classifier}_{year}_{month}_{day}_{hour}_{min}_{sec}.png",dpi=1080)

    if y_test is not None:
        y_pred_test = best_model.predict(X_test)
        f1_classifier, precision_classifier, recall_classifier = f1_score(y_pred_test,y_test), precision_score(y_pred_test,y_test), recall_score(y_pred_test,y_test)

        print("Test Precision:", precision_classifier)
        print("Test Recall:", recall_classifier)
        print("Test F-1 Score:",f1_classifier)

    else:
        pass # no test set passed. 
    
    if save_scatter and y_test is not None:
        plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test,colorbar=False,cmap='gray')
        path = base_path+'Test_Confusion'
        plt.plot()
        plt.title(f"Test Data Confusion Matrix with class_weights: {class_weight}")
        plt.savefig(f"{path}_{classifier}_{year}_{month}_{day}_{hour}_{min}_{sec}.png",dpi=1080)

    if feature_imp and classifier=='BaggingClassifier': # Plotting Feature Importance for Bagging Classifiers

        all_tree_importances = []

        for tree in best_model.estimators_:
            importances = tree.feature_importances_
            all_tree_importances.append(importances)
        plt.rcParams.update({'font.size': 15})
        average_importances = np.mean(all_tree_importances, axis=0)
        std_importances = np.std(all_tree_importances,axis=0)#/np.sqrt(len(all_tree_importances)) # standard deviation / number of trees
        
        feature_importance_df_all = pd.DataFrame({'Feature': X_train.columns, 'Importance': average_importances,'Deviations':std_importances})
        feature_importance_df_all = feature_importance_df_all.sort_values('Importance', ascending=False)
        feature_importance_df = feature_importance_df_all.head(12)
        
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=[20,10])
        plt.title("Feature Importances")
        plt.bar(range(feature_importance_df.shape[0]), feature_importance_df['Importance'], align="center",color='gray')
        plt.errorbar(range(feature_importance_df.shape[0]),feature_importance_df['Importance'],yerr=feature_importance_df['Deviations'],fmt="o",ecolor='black')
        plt.xticks(range(feature_importance_df.shape[0]), feature_importance_df['Feature'], rotation=90)
        plt.ylabel('Relative Importance')
        path = base_path+'Feature_Importance'
        plt.savefig(f"{path}_{classifier}_{year}_{month}_{day}_{hour}_{min}_{sec}.png",bbox_inches='tight',dpi=1080) 
    else:
        feature_importance_df = None 
        feature_importance_df_all = None

    # saving the results:
    if save_results:
        save_path = '../Result Files/'
    
        df = pd.DataFrame.from_dict({'Train Ground Truth':list(y_train['age']), 'Train Predictions':list(y_pred_train)})
        df.to_csv(f"{save_path}_{classifier}_Train_Predictions.csv")

        if y_test is not None:
            df = pd.DataFrame.from_dict({'Test Ground Truth':list(y_test), 'Train Predictions':list(y_pred_test)})
            df.to_csv(f"{save_path}_{classifier}_Test_Predictions.csv")

        feature_importance_df_all.to_csv(f"{save_path}_{classifier}_Feature_Importances.csv")


    output = {
        'best_model':best_model,
        'cv_results_df':cv_results_df,
        'feature_importance_df':feature_importance_df,
        'feature_importance_df_orig' : feature_importance_df_all
    }

    return output



