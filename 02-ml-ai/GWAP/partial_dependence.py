from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from matplotlib.collections import LineCollection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


 

def partial_dependence_plot(output_regressor, X_train,y_train,plot=True,classifier=False,df_columns={},scaled=True):
    """
    This function computes the Partial Dependence Plots for a trained input model. 


    Parameters:
    ----------
    output_regressor : dictionary 
        Dictionary containing key value pairs as returned from the decision_trees_regressor.py script file 

    X_train : Pandas DataFrame 
        The Train dataset 

    y_train : Pandas DataFrame 
        The Labels for the train dataset 
    
    plot : boolean 
        If true, this plots the top 12 feature importance plots on one figure 

    classifier : boolean
        If true, sets the y-scale limit from 0 to 1 (binary values)
    
    df_columns : Pandas DataFrame 
        meta dataframe of the data as returned by importing_files.py script 

    scaled : boolean 
        If true, normalizes the scale on the subplot y-axis for comparable trends for all features 

    
    Returns: (not currently being used in the main_notebook.ipynb file )
    ----------
    grid_values : list 
        The feature values returned by the partial dependence function

    averages : list 
        The corressponding average ages for the feature values in the grid_values 

    """

    best_model = output_regressor['best_model']
    feature_importance_df = output_regressor['feature_importance_df']
    feature_importance_df_orig = output_regressor['feature_importance_df_orig']
    grid_resolution = X_train.shape[0] # number of unique values set to max number of train samples (useful for decile 
    plt.rcParams.update({'font.size': 8})
    if plot:
        features = list(feature_importance_df['Feature'])
        n_rows = 4
        n_cols = 3

        check = 0
        fig, axs = plt.subplots(n_rows,n_cols,figsize=(10,10))
        fig.tight_layout(pad=5.0) 

        grid_values = []
        averages = []
        
        for row in range(n_rows):
            for col in range(n_cols):
                temp = row+col

                feature = features[temp+check]  
                print(feature)
                feature_index = [X_train.columns.get_loc(feature)]

                results = partial_dependence(best_model, X_train, feature_index,grid_resolution=grid_resolution)
                grid_values.append(results['grid_values'])
                averages.append(results['average'])

                df_temp = pd.DataFrame.from_dict({'grid_values':results['grid_values'][0], 'averages':results['average'][0]})
                df_temp['Quantile Rank'] = pd.qcut(df_temp['grid_values'],10,labels=False)

                global_max = df_temp['grid_values'].max()
                global_min = df_temp['grid_values'].min()
                normalization_factor = global_max-global_min

                decile_groups = df_temp.groupby('Quantile Rank')
                max_values = decile_groups['grid_values'].max()
                min_values = decile_groups['grid_values'].min()
                sample_counts = decile_groups.size()

                bin_size = sample_counts/((max_values - min_values)/normalization_factor)
                df_temp['bin_size'] = (df_temp['Quantile Rank'].map(bin_size)/bin_size.max())*10

                X = df_temp['grid_values']
                Y = df_temp['averages']
                lwidths = df_temp['bin_size']
                points = np.array([X, Y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, linewidths=lwidths,color='gray',linestyle='-')
                
                # if scaled:
                #     if y_train.max().item() == 75.0:
                #         axs[row,col].vlines(min_values,ymin = 20,ymax = 55,linewidth=0.5)
                #     else:
                #         axs[row,col].vlines(min_values,ymin = 10,ymax = 55,linewidth=0.5)
                # else:
                axs[row,col].vlines(min_values,ymin = df_temp['averages'].min()-1,ymax = df_temp['averages'].max()+1,linewidth=0.5)

                axs[row,col].add_collection(lc)

                distribution_scale = df_columns.loc[df_columns['Feature Name']==feature]['Distribution'].item()
                
                if distribution_scale == 'Log':
                    axs[row,col].set_xscale("log")
                
                axs[row,col].set_xlim(min_values.min()-1,max_values.max()+1)

                if classifier:
                    axs[row,col].set_ylim(0,1)
                else:
                    if scaled:
                        if y_train.max().item() == 75.0:
                            axs[row,col].set_ylim(20,55)
                        else:
                            axs[row,col].set_ylim(10,55)
                    else:
                        axs[row,col].set_ylim(df_temp['averages'].min()-1,df_temp['averages'].max()+1)

                axs[row,col].set_xlabel(feature)
                axs[row,col].set_ylabel('Average Age (years)')

                # axs[row,col].plot(results['grid_values'][0],results['average'][0])
                # axs[row,col].set_xlabel(feature)
                # axs[row,col].set_ylabel('Averages (years)')

            check += 2
        
        fig.suptitle('Partial Dependence Plots')
        fig.tight_layout()
        if classifier:
            plt.savefig('../Result Plots/PDP_Classifier.png',dpi=1080)
        else:
            if y_train.max().item() == 75.0:
                plt.savefig('../Result Plots/PDP_Regression_all_ages.png',dpi=1080)
            else:
                plt.savefig('../Result Plots/PDP_Regression_young_ages.png',dpi=1080)
        fig.show()
    else:
        features = list(feature_importance_df_orig['Feature'])
        run_iter = 1

        #features = ['O18O16RAT','depth_or_bottom_m','SO4','gm_top_depth_of_screen_ft','MN','CA','K_sat','distance_to_stream','NA (not nan)','gwe','MG','AS']

        for feature in features:
            feature_index = [X_train.columns.get_loc(feature)] # extract the feature of interest 
            results = partial_dependence(best_model, X_train, feature_index,grid_resolution=grid_resolution) # computes the results
            print(feature)
            
            df_temp = pd.DataFrame.from_dict({'grid_values':results['grid_values'][0], 'averages':results['average'][0]})
            df_temp['Quantile Rank'] = pd.qcut(df_temp['grid_values'],10,labels=False)

            global_max = df_temp['grid_values'].max()
            global_min = df_temp['grid_values'].min()
            normalization_factor = global_max-global_min

            decile_groups = df_temp.groupby('Quantile Rank')
            max_values = decile_groups['grid_values'].max()
            min_values = decile_groups['grid_values'].min()
            sample_counts = decile_groups.size()

            bin_size = sample_counts/((max_values - min_values)/normalization_factor)

            df_temp['bin_size'] = (df_temp['Quantile Rank'].map(bin_size)/bin_size.max())*10

            #df_temp.to_excel(f"/Users/azhara001/Documents/MAR Project/Paper Drafts/Partial_Dependence_Data/Quant_Regression all samples_{feature}_.xlsx",index=False) # remove _ afterwards! 
            print(run_iter)
            run_iter += 1

            X = df_temp['grid_values']
            Y = df_temp['averages']
            lwidths = df_temp['bin_size']
            points = np.array([X, Y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, linewidths=lwidths,color='gray',linestyle='-')
            
            fig,a = plt.subplots()
            a.vlines(min_values,ymin = df_temp['averages'].min()-1,ymax = df_temp['averages'].max()+1,linewidth=0.5)
            a.add_collection(lc)
            a.set_xlim(min_values.min()-1,max_values.max()+1)
            a.set_ylim(df_temp['averages'].min()-1,df_temp['averages'].max()+1)
            a.set_xlabel(feature)
            a.set_ylabel('Average Age (years)')

            plt.savefig(f'../Result Plots/PDP/PDP{feature}.png',dpi=920)
  
            grid_values = None
            averages = None
            


    output = {
            'grid_values':grid_values,
            'averages':averages
    }
    return output

                
    

                


            