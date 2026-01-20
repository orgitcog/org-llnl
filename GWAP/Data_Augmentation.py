import torch
import pandas as pd
#import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

#import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import mean_squared_error
#from sklearn.inspection import PartialDependenceDisplay
#from sklearn.metrics import accuracy_score

from data_prep_imputation_normalizing import data_prep_imputation_normalizing
from data_utils import FlexibleDataset

def augment_data(input_dict={},augmented_size=5,specialized=False,z_score=False,log_normal=True,cols_to_retain=[],impute=True,binary_split=70,plot_dist_split=False):
    """
    This function takes in the output (dict) returned from the importing_files() function and pre-processes it (removes certain columns), then calls another function that further pre-processes the data (imputation, normalization), splits the dataset into train and test followed by loading the dataset as a PyTorch Tensor (for augmentation). 
    Returns an output dictionary of train and test splits (post-processed). 

    The output of this function feeds directly into a function for modelling! 

    Parameters: 
    -----------

    input_dict: dict
        contains a dictionary output from the function importing_files()

    augmented_sized: int
        the number of times you want to augment your dataset (only if df_unc is in input_dict)

    specialized: boolean
        drops ['He4_ter','NGRT','DNe'] if True

    z_score: boolean
        performs z-score normalization if True 

    log_normal : boolean 
        input given to the function data_prep_imputation_normalizing() which performs lognormal computation on columns defined as log-normal in df_columns['Distribution']

    cols_to_retain: list
        only retains the columns passed in this list for further pre-processing. If the list is empty, the code doesn't perform this action! 

    impute : boolean 
        input given passed to the data_prep_imputation_normalizing() function. If True, imputes the missing values in the DataFrame. Observe data_prep_imputation_normalizing()'s docs for more details on imputation

    binary_split: int
        creates a binary split column of the age label for binary classification. This value determines the split. 

    plot_dist_split: boolean
        plots the train-test split of the age distribution in the dataset


    Returns:
    -----------

    output : dict
        X_train : Pandas DataFrame 
            Train dataset
        X_test : Pandas DataFrame 
            Test dataset 
        y_train : Pandas Series 
            Train labels 
        y_train_binary : Pandas Series 
            Train Labels (Binary)
        y_test : Pandas Series 
            Test labels 
        y_test_binary : Pandas Series -> binary 
            Test labels (Binary)
    """
    
    assert 'df' in input_dict and 'df_columns' in input_dict, 'main data frame and/or meta data frame missing in the input param: input_dict'
    df , df_columns = input_dict['df'], input_dict['df_columns']
    uncertainty = True if 'df_unc' in input_dict else False

    if uncertainty: df_unc = input_dict['df_unc']
    
    if specialized: 
        df.drop(['He4_ter','NGRT','DNe'],axis=1,inplace=True)
        if uncertainty: 
            df_unc.drop(['He4_ter','NGRT','DNe'],axis=1,inplace=True)
        print("'He4_ter','NGRT','DNe' Dropped!")
    
    if len(cols_to_retain) != 0:
        df = df[cols_to_retain]

        if uncertainty:
            df_unc = df_unc[cols_to_retain]


    output = data_prep_imputation_normalizing(df,df_columns,tiers_to_remove=[0,4,'Nil'],lognormal=log_normal,z_score=z_score,impute=impute)   

    X_test,y_test = output['X_test'],output['y_test'] #set aside
    X_train,y_train = output['X_train'],output['y_train']
    
    if plot_dist_split:
        # code modified with broken axis 
        n_bins = 20
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0.2)  # adjust space between axes
        ax1.hist([y_train,y_test], bins=n_bins, stacked=False, label=['train distribution', 'test distribution'], alpha=0.7)
        ax2.hist([y_train,y_test], bins=n_bins, stacked=False, label=['train distribution', 'test distribution'], alpha=0.7)
        ax1.set_ylim(380, 390)  # outliers only
        ax2.set_ylim(0, 100)  # most of the data 
        # hide the spines between ax and ax2
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        d = 1.0  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        ax2.set_xlabel('Age (years)')
        ax1.set_ylabel('Count of Samples')
        ax1.yaxis.set_label_coords(-0.1,-0.05)
        ax1.legend()
        #ax2.title('Count of number of samples for train and test split')
        plt.savefig('../Result Plots/Age_Distribution_Split.png',dpi=720)

    
    X_train_torch,y_train_torch = torch.tensor(X_train.values),torch.tensor(y_train.values)
    #X_test_torch,y_test_torch = torch.tensor(X_test.values),torch.tensor(y_train.values)
    columns = X_train.columns

    mean_X,std_X = torch.tensor(X_train.mean()),torch.tensor(X_train.std())
    mean_Y,std_Y = torch.tensor(y_train.mean()),torch.tensor(y_train.std())
    
    if uncertainty:
        df_unc['age'] = df_unc['age_unc'].copy()
        df_unc = df_unc.drop(['age_unc'],axis=1)

        output_unc = data_prep_imputation_normalizing(df_unc,df_columns,tiers_to_remove=[0,4,'Nil '],lognormal=log_normal,z_score=z_score,impute=impute)
        
        #X_test_unc,y_test_unc = output_unc['X_test'],output_unc['y_test'] #set aside
        X_train_unc,y_train_unc = output_unc['X_train'],output_unc['y_train']

        X_train_torch_unc,y_train_torch_unc = torch.tensor(X_train_unc.values),torch.tensor(y_train_unc.values)
        #X_test_torch_unc,y_test_torch_unc = torch.tensor(X_test_unc.values),torch.tensor(y_train_unc.values)

    if augmented_size > 0 and uncertainty:
        train_flexi = FlexibleDataset(
            X_data= X_train_torch,
            y_data = y_train_torch,
            mean_x = mean_X, 
            std_x= std_X, 
            mean_y = mean_Y, 
            std_y = std_Y, 
            X_unc = X_train_torch_unc,
            y_unc = y_train_torch_unc
        )
    elif augmented_size == 0 and not uncertainty:
        train_flexi = FlexibleDataset(
            X_data= X_train_torch,
            y_data = y_train_torch,
            mean_x = None, # passed as none as normalization not done
            std_x= None, # passed as none as normalization not done
            mean_y = None, # passed as none as normalization not done
            std_y = None, # passed as none as normalization not done
            X_unc = None,
            y_unc = None
        )
    else:
        raise Exception('Pass a greater than 0 augmented_size value when uncertainty is False!')

    X_train_augmented,y_train_augmented = X_train_torch,y_train_torch
    train_loader = torch.utils.data.DataLoader(train_flexi,batch_size=X_train_torch.shape[0]) 
    

    for i in range(augmented_size):    
        x_i,y_i = next(iter(train_loader))
        
        X_train_augmented = torch.cat((X_train_augmented,x_i))
        y_train_augmented = torch.cat((y_train_augmented,y_i))
        
        print(f"{i}th augmentation")
    
    # converting torch tensors back to pandas dataframes 
    X_train_augmented_df = pd.DataFrame(X_train_augmented.numpy(),columns=columns)
    y_train_augmented_df = pd.DataFrame(y_train_augmented.numpy(),columns=['age'])

    y_train_augmented_df_binary = (y_train_augmented_df >= binary_split).astype(int)
    y_test_binary = (y_test >= binary_split).astype(int)

    output = {
        'X_train_augmented': X_train_augmented_df,
        'y_train_augmented': y_train_augmented_df,
        'X_test' : X_test,
        'y_test' : y_test,
        'y_train_binary' : y_train_augmented_df_binary,
        'y_test_binary' : y_test_binary
        }
    
    return output 
