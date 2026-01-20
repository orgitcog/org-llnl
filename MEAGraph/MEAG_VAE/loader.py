import torch
import os
import numpy as np
from torch_geometric.data import Dataset
from MEAG_VAE.dataprocessing.load_data import StructureDataset
from torch_geometric.loader import DataLoader

from MEAG_VAE.config import cfg


def data_loader(data_set,batch_size=8,train_val_ratio=0.9):
    r"""
    load the train and test data
    """

    torch.manual_seed(torch.initial_seed())
    device=torch.device(cfg.device)
  
    input_size = data_set.num_features

    N_str = data_set.len()
        
    # split N_str into training, validation 
    N_trn = int(train_val_ratio * N_str)
    N_test = N_str - N_trn


    if train_val_ratio == 1.0: 
  
        train_set = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        test_set=None

    else:
        train_dataset,test_dataset = torch.utils.data.random_split(data_set, [N_trn,N_test])
        train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_set = DataLoader(test_dataset, batch_size = N_test)
    return train_set,test_set