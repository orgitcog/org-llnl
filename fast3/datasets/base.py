import numpy as np
import random
import torch

# from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataListLoader
# from torch_geometric.loader import 
# from typing import List
from pdb import set_trace
from datasets.sampler import BalancedBatchSampler

def worker_init_fn(worker_id):
    np.random.seed(int(0))


# How to define a balanced dataset

class CustomDataLoaders(object):
    def __init__(self, dataset_dict, configs, balanced=False):
        self.dataloader = {}
        self.steps_in_epoch = {}

        for mode, dataset in dataset_dict.items():
            data_size = len(dataset)
            batch_size = int(configs["batch_size"])
            self.steps_in_epoch[mode] = data_size // batch_size
            shuffle = True
            drop_last = False
            num_workers = configs["num_workers"] if "num_workers" in configs else 0
            self.dataloader[mode] = DataListLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                drop_last=drop_last,
                pin_memory=True
            )

    def __iter__(self):
        pass

    def get(self, mode):
        return self.dataloader[mode]
