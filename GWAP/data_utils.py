from torch.utils.data import Dataset, DataLoader
import numpy as np

class FlexibleDataset(Dataset):
    
    def __init__(self, 
                 X_data, 
                 y_data,
                 mean_x=None,
                 std_x=None,
                 mean_y=None,
                 std_y=None,
                 X_unc=None,
                 y_unc=None,
                 mode='r', 
                 n_bin=14, 
                 uniform=True):
        
        self.X_data = X_data
        self.y_data = y_data
        
        # get feature dimension
        self.dim = len(X_data[0])
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y
        
        # uncertainty
        self.X_unc = X_unc
        self.y_unc = y_unc
        
        if self.std_x is not None and self.mean_x is not None:
            self.X_data =  (self.X_data - self.mean_x) / self.std_x
            print('Normalize X')    
        if X_unc is not None:
            if self.std_x is not None and self.mean_x is not None:
                self.X_unc =  self.X_unc / self.std_x
            print("X uncertainty")
        else:
            print("No X uncertainty")
        
        if self.std_y is not None and self.mean_y is not None:
            self.y_data =  (self.y_data - self.mean_y) / self.std_y
            print('Normalize Y')  
        if y_unc is not None:
            if self.std_y is not None and self.mean_y is not None:
                self.y_unc =  self.y_unc / self.std_y
            print("Y uncertainty")
        else:
            print("No Y uncertainty")
            
        # "r" or "c" for regression or classification
        self.mode = mode
        # number of bins for regression
        self.n_bin = n_bin
        # whether to use bins to generate uniform class distribution
        self.uniform = uniform
    
    def classify(self):
        return 
    
    def __getitem__(self, index):
        if self.mode == 'r':
            if self.X_unc is not None:
                X = self.X_data[index] + abs(self.X_unc[index]) * np.random.normal(0,1,(1,self.dim))
            else:
                X = self.X_data[index]

            if self.y_unc is not None:
                y = self.y_data[index] + abs(self.y_unc[index]) * np.random.normal(0,1,1)
            else:
                y = self.y_data[index]
        # WIP, don't use         
        elif self.mode == 'c':
            if self.X_unc is not None:
                X = self.X_data[index] + abs(self.X_unc[index]) * np.random.normal(0,1,(1,self.dim))
            else:
                X = self.X_data[index]
            
            y = self.y_data[index]
        else:
            print("not implemented, choose between mode = r or c")
        return X.squeeze().float(), y.squeeze().float()
        
    def __len__ (self):
        return len(self.X_data)