import os
import torch
from torch_geometric.data import Dataset,Data
import glob
import ase
from multiprocessing import Pool
from MEAG_VAE.dataprocessing.helper import feature_matrix
from MEAG_VAE.config import cfg
import numpy as np
import pickle


class StructureDataset(Dataset):
    r"""
    Dataset class for processing and saving data into PyG format.

    Args:
        root (str): Root directory of the dataset.
        element (str): Element symbol (default: 'Nb').
        feature_type (str): Type of features to use (default: 'soap').
        transform (callable, optional): A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version.
        pre_transform (callable, optional): A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version.

    Returns:
        torch_geometric.data.Dataset: Dataset containing processed structures.
    """
    def __init__(self,root,
                 element='Nb',
                 feature_type='soap',
                 num_cores=cfg.num_threads,
                 transform=None,pre_transform=None):
        

        self.root=root
        self.elem=element
        self.num_cores=num_cores
        self.rcut=3.0
        self.feature_type=feature_type

        #folder for raw and processed files
        self.raw=os.path.join(root,'raw',f'raw_{self.elem}_{self.feature_type}')
        self.processed=os.path.join(root,'processed',f'processed_{self.elem}_{self.feature_type}')

        if not os.path.isdir(self.processed):
            os.makedirs(self.processed)

        #load dataset
        if 'snap' in self.feature_type:
            with open(f'{self.raw}/{self.elem}_dataset.pkl','rb') as file:
                self.data_info=pickle.load(file)
            self.N_structures=len(self.data_info.keys())
            column_names = next(iter(self.data_info.values())).columns
            self.num_descriptors = sum(str(col).isdigit() for col in column_names)
        else:
            self.structures=ase.io.read(f'{self.raw}/train.xyz',":")
            self.N_structures= len(self.structures)
            self.feats_amat=feature_matrix(self.structures,self.elem,self.feature_type)
       
       # self.feats_amat=np.loadtxt('feature_amat.txt')
        super().__init__(self.root, transform, pre_transform)



    @property
    def raw_file_names(self):
        return f'{self.elem}.xyz'

    @property
    def processed_dir(self):
        return self.processed
    
    @property
    def processed_file_names(self):

        processed_files=glob.glob(f'{self.processed}'+'/'+'data*.pt')
        processed_file_list=[os.path.basename(p) for p in processed_files]

        return processed_file_list
    def download(self):
        pass

    # Uncomment the lines below to process the data in parallel with num_cores of CPU. Need to define self.process_each()
    # def process(self):
    #     with Pool(self.num_cores) as p:
    #         p.map(self.process_each,range(self.N_structures))

    #transform the data into PyG object
    def process(self) -> None:
        if 'snap' in self.feature_type:
            each=0
            for _, df in self.data_info.items():
                # Extract node features and convert to tensor
                node_feats=torch.tensor(df.iloc[:,0:self.num_descriptors].to_numpy()+1e-15,dtype=eval(cfg.torch_real))
                # Create node index labels
                y=np.array([[each,i] for i in range(len(node_feats))])
                y=torch.tensor(y,dtype=torch.long)
                # Create Data object
                data=Data(x=node_feats,y=y)
                torch.save(data,os.path.join(self.processed,f'data_{each}.pt'))
                each+=1
        else:
            for each in range(self.N_structures):
                node_feats=self.feats_amat[each]
                pos=self.structures[each].get_positions()
                y=np.array([[each,i] for i in range(len(pos))])
                node_feats=torch.tensor(node_feats,dtype=eval(cfg.torch_real))
                pos=torch.tensor(pos,dtype=eval(cfg.torch_real))
                y=torch.tensor(y,dtype=torch.long)
                data=Data(x=node_feats,pos=pos,y=y)
                torch.save(data,os.path.join(self.processed,f'data_{each}.pt'))
    def len(self):
        return self.N_structures
    def get(self,idx):
        return torch.load(os.path.join(self.processed,f'data_{idx}.pt'))





class CustomDataset(Dataset):
    r"""
    Custom dataset class for processing and saving data into PyG format.

    Args:
        root (str): Root directory of the dataset.
        element (str): Element symbol (default: 'Ta').
        feature_type (str): Type of features to use (default: 'snap').
        group_list (list): List of group names to include in the dataset (default: None).
        transform (callable, optional): A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version.
        pre_transform (callable, optional): A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version.

    Returns:
        torch_geometric.data.Dataset: Custom dataset containing processed data.
    """

    def __init__(self,root,
                 element='Ta',
                 feature_type='snap',
                 group_list=None,
                 num_cores=cfg.num_threads,
                 transform=None,pre_transform=None):
        
        self.root=root
        self.elem=element
        self.num_cores=num_cores
        self.rcut=3.0
        self.feature_type=feature_type
        self.group_list=group_list
        #folder for raw and processed files
        self.raw=os.path.join(root,'raw',f'raw_{self.elem}_{self.feature_type}')
  
        self.dataset=[]
        #load dataset
        if 'snap' in self.feature_type:
            with open(f'{self.raw}/{self.elem}_dataset.pkl','rb') as file:
                self.data_info=pickle.load(file)
            self.N_structures=len(self.data_info.keys())
            column_names = next(iter(self.data_info.values())).columns
            self.num_descriptors = sum(str(col).isdigit() for col in column_names)
        else:
            self.structures=ase.io.read(f'{self.raw}/train.xyz',":")
            self.N_structures= len(self.structures)
            self.feats_amat=feature_matrix(self.structures,self.elem,self.feature_type)
   
        self.process_data()
        super().__init__(self.root, transform, pre_transform)
    
      
    def process_data(self):
        if 'snap' in self.feature_type:
            each=0
            for _, df in self.data_info.items():
                # Skip if the group name of the structure is not in the specified group list
                if df['Groups'].iloc[0] not in self.group_list: continue
                for i in range(len(df)):
                    # Extract node features and convert to tensor
                    node_feats_data = df.iloc[i, 0:self.num_descriptors].values.reshape(1,-1).astype(np.float32) + 1e-15
                    node_feats = torch.tensor(node_feats_data, dtype=eval(cfg.torch_real))
                    y=np.array([[each,i]])
                    y=torch.tensor(y,dtype=torch.long)
                    data=Data(x=node_feats,y=y)

                    # Define columns and attributes for force predictions and truths
                    columns = ['preds_fx', 'truths_fx', 'preds_fy', 'truths_fy', 'preds_fz', 'truths_fz']
                    attributes = ['forcex_preds', 'forcex_truths', 'forcey_preds', 'forcey_truths', 'forcez_preds', 'forcez_truths']
                    # Set force predictions and truths as attributes of the Data object
                    for col, attr in zip(columns, attributes):
                        if col in df.columns:
                            setattr(data, attr, torch.tensor(df[col].to_numpy()[i].reshape(1,-1), dtype=eval(cfg.torch_real)))
                
                # Add configuration and group information to the Data object
                    label = [{
                        'Configs': df['Configs'].iloc[0],
                        'Groups': df['Groups'].iloc[0]
                    }]
                    data.labels=label

                    self.dataset.append(data)

                each+=1
            
        else:
            for each in range(self.N_structures):
                if str(each) not in self.group_list:continue
                node_feats=self.feats_amat[each]
                pos=self.structures[each].get_positions()
                y=np.array([[each,i] for i in range(len(pos))])
                node_feats=torch.tensor(node_feats,dtype=eval(cfg.torch_real))
                pos=torch.tensor(pos,dtype=eval(cfg.torch_real))
                y=torch.tensor(y,dtype=torch.long)
                data=Data(x=node_feats,pos=pos,y=y)
                self.dataset.append(data)
      
    def len(self):
        return len(self.dataset)
    
    def get(self,idx):
        return self.dataset[idx]        





