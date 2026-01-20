import torch
import os
from torch_geometric.loader import DataLoader
import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Dataset,Data
from MEAG_VAE.args import parse_args
from MEAG_VAE.config import load_cfg,update_out_dir,cfg,dump_cfg
from MEAG_VAE.models.utils import build_graph

from MEAG_VAE.dataprocessing.load_data import CustomDataset
from MEAG_VAE.models.model_Net import Net
from MEAG_VAE.optimizer import build_optimizer
from MEAG_VAE.loader import data_loader
from MEAG_VAE.train import train_cp
import ase.io
import json
import numpy as np
import torch
import optparse
import os
torch.manual_seed(42)

#user inputs
args=parse_args()
load_cfg(cfg,args)
update_out_dir(cfg.out_dir,args.cfg_file)

group_name_list = args.group_name.split("+")
device=torch.device('cpu')
data_set=CustomDataset(root=cfg.dataset.dir_name,
                            element=cfg.dataset.elem,
                            feature_type=cfg.dataset.feature_type,
                            group_list=group_name_list)

channels = list(map(int, cfg.model.channels.split(",")))

model = Net(data_set.num_features, cfg.model.num_kernels,cfg.model.pooling_rate,channels,cfg.model.edge_reduction).to(device)
model.load_state_dict(torch.load(f'{cfg.out_dir}/{cfg.model.file_name}.pth',map_location='cpu'))
model.eval()
model.update_parameters(rate=args.rate)

N_str = data_set.len()

train_val_ratio=args.train_val_ratio
N_trn = int(train_val_ratio * N_str)
N_test = N_str - N_trn


if train_val_ratio == 1.0: 
    train_set = list(DataLoader(data_set, batch_size=N_trn, shuffle=False))
    test_set=None
    test_data=None
#    test_set = DataLoader(test_dataset, batch_size = N_test)
else:
    generator = torch.Generator().manual_seed(42)
    train_dataset,test_dataset = torch.utils.data.random_split(data_set, [N_trn,N_test],generator=generator)
    train_set = DataLoader(train_dataset, batch_size=N_trn, shuffle=False)
    test_set = DataLoader(test_dataset, batch_size = N_test)
    test_data=next(iter(test_set))



#load the dataset
device=torch.device('cpu')

#build the test folder and save the graph clusters to file
saved_xyz_dir = f'{cfg.out_dir}/{cfg.test.xyz_dir}'
os.makedirs(saved_xyz_dir,exist_ok=True)
final_graph_x=[]


with torch.no_grad():
    for data in train_set:
        data=data.to(device)
        node_mask,x,edge_index=build_graph(data,args.fixed_rate_l,args.fixed_rate_r)
        assert len(node_mask) > 0, print("no edges")
        data_x = data.x.detach().cpu().numpy() 
        config_idx=(data.y).detach().cpu().numpy()
        num_nodes_tot=len(data_x)
        z, latent_x, latent_edge,_ = model(x,edge_index)
        loss=torch.nn.MSELoss()(z,x)
        print(f'testing loss is: {loss}')
        edge_index = latent_edge
        edge_index = node_mask[edge_index].view(2, -1)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        
        #constructed similarity sub graphs based on the edge_index info in the reduced latent graph
        graph = nx.Graph()
        graph.add_edges_from(edge_index.t().tolist())
        connected_subgraphs= list(nx.connected_components(graph)) 
        isolated_nodes=set(range(num_nodes_tot))-set(graph.nodes)
        print(f'total number of atoms: {num_nodes_tot}, number of isolated atoms: {len(isolated_nodes)}')

        node_info={}
        #node info for connected clusters
        for cluster_idx,cluster in enumerate(connected_subgraphs):
            for node in cluster:
                size=len(cluster)
            #    node_info[node]=[cluster_idx,size,int(config_idx[node][0]),int(config_idx[node][1])]
                node_info[node]=[cluster_idx,size]
       
        #node info for isolatedd clusters
        count=0
        for node in isolated_nodes:
            count+=1
        #    node_info[node]=[cluster_idx+count,1,int(config_idx[node][0]),int(config_idx[node][1])]
            node_info[node]=[cluster_idx+count,1]

file_path = f"{saved_xyz_dir}/clusters_{args.group_name}_r{args.rate}_train{args.train_val_ratio}.json"
with open(file_path, 'w') as json_file:
    json.dump(node_info, json_file)

datafile_path=f"{saved_xyz_dir}/graph_{args.group_name}_train{args.train_val_ratio}_data.pt"
testdatafile_path=f"{saved_xyz_dir}/graph_{args.group_name}_train{args.train_val_ratio}_testdata.pt"
if not os.path.exists(datafile_path):
    torch.save(data,f"{saved_xyz_dir}/graph_{args.group_name}_train{args.train_val_ratio}_data.pt")
if not os.path.exists(testdatafile_path):
    torch.save(test_data,f"{saved_xyz_dir}/graph_{args.group_name}_train{args.train_val_ratio}_testdata.pt")

