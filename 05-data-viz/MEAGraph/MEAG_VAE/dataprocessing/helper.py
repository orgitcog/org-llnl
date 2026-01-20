import ase
import ase.io
from ase.neighborlist import neighbor_list
import functools
from typing import List, Union
import numpy as np
from ase import Atoms


import numpy as np
import torch
from torch import linalg as LA

from dscribe.descriptors import SOAP,ACSF
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
from MEAG_VAE.config import cfg


def feature_matrix(structures=None,elem='Ta',feature_type='soap'):
    r"""
    Generate feature matrices for a list of atomic structures.

    Args:
        structures (List[Atoms], optional): List of atomic structures. Defaults to None.
        elem (str, optional): Element symbol. Defaults to 'Ta'.
        feature_type (str, optional): Type of feature to generate. Supported types: 'nequip', 'soap'. Defaults to 'soap'.

    Returns:
        List[np.ndarray]: List of feature matrices for each structure.

    Raises:
        ValueError: If the specified feature type is not supported.
    """
  
    if structures is not None:
        species = set()
        for structure in structures:
            species.update(structure.get_chemical_symbols())     
    if feature_type=='nequip':
        # Load pre-computed NeQuIP embeddings from a file in the raw folder
        feats_amat=np.load(f'{cfg.dataset.dir_name}/raw/raw_{elem}_{feature_type}/embeddings.npy')
        # Get the number of atoms in each structure
        N_atom_list=[len(atom.positions) for atom in structures]
    elif feature_type == 'soap':
        soap = SOAP(
            species=species,
            periodic=True,
            r_cut=5.0, 
            n_max=3,
            l_max=3,
            average="off",
            sparse=False
        )
        feats_amat=[]
        for atoms in structures:
            feature_vectors = soap.create(atoms,n_jobs=8)
            feats_amat.append(np.array(feature_vectors)+1e-15)
        return feats_amat
    else:
        raise ValueError("feature type not available")
    
    # Add a small constant to the feature matrices to avoid zero values
    scaled_feats_amat=feats_amat+1e-15
    
    #Change the feats_amat of the whole dataset to a list of per-structure feats matrix
    scaled_feats_groups=[scaled_feats_amat[0:N_atom_list[0],:]]
    last_sum=N_atom_list[0]
    for i in range(1,len(N_atom_list)):
        scaled_feats_groups.append(scaled_feats_amat[last_sum:last_sum+N_atom_list[i],:])
        last_sum=last_sum+N_atom_list[i]

    
    return scaled_feats_groups 



def build_sim_graph(feats_amat,threshold):
    r"""
    Build the similarity graph based feature matrix of a structure,cutoff=threshold
    -feats_amat:(N_atom,N_feats)
    -threshold: 1e-4
    -return: edge_index (N_edge,2), edge_weight (N_edge,)
    """
    
    sim_amat=euclidean_distances(feats_amat)

    # Uncomment the following lines to perform Min-Max scaling on the similarity matrix
    # # scaler=MinMaxScaler()
    # # exp_sim=scaler.fit_transform(sim_amat)
    
    exp_sim=np.exp(-sim_amat)
    #Uncomment to use cosine similarity
    #exp_sim=(cosine_similarity(feats_amat)+1)/2
    print(f'max and min sim:{np.min(exp_sim),np.max(exp_sim)}')

    edge_index=np.argwhere(exp_sim > np.mean(exp_sim))
#    edge_index=np.argwhere(exp_sim > threshold)
    assert edge_index.size > 0, print("Error: empty edge_index")
    edge_weight=exp_sim[exp_sim > threshold]
    return edge_index,edge_weight
    


def constr_adj_list(at: ase.Atoms):
    r"""
    construct the adj_list for each ase.Atoms object with periodic boundary conditions
    args: input at:ase.Atoms object
          return: adj_list
    """
    torch_real=torch.float32
    r_cut=3.0 
    max_neighbors=12

    neighbor_func=functools.partial(neighbor_list,cutoff=r_cut,self_interaction=False)

    edge_i,edge_j,d,D=neighbor_func(at)

    #adj_list:List([List([(idx_j,dis_j,dis_vec_j),(),...])])
    adj_list=[[] for i in range(len(at))]

    #convert to adj_list
    for i in range(len(edge_i)):
        adj_list[edge_i[i]].append((edge_j[i],d[i],D[i]))
    
    #only keep adj_list up to max_neighbors
    for adj in adj_list:
        adj.sort(key=lambda x:x[1])
        adj[:]=adj[:max_neighbors]
    
    #create the edge_index and edge_vec list (Num_edge,2) (Num_edge,3)
    edge_index,edge_vec=[],[]
    for i in range(len(adj)):
        for j in adj[i]:
            edge_index.append([i,j[0]])
            edge_vec.append(j[2])
    
    #Uncomment the lines below to convert the edge_index to (2,Num_edge) for pyG data
   # edge_index=torch.transpose(torch.tensor(edge_index,dtype=torch.long),0,1)
   # edge_vec=torch.tensor(np.array(edge_vec),dtype=torch_real)
    # return edge_index,edge_vec
    return adj



    

    


        

