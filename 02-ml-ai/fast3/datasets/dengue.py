import builtins
import h5py
import numpy as np
import os
import pandas as pd
import torch

from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from datasets.pdbbind import PDBBind
from utils.shape_util import affine_transform

class Dengue(Dataset):

    feat_dims = {
        "graph": 20,
        "3d": 19,
        "point": 20,
    }

    def __init__(
        self,
        mode,
        subset,
        configs,
        data_dir="data/dengue/",
        add_info=True,
        cache_data=False
    ):
        super(Dengue, self).__init__()
        self.mode = mode
        self.subset = subset
        self.configs = configs
        self.input_type = configs["input_type"]
        if self.mode == 'test':
            data_dir = os.path.join(data_dir, configs["split"]["test"][0][0])
        else:
            data_dir = os.path.join(data_dir, configs["split"]["train"][0][0])

        if not isinstance(self.input_type, list):
            self.input_type = [self.input_type]
        
        self.use_docking = configs["use_docking"]
        self.feature_type = configs["feature_type"]
        self.preprocessing_type = configs["preprocessing_type"]
        file_path = os.path.join(data_dir, "dengue-"+configs["pocket"]+".hdf")
        self.df = h5py.File(file_path, "r")
        self.add_info = add_info
        self.max_atoms = configs["max_atoms"] if "max_atoms" in configs else 2000
        self.affine = configs.pop("affine", False) and (self.mode in ("train", "val"))
        self.cache_data = cache_data
        comp_infos = pd.read_csv(os.path.join(data_dir, "protease_ligand_prep.csv"))
        comp_ids = pd.read_csv(os.path.join(data_dir, "train_test_valid_ids_"+configs["type"]+".csv"))

        comp_ids = comp_ids[comp_ids["subset"]==subset]["cmpd_id"].tolist()
        comp_infos = comp_infos[comp_infos["compound_id"].isin(comp_ids)]
        comp_ids = comp_infos["id"].tolist()
        
        self.df = h5py.File(file_path, "r")
        self.threshold = configs["threshold"] if "threshold" in configs else None
        self.data_list = []
        self.data_dict = {}
        self.zero_indices = []
        self.not_zero_indices = []
        self.flag = True
        pose_choice = set(np.random.choice([i+1 for i in range(20)], size=configs['poses'], replace=False))
        self.lower_bound = configs['affinity'] if 'affinity' in configs else 0.
        for id in comp_ids:
            if str(id) in self.df:
                poses = self.df[str(id)]["pybel"]["processed"]["docking"]
                affinity = np.asarray(self.df[str(id)].attrs["affinity"]).reshape(1, -1)
                if not configs["with_zeros"]:
                    if affinity != 0:
                        for pose in poses:
                            if int(pose) in pose_choice:
                                self.data_list.append((str(id), pose, affinity))

                else:
                    for pose in poses:
                        if int(pose) in pose_choice:
                            if self.threshold:
                                if affinity <= self.threshold:
                                    if affinity == 0.:
                                        self.data_list.append((str(id), pose, np.asarray(self.lower_bound).reshape(1,-1)))
                                    else:
                                        self.data_list.append((str(id), pose, affinity))

                                else:
                                    self.data_list.append((str(id), pose, affinity))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                return self.data_dict[item]
            else:
                pass
        
        pdbid, pose, affinity = self.data_list[item]
        node_feats, coords = None, None

        if self.use_docking:
            # TODO(june): implement if needed
            raise ValueError("Not implemented for docking yet.")
        else:
            data = self.df[pdbid.split('_')[0]][self.feature_type][self.preprocessing_type]["docking"][pose]["data"]
            vdw_radii = (
                self.df[pdbid.split('_')[0]][self.feature_type][self.preprocessing_type]["docking"][pose]
                .attrs["van_der_waals"]
                .reshape(-1, 1)
            )

        y=torch.tensor(affinity).float()
        _data = Data(
            y=y
        )
        if "graph" in self.input_type:
            coords = data[:, 0:3]
            node_feats = np.concatenate([vdw_radii, data[:, 3:22]], axis=1)

            # Account for the vdw radii in distance cacluations
            # (consider each atom as a sphere, distance between spheres)
            dists = pairwise_distances(coords, metric="euclidean")
            edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())
            x = torch.from_numpy(node_feats).float()
            # y = torch.FloatTensor(affinity).view(-1, 1)
            pos = torch.from_numpy(coords)
            
            _data.x = x
            _data.edge_index = edge_index
            _data.edge_attr = edge_attr.view(-1, 1)
            _data.pos = pos
            _data.label = torch.tensor([[1., 0.]]) if y == self.lower_bound else torch.tensor([[0., 1.]])

        if "3d" in self.input_type:
            data_3d = np.zeros((self.max_atoms, self.feat_dims["3d"] + 3), dtype=np.float32)
            actual_data = data[:]
            data_3d[:actual_data.shape[0],:] = actual_data
            _data.x_3d = torch.tensor(np.expand_dims(data_3d, axis=0)).float()

        if "point" in self.input_type:
            data_3d = np.zeros((self.max_atoms, self.feat_dims["point"] + 3), dtype=np.float32)
            actual_data = data[:]
            if self.affine:
                actual_data[: , 0:3] = affine_transform(actual_data[: , 0:3])
            data_3d[:actual_data.shape[0],:] = actual_data
            _data.x_point = torch.tensor(np.expand_dims(data_3d, axis=0)).float()

        # When `num_nodes` cannot be automatically inferred, set dummy value.
        if _data.x is None:
            _data.num_nodes = y.shape[0]

        inputs = {"data": _data}

        if self.add_info:
            inputs["id"] = pdbid
            inputs["pose"] = pose

        if self.cache_data:
            self.data_dict[item] = inputs

        return inputs
    
