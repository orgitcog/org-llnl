import builtins
import h5py
import numpy as np
import os
import torch

from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from pdb import set_trace
# from datasets.base import CustomData
from utils.shape_util import affine_transform


class PDBBind(Dataset):

    def __init__(
        self,
        mode,
        subset,
        configs,
        file_path,
        add_info=True,
        cache_data=False,
        h5_file_driver=None
    ):
        super(PDBBind, self).__init__()

        self.mode = mode
        self.subset = subset
        self.configs = configs
        self.df = h5py.File(file_path, "r", driver=h5_file_driver)

        self.input_type = configs["input_type"]
        if not isinstance(self.input_type, list):
            self.input_type = [self.input_type]
        self.add_info = add_info
        self.cache_data = cache_data

        # self.feat_dim = configs["feat_dim"] + 3 if "feat_dim" in configs else 22
        self.max_atoms = configs["max_atoms"] if "max_atoms" in configs else 2000
        self.affine = configs.pop("affine", False) and (self.mode in ("train", "val"))

        self.data_dict = {}  # store data once it has been computed if `cache_data` is True
        self.data_list = []  # store ids for data

    def __len__(self):
        return len(self.data_list)


class PDBBind2016(PDBBind):
    feat_dims = {
        "graph": 20,
        "3d": 19,
        "point": 19,
    }

    def __init__(
        self,
        mode,
        subset,
        configs,
        data_dir="data/pdbbind2016",
    ):
        file_path = os.path.join(data_dir, subset + ".hdf")
        super(PDBBind2016, self).__init__(mode, subset, configs, file_path, cache_data=False)

        self.feature_type = configs["feature_type"]
        self.preprocessing_type = configs["preprocessing_type"]
        self.use_docking = configs["use_docking"]

        if self.use_docking:
            # TODO(june): implement if needed
            raise ValueError("Not implemented for docking yet.")
        else:
            for name in list(self.df):
                # if the feature type (pybel or rdkit) not available, skip over it
                if self.feature_type in list(self.df[name]):
                    affinity = np.asarray(self.df[name].attrs["affinity"]).reshape(1, -1)

                    # Putting 0 for pose to denote experimental structure
                    # and to be consistent with docking data format
                    self.data_list.append((name, 0, affinity))



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
            data = self.df[pdbid][self.feature_type][self.preprocessing_type]["data"]
            vdw_radii = (
                self.df[pdbid][self.feature_type][self.preprocessing_type]
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


class PDBBind2020(PDBBind):
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
        data_dir="data/pdbbind2020",
    ):
        file_path = os.path.join(data_dir, subset + ".h5")
        super(PDBBind2020, self).__init__(mode, subset, configs, file_path, cache_data=False)

        # self.input_key = "voxelized" if self.input_type == "3d" else "spatial"

        for name in list(self.df):
            affinity = np.asarray(self.df[name].attrs["affinity"]).reshape(1, -1)

            # Putting 0 for pose to denote experimental structure
            # and to be consistent with docking data format
            self.data_list.append((name, 0, affinity))
        

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                return self.data_dict[item]
            else:
                pass

        pdbid, pose, affinity = self.data_list[item]
        node_feats, coords = None, None

        data_all = self.df[pdbid]

        y = torch.tensor(affinity).float()
        _data = Data(
            y=y,
        )

        if "graph" in self.input_type:
            data = data_all["spatial"]
            coords = data["coords"][:]
            node_feats = data["node_feats"][:]
            dists = data["dists"][:]
            edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())

            # from pdb import set_trace
            # set_trace()
            x = torch.from_numpy(node_feats).float()
            # y = torch.FloatTensor(affinity).view(-1, 1)
            pos = torch.from_numpy(coords)

            _data.x = x
            _data.edge_index = edge_index
            _data.edge_attr = edge_attr.view(-1, 1)
            _data.pos = pos
            _data.label = torch.tensor([0., 1.]) if y == 0 else torch.tensor([1., 0.])


        if "3d" in self.input_type:
            data = data_all["voxelized"]
            _data.x_3d = torch.tensor(np.expand_dims(data["data0"][:], axis=0)).float()

        if "point" in self.input_type:
            data = data_all["spatial"]
            coords = data["coords"][:]
            if self.affine:
                coords = affine_transform(coords)

            node_feats = data["node_feats"][:]
            actual_data = np.concatenate((coords, node_feats), axis=-1)

            data_point = np.zeros((self.max_atoms, self.feat_dims["point"] + 3), dtype=np.float32)
            data_point[:actual_data.shape[0],:] = actual_data
            _data.x_point = torch.tensor(np.expand_dims(data_point, axis=0)).float()

        # When `num_nodes` cannot be automatically inferred, set dummy value.
        if _data.x is None:
            _data.num_nodes = y.shape[0]

        inputs = {"data": _data}

        if self.add_info:
            inputs["id"] = pdbid
            inputs["pose"] = pose

        # if self.cache_data:
        #     self.data_dict[item] = inputs

        return inputs
    
    def get_id(self):
        pass