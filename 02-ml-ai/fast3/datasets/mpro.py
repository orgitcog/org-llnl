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
from pdb import set_trace

from utils.shape_util import affine_transform



class Mpro(Dataset):

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
        data_dir="data/mpro",
        add_info=True,
    ):
        super(Mpro, self).__init__()

        self.mode = mode
        self.subset = subset
        self.configs = configs
        self.input_type = configs["input_type"]
        if not isinstance(self.input_type, list):
            self.input_type = [self.input_type]
        self.add_info = add_info
        self.max_atoms = configs["max_atoms"] if "max_atoms" in configs else 2000
        self.affine = configs.pop("affine", False) and (self.mode in ("train", "val"))

        file_path = os.path.join(data_dir, "gmd_postera_protease_pos.h5")
        self.df = h5py.File(file_path, "r")
        comp_ids_all = list(self.df)

        comp_infos = pd.read_csv(os.path.join(data_dir, "updated_mpro_positive_compounds.csv"))
        comp_infos = comp_infos[comp_infos.activity_type == "pIC50"]

        with open(os.path.join(data_dir, subset + "_ids.txt"), "r") as f:
            comp_ids = f.readlines()
        comp_ids = [comp_id.rstrip() for comp_id in comp_ids]

        self.data_list = []
        for comp_id in comp_ids:
            comp_id_poses = [x for x in comp_ids_all if comp_id + "_" in x]
            assert len(comp_id_poses) == 10, "Poses of {}: {}".format(comp_id, comp_id_poses)

            comp_info = comp_infos[comp_infos["compound_id"].str.endswith(comp_id)]
            assert len(comp_info) == 1, "# infos of {}: {}".format(comp_id, comp_info)
            affinity = np.asarray(comp_info["activity"]).reshape(1, -1)

            for comp_id_pose in comp_id_poses:
                pose = int(comp_id_pose.rsplit("_", 1)[-1])
                self.data_list.append((comp_id_pose, pose, affinity))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        comp_id, pose, affinity = self.data_list[item]
        node_feats, coords = None, None

        data_all = self.df[comp_id]

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

            x = torch.from_numpy(node_feats).float()
            # y = torch.FloatTensor(affinity).view(-1, 1)
            pos = torch.from_numpy(coords)

            _data.x = x
            _data.edge_index = edge_index
            _data.edge_attr = edge_attr.view(-1, 1)
            _data.pos = pos

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
            inputs["id"] = comp_id
            inputs["pose"] = pose

        return inputs
