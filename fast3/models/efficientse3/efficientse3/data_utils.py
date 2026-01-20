################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Spatial Graph Convolutional Network data loading utilities
################################################################################

import torch
import os.path as osp

from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
import scipy
import h5py
import json
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from tqdm import tqdm






'''
class PDBBindDataset(Dataset):

    def __init__(self,
                 data_file,
                 dataset_name,
                 feature_type,
                 preprocessing_type,
                 use_docking=False,
                 output_info=False,
                 cache_data=True,
                 h5_file_driver=None,
                 non_int_distance_cutoff=5.,
                 int_distance_cutoff=5.,
                 ligand_only=False):
        super().__init__()
        self.dataset_name = dataset_name

        self.data_file = data_file
        self.feature_type = feature_type
        self.preprocessing_type = preprocessing_type
        self.use_docking = use_docking
        self.output_info = output_info
        self.cache_data = cache_data
        self.data_dict = {
        }  # will use this to store data once it has been computed if cache_data is True

        self.data_list = []  # will use this to store ids for data

        self.h5_file_driver = h5_file_driver
        self.int_distance_cutoff = int_distance_cutoff
        self.non_int_distance_cutoff = non_int_distance_cutoff
        self.ligand_only = ligand_only

        if self.use_docking:

            with h5py.File(data_file, "r") as f:

                for name in list(f):
                    # if the feature type (pybel or rdkit) not available, skip over it
                    if self.feature_type in list(f[name]):
                        affinity = np.asarray(
                            f[name].attrs["affinity"]).reshape(1, -1)
                        if self.preprocessing_type in f[name][
                                self.feature_type]:
                            if self.dataset_name in list(
                                    f[name][self.feature_type][
                                        self.preprocessing_type]):
                                for pose in f[name][self.feature_type][
                                        self.preprocessing_type][
                                            self.dataset_name]:
                                    self.data_list.append(
                                        (name, pose, affinity))

        else:

            with h5py.File(data_file, "r", driver=self.h5_file_driver) as f:

                for name in list(f):
                    # if the feature type (pybel or rdkit) not available, skip over it
                    if self.feature_type in list(f[name]):
                        affinity = np.asarray(
                            f[name].attrs["affinity"]).reshape(1, -1)

                        self.data_list.append(
                            (name, 0, affinity)
                        )  # Putting 0 for pose to denote experimental structure and to be consistent with docking data format

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
        with h5py.File(self.data_file, "r") as f:

            if (not self.dataset_name in f["{}/{}/{}".format(
                    pdbid, self.feature_type,
                    self.preprocessing_type)].keys()):
                print(pdbid)
                return None

            if self.use_docking:
                # TODO: the next line will cuase runtime error because not selelcting poses
                data = f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )][pose]["data"]
                vdw_radii = (f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )][pose].attrs["van_der_waals"].reshape(-1, 1))

            else:
                data = f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )]["data"]
                vdw_radii = (f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )].attrs["van_der_waals"].reshape(-1, 1))

            if self.feature_type == "pybel":
                coords = data[:, 0:3]
                node_feats = np.concatenate([vdw_radii, data[:, 3:22]], axis=1)

            else:
                raise NotImplementedError

        # account for the vdw radii in distance cacluations (consider each atom as a sphere, distance between spheres)

        if self.ligand_only:
            ligand_mask = node_feats[:, 14] == 1
            node_feats = node_feats[ligand_mask]
            coords = coords[ligand_mask]
        dists = pairwise_distances(coords, metric="euclidean")

        edge_index, edge_attr = dense_to_sparse(
            torch.from_numpy(dists).float())

        is_ligand = torch.BoolTensor(node_feats[:, 14] == 1)
        is_protein = torch.BoolTensor(node_feats[:, 14] == -1)
        ligand_index = is_ligand.nonzero().view(-1)
        protein_index = is_protein.nonzero().view(-1)
        i = edge_index[0]
        j = edge_index[1]
        from_ligand = (i[..., None] == ligand_index).any(-1).squeeze()
        to_ligand = (j[..., None] == ligand_index).any(-1).squeeze()
        int_mask = (from_ligand != to_ligand)

        mask = (edge_attr < self.non_int_distance_cutoff).logical_or(
            int_mask.logical_and(edge_attr < self.int_distance_cutoff))
        edge_index = edge_index[:, mask.view(-1)]
        edge_attr = edge_attr[mask]

        x = torch.from_numpy(node_feats).float()

        y = torch.FloatTensor(affinity).view(-1, 1)
        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
        data.coords = torch.FloatTensor(coords)

        if self.cache_data:
            if self.output_info:
                self.data_dict[item] = (pdbid, pose, data)

            else:
                self.data_dict[item] = data

            return self.data_dict[item]

        else:
            if self.output_info:
                return (pdbid, pose, data)
            else:
                return data


class PDBBind2019Dataset(Dataset):

    def __init__(
        self,
        data_file,
        cache_data=True,
        h5_file_driver=None,
        distance_cutoff=5.,
    ):
        super().__init__()

        self.data_file = data_file
        self.cache_data = cache_data
        self.data_dict = {
        }  # will use this to store data once it has been computed if cache_data is True

        self.data_list = []  # will use this to store ids for data

        self.h5_file_driver = h5_file_driver
        self.distance_cutoff = distance_cutoff

        self.file = h5py.File(self.data_file, "r")
        names = list(self.file.keys())
        self.data_list = names

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                return self.data_dict[item]
            else:
                pass

        name = self.data_list[item]
        node_feats = torch.FloatTensor(
            np.asarray(self.file[name]['spatial']['node_feats']))
        #affinity = torch.FloatTensor(np.asarray(self.file[name].attrs['affinity'])).reshape(1, -1)
        affinity = torch.FloatTensor(
            np.asarray(self.file[name].attrs['affinity']))
        coords = torch.FloatTensor(
            np.asarray(self.file[name]['spatial']['coords']))
        dists = torch.from_numpy(
            np.asarray(self.file[name]['spatial']['dists']))
        edge_index, edge_attr = dense_to_sparse(dists)
        mask = edge_attr < self.distance_cutoff
        edge_index = edge_index[:, mask.view(-1)]
        x = node_feats
        data = Data(x=x,
                    coords=coords,
                    edge_index=edge_index,
                    y=affinity,
                    name=name)
        if self.cache_data:
            self.data_dict[item] = data
            return self.data_dict[item]
        else:
            return data

class PDBBind2019GraphDatasetOld(Dataset):
    def __init__(self, data_file, distance_cutoff=5.,):
        super().__init__()
        f_name = Path(data_file).name
        suffix = Path(data_file).suffix
        if suffix == ".h5":
            processed = []
            with h5py.File(data_file, 'r') as file:
                names = list(file.keys())
                for name in tqdm(names, desc=f"Preprocessing data from {f_name}"):
                    node_feats = torch.from_numpy(
                        # np.asarray(file[name]['spatial']['node_feats']))
                        np.asarray(file[name]['md']['data']))
                    coords = torch.from_numpy(
                        # np.asarray(file[name]['spatial']['coords']))
                        np.asarray(file[name]['md']['coords']))
                    dists = np.asarray(file[name]['md']['pdists'])
                    dists = torch.from_numpy(scipy.spatial.distance.squareform(dists))
                    affinity = torch.from_numpy(
                        np.asarray(file[name].attrs['affinity']))
                    
                    edge_index, edge_attr = dense_to_sparse(dists)
                    mask = edge_attr < distance_cutoff
                    edge_index = edge_index[:, mask.view(-1)]
                    x = node_feats
                    data = Data(x=x,
                                coords=coords,
                                edge_index=edge_index,
                                y=affinity,
                                name=name)
                    processed.append(data)
            self.data = processed
        elif suffix == ".pt":
            self.data = torch.load(data_file).data
        else:
            raise ValueError("Only suffixes ending in .h5 or .pt are supported")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]
'''


'''
class PDBBind2019GraphDataset(Dataset):
    def __init__(
        self,
        data_file,
        crystal_scale=1.0,
        mmgbsa_scale=1.0,
        filter_list_file=None,
        task_idx=None,
        pred_mode=False,
    ):
        super(PDBBind2019GraphDataset, self).__init__()
        self.dataset_name = "md"
        self.data_file = data_file
        self.crystal_scale = crystal_scale
        self.mmgbsa_scale = mmgbsa_scale
        self.data_dict = (
            {}
        )  # will use this to store data once it has been computed if cache_data is True
        self.task_idx = task_idx
        self.data_list = []  # will use this to store ids for data
        self.node_feat_dict = {}
        self.affinity_dict = {}
        self.filter_list_file = filter_list_file
        self.pred_mode = pred_mode

        if self.filter_list_file is not None:
            self.filter_list = pd.read_csv(self.filter_list_file, index_col=0)['id'].values.tolist()

        with h5py.File(data_file, "r") as f:

            name_list = list(f)
            num_ids = len(name_list)
            if filter_list_file is not None:
                name_list = [x for x in name_list if x in self.filter_list]
                print(f"removed {num_ids - len(name_list)} from dataset.")

            for name in name_list:

                affinity = np.asarray(
                    f[name].attrs["affinity"][0]
                )  # todo: for some reason the affinity attribute may have shape of 2??? (5/11/21)


                # we will only use the data for which we have mmgbsa scores for
                n_frames = f[name][self.dataset_name]['DELTA TOTAL'].shape[
                    0
                ]  # this corrresponds to the number of frames

                if not self.pred_mode:
                    node_feats = f[name][self.dataset_name]["data"][:]
                    vdw_radii = f[name][self.dataset_name]["van_der_waals"][:].reshape(
                        -1, 1
                    )

                    node_feats = np.concatenate([vdw_radii, node_feats], axis=1)

                    self.node_feat_dict[name] = node_feats
                self.affinity_dict[name] = affinity


                #TODO: use argument to determine whether to use all simulation frames or allow the user to supply a range of frames?



                for frame_idx in range(n_frames):

                    self.data_list.append((name, frame_idx))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        name, frame_idx = self.data_list[item]

        node_feats = None
        if not self.pred_mode:
            node_feats = self.node_feat_dict[name]
            affinity = self.affinity_dict[name]
        with h5py.File(self.data_file, "r") as f:

            mmgbsa_data = f[name][self.dataset_name]["DELTA TOTAL"][frame_idx]
            mmgbsa_score = torch.from_numpy(np.asarray([mmgbsa_data]))

            pdists = f[name][self.dataset_name]["pdists"][frame_idx] * 10 #TODO: putting this here bc dists were computed in nanometers
            # pdists = torch.from_numpy(scipy.spatial.distance.squareform(pdists))
            #pdists = torch.from_numpy(pdists)



            coords = torch.from_numpy(np.asarray(f[name]['md']['coords'][frame_idx]))

            if self.pred_mode:
                node_feats = f[name][self.dataset_name]["data"][:]
                vdw_radii = f[name][self.dataset_name]["van_der_waals"][:].reshape(-1, 1)
                node_feats = np.concatenate([vdw_radii, node_feats], axis=1)


        #edge_index, edge_attr = dense_to_sparse(torch.from_numpy(pdists))        

        #import ipdb
        #ipdb.set_trace()
        edge_index, _ = dense_to_sparse(torch.from_numpy(scipy.spatial.distance.squareform(pdists)))
        log_ki_kd=torch.FloatTensor(affinity).reshape(-1,1) * self.crystal_scale #TODO: there was a bug with some datafiles with affinty being duplicated, so I had use the 0th index but removing that now
        mmgbsa_score=mmgbsa_score.reshape(-1,1) * self.mmgbsa_scale

        y = None
        try:
            #y = torch.cat([log_ki_kd, mmgbsa_score], dim=1)        
            # import warnings
            # warnings.warn("deprecated use, need to implement multi target training for se3", warnings.UserWarning)
            y = log_ki_kd

        except RuntimeError as e:
            print(e, name, frame_idx, self.data_file)

        

        data = Data(
            x=torch.from_numpy(node_feats).float(),
            y=y,
            log_ki_kd=log_ki_kd,
            mmgbsa_score=mmgbsa_score,
            frame_idx=torch.tensor([frame_idx]).float(),
            pdbid_pose_replicate_id=torch.tensor([ord(x) for x in name]).view(1, -1),
            edge_index=edge_index,
            # edge_attr=edge_attr,
            coords=coords,
            pdists=torch.from_numpy(pdists),
        )

        return data

'''
















'''
class DockedPDBBindDataset(Dataset):

    def __init__(
        self,
        data_file,
        cache_data=True,
        h5_file_driver=None,
        distance_cutoff=5.,
    ):
        super().__init__()

        self.data_file = data_file
        self.cache_data = cache_data
        self.data_dict = {
        }  # will use this to store data once it has been computed if cache_data is True

        self.data_list = []  # will use this to store ids for data

        self.h5_file_driver = h5_file_driver
        self.distance_cutoff = distance_cutoff

        with open(data_file.replace('.h5', '.json')) as f:
            f_ = json.load(f)
            names = list(f_['regression'].keys())
            self.data_list = names
        correct_file = '/content/drive/MyDrive/LBA/data/pdbbindv2019_pocket_rmsd_1.json'
        with open(correct_file) as f:
            f_ = json.load(f)
            names = list(f_.keys())
            self.correct_list = names

        self.file = h5py.File(self.data_file, "r")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                return self.data_dict[item]
            else:
                pass

        name = self.data_list[item]
        node_feats = torch.FloatTensor(
            self.file['regression'][name]['pdbbind_sgcnn']['node_feats'])

        affinity = torch.FloatTensor(
            self.file['regression'][name].attrs["affinity"]).reshape(1, -1)
        dock_score = torch.FloatTensor(
            self.file['regression'][name].attrs["dock_score"]).reshape(1, -1)
        correct = torch.BoolTensor([name in self.correct_list]).reshape(1, -1)

        coords = torch.FloatTensor(
            self.file['regression'][name]['pdbbind_sgcnn']['dists'])
        dists = pairwise_distances(coords, metric="euclidean")
        edge_index, edge_attr = dense_to_sparse(
            torch.from_numpy(dists).float())
        mask = edge_attr < self.distance_cutoff
        edge_index = edge_index[:, mask.view(-1)]

        x = node_feats
        data = Data(x=x,
                    coords=coords,
                    edge_index=edge_index,
                    y=dock_score,
                    affinity=affinity,
                    correct=correct,
                    name=name)

        if self.cache_data:
            self.data_dict[item] = data
            return self.data_dict[item]
        else:
            return data


class NoisyPDBBindDataset(PDBBindDataset):

    def __init__(
        self,
        data_file,
        dataset_name,
        feature_type,
        preprocessing_type,
        use_docking=False,
        output_info=False,
        cache_data=True,
        h5_file_driver=None,
        sigma=0,
    ):
        super().__init__(
            data_file,
            dataset_name,
            feature_type,
            preprocessing_type,
            use_docking,
            output_info,
            cache_data,
            h5_file_driver,
        )
        self.sigma = sigma

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                if self.output_info:
                    pdbid, pose, coords, x, y = self.data_dict[item]
                    eps = np.clip(self.sigma * np.random.randn(*coords.shape),
                                  a_min=-2 * self.sigma,
                                  a_max=2 * self.sigma)
                    dists = pairwise_distances(coords + eps,
                                               metric="euclidean")
                    edge_index, edge_attr = dense_to_sparse(
                        torch.from_numpy(dists).float())
                    data = Data(x=x,
                                edge_index=edge_index,
                                edge_attr=edge_attr.view(-1, 1),
                                y=y)
                    return (pdbid, pose, data)
                else:
                    coords, x, y = self.data_dict[item]
                    eps = np.clip(self.sigma * np.random.randn(*coords.shape),
                                  a_min=-2 * self.sigma,
                                  a_max=2 * self.sigma)
                    dists = pairwise_distances(coords + eps,
                                               metric="euclidean")
                    edge_index, edge_attr = dense_to_sparse(
                        torch.from_numpy(dists).float())
                    x = torch.from_numpy(node_feats).float()
                    y = torch.FloatTensor(affinity).view(-1, 1)
                    data = Data(x=x,
                                edge_index=edge_index,
                                edge_attr=edge_attr.view(-1, 1),
                                y=y)
                    return data
            else:
                pass

        pdbid, pose, affinity = self.data_list[item]

        node_feats, coords = None, None
        with h5py.File(self.data_file, "r") as f:

            if (not self.dataset_name in f["{}/{}/{}".format(
                    pdbid, self.feature_type,
                    self.preprocessing_type)].keys()):
                print(pdbid)
                return None

            if self.use_docking:
                # TODO: the next line will cuase runtime error because not selelcting poses
                data = f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )][pose]["data"]
                vdw_radii = (f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )][pose].attrs["van_der_waals"].reshape(-1, 1))

            else:
                data = f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )]["data"]
                vdw_radii = (f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )].attrs["van_der_waals"].reshape(-1, 1))

            if self.feature_type == "pybel":
                coords = data[:, 0:3]
                node_feats = np.concatenate([vdw_radii, data[:, 3:22]], axis=1)

            else:
                raise NotImplementedError

        # account for the vdw radii in distance cacluations (consider each atom as a sphere, distance between spheres)
        eps = np.clip(self.sigma * np.random.randn(*coords.shape),
                      a_min=-2 * self.sigma,
                      a_max=2 * self.sigma)
        dists = pairwise_distances(coords + eps, metric="euclidean")
        edge_index, edge_attr = dense_to_sparse(
            torch.from_numpy(dists).float())
        x = torch.from_numpy(node_feats).float()
        y = torch.FloatTensor(affinity).view(-1, 1)
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr.view(-1, 1),
                    y=y)

        if self.cache_data:
            if self.output_info:
                self.data_dict[item] = (pdbid, pose, coords, x, y)
                return (pdbid, pose, data)
            else:
                self.data_dict[item] = (coords, x, y)
                return data
        else:
            if self.output_info:
                return (pdbid, pose, data)
            else:
                return data


class EdgeNoisePDBBindDataset(PDBBindDataset):

    def __init__(
        self,
        data_file,
        dataset_name,
        feature_type,
        preprocessing_type,
        use_docking=False,
        output_info=False,
        cache_data=True,
        h5_file_driver=None,
        sigma=0,
    ):
        super().__init__(
            data_file,
            dataset_name,
            feature_type,
            preprocessing_type,
            use_docking,
            output_info,
            cache_data,
            h5_file_driver,
        )
        self.sigma = sigma

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                if self.output_info:
                    pdbid, pose, coords, x, y, edge_attr0 = self.data_dict[
                        item]
                    eps = np.clip(self.sigma * np.random.randn(*coords.shape),
                                  a_min=-2 * self.sigma,
                                  a_max=2 * self.sigma)
                    dists = pairwise_distances(coords + eps,
                                               metric="euclidean")
                    edge_index, edge_attr = dense_to_sparse(
                        torch.from_numpy(dists).float())
                    edge_attr0 = edge_attr0.view(-1, 1)
                    edge_attr = edge_attr.view(-1, 1)
                    if not edge_attr0.shape == edge_attr.shape:  #, f'{dists0.shape} != {dists.shape} or {edge_attr0.shape} != {edge_attr.shape}'
                        print(f'Wrong shape for edge_index, skipping {pdbid}')
                        return self[item + 1]
                    edge_attr_combined = torch.cat([edge_attr0, edge_attr],
                                                   dim=1)
                    data = Data(x=x,
                                edge_index=edge_index,
                                edge_attr=edge_attr_combined.view(-1, 2),
                                y=y)
                    return (pdbid, pose, data)
                else:
                    coords, x, y, edge_attr0 = self.data_dict[item]
                    eps = np.clip(self.sigma * np.random.randn(*coords.shape),
                                  a_min=-2 * self.sigma,
                                  a_max=2 * self.sigma)
                    dists = pairwise_distances(coords + eps,
                                               metric="euclidean")
                    edge_index, edge_attr = dense_to_sparse(
                        torch.from_numpy(dists).float())
                    edge_attr0 = edge_attr0.view(-1, 1)
                    edge_attr = edge_attr.view(-1, 1)
                    if not edge_attr0.shape == edge_attr.shape:  #, f'{dists0.shape} != {dists.shape} or {edge_attr0.shape} != {edge_attr.shape}'
                        print(f'Wrong shape for edge_index, skipping {pdbid}')
                        return self[item + 1]
                    edge_attr_combined = torch.cat([edge_attr0, edge_attr],
                                                   dim=1)
                    data = Data(x=x,
                                edge_index=edge_index,
                                edge_attr=edge_attr_combined.view(-1, 2),
                                y=y)
                    return data
            else:
                pass

        pdbid, pose, affinity = self.data_list[item]

        node_feats, coords = None, None
        with h5py.File(self.data_file, "r") as f:

            if (not self.dataset_name in f["{}/{}/{}".format(
                    pdbid, self.feature_type,
                    self.preprocessing_type)].keys()):
                print(pdbid)
                return None

            if self.use_docking:
                # TODO: the next line will cuase runtime error because not selelcting poses
                data = f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )][pose]["data"]
                vdw_radii = (f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )][pose].attrs["van_der_waals"].reshape(-1, 1))

            else:
                data = f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )]["data"]
                vdw_radii = (f["{}/{}/{}/{}".format(
                    pdbid,
                    self.feature_type,
                    self.preprocessing_type,
                    self.dataset_name,
                )].attrs["van_der_waals"].reshape(-1, 1))

            if self.feature_type == "pybel":
                coords = data[:, 0:3]
                node_feats = np.concatenate([vdw_radii, data[:, 3:22]], axis=1)

            else:
                raise NotImplementedError

        # account for the vdw radii in distance cacluations (consider each atom as a sphere, distance between spheres)
        dists0 = pairwise_distances(coords, metric="euclidean")
        edge_index, edge_attr0 = dense_to_sparse(
            torch.from_numpy(dists0).float())
        x = torch.from_numpy(node_feats).float()
        y = torch.FloatTensor(affinity).view(-1, 1)
        eps = np.clip(self.sigma * np.random.randn(*coords.shape),
                      a_min=-2 * self.sigma,
                      a_max=2 * self.sigma)
        dists = pairwise_distances(coords + eps, metric="euclidean")
        edge_index, edge_attr = dense_to_sparse(
            torch.from_numpy(dists).float())
        edge_attr0 = edge_attr0.view(-1, 1)
        edge_attr = edge_attr.view(-1, 1)
        if not edge_attr0.shape == edge_attr.shape:  #, f'{dists0.shape} != {dists.shape} or {edge_attr0.shape} != {edge_attr.shape}'
            print(f'Wrong shape for edge_index, skipping {pdbid}')
            return self[item + 1]
        edge_attr_combined = torch.cat([edge_attr0, edge_attr], dim=1)
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr_combined.view(-1, 2),
                    y=y)

        if self.cache_data:
            if self.output_info:
                self.data_dict[item] = (pdbid, pose, coords, x, y, edge_attr0)
                return (pdbid, pose, data)
            else:
                self.data_dict[item] = (coords, x, y, edge_attr0)
                return data
        else:
            if self.output_info:
                return (pdbid, pose, data)
            else:
                return data
'''