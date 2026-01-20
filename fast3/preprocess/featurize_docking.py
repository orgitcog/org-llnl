from torch.utils.data import Dataset
import json

import h5py

import numpy as np

import torch

from llnlvision.utils.data_utils import LLNLDataUtils
import scipy.ndimage
import scipy as sp

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
#from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from cupy.cuda.nvtx import RangePush, RangePop
import sys
import glob
# sys.path.insert(1, '/p/gpfs1/gstevens/horovod/aha_distributed_torch/covid19/ml_data/')

from conveyorLC_util import *
import os
import pandas as pd

def get_3D_bound(xyz_array):
    xmin = min(xyz_array[:, 0])
    ymin = min(xyz_array[:, 1])
    zmin = min(xyz_array[:, 2])
    xmax = max(xyz_array[:, 0])
    ymax = max(xyz_array[:, 1])
    zmax = max(xyz_array[:, 2])
    return xmin, ymin, zmin, xmax, ymax, zmax

def get_3D_all2(xyz, feat, vol_dim, relative_size=True, size_angstrom=48, atom_radii=None, atom_radius=1, sigma=0):
    # get 3d bounding box
    xmin, ymin, zmin, xmax, ymax, zmax = get_3D_bound(xyz)
    print(xmin, ymin, zmin, xmax, ymax, zmax)
    # initialize volume
    vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)

    if relative_size:
            # voxel size (assum voxel size is the same in all axis
        vox_size = float(zmax - zmin) / float(vol_dim[0])
    else:
        vox_size = float(size_angstrom) / float(vol_dim[0])
        xmid = (xmin + xmax) / 2.0
        ymid = (ymin + ymax) / 2.0
        zmid = (zmin + zmax) / 2.0
        xmin = xmid - (size_angstrom / 2)
        ymin = ymid - (size_angstrom / 2)
        zmin = zmid - (size_angstrom / 2)
        xmax = xmid + (size_angstrom / 2)
        ymax = ymid + (size_angstrom / 2)
        zmax = zmid + (size_angstrom / 2)
        vox_size2 = float(size_angstrom) / float(vol_dim[0])
        #print(vox_size, vox_size2)

    # assign each atom to voxels
    for ind in range(xyz.shape[0]):
        x = xyz[ind, 0]
        y = xyz[ind, 1]
        z = xyz[ind, 2]
        if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
            continue
        # compute van der Waals radius and atomic density, use 1 if not available
        if not atom_radii is None:
            vdw_radius = atom_radii[ind]
            atom_radius = 1 + vdw_radius * vox_size

        cx = (x - xmin) / (xmax - xmin) * (vol_dim[2] - 1)
        cy = (y - ymin) / (ymax - ymin) * (vol_dim[1] - 1)
        cz = (z - zmin) / (zmax - zmin) * (vol_dim[0] - 1)

        vx_from = max(0, int(cx - atom_radius))
        vx_to = min(vol_dim[2] - 1, int(cx + atom_radius))
        vy_from = max(0, int(cy - atom_radius))
        vy_to = min(vol_dim[1] - 1, int(cy + atom_radius))
        vz_from = max(0, int(cz - atom_radius))
        vz_to = min(vol_dim[0] - 1, int(cz + atom_radius))

        for vz in range(vz_from, vz_to + 1):
            for vy in range(vy_from, vy_to + 1):
                for vx in range(vx_from, vx_to + 1):
                    vol_data[vz, vy, vx, :] += feat[ind, :]

    # gaussian filter
    if sigma > 0:
        for i in range(vol_data.shape[-1]):
            vol_data[:,:,:,i] = sp.ndimage.filters.gaussian_filter(vol_data[:,:,:,i], sigma=sigma, truncate=2)

    return vol_data


set_json = '/p/lustre1/gstevens/horovod/data2/test.json'
outfile = 'pdbbind2020_docking_refined.h5'
# outfile = '/usr/workspace/atom/ml_fusion/pdbbind2019_for_hunter/w_coords/pdbbind2019_docking_w_coords.h5'

dock_hdf = h5py.File(outfile, 'a')
# reg_group = dock_hdf.create_group('regression')
test_h5 = h5py.File('/p/lustre1/gstevens/horovod/data2/test.h5', 'r')
val_h5 = h5py.File('/p/lustre1/gstevens/horovod/data2/val.h5', 'r')
train_h5 = h5py.File('/p/lustre1/gstevens/horovod/data2/train.h5', 'r')

with open(set_json, "r") as opened_json_file:
    json_data = json.load(opened_json_file)

new_json = {'regression': {}}
missing_ids = []
# missing_ids = ['1c5z', '1o5b', '3aru', '3arv', '3ary', '3d6q', '4m0z', '4u4s', '4x6p', '3dxg' , '3f3a','3g2n' , '3l7b', '3nw9','3pxf' ,'3syr' ,'4de3' , '4dli' , '4eky']
# test_csv = pd.read_csv('/p/lustre1/zhang30/pdbbind/pdbbind_2019_pocket/test.csv')
test_csv = pd.read_csv('/p/lustre2/zhang30/PDBBIND2020/pdbbind2020/dock.csv')

print(test_csv)
# print(len(json_data['regression']))
# all_pockets = glob.glob('/p/lustre1/zhang30/pdbbind/pdbbind_2019_pocket/pdb/*_pocket.pdb')
all_pockets = glob.glob('/p/lustre1/gstevens/aha_data/flat/pdbbind_v2020/**/*_pocket.pdb')
for i in range(len(all_pockets)):
    temp = all_pockets[i]
    temp = temp[temp.rfind('/')+1:temp.rfind('_')]
    all_pockets[i] = temp#.replace('/p/lustre1/zhang30/pdbbind/pdbbind_2019_pocket/pdb/', '').replace('_pocket.pdb', '')
# for key in json_data['regression']:
training = 0
valid = 0
test = 0
# print(all_pockets)
for key in all_pockets:
# for key in json_data['regression']:
    print(key)
    if key in missing_ids:
        continue
    elif key+'_'+str(1) in list(dock_hdf.keys()):
        continue


    # if key in test_h5['regression'].keys():
    #     # affinity = np.squeeze(np.asarray(test_h5["{}/{}".format('regression' , key)].attrs['affinity']))
    #     test += 1
    # elif key in val_h5['regression'].keys():
    #     # affinity = np.squeeze(np.asarray(val_h5["{}/{}".format('regression' , key)].attrs['affinity']))
    #     valid += 1
    # else:
    #     # affinity = np.squeeze(np.asarray(train_h5["{}/{}".format('regression' , key)].attrs['affinity']))
    #     training += 1
    dockingfile = test_csv.loc[test_csv[' ligName'].str.contains(key), 'file']
    if len(dockingfile) < 1:
        continue
    dockingfile = dockingfile.item().replace('/p/vast1/zhang30/PDBBIND/', '/p/lustre2/zhang30/PDBBIND2020/')
    os.system('grep -v "HOH" '+ '/p/lustre1/gstevens/aha_data/flat/pdbbind_v2020/'+str(key)+'/'+str(key)+'_pocket.pdb' + ' > ' + '/p/lustre1/gstevens/aha_data/flat/pdbbind_v2020/'+str(key)+'/'+str(key)+'_pocket_no_water.pdb')

    featurizer = ComplexFeaturizer('/p/lustre1/gstevens/aha_data/flat/pdbbind_v2020/'+str(key)+'/'+str(key)+'_pocket_no_water.pdb', '/usr/workspace/atom/glo_spl/fusion/covid19/ml_data/elements.xml', include_vdw=True)
    path = test_csv.loc[test_csv[' ligName'].str.contains(key), ' key'].item()
    with DockHDF_pdbqt(dockingfile, 'r') as h5_file_object:
        poses = h5_file_object.get_pdbqt_pybel(path[path[:path.rfind('/')].rfind('/')+1:path.rfind('/')], path[path.rfind('/')+1:], compute_partial_charge=True)
        for pose_ind, pose in enumerate(poses):
            lig_id_group = dock_hdf.create_group(key+'_'+str(pose_ind+1))
            # new_json['regression'][key+'_'+str(pose_ind+1)] = {}
            # new_json['regression'][key+'_'+str(pose_ind+1)]['pybel'] = {}

            # new_json['regression'][key+'_'+str(pose_ind+1)]['pybel']['processed'] = {}
            vina = test_csv.loc[test_csv[' ligName'] == ' ' + str(key)+'_ligand', ' scores/'+str(pose_ind+1)]
            if len(vina) == 1:
                vina = vina.item()
            try:
                lig_id_group.attrs['dock_score'] = [float(vina)]
            except:
                pass
            
            node_feats, coords = None, None
            pose_feature = featurizer.featurize(pose)
            coords = pose_feature[:, 0:3]
            node_feats = np.concatenate([pose_feature[:,-1][...,np.newaxis], pose_feature[:, 3:-1]], axis=1)
            dists = squareform(pdist(coords,'euclidean'))
            # dists = coords
            input_radii = None
            cnn3d_3D_dim = [48, 48, 48, 19]
            cnn3d_3D_relative_size = False
            cnn3d_3D_size_angstrom = 48 # 48, valid only when g_3D_relative_size = False
            cnn3d_3D_atom_radius = 1
            cnn3d_3D_sigma = 1
            output_3d_data = get_3D_all2(coords, node_feats[:,1:], cnn3d_3D_dim, cnn3d_3D_relative_size, cnn3d_3D_size_angstrom, input_radii, cnn3d_3D_atom_radius, cnn3d_3D_sigma)
            print(np.min(output_3d_data))
            print(np.max(output_3d_data))
            cnn3d_data = torch.from_numpy(np.moveaxis(output_3d_data, -1 ,0))
            cnn_group = lig_id_group.create_group("voxelized")
            sgcnn_group = lig_id_group.create_group("spatial")
            # new_json['regression'][key+'_'+str(pose_ind+1)]['pybel']['processed']['pdbbind_cnn3d'] = {"data0": {}}
            # new_json['regression'][key+'_'+str(pose_ind+1)]['pybel']['processed']['pdbbind_sgcnn'] = {"data0": {}}
            sgcnn_group.create_dataset("coords", data=coords,
                                                shape=coords.shape, dtype='float32', compression='lzf')
            cnn_group.create_dataset("data0", data=cnn3d_data, shape=cnn3d_data.shape, dtype='float32', compression='lzf')
            sgcnn_group.create_dataset("dists", data=dists, shape=dists.shape, dtype='float32', compression='lzf')
            sgcnn_group.create_dataset("node_feats", data=node_feats, shape=node_feats.shape, dtype='float32', compression='lzf')

            # if key in test_h5['regression'].keys():
            #     affinity = np.squeeze(np.asarray(test_h5["{}/{}".format('regression' , key)].attrs['affinity']))
            #     # test += 1
            # elif key in val_h5['regression'].keys():
            #     affinity = np.squeeze(np.asarray(val_h5["{}/{}".format('regression' , key)].attrs['affinity']))
            #     # valid += 1
            # else:
            #     affinity = np.squeeze(np.asarray(train_h5["{}/{}".format('regression' , key)].attrs['affinity']))
            #     # training += 1
            # lig_id_group.attrs['affinity'] = [float(affinity)]

dock_hdf.close()
for key in missing_ids:
    del json_data['regression'][key]
print(training, "Train")
print(valid, "Val")
print(test, "Test")

# with open('/p/lustre1/gstevens/all_docking_3d.json', 'w') as fp:
#     json.dump(new_json, fp, sort_keys=True, indent=4)
# outfile = '/p/lustre1/gstevens/horovod/data2/test_docking.h5'
# if os.path.isfile(outfile):
#     try:
#         dock_hdf = h5py.File(outfile, 'r+')
#     except OSError as e:
#         os.remove(outfile)
#         dock_hdf = h5py.File(outfile, 'a')
# else:

#     dock_hdf = h5py.File(outfile, 'a')

# class CovidDocking(Dataset):

#     def __init__(self, dataset, seed, h5_file, json_file, featurization, variation, representation, observations, transforms=None, transform=None, target_transform=None, problem_type='muti-class', data_manipulations=[]):
#         super(CovidDocking, self).__init__()

#         self.dataset = dataset
#         self.h5_file = h5_file
#         self.json_file = json_file
#         self.featurization = featurization
#         self.variation = variation
#         self.observations = observations
#         self.representation = representation
#         self.seed = seed
#         self.problem_type = problem_type
#         self.data_manipulations = data_manipulations
#         self.featurizer = 0
#         has_transforms = transforms is not None
#         has_separate_transform = transform is not None or target_transform is not None
#         if has_transforms and has_separate_transform:
#             raise ValueError("Only transforms or transform/target_transform can "
#                              "be passed as argument")

#         # for backwards-compatibility
#         self.transform = transform
#         self.target_transform = target_transform

#         if has_separate_transform:
#             transforms = StandardTransform(transform, target_transform)
#         self.transforms = transforms


#         self.output_info = False #TODO: Not an option from command line yet

#         data_list = []  # will use this to store ids for data

#         with open(self.json_file, "r") as opened_json_file:
#             json_data = json.load(opened_json_file)

#         self.num_groups = len(json_data)
#         self.groups = []

#         self.representations_to_combine = []
#         counter = 0
#         stop = False
#         self.bind_to_dirs = []
#         self.dock_hdf_names = []
#         self.dock_ids = []
#         self.lig_ids = []
#         for bind_to_dir in json_data:
#             if not bind_to_dir in self.bind_to_dirs:
#                 self.bind_to_dirs.append(bind_to_dir)
#             for dock_hdf_name in json_data[bind_to_dir]:
#                 if not dock_hdf_name in self.dock_hdf_names:
#                     self.dock_hdf_names.append(dock_hdf_name)
#                 for dock_id in json_data[bind_to_dir][dock_hdf_name]:
#                     if not dock_id in self.dock_ids:
#                         self.dock_ids.append(dock_id)
#                     for lig_id in json_data[bind_to_dir][dock_hdf_name][dock_id]:
#                         if not lig_id in self.lig_ids:
#                             self.lig_ids.append(lig_id)
#         for bind_to_dir in json_data:
#             btd = self.bind_to_dirs.index(bind_to_dir)
#             bind_json = json_data[bind_to_dir]
#         #    half = 350000
#             for i, dock_hdf_name in enumerate(bind_json):
#                 dhf = self.dock_hdf_names.index(dock_hdf_name)
#                 dock_hdf_json = bind_json[dock_hdf_name]
#          #       if stop == True:
#           #          break
#                 for x, dock_id in enumerate(dock_hdf_json):
#                     dckid = self.dock_ids.index(dock_id)
#                     dock_id_json = dock_hdf_json[dock_id]
#                     for j, lig_id in enumerate(dock_id_json):
#                         li = self.lig_ids.index(lig_id)
# #          if counter > (half*18):
#           #              stop = True
#           #              break
#                         lig_json = dock_id_json[lig_id]
#                         for pose_id in lig_json:
#                             counter += 1
#           #              if counter < (half*17):
#           #                  continue
#                             data_list.append((int(btd), int(dhf), int(dckid), int(li), int(pose_id)))
#         # for group, group_name in enumerate(json_data):
#         #     group_json = json_data[group_name]
#         #     self.groups.append(group_name)
#         #     for identifier in group_json:
#         #         identifier_json = group_json[identifier]
#         #         for json_featurization_name in identifier_json:
#         #             if featurization == json_featurization_name:
#         #                 featurization_json = identifier_json[featurization]
#         #                 for json_variation_name in featurization_json:
#         #                     variation_found = False
#         #                     if variation == 'all':
#         #                         variation_json = featurization_json[json_variation_name]
#         #                         variation_found = True
#         #                     elif variation == json_variation_name:
#         #                         variation_json = featurization_json[variation]
#         #                         variation_found = True
#         #                     if variation_found:
#         #                         representation_to_index_by = ''
#         #                         if representation == 'all':
#         #                             for json_representation_name in variation_json:
#         #                                 if json_representation_name != 'gt' and representation == 'all' or representation == json_representation_name:
#         #                                     representation_to_index_by = json_representation_name
#         #                                     if not json_representation_name in self.representations_to_combine:
#         #                                         self.representations_to_combine.append(json_representation_name)
#         #                         else:
#         #                             representation_to_index_by = representation
#         #                         if representation_to_index_by != '':
#         #                             representation_json = variation_json[representation_to_index_by]
#         #                             for json_observation_name in representation_json:
#         #                                 if observations == 'all':
#         #                                     data_list.append((identifier, json_observation_name, group, len(data_list), json_variation_name))
#         #                                 elif json_observation_name == observations:
#         #                                     data_list.append((identifier, json_observation_name, group, len(data_list), json_variation_name))


#         tgt_idx = self.bind_to_dirs[0].find('_')
#         self.tgt = self.bind_to_dirs[0][:tgt_idx]
#         self.data_np = np.array(data_list)
#        # self.featurizer = ComplexFeaturizer('/p/vast1/gstevens/cut8_targets/'+tgt+'_cut8.pdb', '/p/gpfs1/gstevens/horovod/aha_distributed_torch/covid19/ml_data/elements.xml')

#     def __len__(self):
#         print(self.dataset + " samples #: " + str(len(self.data_np)))

#         return len(self.data_np)

#     def __getitem__(self, item):
#         RangePush("__getitem__")
#         if self.featurizer == 0:
#             self.featurizer = ComplexFeaturizer('/p/gpfs1/gstevens/cut8_targets/'+self.tgt+'_cut8.pdb', '/p/gpfs1/gstevens/horovod/aha_distributed_torch/covid19/ml_data/elements.xml', check_partial_charge=False, include_vdw=True)
#         bind_to_dir, dock_hdf_name, dock_id, lig_id, pose_id = self.data_np[item]
#         btd = self.bind_to_dirs[bind_to_dir]
#         tgt_idx = btd.find('_')
#         tgt = btd[:tgt_idx]
#         part_idx = btd.rfind('_')
#         part_start = btd[:part_idx].rfind('_')
#         part = btd[part_start+1:part_idx]
#         with DockHDF_pdbqt('/p/gpfs1/gstevens/raw_docking/'+ tgt +'/en/'+ part+ '/' + btd[part_idx+1:] + '/scratch/dockHDF5/' + self.dock_hdf_names[dock_hdf_name] + '.hdf5', 'r') as h5_file_object:
#         #with h5py.File('/usr/WS2/atom/ml_fusion/pdbbind_eval/'+ bind_to_dir + '/' + dock_hdf_name + '.hdf', 'r') as h5_file_object:
#             second_level = list(h5_file_object.hdf_file['dock'].keys())[0]
#             poses = h5_file_object.get_pdbqt_pybel(second_level, self.dock_ids[dock_id], compute_partial_charge=True)
#             pose_feature = self.featurizer.featurize(poses[pose_id-1])
#             node_feats, coords = None, None
            
#             sg_data_raw = pose_feature
#             #vdw_radii = self.featurizer.parse_vdw(poses[pose_id-1],self.featurizer.element_dict).reshape(-1,1)
#             coords = sg_data_raw[:, 0:3]
#             node_feats = np.concatenate([sg_data_raw[:,-1][...,np.newaxis], sg_data_raw[:, 3:-1]], axis=1)
             
#             # else:
#             #     raise NotImplementedError
#             dists = squareform(pdist(coords,'euclidean'))

#             edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())

#             x = torch.from_numpy(node_feats).float()

#             y = torch.FloatTensor([0]).view(-1,1)

#             #input_feat = input_data_[:,3:]
#             input_radii = False
#             cnn3d_3D_dim = [48, 48, 48, 19]
#             cnn3d_3D_relative_size = False
#             cnn3d_3D_size_angstrom = 0 # 48, valid only when g_3D_relative_size = False
#             cnn3d_3D_atom_radius = 1
#             cnn3d_3D_sigma = 1
#             output_3d_data = get_3D_all2(coords, node_feats, cnn3d_3D_dim, cnn3d_3D_relative_size, cnn3d_3D_size_angstrom, input_radii, cnn3d_3D_atom_radius, cnn3d_3D_sigma)
#             cnn3d_data = torch.from_numpy(np.moveaxis(output_3d_data, -1 ,0))
#             #x = torch.from_numpy(np.array(h5_file_object["{}/{}/{}/{}".format(self.groups[int(group)], identifier,'pdbbind_sgcnn', "node_feats")])).float()

#             #y = torch.FloatTensor(affinity).view(-1,1)
#             input_radii = None
#             sg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1,1), y=y)




#             #cnn3d_data = torch.as_tensor(np.array(h5_file_object["{}/{}/{}/{}".format(self.groups[int(group)], identifier, 'pdbbind_cnn3d', observation_name)]))
#             data = np.array([sg_data, cnn3d_data])

#             # if not 'aha' in self.data_manipulations:
#             #     return data
#             # else:
#             # # if self.output_info:
#             # #     self.data_dict[item] = (identifier, observation_name, group)

#             # # else:
#             # #     self.data_dict[item] = data

#             return np.array([data, bind_to_dir, dock_hdf_name, lig_id, pose_id])
