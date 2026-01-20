from torch.utils.data import Dataset
import json
import h5py
import numpy as np
import torch
import scipy.ndimage
import scipy as sp

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import sys
# sys.path.insert(1, '../covid19/ml_data/')
sys.path.insert(1, '/usr/workspace/atom/glo_spl/fusion/covid19/ml_data')

from conveyorLC_util import *
import pandas as pd
from pathlib import Path
import bz2

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
                #     print("here!")
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



# outfile = 'gmd_test_protease2_update_no_voxel.h5'
outfile = 'gmd_protese2_docking_v4_no_voxel.h5'
dock_hdf = h5py.File(outfile, 'w')
# reg_group = dock_hdf.create_group('regression')
# new_json = {}
# print(list(dock_hdf.keys()))
dock_procs = []
# for path in Path('/usr/workspace/atom/PostEra/postera_mpro_posneg/protease1_neg').rglob('dock_proc*.hdf5'):
# for path in Path('/p/lustre1/allen99/gmdruns/mpro_virtual_screening/postera_mpro_exp3neg2').rglob('dock_proc*.hdf5'):
# for path in Path('/usr/WS2/atom/glo_spl/gmd_mpro_new_leads_vscreen_1/results/structures_corrected').rglob('coval*.hdf5'):

for path in Path('/usr/workspace/atom/mpro_inhibitors/lead_compounds/updated_dset/docking').rglob('*.hdf5'):

    dock_procs.append(path)
for dp in dock_procs:
    print(dp)
    featurizer = ComplexFeaturizer('/usr/workspace/atom/glo_spl/fusion/protease2_cut8.pdb', '/usr/workspace/atom/glo_spl/fusion/covid19/ml_data/elements.xml', include_vdw=True)
    dockingfile = dp
    with h5py.File(dockingfile, 'r') as first_open_obj:
        print(list(first_open_obj.keys()))
        # dock_ids = list(first_open_obj['dock/coval'].keys())
        # lig_names = []
        # for dock_id in dock_ids:
        #         lig_name = ''.join(np.array(first_open_obj['dock/coval/'+ str(dock_id) + '/meta/ligName']).astype(str))
        #         lig_names.append(lig_name)
        lig_names = []
        dock_ids = []
        for i in range(first_open_obj['metadata'][:].shape[0]):
                mdata = json.loads(bz2.decompress(first_open_obj['metadata'][i].tobytes()).decode())
                lig_names.append(mdata['ligand_name'])
                dock_ids.append(int(mdata['data_id']))

        print(len(lig_names))
    with DockHDF_pdbqt(dockingfile, 'r') as h5_file_object:
        for idx, dock_id in enumerate(dock_ids):
                # dock_id = str(dock_id)
                lig_name = lig_names[idx]
                poses = h5_file_object.get_pdbqt_pybel2('pdbqt', dock_id, compute_partial_charge=True)
                for pose_ind, pose in enumerate(poses):
                        print(dock_id, pose_ind, lig_name)    
                        # if not lig_name+'_protease2_'+str(pose_ind+1) in list(dock_hdf.keys()):
                        lig_id_group = dock_hdf.create_group(lig_name+'_protease2_'+str(pose_ind+1))
                                # new_json['regression'][lig_name+'_protease_'+str(pose_ind+1)] = {}
#                                 # new_json['regression'][lig_name+'_protease_'+str(pose_ind+1)]['pybel'] = {}
#                                 # new_json['regression'][lig_name+'_protease_'+str(pose_ind+1)]['pybel']['processed'] = {}

                        node_feats, coords = None, None
                        pose_feature = featurizer.featurize(pose)
                        coords = pose_feature[:, 0:3]
                        node_feats = np.concatenate([pose_feature[:,-1][...,np.newaxis], pose_feature[:, 3:-1]], axis=1)
                        dists = squareform(pdist(coords,'euclidean'))
                        print(coords.shape, node_feats.shape)

                        input_radii = None
                        cnn3d_3D_dim = [48, 48, 48, 19]
                        cnn3d_3D_relative_size = False
                        cnn3d_3D_size_angstrom = 48 # 48, valid only when g_3D_relative_size = False
                        cnn3d_3D_atom_radius = 1
                        cnn3d_3D_sigma = 1
                        # output_3d_data = get_3D_all2(coords, node_feats[:,1:], cnn3d_3D_dim, cnn3d_3D_relative_size, cnn3d_3D_size_angstrom, input_radii, cnn3d_3D_atom_radius, cnn3d_3D_sigma)
                        # # # print(np.min(output_3d_data))
                        # # # print(np.max(output_3d_data))
                        # cnn3d_data = torch.from_numpy(np.moveaxis(output_3d_data, -1 ,0))
                        # cnn_group = lig_id_group.create_group("voxelized")
                        sgcnn_group = lig_id_group.create_group("spatial")
                        # # new_json['regression'][lig_name+'_protease_'+str(pose_ind+1)]['pybel']['processed']['pdbbind_cnn3d'] = {"data0": {}}
                        # # new_json['regression'][lig_name+'_protease_'+str(pose_ind+1)]['pybel']['processed']['pdbbind_sgcnn'] = {"data0": {}}
                        # cnn_group.create_dataset("data0", data=cnn3d_data, shape=cnn3d_data.shape, dtype='float32', compression='lzf')
                        sgcnn_group.create_dataset("dists", data=dists, shape=dists.shape, dtype='float32', compression='lzf')
                        sgcnn_group.create_dataset("node_feats", data=node_feats, shape=node_feats.shape, dtype='float32', compression='lzf')
                        sgcnn_group.create_dataset("coords", data=coords, shape=coords.shape, dtype='float32', compression='lzf')

dock_hdf.close()

# with open('gmd_protease_docking3.json', 'w') as fp:
#     json.dump(new_json, fp, sort_keys=True, indent=4)