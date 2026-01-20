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
sys.path.insert(1, '../covid19/ml_data/')

from conveyorLC_util import *
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
                    print("here!")
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


for pocket_id, dock_id in zip(['protease'], ['493124']):
    print(pocket_id)
    featurizer = ComplexFeaturizer('/usr/workspace/atom/glo_spl/fusion/protease_cut8.pdb', '/usr/workspace/atom/glo_spl/fusion/covid19/ml_data/elements.xml', include_vdw=True)
    dockingfile = '/usr/workspace/atom/glo_spl/docking/scratch/dockHDF5/dock_proc1.hdf5'
    with DockHDF_pdbqt(dockingfile, 'r') as h5_file_object:
        poses = h5_file_object.get_pdbqt_pybel(pocket_id, dock_id, compute_partial_charge=True)
        for pose_ind, pose in enumerate(poses):
            node_feats, coords = None, None
            pose_feature = featurizer.featurize(pose)
            coords = pose_feature[:, 0:3]
            node_feats = np.concatenate([pose_feature[:,-1][...,np.newaxis], pose_feature[:, 3:-1]], axis=1)
            dists = squareform(pdist(coords,'euclidean'))
            print(coords.shape, node_feats.shape)

            input_radii = None
            cnn3d_3D_dim = [48, 48, 48, 19]
            cnn3d_3D_relative_size = False
            cnn3d_3D_size_angstrom = 32 # 48, valid only when g_3D_relative_size = False
            cnn3d_3D_atom_radius = 1
            cnn3d_3D_sigma = 1
            output_3d_data = get_3D_all2(coords, node_feats[:,1:], cnn3d_3D_dim, cnn3d_3D_relative_size, cnn3d_3D_size_angstrom, input_radii, cnn3d_3D_atom_radius, cnn3d_3D_sigma)
            print(np.min(output_3d_data))
            print(np.max(output_3d_data))
            cnn3d_data = torch.from_numpy(np.moveaxis(output_3d_data, -1 ,0))
            
