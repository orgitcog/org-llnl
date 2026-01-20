################################################################################
# Copyright 2019-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# File utility functions
################################################################################

import os
import sys
import math
import numbers
import numpy as np
import scipy as sp
import torch
import torch.nn as nn

from models.base import BaseModel
from models.voxelizer import Voxelizer3D, GaussianFilter


class Conv3DNet(BaseModel):

    # num_filters=[64,128,256] or [96,128,128]
    def __init__(
        self,
        in_channels,
        out_channels,
        loss_fn,
        preprocess=True,
        num_filters=[64, 128, 256],
        deeper=False,
        verbose=0
    ):
        super(Conv3DNet, self).__init__(in_channels, out_channels, loss_fn)

        self.preprocess = preprocess
        self.num_filters = num_filters
        self.deeper = deeper
        self.verbose = verbose

        if self.preprocess:
            self.voxelizer = Voxelizer3D()
            self.gaussian_filter = GaussianFilter(dim=3, channels=in_channels, kernel_size=11, sigma=1)

        self.conv_block1 = self.__conv_layer_set__(self.in_channels, self.num_filters[0], 7, 2, 3)
        self.res_block1 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)
        self.res_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)

        self.conv_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[1], 7, 3, 3)
        self.max_pool2 = nn.MaxPool3d(2)

        self.conv_block3 = self.__conv_layer_set__(self.num_filters[1], self.num_filters[2], 5, 2, 2)
        self.max_pool3 = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(2048, 100)
        torch.nn.init.normal_(self.fc1.weight, 0, 1)
        self.fc1_bn = nn.BatchNorm1d(num_features=100, affine=True, momentum=0.1) # .train()

        if deeper:
            self.fc2 = nn.Linear(100, 10)
            torch.nn.init.normal_(self.fc2.weight, 0, 1)
            self.fc2_bn = nn.BatchNorm1d(num_features=10, affine=True, momentum=0.1)
            self.fc3 = nn.Linear(10, 1)
            torch.nn.init.normal_(self.fc3.weight, 0, 1)
        else:
            self.fc2 = nn.Linear(100, 1)
            torch.nn.init.normal_(self.fc2.weight, 0, 1)

        self.relu = nn.ReLU()
        #self.drop=nn.Dropout(p=0.15)

    def __conv_layer_set__(self, in_c, out_c, k_size, stride, padding):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_c))
        return conv_layer

    def _forward(self, data, get_feature=False):
        x = data.x_3d

        if self.preprocess:
            vol_batch = torch.zeros((x.shape[0],19,48,48,48)).float().to(x.device)
            for i in range(x.shape[0]):
                xyz, feat = x[i,:,:3], x[i,:,3:]
                vol_batch[i,:,:,:,:] = self.voxelizer(xyz, feat)
            vol_batch = self.gaussian_filter(vol_batch)
            x = vol_batch[:x.shape[0]]

        if x.dim() == 1:
            x = x.unsqueeze(-1)

        conv1_h = self.conv_block1(x)
        if self.verbose != 0:
            print(conv1_h.shape)

        conv1_res1_h = self.res_block1(conv1_h)
        if self.verbose != 0:
            print(conv1_res1_h.shape)

        conv1_res1_h2 = conv1_res1_h + conv1_h
        if self.verbose != 0:
            print(conv1_res1_h2.shape)

        conv1_res2_h = self.res_block2(conv1_res1_h2)
        if self.verbose != 0:
            print(conv1_res2_h.shape)

        conv1_res2_h2 = conv1_res2_h + conv1_h
        if self.verbose != 0:
            print(conv1_res2_h2.shape)

        conv2_h = self.conv_block2(conv1_res2_h2)
        if self.verbose != 0:
            print(conv2_h.shape)

        pool2_h = self.max_pool2(conv2_h)
        if self.verbose != 0:
            print(pool2_h.shape)

        conv3_h = self.conv_block3(pool2_h)
        if self.verbose != 0:
            print(conv3_h.shape)

        pool3_h = conv3_h
        #pool3_h = self.max_pool3(conv3_h)
        #if self.verbose != 0:
        #    print(pool3_h.shape)

        flatten_h = pool3_h.view(pool3_h.size(0), -1)
        if self.verbose != 0:
            print(flatten_h.shape)

        fc1_z = self.fc1(flatten_h)
        fc1_y = self.relu(fc1_z)
        fc1_h = self.fc1_bn(fc1_y) if fc1_y.shape[0] > 1 else fc1_y  #batchnorm train require more than 1 batch
        if self.verbose != 0:
            print(fc1_h.shape)

        if self.deeper:
            fc2_z = self.fc2(fc1_h)
            if get_feature: return fc2_z
            fc2_y = self.relu(fc2_z)
            fc2_h = self.fc2_bn(fc2_y) if fc2_y.shape[0] > 1 else fc2_y
            fc3_z = self.fc3(fc2_h)
            return fc3_z
        else:
            if get_feature: return fc1_z
            fc2_z = self.fc2(fc1_h)
            if self.verbose != 0:
                print(fc2_z.shape)
            return fc2_z
