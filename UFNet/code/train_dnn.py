#!/usr/bin/env python
# coding: utf-8
# In[ ]:

#This code was used to train the DNN model to simulate the CDF in each grid


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py
import random
import numpy as np


# step 1. data loaded
B = np.load('./data/obs_all_cesmgrid.npy')
train_obs_Y = B[:16425]
test_obs_Y = B[16425:]

C = np.load('./data/e3smprep_all_cesmgrid.npy')
train_sim_Y = C[:16425]
test_sim_Y = C[16425:]

A = np.load('./data/Dynamics_ens1.npy')
train_sim_X = A[:16425]
test_sim_X = A[16425:]

trainObser_y = train_obs_Y
validationObser_y = test_obs_Y
trainGCM_x = train_sim_X
trainGCM_y = train_sim_Y
validationGCM_x = test_sim_X
validationGCM_y = test_sim_Y
obs_y_all = np.concatenate((trainObser_y,validationObser_y),axis = 0)


# step 2. Generate training and test dataset for DNN
batchsize = 100
seed_n=47
np.random.seed(seed_n)
torch.manual_seed(seed_n)
length = np.shape(trainObser_y)[0]
trainGCM_x1 = []
trainGCM_y1 = []
trainObser_y1 = []
for j in range(1):
    for i in tqdm(range(int(length / batchsize))):  #
        import random
        random.seed(seed_n + i)
        base = random.sample(range(0, length - 1), batchsize)
        daylag = random.sample(range(-15, 15), 1)[0]
        yearlag = random.sample(range(-5, 5), 1)[0]
        tempt = [x + daylag + yearlag * 365 for x in base]
        choice = [x if (x >= 0 and x < length) else x - daylag - yearlag * 365 for x in tempt]
        if trainGCM_x1 == []:
            trainGCM_x1 = trainGCM_x[base, :, :, :]
            trainGCM_y1 = trainGCM_y[choice, :, :].squeeze(1)
            trainObser_y1 = trainObser_y[choice, :, :].squeeze(1)
        else:
            trainGCM_x1 = np.concatenate((trainGCM_x1, trainGCM_x[base, :, :, :]))
            trainGCM_y1 = np.concatenate((trainGCM_y1, trainGCM_y[choice, :, :].squeeze(1)))
            trainObser_y1 = np.concatenate((trainObser_y1, trainObser_y[choice, :, :].squeeze(1)))

train_sim_y = trainGCM_y1
train_obs_y = trainObser_y1
test_sim_y = validationGCM_y.squeeze()
test_obs_y =  validationObser_y.squeeze()

plat = np.load('./data/Precip_Latitude.npy')[1:]
plon = np.load('./data/Precip_Longitude.npy')

#Each grid point randomly selects 3000 data, and selects randomly 10 times.
def all_cdf (cdfn,obs,mod,global_max):
    wide=global_max/cdfn
    obs=np.sort(obs)
    mod[mod==0]=np.random.uniform(0,wide,np.where(mod==0)[0].shape) # too many 0 value will cause problem. random give value from 0-wide.
    index = mod.argsort()
    mod=np.sort(mod)
    # calculate the global max and bins.
    xbins=np.arange(0.0,global_max+wide,wide)
    # create PDF
    pdfobs,bins=np.histogram(obs,bins=xbins)
    pdfmod,bins=np.histogram(mod,bins=xbins)
    # create CDF with zero in first entry.
    cdfobs = np.insert(np.cumsum(pdfobs), 0, 0.0) / obs.shape[0]
    cdfmod = np.insert(np.cumsum(pdfmod), 0, 0.0) / mod.shape[0]
    return cdfobs,cdfmod,xbins


train_obs_Y = train_obs_y
test_obs_Y = test_obs_y
unet_train = train_sim_y
unet_test = test_sim_y

mask_numpy = np.load("./data/mask.npy")
land_index = np.argwhere((mask_numpy ==1) & (np.max(obs_y_all,0).squeeze() >0)) #only land was selected.

seed_n =47
np.random.seed(seed_n)
torch.manual_seed(seed_n)

datasize = 300
bitch_size = 16
length = train_obs_Y.shape[0]
cdfn = 500
sample_n = 1000
global_max = np.max(obs_y_all,0).squeeze()
all_max = np.max(obs_y_all,0).squeeze()
cdf_train_x = np.zeros ((land_index.shape[0]*sample_n,cdfn+1+3))
cdf_train_y = np.zeros((land_index.shape[0]*sample_n,cdfn+1))
for i in range (land_index.shape[0]):
    for j in range (sample_n):
        random.seed(i+j)
        loc = np.array(random.sample(range(0, length - 1), datasize))
        max_value = all_max[land_index[i][0],land_index[i][1]] / all_max.max()
        lat = plat [land_index[i][0]] / plat.max()
        lon = plon [land_index[i][1]] / plon.max()
        data_sim = unet_train.squeeze()[loc,land_index[i][0],land_index[i][1]]
        data_obs = train_obs_Y [loc,land_index[i][0],land_index[i][1]]
        cdf_obs,cdf_sim,xbins = all_cdf (cdfn,data_obs,data_sim,global_max[land_index[i][0],land_index[i][1]])
        cdf_train_x[i*sample_n+j,:] = np.concatenate((cdf_sim[0:cdfn+1],np.array([lat,lon,max_value])),axis = 0)
        cdf_train_y[i*sample_n+j,:] = cdf_obs[0:cdfn+1]
    print(i)

np.save('./data/cdf_train_sim_p.npy',cdf_train_x)
np.save('./data/cdf_train_obs_p.npy',cdf_train_y)

cdf_test_x = np.zeros ((land_index.shape[0],cdfn+4))
cdf_test_y = np.zeros((land_index.shape[0],cdfn+1))
for i in range (land_index.shape[0]):
    max_value = all_max[land_index[i][0], land_index[i][1]] / all_max.max()
    lat = plat[land_index[i][0]] / plat.max()
    lon = plon[land_index[i][1]] / plon.max()
    data_sim = unet_test.squeeze()[:,land_index[i][0],land_index[i][1]]
    data_obs = test_obs_Y [:,land_index[i][0],land_index[i][1]]
    cdf_obs,cdf_sim,xbins = all_cdf (cdfn,data_obs,data_sim,global_max[land_index[i][0],land_index[i][1]])
    cdf_test_x[i,:] = np.concatenate((cdf_sim[0:cdfn+1],np.array([lat,lon,max_value])),axis = 0)
    cdf_test_y[i,:] = cdf_obs[0:cdfn+1]
    print(i)

np.save('./data/cdf_test_sim_p.npy',cdf_test_x)
np.save('./data/cdf_test_obs_p.npy',cdf_test_y)

batch_size =32

mask_numpy = np.load("./data/mask.npy")
cdf_train_x = np.load('./data/cdf_train_sim_p.npy')
cdf_train_y = np.load('./data/cdf_train_obs_p.npy')
cdf_test_x = np.load('./data/cdf_test_sim_p.npy')
cdf_test_y = np.load('./data/cdf_test_obs_p.npy')

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

train_x = torch.from_numpy(cdf_train_x).float().to(device)
train_y = torch.from_numpy(cdf_train_y).float().to(device)
test_x = torch.from_numpy(cdf_test_x).float().to(device)
test_y = torch.from_numpy(cdf_test_y).float().to(device)

train = torch.utils.data.TensorDataset(train_x, train_y)
test = torch.utils.data.TensorDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

#step 3. Train the model of DNN
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD, Adam
from torch.nn import MSELoss
from torch.nn import Flatten
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(504, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 16)
        self.layer6 = nn.Linear(16, 501)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(501, 501)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        x = self.relu(x)
        x = self.layer6(x)
        x = torch.sigmoid(self.linear(x))
        return x




def train_model(train_dl, test_dl, model):
    ep = []
    trainloss = []
    testloss =[]
    criterion = MSELoss()#
    criterion.cuda()#
    optimizer = Adam(model.parameters(), lr=0.0001)
    min_loss = 1
    for epoch in range(100):
        # initialize for epoch average loss
        epoch_loss = 0.0
        validation_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs.cuda())
            loss1 = 100 * criterion(yhat, targets)
            loss2 =  2 * 100 * torch.nn.functional.mse_loss(yhat[:,10:120],targets[:,10:120])
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
        epoch_loss += targets.shape[0] * loss.item()
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, epoch_loss))
        for i, (inputs2,targets2) in enumerate(test_dl):
            yhat2 = model (inputs2.cuda())
            vloss = criterion(yhat2*100,targets2*100)
        validation_loss +=  targets2.shape[0] * vloss.item()
        print('====> Epoch: {} Test loss: {:.6f}'.format(epoch, validation_loss))
        torch.save(model.state_dict(), './models/model' + str(epoch) + '.pt')


model = DNN().cuda()
train_model(train_loader, test_loader, model)





