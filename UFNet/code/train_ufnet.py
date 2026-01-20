#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import numpy as np
from skimage.transform import resize
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

#step 1. Data loaded
A = np.load('./data/e3sm_dynamic_all_interpcesmgrid_10_04_small0.npy')
C = np.load('./data/e3smprep_all_cesmgrid.npy')
train_sim_Y = C[:16425]
test_sim_Y = C[16425:]
np.save('./data/np4GCM_test.npy', test_sim_Y)
B = np.load('./data/obs_all_cesmgrid.npy')
train_sim_X = A[:16425]
test_sim_X = A[16425:]
train_obs_Y = B[:16425]
test_obs_Y = B[16425:]

trainObser_y = train_obs_Y
validationObser_y = test_obs_Y
trainGCM_x = train_sim_X
trainGCM_y = train_sim_Y
validationGCM_x = test_sim_X
validationGCM_y = test_sim_Y


# step 2. Generate the training and test dataset.
seed_n=47
np.random.seed(seed_n)
torch.manual_seed(seed_n)
batch_size = 300
trainObser_y = (trainObser_y - trainObser_y.min()) / (trainObser_y.max() - trainObser_y.min())
validationObser_y = (validationObser_y - validationObser_y.min()) / (validationObser_y.max() - validationObser_y.min())
trainGCM_x = (trainGCM_x - trainGCM_x.min()) / (trainGCM_x.max() - trainGCM_x.min())
trainGCM_y = (trainGCM_y - trainGCM_y.min()) / (trainGCM_y.max() - trainGCM_y.min())
validationGCM_x = (validationGCM_x - validationGCM_x.min()) / (validationGCM_x.max() - validationGCM_x.min())
validationGCM_y = (validationGCM_y - validationGCM_y.min()) / (validationGCM_y.max() - validationGCM_y.min())
length = np.shape(trainObser_y)[0]
batchsize = 100
trainGCM_x1 = []
trainGCM_y1 = []
trainObser_y1 = []
for j in range(1):
    for i in tqdm(range(int(length / batchsize))):  # int(length/batchsize)
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

train_x = trainGCM_x1
train_y = np.concatenate((trainObser_y1[:,np.newaxis,:,:],trainGCM_y1[:,np.newaxis,:,:]*train_sim_Y.max()),axis = 1)
test_x = validationGCM_x
test_y = np.concatenate((validationObser_y,validationGCM_y* test_sim_Y.max()),axis = 1)

np.save('./data/test_x_normalized.npy', test_x)
np.save('./data/test_y.npy', test_y)

train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()

train = torch.utils.data.TensorDataset(train_x, train_y)
test = torch.utils.data.TensorDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)


#Step 3. build the structure of UFNet model
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

class Net_Obser(nn.Module):
    def __init__(self):
        super(Net_Obser, self).__init__()
        M = 120
        self.contract1_conv = nn.Conv2d(6, M, 1)
        self.contract11_conv = nn.Conv2d(M, M, 1)
        self.contract1_pool = nn.MaxPool2d((2, 2))
        self.contract2_conv = nn.Conv2d(M, 2 * M, 1)
        self.contract22_conv = nn.Conv2d(2 * M, 2 * M, 1)
        self.contract3_conv = nn.Conv2d(2 * M, 4 * M, 1)
        self.contract33_conv = nn.Conv2d(4 * M, 4 * M, 1)
        self.contract4_conv = nn.Conv2d(4 * M, 8 * M, 1)
        self.contract44_conv = nn.Conv2d(8 * M, 8 * M, 1)
        self.contract5_conv = nn.Conv2d(8 * M, 8 * M, 1)
        self.contract6_conv = nn.Conv2d(16 * M, 8 * M, 1)
        self.contract66_conv = nn.Conv2d(8 * M, 8 * M, 1)
        self.contract7_conv = nn.Conv2d(8 * M, 4 * M, 1)
        self.contract77_conv = nn.Conv2d(4 * M, 4 * M, 1)
        self.contract8_conv = nn.Conv2d(4 * M, 2 * M, 1)
        self.contract88_conv = nn.Conv2d(2 * M, 2 * M, 1)
        self.contract9_conv = nn.Conv2d(2 * M, M, 1)
        self.contract99_conv = nn.Conv2d(M, M, 1)
        self.contract10_conv = nn.Conv2d(M, 10, 1)
        self.dropout = nn.Dropout(0.05)
        self.deconv1 = nn.ConvTranspose2d(8 * M, 8 * M, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(8 * M, 4 * M, 2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(4 * M, 2 * M, 2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(2 * M, M, 2, stride=2)
        self.m = nn.Conv2d(10, 10, (5, 1))
        self.m1 = nn.Conv2d(10, 1, (3, 1))
        self.b1 = nn.Conv2d(M, M, (3, 6))
        self.b2 = nn.Conv2d(M, M, (3, 4))
        self.b3 = nn.Conv2d(2 * M, 2 * M, (2, 4))
        self.b4 = nn.Conv2d(2 * M, 2 * M, (2, 2))
        self.b5 = nn.Conv2d(4 * M, 4 * M, (1, 2))
        self.b6 = nn.Conv2d(4 * M, 4 * M, (2, 2))
        self.b7 = nn.Conv2d(8 * M, 8 * M, (1, 2))
        self.b8 = nn.BatchNorm2d(M)
        self.b9 = nn.BatchNorm2d(4 * M)
        self.b10 = nn.BatchNorm2d(8 * M)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, b, cdfn,all_max,max_value,plat,plon,model_dnn):
        x = self.b8(F.selu(self.contract11_conv(F.selu(self.contract1_conv(x)))))
        x1 = self.contract1_pool(x)
        x2 = F.selu(self.b2(self.b1(x)))
        x11 = F.selu(self.contract22_conv(F.selu(self.contract2_conv(x1))))
        x12 = self.contract1_pool(x11)
        x3 = F.selu(self.b4(self.b3(x11)))
        x121 = self.b9(F.selu(self.contract33_conv(F.selu(self.contract3_conv(x12)))))
        x122 = self.contract1_pool(x121)
        x4 = F.selu(self.b6(self.b5(x121)))
        x1221 = self.b10(F.selu(self.contract44_conv(F.selu(self.contract4_conv(x122)))))
        x1222 = self.contract1_pool(x1221)
        x5 = F.selu(self.b7(x1221))
        x12221 = self.dropout(
            F.selu(self.contract5_conv(F.selu(self.contract5_conv(x1222)))))
        x122211 = F.selu(self.deconv1(x12221))
        x6 = torch.cat((x5, x122211), 1)
        x7 = F.relu(self.contract66_conv(F.relu(self.contract6_conv(x6))))
        x8 = F.selu(self.deconv2(x7))
        x9 = torch.cat((x4, x8), 1)
        x10 = F.relu(self.contract77_conv(F.relu(self.contract7_conv(x9))))
        x11 = F.selu(self.deconv3(x10))
        x12 = torch.cat((x3, x11), 1)
        x13 = F.relu(self.contract88_conv(F.relu(self.contract8_conv(x12))))
        x14 = F.selu(self.deconv4(x13))
        x15 = torch.cat((x2, x14), 1)
        x16 = F.relu(self.contract99_conv(F.relu(self.contract9_conv(x15))))
        x17 = F.relu(self.m1(F.selu(self.m((self.contract10_conv(x16))))))
        x = x17.view(-1, 26, 48)
        x = x.cpu()
        last = x
        b = b.cpu()
        x_qm = torch.zeros((x.shape[0], 26, 48))
        def interp(x, xp, fp):
            indices = torch.searchsorted(xp, x)
            indices = torch.clamp(indices, 1, len(xp) - 1)
            interp_weights = (x - xp[indices - 1]) / (xp[indices] - xp[indices - 1])
            interpolated = (1 - interp_weights) * fp[indices - 1] + interp_weights * fp[indices]
            return interpolated
        for w in range(26):
            for h in range(48):
                obs = b[:, w, h]
                mod = last[:, w, h] * 233.898239415353# train_obs_Y.max()
                obs[torch.isnan(obs)] = 0
                mod[torch.isnan(mod)] = 0
                if obs.max() > 0:
                    global_max = all_max[w,h]
                    wide = global_max / cdfn  #
                    xbins = torch.arange(0.0, float(global_max + wide), float(wide))[0:cdfn + 1]
                    mod[mod == 0] = torch.rand(torch.where(mod == 0)[0].shape) * wide * 0.1
                    obs[obs == 0] = torch.rand(torch.where(obs == 0)[0].shape) * wide * 0.1
                    index = mod.argsort()
                    mod = torch.sort(mod).values
                    obs = torch.sort(obs).values
                    histmod = torch.bincount(torch.bucketize(mod, torch.linspace(0, float(global_max), cdfn + 1)),
                                             minlength=cdfn)[1:, ]
                    histobs = torch.bincount(torch.bucketize(obs, torch.linspace(0, float(global_max), cdfn + 1)),
                                             minlength=cdfn)[1:, ]
                    cdfmod = torch.cat((torch.tensor([0.0]), torch.cumsum(histmod, dim=0),torch.cumsum(histmod, dim=0)[-2:])) / mod.shape[0]
                    cdfsim = torch.cat((torch.tensor([0.0]), torch.cumsum(histobs, dim=0),torch.cumsum(histobs, dim=0)[-2:])) / obs.shape[0]
                    lat = plat[w]/48.5340#maximum value of the lat
                    lon = plon[h]/293.7500#maximum value of the lon
                    cdf_input = torch.cat((cdfsim[0:501], torch.tensor([lat, lon, max_value[w, h]/ max_value.max()])), dim=0)
                    cdfobs = model_dnn(cdf_input.cuda())
                    cdfobs = cdfobs.cpu()
                    cdf1 = interp(mod, xbins, cdfmod)
                    x_qm_pre = interp(cdf1, cdfobs, xbins)
                    a = torch.zeros((mod.shape[0],))
                    a[index] = x_qm_pre
                    a[torch.isnan(a)] = 0
                    x_qm[:, w, h] = a[0:x.shape[0]]
                else:
                    x_qm[:, w, h] = 0.000000000001 * mod
        return x.cuda(), x_qm.cuda()


model = Net_Obser()
import numpy as np
import torch

#step 4. Define the training process
class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter
            """Training block"""
            self._train()
            """Validation block"""
            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(
                        self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):
        _w = torch.Tensor([4.96282349e-03, 3.68031327e-02, 1.82942446e-02, 7.11085828e-02,
                           2.28154232e-02, 6.76589986e-02, 1.66112957e-02, 4.11522634e-02,
                           9.61538462e-02, 2.12765957e-02, 5.88235294e-02, 1.00000000e-01,
                           2.00000000e-01, 1.00000000e+00, 1.00000000e+00])
        loss_function = nn.CrossEntropyLoss(
            _w)
        def my_cross_entropy(x, y):
            loss = torch.nn.functional.kl_div(x, y) + torch.nn.functional.l1_loss(x, y)
            loss = torch.nn.functional.mse_loss(x, y)
            return loss
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)
        mean_cesm_r = 0.
        mean_cesm_r_1 = 0.

        all_max = np.nanmax(np.load('./data/obs_all_cesmgrid.npy'),0).squeeze()#np.load('/p/lustre2/yu42/RADA_model/unet_bias_predictor/v54/v54.34.4/all_max_obs_only.npy')
        all_max = torch.from_numpy(all_max).float()
        plat = np.load('/data/Precip_Latitude.npy')[1:]
        plon = np.load('/data/Precip_Longitude.npy')
        plat = torch.from_numpy(plat).float()
        plon = torch.from_numpy(plon).float()
        max_value = np.nanmax(np.load('./data/obs_all_cesmgrid.npy'),0).squeeze()#np.load('/p/lustre2/yu42/RADA_model/unet_bias_predictor/v54/v54.34.4/all_max_obs_only.npy')
        max_value = torch.from_numpy(max_value).float()

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

        model_path = './models/model_dnn_89.pt'
        model_dnn = DNN()
        model_dnn.load_state_dict(torch.load(model_path))
        model_dnn = model_dnn.cuda()
        model_dnn.train(False)
        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out, out_qm = self.model(input, target[:, 1, :, :], 500, all_max, max_value, plat, plon, model_dnn)
            target = target[:, 0, :, :]
            _mean_out = torch.mean(out, 0)
            _std_out = torch.std(out, 0)
            _mean_target = torch.mean(target, 0)
            _std_target = torch.std(target, 0)
            _mean_outqm = torch.mean(out_qm / 209.26671, 0)
            loss = torch.nn.functional.mse_loss(out, target) + 5 * torch.nn.functional.mse_loss(_mean_out,
                                                                                                _mean_target) + 5 * torch.nn.functional.mse_loss(
                _std_out, _std_target) + torch.nn.functional.mse_loss(_mean_outqm, _mean_target)
            y1 = out.detach().cpu().numpy()
            test_y_mean = np.std(target.detach().cpu().numpy(), 0)
            pred_y_mean = np.std(y1, 0)
            test_y_mean_1 = np.mean(target.detach().cpu().numpy(), 0)
            pred_y_mean_1 = np.mean(y1, 0)
            from scipy.stats import pearsonr
            from scipy.stats import skew
            from scipy.stats import kurtosis
            def RMSE(x, y):
                y = y[~np.isnan(x)]
                x = x[~np.isnan(x)]
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                return np.sqrt(np.mean((y - x) ** 2))
            def correlation(x, y):
                y = y[~np.isnan(x)]
                x = x[~np.isnan(x)]
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                corr = pearsonr(x, y)[0]
                return corr
            mean_cesm_r += correlation(pred_y_mean, test_y_mean)
            mean_cesm_r_1 += correlation(pred_y_mean_1, test_y_mean_1)
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
        print(mean_cesm_r, mean_cesm_r_1)
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        test_batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation',
                               total=len(self.validation_DataLoader),
                               leave=False)
        for j, (x, y) in test_batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            out,out_qm = self.model(input, target[:, 1, :, :], 500,all_max,max_value,plat,plon,model_dnn)
        torch.save(self.model.state_dict(), './models/model' + str(self.epoch) + '.pt')
        batch_iter.close()


    def _validate(self):
        from scipy.stats import pearsonr
        from scipy.stats import skew
        from scipy.stats import kurtosis
        def correlation(x, y):
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]
            corr = pearsonr(x, y)[0]
            return corr
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)
        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out, out_qm = self.model(input, target[:, 1, :, :], 500, all_max, model_dnn)
                target = target[:,0,:,:]
                loss = correlation(out, target)

                batch_iter.set_description(f'Validation: (loss {loss:.4f})')


        batch_iter.close()


# step 5. Parameter definition
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


model = Net_Obser().cuda()
criterion = torch.nn.KLDivLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=train_loader,
                  validation_DataLoader=test_loader,
                  lr_scheduler=None,
                  epochs=300,
                  epoch=0,
                  notebook=False)

# step 6. start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()




# step 7. applied the trained model to get the bias corrected results.

import numpy as np
import torch
import numpy as np
import torch
import numpy as np
import torch.nn as nn
import torch, tqdm, copy, os, ssl, h5py
import torch.nn.functional as F
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import pearsonr
from cartopy.io import shapereader
import matplotlib.colors
from cartopy import crs as ccrs
import matplotlib.pyplot as plt

def RMSE(x, y):
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    return np.sqrt(np.mean((y - x) ** 2))

def correlation(x, y):
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    corr = pearsonr(x, y)[0]
    return corr

mask_numpy = np.load("./data/mask.npy")

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

model_path = './models/model_dnn_89.pt'
model_dnn = DNN()
model_dnn.load_state_dict(torch.load(model_path))
model_dnn = model_dnn.cuda()
model_dnn.train(False)
model_path = './models/model_ufnet.pt'
model = Net_Obser()
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.train(False)
nP4GCM = test_sim_Y
nP4Obser = test_obs_Y
preds = []
test_x = np.load('./data/test_x_normalized.npy')
test_y = np.load('./data/test_y.npy')
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()
test = torch.utils.data.TensorDataset(test_x, test_y)
batch_size = 300
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
for j, (x, y) in enumerate(test_loader):
    input, target = x.cuda(), y.cuda()  # send to device (GPU or CPU)
    all_max = np.nanmax(np.load('./data/obs_all_cesmgrid.npy'),0).squeeze()
    all_max = torch.from_numpy(all_max).float()
    plat = np.load('./data/Precip_Latitude.npy')[1:]
    plon = np.load('./data/Precip_Longitude.npy')
    plat = torch.from_numpy(plat).float()
    plon = torch.from_numpy(plon).float()
    max_value = np.nanmax(np.load('./data/obs_all_cesmgrid.npy'),0).squeeze()
    max_value = torch.from_numpy(max_value).float()
    out,qm = model(input, target[:, 1, :, :], 500,all_max,max_value,plat,plon,model_dnn)
    for i in qm.detach().cpu().numpy():
        preds.append(i)
preds = np.array(preds)
preds_path = './data/preds_ep_ufnet.npy'
np.save(preds_path,preds)









