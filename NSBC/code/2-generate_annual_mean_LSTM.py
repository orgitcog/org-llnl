'''
This code is used to train a Unet model to simulate the residual
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch, tqdm, copy, os, ssl, h5py
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py
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
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import pearsonr
from cartopy.io import shapereader

#step 1. generate the training and test dataset，E3SM remove the annual mean; obs remove the annual mean calculated based on LSTM

train_x = np.load('./data/train_dynamic_interp_withoutstand.npy')
test_x = np.load('./data/test_dynamic_interp_withoutstand.npy')
future_x = np.load('./data/future_dynamic_interp_withoutstand.npy')
all_x = np.concatenate((train_x,test_x,future_x),axis = 0)
train_x = all_x[0*365:72*365,:,:,:]
test_x = all_x[72*365:82*365,:,:,:]
future_x = all_x[82*365:,:,:,:]
all_x = np.concatenate((train_x,test_x,future_x),axis=0)
all_pre = np.zeros((all_x.shape[0],1,36,56))
for w in range(36):
    for h in range(56):
        data = all_x[:,0,w,h]
        data_mean = np.nanmean(all_x[:,0,w,h].reshape(-1,365),axis=1)
        all_pre[:,0,w,h] = data - np.repeat(data_mean, 365, axis=0)
train_x = all_pre[0*365:72*365,:,:,:]
test_x = all_pre[72*365:82*365,:,:,:]
future_x = all_pre[82*365:,:,:,:]


train_y_obs = np.load('./data/train_temp_obs_interp.npy')
test_y_obs = np.load('./data/test_temp_obs_interp.npy')
future_y_obs = np.load('./data/future_temp_obs_interp.npy')
all_y_obs = np.concatenate((train_y_obs,test_y_obs,future_y_obs),axis = 0)
train_y_obs = all_y_obs[0*365:72*365,:,:]
test_y_obs = all_y_obs[72*365:82*365,:,:]
future_y_obs = all_y_obs[82*365:,:,:]
np.save('./data/future_y.npy', future_y_obs)
np.save('./data/test_obs.npy', test_y_obs)
#every grid remove their annual mean，this annual mean was calculated based on LSTM
all_y = np.concatenate((train_y_obs,test_y_obs,future_y_obs),axis=0)
all_pre = np.zeros((all_y.shape[0],26,48))
correct_mean = np.zeros((150,26,48))
corrected_all = np.load('./data/LSTM_gfdl_1950_2099.npy')

for w in range(26):
    for h in range(48):
        corrected = corrected_all[:,w,h]
        correct_mean[:,w,h] = corrected
        data = all_y[:,w,h]
        all_pre[:,w,h] = data - np.repeat(corrected, 365, axis=0)
    print(w)
np.save('./data/lstm_correct_daily_mean.npy', np.repeat(correct_mean,365,axis=0))
train_y_obs = all_pre[0*365:72*365,:,:]
test_y_obs = all_pre[72*365:82*365,:,:]
future_y_obs = all_pre[82*365:,:,:]

train_y_e3sm = np.load('./data/train_temp_e3sm_interp.npy')
test_y_e3sm = np.load('./data/test_temp_e3sm_interp.npy')
future_y_e3sm = np.load('./data/future_temp_e3sm_interp.npy')
all_y_e3sm = np.concatenate((train_y_e3sm,test_y_e3sm,future_y_e3sm),axis = 0)
train_y_e3sm = all_y_e3sm[0*365:72*365,:,:]
test_y_e3sm = all_y_e3sm[72*365:82*365,:,:]
future_y_e3sm = all_y_e3sm[82*365:,:,:]
np.save('./data/np4GCM_future.npy', future_y_e3sm)
train_sim_Y = train_y_e3sm
test_sim_Y = test_y_e3sm
np.save('./data/np4GCM_test.npy', test_sim_Y)


seed_n=47
np.random.seed(seed_n)
torch.manual_seed(seed_n)
batch_size = 100

train_sim_X = train_x
test_sim_X = test_x
train_obs_Y = train_y_obs
test_obs_Y = test_y_obs

trainObser_y = train_obs_Y
validationObser_y = test_obs_Y
trainGCM_x = train_sim_X
trainGCM_y = train_sim_Y

validationGCM_x = test_sim_X
validationGCM_y = test_sim_Y
futureGCM_x = future_x
futureGCM_y = future_y_e3sm
futureObser_y = future_y_obs


train_x = trainGCM_x
train_y = trainObser_y
test_x = validationGCM_x
test_y = validationObser_y


np.save('./data/test_x_normalized.npy', test_x)
np.save('./data/test_y.npy', test_y)
np.save('./data/future_x_normalized.npy', futureGCM_x)


train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()

train = torch.utils.data.TensorDataset(train_x, train_y)
test = torch.utils.data.TensorDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

#step 2. build the UNet structure.
class Net_Obser(nn.Module):
    def __init__(self):
        super(Net_Obser, self).__init__()
        M = 120
        self.contract1_conv = nn.Conv2d(1, M, 1)
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
        # self.m2=nn.Softmax(dim=1)
        self.b1 = nn.Conv2d(M, M, (3, 6))
        self.b2 = nn.Conv2d(M, M, (3, 4))
        self.b3 = nn.Conv2d(2 * M, 2 * M, (2, 4))
        self.b4 = nn.Conv2d(2 * M, 2 * M, (2, 2))
        self.b5 = nn.Conv2d(4 * M, 4 * M, (1, 2))
        self.b6 = nn.Conv2d(4 * M, 4 * M, (2, 2))
        self.b7 = nn.Conv2d(8 * M, 8 * M, (1, 2))
        # self.b6=nn.Conv2d(4*M,4*M,(2,2))
        self.b8 = nn.BatchNorm2d(M)
        self.b9 = nn.BatchNorm2d(4 * M)
        self.b10 = nn.BatchNorm2d(8 * M)
        self.softmax = nn.Softmax(dim=1)
        self.conv_last1 = nn.Conv1d(26, 27, 1)
        self.conv_last2 = nn.Conv1d(48, 62, 1)
    def forward(self, x):
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
        x17 = (self.m1(F.selu(self.m((self.contract10_conv(x16))))))
        x = x17.view(-1, 26, 48)
        return x


#step 3. prepare the code to train the Unet model

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
            self.epoch += 1

            """Training block"""
            self._train()

            """Validation block"""

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(
                        self.validation_loss[i])
                else:
                    self.lr_scheduler.batch()
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

        self.model.train()
        train_losses = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)
        mean_cesm_r = 0.
        mean_cesm_r_1 = 0.

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(input)
            _mean_out = torch.mean(out, 0)
            _std_out = torch.std(out, 0)
            _mean_target = torch.mean(target, 0)
            _std_target = torch.std(target, 0)
            _median_out=torch.median(out, 0)[0]
            _median_target=torch.median(target, 0)[0]
            loss = torch.nn.functional.mse_loss(out, target)+ 2.0 * torch.nn.functional.mse_loss(_median_out, _median_target)+ 0.5 * torch.nn.functional.mse_loss(_mean_out, _mean_target)
            + 0.5 * torch.nn.functional.mse_loss(_std_out, _std_target)

            y1 = out.detach().cpu().numpy()
            test_y_mean = np.std(target.detach().cpu().numpy(), 0)
            pred_y_mean = np.std(y1, 0)
            test_y_mean_1 = np.mean(target.detach().cpu().numpy(), 0)
            pred_y_mean_1 = np.mean(y1, 0)
            from scipy.stats import pearsonr
            from scipy.stats import skew
            from scipy.stats import kurtosis
            def correlation(x, y):
                x = x[~np.isnan(x)]
                y = y[~np.isnan(y)]
                corr = pearsonr(x, y)[0]
                return corr

            def RMSE(x, y):
                x = x[~np.isnan(x)]
                y = y[~np.isnan(y)]
                return np.sqrt(np.mean((y - x) ** 2))

            mean_cesm_r += correlation(pred_y_mean, test_y_mean)
            mean_cesm_r_1 += correlation(pred_y_mean_1, test_y_mean_1)

            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()
            self.optimizer.step()

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')

        print(mean_cesm_r, mean_cesm_r_1,np.mean(np.mean(pred_y_mean)))
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        test_batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation',
                               total=len(self.validation_DataLoader),
                               leave=False)
        for j, (x, y) in test_batch_iter:
            input, target = x.to(self.device), y.to(self.device)
            out = self.model(input)
        torch.save(self.model.state_dict(), './' + str(v) + '/models/model' + str(self.epoch) + '.pt')
        batch_iter.close()
        test_batch_iter

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

        self.model.eval()
        valid_losses = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                out = self.model(input)

                loss = correlation(out, target)


                batch_iter.set_description(f'Validation: (loss {loss:.4f})')



        batch_iter.close()



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# step 4. set parameters for the model training.
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

# step 5. start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

# step 6. apply the trained model to the future period and generate the final daily temperature dataset

model_path = './models/final_unet.pt'
model = Net_Obser()
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.train(False)
nP4GCM = np.load('./data/np4GCM_test.npy')
nP4Obser = np.load('./data/test_obs.npy')
preds = []
test_x = np.load('./data/test_x_normalized.npy')
test_y = np.load('./data/test_y.npy')
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()
test = torch.utils.data.TensorDataset(test_x, test_y)
batch_size = 100
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
for j, (x, y) in enumerate(test_loader):
    input, target = x.cuda(), y.cuda()  # send to device (GPU or CPU)
    out = model(input)  # one forward pass
    for i in out.detach().cpu().numpy():
        preds.append(i)
preds = preds + correct[72*365:82*365,:,:]
preds_path = './data/tas_test_nsbc.npy'
np.save(preds_path,preds)



future_x_pre = futureGCM_x
model_path = './models/final_unet.pt'
model = Net_Obser()
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.train(False)
nP4GCM = np.load('./data/np4GCM_test.npy')
nP4Obser = np.load('./data/test_obs.npy')
preds = []
test_x = future_x_pre
test_y = np.load('./data/future_y.npy')
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()
test = torch.utils.data.TensorDataset(test_x, test_y)
batch_size = 100
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
for j, (x, y) in enumerate(test_loader):
    input, target = x.cuda(), y.cuda()  # send to device (GPU or CPU)
    out = model(input)
    for i in out.detach().cpu().numpy():
        preds.append(i)
preds = preds + correct[82*365:,:,:]
preds_path = './data/tas_future_nsbc.npy'
np.save(preds_path,preds)
