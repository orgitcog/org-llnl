'''
This code is used to generate the annual mean temperature based on the trained LSTM model
'''
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import skew
from scipy.stats import kurtosis
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import netCDF4 as nc

#step1. generate the input data of LSTM

greengas = nc.Dataset('./data/GHG_CMIP_SSP585-1-2-1_Annual_Global_2015-2500_c20190310.nc')
co2 = greengas['CO2'][:]
ch4 = greengas['CH4'][:]
date = greengas['date'][:]
time=greengas['time']
#nc.num2date(time,time.units)
nc.num2date(time[:], time.units, calendar = time.calendar)
ghg_his = nc.Dataset('./data/GHG_CMIP-1-2-0_Annual_Global_0000-2014_c20180105.nc')
ghg_fut = nc.Dataset('./data/GHG_CMIP_SSP585-1-2-1_Annual_Global_2015-2500_c20190310.nc')
co2_his = ghg_his['CO2'][:]
co2_fut = ghg_fut['CO2'][:]
ch4_his = ghg_his['CH4'][:]
ch4_fut = ghg_fut['CH4'][:]
n2o_his = ghg_his['N2O'][:]
n2o_fut = ghg_fut['N2O'][:]
f11_his = ghg_his['f11'][:]
f11_fut = ghg_fut['f11'][:]
f12_his = ghg_his['f12'][:]
f12_fut = ghg_fut['f12'][:]
time=ghg_his['time']
time_his = nc.num2date(time[:], time.units, calendar = time.calendar,has_year_zero = True)
time_fut=ghg_fut['time']
time_fut = nc.num2date(time_fut[:], time_fut.units, calendar = time_fut.calendar)
co2 = np.concatenate((co2_his[0:-2],co2_fut[0:-2]))[1950:2100]
ch4 = np.concatenate((ch4_his[0:-2],ch4_fut[0:-2]))[1950:2100]
n2o = np.concatenate((n2o_his[0:-2],n2o_fut[0:-2]))[1950:2100]
f11 = np.concatenate((f11_his[0:-2],f11_fut[0:-2]))[1950:2100]
f12 = np.concatenate((f12_his[0:-2],f12_fut[0:-2]))[1950:2100]
co2_nor = ((co2-co2.min())/(co2.max()-co2.min()))
ch4_nor =((ch4-ch4.min())/(ch4.max()-ch4.min()))
n2o_nor =((n2o-n2o.min())/(n2o.max()-n2o.min()))
f11_nor =((f11-f11.min())/(f11.max()-f11.min()))
f12_nor =((f12-f12.min())/(f12.max()-f12.min()))
all_ghg = ((co2-co2.min())/(co2.max()-co2.min()))+((ch4-ch4.min())/(ch4.max()-ch4.min()))+((n2o-n2o.min())/(n2o.max()-n2o.min()))+((f11-f11.min())/(f11.max()-f11.min()))+((f12-f12.min())/(f12.max()-f12.min()))
all_ghg = co2_nor+ch4_nor+n2o_nor+f11_nor+f12_nor
all_ghg_annual = all_ghg
ghg = np.zeros((365*150,1))
for i in range (150):
    ghg[i*365:(i+1)*365,0] = all_ghg_annual[i]

import numpy as np
import numpy as np

X = np.load('./data/x_54750_50.npy')
Y = np.load('./data/y_54750_50.npy')
X = np.concatenate((X,ghg),axis=1)
X_min = np.nanmin(X,axis=0)
X_max = np.nanmax(X,axis = 0)
Y_min = np.nanmin(Y[0:365*72,:],axis = 0)
Y_max = np.nanmax(Y[0:365*72,:],axis = 0)*2
X = (X-X_min)/(X_max - X_min)
Y = (Y-Y_min)/(Y_max - Y_min)

seq_len = 30
def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i + seq_len)] for i in range(n_seq)])

train_len = 365 * 72
train_frac = train_len

start = 0
def get_train_test_sets(data, seq_len, train_frac):
    sequences_x = split_into_sequences(data, seq_len)
    n_train = train_frac
    x_train = sequences_x[:n_train, :-1]
    x_test = sequences_x[n_train:, :-1]
    return x_train, x_test

x_train, x_test = get_train_test_sets(X, seq_len,train_frac=train_len)

y_train = Y[start+29:start+29+x_train.shape[0],]
y_test = Y[start+29+x_train.shape[0]:,]

X_train = x_train
X_test = x_test
Y_train = y_train
Y_test =  y_test

X_seq = torch.from_numpy(np.concatenate((X_train,X_test),axis=0)).float()
Y_seq = torch.from_numpy(np.concatenate((Y_train,Y_test),axis=0)).float()
test = torch.utils.data.TensorDataset(X_seq, Y_seq)
batch_size = 300
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

# step 2. generate the autoencoder and LSTM model.

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Upsample(size=(28, 28), mode='nearest'),
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, 50),
            # 10
            # nn.Softmax(),
        )
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(50, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            nn.Upsample(size=(26, 48), mode='nearest'),
            # 28 x 28
            # nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
num_hidden_state = 40
class Pred_lstm(nn.Module):
    def __init__(self):
        super(Pred_lstm, self).__init__()
        self.layer_aug = nn.Sequential(
            nn.Flatten(),
            nn.Unflatten(1, (1, (seq_len - 1) * 51)),

        )
        self.LSTM = nn.Sequential(
            nn.LSTM(input_size=(seq_len - 1) * 51, hidden_size=num_hidden_state,
                    num_layers=1, batch_first=True),
        )
        self.linearlayer = nn.Sequential(
            nn.Linear(num_hidden_state, 60),
            # nn.ReLU(),
            nn.Linear(60, 50),
            # nn.ReLU(),
        )
    def forward(self, x):
        x = self.layer_aug(x)
        output, (hn, cn) = self.LSTM(x)
        x = output
        x = self.linearlayer(x)
        x = x.squeeze(1)
        return x

device = torch.device('cuda')
model = Pred_lstm()
model.cuda()

#step3. apply the dataset into the trained LSTM to predict latent space

import numpy as np
import torch
model = Autoencoder().cuda()
model.load_state_dict(torch.load('./models/final_sim_auto.pt', map_location='cuda'))
model1 = Pred_lstm().cuda()
from torchsummary import summary
model1.load_state_dict(torch.load('./models/new_test_lstm_final.pt', map_location='cuda'))
model1.train(False)
predictions = []
for j, (x, y) in enumerate(test_loader):
    input,target = x.cuda(),y.cuda()
    out = model1(input)
    for i in out.detach().cpu().numpy():
        predictions.append(i)

predictions = np.array(predictions)
pred_y_all = np.zeros((predictions.shape[0],50))

for i in range(50):
    preds = predictions[:,i:(i+1)] * (Y_max[i] - Y_min[i]) + Y_min[i]
    pred_y_all[:,i] = preds[:,0]
    pred = np.concatenate((preds[0:29,0],preds[:, 0]),axis=0)

# step 4. apply the predict latent space into the trained decoder model

import numpy as np
import numpy as np
model = Autoencoder().cuda()
model.load_state_dict(torch.load('./models/final_sim_auto.pt', map_location='cuda'))

import numpy as np
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

pred_y = pred_y_all
train_sim = np.load('./data/train_temp_gfdl_interp.npy')
test_sim = np.load('./data/test_temp_gfdl_interp.npy')
future_sim = np.load('./data/future_temp_gfdl_interp.npy')
sim_all = np.concatenate((train_sim, test_sim, future_sim), axis=0)
Y = sim_all
train_sim = np.load('./data/train_temp_e3sm_interp.npy')
test_sim = np.load('./data/test_temp_e3sm_interp.npy')
future_sim = np.load('./data/future_temp_e3sm_interp.npy')
sim_all = np.concatenate((train_sim, test_sim, future_sim), axis=0)
from tqdm import tqdm
pred_final = np.zeros((pred_y.shape[0], 26, 48))
for index in tqdm(range(pred_y.shape[0])):
    pred_final[index:index + 1, :, :] = model.decoder(
        torch.from_numpy(pred_y[index:index + 1, :]).float().cuda()).detach().cpu().numpy().squeeze(1)

# step5. calculate and save the annual mean temperature dataset from the LSTM

actual_year = np.zeros((150, 26, 48))
pred_year = np.zeros((150, 26, 48))
pred_final_pre = np.concatenate((Y[0:29, :, :], pred_final), axis=0)
gcm_pre = np.zeros((150, 26, 48))
for index in range((150)):
    actual_year[index, :, :] = np.mean((Y[365 * index:(index + 1) * 365]), 0)
    pred_year[index, :, :] = np.mean((pred_final_pre[365 * index:(index + 1) * 365]), 0)
    gcm_pre[index, :, :] = np.mean((sim_all[365 * index:(index + 1) * 365]), 0)

np.save('./data/LSTM_gfdl_1950_2099.npy', pred_year)



