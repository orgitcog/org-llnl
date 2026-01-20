'''
This code is used to train a LSTM, the input and output will be the latent space from auto-encoder model
'''

from scipy.stats import pearsonr
from scipy.stats import skew
from scipy.stats import kurtosis
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import netCDF4 as nc

#step 1. input the trained auto-encoder mdoel.
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Upsample(size=(28, 28), mode='nearest'),

            nn.Conv2d(1, 4, kernel_size=5),

            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),

            nn.Flatten(),
            nn.Linear(3200, 50),

        )
        self.decoder = nn.Sequential(

            nn.Linear(50, 400),

            nn.ReLU(True),
            nn.Linear(400, 4000),

            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),

            nn.ConvTranspose2d(10, 10, kernel_size=5),

            nn.ConvTranspose2d(10, 1, kernel_size=5),
            nn.Upsample(size=(26, 48), mode='nearest'),

        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder()
model.load_state_dict(torch.load('./models/final_sim_auto.pt', map_location='cpu'))


#step 2. Generate the train and test datasets for the LSTM
import numpy as np
import numpy as np
# step2.1 GHG forcing
greengas = nc.Dataset('./data/GHG_CMIP_SSP585-1-2-1_Annual_Global_2015-2500_c20190310.nc')
co2 = greengas['CO2'][:]
ch4 = greengas['CH4'][:]
date = greengas['date'][:]
time=greengas['time']
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
# step2.2 Latent space
'''
train_sim = np.load('./data/train_temp_e3sm_interp.npy')
test_sim = np.load('./data/test_temp_e3sm_interp.npy')
future_sim = np.load('./data/future_temp_e3sm_interp.npy')
sim_all = np.concatenate((train_sim, test_sim, future_sim), axis=0)
test_obs = sim_all
test_obs = torch.from_numpy(np.expand_dims(test_obs, 1)).float()
from tqdm import tqdm
pred_y = np.zeros((test_obs.shape[0], 50))
pred_y1 = np.zeros((test_obs.shape[0], test_obs.shape[1], test_obs.shape[2], test_obs.shape[3]))
for index in tqdm(range(test_obs.shape[0])):
    pred_y[index:index + 1, :] = model.encoder(test_obs[index:index + 1, :, :, :]).detach().cpu().numpy()
X = pred_y
train_obs = np.load('./data/train_temp_gfdl_interp.npy')
test_obs = np.load('./data/test_temp_gfdl_interp.npy')
future_obs = np.load('./data/future_temp_gfdl_interp.npy')
obs_all = np.concatenate((train_obs, test_obs, future_obs), axis=0)
test_obs = obs_all
test_obs = torch.from_numpy(np.expand_dims(test_obs, 1)).float()
from tqdm import tqdm
pred_y = np.zeros((test_obs.shape[0], 50))
for index in tqdm(range(test_obs.shape[0])):
    pred_y[index:index + 1, :] = model.encoder(test_obs[index:index + 1, :, :, :]).detach().cpu().numpy()
Y = pred_y
np.save('/data/x_54750_50.npy',X)
np.save('/data/y_54750_50.npy',Y)
'''

X = np.load('./data/x_54750_50.npy')

# step2.3 combine latent space and GHG
X = np.concatenate((X,ghg),axis=1)
Y = np.load('./data/y_54750_50.npy')

# step2.4 normalization
X_min = np.nanmin(X,axis=0)
X_max = np.nanmax(X,axis = 0)
Y_min = np.nanmin(Y[0:365*72,:],axis = 0)
Y_max = np.nanmax(Y[0:365*72,:],axis = 0)*2
X = (X-X_min)/(X_max - X_min)
Y = (Y-Y_min)/(Y_max - Y_min)

#step2.5 generate the train and test dataset
seq_len = 30
def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i + seq_len)] for i in range(n_seq)])
train_len = 365 * 72
train_frac = train_len
def get_train_test_sets(data, seq_len, train_frac):
    sequences_x = split_into_sequences(data, seq_len)
    n_train = train_frac
    x_train = sequences_x[:n_train, :-1]
    x_test = sequences_x[n_train:, :-1]
    return x_train, x_test

x_train, x_test = get_train_test_sets(X, seq_len, train_frac=train_len)
y_train = Y[29:29+x_train.shape[0],]
y_test = Y[29+x_train.shape[0]:,]
num_hidden_state = 200

train_x = torch.from_numpy(x_train).float()
train_y = torch.from_numpy(y_train).float()
test_x = torch.from_numpy(x_test).float()
test_y = torch.from_numpy(y_test).float()

batch_size=100
train = torch.utils.data.TensorDataset(train_x, train_y)
test = torch.utils.data.TensorDataset(test_x, test_y)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

#step 3. build the LSTM structure.

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

#step 4. build the training

def train(model, train_loader, num_epochs, learning_rate):
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()  # Set the model to training mode
    loss_show = []
    # Loop over epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
        loss_show.append(running_loss / len(train_loader))
    return loss_show


n_steps = 29  # Number of time steps
n_features = 2  # Number of input features
device = torch.device('cuda')
# step5 start training
model = Pred_lstm()
model.cuda()
loss_show = train(model, train_loader, num_epochs=500, learning_rate=0.0001)
torch.save(model.state_dict(), './models/new_test_lstm_final.pt')



