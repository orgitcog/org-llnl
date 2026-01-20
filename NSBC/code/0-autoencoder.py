'''
This code is to train a autoencoder model
'''
from scipy.stats import pearsonr
from scipy.stats import skew
from scipy.stats import kurtosis
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

#step 1. input the E3SM and GFDL datasets
train_sim = np.load('./data/train_temp_e3sm_interp.npy')
test_sim = np.load('./data/test_temp_e3sm_interp.npy')
future_sim = np.load('./data/future_temp_e3sm_interp.npy')
sim_all = np.concatenate((train_sim, test_sim, future_sim), axis=0)
min_sim = np.min(sim_all, 0)
max_sim = np.max(sim_all, 0)
train_obs = np.load('./data/train_temp_gfdl_interp.npy')
test_obs = np.load('./data/test_temp_gfdl_interp.npy')
future_obs = np.load('./data/future_temp_gfdl_interp.npy')
obs_all = np.concatenate((train_obs, test_obs, future_obs), axis=0)

train_len = 20000
sim_train = sim_all[0:train_len]
sim_test = sim_all[train_len:]
obs_train = obs_all[0:train_len]
obs_test = obs_all[train_len:]

x_train = np.expand_dims(sim_train, 1)
x_test = np.expand_dims(sim_test, 1)
y_train = np.expand_dims(sim_train, 1)
y_test = np.expand_dims(sim_test, 1)


batch_size = 100
train_x = torch.from_numpy(x_train).float()
train_y = torch.from_numpy(y_train).float()
test_x = torch.from_numpy(x_test).float()
test_y = torch.from_numpy(y_test).float()

train = torch.utils.data.TensorDataset(train_x, train_y)
test = torch.utils.data.TensorDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)


#step 2. build the Auto-encoder model structure.

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
            nn.Linear(3200, 30),
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 400),
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




# step 3. prepare the code to train the Auto-encoder model
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
            loss = 0 * torch.nn.functional.mse_loss(_mean_out, _mean_target) + 1 * torch.nn.functional.mse_loss(out,
                                                                                                                target) + 0 * torch.nn.functional.mse_loss(
                _std_out, _std_target)

            y1 = out.detach().cpu().numpy()

            test_y_mean = np.std(target.detach().cpu().numpy(), 0)
            pred_y_mean = np.std(y1, 0)

            test_y_mean_1 = np.mean(target.detach().cpu().numpy(), 0)
            pred_y_mean_1 = np.mean(y1, 0)
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

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        print(mean_cesm_r, mean_cesm_r_1, np.mean(np.mean(pred_y_mean)))
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        test_batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation',
                               total=len(self.validation_DataLoader),
                               leave=False)
        for j, (x, y) in test_batch_iter:
            input, target = x.to(self.device), y.to(self.device)
            out = self.model(input)
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
model = Autoencoder().cuda()
criterion = torch.nn.KLDivLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=train_loader,
                  validation_DataLoader=test_loader,
                  lr_scheduler=None,
                  epochs=500,
                  epoch=0,
                  notebook=False)

# step 5. start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()
# step 6. save the trained model
torch.save(model.state_dict(), './models/final_sim_auto.pt')