import os
import itertools
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LeakyReLU, LeakyReLU, BatchNorm1d, Dropout
from torch_geometric.utils import softmax
import logging
import time
import datetime
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
import wandb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def count_parameter(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def append_to_dict(d1, d2):
	for key, value in d2.items():
		if key in d1:
			d1[key].append(value)
		else:
			d1[key] = [value]
	return d1

def average_dict(d):
	for k in d.keys():
		d[k] = [x for x in d[k] if x == x] # skip nans
	return {k: sum(v) / len(v) for k, v in d.items()}

def add_prefix_to_dict(d, prefix):
	return {f'{prefix}_{k}': v for k, v in d.items()}

class BaseModel(torch.nn.Module):
	def __init__(self, in_dim, out_dim, output_dir, use_gpu=True, optimizer=None, schedule_lr=False, resume=True):
		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.output_dir = output_dir
		self.logger = self.get_logger()
		self.saved_models_dir = os.path.join(self.output_dir, 'saved_models')
		if not os.path.exists(self.saved_models_dir):
			os.makedirs(self.saved_models_dir)
		self.history = {}
		self.define_modules()
		if not optimizer:
			self.logger.info(f'Using default Adam optimizer with lr=1e-4')
			optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
		self.optimizer = optimizer
		self.schedule_lr = schedule_lr
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, threshold=0.01, verbose=True)
		if resume and self.load('min_val_loss_model.pt', warn_when_fail=False):
			pass
		else:
			self.logger.info('Initializing a new model from scratch')
		self.logger.info(f'Model has {count_parameter(self) / 1e6:.2g}M parameters')
		self.log_parameters()
		self.epoch = 1

	def define_modules(self):
		self.out_mlp = Linear(self.in_dim, self.out_dim)

	def log_parameters(self):
		return

	def set_scheduler(self, scheduler):
		self.scheduler = scheduler

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def get_logger(self):
		logger = logging.getLogger('train_logger')
		if (logger.hasHandlers()):
			logger.handlers.clear()
		logger.setLevel(logging.DEBUG)
		fh = logging.FileHandler(f'{self.output_dir}/train.log')
		fh.setLevel(logging.DEBUG)
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		logger.addHandler(fh)
		logger.addHandler(ch)
		return logger

	def save(self, name):
		checkpoint_dict = {
			'model_state_dict': self.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
			'history': self.history
		}
		torch.save(checkpoint_dict, os.path.join(self.saved_models_dir, name))
		self.logger.info(f'Saved model at {os.path.join(self.saved_models_dir, name)}')

	def load(self, name, warn_when_fail=True, strict=True, absolute=False, load_optimizer=True):
		if not absolute:
			checkpoint_dir = os.path.join(self.saved_models_dir, name)
		else:
			checkpoint_dir = name
		if not os.path.exists(checkpoint_dir):
			if warn_when_fail:
				print(f'Checkpoint directory does not exist: {checkpoint_dir}')
			return False
		checkpoint_dict = torch.load(checkpoint_dir)
		self.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
		if load_optimizer:
			self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
			self.scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
		self.history = checkpoint_dict['history']
		self.logger.info(f'Loaded model from {os.path.join(self.saved_models_dir, name)}')
		return True

	def forward_and_return_loss(self, data, return_y=False):
		y_target = data['y'].float().to(self.device)
		y_pred = self(data)
		loss, loss_info = self.loss(y_pred, y_target)
		if return_y:
			return loss, loss_info, y_target, y_pred
		return loss, loss_info

	def update_model(self, data):
		self.optimizer.zero_grad()
		loss, loss_info, y_target, y_pred = self.forward_and_return_loss(data, return_y=True)
		loss.backward()
		total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
		loss_info['grad_norm'] = total_norm.item()
		if total_norm == total_norm:
			self.optimizer.step()
		else:
			warnings.warn('Gradient update skipped because of NaNs')
			loss_info = {}
		return loss_info, y_target, y_pred
		
	def train_epoch(self, train_loader):
		self.train()
		sum_loss_info = {}
		num_batch = 0
		y_target_list = []
		y_pred_list = []
		for data in tqdm(train_loader):
			loss_info, y_target, y_pred = self.update_model(data)
			y_target_list.extend(y_target.cpu().tolist())
			y_pred_list.extend(y_pred.cpu().tolist())
			sum_loss_info = append_to_dict(sum_loss_info, loss_info)
			num_batch += 1
		average_loss_info = average_dict(sum_loss_info)
		self.epoch += 1
		y_true = np.array(y_target_list)
		y_pred = np.array(y_pred_list)
		r2 = r2_score(y_true=y_true, y_pred=y_pred)
		mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
		mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
		pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
		spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))
		info = {
			'loss': mse,
			'mae': mae,
			'mse': mse,
			'pearsonr': pearsonr[0],
			'spearmanr': spearmanr[0],
			'r2': r2,
			# 'y_true': y_true,
			# 'y_pred': y_pred,
		}
		info.update(average_loss_info)
		return info

	def validate_model(self, loader, prefix):
		self.eval()
		sum_loss_info = {}
		num_batch = 0
		y_target_list = []
		y_pred_list = []
		with torch.no_grad():
			for data in tqdm(loader):
				loss, loss_info, y_target, y_pred = self.forward_and_return_loss(data, return_y=True)
				y_target_list.extend(y_target.cpu().tolist())
				y_pred_list.extend(y_pred.cpu().tolist())
				sum_loss_info = append_to_dict(sum_loss_info, loss_info)
				num_batch += 1
		average_loss_info = average_dict(sum_loss_info)
		y_true = np.array(y_target_list)
		y_pred = np.array(y_pred_list)
		r2 = r2_score(y_true=y_true, y_pred=y_pred)
		mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
		mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
		pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
		spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))
		info = {
			'loss': mse,
			'mae': mae,
			'mse': mse,
			'pearsonr': pearsonr[0],
			'spearmanr': spearmanr[0],
			'r2': r2,
			# 'y_true': y_true,
			# 'y_pred': y_pred,
		}
		info.update(average_loss_info)
		return info

	def update_history(self, info):
		for key, value in info.items():
			if key in self.history:
				self.history[key].append(value)
			else:
				self.history[key] = [value]
		wandb.log(info, commit=True)

	def plot_y(self, prefix):
		y_true = self.history[f'{prefix}_y_true'][-1].reshape(-1)
		y_pred = self.history[f'{prefix}_y_pred'][-1].reshape(-1)
		plt.scatter(y_true, y_pred, s=1)
		x = np.linspace(0, 14, 1000)
		plt.plot(x, x)
		plt.title(prefix)
		plt.xlabel('$y_\mathrm{true}$')
		plt.ylabel('$y_\mathrm{pred}$')
		plt.savefig(os.path.join(self.output_dir, f'{prefix}_pred_vs_truth.png'))
		plt.clf()

	def plot_history(self):
		for key in self.history:
			if 'train' in key and 'loss' in key:
				plt.plot(self.history[key], label=key)
				plt.legend()
				plt.savefig(os.path.join(self.output_dir, 'train_losses.png'))
		# for prefix in ['train', 'val', 'test_2016', 'test_2019']:
		# 	self.plot_y(prefix)
		plt.clf()
		plt.plot(self.history['train_loss'], label='$L^{(train)}$')
		plt.plot(self.history['val_loss'], label='$L^{(val)}$')
		plt.plot(self.history['test_2016_loss'], label='$L^{(test 2016)}$')
		plt.plot(self.history['test_2019_loss'], label='$L^{(test 2019)}$')
		plt.legend()
		plt.savefig(os.path.join(self.output_dir, 'loss_history.png'))
		plt.clf()
		plt.plot(self.history['val_pearsonr'], label='$Pearson^{(val)}$')
		plt.plot(self.history['test_2016_pearsonr'], label='$Pearson^{(test 2016)}$')
		plt.plot(self.history['test_2019_pearsonr'], label='$Pearson^{(test 2019)}$')
		plt.legend()
		plt.savefig(os.path.join(self.output_dir, 'corr_history.png'))
		plt.clf()
		plt.plot(self.history['val_mae'], label='$MAE^{(val)}$')
		plt.plot(self.history['test_2016_mae'], label='$MAE^{(test 2016)}$')
		plt.plot(self.history['test_2019_mae'], label='$MAE^{(test 2019)}$')
		plt.legend()
		plt.savefig(os.path.join(self.output_dir, 'mae_history.png'))
		plt.clf()
		for key in self.history:
			if 'train' in key and ('loss' in key or 'acc' in key):
				plt.plot(self.history[key], label=key)
				plt.legend()
				plt.savefig(os.path.join(self.output_dir, 'train_history.png'))
		for key in self.history:
			if key not in ['loss', 'acc', 'pearsonr', 'mae', 'mse']:
				plt.clf()
				if len(np.array(self.history[key]).shape) == 1:
					plt.plot(self.history[key], label=key)
					plt.legend()
					plt.savefig(os.path.join(self.output_dir, f'{key}.png'))


	def train_model(self, num_epochs, train_loader, val_loader, test_2016_loader, test_2019_loader):
		self.logger.info(f'Traning on {len(train_loader) * train_loader.batch_size} events')
		for epoch in range(num_epochs):
			start = time.time()
			train_info = self.train_epoch(train_loader)
			elapsed = (time.time() - start)
			val_info = self.validate_model(val_loader, prefix='val')
			test_2016_info = self.validate_model(test_2016_loader, prefix='test_2016') if test_2016_loader else val_info
			test_2019_info = self.validate_model(test_2019_loader, prefix='test_2019') if test_2019_loader else val_info
			if self.schedule_lr:
				self.scheduler.step(val_info['loss'])
			self.logger.info('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
			self.logger.info('\tTrain loss: {:.3f}, Val loss: {:.3f}, Test 2016 loss: {:.3f}, Test 2019 loss: {:.3f}'.format(train_info['loss'], val_info['loss'], test_2016_info['loss'], test_2019_info['loss']))
			if 'acc' in train_info:
				self.logger.info('\tTrain acc: {:.3f}, Val acc: {:.3f}, Test 2016 acc: {:.3f}, Test 2019 acc: {:.3f}'.format(train_info['acc'], val_info['acc'], test_2016_info['acc'], test_2019_info['acc']))
			train_info = add_prefix_to_dict(train_info, 'train')
			val_info = add_prefix_to_dict(val_info, 'val')
			test_2016_info = add_prefix_to_dict(test_2016_info, 'test_2016')
			test_2019_info = add_prefix_to_dict(test_2019_info, 'test_2019')
			self.update_history({**train_info, **val_info, **test_2016_info, **test_2019_info})
			# self.plot_history()
			self.save('most_recent_epoch_model.pt')
			if val_info['val_loss'] <= min(self.history['val_loss']):
				self.save('min_val_loss_model.pt')
			if test_2016_info['test_2016_loss'] <= min(self.history['test_2016_loss']):
				self.save('min_test_2016_loss_model.pt')
			if test_2019_info['test_2019_loss'] <= min(self.history['test_2019_loss']):
				self.save('min_test_2019_loss_model.pt')

	def loss(self, y_pred, y_target):
		raise NotImplementedError

	def forward(self, data):
		raise NotImplementedError
